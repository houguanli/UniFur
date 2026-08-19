#!/usr/bin/env python3
"""Render a carrier-aware UniFur edit video from one calibrated view."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from dpd3dgs_animal.config import load_config
from dpd3dgs_animal.fiber import (
    CARRIER_NAMES,
    HARD_ROUTE_POLICIES,
    ROUTE_NAMES,
    create_unified_fiber_field,
    deform_simulation_asset,
    simulation_asset_summary,
)
from dpd3dgs_animal.fiber_optimize import _render
from dpd3dgs_animal.scaffold import (
    DifferentiableSurfaceScaffold,
    _frame_paths,
    _resolve_device,
    _resolve_render_size,
)
from dpd3dgs_animal.observations import resolve_observations


def _to_image(rgb: torch.Tensor, label: str) -> Image.Image:
    array = (
        rgb.detach().clamp(0.0, 1.0).mul(255.0).byte().cpu().numpy()
    )
    image = Image.fromarray(array, mode="RGB")
    draw = ImageDraw.Draw(image)
    box = draw.textbbox((0, 0), label)
    draw.rectangle((4, 4, box[2] + 12, box[3] + 12), fill=(0, 0, 0))
    draw.text((8, 8), label, fill=(255, 255, 255))
    return image


def _grid(images: list[Image.Image]) -> Image.Image:
    width, height = images[0].size
    canvas = Image.new("RGB", (2 * width, 2 * height), (0, 0, 0))
    for index, image in enumerate(images):
        canvas.paste(image, ((index % 2) * width, (index // 2) * height))
    return canvas


def _load_field(args, cfg, device):
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    metadata = payload.get("metadata", {})
    point_count = int(metadata.get("point_count", cfg.fiber_max_points))
    point_sampling_mode = str(
        metadata.get("point_sampling_mode", cfg.fiber_point_sampling_mode)
    )
    exact_vertex_binding = bool(
        metadata.get("exact_vertex_binding", cfg.fiber_exact_vertex_binding)
    )
    motion = DifferentiableSurfaceScaffold(args.stage1_npz, device=device)
    motion.joints.requires_grad_(False)
    with np.load(args.stage1_npz, allow_pickle=False) as stage1:
        scalp_faces = (
            stage1["scalp_face_indices"].astype(np.int64)
            if "scalp_face_indices" in stage1.files
            else None
        )
    field = create_unified_fiber_field(
        args.gaussian_ply,
        motion.rest_surface_vertices.detach().cpu().numpy(),
        motion.surface_faces.detach().cpu().numpy(),
        device=device,
        max_points=point_count,
        point_sampling_mode=point_sampling_mode,
        exact_vertex_binding=exact_vertex_binding,
        binding_mode=str(metadata.get("binding_mode", cfg.fiber_binding_mode)),
        source_mask_mode=str(
            metadata.get("source_mask_mode", cfg.fiber_source_mask_mode)
        ),
        source_mask_threshold=float(
            metadata.get("source_mask_threshold", cfg.fiber_source_mask_threshold)
        ),
        source_min_opacity=float(
            metadata.get("source_min_opacity", cfg.fiber_source_min_opacity)
        ),
        residual_max_scale_fraction=float(
            metadata.get(
                "residual_max_scale_fraction",
                cfg.fiber_residual_max_scale_fraction,
            )
        ),
        semantic_mask_from_source=bool(
            metadata.get(
                "semantic_mask_from_source", cfg.fiber_semantic_mask_from_source
            )
        ),
        structured_foreground_only=bool(
            metadata.get(
                "structured_foreground_only",
                cfg.fiber_structured_foreground_only,
            )
        ),
        shell_propagated_direction_weight=float(
            metadata.get(
                "shell_propagated_direction_weight",
                cfg.fiber_shell_propagated_direction_weight,
            )
        ),
        initial_residual_trust=float(cfg.fiber_initial_residual_trust),
        scalp_face_indices=scalp_faces,
        binding_cache=cfg.fiber_binding_cache,
    )
    state = payload["state_dict"]
    gate = state.get("strand_visibility_gate")
    if isinstance(gate, torch.Tensor):
        field.strand_visibility_gate = torch.empty_like(
            gate, device=field.route_logits.device
        )
    incompatible = field.load_state_dict(state, strict=False)
    allowed_missing = {
        "carrier_logits",
        "carrier_root_tip_raw",
        "initial_carrier_probabilities",
        "initial_carrier_root_tip",
        "route_active_gate",
    }
    unexpected_missing = set(incompatible.missing_keys) - allowed_missing
    if unexpected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={sorted(unexpected_missing)}, "
            f"unexpected={sorted(incompatible.unexpected_keys)}"
        )
    field.eval()
    return field, motion, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage1-npz", required=True)
    parser.add_argument("--gaussian-ply", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--frame-dir", required=True)
    parser.add_argument("--camera-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--frames", type=int, default=96)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--render-width", type=int)
    parser.add_argument("--render-height", type=int)
    parser.add_argument("--wind-scale", type=float, default=0.04)
    parser.add_argument("--length-amplitude", type=float, default=0.18)
    parser.add_argument("--hard-carriers", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = _resolve_device(cfg.device)
    out_dir = Path(args.out_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    field, motion, metadata = _load_field(args, cfg, device)
    frame_paths = _frame_paths(args.frame_dir)
    render_size = (
        (args.render_width, args.render_height)
        if args.render_width and args.render_height
        else None
    )
    width, height = _resolve_render_size(render_size, frame_paths, args.stage1_npz)
    observations = resolve_observations(
        frame_paths,
        args.stage1_npz,
        motion.rest_surface_vertices.detach().cpu().numpy(),
        width,
        height,
        camera_manifest=args.camera_manifest,
    )
    frame_index = int(args.frame_index) % len(frame_paths)
    camera = observations.cameras[frame_index]
    _tet, surface_vertices, _joints = motion.driven_points(
        observations.motion_indices[frame_index]
    )
    shell_samples = int(metadata.get("shell_samples", cfg.fiber_shell_samples))
    strand_samples = int(metadata.get("strand_samples", cfg.fiber_strand_samples))
    hard_policy = str(metadata.get("hard_route_policy", cfg.fiber_hard_route_policy))
    if hard_policy not in HARD_ROUTE_POLICIES:
        raise ValueError(f"Unknown hard route policy {hard_policy!r}")
    route_palette = torch.tensor(
        [[0.15, 0.55, 1.0], [1.0, 0.25, 0.1], [0.4, 1.0, 0.2]],
        device=device,
    )
    carrier_palette = torch.tensor(
        [[0.55, 0.55, 0.55], [0.1, 0.65, 1.0], [1.0, 0.2, 0.55]],
        device=device,
    )
    with torch.no_grad():
        primitives = field.primitives(
            surface_vertices,
            motion.surface_faces,
            shell_samples=shell_samples,
            strand_samples=strand_samples,
            temperature=cfg.fiber_final_temperature,
            hard_route=False,
            hard_route_policy=hard_policy,
                fin_aspect_ratio=cfg.fiber_fin_aspect_ratio,
                additive_teacher=cfg.fiber_additive_teacher_mode,
                teacher_opacity_transfer=cfg.fiber_teacher_opacity_transfer,
            )
        base = _render(primitives, camera, cfg, cfg.fiber_renderer)
        route = _render(
            replace(primitives, color=route_palette[primitives.route_id]),
            camera,
            cfg,
            cfg.fiber_renderer,
        )
        carrier_ids = primitives.carrier_probabilities.argmax(dim=-1)
        carrier = _render(
            replace(primitives, color=carrier_palette[carrier_ids]),
            camera,
            cfg,
            cfg.fiber_renderer,
        )
        scene_scale = float(field.scene_scale.detach().cpu())
        for video_frame in range(int(args.frames)):
            phase = 2.0 * math.pi * video_frame / max(int(args.frames), 1)
            length_scale = 1.0 + float(args.length_amplitude) * math.sin(phase)
            wind = torch.tensor(
                [
                    float(args.wind_scale) * scene_scale * math.sin(phase),
                    0.25 * float(args.wind_scale) * scene_scale * math.sin(2.0 * phase),
                    0.0,
                ],
                dtype=primitives.xyz.dtype,
                device=device,
            )
            edited = deform_simulation_asset(
                primitives,
                length_scale=length_scale,
                wind_displacement=wind,
                hard_carriers=bool(args.hard_carriers),
            )
            prediction = _render(edited, camera, cfg, cfg.fiber_renderer)
            panel = _grid(
                [
                    _to_image(base["rgb"], "Reconstruction"),
                    _to_image(
                        prediction["rgb"],
                        f"Carrier edit  length={length_scale:.2f}",
                    ),
                    _to_image(route["rgb"], "Render route  S/Str/R"),
                    _to_image(carrier["rgb"], "Motion carrier  surf/shell/strand"),
                ]
            )
            panel.save(frames_dir / f"{video_frame:05d}.png")

    video_path = out_dir / "simulation_edit.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(args.fps),
            "-i",
            str(frames_dir / "%05d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            str(video_path),
        ],
        check=True,
    )
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "frame_index": frame_index,
        "frames": int(args.frames),
        "fps": int(args.fps),
        "video": str(video_path.resolve()),
        "carrier_summary": field.carrier_summary(cfg.fiber_final_temperature),
        "asset_summary": simulation_asset_summary(primitives),
        "edit": {
            "length_amplitude": float(args.length_amplitude),
            "wind_scale_scene": float(args.wind_scale),
            "hard_carriers": bool(args.hard_carriers),
        },
    }
    with (out_dir / "simulation_edit_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
