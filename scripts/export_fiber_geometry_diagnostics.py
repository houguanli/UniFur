#!/usr/bin/env python3
"""Export scalp-frame and shell/strand geometry diagnostics.

The bundle is intentionally renderer-independent.  It contains PLY line sets
for root-to-tip geometry, a root-density/scalp-occupancy PLY, per-view image
projections, and numeric multi-view mask-support statistics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from dpd3dgs_animal.fiber import create_unified_fiber_field
from dpd3dgs_animal.fiber_optimize import _sample_mask_at_world_points
from dpd3dgs_animal.observations import resolve_observations
from dpd3dgs_animal.optimize import (
    DifferentiableSkeletonTetModel,
    _frame_paths,
    _load_gt_frame_torch,
    _resolve_device,
    _resolve_render_size,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-npz", required=True)
    parser.add_argument("--gaussian-ply", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--frame-dir", required=True)
    parser.add_argument("--camera-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-views", type=int, default=8)
    parser.add_argument("--max-arrows", type=int, default=1600)
    parser.add_argument("--render-width", type=int, default=1024)
    parser.add_argument("--render-height", type=int, default=1024)
    return parser.parse_args()


def _load_field(args: argparse.Namespace, motion, device: str):
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    metadata = payload.get("metadata", {})
    with np.load(args.stage1_npz, allow_pickle=False) as stage1_payload:
        scalp_face_indices = (
            stage1_payload["scalp_face_indices"].astype(np.int64)
            if "scalp_face_indices" in stage1_payload.files
            else None
        )
    field = create_unified_fiber_field(
        args.gaussian_ply,
        motion.rest_surface_vertices.detach().cpu().numpy(),
        motion.surface_faces.detach().cpu().numpy(),
        device=device,
        max_points=int(metadata.get("point_count", 20_000)),
        scalp_face_indices=scalp_face_indices,
    )
    state = payload["state_dict"]
    gate = state.get("strand_visibility_gate")
    if isinstance(gate, torch.Tensor):
        field.strand_visibility_gate = torch.empty_like(
            gate, device=field.route_logits.device
        )
    incompatible = field.load_state_dict(state, strict=False)
    allowed = {
        "expert_color_delta",
        "bend_cubic_local",
        "residual_log_scale_delta",
        "residual_rotation_raw",
        "residual_trust_logits",
        "structured_delta_raw",
        "strand_visibility_gate",
    }
    unexpected_missing = set(incompatible.missing_keys) - allowed
    if unexpected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={sorted(unexpected_missing)}, "
            f"unexpected={sorted(incompatible.unexpected_keys)}"
        )
    if "structured_delta_raw" in incompatible.missing_keys:
        with torch.no_grad():
            field.structured_delta_raw.fill_(1.0)
    return field.eval(), metadata


def _geometry(field, vertices, faces):
    root, tangent, bitangent, normal = field.surface_frame(vertices, faces)
    direction = F.normalize(
        field.direction_local[:, :1] * tangent
        + field.direction_local[:, 1:2] * bitangent
        + field.direction_local[:, 2:3] * normal,
        dim=-1,
        eps=1e-8,
    )
    bend = field.bend_local[:, :1] * tangent + field.bend_local[:, 1:] * bitangent
    cubic = (
        field.bend_cubic_local[:, :1] * tangent
        + field.bend_cubic_local[:, 1:] * bitangent
    )
    origin = root + field.height[:, None] * normal
    shell_target = origin + field.shell_length[:, None] * direction
    strand_target = origin + field.strand_length[:, None] * (direction + bend + cubic)
    residual = root + (
        field.residual_offset_local[:, :1] * tangent
        + field.residual_offset_local[:, 1:2] * bitangent
        + field.residual_offset_local[:, 2:3] * normal
    )
    gains = field.structured_delta_gain
    shell_deployed = torch.lerp(residual, shell_target, gains[:, 0:1])
    strand_deployed = torch.lerp(residual, strand_target, gains[:, 1:2])
    return {
        "root": root,
        "normal": normal,
        "direction": direction,
        "residual": residual,
        "shell_target": shell_target,
        "strand_target": strand_target,
        "shell_tip": shell_deployed,
        "strand_tip": strand_deployed,
    }


def _write_line_ply(path: Path, starts: np.ndarray, ends: np.ndarray, colors) -> None:
    count = int(starts.shape[0])
    vertices = np.concatenate([starts, ends], axis=0)
    start_color = np.tile(np.asarray(colors[0], dtype=np.uint8), (count, 1))
    end_color = np.tile(np.asarray(colors[1], dtype=np.uint8), (count, 1))
    vertex_colors = np.concatenate([start_color, end_color], axis=0)
    with open(path, "w", encoding="utf-8") as file:
        file.write("ply\nformat ascii 1.0\n")
        file.write(f"element vertex {2 * count}\n")
        file.write("property float x\nproperty float y\nproperty float z\n")
        file.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        file.write(f"element edge {count}\n")
        file.write("property int vertex1\nproperty int vertex2\nend_header\n")
        for point, color in zip(vertices, vertex_colors):
            file.write(
                f"{point[0]:.8g} {point[1]:.8g} {point[2]:.8g} "
                f"{color[0]} {color[1]} {color[2]}\n"
            )
        for index in range(count):
            file.write(f"{index} {index + count}\n")


def _write_occupancy_ply(path: Path, roots: np.ndarray) -> dict[str, float]:
    from scipy.spatial import cKDTree

    k = min(9, len(roots))
    distances, _indices = cKDTree(roots).query(roots, k=k)
    density_distance = np.asarray(distances)[:, -1] if k > 1 else np.zeros(len(roots))
    low, high = np.quantile(density_distance, [0.02, 0.98])
    occupancy = 1.0 - np.clip((density_distance - low) / max(high - low, 1e-8), 0.0, 1.0)
    colors = np.stack(
        [255.0 * occupancy, 255.0 * (1.0 - np.abs(2.0 * occupancy - 1.0)), 255.0 * (1.0 - occupancy)],
        axis=-1,
    ).astype(np.uint8)
    with open(path, "w", encoding="utf-8") as file:
        file.write("ply\nformat ascii 1.0\n")
        file.write(f"element vertex {len(roots)}\n")
        file.write("property float x\nproperty float y\nproperty float z\n")
        file.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        file.write("property float occupancy\nend_header\n")
        for point, color, score in zip(roots, colors, occupancy):
            file.write(
                f"{point[0]:.8g} {point[1]:.8g} {point[2]:.8g} "
                f"{color[0]} {color[1]} {color[2]} {score:.8g}\n"
            )
    return {
        "occupancy_mean": float(occupancy.mean()),
        "occupancy_p05": float(np.quantile(occupancy, 0.05)),
        "occupancy_p95": float(np.quantile(occupancy, 0.95)),
        "knn_distance_mean": float(density_distance.mean()),
    }


def _project(points: np.ndarray, camera) -> tuple[np.ndarray, np.ndarray]:
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    camera_points = homogeneous @ np.asarray(camera.world_to_camera).T
    z = camera_points[:, 2]
    safe = np.maximum(z, 1e-8)
    x = float(camera.fx) * camera_points[:, 0] / safe + float(camera.cx)
    y_sign = 1.0 if camera.image_y_down else -1.0
    y = float(camera.cy) + y_sign * float(camera.fy) * camera_points[:, 1] / safe
    xy = np.stack([x, y], axis=-1)
    valid = (
        (z > 1e-5)
        & (x >= 0)
        & (x < camera.width)
        & (y >= 0)
        & (y < camera.height)
    )
    return xy, valid


def _draw_projection(image_path: Path, camera, geometry, out_path: Path, max_arrows: int) -> None:
    image = Image.open(image_path).convert("RGB").resize(
        (camera.width, camera.height), Image.Resampling.LANCZOS
    )
    draw = ImageDraw.Draw(image)
    count = len(geometry["root"])
    indices = np.linspace(0, count - 1, min(max_arrows, count)).astype(np.int64)
    projected = {}
    validity = {}
    for key in ("root", "residual", "shell_tip", "strand_tip"):
        projected[key], validity[key] = _project(geometry[key], camera)
    for index in indices:
        root = projected["root"][index]
        if validity["root"][index] and validity["shell_tip"][index]:
            tip = projected["shell_tip"][index]
            draw.line([tuple(root), tuple(tip)], fill=(0, 220, 255), width=1)
        if validity["root"][index] and validity["strand_tip"][index]:
            tip = projected["strand_tip"][index]
            draw.line([tuple(root), tuple(tip)], fill=(255, 80, 20), width=1)
        if validity["residual"][index]:
            x, y = projected["residual"][index]
            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(255, 230, 0))
        if validity["root"][index]:
            x, y = root
            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(40, 255, 80))
    draw.rectangle((8, 8, 355, 34), fill=(0, 0, 0))
    draw.text(
        (13, 12),
        "root green | residual yellow | shell cyan | strand orange",
        fill=(255, 255, 255),
    )
    image.save(out_path)


def main() -> None:
    args = _arguments()
    out_dir = Path(args.out_dir)
    projections_dir = out_dir / "projections"
    projections_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    motion = DifferentiableSkeletonTetModel(args.stage1_npz, device=device)
    motion.joints.requires_grad_(False)
    field, metadata = _load_field(args, motion, device)
    frame_paths = _frame_paths(args.frame_dir)
    width, height = _resolve_render_size(
        (args.render_width, args.render_height), frame_paths, args.stage1_npz
    )
    observations = resolve_observations(
        frame_paths,
        args.stage1_npz,
        motion.rest_surface_vertices.detach().cpu().numpy(),
        width,
        height,
        camera_manifest=args.camera_manifest,
    )
    view_count = min(max(args.max_views, 1), len(frame_paths))
    view_indices = np.linspace(0, len(frame_paths) - 1, view_count).astype(int).tolist()

    with torch.no_grad():
        rest_geometry = _geometry(
            field, motion.rest_surface_vertices, motion.surface_faces
        )
    rest_numpy = {
        key: value.detach().cpu().numpy() for key, value in rest_geometry.items()
    }
    _write_line_ply(
        out_dir / "shell_root_tip.ply",
        rest_numpy["root"],
        rest_numpy["shell_tip"],
        ((40, 255, 80), (0, 220, 255)),
    )
    _write_line_ply(
        out_dir / "strand_root_tip.ply",
        rest_numpy["root"],
        rest_numpy["strand_tip"],
        ((40, 255, 80), (255, 80, 20)),
    )
    occupancy_report = _write_occupancy_ply(
        out_dir / "scalp_occupancy.ply", rest_numpy["root"]
    )

    support = {"root": [], "residual": [], "shell_tip": [], "strand_tip": []}
    valid_projection = {key: [] for key in support}
    direction_normal_cosine = []
    projection_paths = []
    with torch.no_grad():
        for view_index in view_indices:
            _tet, vertices, _joints = motion.driven_points(
                observations.motion_indices[view_index]
            )
            geometry = _geometry(field, vertices, motion.surface_faces)
            direction_normal_cosine.append(
                torch.sum(geometry["direction"] * geometry["normal"], dim=-1)
                .detach()
                .cpu()
                .numpy()
            )
            target = _load_gt_frame_torch(
                frame_paths[view_index], width, height, device
            )
            for key in support:
                sampled, valid = _sample_mask_at_world_points(
                    geometry[key], observations.cameras[view_index], target["mask"], 3
                )
                valid_values = sampled[valid]
                valid_projection[key].append(float(valid.float().mean().cpu()))
                support[key].append(
                    float((valid_values >= 0.5).float().mean().cpu())
                    if valid_values.numel()
                    else 0.0
                )
            geometry_numpy = {
                key: value.detach().cpu().numpy() for key, value in geometry.items()
            }
            projection_path = projections_dir / f"{view_index:03d}_{frame_paths[view_index].stem}.png"
            _draw_projection(
                frame_paths[view_index],
                observations.cameras[view_index],
                geometry_numpy,
                projection_path,
                args.max_arrows,
            )
            projection_paths.append(projection_path)

    cosine = np.concatenate(direction_normal_cosine)
    barycentric = field.barycentric.detach().cpu().numpy()
    gain = field.structured_delta_gain.detach().cpu().numpy()
    gate = field.strand_visibility_gate.detach().cpu().numpy()
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "point_count": field.point_count,
        "views": view_indices,
        "camera_source": observations.source,
        "root_attachment": {
            "construction": "exact barycentric point on bound scalp triangle",
            "barycentric_sum_max_abs_error": float(
                np.max(np.abs(barycentric.sum(axis=1) - 1.0))
            ),
            "barycentric_min": float(barycentric.min()),
            "root_to_bound_triangle_distance": 0.0,
        },
        "direction_vs_scalp_normal": {
            "cosine_mean": float(cosine.mean()),
            "cosine_p05": float(np.quantile(cosine, 0.05)),
            "cosine_p95": float(np.quantile(cosine, 0.95)),
            "inward_fraction": float((cosine < 0.0).mean()),
        },
        "multi_view_hair_mask_support": {
            key: {
                "mean": float(np.mean(values)),
                "per_view": values,
                "valid_projection_mean": float(
                    np.mean(valid_projection[key])
                ),
                "valid_projection_per_view": valid_projection[key],
            }
            for key, values in support.items()
        },
        "structured_delta_gain": {
            "shell_mean": float(gain[:, 0].mean()),
            "strand_mean": float(gain[:, 1].mean()),
        },
        "stored_visual_hull_gate": {
            "shape": list(gate.shape),
            "kept_fraction": float(gate.mean()) if gate.size else None,
        },
        "scalp_occupancy": occupancy_report,
        "checkpoint_metadata": metadata,
        "files": {
            "shell_root_tip": str(out_dir / "shell_root_tip.ply"),
            "strand_root_tip": str(out_dir / "strand_root_tip.ply"),
            "scalp_occupancy": str(out_dir / "scalp_occupancy.ply"),
            "projections": [str(path) for path in projection_paths],
        },
    }
    with open(out_dir / "geometry_audit.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
