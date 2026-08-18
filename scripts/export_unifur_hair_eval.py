#!/usr/bin/env python3
"""Export UniFur geometry in the representation used by HairGS metrics.

The primary ``strand_deployed`` mode exports the actual hard-routed strand
centerlines after residual-to-structure continuation. ``strand_target`` is a
diagnostic upper bound before that continuation gain is applied.
``residual_points`` exports no connectivity because residual Gaussians are not
hair strands.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dpd3dgs_animal.fiber import (
    ROUTE_NAMES,
    _quaternion_to_matrix_torch,
    create_unified_fiber_field,
)
from dpd3dgs_animal.optimize import DifferentiableSkeletonTetModel


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-npz", required=True)
    parser.add_argument("--gaussian-ply", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-npz", required=True)
    parser.add_argument("--out-report", required=True)
    parser.add_argument(
        "--mode",
        choices=(
            "strand_deployed",
            "strand_target",
            "structured_deployed",
            "residual_points",
        ),
        default="strand_deployed",
    )
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument(
        "--hard-policy",
        choices=("argmax", "mass_preserving"),
        default="mass_preserving",
    )
    parser.add_argument("--spacing-mm", type=float, default=3.0)
    parser.add_argument("--dense-samples", type=int, default=129)
    parser.add_argument("--max-segments", type=int, default=256)
    parser.add_argument("--min-length-mm", type=float, default=0.1)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _load_field(args: argparse.Namespace, motion: DifferentiableSkeletonTetModel):
    payload = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    metadata = payload.get("metadata", {})
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
        device=args.device,
        max_points=int(metadata.get("point_count", 20_000)),
        point_sampling_mode=str(metadata.get("point_sampling_mode", "uniform_index")),
        exact_vertex_binding=bool(metadata.get("exact_vertex_binding", False)),
        binding_mode=str(metadata.get("binding_mode", "closest_surface")),
        source_mask_mode=str(metadata.get("source_mask_mode", "all")),
        source_mask_threshold=float(metadata.get("source_mask_threshold", 0.25)),
        source_min_opacity=float(metadata.get("source_min_opacity", 0.0)),
        residual_max_scale_fraction=float(
            metadata.get("residual_max_scale_fraction", 0.0)
        ),
        semantic_mask_from_source=bool(
            metadata.get("semantic_mask_from_source", False)
        ),
        structured_foreground_only=bool(
            metadata.get("structured_foreground_only", False)
        ),
        scalp_face_indices=scalp_faces,
    )
    # Visual-hull gates are data-dependent buffers whose sample dimension is
    # established during training.  Recreate both buffers from checkpoint
    # shapes before loading; ``strict=False`` does not suppress tensor shape
    # mismatches.
    for name in ("shell_visibility_gate", "strand_visibility_gate"):
        gate = payload["state_dict"].get(name)
        if isinstance(gate, torch.Tensor):
            setattr(
                field,
                name,
                torch.empty_like(gate, device=field.route_logits.device),
            )
    incompatible = field.load_state_dict(payload["state_dict"], strict=False)
    allowed_missing = {
        "expert_color_delta",
        "bend_cubic_local",
        "residual_log_scale_delta",
        "residual_rotation_raw",
        "residual_trust_logits",
        "structured_delta_raw",
        "shell_visibility_gate",
        "strand_visibility_gate",
        "route_active_gate",
    }
    unexpected_missing = set(incompatible.missing_keys) - allowed_missing
    if unexpected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={sorted(unexpected_missing)}, "
            f"unexpected={sorted(incompatible.unexpected_keys)}"
        )
    return field.eval(), metadata


def _world_geometry(field, vertices: torch.Tensor, faces: torch.Tensor):
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
    residual = root + (
        field.residual_offset_local[:, :1] * tangent
        + field.residual_offset_local[:, 1:2] * bitangent
        + field.residual_offset_local[:, 2:3] * normal
    )
    return direction, bend, cubic, origin, residual


def _resample_curves(
    curves: np.ndarray,
    *,
    spacing: float,
    max_segments: int,
    min_length: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    all_points: list[np.ndarray] = []
    all_directions: list[np.ndarray] = []
    all_ids: list[np.ndarray] = []
    all_edges: list[np.ndarray] = []
    lengths: list[float] = []
    dropped = 0
    point_offset = 0
    output_strand_id = 0
    for curve in curves:
        delta = np.diff(curve, axis=0)
        segment_length = np.linalg.norm(delta, axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(segment_length)])
        length = float(cumulative[-1])
        if not math.isfinite(length) or length < min_length:
            dropped += 1
            continue
        segments = min(max(int(math.ceil(length / spacing)), 1), max_segments)
        distance = np.linspace(0.0, length, segments + 1)
        nodes = np.stack(
            [np.interp(distance, cumulative, curve[:, axis]) for axis in range(3)],
            axis=1,
        )
        directions = np.diff(nodes, axis=0)
        norm = np.linalg.norm(directions, axis=1, keepdims=True)
        valid = norm[:, 0] > 1e-12
        if not np.any(valid):
            dropped += 1
            continue
        points = nodes[:-1][valid]
        directions = directions[valid] / norm[valid]
        count = len(points)
        all_points.append(points)
        all_directions.append(directions)
        all_ids.append(np.full(count, output_strand_id, dtype=np.int64))
        if count > 1:
            start = np.arange(point_offset, point_offset + count - 1, dtype=np.int64)
            all_edges.append(np.stack([start, start + 1], axis=1))
        point_offset += count
        output_strand_id += 1
        lengths.append(length)
    if not all_points:
        raise RuntimeError("No non-degenerate curves survived export")
    points = np.concatenate(all_points, axis=0)
    directions = np.concatenate(all_directions, axis=0)
    strand_ids = np.concatenate(all_ids, axis=0)
    edges = (
        np.concatenate(all_edges, axis=0)
        if all_edges
        else np.empty((0, 2), dtype=np.int64)
    )
    length_array = np.asarray(lengths)
    report = {
        "input_curve_count": int(len(curves)),
        "output_strand_count": int(output_strand_id),
        "dropped_degenerate_count": int(dropped),
        "point_count": int(len(points)),
        "length_m_mean": float(length_array.mean()),
        "length_m_median": float(np.median(length_array)),
        "length_m_p90": float(np.quantile(length_array, 0.9)),
    }
    return points, directions, strand_ids, edges, report


def main() -> int:
    args = _arguments()
    if args.spacing_mm <= 0 or args.min_length_mm < 0:
        raise ValueError("Spacing must be positive and minimum length non-negative")
    if args.dense_samples < 2 or args.max_segments < 1:
        raise ValueError("Need at least two dense samples and one output segment")
    motion = DifferentiableSkeletonTetModel(args.stage1_npz, device=args.device)
    field, metadata = _load_field(args, motion)
    vertices = motion.rest_surface_vertices
    faces = motion.surface_faces
    with torch.no_grad():
        direction, bend, cubic, origin, residual = _world_geometry(
            field, vertices, faces
        )
        probabilities = field.route_probabilities(
            temperature=args.temperature,
            hard=True,
            hard_policy=args.hard_policy,
        )
        route_ids = probabilities.argmax(dim=-1)

        if args.mode == "residual_points":
            residual_primitives = field.residual_primitives(vertices, faces)
            matrix = _quaternion_to_matrix_torch(residual_primitives.rotation)
            major_axis = residual_primitives.scaling.argmax(dim=-1)
            row = torch.arange(field.point_count, device=matrix.device)
            directions_tensor = matrix[row, :, major_axis]
            points_np = residual_primitives.xyz.cpu().numpy().astype(np.float64)
            directions_np = (
                F.normalize(directions_tensor, dim=-1).cpu().numpy().astype(np.float64)
            )
            strand_ids_np = edges_np = None
            geometry_report = {
                "point_count": int(len(points_np)),
                "output_strand_count": None,
                "topology": "not_applicable_unstructured_gaussians",
            }
        else:
            t = torch.linspace(
                0.0,
                1.0,
                args.dense_samples,
                device=vertices.device,
                dtype=vertices.dtype,
            )
            curve_target = origin[:, None, :] + field.strand_length[:, None, None] * (
                t[None, :, None] * direction[:, None, :]
                + t[None, :, None].square() * bend[:, None, :]
                + t[None, :, None].pow(3) * cubic[:, None, :]
            )
            strand_gain = field.structured_delta_gain[:, 1]
            curve_deployed = torch.lerp(
                residual[:, None, :].expand_as(curve_target),
                curve_target,
                strand_gain[:, None, None],
            )
            strand_mask = route_ids == ROUTE_NAMES.index("strand")
            selected_curves = [
                curve_target[strand_mask]
                if args.mode == "strand_target"
                else curve_deployed[strand_mask]
            ]

            if args.mode == "structured_deployed":
                shell_mask = route_ids == ROUTE_NAMES.index("shell")
                shell_target = origin[:, None, :] + (
                    field.shell_length[:, None, None]
                    * t[None, :, None]
                    * direction[:, None, :]
                )
                shell_gain = field.structured_delta_gain[:, 0]
                shell_deployed = torch.lerp(
                    residual[:, None, :].expand_as(shell_target),
                    shell_target,
                    shell_gain[:, None, None],
                )
                selected_curves.append(shell_deployed[shell_mask])

            curves_np = torch.cat(selected_curves, dim=0).cpu().numpy().astype(np.float64)
            points_np, directions_np, strand_ids_np, edges_np, geometry_report = (
                _resample_curves(
                    curves_np,
                    spacing=args.spacing_mm * 1e-3,
                    max_segments=args.max_segments,
                    min_length=args.min_length_mm * 1e-3,
                )
            )
            geometry_report["route_selected_count"] = int(len(curves_np))
            geometry_report["spacing_mm"] = float(args.spacing_mm)

    out_npz = Path(args.out_npz)
    out_report = Path(args.out_report)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"points": points_np, "directions": directions_np}
    if strand_ids_np is not None:
        arrays["points_id_to_strand_id"] = strand_ids_np
        arrays["edges"] = edges_np
    np.savez_compressed(out_npz, **arrays)
    payload = {
        "schema": "unifur-hairgs-geometry-export-v1",
        "mode": args.mode,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "stage1": str(Path(args.stage1_npz).resolve()),
        "temperature": float(args.temperature),
        "hard_policy": args.hard_policy,
        "route_counts": {
            name: int((route_ids == index).sum().cpu())
            for index, name in enumerate(ROUTE_NAMES)
        },
        "geometry": geometry_report,
        "bbox_min": [float(value) for value in points_np.min(axis=0)],
        "bbox_max": [float(value) for value in points_np.max(axis=0)],
        "checkpoint_metadata": metadata,
        "output_npz": str(out_npz.resolve()),
    }
    with open(out_report, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
