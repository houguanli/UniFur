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
from dpd3dgs_animal.gaussian import load_gaussian_ply
from dpd3dgs_animal.fiber_optimize import _sample_mask_at_world_points
from dpd3dgs_animal.observations import resolve_observations
from dpd3dgs_animal.scaffold import (
    DifferentiableSurfaceScaffold,
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
    # Training stores most reconstruction arguments under metadata["config"],
    # while counts and reports live at the top level.  Reconstruct the runtime
    # policy from both layers so non-persistent buffers (notably surface KNN)
    # are identical during post-training audits.
    field_config = dict(metadata.get("config", {}))
    field_config.update(metadata)
    with np.load(args.stage1_npz, allow_pickle=False) as stage1_payload:
        scalp_face_indices = (
            stage1_payload["scalp_face_indices"].astype(np.int64)
            if "scalp_face_indices" in stage1_payload.files
            else None
        )
    source_count = int(load_gaussian_ply(args.gaussian_ply).xyz.shape[0])
    checkpoint_count = int(field_config.get("point_count", source_count))
    field = create_unified_fiber_field(
        args.gaussian_ply,
        motion.rest_surface_vertices.detach().cpu().numpy(),
        motion.surface_faces.detach().cpu().numpy(),
        device=device,
        max_points=int(field_config.get("point_count", 20_000)),
        point_sampling_mode=str(
            field_config.get("point_sampling_mode", "uniform_index")
        ),
        exact_vertex_binding=bool(field_config.get("exact_vertex_binding", False)),
        binding_mode=str(field_config.get("binding_mode", "closest_surface")),
        source_mask_mode=str(field_config.get("source_mask_mode", "all")),
        source_mask_threshold=float(field_config.get("source_mask_threshold", 0.25)),
        source_min_opacity=float(field_config.get("source_min_opacity", 0.0)),
        residual_max_scale_fraction=float(
            field_config.get("residual_max_scale_fraction", 0.0)
        ),
        semantic_mask_from_source=bool(
            field_config.get("semantic_mask_from_source", False)
        ),
        structured_foreground_only=bool(
            field_config.get("structured_foreground_only", False)
        ),
        shell_propagated_direction_weight=float(
            field_config.get("shell_propagated_direction_weight", 1.0)
        ),
        neighbor_k=int(
            field_config.get(
                "route_neighbor_k",
                metadata.get("surface_propagation_report", {}).get("neighbor_k", 0),
            )
        ),
        root_barycentric_max_delta=float(
            field_config.get("root_barycentric_max_delta", 0.0)
        ),
        expert_sh_max_delta=float(field_config.get("expert_sh_max_delta", 0.5)),
        expert_sh_degree=int(field_config.get("expert_sh_degree", 0)),
        initial_residual_trust=float(
            field_config.get("initial_residual_trust", 0.95)
        ),
        initial_shell_length_scale=field_config.get("initial_shell_length_scale"),
        initial_strand_length_scale=field_config.get("initial_strand_length_scale"),
        initialize_direction_from_normal=bool(
            field_config.get("initialize_direction_from_normal", False)
        ),
        scalp_face_indices=scalp_face_indices,
        unbound_root_capacity=max(checkpoint_count - source_count, 0),
        binding_cache=field_config.get("binding_cache"),
    )
    state = payload["state_dict"]
    for gate_name in ("shell_visibility_gate", "strand_visibility_gate"):
        gate = state.get(gate_name)
        if isinstance(gate, torch.Tensor):
            setattr(
                field,
                gate_name,
                torch.empty_like(gate, device=field.route_logits.device),
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
        "shell_visibility_gate",
        "route_active_gate",
        "route_neighbor_index",
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


def _quantiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {key: 0.0 for key in ("min", "p01", "p05", "mean", "p50", "p95", "p99", "max")}
    return {
        "min": float(values.min()),
        "p01": float(np.quantile(values, 0.01)),
        "p05": float(np.quantile(values, 0.05)),
        "mean": float(values.mean()),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
    }


def _threshold_fractions(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"above_0p1": 0.0, "above_0p25": 0.0, "above_0p5": 0.0, "above_0p9": 0.0}
    return {
        "above_0p1": float((values > 0.1).mean()),
        "above_0p25": float((values > 0.25).mean()),
        "above_0p5": float((values > 0.5).mean()),
        "above_0p9": float((values > 0.9).mean()),
    }


def _gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[values >= 0.0]
    if values.size == 0 or float(values.sum()) <= 0.0:
        return 0.0
    ordered = np.sort(values)
    index = np.arange(1, ordered.size + 1, dtype=np.float64)
    return float(
        (2.0 * np.sum(index * ordered) / (ordered.size * ordered.sum()))
        - (ordered.size + 1.0) / ordered.size
    )


def _root_distribution(
    roots: np.ndarray,
    face_index: np.ndarray,
    mask: np.ndarray,
    scene_scale: float,
) -> dict[str, object]:
    from scipy.spatial import cKDTree

    selected = roots[mask]
    selected_faces = face_index[mask]
    if len(selected) < 2:
        return {"count": int(len(selected)), "valid": False}
    distances, _ = cKDTree(selected).query(selected, k=min(9, len(selected)))
    distances = np.asarray(distances)
    nearest = distances[:, 1]
    kth = distances[:, -1]
    unique_faces, face_counts = np.unique(selected_faces, return_counts=True)
    scale = max(float(scene_scale), 1e-8)
    return {
        "count": int(len(selected)),
        "valid": True,
        "occupied_face_count": int(len(unique_faces)),
        "roots_per_occupied_face": _quantiles(face_counts),
        "face_count_gini": _gini(face_counts),
        "nearest_neighbor_distance": _quantiles(nearest),
        "k8_distance": _quantiles(kth),
        "near_duplicate_fraction_scene_1e-5": float((nearest < scale * 1e-5).mean()),
        "near_duplicate_fraction_scene_1e-4": float((nearest < scale * 1e-4).mean()),
        "isolated_fraction_above_3x_median_k8": float(
            (kth > 3.0 * max(float(np.median(kth)), 1e-12)).mean()
        ),
    }


def _distribution_audit(
    field,
    geometry,
    shell_samples: int,
    strand_samples: int,
) -> dict[str, object]:
    roots = geometry["root"].detach().cpu().numpy()
    direction = geometry["direction"].detach()
    normal = geometry["normal"].detach()
    face_index = field.face_index.detach().cpu().numpy()
    active = field.route_active_gate.detach().cpu().numpy() > 0.5
    shell_active = active[:, 0]
    strand_active = active[:, 1]
    shell_active_torch = torch.as_tensor(shell_active, device=direction.device)
    strand_active_torch = torch.as_tensor(strand_active, device=direction.device)
    scene_scale = float(field.scene_scale.detach().cpu())

    neighbor_index = field.route_neighbor_index
    has_neighbor_graph = neighbor_index.numel() > 0
    if has_neighbor_graph:
        neighbor_direction = direction[neighbor_index]
        axial_cosine = torch.abs(
            torch.sum(direction[:, None, :] * neighbor_direction, dim=-1)
        ).clamp(0.0, 1.0)
        axial_angle_all = torch.rad2deg(torch.acos(axial_cosine))
        signed_disagreement_all = (
            torch.sum(direction[:, None, :] * neighbor_direction, dim=-1) < 0.0
        ).float()
        active_pair = (
            strand_active_torch[:, None]
            & strand_active_torch[neighbor_index]
        )
        axial_angle = axial_angle_all[active_pair]
        signed_disagreement = signed_disagreement_all[active_pair]
    else:
        active_pair = torch.zeros((0,), dtype=torch.bool, device=direction.device)
        axial_angle = torch.zeros(0, device=direction.device)
        signed_disagreement = torch.zeros_like(axial_angle)

    root = geometry["root"]
    strand_tip = geometry["strand_tip"]
    shell_tip = geometry["shell_tip"]
    residual = geometry["residual"]
    original_scale = field.original_scaling.detach().cpu().numpy()
    scale_max = original_scale.max(axis=1)
    scale_min = original_scale.min(axis=1)
    scale_aspect = scale_max / np.maximum(scale_min, 1e-12)
    barycentric_shift = (
        field.current_barycentric - field.barycentric
    ).norm(dim=-1)
    direction_normal_cosine = torch.sum(direction * normal, dim=-1)
    shell_delta = field.structured_delta_gain[:, 0]
    strand_delta = field.structured_delta_gain[:, 1]
    shell_opacity = field.structured_opacity_gain[:, 0]
    strand_opacity = field.structured_opacity_gain[:, 1]
    residual_scaling = field.residual_scaling.detach()
    shell_target_scaling = torch.stack(
        [
            field.shell_length / (2.0 * max(int(shell_samples), 1)),
            field.radius,
            field.radius,
        ],
        dim=-1,
    )
    strand_target_scaling = torch.stack(
        [
            field.strand_length / (2.0 * max(int(strand_samples), 1)),
            field.radius,
            field.radius,
        ],
        dim=-1,
    )
    shell_effective_scaling = torch.lerp(
        residual_scaling, shell_target_scaling, shell_delta[:, None]
    )
    strand_effective_scaling = torch.lerp(
        residual_scaling, strand_target_scaling, strand_delta[:, None]
    )

    def scale_audit(scaling: torch.Tensor, mask: torch.Tensor) -> dict[str, object]:
        selected = scaling[mask]
        if selected.numel() == 0:
            return {"count": 0}
        ordered = torch.sort(selected, dim=-1).values
        aspect = ordered[:, -1] / ordered[:, 0].clamp_min(1e-12)
        return {
            "count": int(selected.shape[0]),
            "minimum_axis": _quantiles(ordered[:, 0].detach().cpu().numpy()),
            "maximum_axis": _quantiles(ordered[:, -1].detach().cpu().numpy()),
            "aspect_ratio": _quantiles(aspect.detach().cpu().numpy()),
        }
    report = {
        "scene_scale": scene_scale,
        "active_sources": {
            "shell": int(shell_active.sum()),
            "strand": int(strand_active.sum()),
            "residual": int(active[:, 2].sum()),
        },
        "root_distribution_all": _root_distribution(
            roots, face_index, np.ones(len(roots), dtype=bool), scene_scale
        ),
        "root_distribution_shell": _root_distribution(
            roots, face_index, shell_active, scene_scale
        ),
        "root_distribution_strand": _root_distribution(
            roots, face_index, strand_active, scene_scale
        ),
        "direction_field": {
            "neighbor_graph_shape": list(neighbor_index.shape),
            "neighbor_graph_loaded": bool(has_neighbor_graph),
            "neighbor_axial_angle_deg": _quantiles(
                axial_angle.detach().cpu().numpy()
            ),
            "signed_neighbor_disagreement": _quantiles(
                signed_disagreement.detach().cpu().numpy()
            ),
            "active_strand_pair_count": int(active_pair.sum().detach().cpu()),
            "direction_normal_cosine": _quantiles(
                direction_normal_cosine[strand_active_torch].detach().cpu().numpy()
            ),
            "active_strand_inward_fraction": float(
                (direction_normal_cosine[strand_active_torch] < 0.0).float().mean().cpu()
            ) if int(strand_active.sum()) else 0.0,
        },
        "strand_geometry": {
            "parameter_length": _quantiles(
                field.strand_length[strand_active_torch].detach().cpu().numpy()
            ),
            "deployed_root_tip_distance": _quantiles(
                torch.linalg.vector_norm(
                    strand_tip[strand_active_torch] - root[strand_active_torch],
                    dim=-1,
                ).detach().cpu().numpy()
            ),
            "teacher_to_target_distance": _quantiles(
                torch.linalg.vector_norm(
                    geometry["strand_target"][strand_active_torch]
                    - residual[strand_active_torch],
                    dim=-1,
                ).detach().cpu().numpy()
            ),
            "deployed_to_target_error": _quantiles(
                torch.linalg.vector_norm(
                    geometry["strand_target"][strand_active_torch]
                    - strand_tip[strand_active_torch],
                    dim=-1,
                ).detach().cpu().numpy()
            ),
            "bend_magnitude": _quantiles(
                torch.linalg.vector_norm(
                    field.bend_local[strand_active_torch], dim=-1
                ).detach().cpu().numpy()
            ),
            "cubic_bend_magnitude": _quantiles(
                torch.linalg.vector_norm(
                    field.bend_cubic_local[strand_active_torch], dim=-1
                ).detach().cpu().numpy()
            ),
        },
        "shell_geometry": {
            "parameter_length": _quantiles(
                field.shell_length[shell_active_torch].detach().cpu().numpy()
            ),
            "deployed_root_tip_distance": _quantiles(
                torch.linalg.vector_norm(
                    shell_tip[shell_active_torch] - root[shell_active_torch], dim=-1
                ).detach().cpu().numpy()
            ),
        },
        "gaussian_scaffold": {
            "original_scale_max": _quantiles(scale_max),
            "original_scale_min": _quantiles(scale_min),
            "original_scale_aspect_ratio": _quantiles(scale_aspect),
            "radius": _quantiles(field.radius.detach().cpu().numpy()),
            "opacity": _quantiles(field.opacity.detach().cpu().numpy()),
            "residual_to_root_distance": _quantiles(
                torch.linalg.vector_norm(residual - root, dim=-1).detach().cpu().numpy()
            ),
        },
        "effective_covariance": {
            "shell_active": scale_audit(shell_effective_scaling, shell_active_torch),
            "strand_active": scale_audit(strand_effective_scaling, strand_active_torch),
        },
        "optimization_state": {
            "barycentric_root_shift": _quantiles(
                barycentric_shift.detach().cpu().numpy()
            ),
            "structured_delta_shell_active": {
                "quantiles": _quantiles(shell_delta[shell_active_torch].detach().cpu().numpy()),
                "fractions": _threshold_fractions(shell_delta[shell_active_torch].detach().cpu().numpy()),
            },
            "structured_delta_strand_active": {
                "quantiles": _quantiles(strand_delta[strand_active_torch].detach().cpu().numpy()),
                "fractions": _threshold_fractions(strand_delta[strand_active_torch].detach().cpu().numpy()),
            },
            "structured_opacity_shell_active": _quantiles(
                shell_opacity[shell_active_torch].detach().cpu().numpy()
            ),
            "structured_opacity_strand_active": _quantiles(
                strand_opacity[strand_active_torch].detach().cpu().numpy()
            ),
            "effective_deployed_mass": {
                # Geometry realization is the relevant quantity when teacher
                # optical thickness is transferred rather than additively
                # gated.  Keep both measures explicit for auditability.
                "shell_geometry_mean": float(
                    shell_delta[shell_active_torch].mean().cpu()
                ) if int(shell_active.sum()) else 0.0,
                "strand_geometry_mean": float(
                    strand_delta[strand_active_torch].mean().cpu()
                ) if int(strand_active.sum()) else 0.0,
                "shell_opacity_weighted_mean": float(
                    (shell_delta[shell_active_torch] * shell_opacity[shell_active_torch]).mean().cpu()
                ) if int(shell_active.sum()) else 0.0,
                "strand_opacity_weighted_mean": float(
                    (strand_delta[strand_active_torch] * strand_opacity[strand_active_torch]).mean().cpu()
                ) if int(strand_active.sum()) else 0.0,
            },
        },
    }
    return report


def _geometry_quality_gates(
    distribution: dict[str, object],
    active_tip_support: float,
) -> dict[str, object]:
    """Return explicit diagnostic gates, not a replacement for paper metrics."""

    direction = distribution["direction_field"]
    roots = distribution["root_distribution_strand"]
    optimization = distribution["optimization_state"]
    strand_delta = optimization["structured_delta_strand_active"]
    values = {
        "neighbor_graph_reconstructed": (
            1.0 if direction["neighbor_graph_loaded"] else 0.0,
            ">=",
            1.0,
        ),
        "signed_neighbor_disagreement": (
            float(direction["signed_neighbor_disagreement"]["mean"]),
            "<=",
            0.15,
        ),
        "active_strand_inward_fraction": (
            float(direction["active_strand_inward_fraction"]),
            "<=",
            0.02,
        ),
        "near_duplicate_root_fraction": (
            float(roots["near_duplicate_fraction_scene_1e-4"]),
            "<=",
            0.02,
        ),
        "median_structured_deployment": (
            float(strand_delta["quantiles"]["p50"]),
            ">=",
            0.35,
        ),
        "effective_deployed_mass": (
            float(optimization["effective_deployed_mass"]["strand_geometry_mean"]),
            ">=",
            0.35,
        ),
        "multiview_active_tip_mask_support": (
            float(active_tip_support),
            ">=",
            0.85,
        ),
    }
    gates = {}
    for name, (value, relation, threshold) in values.items():
        passed = value >= threshold if relation == ">=" else value <= threshold
        gates[name] = {
            "value": value,
            "relation": relation,
            "threshold": threshold,
            "passed": bool(passed),
        }
    failed = [name for name, item in gates.items() if not item["passed"]]
    return {
        "status": "pass" if not failed else "structurally_unresolved",
        "failed": failed,
        "note": "Heuristic engineering gates; report alongside, never instead of, held-out image metrics.",
        "gates": gates,
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


def _draw_projection(
    image_path: Path,
    camera,
    geometry,
    out_path: Path,
    max_arrows: int,
    active_routes: np.ndarray,
) -> None:
    image = Image.open(image_path).convert("RGB").resize(
        (camera.width, camera.height), Image.Resampling.LANCZOS
    )
    draw = ImageDraw.Draw(image)
    active_indices = np.flatnonzero(active_routes.any(axis=1))
    if len(active_indices) > max_arrows:
        sample = np.linspace(0, len(active_indices) - 1, max_arrows).astype(np.int64)
        indices = active_indices[sample]
    else:
        indices = active_indices
    projected = {}
    validity = {}
    for key in ("root", "residual", "shell_tip", "strand_tip"):
        projected[key], validity[key] = _project(geometry[key], camera)
    for index in indices:
        root = projected["root"][index]
        if active_routes[index, 0] and validity["root"][index] and validity["shell_tip"][index]:
            tip = projected["shell_tip"][index]
            draw.line([tuple(root), tuple(tip)], fill=(0, 220, 255), width=1)
        if active_routes[index, 1] and validity["root"][index] and validity["strand_tip"][index]:
            tip = projected["strand_tip"][index]
            draw.line([tuple(root), tuple(tip)], fill=(255, 80, 20), width=1)
        if active_routes[index, 2] and validity["residual"][index]:
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
    motion = DifferentiableSurfaceScaffold(args.stage1_npz, device=device)
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
    active_routes = field.route_active_gate.detach().cpu().numpy() > 0.5
    shell_active = active_routes[:, 0]
    strand_active = active_routes[:, 1]
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
    _write_line_ply(
        out_dir / "shell_active_root_tip.ply",
        rest_numpy["root"][shell_active],
        rest_numpy["shell_tip"][shell_active],
        ((40, 255, 80), (0, 220, 255)),
    )
    _write_line_ply(
        out_dir / "strand_active_root_tip.ply",
        rest_numpy["root"][strand_active],
        rest_numpy["strand_tip"][strand_active],
        ((40, 255, 80), (255, 80, 20)),
    )
    occupancy_report = _write_occupancy_ply(
        out_dir / "scalp_occupancy.ply", rest_numpy["root"]
    )
    strand_occupancy_report = _write_occupancy_ply(
        out_dir / "strand_active_scalp_occupancy.ply",
        rest_numpy["root"][strand_active],
    )

    support = {"root": [], "residual": [], "shell_tip": [], "strand_tip": []}
    active_support = {key: [] for key in support}
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
                if key == "strand_tip":
                    route_active = field.route_active_gate[:, 1] > 0.5
                elif key == "shell_tip":
                    route_active = field.route_active_gate[:, 0] > 0.5
                elif key == "residual":
                    route_active = field.route_active_gate[:, 2] > 0.5
                else:
                    route_active = (field.route_active_gate > 0.5).any(dim=1)
                active_valid = valid & route_active
                active_values = sampled[active_valid]
                active_support[key].append(
                    float((active_values >= 0.5).float().mean().cpu())
                    if active_values.numel()
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
                active_routes,
            )
            projection_paths.append(projection_path)

    cosine = np.concatenate(direction_normal_cosine)
    barycentric = field.current_barycentric.detach().cpu().numpy()
    gain = field.structured_delta_gain.detach().cpu().numpy()
    gate = field.strand_visibility_gate.detach().cpu().numpy()
    shell_samples = int(metadata.get("shell_samples", 2))
    strand_samples = int(metadata.get("strand_samples", 5))
    distribution_audit = _distribution_audit(
        field,
        rest_geometry,
        shell_samples=shell_samples,
        strand_samples=strand_samples,
    )
    active_tip_support = float(np.mean(active_support["strand_tip"]))
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
                "active_mean": float(np.mean(active_support[key])),
                "active_per_view": active_support[key],
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
        "distribution_audit": distribution_audit,
        "geometry_quality_gates": _geometry_quality_gates(
            distribution_audit, active_tip_support
        ),
        "scalp_occupancy": occupancy_report,
        "strand_active_scalp_occupancy": strand_occupancy_report,
        "checkpoint_metadata": metadata,
        "files": {
            "shell_root_tip": str(out_dir / "shell_root_tip.ply"),
            "strand_root_tip": str(out_dir / "strand_root_tip.ply"),
            "shell_active_root_tip": str(out_dir / "shell_active_root_tip.ply"),
            "strand_active_root_tip": str(out_dir / "strand_active_root_tip.ply"),
            "scalp_occupancy": str(out_dir / "scalp_occupancy.ply"),
            "strand_active_scalp_occupancy": str(
                out_dir / "strand_active_scalp_occupancy.ply"
            ),
            "projections": [str(path) for path in projection_paths],
        },
    }
    with open(out_dir / "geometry_audit.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
