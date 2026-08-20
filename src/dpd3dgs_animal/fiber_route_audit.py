from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from .config import PipelineConfig
from .fiber import (
    HARD_ROUTE_POLICIES,
    ROUTE_NAMES,
    attach_fixed_gaussian_base,
    create_unified_fiber_field,
    mass_preserving_route_ids,
    partition_binding_cache,
)
from .fiber_evaluate import _frame_metrics, _select_frame_indices
from .fiber_optimize import _render
from .scaffold import (
    DifferentiableSurfaceScaffold,
    _frame_paths,
    _load_gt_frame_torch,
    _resolve_device,
    _resolve_render_size,
)
from .observations import resolve_observations


@dataclass
class FiberRouteAuditArtifacts:
    out_dir: Path
    report_json: Path
    plot_png: Path


def audit_unified_fiber_routes(
    stage1_npz: str | Path,
    gaussian_ply: str | Path,
    checkpoint_pt: str | Path,
    frame_dir: str | Path,
    out_dir: str | Path,
    cfg: PipelineConfig,
    *,
    renderer: str | None = None,
    render_size: tuple[int, int] | None = None,
    max_frames: int | None = None,
    frame_start: int = 0,
    frame_stride: int = 1,
    camera_manifest: str | Path | None = None,
    fixed_base_gaussian_ply: str | Path | None = None,
) -> FiberRouteAuditArtifacts:
    """Audit whether routing confidence agrees with measurable rendering impact.

    Route probabilities have no ground-truth class labels, so they cannot be
    interpreted as calibrated epistemic probabilities. This audit reports their
    sharpness and spatial coherence, then compares mean probability mass with a
    leave-one-route-out rendering contribution.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(cfg.device)
    renderer_name = str(renderer or cfg.fiber_renderer).lower()
    payload = torch.load(checkpoint_pt, map_location=device)
    metadata = payload.get("metadata", {})
    point_count = int(metadata.get("point_count", cfg.fiber_max_points))
    point_sampling_mode = str(
        metadata.get("point_sampling_mode", cfg.fiber_point_sampling_mode)
    )
    exact_vertex_binding = bool(
        metadata.get("exact_vertex_binding", cfg.fiber_exact_vertex_binding)
    )
    split_fixed_base = bool(
        metadata.get("split_fixed_base", cfg.fiber_split_fixed_base)
    )
    fixed_base_source = Path(
        fixed_base_gaussian_ply
        or metadata.get("fixed_base_gaussian_ply")
        or gaussian_ply
    )
    fixed_base_max_scale_fraction = float(
        cfg.fiber_fixed_base_max_scale_fraction
    )
    if fixed_base_max_scale_fraction <= 0.0:
        fixed_base_max_scale_fraction = float(
            metadata.get("fixed_base_max_scale_fraction", 0.0)
        )
    hard_route_policy = str(
        metadata.get("hard_route_policy", cfg.fiber_hard_route_policy)
    )
    if hard_route_policy not in HARD_ROUTE_POLICIES:
        raise ValueError(
            f"Unknown checkpoint hard-route policy {hard_route_policy!r}"
        )

    motion = DifferentiableSurfaceScaffold(stage1_npz, device=device)
    motion.joints.requires_grad_(False)
    with np.load(stage1_npz, allow_pickle=False) as stage1_payload:
        scalp_face_indices = (
            stage1_payload["scalp_face_indices"].astype(np.int64)
            if "scalp_face_indices" in stage1_payload.files
            else None
        )
    field = create_unified_fiber_field(
        gaussian_ply,
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
            cfg.fiber_shell_propagated_direction_weight
        ),
        root_barycentric_max_delta=float(
            metadata.get(
                "root_barycentric_max_delta",
                cfg.fiber_root_barycentric_max_delta,
            )
        ),
        expert_sh_max_delta=float(
            metadata.get("expert_sh_max_delta", cfg.fiber_expert_sh_max_delta)
        ),
        expert_sh_degree=int(
            metadata.get("expert_sh_degree", cfg.fiber_expert_sh_degree)
        ),
        initial_residual_trust=float(cfg.fiber_initial_residual_trust),
        scalp_face_indices=scalp_face_indices,
        binding_cache=(
            partition_binding_cache(cfg.fiber_binding_cache, "foreground")
            if split_fixed_base
            else cfg.fiber_binding_cache
        ),
    )
    if split_fixed_base:
        attach_fixed_gaussian_base(
            field,
            fixed_base_source,
            motion.rest_surface_vertices.detach().cpu().numpy(),
            motion.surface_faces.detach().cpu().numpy(),
            device=device,
            point_sampling_mode=point_sampling_mode,
            exact_vertex_binding=exact_vertex_binding,
            binding_mode=str(metadata.get("binding_mode", cfg.fiber_binding_mode)),
            source_mask_threshold=float(
                metadata.get("source_mask_threshold", cfg.fiber_source_mask_threshold)
            ),
            source_min_opacity=float(
                metadata.get("source_min_opacity", cfg.fiber_source_min_opacity)
            ),
            residual_max_scale_fraction=fixed_base_max_scale_fraction,
            scalp_face_indices=scalp_face_indices,
            binding_cache=cfg.fiber_binding_cache,
        )
    checkpoint_state = payload["state_dict"]
    checkpoint_shell_gate = checkpoint_state.get("shell_visibility_gate")
    if isinstance(checkpoint_shell_gate, torch.Tensor):
        field.shell_visibility_gate = torch.empty_like(
            checkpoint_shell_gate, device=field.route_logits.device
        )
    checkpoint_gate = checkpoint_state.get("strand_visibility_gate")
    if isinstance(checkpoint_gate, torch.Tensor):
        field.strand_visibility_gate = torch.empty_like(
            checkpoint_gate, device=field.route_logits.device
        )
    incompatible = field.load_state_dict(checkpoint_state, strict=False)
    unexpected_missing = set(incompatible.missing_keys) - {
        "expert_color_delta",
        "bend_cubic_local",
        "structured_delta_raw",
        "structured_opacity_raw",
        "shell_visibility_gate",
        "strand_visibility_gate",
        "route_active_gate",
        "route_neighbor_index",
        "carrier_logits",
        "carrier_root_tip_raw",
        "initial_carrier_probabilities",
        "initial_carrier_root_tip",
        "rest_surface_frame",
        "barycentric_offset_raw",
        "strand_root_occupancy",
        "expert_sh_delta_raw",
    }
    if unexpected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint state mismatch: "
            f"missing={sorted(unexpected_missing)}, "
            f"unexpected={sorted(incompatible.unexpected_keys)}"
        )
    if "structured_delta_raw" in incompatible.missing_keys:
        with torch.no_grad():
            field.structured_delta_raw.fill_(1.0)
    field.eval()

    frame_paths = _frame_paths(frame_dir)
    width, height = _resolve_render_size(render_size, frame_paths, stage1_npz)
    observation_set = resolve_observations(
        frame_paths,
        stage1_npz,
        motion.rest_surface_vertices.detach().cpu().numpy(),
        width,
        height,
        camera_manifest=camera_manifest,
    )
    cameras = observation_set.cameras
    motion_indices = observation_set.motion_indices
    invalid_motion = [
        index
        for index, motion_index in enumerate(motion_indices)
        if motion_index >= int(motion.joints.shape[0])
    ]
    if invalid_motion:
        first = invalid_motion[0]
        raise ValueError(
            f"Observation {frame_paths[first].name!r} maps to motion state "
            f"{motion_indices[first]}, but Stage 1 only has "
            f"{int(motion.joints.shape[0])} states"
        )
    frame_indices = _select_frame_indices(
        len(frame_paths),
        frame_start,
        frame_stride,
        max_frames,
    )
    if not frame_indices:
        raise ValueError("No route-audit frames selected")

    probabilities = field.route_probabilities(
        temperature=cfg.fiber_final_temperature
    ).detach()
    hard_labels = (
        probabilities.argmax(dim=-1)
        if hard_route_policy == "argmax"
        else mass_preserving_route_ids(probabilities)
    )
    confidence = probabilities.max(dim=-1).values
    sorted_probabilities = probabilities.sort(dim=-1, descending=True).values
    margin = sorted_probabilities[:, 0] - sorted_probabilities[:, 1]
    normalized_entropy = (
        -torch.sum(
            probabilities * torch.log(probabilities.clamp_min(1e-8)), dim=-1
        )
        / np.log(len(ROUTE_NAMES))
    )
    mean_probabilities = probabilities.mean(dim=0).cpu().numpy()
    hard_fractions_tensor = torch.bincount(
        hard_labels, minlength=len(ROUTE_NAMES)
    ).float()
    hard_fractions_tensor /= hard_fractions_tensor.sum().clamp_min(1.0)
    hard_fractions = hard_fractions_tensor.cpu().numpy()

    with torch.no_grad():
        roots, _tangent, _bitangent, _normal = field.surface_frame(
            motion.rest_surface_vertices, motion.surface_faces
        )
    spatial = _spatial_route_statistics(
        roots.detach().cpu().numpy(),
        probabilities.cpu().numpy(),
        hard_labels.cpu().numpy(),
        hard_fractions,
    )

    full_metrics: list[dict[str, float]] = []
    hard_metrics: list[dict[str, float]] = []
    ablated_metrics: dict[str, list[dict[str, float]]] = {
        route: [] for route in ROUTE_NAMES
    }
    with torch.no_grad():
        for frame_index in frame_indices:
            camera = cameras[frame_index]
            _tet_nodes, surface_vertices, _joints = motion.driven_points(
                motion_indices[frame_index]
            )
            target = _load_gt_frame_torch(
                frame_paths[frame_index], width, height, device
            )
            soft_primitives = field.primitives(
                surface_vertices,
                motion.surface_faces,
                shell_samples=cfg.fiber_shell_samples,
                strand_samples=cfg.fiber_strand_samples,
                temperature=cfg.fiber_final_temperature,
                hard_route=False,
                fin_aspect_ratio=cfg.fiber_fin_aspect_ratio,
                additive_teacher=cfg.fiber_additive_teacher_mode,
                teacher_opacity_transfer=cfg.fiber_teacher_opacity_transfer,
            )
            full_prediction = _render(
                soft_primitives, camera, cfg, renderer_name
            )
            full_metrics.append(_frame_metrics(full_prediction, target))

            hard_primitives = field.primitives(
                surface_vertices,
                motion.surface_faces,
                shell_samples=cfg.fiber_shell_samples,
                strand_samples=cfg.fiber_strand_samples,
                temperature=cfg.fiber_final_temperature,
                hard_route=True,
                hard_route_policy=hard_route_policy,
                fin_aspect_ratio=cfg.fiber_fin_aspect_ratio,
                additive_teacher=cfg.fiber_additive_teacher_mode,
                teacher_opacity_transfer=cfg.fiber_teacher_opacity_transfer,
            )
            hard_prediction = _render(
                hard_primitives, camera, cfg, renderer_name
            )
            hard_metrics.append(_frame_metrics(hard_prediction, target))

            for route_index, route in enumerate(ROUTE_NAMES):
                keep = soft_primitives.route_id != route_index
                ablated = replace(
                    soft_primitives,
                    opacity=torch.where(
                        keep,
                        soft_primitives.opacity,
                        torch.zeros_like(soft_primitives.opacity),
                    ),
                )
                prediction = _render(ablated, camera, cfg, renderer_name)
                ablated_metrics[route].append(_frame_metrics(prediction, target))

    aggregate_full = _aggregate_metrics(full_metrics)
    aggregate_hard = _aggregate_metrics(hard_metrics)
    aggregate_ablated = {
        route: _aggregate_metrics(values)
        for route, values in ablated_metrics.items()
    }
    contribution = _route_contribution(aggregate_full, aggregate_ablated)
    normalized_impact = np.asarray(
        [
            contribution[route]["normalized_appearance_impact"]
            for route in ROUTE_NAMES
        ]
    )
    alignment_total_variation = 0.5 * float(
        np.abs(mean_probabilities - normalized_impact).sum()
    )

    confidence_values = confidence.cpu().numpy()
    margin_values = margin.cpu().numpy()
    entropy_values = normalized_entropy.cpu().numpy()
    report = {
        "interpretation": {
            "calibration_status": "not_identifiable_without_route_class_labels",
            "probability_semantics": "optimization gate mass, not epistemic confidence",
            "contribution_definition": (
                "leave-one-route-out appearance and silhouette changes are "
                "reported separately; appearance impact is normalized from "
                "positive foreground-PSNR drops"
            ),
        },
        "checkpoint": str(checkpoint_pt),
        "renderer": renderer_name,
        "hard_route_policy": hard_route_policy,
        "camera_source": observation_set.source,
        "frame_indices": frame_indices,
        "motion_indices": [motion_indices[index] for index in frame_indices],
        "render_size": [width, height],
        "route_probability_mean": {
            route: float(mean_probabilities[index])
            for index, route in enumerate(ROUTE_NAMES)
        },
        "hard_route_fraction": {
            route: float(hard_fractions[index])
            for index, route in enumerate(ROUTE_NAMES)
        },
        "confidence": {
            "max_probability_mean": float(confidence_values.mean()),
            "max_probability_quantiles": _quantiles(confidence_values),
            "top1_top2_margin_mean": float(margin_values.mean()),
            "top1_top2_margin_quantiles": _quantiles(margin_values),
            "normalized_entropy_mean": float(entropy_values.mean()),
            "normalized_entropy_quantiles": _quantiles(entropy_values),
            "fraction_below_0_6": float((confidence_values < 0.6).mean()),
            "fraction_below_0_8": float((confidence_values < 0.8).mean()),
        },
        "spatial_coherence": spatial,
        "soft_full_metrics": aggregate_full,
        "hard_metrics": aggregate_hard,
        "hard_minus_soft_gap": {
            "psnr_drop": aggregate_full["foreground_psnr"]
            - aggregate_hard["foreground_psnr"],
            "mask_iou_drop": aggregate_full["mask_iou"]
            - aggregate_hard["mask_iou"],
        },
        "leave_one_route_out": aggregate_ablated,
        "route_contribution": contribution,
        "probability_appearance_contribution_total_variation": (
            alignment_total_variation
        ),
    }
    report_json = out_dir / "route_audit.json"
    with open(report_json, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    plot_png = out_dir / "route_audit.png"
    _plot_route_audit(
        confidence_values,
        margin_values,
        mean_probabilities,
        hard_fractions,
        normalized_impact,
        contribution,
        plot_png,
    )
    return FiberRouteAuditArtifacts(out_dir, report_json, plot_png)


def _aggregate_metrics(values: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: float(np.mean([value[key] for value in values]))
        for key in values[0]
    }


def _route_contribution(
    full: dict[str, float],
    ablated: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    raw: dict[str, float] = {}
    result: dict[str, dict[str, float]] = {}
    full_proxy = full["foreground_l1"] + 10.0 * full["mask_mae"]
    for route in ROUTE_NAMES:
        route_metrics = ablated[route]
        route_proxy = (
            route_metrics["foreground_l1"] + 10.0 * route_metrics["mask_mae"]
        )
        raw[route] = route_proxy - full_proxy
        result[route] = {
            "objective_proxy_increase": raw[route],
            "foreground_l1_increase": route_metrics["foreground_l1"]
            - full["foreground_l1"],
            "psnr_drop": full["foreground_psnr"]
            - route_metrics["foreground_psnr"],
            "mask_mae_increase": route_metrics["mask_mae"] - full["mask_mae"],
            "mask_iou_drop": full["mask_iou"] - route_metrics["mask_iou"],
            "mask_f1_drop": full["mask_f1"] - route_metrics["mask_f1"],
        }
    positive_appearance_sum = sum(
        max(result[route]["psnr_drop"], 0.0) for route in ROUTE_NAMES
    )
    positive_silhouette_sum = sum(
        max(result[route]["mask_iou_drop"], 0.0) for route in ROUTE_NAMES
    )
    for route in ROUTE_NAMES:
        result[route]["normalized_appearance_impact"] = (
            max(result[route]["psnr_drop"], 0.0) / positive_appearance_sum
            if positive_appearance_sum > 0.0
            else 0.0
        )
        result[route]["normalized_silhouette_impact"] = (
            max(result[route]["mask_iou_drop"], 0.0) / positive_silhouette_sum
            if positive_silhouette_sum > 0.0
            else 0.0
        )
    return result


def _spatial_route_statistics(
    roots: np.ndarray,
    probabilities: np.ndarray,
    hard_labels: np.ndarray,
    hard_fractions: np.ndarray,
) -> dict[str, float | int]:
    from scipy.spatial import cKDTree

    neighbor_count = min(8, max(len(roots) - 1, 1))
    _distance, indices = cKDTree(roots).query(roots, k=neighbor_count + 1)
    neighbors = indices[:, 1:]
    same_label = hard_labels[neighbors] == hard_labels[:, None]
    probability_l1 = np.abs(
        probabilities[neighbors] - probabilities[:, None, :]
    ).mean(axis=(1, 2))
    random_baseline = float(np.square(hard_fractions).sum())
    consistency = float(same_label.mean())
    return {
        "neighbors": neighbor_count,
        "same_hard_route_fraction": consistency,
        "random_label_baseline": random_baseline,
        "excess_over_random": consistency - random_baseline,
        "neighbor_probability_l1_mean": float(probability_l1.mean()),
    }


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        name: float(np.quantile(values, quantile))
        for name, quantile in (
            ("q10", 0.10),
            ("q25", 0.25),
            ("q50", 0.50),
            ("q75", 0.75),
            ("q90", 0.90),
        )
    }


def _plot_route_audit(
    confidence: np.ndarray,
    margin: np.ndarray,
    mean_probabilities: np.ndarray,
    hard_fractions: np.ndarray,
    normalized_impact: np.ndarray,
    contribution: dict[str, dict[str, float]],
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].hist(confidence, bins=30, alpha=0.75, label="max probability")
    axes[0].hist(margin, bins=30, alpha=0.65, label="top1-top2 margin")
    axes[0].set_xlabel("routing score")
    axes[0].set_ylabel("source Gaussian count")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    x = np.arange(len(ROUTE_NAMES))
    width = 0.25
    axes[1].bar(x - width, mean_probabilities, width, label="soft mass")
    axes[1].bar(x, hard_fractions, width, label="hard fraction")
    axes[1].bar(x + width, normalized_impact, width, label="LOO impact")
    axes[1].set_xticks(x, ROUTE_NAMES)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.2)

    psnr_drop = [contribution[route]["psnr_drop"] for route in ROUTE_NAMES]
    axes[2].bar(ROUTE_NAMES, psnr_drop)
    axes[2].axhline(0.0, color="black", linewidth=1)
    axes[2].set_ylabel("foreground PSNR drop when route is removed (dB)")
    axes[2].grid(axis="y", alpha=0.2)
    figure.suptitle("Unified fiber route confidence and contribution audit")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
