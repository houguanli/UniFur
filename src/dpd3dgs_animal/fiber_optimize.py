from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .config import PipelineConfig
from .fiber import (
    CARRIER_NAMES,
    HARD_ROUTE_POLICIES,
    ROUTE_NAMES,
    UnifiedFiberField,
    apply_fin_view_gate,
    create_unified_fiber_field,
    render_fiber_primitives,
)
from .optimize import (
    DifferentiableSkeletonTetModel,
    _frame_paths,
    _load_gt_frame_torch,
    _resolve_device,
    _resolve_render_size,
    differentiable_render_loss,
)
from .observations import resolve_observations


@dataclass
class FiberOptimizationArtifacts:
    out_dir: Path
    checkpoint_pt: Path
    state_npz: Path
    report_json: Path
    metrics_jsonl: Path
    loss_curve_png: Path


_RESIDUAL_BOOTSTRAP_KEYS = (
    "color_logits",
    "opacity_logits",
    "residual_offset_local",
    "residual_log_scale_delta",
    "residual_rotation_raw",
)


def _load_residual_bootstrap_checkpoint(
    field: UnifiedFiberField,
    checkpoint_path: str | Path,
    bootstrap_route_mass: list[float] | tuple[float, float, float] | None = None,
    bootstrap_route_temperature: float = 1.0,
) -> dict[str, object]:
    """Transfer only the photometric residual-Gaussian scaffold.

    A residual-only multi-view fit supplies a valid 3DGS appearance bootstrap
    for the unified model.  Route logits, structured geometry, and route
    buffers deliberately remain freshly initialized, preventing a residual-only
    checkpoint from reintroducing a collapsed routing prior.
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Residual bootstrap checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise ValueError("Residual bootstrap checkpoint must contain a state_dict")
    metadata = payload.get("metadata", {})
    if isinstance(metadata, dict):
        representation = metadata.get("representation")
        if representation is not None and representation != "residual_only":
            raise ValueError(
                "Residual bootstrap checkpoint must be trained with "
                f"representation='residual_only', got {representation!r}"
            )
        point_count = metadata.get("point_count")
        if point_count is not None and int(point_count) != field.point_count:
            raise ValueError(
                "Residual bootstrap point count does not match current field: "
                f"{point_count} != {field.point_count}"
            )

    source_state = payload["state_dict"]
    target_state = field.state_dict()
    for key in _RESIDUAL_BOOTSTRAP_KEYS:
        source = source_state.get(key)
        target = target_state.get(key)
        if not isinstance(source, torch.Tensor) or not isinstance(target, torch.Tensor):
            raise ValueError(f"Residual bootstrap checkpoint lacks tensor {key!r}")
        if tuple(source.shape) != tuple(target.shape):
            raise ValueError(
                f"Residual bootstrap tensor {key!r} shape mismatch: "
                f"{tuple(source.shape)} != {tuple(target.shape)}"
            )
        target_state[key] = source.to(device=target.device, dtype=target.dtype)
    field.load_state_dict(target_state, strict=True)
    routing_bootstrap_mass = None
    routing_bootstrap_temperature = None
    if bootstrap_route_mass is not None:
        requested_mass = torch.as_tensor(
            bootstrap_route_mass,
            dtype=field.route_logits.dtype,
            device=field.route_logits.device,
        ).reshape(-1)
        if requested_mass.numel() != len(ROUTE_NAMES):
            raise ValueError(
                "fiber_bootstrap_route_mass must contain effective "
                f"[{', '.join(ROUTE_NAMES)}] mass"
            )
        if not torch.isfinite(requested_mass).all() or torch.any(requested_mass <= 0):
            raise ValueError("fiber_bootstrap_route_mass must be finite and positive")
        if float(bootstrap_route_temperature) <= 0.0:
            raise ValueError("bootstrap_route_temperature must be positive")
        requested_mass = requested_mass / requested_mass.sum()
        residual_index = ROUTE_NAMES.index("residual")
        trust = field.residual_trust.detach().reshape(-1, 1)
        if torch.any(requested_mass[residual_index] <= trust[:, 0]):
            raise ValueError(
                "fiber_bootstrap_route_mass residual fraction must exceed the "
                "initial residual trust"
            )
        base_mass = requested_mass[None, :].expand(field.point_count, -1).clone()
        base_mass[:, :residual_index] /= (1.0 - trust).clamp_min(1e-6)
        base_mass[:, residual_index] = (
            requested_mass[residual_index] - trust[:, 0]
        ) / (1.0 - trust[:, 0]).clamp_min(1e-6)
        with torch.no_grad():
            # route_probabilities uses softmax(logits / temperature).  Scale
            # the logits here so the requested mass is exact at deployment,
            # rather than becoming spuriously shell-dominant as temperature
            # is annealed below one.
            field.route_logits.copy_(
                float(bootstrap_route_temperature)
                * torch.log(base_mass.clamp_min(1e-6))
            )
            field.initial_route_probabilities.copy_(
                requested_mass[None, :].expand_as(field.initial_route_probabilities)
            )
        routing_bootstrap_mass = [float(value) for value in requested_mass.cpu()]
        routing_bootstrap_temperature = float(bootstrap_route_temperature)
    return {
        "checkpoint": str(path),
        "loaded_keys": list(_RESIDUAL_BOOTSTRAP_KEYS),
        "source_representation": metadata.get("representation")
        if isinstance(metadata, dict)
        else None,
        "routing_bootstrap_mass": routing_bootstrap_mass,
        "routing_bootstrap_temperature": routing_bootstrap_temperature,
    }


def _freeze_residual_teacher_scaffold(field: UnifiedFiberField) -> None:
    """Freeze the exact residual checkpoint while structured deltas learn."""

    for name in _RESIDUAL_BOOTSTRAP_KEYS:
        parameter = getattr(field, name)
        parameter.requires_grad_(False)
    # Keep the residual expert's appearance delta exactly zero while allowing
    # shell and strand appearance residuals to specialize.
    field.freeze_residual_teacher = True
    field.expert_color_delta.register_hook(
        lambda gradient: torch.cat(
            [gradient[:, :2], torch.zeros_like(gradient[:, 2:3])], dim=1
        )
    )


def _resolve_fiber_point_budget(
    cfg: PipelineConfig,
    render_size: tuple[int, int],
    *,
    explicit_max_points: int | None = None,
) -> int:
    """Resolve a reproducible source-Gaussian budget for one experiment."""

    if explicit_max_points is not None:
        if int(explicit_max_points) < 0:
            raise ValueError("max_points override must be non-negative")
        return int(explicit_max_points)
    hard_max = int(cfg.fiber_max_points)
    if hard_max < 0:
        raise ValueError("fiber_max_points must be non-negative")
    mode = str(cfg.fiber_capacity_mode).lower()
    if mode == "fixed":
        return hard_max
    if mode != "pixel_adaptive":
        raise ValueError(
            "fiber_capacity_mode must be 'fixed' or 'pixel_adaptive'"
        )
    pixels_per_point = float(cfg.fiber_target_pixels_per_point)
    if not math.isfinite(pixels_per_point) or pixels_per_point <= 0.0:
        raise ValueError("fiber_target_pixels_per_point must be positive")
    minimum = int(cfg.fiber_min_points)
    if minimum < 0:
        raise ValueError("fiber_min_points must be non-negative")
    width, height = (int(render_size[0]), int(render_size[1]))
    if width <= 0 or height <= 0:
        raise ValueError("render_size must be positive")
    adaptive = max(minimum, int(math.ceil(width * height / pixels_per_point)))
    return min(adaptive, hard_max) if hard_max > 0 else adaptive


def optimize_unified_fiber_stage2(
    stage1_npz: str | Path,
    gaussian_ply: str | Path,
    frame_dir: str | Path,
    out_dir: str | Path,
    cfg: PipelineConfig,
    *,
    steps: int | None = None,
    lr: float | None = None,
    max_points: int | None = None,
    renderer: str | None = None,
    render_size: tuple[int, int] | None = None,
    max_frames: int | None = None,
    frame_start: int = 0,
    frame_stride: int = 1,
    log_every: int | None = None,
    checkpoint_every: int | None = None,
    camera_manifest: str | Path | None = None,
    residual_bootstrap_checkpoint: str | Path | None = None,
) -> FiberOptimizationArtifacts:
    """Optimize a unified field on monocular or calibrated multi-view data.

    The first phase is an ordinary residual-Gaussian scaffold.  The second
    phase learns soft shell/strand/residual routing, and the last phase lowers
    routing temperature and jointly refines the structured primitives.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(cfg.device)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    renderer_name = str(renderer or cfg.fiber_renderer).lower()
    if renderer_name not in {"torch", "hairgs"}:
        raise ValueError("renderer must be 'torch' or 'hairgs'")
    if float(cfg.fiber_orientation_weight) > 0.0 and renderer_name != "hairgs":
        raise ValueError("fiber_orientation_weight requires the HairGS renderer")
    representation = str(cfg.fiber_representation).lower()
    if representation not in {"unified", "residual_only"}:
        raise ValueError(
            "fiber_representation must be 'unified' or 'residual_only'"
        )
    frame_paths = _frame_paths(frame_dir)
    width, height = _resolve_render_size(render_size, frame_paths, stage1_npz)
    point_budget = _resolve_fiber_point_budget(
        cfg,
        (width, height),
        explicit_max_points=max_points,
    )
    motion = DifferentiableSkeletonTetModel(stage1_npz, device=device)
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
        max_points=point_budget,
        point_sampling_mode=str(cfg.fiber_point_sampling_mode),
        exact_vertex_binding=bool(cfg.fiber_exact_vertex_binding),
        default_opacity=float(cfg.fiber_default_opacity),
        default_opacity_reference_points=int(
            cfg.fiber_default_opacity_reference_points
        ),
        neighbor_k=(
            int(cfg.fiber_route_neighbor_k) if representation == "unified" else 0
        ),
        initial_residual_trust=float(cfg.fiber_initial_residual_trust),
        initial_shell_length_scale=cfg.fiber_initial_shell_length_scale,
        initial_strand_length_scale=cfg.fiber_initial_strand_length_scale,
        initialize_direction_from_normal=bool(
            cfg.fiber_initialize_direction_from_normal
        ),
        scalp_face_indices=scalp_face_indices,
        binding_cache=cfg.fiber_binding_cache,
    )
    bootstrap_metadata = None
    if residual_bootstrap_checkpoint is not None:
        bootstrap_metadata = _load_residual_bootstrap_checkpoint(
            field,
            residual_bootstrap_checkpoint,
            bootstrap_route_mass=cfg.fiber_bootstrap_route_mass,
            bootstrap_route_temperature=cfg.fiber_final_temperature,
        )
    teacher_requested = representation == "unified" and (
        bool(cfg.fiber_freeze_residual_teacher)
        or float(cfg.fiber_teacher_nonregression_weight) > 0.0
    )
    if teacher_requested and bootstrap_metadata is None:
        raise ValueError(
            "A residual bootstrap checkpoint is required for the fixed-teacher "
            "or held-out non-regression configuration"
        )
    teacher_field: UnifiedFiberField | None = None
    if teacher_requested:
        teacher_field = copy.deepcopy(field).eval()
        for parameter in teacher_field.parameters():
            parameter.requires_grad_(False)
    if representation == "unified" and bool(cfg.fiber_freeze_residual_teacher):
        _freeze_residual_teacher_scaffold(field)

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
    camera_source = observation_set.source
    available_frames = len(frame_paths)
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
    if frame_start < 0:
        raise ValueError("frame_start must be non-negative")
    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    frame_indices = list(range(int(frame_start), available_frames, int(frame_stride)))
    frame_limit = max_frames if max_frames is not None else cfg.fiber_max_frames
    if frame_limit is not None and int(frame_limit) > 0:
        frame_indices = frame_indices[: int(frame_limit)]
    calibration_count = (
        int(cfg.fiber_calibration_frames) if representation == "unified" else 0
    )
    frame_indices, calibration_frame_indices = _split_training_and_calibration_frames(
        frame_indices, calibration_count
    )
    n_frames = len(frame_indices)
    if n_frames <= 0:
        raise ValueError(
            f"No frames available in {frame_dir} for start={frame_start}, "
            f"stride={frame_stride}"
        )
    ground_truth = [
        _load_gt_frame_torch(frame_paths[index], width, height, device)
        for index in frame_indices
    ]
    calibration_ground_truth = [
        _load_gt_frame_torch(frame_paths[index], width, height, device)
        for index in calibration_frame_indices
    ]
    if (
        representation == "unified"
        and float(cfg.fiber_teacher_nonregression_weight) > 0.0
        and not calibration_frame_indices
    ):
        raise ValueError(
            "fiber_teacher_nonregression_weight requires fiber_calibration_frames > 0"
        )
    orientation_targets = _load_orientation_targets(
        frame_paths, frame_indices, width, height, device, cfg.fiber_orientation_dir
    )

    visual_hull_frame_indices = frame_indices + calibration_frame_indices
    visual_hull_ground_truth = ground_truth + calibration_ground_truth
    visual_hull_vertices: list[torch.Tensor] = []
    if (
        representation == "unified"
        and int(cfg.fiber_visual_hull_update_every) > 0
    ):
        with torch.no_grad():
            for visual_frame in visual_hull_frame_indices:
                _tet_nodes, vertices, _joints = motion.driven_points(
                    motion_indices[visual_frame]
                )
                visual_hull_vertices.append(vertices)

    _validate_route_training_config(cfg)
    route_rng = np.random.default_rng(int(cfg.fiber_random_seed))

    base_lr = float(lr or cfg.optimize_lr)
    optimizer = torch.optim.Adam(
        _optimizer_parameter_groups(field, cfg, base_lr, representation), eps=1e-8
    )
    total_steps = int(steps or cfg.optimize_steps)
    warmup_steps = (
        min(int(cfg.fiber_warmup_steps), max(total_steps // 3, 0))
        if representation == "unified"
        else 0
    )
    routing_end = (
        min(max(2 * warmup_steps, warmup_steps + 1), total_steps)
        if representation == "unified"
        else 0
    )
    history: list[dict[str, float | int | str | dict[str, float]]] = []
    log_interval = max(int(log_every or cfg.fiber_log_every), 1)
    checkpoint_interval = int(
        checkpoint_every
        if checkpoint_every is not None
        else cfg.fiber_checkpoint_every
    )
    ema_decay = float(cfg.fiber_ema_decay)
    if not 0.0 <= ema_decay < 1.0:
        raise ValueError("fiber_ema_decay must be in [0, 1)")
    metrics_jsonl = out_dir / "training_metrics.jsonl"
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ema: dict[str, float] = {}
    started_at = time.perf_counter()
    risk_target: torch.Tensor | None = None
    latest_risk = torch.zeros(len(ROUTE_NAMES), device=device)
    latest_negative_contribution = torch.zeros(len(ROUTE_NAMES), device=device)
    latest_calibration_loss: float | None = None
    latest_calibration_frames: list[int] | None = None
    risk_update_count = 0
    dropout_counts = {name: 0 for name in ROUTE_NAMES}
    strand_visibility: torch.Tensor | None = None
    latest_visual_hull_report: dict[str, float] | None = None
    visual_hull_update_count = 0

    with open(metrics_jsonl, "w", encoding="utf-8", buffering=1) as metrics_file:
        for step in range(total_steps):
            optimizer.zero_grad(set_to_none=True)
            local_frame_index = step % n_frames
            frame_index = frame_indices[local_frame_index]
            camera = cameras[frame_index]
            with torch.no_grad():
                _tet_nodes, surface_vertices, _joints = motion.driven_points(
                    motion_indices[frame_index]
                )

            if representation == "residual_only":
                phase = "residual_only_3dgs"
                temperature = float(cfg.fiber_final_temperature)
                route_blend = 0.0
                geometry_blend = 0.0
                route_hardening = 1.0
                dropped_route = None
                primitives = field.residual_primitives(
                    surface_vertices, motion.surface_faces
                )
            else:
                phase, temperature, forced_route = _phase_for_step(
                    step, total_steps, warmup_steps, routing_end, cfg
                )
                route_blend, route_hardening = _routing_continuation(
                    step, total_steps, warmup_steps, routing_end, phase, cfg
                )
                geometry_blend = route_blend * route_blend
                route_dropout_probability = _scheduled_route_dropout_probability(
                    step,
                    total_steps,
                    routing_end,
                    phase,
                    float(cfg.fiber_route_dropout_probability),
                    float(cfg.fiber_route_dropout_final_fraction),
                )
                dropped_route = _sample_dropped_route(
                    route_rng,
                    phase,
                    route_dropout_probability,
                    float(cfg.fiber_route_dropout_residual_bias),
                )
                if dropped_route is not None:
                    dropout_counts[dropped_route] += 1
                visual_update_every = int(cfg.fiber_visual_hull_update_every)
                should_update_visual_hull = (
                    visual_update_every > 0
                    and phase != "gaussian_scaffold"
                    and (
                        strand_visibility is None
                        or (step - warmup_steps) % visual_update_every == 0
                    )
                )
                if should_update_visual_hull:
                    strand_visibility, latest_visual_hull_report = (
                        _compute_visual_hull_gate(
                            field,
                            visual_hull_vertices,
                            motion.surface_faces,
                            [cameras[index] for index in visual_hull_frame_indices],
                            visual_hull_ground_truth,
                            cfg,
                            temperature,
                            geometry_blend,
                        )
                    )
                    field.strand_visibility_gate = strand_visibility.detach()
                    visual_hull_update_count += 1
                primitives = field.primitives(
                    surface_vertices,
                    motion.surface_faces,
                    shell_samples=cfg.fiber_shell_samples,
                    strand_samples=cfg.fiber_strand_samples,
                    temperature=temperature,
                    forced_route=forced_route,
                    hard_route=False,
                    route_blend=route_blend,
                    geometry_blend=geometry_blend,
                    route_hardening=route_hardening,
                    dropped_route=dropped_route,
                    hard_route_policy=cfg.fiber_hard_route_policy,
                    strand_visibility=strand_visibility,
                    fin_aspect_ratio=cfg.fiber_fin_aspect_ratio,
                    additive_teacher=cfg.fiber_additive_teacher_mode,
                )
            orientation_target = (
                orientation_targets[local_frame_index]
                if phase != "gaussian_scaffold"
                else None
            )
            prediction = _render(
                primitives,
                camera,
                cfg,
                renderer_name,
                render_orientation=orientation_target is not None,
            )
            render_loss, render_parts = differentiable_render_loss(
                prediction,
                ground_truth[local_frame_index]["rgb"],
                ground_truth[local_frame_index]["mask"],
                cfg.color_loss_weight,
                cfg.mask_loss_weight,
                cfg.mask_boundary_weight,
                cfg.mask_boundary_radius,
                cfg.mask_balance_weight,
            )
            orientation_loss = render_loss.new_zeros(())
            if orientation_target is not None:
                orientation_loss = _orientation_consistency_loss(
                    prediction["orientation"], orientation_target,
                    ground_truth[local_frame_index]["mask"],
                )
                render_loss = render_loss + (
                    float(cfg.fiber_orientation_weight) * orientation_loss
                )
            visual_hull_loss = render_loss.new_zeros(())
            if (
                representation == "unified"
                and phase != "gaussian_scaffold"
                and float(cfg.fiber_visual_hull_weight) > 0.0
            ):
                strand_points = _strand_points_from_primitives(
                    primitives, field.point_count, int(cfg.fiber_strand_samples)
                )
                visual_hull_loss = _visual_hull_soft_loss(
                    strand_points,
                    primitives.route_probabilities,
                    camera,
                    ground_truth[local_frame_index]["mask"],
                    int(cfg.fiber_visual_hull_margin_px),
                )
            fin_silhouette_loss = render_loss.new_zeros(())
            if (
                representation == "unified"
                and phase != "gaussian_scaffold"
                and float(cfg.fiber_fin_silhouette_weight) > 0.0
            ):
                fin_silhouette_loss = _fin_point_support_loss(
                    field,
                    primitives,
                    camera,
                    ground_truth[local_frame_index]["mask"],
                    int(cfg.fiber_fin_silhouette_radius),
                )
            strand_support_loss = render_loss.new_zeros(())
            if (
                representation == "unified"
                and phase != "gaussian_scaffold"
                and float(cfg.fiber_strand_support_weight) > 0.0
            ):
                strand_support_loss = _strand_support_activation_loss(
                    field,
                    primitives,
                    camera,
                    ground_truth[local_frame_index]["mask"],
                    int(cfg.fiber_visual_hull_margin_px),
                )
            regularizers = field.regularizers(
                surface_vertices, motion.surface_faces, temperature=temperature
            )
            calibration_every = int(cfg.fiber_risk_calibration_every)
            should_update_risk = (
                representation == "unified"
                and phase != "gaussian_scaffold"
                and bool(calibration_frame_indices)
                and (
                    float(cfg.fiber_risk_calibration_weight) > 0.0
                    or float(cfg.fiber_negative_contribution_weight) > 0.0
                )
                and calibration_every > 0
                and geometry_blend
                >= float(cfg.fiber_risk_calibration_start_geometry_blend)
                and (
                    risk_target is None
                    or (step - warmup_steps) % calibration_every == 0
                )
            )
            if should_update_risk:
                latest_calibration_frames = list(calibration_frame_indices)
                calibration_vertices = []
                calibration_cameras = []
                with torch.no_grad():
                    for calibration_frame in calibration_frame_indices:
                        _tet_nodes, vertices, _joints = motion.driven_points(
                            motion_indices[calibration_frame]
                        )
                        calibration_vertices.append(vertices)
                        calibration_cameras.append(cameras[calibration_frame])
                (
                    new_target,
                    latest_risk,
                    latest_negative_contribution,
                    latest_calibration_loss,
                ) = (
                    _estimate_route_ablation_risk(
                        field,
                        calibration_vertices,
                        motion.surface_faces,
                        calibration_cameras,
                        calibration_ground_truth,
                        cfg,
                        renderer_name,
                        temperature,
                        geometry_blend,
                        strand_visibility,
                    )
                )
                new_target = _apply_route_mass_floor(
                    new_target, cfg.fiber_route_minimum_mass
                )
                prior_blend = float(cfg.fiber_risk_target_prior_blend)
                initial_mass = field.initial_route_probabilities.mean(dim=0)
                new_target = (
                    (1.0 - prior_blend) * new_target
                    + prior_blend * initial_mass
                )
                new_target = new_target / new_target.sum().clamp_min(1e-8)
                if risk_target is None:
                    risk_target = new_target
                else:
                    decay = float(cfg.fiber_risk_calibration_ema)
                    risk_target = decay * risk_target + (1.0 - decay) * new_target
                    risk_target = risk_target / risk_target.sum().clamp_min(1e-8)
                risk_update_count += 1

            risk_calibration = render_loss.new_zeros(())
            negative_contribution = render_loss.new_zeros(())
            if risk_target is not None and phase != "gaussian_scaffold":
                risk_calibration = _risk_calibration_kl(
                    field.route_probabilities(temperature), risk_target
                )
                negative_contribution = _negative_contribution_penalty(
                    field.route_probabilities(temperature),
                    latest_negative_contribution,
                )

            teacher_nonregression = render_loss.new_zeros(())
            teacher_calibration_student = None
            teacher_calibration_residual = None
            nonreg_every = int(cfg.fiber_teacher_nonregression_every)
            if (
                representation == "unified"
                and phase != "gaussian_scaffold"
                and teacher_field is not None
                and calibration_frame_indices
                and float(cfg.fiber_teacher_nonregression_weight) > 0.0
                and (step - warmup_steps) % nonreg_every == 0
            ):
                calibration_local = (
                    (step - warmup_steps) // nonreg_every
                ) % len(calibration_frame_indices)
                calibration_frame = calibration_frame_indices[calibration_local]
                with torch.no_grad():
                    _tet_nodes, calibration_vertices, _joints = motion.driven_points(
                        motion_indices[calibration_frame]
                    )
                calibration_student_primitives = field.primitives(
                    calibration_vertices,
                    motion.surface_faces,
                    shell_samples=cfg.fiber_shell_samples,
                    strand_samples=cfg.fiber_strand_samples,
                    temperature=temperature,
                    hard_route=False,
                    route_blend=route_blend,
                    geometry_blend=geometry_blend,
                    route_hardening=route_hardening,
                    hard_route_policy=cfg.fiber_hard_route_policy,
                    strand_visibility=strand_visibility,
                    fin_aspect_ratio=cfg.fiber_fin_aspect_ratio,
                    additive_teacher=cfg.fiber_additive_teacher_mode,
                )
                calibration_student_prediction = _render(
                    calibration_student_primitives,
                    cameras[calibration_frame],
                    cfg,
                    renderer_name,
                )
                calibration_student_loss, _parts = differentiable_render_loss(
                    calibration_student_prediction,
                    calibration_ground_truth[calibration_local]["rgb"],
                    calibration_ground_truth[calibration_local]["mask"],
                    cfg.color_loss_weight,
                    cfg.mask_loss_weight,
                    cfg.mask_boundary_weight,
                    cfg.mask_boundary_radius,
                    cfg.mask_balance_weight,
                )
                with torch.no_grad():
                    teacher_primitives = teacher_field.residual_primitives(
                        calibration_vertices, motion.surface_faces
                    )
                    teacher_prediction = _render(
                        teacher_primitives,
                        cameras[calibration_frame],
                        cfg,
                        renderer_name,
                    )
                    teacher_loss, _parts = differentiable_render_loss(
                        teacher_prediction,
                        calibration_ground_truth[calibration_local]["rgb"],
                        calibration_ground_truth[calibration_local]["mask"],
                        cfg.color_loss_weight,
                        cfg.mask_loss_weight,
                        cfg.mask_boundary_weight,
                        cfg.mask_boundary_radius,
                        cfg.mask_balance_weight,
                    )
                teacher_nonregression = F.relu(
                    calibration_student_loss
                    - teacher_loss
                    - float(cfg.fiber_teacher_nonregression_margin)
                )
                teacher_calibration_student = float(
                    calibration_student_loss.detach().cpu()
                )
                teacher_calibration_residual = float(teacher_loss.detach().cpu())
            if representation == "residual_only":
                regularization = (
                    cfg.fiber_residual_drift_weight
                    * field.residual_drift_regularizer()
                )
            elif phase == "gaussian_scaffold":
                regularization = render_loss.new_zeros(())
            else:
                prior_decay = max(
                    float(cfg.fiber_route_prior_final_fraction),
                    1.0 - (step - warmup_steps) / max(total_steps - warmup_steps, 1),
                )
                regularization = (
                    cfg.fiber_route_entropy_weight * regularizers["route_entropy"]
                    + cfg.fiber_route_prior_weight
                    * prior_decay
                    * regularizers["route_prior"]
                    + cfg.fiber_route_neighbor_weight
                    * regularizers["route_neighbor"]
                    + cfg.fiber_risk_calibration_weight * risk_calibration
                    + cfg.fiber_negative_contribution_weight
                    * negative_contribution
                    + cfg.fiber_teacher_nonregression_weight
                    * teacher_nonregression
                    + cfg.fiber_visual_hull_weight * visual_hull_loss
                    + cfg.fiber_fin_silhouette_weight * fin_silhouette_loss
                    + cfg.fiber_strand_support_weight * strand_support_loss
                    + cfg.fiber_shell_normal_weight * regularizers["shell_normal"]
                    + cfg.fiber_shell_length_weight * regularizers["shell_length"]
                    + cfg.fiber_strand_thinness_weight * regularizers["strand_thinness"]
                    + cfg.fiber_height_weight * regularizers["height"]
                    + cfg.fiber_bend_weight * regularizers["bend"]
                    + cfg.fiber_residual_drift_weight * regularizers["residual_drift"]
                    + cfg.fiber_residual_trust_weight * regularizers["residual_trust"]
                    + cfg.fiber_expert_appearance_weight
                    * regularizers["expert_appearance"]
                    + cfg.fiber_carrier_entropy_weight
                    * regularizers["carrier_entropy"]
                    + cfg.fiber_carrier_prior_weight
                    * regularizers["carrier_prior"]
                    + cfg.fiber_carrier_neighbor_weight
                    * regularizers["carrier_neighbor"]
                    + cfg.fiber_carrier_tip_neighbor_weight
                    * regularizers["carrier_tip_neighbor"]
                    + cfg.fiber_carrier_attachment_weight
                    * regularizers["carrier_attachment"]
                    + cfg.fiber_carrier_tip_prior_weight
                    * regularizers["carrier_tip_prior"]
                    + cfg.fiber_carrier_family_alignment_weight
                    * regularizers["carrier_family_alignment"]
                    + cfg.fiber_carrier_structure_floor_weight
                    * regularizers["carrier_structure_floor"]
                )
            objective = render_loss + regularization
            if not torch.isfinite(objective):
                emergency_path = checkpoint_dir / f"nonfinite_step_{step:06d}.pt"
                _save_training_checkpoint(
                    emergency_path,
                    field,
                    optimizer,
                    step,
                    phase,
                    frame_indices,
                    cfg,
                )
                raise FloatingPointError(
                    f"Non-finite objective at step {step}; checkpoint={emergency_path}"
                )
            objective.backward()
            gradient_norm = _gradient_norm(field)
            if not math.isfinite(gradient_norm):
                raise FloatingPointError(f"Non-finite gradient norm at step {step}")
            torch.nn.utils.clip_grad_norm_(field.parameters(), cfg.fiber_gradient_clip)
            optimizer.step()

            scalar_values = {
                "total": float(objective.detach().cpu()),
                "render": float(render_loss.detach().cpu()),
                "regularization": float(regularization.detach().cpu()),
                "color": float(render_parts["color"].detach().cpu()),
                "mask_loss": float(render_parts["mask_loss"].detach().cpu()),
                "mask_soft": float(render_parts["mask_soft"].detach().cpu()),
                "mask_01": float(render_parts["mask_01"].detach().cpu()),
                "mask_balanced": float(
                    render_parts["mask_balanced"].detach().cpu()
                ),
                "mask_foreground": float(
                    render_parts["mask_foreground"].detach().cpu()
                ),
                "mask_background": float(
                    render_parts["mask_background"].detach().cpu()
                ),
                "mask_boundary": float(render_parts["mask_boundary"].detach().cpu()),
                "orientation": float(orientation_loss.detach().cpu()),
                "risk_calibration": float(risk_calibration.detach().cpu()),
                "negative_contribution": float(
                    negative_contribution.detach().cpu()
                ),
                "teacher_nonregression": float(
                    teacher_nonregression.detach().cpu()
                ),
                "visual_hull": float(visual_hull_loss.detach().cpu()),
                "fin_silhouette": float(fin_silhouette_loss.detach().cpu()),
                "strand_support": float(strand_support_loss.detach().cpu()),
            }
            for name, value in scalar_values.items():
                ema[name] = value if name not in ema else (
                    ema_decay * ema[name] + (1.0 - ema_decay) * value
                )
            should_log = (
                step == 0
                or step == total_steps - 1
                or (step + 1) % log_interval == 0
            )
            if should_log:
                effective_route_mean = (
                    primitives.route_probabilities.detach().mean(dim=0).cpu()
                )
                record = {
                    "step": step,
                    "frame": frame_index,
                    "local_frame": local_frame_index,
                    "phase": phase,
                    "temperature": float(temperature),
                    "route_blend": route_blend,
                    "geometry_blend": geometry_blend,
                    "route_hardening": route_hardening,
                    "dropped_route": dropped_route,
                    "route_dropout_probability": (
                        route_dropout_probability
                        if representation == "unified"
                        else 0.0
                    ),
                    "elapsed_seconds": time.perf_counter() - started_at,
                    **scalar_values,
                    "gradient_norm": gradient_norm,
                    "ema": dict(ema),
                    "routes": _representation_route_summary(
                        field, temperature, representation
                    ),
                    "effective_routes": {
                        name: float(effective_route_mean[index])
                        for index, name in enumerate(ROUTE_NAMES)
                    },
                    "risk_target": (
                        {
                            name: float(risk_target[index].detach().cpu())
                            for index, name in enumerate(ROUTE_NAMES)
                        }
                        if risk_target is not None
                        else None
                    ),
                    "latest_ablation_risk": {
                        name: float(latest_risk[index].detach().cpu())
                        for index, name in enumerate(ROUTE_NAMES)
                    },
                    "latest_negative_contribution": {
                        name: float(
                            latest_negative_contribution[index].detach().cpu()
                        )
                        for index, name in enumerate(ROUTE_NAMES)
                    },
                    "visual_hull_report": latest_visual_hull_report,
                    "teacher_calibration_student": teacher_calibration_student,
                    "teacher_calibration_residual": teacher_calibration_residual,
                    "latest_calibration_frames": latest_calibration_frames,
                    "latest_calibration_loss": latest_calibration_loss,
                    **{
                        name: float(value.detach().cpu())
                        for name, value in regularizers.items()
                    },
                }
                history.append(record)
                line = json.dumps(record, sort_keys=True)
                metrics_file.write(line + "\n")
                print(f"fiber_metric={line}", flush=True)

            if (
                checkpoint_interval > 0
                and (step + 1) % checkpoint_interval == 0
                and step != total_steps - 1
            ):
                _save_training_checkpoint(
                    checkpoint_dir / f"step_{step + 1:06d}.pt",
                    field,
                    optimizer,
                    step,
                    phase,
                    frame_indices,
                    cfg,
                )

    checkpoint_pt = out_dir / "unified_fiber_field.pt"
    torch.save(
        {
            "state_dict": field.state_dict(),
            "metadata": {
                "routes": ROUTE_NAMES,
                "point_count": field.point_count,
                "requested_point_budget": point_budget,
                "capacity_mode": cfg.fiber_capacity_mode,
                "point_sampling_mode": cfg.fiber_point_sampling_mode,
                "exact_vertex_binding": bool(cfg.fiber_exact_vertex_binding),
                "scene_scale": float(field.scene_scale.detach().cpu()),
                "shell_samples": cfg.fiber_shell_samples,
                "strand_samples": cfg.fiber_strand_samples,
                "frame_indices": frame_indices,
                "calibration_frame_indices": calibration_frame_indices,
                "render_size": [width, height],
                "representation": representation,
                "hard_route_policy": cfg.fiber_hard_route_policy,
                "deployment_route_mode": (
                    "residual"
                    if representation == "residual_only"
                    else ("hard" if cfg.fiber_route_hardening else "soft")
                ),
                "baseline": "HairGS@16588656b1f6f048bc3bc83f3cb98c2da8596754",
                "residual_bootstrap": bootstrap_metadata,
                "fixed_residual_teacher": bool(cfg.fiber_freeze_residual_teacher),
                "visual_hull_update_count": visual_hull_update_count,
            },
        },
        checkpoint_pt,
    )
    state_npz = out_dir / "unified_fiber_field.npz"
    _save_field_npz(field, state_npz)
    _save_previews(
        field,
        motion,
        cameras,
        motion_indices,
        out_dir,
        frame_indices[:4],
        cfg,
        renderer_name,
        representation,
    )

    loss_curve_png = out_dir / "loss_curves.png"
    _plot_training_curves(history, loss_curve_png)

    report_json = out_dir / "unified_fiber_report.json"
    payload = {
        "baseline": {
            "name": "HairGS",
            "repository": "https://github.com/yimin-pan/hair-gs",
            "commit": "16588656b1f6f048bc3bc83f3cb98c2da8596754",
            "role": "Gaussian-to-strand implementation reference; upstream kept unmodified",
        },
        "implementation": (
            "surface-anchored residual-only skinned anisotropic 3DGS"
            if representation == "residual_only"
            else "optimization-first surface-anchored shell/strand/residual field"
        ),
        "representation": representation,
        "steps": total_steps,
        "warmup_steps": warmup_steps,
        "routing_end": routing_end,
        "frames": n_frames,
        "frame_indices": frame_indices,
        "motion_indices": [motion_indices[index] for index in frame_indices],
        "calibration_frames": len(calibration_frame_indices),
        "calibration_frame_indices": calibration_frame_indices,
        "points": field.point_count,
        "render_size": [width, height],
        "camera_source": camera_source,
        "renderer": renderer_name,
        "resource_usage": {
            "elapsed_training_seconds": time.perf_counter() - started_at,
            "peak_cuda_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.startswith("cuda")
                else 0
            ),
            "peak_cuda_reserved_bytes": (
                int(torch.cuda.max_memory_reserved(device))
                if device.startswith("cuda")
                else 0
            ),
        },
        "final_routes": _representation_route_summary(
            field, cfg.fiber_final_temperature, representation
        ),
        "final_hard_routes": _representation_route_summary(
            field,
            cfg.fiber_final_temperature,
            representation,
            hard=True,
            hard_policy=cfg.fiber_hard_route_policy,
        ),
        "final_risk_target": (
            {
                name: float(risk_target[index].detach().cpu())
                for index, name in enumerate(ROUTE_NAMES)
            }
            if risk_target is not None
            else None
        ),
        "deployment_route_mode": (
            "residual"
            if representation == "residual_only"
            else ("hard" if cfg.fiber_route_hardening else "soft")
        ),
        "route_dropout_counts": dropout_counts,
        "risk_update_count": risk_update_count,
        "visual_hull_update_count": visual_hull_update_count,
        "final_visual_hull_report": latest_visual_hull_report,
        "hard_route_policy": cfg.fiber_hard_route_policy,
        "config": {
            "representation": representation,
            "shell_samples": cfg.fiber_shell_samples,
            "strand_samples": cfg.fiber_strand_samples,
            "max_points": int(max_points or cfg.fiber_max_points),
            "base_lr": base_lr,
            "residual_bootstrap": bootstrap_metadata,
            "route_neighbor_k": int(cfg.fiber_route_neighbor_k),
            "route_neighbor_weight": float(cfg.fiber_route_neighbor_weight),
            "initial_residual_trust": float(cfg.fiber_initial_residual_trust),
            "residual_trust_weight": float(cfg.fiber_residual_trust_weight),
            "route_dropout_probability": float(
                cfg.fiber_route_dropout_probability
            ),
            "route_dropout_final_fraction": float(
                cfg.fiber_route_dropout_final_fraction
            ),
            "route_dropout_residual_bias": float(
                cfg.fiber_route_dropout_residual_bias
            ),
            "route_prior_final_fraction": float(
                cfg.fiber_route_prior_final_fraction
            ),
            "risk_calibration_every": int(cfg.fiber_risk_calibration_every),
            "risk_calibration_start_geometry_blend": float(
                cfg.fiber_risk_calibration_start_geometry_blend
            ),
            "risk_calibration_weight": float(cfg.fiber_risk_calibration_weight),
            "risk_target_prior_blend": float(
                cfg.fiber_risk_target_prior_blend
            ),
            "freeze_residual_teacher": bool(cfg.fiber_freeze_residual_teacher),
            "teacher_nonregression_weight": float(
                cfg.fiber_teacher_nonregression_weight
            ),
            "teacher_nonregression_margin": float(
                cfg.fiber_teacher_nonregression_margin
            ),
            "negative_contribution_weight": float(
                cfg.fiber_negative_contribution_weight
            ),
            "visual_hull_weight": float(cfg.fiber_visual_hull_weight),
            "visual_hull_update_every": int(cfg.fiber_visual_hull_update_every),
            "visual_hull_min_views": int(cfg.fiber_visual_hull_min_views),
            "visual_hull_min_fraction": float(
                cfg.fiber_visual_hull_min_fraction
            ),
        },
        "history": history,
        "metrics_jsonl": str(metrics_jsonl),
        "loss_curve_png": str(loss_curve_png),
    }
    with open(report_json, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    return FiberOptimizationArtifacts(
        out_dir, checkpoint_pt, state_npz, report_json, metrics_jsonl, loss_curve_png
    )


def _optimizer_parameter_groups(
    field: UnifiedFiberField,
    cfg: PipelineConfig,
    base_lr: float,
    representation: str,
) -> list[dict[str, object]]:
    appearance_parameters = [field.color_logits, field.opacity_logits]
    if representation == "unified":
        appearance_parameters.append(field.expert_color_delta)
    groups: list[dict[str, object]] = []

    def add_group(name: str, parameters: list[torch.nn.Parameter], scale: float) -> None:
        trainable = [parameter for parameter in parameters if parameter.requires_grad]
        if trainable:
            groups.append(
                {
                    "params": trainable,
                    "lr": base_lr * scale,
                    "name": name,
                }
            )

    add_group("appearance", appearance_parameters, cfg.fiber_appearance_lr_scale)
    residual_geometry = [
        field.residual_offset_local,
        field.residual_log_scale_delta,
        field.residual_rotation_raw,
    ]
    if representation == "residual_only":
        add_group("residual_geometry", residual_geometry, cfg.fiber_geometry_lr_scale)
        return groups
    add_group(
        "geometry",
        residual_geometry
        + [
            field.direction_local_raw,
            field.bend_local,
            field.bend_cubic_local,
            field.height_raw,
            field.shell_length_raw,
            field.strand_length_raw,
            field.radius_raw,
            field.carrier_root_tip_raw,
        ],
        cfg.fiber_geometry_lr_scale,
    )
    add_group(
        "structure_activation",
        [field.structured_delta_raw, field.structured_opacity_raw],
        cfg.fiber_structure_activation_lr_scale,
    )
    add_group(
        "routing",
        [field.route_logits, field.residual_trust_logits, field.carrier_logits],
        cfg.fiber_route_lr_scale,
    )
    return groups


def _representation_route_summary(
    field: UnifiedFiberField,
    temperature: float,
    representation: str,
    *,
    hard: bool = False,
    hard_policy: str = "argmax",
) -> dict[str, float]:
    if representation == "residual_only":
        return {"shell": 0.0, "strand": 0.0, "residual": 1.0}
    return (
        _hard_route_summary(field, temperature, hard_policy)
        if hard
        else field.route_summary(temperature)
    )


def _split_training_and_calibration_frames(
    frame_indices: list[int], calibration_frames: int
) -> tuple[list[int], list[int]]:
    calibration_count = max(int(calibration_frames), 0)
    if calibration_count == 0:
        return list(frame_indices), []
    if calibration_count >= len(frame_indices):
        raise ValueError(
            "fiber_calibration_frames must leave at least one photometric training frame"
        )
    split = len(frame_indices) - calibration_count
    return list(frame_indices[:split]), list(frame_indices[split:])


def _validate_route_training_config(cfg: PipelineConfig) -> None:
    if cfg.fiber_hard_route_policy not in HARD_ROUTE_POLICIES:
        raise ValueError(
            "fiber_hard_route_policy must be one of "
            f"{HARD_ROUTE_POLICIES}, got {cfg.fiber_hard_route_policy!r}"
        )
    dropout = float(cfg.fiber_route_dropout_probability)
    if not 0.0 <= dropout < 1.0:
        raise ValueError("fiber_route_dropout_probability must be in [0, 1)")
    dropout_final = float(cfg.fiber_route_dropout_final_fraction)
    if not 0.0 <= dropout_final <= 1.0:
        raise ValueError("fiber_route_dropout_final_fraction must be in [0, 1]")
    residual_bias = float(cfg.fiber_route_dropout_residual_bias)
    if not 0.0 <= residual_bias <= 1.0:
        raise ValueError("fiber_route_dropout_residual_bias must be in [0, 1]")
    minimum_mass = cfg.fiber_route_minimum_mass
    if minimum_mass is not None:
        values = np.asarray(minimum_mass, dtype=np.float64).reshape(-1)
        if values.shape != (len(ROUTE_NAMES),):
            raise ValueError(
                "fiber_route_minimum_mass must contain "
                f"[{', '.join(ROUTE_NAMES)}] mass"
            )
        if not np.isfinite(values).all() or np.any(values < 0.0):
            raise ValueError("fiber_route_minimum_mass must be finite and non-negative")
        if float(values.sum()) >= 1.0:
            raise ValueError("fiber_route_minimum_mass must sum to less than one")
    prior_floor = float(cfg.fiber_route_prior_final_fraction)
    if not 0.0 <= prior_floor <= 1.0:
        raise ValueError("fiber_route_prior_final_fraction must be in [0, 1]")
    risk_decay = float(cfg.fiber_risk_calibration_ema)
    if not 0.0 <= risk_decay < 1.0:
        raise ValueError("fiber_risk_calibration_ema must be in [0, 1)")
    target_prior = float(cfg.fiber_risk_target_prior_blend)
    if not 0.0 <= target_prior <= 1.0:
        raise ValueError("fiber_risk_target_prior_blend must be in [0, 1]")
    if int(cfg.fiber_route_neighbor_k) < 0:
        raise ValueError("fiber_route_neighbor_k must be non-negative")
    fin_strength = float(cfg.fiber_fin_gate_strength)
    if not 0.0 <= fin_strength <= 1.0:
        raise ValueError("fiber_fin_gate_strength must be in [0, 1]")
    fin_threshold = float(cfg.fiber_fin_grazing_threshold)
    if not 0.0 <= fin_threshold <= 1.0:
        raise ValueError("fiber_fin_grazing_threshold must be in [0, 1]")
    if float(cfg.fiber_fin_grazing_softness) <= 0.0:
        raise ValueError("fiber_fin_grazing_softness must be positive")
    if float(cfg.fiber_fin_aspect_ratio) < 1.0:
        raise ValueError("fiber_fin_aspect_ratio must be at least one")
    if float(cfg.fiber_fin_silhouette_weight) < 0.0:
        raise ValueError("fiber_fin_silhouette_weight must be non-negative")
    if int(cfg.fiber_fin_silhouette_radius) <= 0:
        raise ValueError("fiber_fin_silhouette_radius must be positive")
    if float(cfg.fiber_strand_support_weight) < 0.0:
        raise ValueError("fiber_strand_support_weight must be non-negative")
    if int(cfg.fiber_risk_calibration_every) < 0:
        raise ValueError("fiber_risk_calibration_every must be non-negative")
    calibration_start_blend = float(
        cfg.fiber_risk_calibration_start_geometry_blend
    )
    if not 0.0 <= calibration_start_blend <= 1.0:
        raise ValueError(
            "fiber_risk_calibration_start_geometry_blend must be in [0, 1]"
        )
    if float(cfg.fiber_risk_floor) <= 0.0:
        raise ValueError("fiber_risk_floor must be positive")
    if int(cfg.fiber_teacher_nonregression_every) <= 0:
        raise ValueError("fiber_teacher_nonregression_every must be positive")
    if float(cfg.fiber_teacher_nonregression_weight) < 0.0:
        raise ValueError("fiber_teacher_nonregression_weight must be non-negative")
    if float(cfg.fiber_negative_contribution_weight) < 0.0:
        raise ValueError("fiber_negative_contribution_weight must be non-negative")
    if int(cfg.fiber_visual_hull_update_every) < 0:
        raise ValueError("fiber_visual_hull_update_every must be non-negative")
    if int(cfg.fiber_visual_hull_min_views) <= 0:
        raise ValueError("fiber_visual_hull_min_views must be positive")
    visual_fraction = float(cfg.fiber_visual_hull_min_fraction)
    if not 0.0 <= visual_fraction <= 1.0:
        raise ValueError("fiber_visual_hull_min_fraction must be in [0, 1]")
    if int(cfg.fiber_visual_hull_margin_px) < 0:
        raise ValueError("fiber_visual_hull_margin_px must be non-negative")
    for name in (
        "fiber_carrier_entropy_weight",
        "fiber_carrier_prior_weight",
        "fiber_carrier_neighbor_weight",
        "fiber_carrier_tip_neighbor_weight",
        "fiber_carrier_attachment_weight",
        "fiber_carrier_tip_prior_weight",
        "fiber_carrier_family_alignment_weight",
        "fiber_carrier_structure_floor_weight",
    ):
        if float(getattr(cfg, name)) < 0.0:
            raise ValueError(f"{name} must be non-negative")


def _sample_dropped_route(
    rng: np.random.Generator,
    phase: str,
    probability: float,
    residual_bias: float = 1.0 / 3.0,
) -> str | None:
    if phase == "gaussian_scaffold" or probability <= 0.0:
        return None
    if float(rng.random()) >= probability:
        return None
    residual_bias = min(max(float(residual_bias), 0.0), 1.0)
    remaining = (1.0 - residual_bias) / 2.0
    index = int(
        rng.choice(
            len(ROUTE_NAMES),
            p=np.asarray([remaining, remaining, residual_bias], dtype=np.float64),
        )
    )
    return ROUTE_NAMES[index]


def _scheduled_route_dropout_probability(
    step: int,
    total_steps: int,
    routing_end: int,
    phase: str,
    probability: float,
    final_fraction: float,
) -> float:
    """Anneal route removal only after structured experts become visible."""

    if phase == "gaussian_scaffold" or probability <= 0.0:
        return 0.0
    if phase != "structured_refinement":
        return float(probability)
    progress = (step - routing_end) / max(total_steps - routing_end - 1, 1)
    progress = min(max(float(progress), 0.0), 1.0)
    multiplier = (1.0 - progress) + progress * float(final_fraction)
    return float(probability) * multiplier


def _normalize_positive_risk(risk: torch.Tensor, floor: float) -> torch.Tensor:
    """Allocate reward only to experts whose removal hurts calibration.

    ``floor`` stabilizes the denominator but is deliberately not added to
    every route: doing so would reward a route with zero or negative measured
    contribution.  If no route is useful, the residual teacher is the only
    safe fallback target.
    """

    positive = risk.clamp_min(0.0)
    total = positive.sum()
    if float(total.detach().cpu()) <= float(floor):
        fallback = torch.zeros_like(positive)
        fallback[ROUTE_NAMES.index("residual")] = 1.0
        return fallback
    return positive / total.clamp_min(float(floor))


def _apply_route_mass_floor(
    target: torch.Tensor,
    minimum_mass: list[float] | None,
) -> torch.Tensor:
    """Reserve only a small expert floor; let observed contribution allocate the rest."""

    target = target / target.sum().clamp_min(1e-8)
    if minimum_mass is None:
        return target
    floor = torch.as_tensor(minimum_mass, dtype=target.dtype, device=target.device)
    if floor.numel() != target.numel():
        raise ValueError("fiber_route_minimum_mass has an invalid size")
    return floor + (1.0 - floor.sum()) * target


def _risk_calibration_kl(
    probabilities: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    mass = probabilities.mean(dim=0).clamp_min(1e-8)
    target = target.detach().to(device=mass.device, dtype=mass.dtype)
    target = target / target.sum().clamp_min(1e-8)
    return F.kl_div(mass.log(), target, reduction="sum")


def _negative_contribution_penalty(
    probabilities: torch.Tensor, negative_contribution: torch.Tensor
) -> torch.Tensor:
    """Penalize mass on routes whose removal improves held-out loss."""

    mass = probabilities.mean(dim=0)
    negative = negative_contribution.detach().to(
        device=mass.device, dtype=mass.dtype
    )
    return torch.sum(mass * negative.clamp_min(0.0))


def _sample_mask_at_world_points(
    points: torch.Tensor,
    camera,
    mask: torch.Tensor,
    margin_px: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiably sample a (possibly dilated) mask at projected 3D points."""

    original_shape = points.shape[:-1]
    flat = points.reshape(-1, 3)
    dtype, device = flat.dtype, flat.device
    world_to_camera = torch.as_tensor(
        camera.world_to_camera, dtype=dtype, device=device
    )
    homogeneous = torch.cat(
        [flat, torch.ones((flat.shape[0], 1), dtype=dtype, device=device)], dim=-1
    )
    camera_xyz = homogeneous @ world_to_camera.T
    safe_z = camera_xyz[:, 2].clamp_min(1e-6)
    x = float(camera.fx) * camera_xyz[:, 0] / safe_z + float(camera.cx)
    y_sign = 1.0 if camera.image_y_down else -1.0
    y = float(camera.cy) + y_sign * float(camera.fy) * camera_xyz[:, 1] / safe_z
    width = int(camera.width)
    height = int(camera.height)
    valid = (
        (camera_xyz[:, 2] > 1e-5)
        & (x >= 0.0)
        & (x <= max(width - 1, 0))
        & (y >= 0.0)
        & (y <= max(height - 1, 0))
    )
    grid_x = 2.0 * x / max(width - 1, 1) - 1.0
    grid_y = 2.0 * y / max(height - 1, 1) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 1, 2)
    mask_image = mask.to(device=device, dtype=dtype).reshape(1, 1, height, width)
    margin = max(int(margin_px), 0)
    if margin > 0:
        kernel = 2 * margin + 1
        mask_image = F.max_pool2d(mask_image, kernel, stride=1, padding=margin)
    sampled = F.grid_sample(
        mask_image,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).reshape(-1)
    sampled = sampled * valid.to(dtype)
    return sampled.reshape(original_shape), valid.reshape(original_shape)


def _silhouette_band_loss(
    prediction: torch.Tensor,
    target_mask: torch.Tensor,
    radius: int,
) -> torch.Tensor:
    """Supervise Fin coverage only in a narrow target silhouette band.

    The target is a morphological gradient around the foreground boundary.
    Positive and negative pixels are averaged separately so sparse fine fur is
    not overwhelmed by background.  A small outside-band term discourages
    floating fins while leaving the residual teacher responsible for RGB.
    """

    if prediction.shape != target_mask.shape or prediction.ndim != 2:
        raise ValueError("prediction and target_mask must be aligned HxW tensors")
    radius = int(radius)
    if radius <= 0:
        raise ValueError("silhouette band radius must be positive")
    mask = target_mask.to(dtype=prediction.dtype).clamp(0.0, 1.0)
    image = mask[None, None]
    kernel = 2 * radius + 1
    dilated = F.max_pool2d(image, kernel, stride=1, padding=radius)[0, 0]
    eroded = -F.max_pool2d(-image, kernel, stride=1, padding=radius)[0, 0]
    band = (dilated - eroded).clamp(0.0, 1.0)
    positive = band > 0.5
    negative = ~positive
    positive_loss = (
        (1.0 - prediction[positive]).abs().mean()
        if bool(positive.any())
        else prediction.new_zeros(())
    )
    negative_loss = (
        prediction[negative].abs().mean()
        if bool(negative.any())
        else prediction.new_zeros(())
    )
    return positive_loss + 0.25 * negative_loss


def _silhouette_band(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if mask.ndim != 2:
        raise ValueError("silhouette mask must be HxW")
    radius = int(radius)
    if radius <= 0:
        raise ValueError("silhouette band radius must be positive")
    image = mask[None, None]
    kernel = 2 * radius + 1
    dilated = F.max_pool2d(image, kernel, stride=1, padding=radius)[0, 0]
    eroded = -F.max_pool2d(-image, kernel, stride=1, padding=radius)[0, 0]
    return (dilated - eroded).clamp(0.0, 1.0)


def _fin_point_support_loss(
    field: UnifiedFiberField,
    primitives,
    camera,
    target_mask: torch.Tensor,
    radius: int,
) -> torch.Tensor:
    """Activate zero-opacity fins from differentiable projected support.

    HairGS culls exactly transparent Gaussians before backward.  This surrogate
    bypasses that discrete renderer branch while preserving a mathematically
    zero initial render.  Once activated, ordinary RGB/mask rendering and LOO
    calibration remain the governing losses.
    """

    shell_points = primitives.xyz[
        primitives.route_id == ROUTE_NAMES.index("shell")
    ]
    if shell_points.shape[0] % field.point_count != 0:
        raise RuntimeError("Shell primitive count is not divisible by source count")
    samples = shell_points.shape[0] // field.point_count
    shell_points = shell_points.reshape(field.point_count, samples, 3)
    band = _silhouette_band(
        target_mask.to(dtype=shell_points.dtype).clamp(0.0, 1.0), radius
    )
    support, valid = _sample_mask_at_world_points(
        shell_points, camera, band, margin_px=0
    )
    source_support = (support * valid.to(support.dtype)).amax(dim=1)
    gate = field.structured_opacity_gain[:, ROUTE_NAMES.index("shell")]
    positive = source_support > 0.25
    negative = ~positive
    positive_loss = (
        (1.0 - gate[positive]).mean()
        if bool(positive.any())
        else gate.new_zeros(())
    )
    negative_loss = (
        gate[negative].mean()
        if bool(negative.any())
        else gate.new_zeros(())
    )
    return positive_loss + 0.05 * negative_loss


def _strand_support_activation_loss(
    field: UnifiedFiberField,
    primitives,
    camera,
    target_mask: torch.Tensor,
    margin_px: int,
) -> torch.Tensor:
    """Boot zero-opacity strands from projected foreground support.

    This only decides whether a strand is allowed to become visible.  The
    HairGS orientation render, RGB/mask objective, visual hull, and signed LOO
    evidence still decide its direction, geometry, and retained route mass.
    """

    strand_points = _strand_points_from_primitives(
        primitives,
        field.point_count,
        int((primitives.route_id == ROUTE_NAMES.index("strand")).sum().item())
        // field.point_count,
    )
    support, valid = _sample_mask_at_world_points(
        strand_points,
        camera,
        target_mask,
        margin_px=max(int(margin_px), 0),
    )
    valid_float = valid.to(support.dtype)
    source_support = (support * valid_float).sum(dim=1) / valid_float.sum(
        dim=1
    ).clamp_min(1.0)
    gate = field.structured_opacity_gain[:, ROUTE_NAMES.index("strand")]
    positive = source_support > 0.5
    negative = ~positive
    positive_loss = (
        (1.0 - gate[positive]).mean()
        if bool(positive.any())
        else gate.new_zeros(())
    )
    negative_loss = (
        gate[negative].mean()
        if bool(negative.any())
        else gate.new_zeros(())
    )
    return positive_loss + 0.05 * negative_loss


def _strand_points_from_primitives(
    primitives, point_count: int, strand_samples: int
) -> torch.Tensor:
    strand = primitives.xyz[primitives.route_id == ROUTE_NAMES.index("strand")]
    expected = int(point_count) * int(strand_samples)
    if strand.shape[0] != expected:
        raise RuntimeError(
            f"Expected {expected} strand samples, found {strand.shape[0]}"
        )
    return strand.reshape(point_count, strand_samples, 3)


def _compute_visual_hull_gate(
    field: UnifiedFiberField,
    surface_vertices: list[torch.Tensor],
    surface_faces: torch.Tensor,
    cameras: list,
    ground_truth: list[dict[str, torch.Tensor]],
    cfg: PipelineConfig,
    temperature: float,
    geometry_blend: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Cull strand samples unsupported by calibrated multi-view hair masks.

    Prefix connectivity is enforced root-to-tip: after the first rejected
    sample, more distal samples are rejected as well.  This avoids isolated
    floating tips even when a distal projection happens to re-enter a mask.
    """

    if not surface_vertices or not (
        len(surface_vertices) == len(cameras) == len(ground_truth)
    ):
        raise ValueError("Visual-hull views, cameras, and masks must be aligned")
    strand_samples = int(cfg.fiber_strand_samples)
    supported = field.route_logits.new_zeros((field.point_count, strand_samples))
    valid_count = torch.zeros_like(supported)
    with torch.no_grad():
        for vertices, camera, target in zip(surface_vertices, cameras, ground_truth):
            primitives = field.primitives(
                vertices,
                surface_faces,
                shell_samples=cfg.fiber_shell_samples,
                strand_samples=strand_samples,
                temperature=temperature,
                hard_route=False,
                geometry_blend=geometry_blend,
                route_hardening=0.0,
                fin_aspect_ratio=cfg.fiber_fin_aspect_ratio,
                additive_teacher=cfg.fiber_additive_teacher_mode,
            )
            points = _strand_points_from_primitives(
                primitives, field.point_count, strand_samples
            )
            mask_support, valid = _sample_mask_at_world_points(
                points,
                camera,
                target["mask"],
                int(cfg.fiber_visual_hull_margin_px),
            )
            valid_float = valid.to(supported.dtype)
            supported += (mask_support >= 0.5).to(supported.dtype) * valid_float
            valid_count += valid_float
        fraction = supported / valid_count.clamp_min(1.0)
        gate = (
            (supported >= max(int(cfg.fiber_visual_hull_min_views), 1))
            & (fraction >= float(cfg.fiber_visual_hull_min_fraction))
        ).to(supported.dtype)
        gate = torch.cumprod(gate, dim=1)
    report = {
        "kept_fraction": float(gate.mean().cpu()),
        "fully_kept_strands": float((gate[:, -1] > 0.5).float().mean().cpu()),
        "mean_support_fraction": float(fraction.mean().cpu()),
        "views": float(len(cameras)),
    }
    return gate, report


def _visual_hull_soft_loss(
    strand_points: torch.Tensor,
    route_probabilities: torch.Tensor,
    camera,
    mask: torch.Tensor,
    margin_px: int,
) -> torch.Tensor:
    support, _valid = _sample_mask_at_world_points(
        strand_points, camera, mask, margin_px
    )
    route_mass = route_probabilities[:, ROUTE_NAMES.index("strand")][:, None]
    weights = route_mass.expand_as(support)
    return ((1.0 - support) * weights).sum() / weights.sum().clamp_min(1e-8)


def _estimate_route_ablation_risk(
    field: UnifiedFiberField,
    surface_vertices: list[torch.Tensor],
    surface_faces: torch.Tensor,
    cameras: list,
    ground_truth: list[dict[str, torch.Tensor]],
    cfg: PipelineConfig,
    renderer_name: str,
    temperature: float,
    geometry_blend: float,
    strand_visibility: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Estimate global expert importance by aggregating held-out soft LOO renders."""

    if (
        len(surface_vertices) != len(ground_truth)
        or len(cameras) != len(ground_truth)
        or not surface_vertices
    ):
        raise ValueError(
            "Calibration vertices, cameras, and ground truth must be non-empty and aligned"
        )
    with torch.no_grad():
        accumulated_risk = field.route_logits.new_zeros(len(ROUTE_NAMES))
        accumulated_negative = field.route_logits.new_zeros(len(ROUTE_NAMES))
        accumulated_full_loss = field.route_logits.new_zeros(())
        for vertices, camera, target_frame in zip(
            surface_vertices, cameras, ground_truth
        ):
            primitives = field.primitives(
                vertices,
                surface_faces,
                shell_samples=cfg.fiber_shell_samples,
                strand_samples=cfg.fiber_strand_samples,
                temperature=temperature,
                hard_route=False,
                geometry_blend=geometry_blend,
                route_hardening=0.0,
                strand_visibility=strand_visibility,
                fin_aspect_ratio=cfg.fiber_fin_aspect_ratio,
                additive_teacher=cfg.fiber_additive_teacher_mode,
            )
            full_prediction = _render(primitives, camera, cfg, renderer_name)
            full_loss, _parts = differentiable_render_loss(
                full_prediction,
                target_frame["rgb"],
                target_frame["mask"],
                cfg.color_loss_weight,
                cfg.mask_loss_weight,
                cfg.mask_boundary_weight,
                cfg.mask_boundary_radius,
                cfg.mask_balance_weight,
            )
            accumulated_full_loss += full_loss
            for route_index in range(len(ROUTE_NAMES)):
                keep = (primitives.route_id != route_index).to(
                    primitives.opacity.dtype
                )
                ablated_prediction = _render(
                    replace(primitives, opacity=primitives.opacity * keep),
                    camera,
                    cfg,
                    renderer_name,
                )
                ablated_loss, _parts = differentiable_render_loss(
                    ablated_prediction,
                    target_frame["rgb"],
                    target_frame["mask"],
                    cfg.color_loss_weight,
                    cfg.mask_loss_weight,
                    cfg.mask_boundary_weight,
                    cfg.mask_boundary_radius,
                    cfg.mask_balance_weight,
                )
                signed_contribution = ablated_loss - full_loss
                accumulated_risk[route_index] += signed_contribution.clamp_min(0.0)
                accumulated_negative[route_index] += (-signed_contribution).clamp_min(0.0)
        count = float(len(surface_vertices))
        risk = accumulated_risk / count
        negative = accumulated_negative / count
        full_loss = accumulated_full_loss / count
        target = _normalize_positive_risk(risk, float(cfg.fiber_risk_floor))
    return target, risk, negative, float(full_loss.detach().cpu())


def _phase_for_step(
    step: int,
    total_steps: int,
    warmup_steps: int,
    routing_end: int,
    cfg: PipelineConfig,
) -> tuple[str, float, str | None]:
    if step < warmup_steps:
        return "gaussian_scaffold", cfg.fiber_initial_temperature, "residual"
    if step < routing_end:
        progress = (step - warmup_steps) / max(routing_end - warmup_steps, 1)
        temperature = (
            cfg.fiber_initial_temperature * (1.0 - progress)
            + 1.0 * progress
        )
        return "soft_routing", temperature, None
    progress = (step - routing_end) / max(total_steps - routing_end - 1, 1)
    temperature = 1.0 * (1.0 - progress) + cfg.fiber_final_temperature * progress
    return "structured_refinement", temperature, None


def _routing_continuation(
    step: int,
    total_steps: int,
    warmup_steps: int,
    routing_end: int,
    phase: str,
    cfg: PipelineConfig,
) -> tuple[float, float]:
    if phase == "gaussian_scaffold":
        return 0.0, 0.0
    if phase == "soft_routing":
        blend = (step - warmup_steps) / max(routing_end - warmup_steps, 1)
        return (float(blend) if cfg.fiber_route_continuation else 1.0), 0.0
    hardness = (step - routing_end) / max(total_steps - routing_end - 1, 1)
    return 1.0, (float(hardness) if cfg.fiber_route_hardening else 0.0)


def _gradient_norm(module: torch.nn.Module) -> float:
    squared = None
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        value = parameter.grad.detach().square().sum()
        squared = value if squared is None else squared + value
    return float(torch.sqrt(squared).cpu()) if squared is not None else 0.0


def _save_training_checkpoint(
    path: Path,
    field: UnifiedFiberField,
    optimizer: torch.optim.Optimizer,
    step: int,
    phase: str,
    frame_indices: list[int],
    cfg: PipelineConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": field.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": int(step),
            "phase": phase,
            "metadata": {
                "routes": ROUTE_NAMES,
                "point_count": field.point_count,
                "point_sampling_mode": cfg.fiber_point_sampling_mode,
                "exact_vertex_binding": bool(cfg.fiber_exact_vertex_binding),
                "frame_indices": list(frame_indices),
                "shell_samples": cfg.fiber_shell_samples,
                "strand_samples": cfg.fiber_strand_samples,
                "representation": cfg.fiber_representation,
                "hard_route_policy": cfg.fiber_hard_route_policy,
            },
        },
        path,
    )


def _hard_route_summary(
    field: UnifiedFiberField,
    temperature: float,
    hard_policy: str = "argmax",
) -> dict[str, float]:
    probabilities = field.route_probabilities(
        temperature, hard=True, hard_policy=hard_policy
    ).detach()
    counts = torch.bincount(
        probabilities.argmax(dim=-1), minlength=len(ROUTE_NAMES)
    ).float()
    counts /= counts.sum().clamp_min(1.0)
    return {
        name: float(counts[index].cpu())
        for index, name in enumerate(ROUTE_NAMES)
    }


def _plot_training_curves(
    history: list[dict[str, float | int | str | dict[str, float]]],
    path: Path,
) -> None:
    if not history:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    steps = np.asarray([int(item["step"]) for item in history])
    figure, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=True)
    axes[0].plot(steps, [float(item["render"]) for item in history], label="render")
    axes[0].plot(
        steps,
        [float(item["ema"]["render"]) for item in history],  # type: ignore[index]
        label="render EMA",
        linewidth=2,
    )
    axes[0].plot(
        steps,
        [float(item["regularization"]) for item in history],
        label="regularization",
        alpha=0.75,
    )
    axes[0].set_ylabel("objective")
    axes[0].set_yscale("symlog", linthresh=1e-4)
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.25)

    axes[1].plot(steps, [float(item["color"]) for item in history], label="foreground L1")
    axes[1].plot(steps, [float(item["mask_soft"]) for item in history], label="mask MAE")
    axes[1].plot(steps, [float(item["mask_01"]) for item in history], label="mask 0/1")
    axes[1].set_ylabel("data losses")
    axes[1].legend(loc="best")
    axes[1].grid(alpha=0.25)

    for route in ROUTE_NAMES:
        axes[2].plot(
            steps,
            [
                float(item.get("effective_routes", item["routes"])[route])  # type: ignore[union-attr,index]
                for item in history
            ],
            label=route,
        )
    axes[2].set_ylabel("mean route probability")
    axes[2].set_xlabel("optimization step")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].legend(loc="best")
    axes[2].grid(alpha=0.25)

    axes[3].plot(
        steps,
        [float(item.get("route_neighbor", 0.0)) for item in history],
        label="surface-neighbor",
    )
    axes[3].plot(
        steps,
        [float(item.get("risk_calibration", 0.0)) for item in history],
        label="risk calibration KL",
    )
    axes[3].set_ylabel("route regularizers")
    axes[3].set_xlabel("optimization step")
    axes[3].set_yscale("symlog", linthresh=1e-5)
    axes[3].legend(loc="best")
    axes[3].grid(alpha=0.25)

    previous_phase = None
    for item in history:
        phase = str(item["phase"])
        if previous_phase is not None and phase != previous_phase:
            for axis in axes:
                axis.axvline(int(item["step"]), color="black", linestyle="--", alpha=0.4)
        previous_phase = phase
    figure.suptitle("Unified fur/hair optimization diagnostics")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _save_field_npz(field: UnifiedFiberField, path: Path) -> None:
    cpu = lambda tensor: tensor.detach().cpu().numpy()
    np.savez_compressed(
        path,
        face_index=cpu(field.face_index),
        barycentric=cpu(field.barycentric),
        color=cpu(field.color),
        opacity=cpu(field.opacity),
        route_probabilities=cpu(
            field.route_probabilities(temperature=1.0)
        ),
        direction_local=cpu(field.direction_local),
        bend_local=cpu(field.bend_local),
        bend_cubic_local=cpu(field.bend_cubic_local),
        structured_delta_gain=cpu(field.structured_delta_gain),
        structured_opacity_gain=cpu(field.structured_opacity_gain),
        strand_visibility_gate=cpu(field.strand_visibility_gate),
        height=cpu(field.height),
        shell_length=cpu(field.shell_length),
        strand_length=cpu(field.strand_length),
        radius=cpu(field.radius),
        residual_offset_local=cpu(field.residual_offset_local),
        original_scaling=cpu(field.original_scaling),
        original_rotation=cpu(field.original_rotation),
        residual_scaling=cpu(field.residual_scaling),
        residual_rotation=cpu(field.residual_rotation),
        residual_trust=cpu(field.residual_trust),
        carrier_probabilities=cpu(field.carrier_probabilities(temperature=1.0)),
        carrier_root_tip=cpu(field.carrier_root_tip),
        scene_scale=np.asarray(float(field.scene_scale.detach().cpu()), dtype=np.float32),
        route_names=np.asarray(ROUTE_NAMES),
        carrier_names=np.asarray(CARRIER_NAMES),
    )


def _save_previews(
    field: UnifiedFiberField,
    motion: DifferentiableSkeletonTetModel,
    cameras,
    motion_indices,
    out_dir: Path,
    frame_indices: list[int],
    cfg: PipelineConfig,
    renderer_name: str,
    representation: str,
) -> None:
    preview_dir = out_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for frame_index in frame_indices:
            camera = cameras[frame_index]
            _tet_nodes, surface_vertices, _joints = motion.driven_points(
                motion_indices[frame_index]
            )
            preview_routes = (
                ("residual",) if representation == "residual_only" else (None, *ROUTE_NAMES)
            )
            for forced_route in preview_routes:
                if representation == "residual_only":
                    primitives = field.residual_primitives(
                        surface_vertices, motion.surface_faces
                    )
                else:
                    primitives = field.primitives(
                        surface_vertices,
                        motion.surface_faces,
                        shell_samples=cfg.fiber_shell_samples,
                        strand_samples=cfg.fiber_strand_samples,
                        temperature=cfg.fiber_final_temperature,
                        forced_route=forced_route,
                        hard_route=(
                            forced_route is None and cfg.fiber_route_hardening
                        ),
                        hard_route_policy=cfg.fiber_hard_route_policy,
                        fin_aspect_ratio=cfg.fiber_fin_aspect_ratio,
                        additive_teacher=cfg.fiber_additive_teacher_mode,
                    )
                prediction = _render(primitives, camera, cfg, renderer_name)
                image = (
                    prediction["rgb"].detach().cpu().numpy().clip(0.0, 1.0) * 255.0
                ).astype(np.uint8)
                name = "unified" if forced_route is None else forced_route
                Image.fromarray(image).save(
                    preview_dir / f"{frame_index:05d}_{name}.png"
                )
                if forced_route is None:
                    palette = torch.tensor(
                        [[0.15, 0.55, 1.0], [1.0, 0.25, 0.1], [0.4, 1.0, 0.2]],
                        dtype=primitives.color.dtype,
                        device=primitives.color.device,
                    )
                    route_prediction = _render(
                        replace(primitives, color=palette[primitives.route_id]),
                        camera,
                        cfg,
                        renderer_name,
                    )
                    route_image = (
                        route_prediction["rgb"]
                        .detach()
                        .cpu()
                        .numpy()
                        .clip(0.0, 1.0)
                        * 255.0
                    ).astype(np.uint8)
                    Image.fromarray(route_image).save(
                        preview_dir / f"{frame_index:05d}_route_map.png"
                    )


def _render(
    primitives,
    camera,
    cfg: PipelineConfig,
    renderer_name: str,
    *,
    render_orientation: bool = False,
):
    primitives = apply_fin_view_gate(
        primitives,
        camera,
        strength=cfg.fiber_fin_gate_strength,
        threshold=cfg.fiber_fin_grazing_threshold,
        softness=cfg.fiber_fin_grazing_softness,
    )
    if renderer_name == "torch":
        if render_orientation:
            raise ValueError("Orientation supervision requires the HairGS renderer")
        return render_fiber_primitives(
            primitives,
            camera,
            radius_px=cfg.render_radius_px,
            sigma_scale=cfg.fiber_sigma_scale,
        )
    from .hairgs_renderer import render_fiber_primitives_hairgs

    return render_fiber_primitives_hairgs(
        primitives, camera, render_orientation=render_orientation
    )


def _load_orientation_targets(
    frame_paths: list[Path],
    frame_indices: list[int],
    width: int,
    height: int,
    device: str,
    orientation_dir: str | None,
) -> list[dict[str, torch.Tensor] | None]:
    """Load HairGS-compatible Gabor angle maps for the selected train views."""

    if orientation_dir is None:
        return [None] * len(frame_indices)
    directory = Path(orientation_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"fiber_orientation_dir does not exist: {directory}")
    targets: list[dict[str, torch.Tensor] | None] = []
    for index in frame_indices:
        stem = frame_paths[index].stem
        angle_path = directory / f"{stem}_orientation.png"
        confidence_path = directory / f"{stem}_confidence.png"
        variance_path = directory / f"{stem}_orientation_var.npy"
        if not angle_path.is_file() or not (
            confidence_path.is_file() or variance_path.is_file()
        ):
            raise FileNotFoundError(
                f"Missing HairGS orientation/confidence maps for {frame_paths[index].name}"
            )
        angle = np.asarray(Image.open(angle_path).convert("L"), dtype=np.float32)
        if confidence_path.is_file():
            confidence = np.asarray(
                Image.open(confidence_path).convert("L"), dtype=np.float32
            ) / 255.0
        else:
            # GaussianHaircut publishes angular variance rather than Hair-GS's
            # confidence PNG. Match its camera loader exactly. Absolute scale
            # cancels in the confidence-normalized orientation objective.
            variance = np.load(variance_path).astype(np.float32)
            confidence = 1.0 / ((variance / math.pi**2) ** 2 + 1e-7)
        theta = torch.as_tensor(angle * (math.pi / 255.0), dtype=torch.float32)
        vectors = torch.stack([torch.cos(2.0 * theta), torch.sin(2.0 * theta)], dim=0)
        confidence_tensor = torch.as_tensor(confidence, dtype=torch.float32)[None, None]
        vectors = F.interpolate(
            vectors[None], size=(height, width), mode="bilinear", align_corners=False
        )[0].permute(1, 2, 0)
        confidence_tensor = F.interpolate(
            confidence_tensor, size=(height, width), mode="bilinear", align_corners=False
        )[0, 0]
        targets.append(
            {
                "vectors": F.normalize(vectors, dim=-1, eps=1e-8).to(device),
                "confidence": confidence_tensor.to(device),
            }
        )
    return targets


def _orientation_consistency_loss(
    predicted: torch.Tensor,
    target: dict[str, torch.Tensor],
    foreground_mask: torch.Tensor,
) -> torch.Tensor:
    """Confidence-weighted sign-invariant image-space tangent discrepancy."""

    predicted_vectors = F.normalize(predicted[..., :2], dim=-1, eps=1e-8)
    agreement = (predicted_vectors * target["vectors"]).sum(dim=-1).clamp(-1.0, 1.0)
    weights = target["confidence"] * foreground_mask.to(target["confidence"].dtype)
    return (0.5 * (1.0 - agreement) * weights).sum() / weights.sum().clamp_min(1e-8)
