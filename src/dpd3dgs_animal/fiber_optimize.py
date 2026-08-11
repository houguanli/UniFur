from __future__ import annotations

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
    ROUTE_NAMES,
    UnifiedFiberField,
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
    representation = str(cfg.fiber_representation).lower()
    if representation not in {"unified", "residual_only"}:
        raise ValueError(
            "fiber_representation must be 'unified' or 'residual_only'"
        )
    motion = DifferentiableSkeletonTetModel(stage1_npz, device=device)
    motion.joints.requires_grad_(False)
    field = create_unified_fiber_field(
        gaussian_ply,
        motion.rest_surface_vertices.detach().cpu().numpy(),
        motion.surface_faces.detach().cpu().numpy(),
        device=device,
        max_points=int(max_points or cfg.fiber_max_points),
        neighbor_k=(
            int(cfg.fiber_route_neighbor_k) if representation == "unified" else 0
        ),
        initial_residual_trust=float(cfg.fiber_initial_residual_trust),
    )

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
    latest_calibration_loss: float | None = None
    latest_calibration_frames: list[int] | None = None
    risk_update_count = 0
    dropout_counts = {name: 0 for name in ROUTE_NAMES}

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
                dropped_route = _sample_dropped_route(
                    route_rng,
                    phase,
                    float(cfg.fiber_route_dropout_probability),
                    float(cfg.fiber_route_dropout_residual_bias),
                )
                if dropped_route is not None:
                    dropout_counts[dropped_route] += 1
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
                )
            prediction = _render(primitives, camera, cfg, renderer_name)
            render_loss, render_parts = differentiable_render_loss(
                prediction,
                ground_truth[local_frame_index]["rgb"],
                ground_truth[local_frame_index]["mask"],
                cfg.color_loss_weight,
                cfg.mask_loss_weight,
            )
            regularizers = field.regularizers(
                surface_vertices, motion.surface_faces, temperature=temperature
            )
            calibration_every = int(cfg.fiber_risk_calibration_every)
            should_update_risk = (
                representation == "unified"
                and phase != "gaussian_scaffold"
                and bool(calibration_frame_indices)
                and float(cfg.fiber_risk_calibration_weight) > 0.0
                and calibration_every > 0
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
                new_target, latest_risk, latest_calibration_loss = (
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
                    )
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
            if risk_target is not None and phase != "gaussian_scaffold":
                risk_calibration = _risk_calibration_kl(
                    field.route_probabilities(temperature), risk_target
                )
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
                    + cfg.fiber_shell_normal_weight * regularizers["shell_normal"]
                    + cfg.fiber_shell_length_weight * regularizers["shell_length"]
                    + cfg.fiber_strand_thinness_weight * regularizers["strand_thinness"]
                    + cfg.fiber_height_weight * regularizers["height"]
                    + cfg.fiber_bend_weight * regularizers["bend"]
                    + cfg.fiber_residual_drift_weight * regularizers["residual_drift"]
                    + cfg.fiber_residual_trust_weight * regularizers["residual_trust"]
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
                "risk_calibration": float(risk_calibration.detach().cpu()),
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
                "scene_scale": float(field.scene_scale.detach().cpu()),
                "shell_samples": cfg.fiber_shell_samples,
                "strand_samples": cfg.fiber_strand_samples,
                "frame_indices": frame_indices,
                "calibration_frame_indices": calibration_frame_indices,
                "render_size": [width, height],
                "representation": representation,
                "deployment_route_mode": (
                    "residual"
                    if representation == "residual_only"
                    else ("hard" if cfg.fiber_route_hardening else "soft")
                ),
                "baseline": "HairGS@16588656b1f6f048bc3bc83f3cb98c2da8596754",
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
            field, cfg.fiber_final_temperature, representation, hard=True
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
        "config": {
            "representation": representation,
            "shell_samples": cfg.fiber_shell_samples,
            "strand_samples": cfg.fiber_strand_samples,
            "max_points": int(max_points or cfg.fiber_max_points),
            "base_lr": base_lr,
            "route_neighbor_k": int(cfg.fiber_route_neighbor_k),
            "route_neighbor_weight": float(cfg.fiber_route_neighbor_weight),
            "initial_residual_trust": float(cfg.fiber_initial_residual_trust),
            "residual_trust_weight": float(cfg.fiber_residual_trust_weight),
            "route_dropout_probability": float(
                cfg.fiber_route_dropout_probability
            ),
            "route_dropout_residual_bias": float(
                cfg.fiber_route_dropout_residual_bias
            ),
            "route_prior_final_fraction": float(
                cfg.fiber_route_prior_final_fraction
            ),
            "risk_calibration_every": int(cfg.fiber_risk_calibration_every),
            "risk_calibration_weight": float(cfg.fiber_risk_calibration_weight),
            "risk_target_prior_blend": float(
                cfg.fiber_risk_target_prior_blend
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
    groups: list[dict[str, object]] = [
        {
            "params": [field.color_logits, field.opacity_logits],
            "lr": base_lr * cfg.fiber_appearance_lr_scale,
            "name": "appearance",
        }
    ]
    residual_geometry = [
        field.residual_offset_local,
        field.residual_log_scale_delta,
        field.residual_rotation_raw,
    ]
    if representation == "residual_only":
        groups.append(
            {
                "params": residual_geometry,
                "lr": base_lr * cfg.fiber_geometry_lr_scale,
                "name": "residual_geometry",
            }
        )
        return groups
    groups.extend(
        [
            {
                "params": residual_geometry
                + [
                    field.direction_local_raw,
                    field.bend_local,
                    field.height_raw,
                    field.shell_length_raw,
                    field.strand_length_raw,
                    field.radius_raw,
                ],
                "lr": base_lr * cfg.fiber_geometry_lr_scale,
                "name": "geometry",
            },
            {
                "params": [field.route_logits, field.residual_trust_logits],
                "lr": base_lr * cfg.fiber_route_lr_scale,
                "name": "routing",
            },
        ]
    )
    return groups


def _representation_route_summary(
    field: UnifiedFiberField,
    temperature: float,
    representation: str,
    *,
    hard: bool = False,
) -> dict[str, float]:
    if representation == "residual_only":
        return {"shell": 0.0, "strand": 0.0, "residual": 1.0}
    return (
        _hard_route_summary(field, temperature)
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
    dropout = float(cfg.fiber_route_dropout_probability)
    if not 0.0 <= dropout < 1.0:
        raise ValueError("fiber_route_dropout_probability must be in [0, 1)")
    residual_bias = float(cfg.fiber_route_dropout_residual_bias)
    if not 0.0 <= residual_bias <= 1.0:
        raise ValueError("fiber_route_dropout_residual_bias must be in [0, 1]")
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
    if int(cfg.fiber_risk_calibration_every) < 0:
        raise ValueError("fiber_risk_calibration_every must be non-negative")
    if float(cfg.fiber_risk_floor) <= 0.0:
        raise ValueError("fiber_risk_floor must be positive")


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


def _normalize_positive_risk(risk: torch.Tensor, floor: float) -> torch.Tensor:
    positive = risk.clamp_min(0.0) + float(floor)
    return positive / positive.sum().clamp_min(1e-8)


def _risk_calibration_kl(
    probabilities: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    mass = probabilities.mean(dim=0).clamp_min(1e-8)
    target = target.detach().to(device=mass.device, dtype=mass.dtype)
    target = target / target.sum().clamp_min(1e-8)
    return F.kl_div(mass.log(), target, reduction="sum")


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
) -> tuple[torch.Tensor, torch.Tensor, float]:
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
            )
            full_prediction = _render(primitives, camera, cfg, renderer_name)
            full_loss, _parts = differentiable_render_loss(
                full_prediction,
                target_frame["rgb"],
                target_frame["mask"],
                cfg.color_loss_weight,
                cfg.mask_loss_weight,
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
                )
                accumulated_risk[route_index] += (
                    ablated_loss - full_loss
                ).clamp_min(0.0)
        count = float(len(surface_vertices))
        risk = accumulated_risk / count
        full_loss = accumulated_full_loss / count
        target = _normalize_positive_risk(risk, float(cfg.fiber_risk_floor))
    return target, risk, float(full_loss.detach().cpu())


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
                "frame_indices": list(frame_indices),
                "shell_samples": cfg.fiber_shell_samples,
                "strand_samples": cfg.fiber_strand_samples,
                "representation": cfg.fiber_representation,
            },
        },
        path,
    )


def _hard_route_summary(
    field: UnifiedFiberField, temperature: float
) -> dict[str, float]:
    probabilities = field.route_probabilities(temperature).detach()
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
        scene_scale=np.asarray(float(field.scene_scale.detach().cpu()), dtype=np.float32),
        route_names=np.asarray(ROUTE_NAMES),
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


def _render(primitives, camera, cfg: PipelineConfig, renderer_name: str):
    if renderer_name == "torch":
        return render_fiber_primitives(
            primitives,
            camera,
            radius_px=cfg.render_radius_px,
            sigma_scale=cfg.fiber_sigma_scale,
        )
    from .hairgs_renderer import render_fiber_primitives_hairgs

    return render_fiber_primitives_hairgs(primitives, camera)
