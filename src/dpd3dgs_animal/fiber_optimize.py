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
    attach_fixed_gaussian_base,
    create_unified_fiber_field,
    partition_binding_cache,
    render_fiber_primitives,
    _surface_knn_indices,
)
from .scaffold import (
    DifferentiableSurfaceScaffold,
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
_RESIDUAL_APPEARANCE_KEYS = ("color_logits", "opacity_logits")
_RESIDUAL_GEOMETRY_KEYS = (
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


def _freeze_residual_teacher_scaffold(
    field: UnifiedFiberField, *, optimize_structured_base_appearance: bool = False
) -> None:
    """Freeze teacher geometry; optionally let the structured student refit appearance."""

    frozen_keys = list(_RESIDUAL_GEOMETRY_KEYS)
    if not bool(optimize_structured_base_appearance):
        frozen_keys.extend(_RESIDUAL_APPEARANCE_KEYS)
    for name in frozen_keys:
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


def _rank_unit_interval(value: torch.Tensor) -> torch.Tensor:
    """Return deterministic, scale-free ranks in [0, 1]."""

    flat = value.detach().reshape(-1)
    if flat.numel() <= 1:
        return torch.zeros_like(flat)
    # Source id is the final tie breaker.  The perturbation is far below the
    # precision of the physical scores but makes repeated migrations exact.
    source_id = torch.arange(flat.numel(), device=flat.device, dtype=flat.dtype)
    tie = source_id * (torch.finfo(flat.dtype).eps / float(flat.numel()))
    order = torch.argsort(flat + tie)
    ranks = torch.empty_like(flat)
    ranks[order] = torch.linspace(
        0.0, 1.0, flat.numel(), device=flat.device, dtype=flat.dtype
    )
    return ranks


def _route_capacity_counts(total: int, mass: torch.Tensor) -> list[int]:
    """Round a normalized route mass without changing the total capacity."""

    raw = mass.detach().cpu().double().numpy() * int(total)
    counts = np.floor(raw).astype(np.int64)
    remainder = int(total) - int(counts.sum())
    fractional = raw - counts
    # Stable route-index tie breaking keeps reports/checkpoints reproducible.
    order = np.argsort(-fractional, kind="stable")
    for index in order[:remainder]:
        counts[int(index)] += 1
    return [int(value) for value in counts]


def _initialize_render_preserving_semantic_migration(
    field: UnifiedFiberField,
    route_mass: list[float] | tuple[float, float, float],
    *,
    temperature: float,
) -> dict[str, object]:
    """Hard-allocate teacher sources while keeping shell/strand co-located.

    Low-confidence, broad, or weakly attached sources retain residual
    capacity.  The remainder is split by anisotropy into shell and strand.
    Every source has exactly one active renderer family, so the configured
    residual share is a real capacity ceiling rather than a soft preference.
    Zero structured displacement plus opacity splitting makes the allocation
    an (up to rasterizer precision) render-preserving reparameterization.
    """

    requested = torch.as_tensor(
        route_mass, device=field.route_logits.device, dtype=field.route_logits.dtype
    ).reshape(-1)
    if requested.numel() != len(ROUTE_NAMES):
        raise ValueError(
            "fiber_teacher_semantic_migration_mass must contain effective "
            f"[{', '.join(ROUTE_NAMES)}] source fractions"
        )
    if (
        not torch.isfinite(requested).all()
        or torch.any(requested < 0.0)
        or float(requested.sum()) <= 0.0
    ):
        raise ValueError(
            "fiber_teacher_semantic_migration_mass must be finite, "
            "non-negative, and sum to a positive value"
        )
    if float(temperature) <= 0.0:
        raise ValueError("semantic migration temperature must be positive")
    requested = requested / requested.sum()
    counts = _route_capacity_counts(field.point_count, requested)
    shell_index = ROUTE_NAMES.index("shell")
    strand_index = ROUTE_NAMES.index("strand")
    residual_index = ROUTE_NAMES.index("residual")

    with torch.no_grad():
        scaling = field.original_scaling.detach().clamp_min(1e-12)
        sorted_scaling = torch.sort(scaling, dim=-1).values
        anisotropy = torch.log(
            (sorted_scaling[:, -1] / sorted_scaling[:, 0]).clamp_min(1.0)
        )
        source_size = torch.log(
            torch.prod(scaling, dim=-1).clamp_min(1e-30).pow(1.0 / 3.0)
            / field.scene_scale.clamp_min(1e-8)
        )
        binding_distance = torch.linalg.vector_norm(
            field.initial_residual_offset_local.detach(), dim=-1
        ) / field.scene_scale.clamp_min(1e-8)
        opacity = field.opacity.detach()
        normal_alignment = field.direction_local.detach()[:, 2].abs()

        anisotropy_rank = _rank_unit_interval(anisotropy)
        opacity_rank = _rank_unit_interval(opacity)
        size_rank = _rank_unit_interval(source_size)
        distance_rank = _rank_unit_interval(binding_distance)
        # High structure confidence means a compact, anisotropic, opaque
        # source with a reliable scalp attachment.
        structure_confidence = (
            0.45 * anisotropy_rank
            + 0.25 * opacity_rank
            - 0.20 * distance_rank
            - 0.10 * size_rank
        )
        # Curves favor highly anisotropic, tangent-aligned sources; the
        # remaining reliable sources form the denser shell/Fin family.
        strand_preference = (
            0.70 * anisotropy_rank
            + 0.20 * distance_rank
            + 0.10 * (1.0 - normal_alignment)
        )

        assignment = torch.full(
            (field.point_count,), shell_index, device=field.route_logits.device,
            dtype=torch.long,
        )
        all_indices = torch.arange(field.point_count, device=assignment.device)
        residual_order = torch.argsort(structure_confidence)
        residual_indices = residual_order[: counts[residual_index]]
        assignment[residual_indices] = residual_index
        structured_eligible = assignment != residual_index
        structured_indices = all_indices[structured_eligible]
        strand_order = torch.argsort(
            strand_preference[structured_indices], descending=True
        )
        strand_indices = structured_indices[
            strand_order[: counts[strand_index]]
        ]
        assignment[strand_indices] = strand_index
        if int((assignment == shell_index).sum()) != counts[shell_index]:
            raise RuntimeError("semantic migration shell capacity allocation failed")

        one_hot = F.one_hot(assignment, num_classes=len(ROUTE_NAMES)).to(
            field.route_logits.dtype
        )
        field.route_active_gate.zero_()
        field.route_active_gate.copy_(one_hot)
        epsilon = 1e-6
        smooth = one_hot * (1.0 - epsilon * (len(ROUTE_NAMES) - 1)) + (
            1.0 - one_hot
        ) * epsilon
        field.route_logits.copy_(float(temperature) * torch.log(smooth))
        field.initial_route_probabilities.copy_(smooth)
        low_trust = torch.logit(
            torch.tensor(1e-4, device=assignment.device, dtype=field.route_logits.dtype)
        )
        high_trust = torch.logit(
            torch.tensor(
                1.0 - 1e-4,
                device=assignment.device,
                dtype=field.route_logits.dtype,
            )
        )
        field.residual_trust_logits.fill_(low_trust)
        field.residual_trust_logits[assignment == residual_index, 0] = high_trust

        carrier_assignment = assignment.clone()
        # Route and carrier enums are [shell, strand, residual] versus
        # [surface, shell, strand].
        carrier_assignment[assignment == shell_index] = CARRIER_NAMES.index("shell")
        carrier_assignment[assignment == strand_index] = CARRIER_NAMES.index("strand")
        carrier_assignment[assignment == residual_index] = CARRIER_NAMES.index("surface")
        carrier_one_hot = F.one_hot(
            carrier_assignment, num_classes=len(CARRIER_NAMES)
        ).to(field.carrier_logits.dtype)
        carrier_smooth = carrier_one_hot * (
            1.0 - epsilon * (len(CARRIER_NAMES) - 1)
        ) + (1.0 - carrier_one_hot) * epsilon
        field.carrier_logits.copy_(float(temperature) * torch.log(carrier_smooth))
        field.initial_carrier_probabilities.copy_(carrier_smooth)

        # This line is the render-preserving contract.  Training unfolds the
        # selected semantic families from an exact residual copy.
        field.structured_delta_raw.zero_()
        field.structured_opacity_raw.zero_()

    route_statistics: dict[str, object] = {}
    for route_index, route_name in enumerate(ROUTE_NAMES):
        selected = assignment == route_index
        route_statistics[route_name] = {
            "count": int(selected.sum().detach().cpu()),
            "fraction": float(selected.float().mean().detach().cpu()),
            "mean_anisotropy": float(anisotropy[selected].mean().detach().cpu())
            if bool(selected.any())
            else None,
            "mean_binding_distance_relative": float(
                binding_distance[selected].mean().detach().cpu()
            )
            if bool(selected.any())
            else None,
            "mean_opacity": float(opacity[selected].mean().detach().cpu())
            if bool(selected.any())
            else None,
        }
    return {
        "requested_mass": [float(value) for value in requested.detach().cpu()],
        "capacity_counts": {
            name: counts[index] for index, name in enumerate(ROUTE_NAMES)
        },
        "residual_source_capacity_ceiling": float(
            counts[residual_index] / max(field.point_count, 1)
        ),
        "selection": "ranked covariance/opacity/scalp-attachment semantic migration",
        "zero_initialized_structured_delta": True,
        "single_active_route_per_source": True,
        "route_statistics": route_statistics,
    }


def _initialize_render_preserving_adaptive_migration(
    field: UnifiedFiberField,
    domain: str,
    *,
    domain_bias: float,
    temperature: float,
) -> dict[str, object]:
    """Remove residual capacity and initialize an adaptive shell/strand router.

    Hair/fur supplies a weak family-level log-odds prior.  Per-source
    anisotropy, scalp distance, opacity, and local direction decide the rest;
    no aggregate route fraction is prescribed.  Both structural routes start
    as co-located teacher copies, and transmittance-conserving opacity splitting
    keeps their soft mixture render-equivalent.
    """

    normalized_domain = str(domain).lower()
    if normalized_domain not in {"hair", "fur", "auto"}:
        raise ValueError(
            "fiber_teacher_adaptive_migration_domain must be 'hair', 'fur', "
            "or 'auto'"
        )
    if not math.isfinite(float(domain_bias)) or float(domain_bias) < 0.0:
        raise ValueError("fiber_teacher_adaptive_migration_bias must be non-negative")
    if float(temperature) <= 0.0:
        raise ValueError("adaptive migration temperature must be positive")
    shell_index = ROUTE_NAMES.index("shell")
    strand_index = ROUTE_NAMES.index("strand")
    residual_index = ROUTE_NAMES.index("residual")

    with torch.no_grad():
        scaling = field.original_scaling.detach().clamp_min(1e-12)
        sorted_scaling = torch.sort(scaling, dim=-1).values
        anisotropy = torch.log(
            (sorted_scaling[:, -1] / sorted_scaling[:, 0]).clamp_min(1.0)
        )
        binding_distance = torch.linalg.vector_norm(
            field.initial_residual_offset_local.detach(), dim=-1
        ) / field.scene_scale.clamp_min(1e-8)
        opacity = field.opacity.detach()
        normal_alignment = field.direction_local.detach()[:, 2].abs()

        # Positive preference selects strand.  Every term is rank-normalized,
        # so the router transfers across reconstruction scale and GS density.
        preference = (
            1.50 * (_rank_unit_interval(anisotropy) - 0.5)
            + 0.45 * (_rank_unit_interval(binding_distance) - 0.5)
            + 0.30 * (_rank_unit_interval(opacity) - 0.5)
            + 0.25 * (1.0 - normal_alignment)
        )
        if normalized_domain == "hair":
            preference = preference + float(domain_bias)
        elif normalized_domain == "fur":
            preference = preference - float(domain_bias)

        field.route_active_gate.zero_()
        field.route_active_gate[:, shell_index] = 1.0
        field.route_active_gate[:, strand_index] = 1.0
        route_logits = torch.full_like(field.route_logits, -20.0)
        route_logits[:, shell_index] = -0.5 * float(temperature) * preference
        route_logits[:, strand_index] = 0.5 * float(temperature) * preference
        field.route_logits.copy_(route_logits)
        field.residual_trust_logits.fill_(
            torch.logit(
                torch.tensor(
                    1e-4,
                    device=field.route_logits.device,
                    dtype=field.route_logits.dtype,
                )
            )
        )
        initial_probabilities = field.route_probabilities(
            temperature=float(temperature), route_blend=1.0
        )
        field.initial_route_probabilities.copy_(
            initial_probabilities.clamp_min(1e-6)
        )

        carrier_logits = torch.full_like(field.carrier_logits, -20.0)
        carrier_logits[:, CARRIER_NAMES.index("shell")] = route_logits[
            :, shell_index
        ]
        carrier_logits[:, CARRIER_NAMES.index("strand")] = route_logits[
            :, strand_index
        ]
        field.carrier_logits.copy_(carrier_logits)
        field.initial_carrier_probabilities.copy_(
            field.carrier_probabilities(float(temperature)).clamp_min(1e-6)
        )
        field.structured_delta_raw.zero_()
        field.structured_opacity_raw.zero_()

    initial_mass = initial_probabilities.mean(dim=0)
    hard_ids = initial_probabilities.argmax(dim=-1)
    return {
        "domain": normalized_domain,
        "domain_log_odds_bias": float(domain_bias),
        "fixed_global_quota": False,
        "residual_source_capacity_ceiling": 0.0,
        "active_routes_per_source": ["shell", "strand"],
        "initial_soft_mass": {
            name: float(initial_mass[index].detach().cpu())
            for index, name in enumerate(ROUTE_NAMES)
        },
        "initial_hard_fraction": {
            name: float((hard_ids == index).float().mean().detach().cpu())
            for index, name in enumerate(ROUTE_NAMES)
        },
        "preference_statistics": {
            "mean": float(preference.mean().detach().cpu()),
            "std": float(preference.std().detach().cpu()),
            "min": float(preference.min().detach().cpu()),
            "max": float(preference.max().detach().cpu()),
        },
        "zero_initialized_structured_delta": True,
        "transmittance_conserving_soft_routes": True,
        "straight_through_hard_forward_required": True,
    }


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
    fixed_base_gaussian_ply: str | Path | None = None,
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
    adaptive_hard_router = (
        representation == "unified"
        and cfg.fiber_teacher_adaptive_migration_domain is not None
        and bool(cfg.fiber_adaptive_migration_hard_router)
    )
    frame_paths = _frame_paths(frame_dir)
    width, height = _resolve_render_size(render_size, frame_paths, stage1_npz)
    point_budget = _resolve_fiber_point_budget(
        cfg,
        (width, height),
        explicit_max_points=max_points,
    )
    motion = DifferentiableSurfaceScaffold(stage1_npz, device=device)
    motion.joints.requires_grad_(False)
    with np.load(stage1_npz, allow_pickle=False) as stage1_payload:
        scalp_face_indices = (
            stage1_payload["scalp_face_indices"].astype(np.int64)
            if "scalp_face_indices" in stage1_payload.files
            else None
        )
    split_fixed_base = bool(cfg.fiber_split_fixed_base)
    learnable_source_mask_mode = (
        "foreground" if split_fixed_base else str(cfg.fiber_source_mask_mode)
    )
    field = create_unified_fiber_field(
        gaussian_ply,
        motion.rest_surface_vertices.detach().cpu().numpy(),
        motion.surface_faces.detach().cpu().numpy(),
        device=device,
        max_points=point_budget,
        point_sampling_mode=str(cfg.fiber_point_sampling_mode),
        exact_vertex_binding=bool(cfg.fiber_exact_vertex_binding),
        binding_mode=str(cfg.fiber_binding_mode),
        source_mask_mode=learnable_source_mask_mode,
        source_mask_threshold=float(cfg.fiber_source_mask_threshold),
        source_min_opacity=float(cfg.fiber_source_min_opacity),
        residual_max_scale_fraction=float(
            cfg.fiber_residual_max_scale_fraction
        ),
        semantic_mask_from_source=bool(cfg.fiber_semantic_mask_from_source),
        structured_foreground_only=bool(cfg.fiber_structured_foreground_only),
        default_opacity=float(cfg.fiber_default_opacity),
        default_opacity_reference_points=int(
            cfg.fiber_default_opacity_reference_points
        ),
        neighbor_k=(
            int(cfg.fiber_route_neighbor_k) if representation == "unified" else 0
        ),
        shell_propagated_direction_weight=float(
            cfg.fiber_shell_propagated_direction_weight
        ),
        root_barycentric_max_delta=float(
            cfg.fiber_root_barycentric_max_delta
        ),
        expert_sh_max_delta=float(cfg.fiber_expert_sh_max_delta),
        expert_sh_degree=int(cfg.fiber_expert_sh_degree),
        initial_residual_trust=float(cfg.fiber_initial_residual_trust),
        initial_shell_length_scale=cfg.fiber_initial_shell_length_scale,
        initial_strand_length_scale=cfg.fiber_initial_strand_length_scale,
        initialize_direction_from_normal=bool(
            cfg.fiber_initialize_direction_from_normal
        ),
        scalp_face_indices=scalp_face_indices,
        binding_cache=(
            partition_binding_cache(cfg.fiber_binding_cache, "foreground")
            if split_fixed_base
            else cfg.fiber_binding_cache
        ),
    )
    fixed_base_count = 0
    fixed_base_source = (
        Path(fixed_base_gaussian_ply)
        if fixed_base_gaussian_ply is not None
        else Path(gaussian_ply)
    )
    if split_fixed_base:
        fixed_base = attach_fixed_gaussian_base(
            field,
            fixed_base_source,
            motion.rest_surface_vertices.detach().cpu().numpy(),
            motion.surface_faces.detach().cpu().numpy(),
            device=device,
            point_sampling_mode=str(cfg.fiber_point_sampling_mode),
            exact_vertex_binding=bool(cfg.fiber_exact_vertex_binding),
            binding_mode=str(cfg.fiber_binding_mode),
            source_mask_threshold=float(cfg.fiber_source_mask_threshold),
            source_min_opacity=float(cfg.fiber_source_min_opacity),
            residual_max_scale_fraction=float(
                cfg.fiber_fixed_base_max_scale_fraction
            ),
            scalp_face_indices=scalp_face_indices,
            binding_cache=cfg.fiber_binding_cache,
        )
        fixed_base_count = fixed_base.point_count
    bootstrap_metadata = None
    if residual_bootstrap_checkpoint is not None:
        bootstrap_metadata = _load_residual_bootstrap_checkpoint(
            field,
            residual_bootstrap_checkpoint,
            bootstrap_route_mass=cfg.fiber_bootstrap_route_mass,
            bootstrap_route_temperature=cfg.fiber_final_temperature,
        )
    teacher_requested = (
        (representation == "unified" and bool(cfg.fiber_freeze_residual_teacher))
        or float(cfg.fiber_teacher_nonregression_weight) > 0.0
        or float(cfg.fiber_structured_spill_weight) > 0.0
        or int(cfg.fiber_coverage_seed_count) > 0
        or cfg.fiber_teacher_semantic_migration_mass is not None
        or cfg.fiber_teacher_adaptive_migration_domain is not None
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
        _freeze_residual_teacher_scaffold(
            field,
            optimize_structured_base_appearance=bool(
                cfg.fiber_optimize_structured_base_appearance
            ),
        )
    if (
        representation == "unified"
        and bool(cfg.fiber_topology_initial_structured_off)
    ):
        # Start exactly from the editable residual student.  Shell/strand
        # capacity exists but is not allowed to consume routing mass until a
        # multi-view seed/grow event explicitly activates it.
        with torch.no_grad():
            field.route_active_gate[:, :2] = 0.0
            field.route_active_gate[:, ROUTE_NAMES.index("residual")] = 1.0
    semantic_migration_report: dict[str, object] | None = None
    if (
        cfg.fiber_teacher_semantic_migration_mass is not None
        and cfg.fiber_teacher_adaptive_migration_domain is not None
    ):
        raise ValueError(
            "Fixed semantic migration and adaptive residual-free migration "
            "are mutually exclusive"
        )
    if (
        representation == "unified"
        and cfg.fiber_teacher_semantic_migration_mass is not None
    ):
        if bootstrap_metadata is None or teacher_field is None:
            raise ValueError(
                "fiber_teacher_semantic_migration_mass requires a frozen "
                "residual bootstrap teacher"
            )
        if bool(cfg.fiber_topology_initial_structured_off):
            raise ValueError(
                "semantic teacher migration is incompatible with "
                "fiber_topology_initial_structured_off"
            )
        semantic_migration_report = (
            _initialize_render_preserving_semantic_migration(
                field,
                cfg.fiber_teacher_semantic_migration_mass,
                temperature=float(cfg.fiber_final_temperature),
            )
        )
    elif (
        representation == "unified"
        and cfg.fiber_teacher_adaptive_migration_domain is not None
    ):
        if bootstrap_metadata is None or teacher_field is None:
            raise ValueError(
                "fiber_teacher_adaptive_migration_domain requires a frozen "
                "residual bootstrap teacher"
            )
        if bool(cfg.fiber_topology_initial_structured_off):
            raise ValueError(
                "adaptive teacher migration is incompatible with "
                "fiber_topology_initial_structured_off"
            )
        semantic_migration_report = (
            _initialize_render_preserving_adaptive_migration(
                field,
                cfg.fiber_teacher_adaptive_migration_domain,
                domain_bias=float(cfg.fiber_teacher_adaptive_migration_bias),
                temperature=float(cfg.fiber_final_temperature),
            )
        )

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
        int(cfg.fiber_calibration_frames)
        if (
            representation == "unified"
            or float(cfg.fiber_teacher_nonregression_weight) > 0.0
        )
        else 0
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
        float(cfg.fiber_teacher_nonregression_weight) > 0.0
        and not calibration_frame_indices
    ):
        raise ValueError(
            "fiber_teacher_nonregression_weight requires fiber_calibration_frames > 0"
        )
    orientation_targets = _load_orientation_targets(
        frame_paths,
        frame_indices,
        width,
        height,
        device,
        cfg.fiber_orientation_dir,
        distribution_radius=int(cfg.fiber_orientation_distribution_radius),
    )

    semantic_migration_equivalence: dict[str, object] | None = None
    if semantic_migration_report is not None:
        if teacher_field is None:
            raise RuntimeError("semantic migration lost its frozen teacher")
        first_frame = frame_indices[0]
        with torch.no_grad():
            _tet_nodes, equivalence_vertices, _joints = motion.driven_points(
                motion_indices[first_frame]
            )
        semantic_migration_equivalence = (
            _measure_semantic_migration_render_equivalence(
                field,
                teacher_field,
                equivalence_vertices,
                motion.surface_faces,
                cameras[first_frame],
                cfg,
                renderer_name,
            )
        )
        equivalence_path = out_dir / "semantic_migration_equivalence.json"
        equivalence_path.write_text(
            json.dumps(semantic_migration_equivalence, indent=2),
            encoding="utf-8",
        )
        tolerance = float(cfg.fiber_teacher_semantic_migration_tolerance)
        if (
            tolerance > 0.0
            and float(semantic_migration_equivalence["maximum_mean_absolute_error"])
            > tolerance
        ):
            raise RuntimeError(
                "Render-preserving semantic migration exceeded tolerance: "
                f"{semantic_migration_equivalence['maximum_mean_absolute_error']:.6g} "
                f"> {tolerance:.6g}; see {equivalence_path}"
            )

    visual_hull_frame_indices = frame_indices + calibration_frame_indices
    visual_hull_ground_truth = ground_truth + calibration_ground_truth
    visual_hull_vertices: list[torch.Tensor] = []
    if (
        representation == "unified"
        and (
            int(cfg.fiber_visual_hull_update_every) > 0
            or int(cfg.fiber_coverage_seed_count) > 0
            or bool(cfg.fiber_surface_propagation_enabled)
        )
    ):
        with torch.no_grad():
            for visual_frame in visual_hull_frame_indices:
                _tet_nodes, vertices, _joints = motion.driven_points(
                    motion_indices[visual_frame]
                )
                visual_hull_vertices.append(vertices)

    _validate_route_training_config(cfg)
    surface_propagation_report: dict[str, object] | None = None
    if representation == "unified" and bool(
        cfg.fiber_surface_propagation_enabled
    ):
        surface_propagation_report = _initialize_intrinsic_surface_propagation(
            field,
            motion.surface_faces,
            visual_hull_vertices[: len(frame_indices)],
            cameras,
            frame_indices,
            ground_truth,
            orientation_targets,
            cfg,
            out_dir,
        )
        if teacher_field is not None:
            first_frame = frame_indices[0]
            propagation_equivalence = (
                _measure_semantic_migration_render_equivalence(
                    field,
                    teacher_field,
                    visual_hull_vertices[0],
                    motion.surface_faces,
                    cameras[first_frame],
                    cfg,
                    renderer_name,
                )
            )
            (out_dir / "surface_propagation_equivalence.json").write_text(
                json.dumps(propagation_equivalence, indent=2), encoding="utf-8"
            )
            tolerance = float(cfg.fiber_teacher_semantic_migration_tolerance)
            if tolerance > 0.0 and float(
                propagation_equivalence["maximum_mean_absolute_error"]
            ) > tolerance:
                raise RuntimeError(
                    "Surface propagation changed the render-preserving "
                    "initialization beyond tolerance"
                )
    coverage_seed_report: dict[str, object] | None = None
    coverage_seed_teacher_masks: dict[int, torch.Tensor] = {}
    if representation == "unified" and int(cfg.fiber_coverage_seed_count) > 0:
        if teacher_field is None:
            raise ValueError("Coverage seeding requires a frozen residual teacher")
        coverage_orientation_targets = None
        if bool(cfg.fiber_coverage_seed_orientation_init):
            coverage_orientation_targets = _load_orientation_targets(
                frame_paths,
                visual_hull_frame_indices,
                width,
                height,
                device,
                cfg.fiber_orientation_dir,
                distribution_radius=int(
                    cfg.fiber_orientation_distribution_radius
                ),
            )
        coverage_seed_report, coverage_seed_teacher_masks = (
            _initialize_multiview_coverage_seeds(
                field,
                teacher_field,
                motion.surface_faces,
                visual_hull_vertices,
                cameras,
                visual_hull_frame_indices,
                visual_hull_ground_truth,
                coverage_orientation_targets,
                cfg,
                renderer_name,
                out_dir,
            )
        )
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
    topology_jsonl = out_dir / "topology_events.jsonl"
    topology_jsonl.write_text("", encoding="utf-8")
    topology_events: list[dict[str, object]] = []
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
    latest_teacher_calibration_student: float | None = None
    latest_teacher_calibration_residual: float | None = None
    latest_teacher_calibration_step: int | None = None
    dropout_counts = {name: 0 for name in ROUTE_NAMES}
    shell_visibility: torch.Tensor | None = None
    strand_visibility: torch.Tensor | None = None
    latest_visual_hull_report: dict[str, float] | None = None
    visual_hull_update_count = 0
    teacher_mask_cache: dict[int, torch.Tensor] = dict(
        coverage_seed_teacher_masks
    )

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
            target_shell_points: torch.Tensor | None = None
            target_strand_points: torch.Tensor | None = None

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
                if adaptive_hard_router:
                    # A one-route forward pass keeps the residual-free
                    # migration exactly one primitive per teacher source.
                    # route_probabilities uses a straight-through estimator,
                    # so shell/strand logits still receive soft gradients.
                    route_hardening = 1.0
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
                    (
                        shell_visibility,
                        strand_visibility,
                        latest_visual_hull_report,
                    ) = _compute_visual_hull_gates(
                            field,
                            visual_hull_vertices,
                            motion.surface_faces,
                            [cameras[index] for index in visual_hull_frame_indices],
                            visual_hull_ground_truth,
                            cfg,
                            temperature,
                            geometry_blend,
                        )
                    field.shell_visibility_gate = shell_visibility.detach()
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
                    shell_visibility=shell_visibility,
                    strand_visibility=strand_visibility,
                    fin_aspect_ratio=cfg.fiber_fin_aspect_ratio,
                    additive_teacher=cfg.fiber_additive_teacher_mode,
                    teacher_opacity_transfer=cfg.fiber_teacher_opacity_transfer,
                )
                if (
                    phase != "gaussian_scaffold"
                    and bool(cfg.fiber_visual_hull_target_geometry)
                ):
                    target_shell_points = field.shell_target_geometry(
                        surface_vertices,
                        motion.surface_faces,
                        shell_samples=int(cfg.fiber_shell_samples),
                    )
                    target_strand_points, _target_directions = (
                        field.strand_target_geometry(
                            surface_vertices,
                            motion.surface_faces,
                            strand_samples=int(cfg.fiber_strand_samples),
                        )
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
            deployment_prediction = None
            deployment_render_loss = render_loss.new_zeros(())
            needs_deployment_prediction = (
                representation == "unified"
                and phase != "gaussian_scaffold"
                and (
                    float(cfg.fiber_deployment_render_weight) > 0.0
                    or float(cfg.fiber_rgb_gradient_weight) > 0.0
                    or float(cfg.fiber_structured_spill_weight) > 0.0
                    or float(cfg.fiber_shell_render_spill_weight) > 0.0
                    or float(cfg.fiber_mask_inside_coverage_weight) > 0.0
                    or float(cfg.fiber_mask_outside_spill_weight) > 0.0
                    or float(
                        cfg.fiber_structured_mask_inside_coverage_weight
                    )
                    > 0.0
                    or float(cfg.fiber_structured_mask_outside_spill_weight)
                    > 0.0
                )
            )
            if needs_deployment_prediction:
                force_target_geometry = bool(
                    cfg.fiber_deployment_force_target_geometry
                )
                deployment_shell_visibility = shell_visibility
                deployment_strand_visibility = strand_visibility
                deployment_delta_override = None
                if force_target_geometry:
                    supported_target = _structured_support_mask(
                        field,
                        float(
                            cfg.fiber_structure_detach_min_support_fraction
                        ),
                    )
                    deployment_delta_override = torch.where(
                        supported_target,
                        torch.ones_like(field.structured_delta_gain),
                        field.structured_delta_gain.detach(),
                    )
                # The optional target render is deliberately stricter than
                # fiber-eval: it exposes the analytic geometry even while the
                # learned main branch remains collapsed near its teacher.
                deployment_primitives = field.primitives(
                    surface_vertices,
                    motion.surface_faces,
                    shell_samples=cfg.fiber_shell_samples,
                    strand_samples=cfg.fiber_strand_samples,
                    temperature=cfg.fiber_final_temperature,
                    hard_route=False,
                    route_blend=1.0,
                    geometry_blend=1.0,
                    route_hardening=(1.0 if adaptive_hard_router else 0.0),
                    hard_route_policy=cfg.fiber_hard_route_policy,
                    shell_visibility=deployment_shell_visibility,
                    strand_visibility=deployment_strand_visibility,
                    fin_aspect_ratio=cfg.fiber_fin_aspect_ratio,
                    additive_teacher=cfg.fiber_additive_teacher_mode,
                    teacher_opacity_transfer=cfg.fiber_teacher_opacity_transfer,
                    structured_delta_override=deployment_delta_override,
                )
                deployment_prediction = _render(
                    deployment_primitives, camera, cfg, renderer_name
                )
                if float(cfg.fiber_deployment_render_weight) > 0.0:
                    deployment_render_loss, _deployment_parts = (
                        differentiable_render_loss(
                            deployment_prediction,
                            ground_truth[local_frame_index]["rgb"],
                            ground_truth[local_frame_index]["mask"],
                            cfg.color_loss_weight,
                            cfg.mask_loss_weight,
                            cfg.mask_boundary_weight,
                            cfg.mask_boundary_radius,
                            cfg.mask_balance_weight,
                        )
                    )
                    render_loss = render_loss + (
                        float(cfg.fiber_deployment_render_weight)
                        * deployment_render_loss
                    )
            rgb_gradient_loss = render_loss.new_zeros(())
            if float(cfg.fiber_rgb_gradient_weight) > 0.0:
                gradient_prediction = (
                    deployment_prediction
                    if deployment_prediction is not None
                    else prediction
                )
                rgb_gradient_loss = _masked_rgb_gradient_loss(
                    gradient_prediction["rgb"],
                    ground_truth[local_frame_index]["rgb"],
                    ground_truth[local_frame_index]["mask"],
                )
                render_loss = render_loss + (
                    float(cfg.fiber_rgb_gradient_weight) * rgb_gradient_loss
                )
            structured_spill_loss = render_loss.new_zeros(())
            teacher_mask = None
            needs_teacher_mask = (
                phase != "gaussian_scaffold"
                and teacher_field is not None
                and (
                    float(cfg.fiber_structured_spill_weight) > 0.0
                    or float(cfg.fiber_mask_outside_spill_weight) > 0.0
                )
            )
            if needs_teacher_mask:
                teacher_mask = teacher_mask_cache.get(frame_index)
                if teacher_mask is None:
                    with torch.no_grad():
                        teacher_primitives = teacher_field.residual_primitives(
                            surface_vertices, motion.surface_faces
                        )
                        teacher_prediction = _render(
                            teacher_primitives,
                            camera,
                            cfg,
                            renderer_name,
                        )
                        teacher_mask = teacher_prediction["mask"].detach()
                        teacher_mask_cache[frame_index] = teacher_mask
            if (
                phase != "gaussian_scaffold"
                and float(cfg.fiber_structured_spill_weight) > 0.0
            ):
                if teacher_field is None:
                    raise RuntimeError(
                        "Structured spill supervision requires a residual teacher"
                    )
                if teacher_mask is None:
                    raise RuntimeError("Residual teacher mask was not rendered")
                structured_spill_loss = _structured_spill_loss(
                    (
                        deployment_prediction["mask"]
                        if deployment_prediction is not None
                        else prediction["mask"]
                    ),
                    teacher_mask,
                    ground_truth[local_frame_index]["mask"],
                )
                render_loss = render_loss + (
                    float(cfg.fiber_structured_spill_weight)
                    * structured_spill_loss
                )
            mask_inside_coverage_loss = render_loss.new_zeros(())
            mask_outside_spill_loss = render_loss.new_zeros(())
            maximum_hole_loss = render_loss.new_zeros(())
            structured_mask_inside_coverage_loss = render_loss.new_zeros(())
            structured_mask_outside_spill_loss = render_loss.new_zeros(())
            if (
                representation == "unified"
                and phase != "gaussian_scaffold"
                and (
                    float(cfg.fiber_mask_inside_coverage_weight) > 0.0
                    or float(cfg.fiber_mask_outside_spill_weight) > 0.0
                )
            ):
                if deployment_prediction is None:
                    raise RuntimeError(
                        "Bidirectional mask supervision requires deployed primitives"
                    )
                (
                    mask_inside_coverage_loss,
                    mask_outside_spill_loss,
                ) = _bidirectional_mask_losses(
                    deployment_prediction["mask"],
                    ground_truth[local_frame_index]["mask"],
                    inside_alpha_target=float(
                        cfg.fiber_mask_inside_alpha_target
                    ),
                    outside_margin_px=int(cfg.fiber_mask_outside_margin_px),
                    outside_reference_mask=(
                        teacher_mask
                        if (
                            teacher_field is not None
                            and bool(cfg.fiber_additive_teacher_mode)
                        )
                        else None
                    ),
                )
                render_loss = render_loss + (
                    float(cfg.fiber_mask_inside_coverage_weight)
                    * mask_inside_coverage_loss
                    + float(cfg.fiber_mask_outside_spill_weight)
                    * mask_outside_spill_loss
                )
            if (
                representation == "unified"
                and phase != "gaussian_scaffold"
                and float(cfg.fiber_max_hole_weight) > 0.0
            ):
                if deployment_prediction is None:
                    raise RuntimeError(
                        "Maximum-hole supervision requires deployed primitives"
                    )
                maximum_hole_loss = _maximum_hole_soft_loss(
                    deployment_prediction["mask"],
                    ground_truth[local_frame_index]["mask"],
                    kernel_size=int(cfg.fiber_max_hole_kernel),
                    topk_fraction=float(cfg.fiber_max_hole_topk_fraction),
                )
                render_loss = render_loss + (
                    float(cfg.fiber_max_hole_weight) * maximum_hole_loss
                )
            if (
                representation == "unified"
                and phase != "gaussian_scaffold"
                and (
                    float(cfg.fiber_structured_mask_inside_coverage_weight)
                    > 0.0
                    or float(cfg.fiber_structured_mask_outside_spill_weight)
                    > 0.0
                )
            ):
                if deployment_prediction is None:
                    raise RuntimeError(
                        "Structured mask supervision requires deployed primitives"
                    )
                structured_keep = (
                    deployment_primitives.route_id
                    != ROUTE_NAMES.index("residual")
                ).to(deployment_primitives.opacity.dtype)
                structured_prediction = _render(
                    replace(
                        deployment_primitives,
                        opacity=deployment_primitives.opacity * structured_keep,
                    ),
                    camera,
                    cfg,
                    renderer_name,
                )
                (
                    structured_mask_inside_coverage_loss,
                    structured_mask_outside_spill_loss,
                ) = _bidirectional_mask_losses(
                    structured_prediction["mask"],
                    ground_truth[local_frame_index]["mask"],
                    inside_alpha_target=float(
                        cfg.fiber_structured_mask_inside_alpha_target
                    ),
                    outside_margin_px=int(cfg.fiber_mask_outside_margin_px),
                )
                render_loss = render_loss + (
                    float(cfg.fiber_structured_mask_inside_coverage_weight)
                    * structured_mask_inside_coverage_loss
                    + float(cfg.fiber_structured_mask_outside_spill_weight)
                    * structured_mask_outside_spill_loss
                )
            shell_render_spill_loss = render_loss.new_zeros(())
            if (
                representation == "unified"
                and phase != "gaussian_scaffold"
                and float(cfg.fiber_shell_render_spill_weight) > 0.0
            ):
                if deployment_prediction is None:
                    raise RuntimeError(
                        "Shell render spill requires deployed primitives"
                    )
                shell_keep = (
                    deployment_primitives.route_id
                    == ROUTE_NAMES.index("shell")
                ).to(deployment_primitives.opacity.dtype)
                shell_prediction = _render(
                    replace(
                        deployment_primitives,
                        opacity=deployment_primitives.opacity * shell_keep,
                    ),
                    camera,
                    cfg,
                    renderer_name,
                )
                shell_render_spill_loss = _rendered_route_spill_loss(
                    shell_prediction["mask"],
                    ground_truth[local_frame_index]["mask"],
                    margin_px=max(int(cfg.fiber_visual_hull_margin_px), 0),
                )
            orientation_loss = render_loss.new_zeros(())
            orientation_distribution_loss = render_loss.new_zeros(())
            if orientation_target is not None:
                orientation_loss = _orientation_consistency_loss(
                    prediction["orientation"], orientation_target,
                    ground_truth[local_frame_index]["mask"],
                )
                orientation_distribution_loss = _orientation_distribution_loss(
                    prediction["orientation"],
                    prediction["orientation4"],
                    orientation_target,
                    ground_truth[local_frame_index]["mask"],
                    prediction.get("physical_mask", prediction["mask"]),
                )
                render_loss = render_loss + (
                    float(cfg.fiber_orientation_weight) * orientation_loss
                    + float(cfg.fiber_orientation_distribution_weight)
                    * orientation_distribution_loss
                )
            visual_hull_loss = render_loss.new_zeros(())
            if (
                representation == "unified"
                and phase != "gaussian_scaffold"
                and float(cfg.fiber_visual_hull_weight) > 0.0
            ):
                strand_points = (
                    target_strand_points
                    if target_strand_points is not None
                    else _strand_points_from_primitives(
                        primitives,
                        field.point_count,
                        int(cfg.fiber_strand_samples),
                    )
                )
                strand_front_visibility = None
                if bool(cfg.fiber_visual_hull_occlusion_aware):
                    strand_front_visibility = _front_visible_sample_gate(
                        strand_points,
                        surface_vertices,
                        camera,
                        bin_px=int(cfg.fiber_visual_hull_occlusion_bin_px),
                        depth_tolerance=(
                            float(field.scene_scale.detach().cpu())
                            * float(cfg.fiber_visual_hull_occlusion_depth_scale)
                        ),
                    )
                visual_hull_loss = _visual_hull_soft_loss(
                    strand_points,
                    primitives.route_probabilities,
                    camera,
                    ground_truth[local_frame_index]["mask"],
                    int(cfg.fiber_visual_hull_margin_px),
                    sample_gate=strand_visibility,
                    visibility_gate=strand_front_visibility,
                )
            shell_visual_hull_loss = render_loss.new_zeros(())
            if (
                representation == "unified"
                and phase != "gaussian_scaffold"
                and float(cfg.fiber_shell_visual_hull_weight) > 0.0
            ):
                shell_points = (
                    target_shell_points
                    if target_shell_points is not None
                    else _shell_points_from_primitives(
                        primitives,
                        field.point_count,
                        int(cfg.fiber_shell_samples),
                    )
                )
                shell_front_visibility = None
                if bool(cfg.fiber_visual_hull_occlusion_aware):
                    shell_front_visibility = _front_visible_sample_gate(
                        shell_points,
                        surface_vertices,
                        camera,
                        bin_px=int(cfg.fiber_visual_hull_occlusion_bin_px),
                        depth_tolerance=(
                            float(field.scene_scale.detach().cpu())
                            * float(cfg.fiber_visual_hull_occlusion_depth_scale)
                        ),
                    )
                shell_visual_hull_loss = _route_visual_hull_soft_loss(
                    shell_points,
                    primitives.route_probabilities,
                    ROUTE_NAMES.index("shell"),
                    camera,
                    ground_truth[local_frame_index]["mask"],
                    int(cfg.fiber_visual_hull_margin_px),
                    sample_gate=shell_visibility,
                    visibility_gate=shell_front_visibility,
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
                    strand_points=target_strand_points,
                )
            regularizers = field.regularizers(
                surface_vertices,
                motion.surface_faces,
                temperature=temperature,
                structure_min_deployment_gain=float(
                    cfg.fiber_structure_min_deployment_gain
                ),
                strand_min_deployment_gain=float(
                    cfg.fiber_strand_min_deployment_gain
                ),
                strand_min_deployed_length_scale=float(
                    cfg.fiber_strand_min_deployed_length_scale
                ),
                strand_coverage_target=float(cfg.fiber_strand_coverage_target),
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
                        float(cfg.fiber_final_temperature),
                        1.0,
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
            teacher_calibration_student = latest_teacher_calibration_student
            teacher_calibration_residual = latest_teacher_calibration_residual
            nonreg_every = int(cfg.fiber_teacher_nonregression_every)
            if (
                phase != "gaussian_scaffold"
                and teacher_field is not None
                and calibration_frame_indices
                and float(cfg.fiber_teacher_nonregression_weight) > 0.0
                and (step - warmup_steps) % nonreg_every == 0
            ):
                event_index = (step - warmup_steps) // nonreg_every
                view_count = min(
                    int(cfg.fiber_teacher_nonregression_views_per_step),
                    len(calibration_frame_indices),
                )
                start_local = (event_index * view_count) % len(
                    calibration_frame_indices
                )
                calibration_locals = [
                    (start_local + offset) % len(calibration_frame_indices)
                    for offset in range(view_count)
                ]
                student_losses: list[torch.Tensor] = []
                teacher_losses: list[torch.Tensor] = []
                for calibration_local in calibration_locals:
                    calibration_frame = calibration_frame_indices[calibration_local]
                    with torch.no_grad():
                        _tet_nodes, calibration_vertices, _joints = motion.driven_points(
                            motion_indices[calibration_frame]
                        )
                    if representation == "residual_only":
                        calibration_student_primitives = field.residual_primitives(
                            calibration_vertices, motion.surface_faces
                        )
                    else:
                        calibration_student_primitives = field.primitives(
                            calibration_vertices,
                            motion.surface_faces,
                            shell_samples=cfg.fiber_shell_samples,
                            strand_samples=cfg.fiber_strand_samples,
                            temperature=cfg.fiber_final_temperature,
                            hard_route=False,
                            route_blend=1.0,
                            geometry_blend=1.0,
                            route_hardening=(
                                1.0 if adaptive_hard_router else 0.0
                            ),
                            hard_route_policy=cfg.fiber_hard_route_policy,
                            strand_visibility=strand_visibility,
                            fin_aspect_ratio=cfg.fiber_fin_aspect_ratio,
                            additive_teacher=cfg.fiber_additive_teacher_mode,
                            teacher_opacity_transfer=(
                                cfg.fiber_teacher_opacity_transfer
                            ),
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
                    student_losses.append(calibration_student_loss)
                    teacher_losses.append(teacher_loss)
                student_loss_stack = torch.stack(student_losses)
                teacher_loss_stack = torch.stack(teacher_losses)
                calibration_violations = F.relu(
                    student_loss_stack
                    - teacher_loss_stack
                    - float(cfg.fiber_teacher_nonregression_margin)
                )
                if cfg.fiber_teacher_nonregression_reduction == "max":
                    teacher_nonregression = calibration_violations.max()
                else:
                    teacher_nonregression = calibration_violations.mean()
                teacher_calibration_student = float(
                    student_loss_stack.detach().mean().cpu()
                )
                teacher_calibration_residual = float(
                    teacher_loss_stack.detach().mean().cpu()
                )
                latest_teacher_calibration_student = teacher_calibration_student
                latest_teacher_calibration_residual = teacher_calibration_residual
                latest_teacher_calibration_step = step
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
                    + cfg.fiber_shell_visual_hull_weight
                    * shell_visual_hull_loss
                    + cfg.fiber_shell_render_spill_weight
                    * shell_render_spill_loss
                    + cfg.fiber_fin_silhouette_weight * fin_silhouette_loss
                    + cfg.fiber_strand_support_weight * strand_support_loss
                    + cfg.fiber_structure_deployment_weight
                    * regularizers["structure_deployment"]
                    + cfg.fiber_strand_field_weight
                    * regularizers["strand_field"]
                    + cfg.fiber_strand_deployability_weight
                    * regularizers["strand_deployability"]
                    + cfg.fiber_strand_coverage_weight
                    * regularizers["strand_coverage_deficit"]
                    + cfg.fiber_shell_normal_weight * regularizers["shell_normal"]
                    + cfg.fiber_shell_length_weight * regularizers["shell_length"]
                    + cfg.fiber_strand_thinness_weight * regularizers["strand_thinness"]
                    + cfg.fiber_height_weight * regularizers["height"]
                    + cfg.fiber_bend_weight * regularizers["bend"]
                    + cfg.fiber_residual_drift_weight * regularizers["residual_drift"]
                    + cfg.fiber_residual_trust_weight * regularizers["residual_trust"]
                    + cfg.fiber_expert_appearance_weight
                    * regularizers["expert_appearance"]
                    + cfg.fiber_expert_sh_weight * regularizers["expert_sh"]
                    + cfg.fiber_root_barycentric_weight
                    * regularizers["root_barycentric"]
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

            completed_step = step + 1
            forced_deployment_floor = _scheduled_structure_detach_floor(
                completed_step, total_steps, cfg
            )
            _enforce_structured_deployment_floor(
                field,
                forced_deployment_floor,
                min_support_fraction=float(
                    cfg.fiber_structure_detach_min_support_fraction
                ),
            )
            sign_projection_report: dict[str, float] | None = None
            sign_projection_every = int(cfg.fiber_strand_sign_projection_every)
            if (
                representation == "unified"
                and sign_projection_every > 0
                and completed_step % sign_projection_every == 0
            ):
                with torch.no_grad():
                    root, tangent, bitangent, normal = field.surface_frame(
                        motion.rest_surface_vertices, motion.surface_faces
                    )
                    del root
                    rest_frame = torch.stack([tangent, bitangent, normal], dim=-1)
                    strand_active = field.route_active_gate[:, 1] > 0.5
                    signed, sign_projection_report = (
                        _synchronize_outward_direction_signs(
                            field.direction_local.detach(),
                            rest_frame,
                            field.route_neighbor_index,
                            anchor_threshold=float(
                                cfg.fiber_strand_outward_anchor_threshold
                            ),
                            steps=int(cfg.fiber_strand_sign_projection_steps),
                            active_mask=strand_active,
                        )
                    )
                    field.direction_local_raw[strand_active] = signed[strand_active]

            topology_event: dict[str, object] | None = None
            topology_every = int(cfg.fiber_topology_update_every)
            topology_start = max(int(cfg.fiber_topology_start_step), warmup_steps)
            topology_stop = int(cfg.fiber_topology_stop_step)
            should_update_topology = (
                representation == "unified"
                and phase != "gaussian_scaffold"
                and topology_every > 0
                and completed_step >= topology_start
                and (topology_stop <= 0 or completed_step <= topology_stop)
                and (completed_step - topology_start) % topology_every == 0
            )
            if should_update_topology:
                topology_event = _update_adaptive_topology(
                    field,
                    optimizer,
                    visual_hull_vertices,
                    motion.surface_faces,
                    [cameras[index] for index in visual_hull_frame_indices],
                    visual_hull_ground_truth,
                    cfg,
                    renderer_name,
                    step=completed_step,
                    validation_surface_vertices=(
                        visual_hull_vertices[-len(calibration_frame_indices) :]
                        if calibration_frame_indices
                        else []
                    ),
                    validation_cameras=[
                        cameras[index] for index in calibration_frame_indices
                    ],
                    validation_targets=calibration_ground_truth,
                )
                topology_events.append(topology_event)
                with open(topology_jsonl, "a", encoding="utf-8") as topology_file:
                    topology_file.write(
                        json.dumps(topology_event, sort_keys=True) + "\n"
                    )
                print(
                    "fiber_topology="
                    + json.dumps(topology_event, sort_keys=True),
                    flush=True,
                )

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
                "rgb_gradient": float(rgb_gradient_loss.detach().cpu()),
                "deployment_render": float(
                    deployment_render_loss.detach().cpu()
                ),
                "structured_spill": float(structured_spill_loss.detach().cpu()),
                "mask_inside_coverage": float(
                    mask_inside_coverage_loss.detach().cpu()
                ),
                "mask_outside_spill": float(
                    mask_outside_spill_loss.detach().cpu()
                ),
                "maximum_hole": float(maximum_hole_loss.detach().cpu()),
                "structured_mask_inside_coverage": float(
                    structured_mask_inside_coverage_loss.detach().cpu()
                ),
                "structured_mask_outside_spill": float(
                    structured_mask_outside_spill_loss.detach().cpu()
                ),
                "shell_render_spill": float(
                    shell_render_spill_loss.detach().cpu()
                ),
                "orientation": float(orientation_loss.detach().cpu()),
                "orientation_distribution": float(
                    orientation_distribution_loss.detach().cpu()
                ),
                "risk_calibration": float(risk_calibration.detach().cpu()),
                "negative_contribution": float(
                    negative_contribution.detach().cpu()
                ),
                "teacher_nonregression": float(
                    teacher_nonregression.detach().cpu()
                ),
                "visual_hull": float(visual_hull_loss.detach().cpu()),
                "shell_visual_hull": float(
                    shell_visual_hull_loss.detach().cpu()
                ),
                "fin_silhouette": float(fin_silhouette_loss.detach().cpu()),
                "strand_support": float(strand_support_loss.detach().cpu()),
                "structure_deployment": float(
                    regularizers["structure_deployment"].detach().cpu()
                    if representation == "unified"
                    else 0.0
                ),
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
                    "forced_deployment_floor": forced_deployment_floor,
                    "sign_projection": sign_projection_report,
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
                    "latest_teacher_calibration_step": (
                        latest_teacher_calibration_step
                    ),
                    "latest_calibration_frames": latest_calibration_frames,
                    "latest_calibration_loss": latest_calibration_loss,
                    "topology_event": topology_event,
                    "active_topology": field.active_topology_summary(
                        shell_samples=int(cfg.fiber_shell_samples),
                        strand_samples=int(cfg.fiber_strand_samples),
                    ),
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
                "binding_mode": str(cfg.fiber_binding_mode),
                "source_mask_mode": learnable_source_mask_mode,
                "split_fixed_base": split_fixed_base,
                "fixed_base_count": fixed_base_count,
                "fixed_base_gaussian_ply": (
                    str(fixed_base_source.resolve()) if split_fixed_base else None
                ),
                "fixed_base_max_scale_fraction": float(
                    cfg.fiber_fixed_base_max_scale_fraction
                ),
                "semantic_mask_from_source": bool(
                    cfg.fiber_semantic_mask_from_source
                ),
                "structured_foreground_only": bool(
                    cfg.fiber_structured_foreground_only
                ),
                "source_mask_threshold": float(cfg.fiber_source_mask_threshold),
                "source_min_opacity": float(cfg.fiber_source_min_opacity),
                "residual_max_scale_fraction": float(
                    cfg.fiber_residual_max_scale_fraction
                ),
                "scene_scale": float(field.scene_scale.detach().cpu()),
                "shell_samples": cfg.fiber_shell_samples,
                "strand_samples": cfg.fiber_strand_samples,
                "shell_propagated_direction_weight": float(
                    cfg.fiber_shell_propagated_direction_weight
                ),
                "route_neighbor_k": int(cfg.fiber_route_neighbor_k),
                "surface_propagation_neighbor_k": int(
                    cfg.fiber_surface_propagation_neighbor_k
                ),
                "root_barycentric_max_delta": float(
                    cfg.fiber_root_barycentric_max_delta
                ),
                "expert_sh_max_delta": float(cfg.fiber_expert_sh_max_delta),
                "expert_sh_degree": int(cfg.fiber_expert_sh_degree),
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
                "coverage_seed_report": coverage_seed_report,
                "surface_propagation_report": surface_propagation_report,
                "semantic_migration_report": semantic_migration_report,
                "semantic_migration_equivalence": (
                    semantic_migration_equivalence
                ),
                "adaptive_topology": bool(
                    int(cfg.fiber_topology_update_every) > 0
                ),
                "active_topology": field.active_topology_summary(
                    shell_samples=int(cfg.fiber_shell_samples),
                    strand_samples=int(cfg.fiber_strand_samples),
                ),
                "topology_update_count": len(topology_events),
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
        "coverage_seed_report": coverage_seed_report,
        "surface_propagation_report": surface_propagation_report,
        "semantic_migration_report": semantic_migration_report,
        "semantic_migration_equivalence": semantic_migration_equivalence,
        "final_structure_deployment": (
            _structure_deployment_summary(field, cfg.fiber_final_temperature)
            if representation == "unified"
            else {"shell": 0.0, "strand": 0.0, "structured_mean": 0.0}
        ),
        "final_active_topology": field.active_topology_summary(
            shell_samples=int(cfg.fiber_shell_samples),
            strand_samples=int(cfg.fiber_strand_samples),
        ),
        "topology_update_count": len(topology_events),
        "topology_events": topology_events,
        "hard_route_policy": cfg.fiber_hard_route_policy,
        "config": {
            "representation": representation,
            "shell_samples": cfg.fiber_shell_samples,
            "strand_samples": cfg.fiber_strand_samples,
            "max_points": int(max_points or cfg.fiber_max_points),
            "binding_mode": str(cfg.fiber_binding_mode),
            "source_mask_mode": learnable_source_mask_mode,
            "split_fixed_base": split_fixed_base,
            "fixed_base_count": fixed_base_count,
            "fixed_base_gaussian_ply": (
                str(fixed_base_source.resolve()) if split_fixed_base else None
            ),
            "fixed_base_max_scale_fraction": float(
                cfg.fiber_fixed_base_max_scale_fraction
            ),
            "semantic_mask_from_source": bool(cfg.fiber_semantic_mask_from_source),
            "structured_foreground_only": bool(
                cfg.fiber_structured_foreground_only
            ),
            "source_mask_threshold": float(cfg.fiber_source_mask_threshold),
            "source_min_opacity": float(cfg.fiber_source_min_opacity),
            "residual_max_scale_fraction": float(
                cfg.fiber_residual_max_scale_fraction
            ),
            "base_lr": base_lr,
            "residual_bootstrap": bootstrap_metadata,
            "route_neighbor_k": int(cfg.fiber_route_neighbor_k),
            "route_neighbor_weight": float(cfg.fiber_route_neighbor_weight),
            "initial_residual_trust": float(cfg.fiber_initial_residual_trust),
            "teacher_semantic_migration_mass": (
                list(cfg.fiber_teacher_semantic_migration_mass)
                if cfg.fiber_teacher_semantic_migration_mass is not None
                else None
            ),
            "teacher_adaptive_migration_domain": (
                str(cfg.fiber_teacher_adaptive_migration_domain)
                if cfg.fiber_teacher_adaptive_migration_domain is not None
                else None
            ),
            "teacher_adaptive_migration_bias": float(
                cfg.fiber_teacher_adaptive_migration_bias
            ),
            "adaptive_migration_hard_router": bool(
                cfg.fiber_adaptive_migration_hard_router
            ),
            "optimize_structured_base_appearance": bool(
                cfg.fiber_optimize_structured_base_appearance
            ),
            "teacher_semantic_migration_tolerance": float(
                cfg.fiber_teacher_semantic_migration_tolerance
            ),
            "structure_deployment_weight": float(
                cfg.fiber_structure_deployment_weight
            ),
            "structure_min_deployment_gain": float(
                cfg.fiber_structure_min_deployment_gain
            ),
            "structure_detach_start_fraction": float(
                cfg.fiber_structure_detach_start_fraction
            ),
            "structure_detach_end_fraction": float(
                cfg.fiber_structure_detach_end_fraction
            ),
            "structure_detach_final_gain": float(
                cfg.fiber_structure_detach_final_gain
            ),
            "structure_detach_min_support_fraction": float(
                cfg.fiber_structure_detach_min_support_fraction
            ),
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
            "teacher_opacity_transfer": float(
                cfg.fiber_teacher_opacity_transfer
            ),
            "teacher_nonregression_weight": float(
                cfg.fiber_teacher_nonregression_weight
            ),
            "teacher_nonregression_margin": float(
                cfg.fiber_teacher_nonregression_margin
            ),
            "teacher_nonregression_views_per_step": int(
                cfg.fiber_teacher_nonregression_views_per_step
            ),
            "teacher_nonregression_reduction": str(
                cfg.fiber_teacher_nonregression_reduction
            ),
            "negative_contribution_weight": float(
                cfg.fiber_negative_contribution_weight
            ),
            "visual_hull_weight": float(cfg.fiber_visual_hull_weight),
            "visual_hull_target_geometry": bool(
                cfg.fiber_visual_hull_target_geometry
            ),
            "visual_hull_update_every": int(cfg.fiber_visual_hull_update_every),
            "visual_hull_min_views": int(cfg.fiber_visual_hull_min_views),
            "visual_hull_min_fraction": float(
                cfg.fiber_visual_hull_min_fraction
            ),
            "visual_hull_occlusion_aware": bool(
                cfg.fiber_visual_hull_occlusion_aware
            ),
            "visual_hull_occlusion_bin_px": int(
                cfg.fiber_visual_hull_occlusion_bin_px
            ),
            "visual_hull_occlusion_depth_scale": float(
                cfg.fiber_visual_hull_occlusion_depth_scale
            ),
            "orientation_distribution_weight": float(
                cfg.fiber_orientation_distribution_weight
            ),
            "orientation_distribution_radius": int(
                cfg.fiber_orientation_distribution_radius
            ),
            "strand_field_weight": float(cfg.fiber_strand_field_weight),
            "strand_deployability_weight": float(
                cfg.fiber_strand_deployability_weight
            ),
            "strand_min_deployment_gain": float(
                cfg.fiber_strand_min_deployment_gain
            ),
            "strand_min_deployed_length_scale": float(
                cfg.fiber_strand_min_deployed_length_scale
            ),
            "strand_coverage_weight": float(cfg.fiber_strand_coverage_weight),
            "strand_coverage_target": float(cfg.fiber_strand_coverage_target),
            "rgb_gradient_weight": float(cfg.fiber_rgb_gradient_weight),
            "deployment_render_weight": float(
                cfg.fiber_deployment_render_weight
            ),
            "structured_spill_weight": float(
                cfg.fiber_structured_spill_weight
            ),
            "mask_inside_coverage_weight": float(
                cfg.fiber_mask_inside_coverage_weight
            ),
            "mask_outside_spill_weight": float(
                cfg.fiber_mask_outside_spill_weight
            ),
            "mask_inside_alpha_target": float(
                cfg.fiber_mask_inside_alpha_target
            ),
            "mask_outside_margin_px": int(
                cfg.fiber_mask_outside_margin_px
            ),
            "structured_mask_inside_coverage_weight": float(
                cfg.fiber_structured_mask_inside_coverage_weight
            ),
            "structured_mask_outside_spill_weight": float(
                cfg.fiber_structured_mask_outside_spill_weight
            ),
            "structured_mask_inside_alpha_target": float(
                cfg.fiber_structured_mask_inside_alpha_target
            ),
            "coverage_seed_count": int(cfg.fiber_coverage_seed_count),
            "coverage_seed_samples": int(cfg.fiber_coverage_seed_samples),
            "coverage_seed_min_views": int(
                cfg.fiber_coverage_seed_min_views
            ),
            "coverage_seed_min_fraction": float(
                cfg.fiber_coverage_seed_min_fraction
            ),
            "coverage_seed_min_deficit": float(
                cfg.fiber_coverage_seed_min_deficit
            ),
            "coverage_seed_voxel_scale": float(
                cfg.fiber_coverage_seed_voxel_scale
            ),
            "coverage_seed_visibility_cull": bool(
                cfg.fiber_coverage_seed_visibility_cull
            ),
            "coverage_seed_visibility_bin_px": int(
                cfg.fiber_coverage_seed_visibility_bin_px
            ),
            "coverage_seed_visibility_depth_scale": float(
                cfg.fiber_coverage_seed_visibility_depth_scale
            ),
            "coverage_seed_structured_opacity": float(
                cfg.fiber_coverage_seed_structured_opacity
            ),
            "coverage_seed_geometry_gain": float(
                cfg.fiber_coverage_seed_geometry_gain
            ),
            "coverage_seed_shell_length_scale": float(
                cfg.fiber_coverage_seed_shell_length_scale
            ),
            "coverage_seed_strand_length_scale": float(
                cfg.fiber_coverage_seed_strand_length_scale
            ),
            "coverage_seed_orientation_init": bool(
                cfg.fiber_coverage_seed_orientation_init
            ),
            "coverage_seed_orientation_normal_bias": float(
                cfg.fiber_coverage_seed_orientation_normal_bias
            ),
            "coverage_seed_orientation_confidence_floor": float(
                cfg.fiber_coverage_seed_orientation_confidence_floor
            ),
            "coverage_seed_route_mass": list(
                cfg.fiber_coverage_seed_route_mass
            ),
            "topology_update_every": int(cfg.fiber_topology_update_every),
            "topology_start_step": int(cfg.fiber_topology_start_step),
            "topology_stop_step": int(cfg.fiber_topology_stop_step),
            "topology_prune_count": int(cfg.fiber_topology_prune_count),
            "topology_grow_count": int(cfg.fiber_topology_grow_count),
            "topology_densify_count": int(cfg.fiber_topology_densify_count),
            "topology_min_views": int(cfg.fiber_topology_min_views),
            "topology_prune_max_support": float(
                cfg.fiber_topology_prune_max_support
            ),
            "topology_prune_footprint_sigma": float(
                cfg.fiber_topology_prune_footprint_sigma
            ),
            "topology_grow_min_support": float(
                cfg.fiber_topology_grow_min_support
            ),
            "topology_grow_min_deficit": float(
                cfg.fiber_topology_grow_min_deficit
            ),
            "topology_boundary_radius": int(
                cfg.fiber_topology_boundary_radius
            ),
            "topology_detail_radius_scale": float(
                cfg.fiber_topology_detail_radius_scale
            ),
            "topology_max_residual_prune_fraction": float(
                cfg.fiber_topology_max_residual_prune_fraction
            ),
            "topology_initial_structured_off": bool(
                cfg.fiber_topology_initial_structured_off
            ),
            "shell_visual_hull_weight": float(
                cfg.fiber_shell_visual_hull_weight
            ),
            "shell_render_spill_weight": float(
                cfg.fiber_shell_render_spill_weight
            ),
        },
        "history": history,
        "metrics_jsonl": str(metrics_jsonl),
        "topology_jsonl": str(topology_jsonl),
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
    if field.expert_sh_delta_raw is not None:
        add_group(
            "route_sh",
            [field.expert_sh_delta_raw],
            cfg.fiber_expert_sh_lr_scale,
        )
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
        "surface_roots",
        [field.barycentric_offset_raw],
        cfg.fiber_root_barycentric_lr_scale,
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


def _structure_deployment_summary(
    field: UnifiedFiberField, temperature: float
) -> dict[str, float]:
    probabilities = field.route_probabilities(temperature).detach()
    gain = field.structured_delta_gain.detach()
    summary: dict[str, float] = {}
    for route_index, route_name in enumerate(ROUTE_NAMES[:2]):
        weight = probabilities[:, route_index]
        summary[route_name] = float(
            (weight * gain[:, route_index]).sum().cpu()
            / weight.sum().clamp_min(1e-8).cpu()
        )
    summary["structured_mean"] = float(
        (
            probabilities[:, :2] * gain
        ).sum().cpu()
        / probabilities[:, :2].sum().clamp_min(1e-8).cpu()
    )
    return summary


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
    # The cameras are ordered around the subject.  A tail split therefore validates
    # several neighbouring cameras and can accept a topology edit that fails on the
    # opposite side of the head.  Reserve evenly distributed leave-one-view-out
    # cameras instead.  Rounded linspace indices are unique because count < length.
    calibration_positions = np.rint(
        np.linspace(0, len(frame_indices) - 1, calibration_count)
    ).astype(np.int64)
    calibration_position_set = set(int(index) for index in calibration_positions)
    training = [
        frame_index
        for position, frame_index in enumerate(frame_indices)
        if position not in calibration_position_set
    ]
    calibration = [
        frame_index
        for position, frame_index in enumerate(frame_indices)
        if position in calibration_position_set
    ]
    return training, calibration


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
    migration_mass = cfg.fiber_teacher_semantic_migration_mass
    if migration_mass is not None:
        values = np.asarray(migration_mass, dtype=np.float64).reshape(-1)
        if values.shape != (len(ROUTE_NAMES),):
            raise ValueError(
                "fiber_teacher_semantic_migration_mass must contain "
                f"[{', '.join(ROUTE_NAMES)}] source fractions"
            )
        if (
            not np.isfinite(values).all()
            or np.any(values < 0.0)
            or float(values.sum()) <= 0.0
        ):
            raise ValueError(
                "fiber_teacher_semantic_migration_mass must be finite, "
                "non-negative, and sum to a positive value"
            )
    adaptive_domain = cfg.fiber_teacher_adaptive_migration_domain
    if migration_mass is not None and adaptive_domain is not None:
        raise ValueError(
            "Fixed semantic migration and adaptive residual-free migration "
            "are mutually exclusive"
        )
    if adaptive_domain is not None and str(adaptive_domain).lower() not in {
        "hair",
        "fur",
        "auto",
    }:
        raise ValueError(
            "fiber_teacher_adaptive_migration_domain must be 'hair', 'fur', "
            "or 'auto'"
        )
    adaptive_bias = float(cfg.fiber_teacher_adaptive_migration_bias)
    if not math.isfinite(adaptive_bias) or adaptive_bias < 0.0:
        raise ValueError(
            "fiber_teacher_adaptive_migration_bias must be finite and non-negative"
        )
    if float(cfg.fiber_teacher_semantic_migration_tolerance) < 0.0:
        raise ValueError(
            "fiber_teacher_semantic_migration_tolerance must be non-negative"
        )
    if float(cfg.fiber_structure_deployment_weight) < 0.0:
        raise ValueError("fiber_structure_deployment_weight must be non-negative")
    structure_gain = float(cfg.fiber_structure_min_deployment_gain)
    if not 0.0 <= structure_gain <= 1.0:
        raise ValueError("fiber_structure_min_deployment_gain must be in [0, 1]")
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
    if float(cfg.fiber_shell_visual_hull_weight) < 0.0:
        raise ValueError("fiber_shell_visual_hull_weight must be non-negative")
    if float(cfg.fiber_shell_render_spill_weight) < 0.0:
        raise ValueError("fiber_shell_render_spill_weight must be non-negative")
    if float(cfg.fiber_rgb_gradient_weight) < 0.0:
        raise ValueError("fiber_rgb_gradient_weight must be non-negative")
    if float(cfg.fiber_structured_spill_weight) < 0.0:
        raise ValueError("fiber_structured_spill_weight must be non-negative")
    for name in (
        "fiber_mask_inside_coverage_weight",
        "fiber_mask_outside_spill_weight",
        "fiber_structured_mask_inside_coverage_weight",
        "fiber_structured_mask_outside_spill_weight",
        "fiber_max_hole_weight",
    ):
        if float(getattr(cfg, name)) < 0.0:
            raise ValueError(f"{name} must be non-negative")
    for name in (
        "fiber_mask_inside_alpha_target",
        "fiber_structured_mask_inside_alpha_target",
    ):
        value = float(getattr(cfg, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    if int(cfg.fiber_mask_outside_margin_px) < 0:
        raise ValueError("fiber_mask_outside_margin_px must be non-negative")
    if int(cfg.fiber_max_hole_kernel) <= 0:
        raise ValueError("fiber_max_hole_kernel must be positive")
    max_hole_topk = float(cfg.fiber_max_hole_topk_fraction)
    if not 0.0 < max_hole_topk <= 1.0:
        raise ValueError("fiber_max_hole_topk_fraction must be in (0, 1]")
    if int(cfg.fiber_coverage_seed_count) < 0:
        raise ValueError("fiber_coverage_seed_count must be non-negative")
    if int(cfg.fiber_coverage_seed_samples) <= 0:
        raise ValueError("fiber_coverage_seed_samples must be positive")
    if int(cfg.fiber_coverage_seed_min_views) <= 0:
        raise ValueError("fiber_coverage_seed_min_views must be positive")
    for name in (
        "fiber_coverage_seed_min_fraction",
        "fiber_coverage_seed_min_deficit",
        "fiber_coverage_seed_structured_opacity",
        "fiber_coverage_seed_geometry_gain",
    ):
        value = float(getattr(cfg, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    for name in (
        "fiber_coverage_seed_voxel_scale",
        "fiber_coverage_seed_shell_length_scale",
        "fiber_coverage_seed_strand_length_scale",
    ):
        if float(getattr(cfg, name)) <= 0.0:
            raise ValueError(f"{name} must be positive")
    if int(cfg.fiber_coverage_seed_visibility_bin_px) <= 0:
        raise ValueError("fiber_coverage_seed_visibility_bin_px must be positive")
    if float(cfg.fiber_coverage_seed_visibility_depth_scale) < 0.0:
        raise ValueError(
            "fiber_coverage_seed_visibility_depth_scale must be non-negative"
        )
    orientation_normal_bias = float(
        cfg.fiber_coverage_seed_orientation_normal_bias
    )
    if not 0.0 <= orientation_normal_bias <= 1.0:
        raise ValueError(
            "fiber_coverage_seed_orientation_normal_bias must be in [0, 1]"
        )
    orientation_confidence_floor = float(
        cfg.fiber_coverage_seed_orientation_confidence_floor
    )
    if not 0.0 <= orientation_confidence_floor <= 1.0:
        raise ValueError(
            "fiber_coverage_seed_orientation_confidence_floor must be in [0, 1]"
        )
    if (
        bool(cfg.fiber_coverage_seed_orientation_init)
        and cfg.fiber_orientation_dir is None
    ):
        raise ValueError(
            "fiber_coverage_seed_orientation_init requires fiber_orientation_dir"
        )
    coverage_route_mass = np.asarray(
        cfg.fiber_coverage_seed_route_mass, dtype=np.float64
    ).reshape(-1)
    if coverage_route_mass.shape != (len(ROUTE_NAMES),):
        raise ValueError(
            "fiber_coverage_seed_route_mass must contain "
            f"[{', '.join(ROUTE_NAMES)}] mass"
        )
    if (
        not np.isfinite(coverage_route_mass).all()
        or np.any(coverage_route_mass < 0.0)
        or float(coverage_route_mass.sum()) <= 0.0
        or float(coverage_route_mass[ROUTE_NAMES.index("residual")]) <= 0.0
    ):
        raise ValueError(
            "fiber_coverage_seed_route_mass must be finite, non-negative, "
            "sum to a positive value, and retain positive residual mass"
        )
    for name in (
        "fiber_topology_update_every",
        "fiber_topology_start_step",
        "fiber_topology_stop_step",
        "fiber_topology_prune_count",
        "fiber_topology_grow_count",
        "fiber_topology_densify_count",
    ):
        if int(getattr(cfg, name)) < 0:
            raise ValueError(f"{name} must be non-negative")
    if int(cfg.fiber_topology_min_views) <= 0:
        raise ValueError("fiber_topology_min_views must be positive")
    for name in (
        "fiber_topology_prune_max_support",
        "fiber_topology_grow_min_support",
        "fiber_topology_grow_min_deficit",
        "fiber_topology_max_residual_prune_fraction",
    ):
        value = float(getattr(cfg, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    footprint_sigma = float(cfg.fiber_topology_prune_footprint_sigma)
    if not math.isfinite(footprint_sigma) or footprint_sigma <= 0.0:
        raise ValueError("fiber_topology_prune_footprint_sigma must be positive")
    if int(cfg.fiber_topology_boundary_radius) <= 0:
        raise ValueError("fiber_topology_boundary_radius must be positive")
    detail_scale = float(cfg.fiber_topology_detail_radius_scale)
    if not 0.0 < detail_scale <= 1.0:
        raise ValueError("fiber_topology_detail_radius_scale must be in (0, 1]")
    if float(cfg.fiber_topology_validation_margin) < 0.0:
        raise ValueError("fiber_topology_validation_margin must be non-negative")
    if (
        bool(cfg.fiber_topology_validate_events)
        and int(cfg.fiber_calibration_frames) < 2
    ):
        raise ValueError(
            "fiber_topology_validate_events requires at least two calibration views"
        )
    if float(cfg.fiber_deployment_render_weight) < 0.0:
        raise ValueError("fiber_deployment_render_weight must be non-negative")
    for name in (
        "fiber_strand_field_weight",
        "fiber_strand_deployability_weight",
        "fiber_strand_min_deployed_length_scale",
        "fiber_strand_coverage_weight",
        "fiber_strand_coverage_target",
    ):
        if float(getattr(cfg, name)) < 0.0:
            raise ValueError(f"{name} must be non-negative")
    deployment_gain = float(cfg.fiber_strand_min_deployment_gain)
    if not 0.0 <= deployment_gain <= 1.0:
        raise ValueError("fiber_strand_min_deployment_gain must be in [0, 1]")
    detach_start = float(cfg.fiber_structure_detach_start_fraction)
    detach_end = float(cfg.fiber_structure_detach_end_fraction)
    detach_gain = float(cfg.fiber_structure_detach_final_gain)
    if not 0.0 <= detach_start <= 1.0:
        raise ValueError("fiber_structure_detach_start_fraction must be in [0, 1]")
    if not detach_start <= detach_end <= 1.0:
        raise ValueError(
            "fiber_structure_detach_end_fraction must be in [start, 1]"
        )
    if not 0.0 <= detach_gain <= 1.0:
        raise ValueError("fiber_structure_detach_final_gain must be in [0, 1]")
    if not 0.0 <= float(cfg.fiber_structure_detach_min_support_fraction) <= 1.0:
        raise ValueError(
            "fiber_structure_detach_min_support_fraction must be in [0, 1]"
        )
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
    if int(cfg.fiber_teacher_nonregression_views_per_step) <= 0:
        raise ValueError(
            "fiber_teacher_nonregression_views_per_step must be positive"
        )
    if cfg.fiber_teacher_nonregression_reduction not in {"mean", "max"}:
        raise ValueError(
            "fiber_teacher_nonregression_reduction must be 'mean' or 'max'"
        )
    if float(cfg.fiber_teacher_nonregression_weight) < 0.0:
        raise ValueError("fiber_teacher_nonregression_weight must be non-negative")
    teacher_transfer = float(cfg.fiber_teacher_opacity_transfer)
    if not 0.0 <= teacher_transfer <= 1.0:
        raise ValueError("fiber_teacher_opacity_transfer must be in [0, 1]")
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
    if float(cfg.fiber_orientation_distribution_weight) < 0.0:
        raise ValueError(
            "fiber_orientation_distribution_weight must be non-negative"
        )
    if int(cfg.fiber_orientation_distribution_radius) < 0:
        raise ValueError(
            "fiber_orientation_distribution_radius must be non-negative"
        )
    if int(cfg.fiber_visual_hull_occlusion_bin_px) <= 0:
        raise ValueError(
            "fiber_visual_hull_occlusion_bin_px must be positive"
        )
    if float(cfg.fiber_visual_hull_occlusion_depth_scale) < 0.0:
        raise ValueError(
            "fiber_visual_hull_occlusion_depth_scale must be non-negative"
        )
    for name in (
        "fiber_root_barycentric_lr_scale",
        "fiber_root_barycentric_weight",
        "fiber_expert_sh_lr_scale",
        "fiber_expert_sh_weight",
        "fiber_expert_sh_max_delta",
    ):
        if float(getattr(cfg, name)) < 0.0:
            raise ValueError(f"{name} must be non-negative")
    if not 0 <= int(cfg.fiber_expert_sh_degree) <= 3:
        raise ValueError("fiber_expert_sh_degree must be in [0, 3]")
    if not 0.0 <= float(cfg.fiber_root_barycentric_max_delta) <= 1.0:
        raise ValueError("fiber_root_barycentric_max_delta must be in [0, 1]")
    if int(cfg.fiber_scalp_occupancy_erosion_px) < 0:
        raise ValueError("fiber_scalp_occupancy_erosion_px must be non-negative")
    if int(cfg.fiber_scalp_occupancy_min_views) <= 0:
        raise ValueError("fiber_scalp_occupancy_min_views must be positive")
    if int(cfg.fiber_scalp_atlas_min_roots_per_face) < 0:
        raise ValueError("fiber_scalp_atlas_min_roots_per_face must be non-negative")
    if float(cfg.fiber_strand_outward_anchor_threshold) < 0.0:
        raise ValueError(
            "fiber_strand_outward_anchor_threshold must be non-negative"
        )
    if int(cfg.fiber_strand_sign_sync_steps) < 0:
        raise ValueError("fiber_strand_sign_sync_steps must be non-negative")
    if not 0.0 <= float(cfg.fiber_topology_birth_strand_mass) <= 1.0:
        raise ValueError("fiber_topology_birth_strand_mass must be in [0, 1]")
    if not 0.0 <= float(cfg.fiber_topology_birth_initial_delta) <= 1.0:
        raise ValueError("fiber_topology_birth_initial_delta must be in [0, 1]")
    if int(cfg.fiber_topology_deficit_min_views) <= 0:
        raise ValueError("fiber_topology_deficit_min_views must be positive")
    if int(cfg.fiber_topology_ray_steps) < 2:
        raise ValueError("fiber_topology_ray_steps must be at least two")
    ray_min = float(cfg.fiber_topology_ray_min_length_scale)
    ray_max = float(cfg.fiber_topology_ray_max_length_scale)
    if not math.isfinite(ray_min) or ray_min <= 0.0:
        raise ValueError("fiber_topology_ray_min_length_scale must be positive")
    if not math.isfinite(ray_max) or ray_max <= ray_min:
        raise ValueError(
            "fiber_topology_ray_max_length_scale must exceed the minimum"
        )
    ray_neighbor_blend = float(cfg.fiber_topology_ray_neighbor_blend)
    if not 0.0 <= ray_neighbor_blend <= 1.0:
        raise ValueError("fiber_topology_ray_neighbor_blend must be in [0, 1]")
    if float(cfg.fiber_topology_bend_neighbor_scale) < 0.0:
        raise ValueError("fiber_topology_bend_neighbor_scale must be non-negative")
    if int(cfg.fiber_topology_local_warmup_steps) < 0:
        raise ValueError("fiber_topology_local_warmup_steps must be non-negative")
    if float(cfg.fiber_topology_local_warmup_lr) < 0.0:
        raise ValueError("fiber_topology_local_warmup_lr must be non-negative")
    if int(cfg.fiber_topology_local_warmup_views) <= 0:
        raise ValueError("fiber_topology_local_warmup_views must be positive")
    for name in (
        "fiber_topology_min_mean_improvement",
        "fiber_topology_min_birth_deployment",
        "fiber_topology_min_birth_effective_mass",
    ):
        if float(getattr(cfg, name)) < 0.0:
            raise ValueError(f"{name} must be non-negative")
    if float(cfg.fiber_topology_min_birth_deployment) > 1.0:
        raise ValueError("fiber_topology_min_birth_deployment must be at most one")
    if float(cfg.fiber_topology_min_birth_effective_mass) > 1.0:
        raise ValueError("fiber_topology_min_birth_effective_mass must be at most one")
    if int(cfg.fiber_strand_sign_projection_every) < 0:
        raise ValueError("fiber_strand_sign_projection_every must be non-negative")
    if int(cfg.fiber_strand_sign_projection_steps) < 0:
        raise ValueError("fiber_strand_sign_projection_steps must be non-negative")
    for name in (
        "fiber_scalp_occupancy_min_fraction",
        "fiber_scalp_initial_strand_fraction",
    ):
        value = float(getattr(cfg, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
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


def _residual_footprint_probe_points(
    center: torch.Tensor,
    tangent: torch.Tensor,
    bitangent: torch.Tensor,
    normal: torch.Tensor,
    scaling: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Return conservative probes for residual Gaussian mask support.

    Center-only pruning opens holes when the center is outside a thin hair
    mask but the Gaussian footprint still covers valid foreground.  Probe the
    center and both signs of the local surface-frame axes at ``sigma`` times
    the largest Gaussian scale.  A residual is prunable only when this whole
    footprint lacks multi-view support.
    """

    if center.ndim != 2 or center.shape[-1] != 3:
        raise ValueError("center must have shape [N, 3]")
    for name, axis in (
        ("tangent", tangent),
        ("bitangent", bitangent),
        ("normal", normal),
    ):
        if axis.shape != center.shape:
            raise ValueError(f"{name} must match center shape")
    if scaling.ndim != 2 or scaling.shape[0] != center.shape[0]:
        raise ValueError("scaling must have shape [N, C]")
    if not math.isfinite(float(sigma)) or float(sigma) <= 0.0:
        raise ValueError("sigma must be positive")
    extent = float(sigma) * scaling.detach().amax(dim=-1, keepdim=True)
    offsets = torch.stack(
        [tangent, -tangent, bitangent, -bitangent, normal, -normal], dim=1
    )
    return torch.cat(
        [center[:, None, :], center[:, None, :] + extent[:, None, :] * offsets],
        dim=1,
    )


def _front_surface_visibility(
    points: torch.Tensor,
    camera,
    *,
    bin_px: int,
    depth_tolerance: float,
) -> torch.Tensor:
    """Approximate head-surface visibility with a point z-buffer.

    Dense head vertices are binned into small image blocks and only points
    near the minimum camera-space depth survive.  This prevents back-side
    face/neck vertices from receiving scalp occupancy merely because their 2D
    projection overlaps a front-side hair silhouette.
    """

    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError("Visibility points must have shape (N, 3)")
    block = int(bin_px)
    if block <= 0:
        raise ValueError("bin_px must be positive")
    if float(depth_tolerance) < 0.0:
        raise ValueError("depth_tolerance must be non-negative")
    dtype, device = points.dtype, points.device
    world_to_camera = torch.as_tensor(
        camera.world_to_camera, dtype=dtype, device=device
    )
    camera_xyz = points @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    z = camera_xyz[:, 2]
    safe_z = z.clamp_min(1e-6)
    x = float(camera.fx) * camera_xyz[:, 0] / safe_z + float(camera.cx)
    y_sign = 1.0 if camera.image_y_down else -1.0
    y = float(camera.cy) + y_sign * float(camera.fy) * camera_xyz[:, 1] / safe_z
    width, height = int(camera.width), int(camera.height)
    valid = (
        (z > 1e-5)
        & (x >= 0.0)
        & (x <= max(width - 1, 0))
        & (y >= 0.0)
        & (y <= max(height - 1, 0))
    )
    bins_x = int(math.ceil(width / block))
    bins_y = int(math.ceil(height / block))
    bin_x = torch.floor(x / block).to(torch.long).clamp(0, bins_x - 1)
    bin_y = torch.floor(y / block).to(torch.long).clamp(0, bins_y - 1)
    linear = bin_y * bins_x + bin_x
    depth = torch.full(
        (bins_x * bins_y,), float("inf"), device=device, dtype=dtype
    )
    if torch.any(valid):
        depth.scatter_reduce_(
            0,
            linear[valid],
            z[valid],
            reduce="amin",
            include_self=True,
        )
    closest = depth[linear]
    return valid & (z <= closest + float(depth_tolerance))


def _front_visible_sample_gate(
    points: torch.Tensor,
    occluders: torch.Tensor,
    camera,
    *,
    bin_px: int,
    depth_tolerance: float,
) -> torch.Tensor:
    """Return front-visible samples using shared geometry as z-buffer occluders."""

    if points.ndim < 2 or points.shape[-1] != 3:
        raise ValueError("points must end in XYZ")
    if occluders.ndim != 2 or occluders.shape[-1] != 3:
        raise ValueError("occluders must have shape [N, 3]")
    original_shape = points.shape[:-1]
    flat = points.detach().reshape(-1, 3)
    with torch.no_grad():
        combined = torch.cat([occluders.detach(), flat], dim=0)
        visible = _front_surface_visibility(
            combined,
            camera,
            bin_px=int(bin_px),
            depth_tolerance=float(depth_tolerance),
        )
    return visible[-flat.shape[0] :].reshape(original_shape)


def _positive_parameter_raw(
    value: torch.Tensor, *, eps: float
) -> torch.Tensor:
    """Invert ``softplus(raw) + eps`` without producing infinities."""

    positive = (value - float(eps)).clamp_min(1e-8)
    return torch.log(torch.expm1(positive).clamp_min(1e-12))


def _topk_masked(
    score: torch.Tensor, eligible: torch.Tensor, count: int
) -> torch.Tensor:
    """Deterministically select the highest-scoring eligible source ids."""

    indices = torch.nonzero(eligible, as_tuple=False).flatten()
    limit = min(max(int(count), 0), int(indices.numel()))
    if limit == 0:
        return torch.empty((0,), dtype=torch.long, device=score.device)
    # Stable tie-breaking by source id keeps topology runs reproducible.
    tie = indices.to(score.dtype) * (
        torch.finfo(score.dtype).eps / float(max(score.numel(), 1))
    )
    order = torch.argsort(score[indices] - tie, descending=True)
    return indices[order[:limit]]


def _local_cluster_masked(
    score: torch.Tensor,
    eligible: torch.Tensor,
    count: int,
    neighbor_index: torch.Tensor,
) -> torch.Tensor:
    """Select one connected high-deficit cluster around the best seed."""

    candidates = torch.nonzero(eligible, as_tuple=False).flatten()
    limit = min(max(int(count), 0), int(candidates.numel()))
    if limit == 0:
        return torch.empty((0,), dtype=torch.long, device=score.device)
    seed = candidates[torch.argmax(score[candidates])]
    selected = torch.zeros_like(eligible)
    frontier = seed.reshape(1)
    selected[seed] = True
    while int(selected.sum()) < limit and frontier.numel() > 0:
        neighbors = neighbor_index[frontier].reshape(-1)
        valid = eligible[neighbors] & (~selected[neighbors])
        frontier = torch.unique(neighbors[valid])
        if frontier.numel() == 0:
            break
        remaining = limit - int(selected.sum())
        order = torch.argsort(score[frontier], descending=True)
        frontier = frontier[order[:remaining]]
        selected[frontier] = True
    if int(selected.sum()) < limit:
        remaining_mask = eligible & (~selected)
        fallback = _topk_masked(
            score, remaining_mask, limit - int(selected.sum())
        )
        selected[fallback] = True
    return torch.nonzero(selected, as_tuple=False).flatten()


def _set_effective_route_mass(
    field: UnifiedFiberField,
    indices: torch.Tensor,
    desired_mass: torch.Tensor,
    *,
    temperature: float,
    residual_trust: float = 0.02,
) -> None:
    """Initialize selected rows to a requested effective route distribution."""

    if indices.numel() == 0:
        return
    desired = desired_mass.to(
        device=field.route_logits.device, dtype=field.route_logits.dtype
    ).reshape(-1)
    if desired.numel() != len(ROUTE_NAMES) or torch.any(desired < 0.0):
        raise ValueError("desired route mass must be non-negative shell/strand/residual")
    desired = desired / desired.sum().clamp_min(1e-8)
    residual_index = ROUTE_NAMES.index("residual")
    trust = min(max(float(residual_trust), 1e-4), 0.95)
    # The effective residual probability contains an explicit trust floor.
    # Keep the requested residual mass just above it before inverting the
    # mixture; topology gates later remove inactive routes exactly.
    effective = desired.clone()
    effective[residual_index] = max(float(effective[residual_index]), trust + 1e-4)
    effective = effective / effective.sum()
    base = effective.clone()
    base[:residual_index] /= 1.0 - trust
    base[residual_index] = (
        effective[residual_index] - trust
    ) / (1.0 - trust)
    base = base.clamp_min(1e-6)
    base = base / base.sum()
    field.residual_trust_logits[indices, 0] = torch.logit(
        torch.tensor(
            trust,
            device=field.residual_trust_logits.device,
            dtype=field.residual_trust_logits.dtype,
        )
    )
    field.route_logits[indices] = float(temperature) * torch.log(base)
    field.initial_route_probabilities[indices] = effective


def _clear_adam_rows(
    optimizer: torch.optim.Optimizer,
    parameters: list[torch.nn.Parameter],
    indices: torch.Tensor,
) -> None:
    """Remove stale Adam momentum from rows changed by a topology event."""

    if indices.numel() == 0:
        return
    for parameter in parameters:
        state = optimizer.state.get(parameter)
        if not state:
            continue
        for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            value = state.get(name)
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                value[indices] = 0.0


def _apply_adaptive_topology_scores(
    field: UnifiedFiberField,
    optimizer: torch.optim.Optimizer,
    cfg: PipelineConfig,
    *,
    visible_views: torch.Tensor,
    residual_support: torch.Tensor,
    structured_support: torch.Tensor,
    deficit_score: torch.Tensor,
    boundary_score: torch.Tensor,
    step: int,
    deficit_views: torch.Tensor | None = None,
    proposed_strand_length: torch.Tensor | None = None,
    proposed_direction_local: torch.Tensor | None = None,
    proposed_bend_local: torch.Tensor | None = None,
    clear_optimizer_state: bool = True,
) -> dict[str, object]:
    """Apply explicit prune/grow/densify decisions from multi-view scores.

    The tensors are a capacity pool; changing route_active_gate changes the
    actual non-zero Gaussian set seen by the rasterizer.  This is equivalent
    to prune/clone ADC without resizing Adam parameters in the middle of a
    training run.
    """

    device = field.route_logits.device
    for name, value in (
        ("visible_views", visible_views),
        ("residual_support", residual_support),
        ("structured_support", structured_support),
        ("deficit_score", deficit_score),
        ("boundary_score", boundary_score),
    ):
        if tuple(value.shape) != (field.point_count,):
            raise ValueError(f"{name} must contain one value per source")
    if deficit_views is None:
        deficit_views = visible_views
    if tuple(deficit_views.shape) != (field.point_count,):
        raise ValueError("deficit_views must contain one value per source")
    for name, value, shape in (
        ("proposed_strand_length", proposed_strand_length, (field.point_count,)),
        ("proposed_direction_local", proposed_direction_local, (field.point_count, 3)),
        ("proposed_bend_local", proposed_bend_local, (field.point_count, 2)),
    ):
        if value is not None and tuple(value.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}")
    active = field.route_active_gate
    shell_index = ROUTE_NAMES.index("shell")
    strand_index = ROUTE_NAMES.index("strand")
    residual_index = ROUTE_NAMES.index("residual")
    min_views = float(cfg.fiber_topology_min_views)
    structured_active = (active[:, :2] > 0.5).any(dim=1)
    residual_active = active[:, residual_index] > 0.5
    foreground_eligible = (
        field.source_foreground > 0.5
        if bool(getattr(field, "structured_foreground_only", False))
        else torch.ones(field.point_count, dtype=torch.bool, device=device)
    )

    max_pruned = int(
        math.floor(
            field.point_count
            * float(cfg.fiber_topology_max_residual_prune_fraction)
        )
    )
    # A semantic migration deliberately disables residual on every source
    # owned by shell/strand.  Those sources are not pruned; only rows with no
    # active family consume the outlier-pruning budget.
    already_pruned = int((~(active > 0.5).any(dim=1)).sum().detach().cpu())
    prune_budget = min(
        int(cfg.fiber_topology_prune_count), max(max_pruned - already_pruned, 0)
    )
    prune_eligible = (
        foreground_eligible
        &
        residual_active
        & (~structured_active)
        & (visible_views >= min_views)
        & (residual_support <= float(cfg.fiber_topology_prune_max_support))
    )
    prune_score = (
        (1.0 - residual_support.clamp(0.0, 1.0))
        * field.opacity.detach()
    )
    pruned = _topk_masked(prune_score, prune_eligible, prune_budget)

    grow_eligible = (
        foreground_eligible
        & (active[:, strand_index] <= 0.5)
        & (
            (field.strand_root_occupancy > 0.5)
            if bool(cfg.fiber_scalp_occupancy_enabled)
            else torch.ones(field.point_count, dtype=torch.bool, device=device)
        )
        & (visible_views >= min_views)
        & (deficit_views >= float(cfg.fiber_topology_deficit_min_views))
        & (structured_support >= float(cfg.fiber_topology_grow_min_support))
        & (deficit_score >= float(cfg.fiber_topology_grow_min_deficit))
    )
    grown = _local_cluster_masked(
        deficit_score * structured_support,
        grow_eligible,
        int(cfg.fiber_topology_grow_count),
        field.route_neighbor_index,
    )
    grow_mask = torch.zeros(field.point_count, dtype=torch.bool, device=device)
    grow_mask[grown] = True

    detail_eligible = (
        foreground_eligible
        & (active[:, shell_index] <= 0.5)
        & (~grow_mask)
        & (visible_views >= min_views)
        & (structured_support >= float(cfg.fiber_topology_grow_min_support))
        & (boundary_score > 0.0)
    )
    densified = _topk_masked(
        boundary_score * structured_support,
        detail_eligible,
        int(cfg.fiber_topology_densify_count),
    )

    with torch.no_grad():
        active[pruned, residual_index] = 0.0
        if grown.numel() > 0:
            incremental_birth = bool(cfg.fiber_topology_incremental_birth)
            if incremental_birth:
                # Keep the existing shell and admit an almost-zero-mass
                # strand at the atlas location.  Optical-thickness splitting
                # makes this nearly render-equivalent; subsequent gradient
                # steps decide whether the strand should unfold and gain mass.
                active[grown, strand_index] = 1.0
            elif (
                cfg.fiber_teacher_semantic_migration_mass is not None
                or bool(cfg.fiber_scalp_occupancy_enabled)
            ):
                # A validated grow event transfers ownership instead of
                # creating a residual escape hatch or double-owned source.
                active[grown] = 0.0
                active[grown, strand_index] = 1.0
            else:
                active[grown, shell_index] = 1.0
                active[grown, strand_index] = 1.0
            if incremental_birth:
                field.structured_delta_raw[grown, strand_index] = float(
                    cfg.fiber_topology_birth_initial_delta
                )
                field.structured_opacity_raw[grown, strand_index] = 0.0
            else:
                field.structured_delta_raw[grown] = 1.0
                field.structured_opacity_raw[grown] = max(
                    float(cfg.fiber_coverage_seed_structured_opacity), 0.25
                )
            _set_effective_route_mass(
                field,
                grown,
                torch.as_tensor(
                    (
                        [
                            1.0 - float(cfg.fiber_topology_birth_strand_mass),
                            float(cfg.fiber_topology_birth_strand_mass),
                            0.0,
                        ]
                        if incremental_birth
                        else
                        [0.0, 1.0, 0.0]
                        if (
                            cfg.fiber_teacher_semantic_migration_mass is not None
                            or bool(cfg.fiber_scalp_occupancy_enabled)
                        )
                        else cfg.fiber_coverage_seed_route_mass
                    ),
                    device=device,
                    dtype=field.route_logits.dtype,
                ),
                temperature=float(cfg.fiber_final_temperature),
            )
            desired_carrier = torch.tensor(
                [0.05, 0.20, 0.75],
                device=device,
                dtype=field.carrier_logits.dtype,
            )
            field.carrier_logits[grown] = float(cfg.fiber_final_temperature) * torch.log(
                desired_carrier
            )
            field.carrier_root_tip_raw[grown] = 0.0
            if proposed_strand_length is not None:
                field.strand_length_raw[grown, 0] = _positive_parameter_raw(
                    proposed_strand_length[grown], eps=field.positive_eps
                )
            if proposed_direction_local is not None:
                field.direction_local_raw[grown] = F.normalize(
                    proposed_direction_local[grown], dim=-1, eps=1e-8
                )
            if proposed_bend_local is not None:
                field.bend_local[grown] = proposed_bend_local[grown]
        if densified.numel() > 0:
            if cfg.fiber_teacher_semantic_migration_mass is not None:
                active[densified] = 0.0
            active[densified, shell_index] = 1.0
            field.structured_delta_raw[densified, shell_index] = 1.0
            field.structured_opacity_raw[densified, shell_index] = max(
                float(cfg.fiber_coverage_seed_structured_opacity), 0.25
            )
            _set_effective_route_mass(
                field,
                densified,
                torch.tensor(
                    (
                        [1.0, 0.0, 0.0]
                        if cfg.fiber_teacher_semantic_migration_mass is not None
                        else [0.78, 0.12, 0.10]
                    ),
                    device=device,
                    dtype=field.route_logits.dtype,
                ),
                temperature=float(cfg.fiber_final_temperature),
            )
            sharpened_radius = (
                field.radius[densified]
                * float(cfg.fiber_topology_detail_radius_scale)
            )
            field.radius_raw[densified, 0] = _positive_parameter_raw(
                sharpened_radius, eps=field.positive_eps
            )
            desired_carrier = torch.tensor(
                [0.05, 0.80, 0.15],
                device=device,
                dtype=field.carrier_logits.dtype,
            )
            field.carrier_logits[densified] = float(cfg.fiber_final_temperature) * torch.log(
                desired_carrier
            )

    changed = torch.unique(torch.cat([pruned, grown, densified]))
    if clear_optimizer_state:
        _clear_adam_rows(
            optimizer,
            [
                field.route_logits,
                field.residual_trust_logits,
                field.structured_delta_raw,
                field.structured_opacity_raw,
                field.radius_raw,
                field.carrier_logits,
                field.carrier_root_tip_raw,
                field.strand_length_raw,
                field.direction_local_raw,
                field.bend_local,
                field.bend_cubic_local,
            ],
            changed,
        )
    topology = field.active_topology_summary(
        shell_samples=int(cfg.fiber_shell_samples),
        strand_samples=int(cfg.fiber_strand_samples),
    )

    def selected_stats(values: torch.Tensor, indices: torch.Tensor) -> dict[str, float]:
        if indices.numel() == 0:
            return {"min": 0.0, "mean": 0.0, "max": 0.0}
        selected = values[indices]
        return {
            "min": float(selected.min().cpu()),
            "mean": float(selected.mean().cpu()),
            "max": float(selected.max().cpu()),
        }

    report: dict[str, object] = {
        "step": int(step),
        "pruned_count": int(pruned.numel()),
        "grown_count": int(grown.numel()),
        "densified_count": int(densified.numel()),
        "pruned_support": selected_stats(residual_support, pruned),
        "grown_deficit": selected_stats(deficit_score, grown),
        "grown_deficit_views": selected_stats(deficit_views, grown),
        "grown_ray_length": selected_stats(
            (
                proposed_strand_length
                if proposed_strand_length is not None
                else field.strand_length.detach()
            ),
            grown,
        ),
        "densified_boundary": selected_stats(boundary_score, densified),
        "topology": topology,
    }
    if not clear_optimizer_state:
        report["_changed_indices"] = changed
    return report


_TOPOLOGY_MUTABLE_FIELD_NAMES = (
    "route_active_gate",
    "route_logits",
    "residual_trust_logits",
    "structured_delta_raw",
    "structured_opacity_raw",
    "radius_raw",
    "carrier_logits",
    "carrier_root_tip_raw",
    "initial_route_probabilities",
    "strand_length_raw",
    "direction_local_raw",
    "bend_local",
    "bend_cubic_local",
)


def _topology_state_snapshot(field: UnifiedFiberField) -> dict[str, torch.Tensor]:
    return {
        name: getattr(field, name).detach().clone()
        for name in _TOPOLOGY_MUTABLE_FIELD_NAMES
    }


def _restore_topology_state(
    field: UnifiedFiberField, snapshot: dict[str, torch.Tensor]
) -> None:
    with torch.no_grad():
        for name, value in snapshot.items():
            getattr(field, name).copy_(value)


def _topology_event_is_accepted(
    before: list[float],
    after: list[float],
    margin: float,
    *,
    allow_neutral: bool = False,
    min_mean_improvement: float = 0.0,
) -> tuple[bool, list[float]]:
    """Require per-view non-regression, not merely a better mean loss."""

    if len(before) != len(after) or len(before) < 2:
        raise ValueError("Topology validation requires at least two aligned views")
    deltas = [float(post - pre) for pre, post in zip(before, after, strict=True)]
    mean_limit = (
        0.25 * float(margin)
        if allow_neutral and float(min_mean_improvement) <= 0.0
        else -max(float(min_mean_improvement), 0.0)
    )
    accepted = (
        max(deltas) <= float(margin)
        and float(np.mean(deltas)) <= mean_limit
    )
    return accepted, deltas


def _topology_validation_losses(
    field: UnifiedFiberField,
    surface_vertices_per_view: list[torch.Tensor],
    surface_faces: torch.Tensor,
    cameras: list,
    targets: list[dict[str, torch.Tensor]],
    cfg: PipelineConfig,
    renderer_name: str,
) -> list[float]:
    if len(surface_vertices_per_view) < 2 or not (
        len(surface_vertices_per_view) == len(cameras) == len(targets)
    ):
        raise ValueError("Topology validation views must contain at least two items")
    losses: list[float] = []
    with torch.no_grad():
        for vertices, camera, target in zip(
            surface_vertices_per_view, cameras, targets, strict=True
        ):
            primitives = field.primitives(
                vertices.to(device=field.route_logits.device),
                surface_faces,
                shell_samples=int(cfg.fiber_shell_samples),
                strand_samples=int(cfg.fiber_strand_samples),
                temperature=float(cfg.fiber_final_temperature),
                hard_route=False,
                route_blend=1.0,
                geometry_blend=1.0,
                route_hardening=0.0,
                hard_route_policy=cfg.fiber_hard_route_policy,
                fin_aspect_ratio=float(cfg.fiber_fin_aspect_ratio),
                additive_teacher=bool(cfg.fiber_additive_teacher_mode),
                teacher_opacity_transfer=float(cfg.fiber_teacher_opacity_transfer),
            )
            prediction = _render(primitives, camera, cfg, renderer_name)
            loss, _parts = differentiable_render_loss(
                prediction,
                target["rgb"],
                target["mask"],
                cfg.color_loss_weight,
                cfg.mask_loss_weight,
                cfg.mask_boundary_weight,
                cfg.mask_boundary_radius,
                cfg.mask_balance_weight,
            )
            if float(cfg.fiber_max_hole_weight) > 0.0:
                loss = loss + float(cfg.fiber_max_hole_weight) * (
                    _maximum_hole_soft_loss(
                        prediction["mask"],
                        target["mask"],
                        kernel_size=int(cfg.fiber_max_hole_kernel),
                        topk_fraction=float(cfg.fiber_max_hole_topk_fraction),
                    )
                )
            losses.append(float(loss.detach().cpu()))
    return losses


def _topology_local_warmup(
    field: UnifiedFiberField,
    surface_vertices_per_view: list[torch.Tensor],
    surface_faces: torch.Tensor,
    cameras: list,
    targets: list[dict[str, torch.Tensor]],
    cfg: PipelineConfig,
    renderer_name: str,
    changed: torch.Tensor,
) -> dict[str, object]:
    """Optimize only a newly born spatial cluster before held-out admission.

    The birth proposal is fitted on a small set of training views.  Parameter
    rows outside the cluster and its one-ring atlas neighborhood are never
    updated, so this cannot silently turn into another global fitting stage.
    The subsequent per-view calibration remains the accept/reject authority.
    """

    steps = int(cfg.fiber_topology_local_warmup_steps)
    learning_rate = float(cfg.fiber_topology_local_warmup_lr)
    if changed.numel() == 0 or steps <= 0 or learning_rate <= 0.0:
        return {"steps": 0, "view_count": 0, "rows": int(changed.numel()), "loss": []}
    view_count = min(
        int(cfg.fiber_topology_local_warmup_views),
        len(surface_vertices_per_view),
    )
    if view_count <= 0:
        return {"steps": 0, "view_count": 0, "rows": int(changed.numel()), "loss": []}

    local_mask = torch.zeros(
        field.point_count, dtype=torch.bool, device=field.route_logits.device
    )
    local_mask[changed] = True
    local_rows = torch.nonzero(local_mask, as_tuple=False)[:, 0]
    parameters = [
        field.route_logits,
        field.residual_trust_logits,
        field.structured_delta_raw,
        field.structured_opacity_raw,
        field.radius_raw,
        field.carrier_logits,
        field.carrier_root_tip_raw,
        field.strand_length_raw,
        field.direction_local_raw,
        field.bend_local,
        field.bend_cubic_local,
        field.expert_color_delta,
    ]
    if field.expert_sh_delta_raw is not None:
        parameters.append(field.expert_sh_delta_raw)
    first_moments = [
        torch.zeros_like(parameter[local_rows]) for parameter in parameters
    ]
    second_moments = [
        torch.zeros_like(parameter[local_rows]) for parameter in parameters
    ]
    beta1, beta2 = 0.9, 0.999
    losses: list[float] = []
    gradient_norms: list[float] = []
    for warmup_step in range(steps):
        view_index = warmup_step % view_count
        vertices = surface_vertices_per_view[view_index].to(
            device=field.route_logits.device
        )
        target = targets[view_index]
        primitives = field.primitives(
            vertices,
            surface_faces,
            shell_samples=int(cfg.fiber_shell_samples),
            strand_samples=int(cfg.fiber_strand_samples),
            temperature=float(cfg.fiber_final_temperature),
            hard_route=False,
            route_blend=1.0,
            geometry_blend=1.0,
            route_hardening=0.0,
            hard_route_policy=cfg.fiber_hard_route_policy,
            fin_aspect_ratio=float(cfg.fiber_fin_aspect_ratio),
            additive_teacher=bool(cfg.fiber_additive_teacher_mode),
            teacher_opacity_transfer=float(cfg.fiber_teacher_opacity_transfer),
        )
        prediction = _render(primitives, cameras[view_index], cfg, renderer_name)
        loss, _parts = differentiable_render_loss(
            prediction,
            target["rgb"],
            target["mask"],
            cfg.color_loss_weight,
            cfg.mask_loss_weight,
            cfg.mask_boundary_weight,
            cfg.mask_boundary_radius,
            cfg.mask_balance_weight,
        )
        inside_loss, outside_loss = _bidirectional_mask_losses(
            prediction["mask"],
            target["mask"],
            inside_alpha_target=float(cfg.fiber_mask_inside_alpha_target),
            outside_margin_px=int(cfg.fiber_mask_outside_margin_px),
        )
        loss = loss + (
            float(cfg.fiber_mask_inside_coverage_weight) * inside_loss
            + float(cfg.fiber_mask_outside_spill_weight) * outside_loss
        )
        if float(cfg.fiber_max_hole_weight) > 0.0:
            loss = loss + float(cfg.fiber_max_hole_weight) * (
                _maximum_hole_soft_loss(
                    prediction["mask"],
                    target["mask"],
                    kernel_size=int(cfg.fiber_max_hole_kernel),
                    topk_fraction=float(cfg.fiber_max_hole_topk_fraction),
                )
            )
        gradients = torch.autograd.grad(
            loss, parameters, allow_unused=True, retain_graph=False
        )
        squared_gradient_norm = loss.new_zeros(())
        with torch.no_grad():
            for index, (parameter, gradient) in enumerate(
                zip(parameters, gradients, strict=True)
            ):
                if gradient is not None:
                    local_gradient = gradient[local_rows]
                    squared_gradient_norm.add_(local_gradient.square().sum())
                    first_moments[index].mul_(beta1).add_(
                        local_gradient, alpha=1.0 - beta1
                    )
                    second_moments[index].mul_(beta2).addcmul_(
                        local_gradient, local_gradient, value=1.0 - beta2
                    )
                    step_index = warmup_step + 1
                    first_hat = first_moments[index] / (1.0 - beta1**step_index)
                    second_hat = second_moments[index] / (1.0 - beta2**step_index)
                    update = learning_rate * first_hat / (
                        second_hat.sqrt() + 1e-8
                    )
                    # Advanced indexing returns a copy. index_copy_ is required
                    # to write the local birth update into the Parameter.
                    parameter.index_copy_(
                        0,
                        local_rows,
                        parameter[local_rows] - update,
                    )
        losses.append(float(loss.detach().cpu()))
        gradient_norms.append(float(squared_gradient_norm.sqrt().cpu()))
    return {
        "steps": steps,
        "view_count": view_count,
        "rows": int(local_rows.numel()),
        "loss": losses,
        "gradient_norm": gradient_norms,
    }


def _update_adaptive_topology(
    field: UnifiedFiberField,
    optimizer: torch.optim.Optimizer,
    surface_vertices_per_view: list[torch.Tensor],
    surface_faces: torch.Tensor,
    cameras: list,
    targets: list[dict[str, torch.Tensor]],
    cfg: PipelineConfig,
    renderer_name: str,
    *,
    step: int,
    validation_surface_vertices: list[torch.Tensor] | None = None,
    validation_cameras: list | None = None,
    validation_targets: list[dict[str, torch.Tensor]] | None = None,
) -> dict[str, object]:
    """Measure multi-view topology evidence and perform one ADC event."""

    if not surface_vertices_per_view or not (
        len(surface_vertices_per_view) == len(cameras) == len(targets)
    ):
        raise ValueError("Topology views, cameras, and targets must be aligned")
    device = field.route_logits.device
    dtype = field.route_logits.dtype
    visible_views = torch.zeros(field.point_count, device=device, dtype=dtype)
    residual_support_sum = torch.zeros_like(visible_views)
    structured_support_sum = torch.zeros_like(visible_views)
    deficit_sum = torch.zeros_like(visible_views)
    deficit_views = torch.zeros_like(visible_views)
    boundary_sum = torch.zeros_like(visible_views)
    structured_valid_views = torch.zeros_like(visible_views)
    scene_scale = float(field.scene_scale.detach().cpu())
    ray_steps = max(int(cfg.fiber_topology_ray_steps), 2)
    ray_lengths = torch.linspace(
        scene_scale * float(cfg.fiber_topology_ray_min_length_scale),
        scene_scale * float(cfg.fiber_topology_ray_max_length_scale),
        ray_steps,
        device=device,
        dtype=dtype,
    )
    ray_support_views = torch.zeros(
        (field.point_count, ray_steps), device=device, dtype=dtype
    )
    ray_support_sum = torch.zeros_like(ray_support_views)
    ray_deficit_sum = torch.zeros_like(ray_support_views)
    ray_observed_views = torch.zeros_like(ray_support_views)

    # Transport neighboring axes/bends through world space before averaging;
    # local tangent coordinates from different scalp triangles are not directly
    # comparable.
    reference_vertices = surface_vertices_per_view[0].to(device=device)
    ref_root, ref_tangent, ref_bitangent, ref_normal = field.surface_frame(
        reference_vertices, surface_faces
    )
    ref_frame = torch.stack(
        [ref_tangent, ref_bitangent, ref_normal], dim=-1
    )
    base_local_direction = field.direction_local.detach()
    base_world_direction = torch.einsum(
        "nij,nj->ni", ref_frame, base_local_direction
    )
    if field.route_neighbor_index.numel() > 0:
        neighbor_world = base_world_direction[field.route_neighbor_index]
        sign = torch.sign(
            torch.sum(
                neighbor_world * base_world_direction[:, None], dim=-1,
                keepdim=True,
            )
        )
        neighbor_world = torch.where(sign < 0.0, -neighbor_world, neighbor_world)
        neighbor_mean_world = F.normalize(
            neighbor_world.mean(dim=1), dim=-1, eps=1e-8
        )
        proposed_direction_local = F.normalize(
            torch.einsum("nij,ni->nj", ref_frame, neighbor_mean_world),
            dim=-1,
            eps=1e-8,
        )
        neighbor_bend = field.bend_local.detach()[field.route_neighbor_index]
        neighbor_tangent = ref_tangent[field.route_neighbor_index]
        neighbor_bitangent = ref_bitangent[field.route_neighbor_index]
        neighbor_bend_world = (
            neighbor_bend[..., :1] * neighbor_tangent
            + neighbor_bend[..., 1:] * neighbor_bitangent
        ).mean(dim=1)
        neighbor_bend_local = torch.stack(
            [
                torch.sum(neighbor_bend_world * ref_tangent, dim=-1),
                torch.sum(neighbor_bend_world * ref_bitangent, dim=-1),
            ],
            dim=-1,
        )
        direction_delta_world = neighbor_mean_world - base_world_direction
        curvature_local = torch.stack(
            [
                torch.sum(direction_delta_world * ref_tangent, dim=-1),
                torch.sum(direction_delta_world * ref_bitangent, dim=-1),
            ],
            dim=-1,
        )
        proposed_bend_local = neighbor_bend_local + float(
            cfg.fiber_topology_bend_neighbor_scale
        ) * curvature_local
        neighbor_length = field.strand_length.detach()[
            field.route_neighbor_index
        ].mean(dim=1)
    else:
        proposed_direction_local = base_local_direction
        proposed_bend_local = field.bend_local.detach().clone()
        neighbor_length = field.strand_length.detach().clone()

    with torch.no_grad():
        for vertices, camera, target in zip(
            surface_vertices_per_view, cameras, targets, strict=True
        ):
            vertices = vertices.to(device=device)
            target_mask = target["mask"].to(device=device, dtype=dtype).clamp(0.0, 1.0)
            primitives = field.primitives(
                vertices,
                surface_faces,
                shell_samples=int(cfg.fiber_shell_samples),
                strand_samples=int(cfg.fiber_strand_samples),
                temperature=float(cfg.fiber_final_temperature),
                hard_route=False,
                route_blend=1.0,
                geometry_blend=1.0,
                route_hardening=0.0,
                hard_route_policy=cfg.fiber_hard_route_policy,
                fin_aspect_ratio=float(cfg.fiber_fin_aspect_ratio),
                additive_teacher=bool(cfg.fiber_additive_teacher_mode),
                teacher_opacity_transfer=float(cfg.fiber_teacher_opacity_transfer),
            )
            prediction = _render(primitives, camera, cfg, renderer_name)
            deficit_image = F.relu(target_mask - prediction["mask"]) * target_mask
            boundary_image = _silhouette_band(
                target_mask, int(cfg.fiber_topology_boundary_radius)
            )

            roots, tangent, bitangent, normal = field.surface_frame(
                vertices, surface_faces
            )
            root_visible = _front_surface_visibility(
                roots,
                camera,
                bin_px=max(int(cfg.fiber_coverage_seed_visibility_bin_px), 1),
                depth_tolerance=(
                    scene_scale
                    * float(cfg.fiber_coverage_seed_visibility_depth_scale)
                ),
            )
            view_frame = torch.stack([tangent, bitangent, normal], dim=-1)
            ray_direction_world = F.normalize(
                torch.einsum(
                    "nij,nj->ni", view_frame, proposed_direction_local
                ),
                dim=-1,
                eps=1e-8,
            )
            ray_origin = roots + field.height[:, None] * normal
            ray_points = (
                ray_origin[:, None, :]
                + ray_lengths[None, :, None] * ray_direction_world[:, None, :]
            )
            ray_support, ray_valid = _sample_mask_at_world_points(
                ray_points,
                camera,
                target_mask,
                margin_px=max(int(cfg.fiber_visual_hull_margin_px), 0),
            )
            ray_deficit, _ = _sample_mask_at_world_points(
                ray_points, camera, deficit_image
            )
            ray_observed = root_visible[:, None] & ray_valid
            ray_support_views.add_(
                (ray_observed & (ray_support >= 0.5)).to(dtype)
            )
            ray_support_sum.add_(ray_support * ray_observed.to(dtype))
            ray_deficit_sum.add_(ray_deficit * ray_observed.to(dtype))
            ray_observed_views.add_(ray_observed.to(dtype))
            residual_xyz = roots + (
                field.residual_offset_local[:, :1] * tangent
                + field.residual_offset_local[:, 1:2] * bitangent
                + field.residual_offset_local[:, 2:3] * normal
            )
            residual_probes = _residual_footprint_probe_points(
                residual_xyz,
                tangent,
                bitangent,
                normal,
                field.residual_scaling,
                float(cfg.fiber_topology_prune_footprint_sigma),
            )
            residual_probe_support, residual_probe_valid = _sample_mask_at_world_points(
                residual_probes,
                camera,
                target_mask,
                margin_px=max(int(cfg.fiber_mask_outside_margin_px), 0),
            )
            residual_valid = residual_probe_valid.any(dim=1)
            residual_support = residual_probe_support.amax(dim=1)
            residual_observed = root_visible & residual_valid
            residual_observed_float = residual_observed.to(dtype)
            visible_views.add_(residual_observed_float)
            residual_support_sum.add_(residual_support * residual_observed_float)

            strand_points, _directions = field.strand_target_geometry(
                vertices,
                surface_faces,
                strand_samples=int(cfg.fiber_strand_samples),
            )
            support, valid = _sample_mask_at_world_points(
                strand_points,
                camera,
                target_mask,
                margin_px=max(int(cfg.fiber_visual_hull_margin_px), 0),
            )
            deficit, _ = _sample_mask_at_world_points(
                strand_points, camera, deficit_image
            )
            boundary, _ = _sample_mask_at_world_points(
                strand_points, camera, boundary_image
            )
            valid_float = valid.to(dtype)
            sample_denominator = valid_float.sum(dim=1).clamp_min(1.0)
            source_support = (support * valid_float).sum(dim=1) / sample_denominator
            source_deficit = (deficit * valid_float).amax(dim=1)
            source_boundary = (boundary * valid_float).amax(dim=1)
            source_observed = root_visible & valid.any(dim=1)
            source_observed_float = source_observed.to(dtype)
            structured_valid_views.add_(source_observed_float)
            structured_support_sum.add_(source_support * source_observed_float)
            deficit_sum.add_(source_deficit * source_observed_float)
            deficit_views.add_(
                (
                    source_observed
                    & (
                        source_deficit
                        >= float(cfg.fiber_topology_grow_min_deficit)
                    )
                ).to(dtype)
            )
            boundary_sum.add_(source_boundary * source_observed_float)

    residual_support = residual_support_sum / visible_views.clamp_min(1.0)
    structured_support = structured_support_sum / structured_valid_views.clamp_min(1.0)
    deficit_score = deficit_sum / structured_valid_views.clamp_min(1.0)
    boundary_score = boundary_sum / structured_valid_views.clamp_min(1.0)
    supported_ray = ray_support_views >= float(
        cfg.fiber_topology_deficit_min_views
    )
    ray_rank = torch.arange(ray_steps, device=device)[None, :].expand(
        field.point_count, -1
    )
    farthest_index = torch.where(
        supported_ray, ray_rank, torch.full_like(ray_rank, -1)
    ).amax(dim=1)
    has_ray = farthest_index >= 0
    safe_ray_index = farthest_index.clamp_min(0)
    ray_length_proposal = ray_lengths[safe_ray_index]
    blend = float(cfg.fiber_topology_ray_neighbor_blend)
    ray_length_proposal = torch.where(
        has_ray,
        (1.0 - blend) * ray_length_proposal + blend * neighbor_length,
        neighbor_length,
    )
    selected_ray_views = ray_support_views.gather(
        1, safe_ray_index[:, None]
    )[:, 0]
    ray_mean_support = (
        ray_support_sum / ray_observed_views.clamp_min(1.0)
    ).mean(dim=1)
    selected_ray_deficit = (
        ray_deficit_sum / ray_observed_views.clamp_min(1.0)
    ).gather(1, safe_ray_index[:, None])[:, 0]
    # The candidate's geometry is now a real 3D root-to-tip ray proposal.
    # Use its multi-view evidence for growth instead of the arbitrary dormant
    # strand length that happened to be stored in the reserve row.
    deficit_score = torch.where(has_ray, selected_ray_deficit, deficit_score)
    deficit_views = torch.where(has_ray, selected_ray_views, deficit_views)
    structured_support = torch.where(
        has_ray, ray_mean_support, structured_support
    )
    validate_event = bool(cfg.fiber_topology_validate_events)
    validation_surface_vertices = validation_surface_vertices or []
    validation_cameras = validation_cameras or []
    validation_targets = validation_targets or []
    before_losses: list[float] = []
    snapshot: dict[str, torch.Tensor] | None = None
    if validate_event:
        before_losses = _topology_validation_losses(
            field,
            validation_surface_vertices,
            surface_faces,
            validation_cameras,
            validation_targets,
            cfg,
            renderer_name,
        )
        snapshot = _topology_state_snapshot(field)
    report = _apply_adaptive_topology_scores(
        field,
        optimizer,
        cfg,
        visible_views=visible_views,
        residual_support=residual_support,
        structured_support=structured_support,
        deficit_score=deficit_score,
        boundary_score=boundary_score,
        step=step,
        deficit_views=deficit_views,
        proposed_strand_length=ray_length_proposal,
        proposed_direction_local=proposed_direction_local,
        proposed_bend_local=proposed_bend_local,
        clear_optimizer_state=False,
    )
    changed = report.pop("_changed_indices")
    if not isinstance(changed, torch.Tensor):
        raise RuntimeError("Topology event did not report changed indices")
    report["local_warmup"] = _topology_local_warmup(
        field,
        surface_vertices_per_view,
        surface_faces,
        cameras,
        targets,
        cfg,
        renderer_name,
        changed,
    )
    optimizer_rows = [
        field.route_logits,
        field.residual_trust_logits,
        field.structured_delta_raw,
        field.structured_opacity_raw,
        field.radius_raw,
        field.carrier_logits,
        field.carrier_root_tip_raw,
        field.strand_length_raw,
        field.direction_local_raw,
        field.bend_local,
        field.bend_cubic_local,
    ]
    optimizer_changed = changed
    if changed.numel() > 0 and field.route_neighbor_index.numel() > 0:
        optimizer_changed = torch.unique(
            torch.cat([changed, field.route_neighbor_index[changed].reshape(-1)])
        )
    if not validate_event:
        _clear_adam_rows(optimizer, optimizer_rows, optimizer_changed)
        return report

    after_losses = _topology_validation_losses(
        field,
        validation_surface_vertices,
        surface_faces,
        validation_cameras,
        validation_targets,
        cfg,
        renderer_name,
    )
    accepted, deltas = _topology_event_is_accepted(
        before_losses,
        after_losses,
        float(cfg.fiber_topology_validation_margin),
        allow_neutral=False,
        min_mean_improvement=float(cfg.fiber_topology_min_mean_improvement),
    )
    birth_rows = changed[field.route_active_gate[changed, 1] > 0.5]
    if birth_rows.numel() > 0:
        birth_delta = field.structured_delta_gain[birth_rows, 1]
        birth_opacity = field.structured_opacity_gain[birth_rows, 1]
        birth_route_mass = field.route_probabilities(
            temperature=float(cfg.fiber_final_temperature)
        )[birth_rows, 1]
        birth_deployment = float(birth_delta.mean().detach().cpu())
        # In non-additive teacher migration, structured opacity is deliberately
        # ignored by the renderer: ownership transfers the teacher optical
        # thickness to shell/strand without creating an additive copy.  Do not
        # reject valid geometry merely because its unused opacity gate is zero.
        birth_mass = birth_delta * birth_route_mass
        if bool(cfg.fiber_additive_teacher_mode):
            birth_mass = birth_mass * birth_opacity
        birth_effective_mass = float(birth_mass.mean().detach().cpu())
    else:
        birth_deployment = 1.0
        birth_effective_mass = 1.0
    # Values originate in float32 parameters while thresholds are Python
    # floats. A source initialized exactly on the configured boundary must not
    # fail because 0.35 is represented as 0.349999994.
    threshold_epsilon = 1e-6
    structural_birth_ok = (
        birth_deployment + threshold_epsilon
        >= float(cfg.fiber_topology_min_birth_deployment)
        and birth_effective_mass + threshold_epsilon
        >= float(cfg.fiber_topology_min_birth_effective_mass)
    )
    accepted = bool(accepted and structural_birth_ok)
    if accepted:
        _clear_adam_rows(optimizer, optimizer_rows, optimizer_changed)
    else:
        if snapshot is None:
            raise RuntimeError("Missing topology snapshot for rejected event")
        _restore_topology_state(field, snapshot)
        report["topology"] = field.active_topology_summary(
            shell_samples=int(cfg.fiber_shell_samples),
            strand_samples=int(cfg.fiber_strand_samples),
        )
        report["reverted_pruned_count"] = report["pruned_count"]
        report["reverted_grown_count"] = report["grown_count"]
        report["reverted_densified_count"] = report["densified_count"]
        report["pruned_count"] = 0
        report["grown_count"] = 0
        report["densified_count"] = 0
    report["validation"] = {
        "accepted": bool(accepted),
        "before": before_losses,
        "after": after_losses,
        "delta": deltas,
        "margin": float(cfg.fiber_topology_validation_margin),
        "view_count": len(before_losses),
        "minimum_mean_improvement": float(
            cfg.fiber_topology_min_mean_improvement
        ),
        "birth_deployment": birth_deployment,
        "birth_effective_mass": birth_effective_mass,
        "structural_birth_ok": bool(structural_birth_ok),
    }
    return report


def _write_coverage_seed_ply(
    path: Path,
    roots: torch.Tensor,
    scores: torch.Tensor,
    lengths: torch.Tensor,
) -> None:
    """Write selected carrier roots as an inspectable ASCII PLY diagnostic."""

    xyz = roots.detach().cpu().numpy()
    score = scores.detach().cpu().numpy()
    length = lengths.detach().cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        file.write("ply\nformat ascii 1.0\n")
        file.write(f"element vertex {xyz.shape[0]}\n")
        file.write("property float x\nproperty float y\nproperty float z\n")
        file.write("property float deficit_score\nproperty float strand_length\n")
        file.write("end_header\n")
        for point, point_score, point_length in zip(
            xyz, score, length, strict=True
        ):
            file.write(
                f"{point[0]:.8g} {point[1]:.8g} {point[2]:.8g} "
                f"{point_score:.8g} {point_length:.8g}\n"
            )


def _parallel_transport_surface_directions(
    local_direction: torch.Tensor,
    surface_frame: torch.Tensor,
    neighbor_index: torch.Tensor,
    observation_confidence: torch.Tensor,
    *,
    steps: int,
    observation_weight: float,
) -> torch.Tensor:
    """Diffuse an axial direction field using normal-aligned transport."""

    if neighbor_index.numel() == 0 or int(steps) <= 0:
        return F.normalize(local_direction, dim=-1, eps=1e-8)
    direction = F.normalize(local_direction, dim=-1, eps=1e-8)
    observed = direction.clone()
    frame = surface_frame
    normal = frame[:, :, 2]
    confidence = observation_confidence.reshape(-1).clamp(0.0, 1.0)
    anchor = (float(observation_weight) * confidence).clamp(0.0, 1.0)

    for _ in range(int(steps)):
        neighbor_frame = frame[neighbor_index]
        neighbor_local = direction[neighbor_index]
        neighbor_world = torch.einsum(
            "nkij,nkj->nki", neighbor_frame, neighbor_local
        )
        source_normal = normal[neighbor_index]
        target_normal = normal[:, None, :].expand_as(source_normal)
        axis = torch.linalg.cross(source_normal, target_normal, dim=-1)
        sin_angle = torch.linalg.vector_norm(axis, dim=-1, keepdim=True)
        cos_angle = torch.sum(
            source_normal * target_normal, dim=-1, keepdim=True
        ).clamp(-1.0, 1.0)
        axis_unit = axis / sin_angle.clamp_min(1e-8)
        rotated = (
            neighbor_world * cos_angle
            + torch.linalg.cross(axis_unit, neighbor_world, dim=-1) * sin_angle
            + axis_unit
            * torch.sum(axis_unit * neighbor_world, dim=-1, keepdim=True)
            * (1.0 - cos_angle)
        )
        transported_world = torch.where(
            (sin_angle > 1e-6).expand_as(rotated), rotated, neighbor_world
        )
        transported_local = torch.einsum(
            "nij,nki->nkj", frame, transported_world
        )
        sign = torch.sum(
            transported_local * direction[:, None, :], dim=-1, keepdim=True
        )
        transported_local = torch.where(
            sign < 0.0, -transported_local, transported_local
        )
        smooth = F.normalize(transported_local.mean(dim=1), dim=-1, eps=1e-8)
        direction = F.normalize(
            anchor[:, None] * observed + (1.0 - anchor[:, None]) * smooth,
            dim=-1,
            eps=1e-8,
        )
    return direction


def _area_stratified_surface_samples(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    eligible_faces: torch.Tensor,
    count: int,
    *,
    min_roots_per_face: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int]]:
    """Sample an area-balanced scalp atlas independently of source GS density."""

    count = max(int(count), 0)
    min_roots_per_face = max(int(min_roots_per_face), 0)
    eligible_faces = eligible_faces.reshape(-1).long()
    if count == 0 or eligible_faces.numel() == 0:
        empty_faces = torch.empty(0, dtype=torch.long, device=faces.device)
        empty_barycentric = torch.empty(
            (0, 3), dtype=vertices.dtype, device=vertices.device
        )
        return empty_faces, empty_barycentric, {
            "atlas_face_count": int(eligible_faces.numel()),
            "atlas_root_count": 0,
            "atlas_covered_face_count": 0,
            "atlas_max_roots_per_face": 0,
        }

    triangles = vertices[faces[eligible_faces]]
    areas = 0.5 * torch.linalg.vector_norm(
        torch.linalg.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
            dim=-1,
        ),
        dim=-1,
    )
    areas = areas.clamp_min(torch.finfo(vertices.dtype).eps)
    face_count = int(eligible_faces.numel())
    guaranteed = min(count, face_count * min_roots_per_face)
    if guaranteed > 0:
        layers = torch.arange(guaranteed, device=faces.device) // face_count
        base_local = torch.arange(guaranteed, device=faces.device) % face_count
        # Rotate successive layers so neighboring faces do not receive the
        # same low-discrepancy phase.
        base_local = (base_local + layers * 2654435761) % face_count
    else:
        base_local = torch.empty(0, dtype=torch.long, device=faces.device)
    remaining = count - guaranteed
    if remaining > 0:
        cdf = torch.cumsum(areas / areas.sum(), dim=0)
        quantiles = (
            torch.arange(remaining, device=faces.device, dtype=vertices.dtype)
            + 0.5
        ) / float(remaining)
        extra_local = torch.searchsorted(cdf, quantiles).clamp_max(face_count - 1)
        local = torch.cat([base_local.long(), extra_local.long()])
    else:
        local = base_local.long()
    selected_faces = eligible_faces[local]

    sample_index = torch.arange(count, device=faces.device, dtype=vertices.dtype)
    face_phase = selected_faces.to(vertices.dtype)
    u = torch.frac((sample_index + 1.0) * 0.754877666 + face_phase * 1.19e-7)
    v = torch.frac((sample_index + 1.0) * 0.569840296 + face_phase * 1.73e-7)
    sqrt_u = torch.sqrt(u.clamp(1e-4, 1.0 - 1e-4))
    barycentric = torch.stack(
        [1.0 - sqrt_u, sqrt_u * (1.0 - v), sqrt_u * v], dim=-1
    )
    barycentric = 0.90 * barycentric + 0.10 / 3.0
    unique_faces, counts = torch.unique(selected_faces, return_counts=True)
    report: dict[str, float | int] = {
        "atlas_face_count": face_count,
        "atlas_root_count": count,
        "atlas_covered_face_count": int(unique_faces.numel()),
        "atlas_max_roots_per_face": int(counts.max().item()),
        "atlas_mean_roots_per_covered_face": float(counts.float().mean().item()),
    }
    return selected_faces, barycentric, report


def _synchronize_outward_direction_signs(
    local_direction: torch.Tensor,
    surface_frame: torch.Tensor,
    neighbor_index: torch.Tensor,
    *,
    anchor_threshold: float,
    steps: int,
    active_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Resolve the axial d/-d ambiguity with outward anchors and continuity."""

    direction = F.normalize(local_direction, dim=-1, eps=1e-8)
    if active_mask is None:
        active_mask = torch.ones(
            direction.shape[0], dtype=torch.bool, device=direction.device
        )
    else:
        active_mask = active_mask.to(device=direction.device, dtype=torch.bool)
    active_count = active_mask.sum().clamp_min(1)
    before_inward = float(
        ((direction[:, 2] < 0.0) & active_mask).float().sum().div(active_count).cpu()
    )
    original_normal = direction[:, 2].abs()
    flip_inward = (direction[:, 2] < 0.0) & active_mask
    direction = torch.where(flip_inward[:, None], -direction, direction)
    anchored = (
        original_normal >= max(float(anchor_threshold), 0.0)
    ) & active_mask
    if neighbor_index.numel() > 0:
        for _ in range(max(int(steps), 0)):
            world = torch.einsum("nij,nj->ni", surface_frame, direction)
            neighbor_world = world[neighbor_index]
            valid_neighbor = active_mask[neighbor_index]
            consensus = F.normalize(
                (neighbor_world * valid_neighbor[..., None]).sum(dim=1),
                dim=-1,
                eps=1e-8,
            )
            has_neighbor = valid_neighbor.any(dim=1)
            flip = (
                (torch.sum(world * consensus, dim=-1) < 0.0)
                & (~anchored)
                & active_mask
                & has_neighbor
            )
            direction = torch.where(flip[:, None], -direction, direction)
    # Never allow the continuity solve to turn a root decisively inward.
    flip_inward = (direction[:, 2] < 0.0) & active_mask
    direction = torch.where(flip_inward[:, None], -direction, direction)
    direction = F.normalize(direction, dim=-1, eps=1e-8)
    world = torch.einsum("nij,nj->ni", surface_frame, direction)
    if neighbor_index.numel() > 0:
        neighbor_dot = torch.sum(world[:, None] * world[neighbor_index], dim=-1)
        valid_pair = active_mask[:, None] & active_mask[neighbor_index]
        disagreement = float(
            ((neighbor_dot < 0.0) & valid_pair).float().sum()
            .div(valid_pair.sum().clamp_min(1))
            .cpu()
        )
    else:
        disagreement = 0.0
    return direction, {
        "inward_fraction_before": before_inward,
        "inward_fraction_after": float(
            (((direction[:, 2] < 0.0) & active_mask).float().sum() / active_count).cpu()
        ),
        "neighbor_sign_disagreement_after": disagreement,
        "outward_anchor_fraction": float(
            anchored.float().sum().div(active_count).cpu()
        ),
    }


def _erode_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Differentiation-free binary/soft erosion preserving input rank."""

    radius = max(int(radius), 0)
    if radius == 0:
        return mask
    original_shape = mask.shape
    image = mask
    if image.ndim == 2:
        image = image[None, None]
    elif image.ndim == 3:
        image = image[None]
    elif image.ndim != 4:
        raise ValueError("mask must be HxW, CxHxW or NxCxHxW")
    eroded = -F.max_pool2d(
        -image,
        kernel_size=2 * radius + 1,
        stride=1,
        padding=radius,
    )
    return eroded.reshape(original_shape)


def _initialize_intrinsic_surface_propagation(
    field: UnifiedFiberField,
    surface_faces: torch.Tensor,
    surface_vertices_per_view: list[torch.Tensor],
    cameras: list,
    frame_indices: list[int],
    targets: list[dict[str, torch.Tensor]],
    orientation_targets: list[dict[str, torch.Tensor] | None],
    cfg: PipelineConfig,
    out_dir: Path,
) -> dict[str, object]:
    """Redistribute duplicate roots and complete their surface direction field."""

    if not surface_vertices_per_view or not (
        len(surface_vertices_per_view) == len(frame_indices) == len(targets)
    ):
        raise ValueError("Surface propagation views and targets must align")
    fraction = float(cfg.fiber_surface_propagation_reassign_fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("surface propagation reassign fraction must be in [0, 1]")
    if not 0.0 <= float(cfg.fiber_surface_propagation_min_fraction) <= 1.0:
        raise ValueError("surface propagation min fraction must be in [0, 1]")

    device = field.route_logits.device
    dtype = field.route_logits.dtype
    faces = surface_faces.to(device=device)
    rest_vertices = surface_vertices_per_view[0].to(device=device)
    with torch.no_grad():
        old_root, old_tangent, old_bitangent, old_normal = field.surface_frame(
            rest_vertices, faces
        )
        old_world = field.transported_residual_xyz(
            old_root, old_tangent, old_bitangent, old_normal
        ).detach()
        anchor_quantized = torch.round(field.barycentric[:, :2] * 1023.0).long()
        anchor_key = (
            field.face_index * (1024 * 1024)
            + anchor_quantized[:, 0] * 1024
            + anchor_quantized[:, 1]
        )
        order = torch.argsort(anchor_key)
        duplicate_sorted = torch.zeros_like(order, dtype=torch.bool)
        duplicate_sorted[1:] = anchor_key[order[1:]] == anchor_key[order[:-1]]
        duplicate_slots = order[duplicate_sorted]
        requested = min(
            int(round(fraction * field.point_count)), int(duplicate_slots.numel())
        )

        face_count = int(faces.shape[0])
        valid_views = torch.zeros(face_count, device=device, dtype=dtype)
        support_sum = torch.zeros_like(valid_views)
        scalp_support_sum = torch.zeros_like(valid_views)
        candidate_roots_rest = rest_vertices[faces].mean(dim=1)
        for vertices, frame_index, target in zip(
            surface_vertices_per_view, frame_indices, targets, strict=True
        ):
            candidate_roots = vertices.to(device=device)[faces].mean(dim=1)
            support, valid = _sample_mask_at_world_points(
                candidate_roots,
                cameras[frame_index],
                target["mask"].to(device=device, dtype=dtype),
                margin_px=int(cfg.fiber_surface_propagation_margin_px),
            )
            visible = valid & _front_surface_visibility(
                candidate_roots,
                cameras[frame_index],
                bin_px=max(int(cfg.fiber_coverage_seed_visibility_bin_px), 1),
                depth_tolerance=float(field.scene_scale.cpu())
                * float(cfg.fiber_coverage_seed_visibility_depth_scale),
            )
            visible_float = visible.to(dtype)
            valid_views.add_(visible_float)
            support_sum.add_(support * visible_float)
            if bool(cfg.fiber_scalp_occupancy_enabled):
                scalp_mask = _erode_mask(
                    target["mask"].to(device=device, dtype=dtype),
                    int(cfg.fiber_scalp_occupancy_erosion_px),
                )
                scalp_support, _ = _sample_mask_at_world_points(
                    candidate_roots,
                    cameras[frame_index],
                    scalp_mask,
                    margin_px=int(cfg.fiber_surface_propagation_margin_px),
                )
                scalp_support_sum.add_(scalp_support * visible_float)

        support_fraction = support_sum / valid_views.clamp_min(1.0)
        eligible_mask = (
            valid_views >= float(cfg.fiber_surface_propagation_min_views)
        ) & (
            support_fraction
            >= float(cfg.fiber_surface_propagation_min_fraction)
        )
        eligible = torch.nonzero(eligible_mask, as_tuple=False).reshape(-1)
        scalp_support_fraction = scalp_support_sum / valid_views.clamp_min(1.0)
        scalp_face_eligible = (
            valid_views >= float(cfg.fiber_scalp_occupancy_min_views)
        ) & (
            scalp_support_fraction
            >= float(cfg.fiber_scalp_occupancy_min_fraction)
        )
        selected_count = requested if int(eligible.numel()) > 0 else 0

        selected_faces = torch.empty(0, device=device, dtype=torch.long)
        selected_barycentric = torch.empty(
            (0, 3), device=device, dtype=dtype
        )
        selected_slots = torch.empty(0, device=device, dtype=torch.long)
        if selected_count > 0:
            eligible_cpu = eligible.detach().cpu().numpy()
            roots_cpu = candidate_roots_rest[eligible].detach().cpu().numpy()
            scores_cpu = support_fraction[eligible].detach().cpu().numpy()
            scene_min = roots_cpu.min(axis=0, keepdims=True)
            scene_extent = np.maximum(
                roots_cpu.max(axis=0, keepdims=True) - scene_min, 1e-8
            )
            bins = max(int(round((2 * selected_count) ** (1.0 / 3.0))), 1)
            quantized = np.minimum(
                ((roots_cpu - scene_min) / scene_extent * bins).astype(np.int64),
                bins - 1,
            )
            voxel_code = (
                quantized[:, 0]
                + bins * quantized[:, 1]
                + bins * bins * quantized[:, 2]
            )
            ranked = np.argsort(-scores_cpu, kind="stable")
            face_order: list[int] = []
            occupied: set[int] = set()
            for local_index in ranked.tolist():
                code = int(voxel_code[local_index])
                if code in occupied:
                    continue
                occupied.add(code)
                face_order.append(local_index)
            if len(face_order) < len(ranked):
                chosen_set = set(face_order)
                face_order.extend(
                    index
                    for index in ranked.tolist()
                    if index not in chosen_set
                )
            face_order_array = np.asarray(face_order, dtype=np.int64)
            repeated_local = face_order_array[
                np.arange(selected_count, dtype=np.int64)
                % len(face_order_array)
            ]
            selected_faces = torch.as_tensor(
                eligible_cpu[repeated_local],
                device=device,
                dtype=torch.long,
            )
            layer = torch.arange(selected_count, device=device) // max(
                len(face_order_array), 1
            )
            face_phase = selected_faces.to(dtype)
            u = torch.frac(
                (layer.to(dtype) + 1.0) * 0.754877666
                + face_phase * 0.000000119
            )
            v = torch.frac(
                (layer.to(dtype) + 1.0) * 0.569840296
                + face_phase * 0.000000173
            )
            sqrt_u = torch.sqrt(u.clamp(1e-4, 1.0 - 1e-4))
            selected_barycentric = torch.stack(
                [
                    1.0 - sqrt_u,
                    sqrt_u * (1.0 - v),
                    sqrt_u * v,
                ],
                dim=-1,
            )
            # Keep roots away from exact triangle edges, where tiny camera or
            # mesh perturbations can change the owning face discontinuously.
            selected_barycentric = (
                0.90 * selected_barycentric + 0.10 / 3.0
            )
            slot_order = torch.argsort(field.opacity[duplicate_slots])
            selected_slots = duplicate_slots[slot_order[:selected_count]]
            field.face_index[selected_slots] = selected_faces
            field.barycentric[selected_slots] = selected_barycentric

            new_root, tangent, bitangent, normal = field.surface_frame(
                rest_vertices, faces
            )
            offset = old_world - new_root
            new_local_offset = torch.stack(
                [
                    torch.sum(offset * tangent, dim=-1),
                    torch.sum(offset * bitangent, dim=-1),
                    torch.sum(offset * normal, dim=-1),
                ],
                dim=-1,
            )
            field.residual_offset_local[selected_slots] = new_local_offset[
                selected_slots
            ]
            field.initial_residual_offset_local[selected_slots] = new_local_offset[
                selected_slots
            ]
            new_frame = torch.stack([tangent, bitangent, normal], dim=-1)
            field.rest_surface_frame[selected_slots] = new_frame[selected_slots]
        else:
            new_root, tangent, bitangent, normal = field.surface_frame(
                rest_vertices, faces
            )
            new_frame = torch.stack([tangent, bitangent, normal], dim=-1)

        neighbor_k = max(int(cfg.fiber_surface_propagation_neighbor_k), 0)
        neighbor_numpy = _surface_knn_indices(
            new_root.detach().cpu().numpy(), neighbor_k
        )
        field.route_neighbor_index = torch.as_tensor(
            neighbor_numpy, device=device, dtype=torch.long
        )

        if orientation_targets and any(
            target is not None for target in orientation_targets
        ):
            local_direction, _world_direction, confidence, orientation_report = (
                _estimate_multiview_orientation_directions(
                    new_root,
                    tangent,
                    bitangent,
                    normal,
                    cameras,
                    frame_indices,
                    targets,
                    orientation_targets,
                    min_views=int(cfg.fiber_surface_propagation_min_views),
                    normal_bias=float(
                        cfg.fiber_surface_propagation_normal_bias
                    ),
                    confidence_floor=float(
                        cfg.fiber_surface_propagation_confidence_floor
                    ),
                )
            )
        else:
            local_direction = field.direction_local.detach().clone()
            confidence = torch.zeros(field.point_count, device=device, dtype=dtype)
            orientation_report = {"source": "source_gaussian_axes"}
        retained = torch.ones(field.point_count, device=device, dtype=torch.bool)
        retained[selected_slots] = False
        confidence[retained] = torch.maximum(
            confidence[retained], torch.full_like(confidence[retained], 0.15)
        )
        propagated = _parallel_transport_surface_directions(
            local_direction,
            new_frame,
            field.route_neighbor_index,
            confidence,
            steps=int(cfg.fiber_surface_propagation_steps),
            observation_weight=float(
                cfg.fiber_surface_propagation_observation_weight
            ),
        )
        field.direction_local_raw.copy_(propagated)

        scalp_source_count = 0
        scalp_initial_strand_count = 0
        scalp_atlas_report: dict[str, float | int] = {}
        strand_sign_report: dict[str, float] = {}
        if bool(cfg.fiber_scalp_occupancy_enabled):
            source_scalp = scalp_face_eligible[field.face_index]
            field.strand_root_occupancy.copy_(source_scalp.to(dtype))
            scalp_sources = torch.nonzero(source_scalp, as_tuple=False).reshape(-1)
            initial_fraction = float(cfg.fiber_scalp_initial_strand_fraction)
            initial_count = min(
                int(round(initial_fraction * int(scalp_sources.numel()))),
                int(scalp_sources.numel()),
            )
            initial_strands = torch.empty(0, device=device, dtype=torch.long)
            if initial_count > 0:
                # Prefer confident, anisotropic and opaque sources for the
                # initial strand set. Remaining scalp slots form a genuine
                # growth reserve that topology can activate from deficits.
                scale = torch.sort(field.original_scaling, dim=-1).values
                anisotropy = torch.log(
                    (scale[:, -1] / scale[:, 0].clamp_min(1e-12)).clamp_min(1.0)
                )
                score = (
                    scalp_support_fraction[field.face_index]
                    + 0.20 * _rank_unit_interval(anisotropy)
                    + 0.10 * field.opacity.detach()
                )
                ranked = scalp_sources[
                    torch.argsort(score[scalp_sources], descending=True)
                ]
                initial_strands = ranked[:initial_count]
            if bool(cfg.fiber_scalp_atlas_enabled) and scalp_sources.numel() > 0:
                atlas_old_world = field.transported_residual_xyz(
                    new_root, tangent, bitangent, normal
                ).detach()
                old_direction_world = torch.einsum(
                    "nij,nj->ni", new_frame, field.direction_local.detach()
                )
                scalp_faces = torch.nonzero(
                    scalp_face_eligible, as_tuple=False
                ).reshape(-1)
                active_atlas_faces, active_atlas_barycentric, active_report = (
                    _area_stratified_surface_samples(
                        rest_vertices, faces, scalp_faces, initial_count,
                        min_roots_per_face=int(
                            cfg.fiber_scalp_atlas_min_roots_per_face
                        ),
                    )
                )
                reserve_count = int(scalp_sources.numel()) - initial_count
                reserve_atlas_faces, reserve_atlas_barycentric, reserve_report = (
                    _area_stratified_surface_samples(
                        rest_vertices, faces, scalp_faces, reserve_count,
                        min_roots_per_face=int(
                            cfg.fiber_scalp_atlas_min_roots_per_face
                        ),
                    )
                )
                atlas_faces = torch.cat(
                    [active_atlas_faces, reserve_atlas_faces], dim=0
                )
                atlas_barycentric = torch.cat(
                    [active_atlas_barycentric, reserve_atlas_barycentric], dim=0
                )
                _, combined_counts = torch.unique(
                    atlas_faces, return_counts=True
                )
                scalp_atlas_report = {
                    "atlas_face_count": int(scalp_faces.numel()),
                    "atlas_root_count": int(atlas_faces.numel()),
                    "atlas_covered_face_count": int(
                        torch.unique(atlas_faces).numel()
                    ),
                    "atlas_max_roots_per_face": int(combined_counts.max().item()),
                    "atlas_mean_roots_per_covered_face": float(
                        combined_counts.float().mean().item()
                    ),
                    "active_covered_face_count": int(
                        active_report["atlas_covered_face_count"]
                    ),
                    "reserve_covered_face_count": int(
                        reserve_report["atlas_covered_face_count"]
                    ),
                    "reserve_root_count": reserve_count,
                }
                initial_mask = torch.zeros(
                    field.point_count, device=device, dtype=torch.bool
                )
                initial_mask[initial_strands] = True
                atlas_sources = torch.cat(
                    [initial_strands, scalp_sources[~initial_mask[scalp_sources]]]
                )
                # The atlas owns the complete scalp capacity, not just the
                # currently active strands.  Inactive rows are genuine 3D
                # deficit candidates distributed over the whole supported
                # scalp instead of copies of teacher-GS clusters.
                # Keep each source's pre-atlas route identity.  Active strands
                # receive the first area-balanced block, then reserve sources
                # receive the remainder; this makes rebinding render-equivalent
                # while both subsets cover the supported scalp.
                field.face_index[atlas_sources] = atlas_faces
                field.barycentric[atlas_sources] = atlas_barycentric
                new_root, tangent, bitangent, normal = field.surface_frame(
                    rest_vertices, faces
                )
                new_frame = torch.stack([tangent, bitangent, normal], dim=-1)
                offset = atlas_old_world - new_root
                new_local_offset = torch.stack(
                    [
                        torch.sum(offset * tangent, dim=-1),
                        torch.sum(offset * bitangent, dim=-1),
                        torch.sum(offset * normal, dim=-1),
                    ],
                    dim=-1,
                )
                field.residual_offset_local[atlas_sources] = new_local_offset[
                    atlas_sources
                ]
                desired_exact_delta = atlas_old_world - field.original_xyz
                desired_exact_delta_local = torch.stack(
                    [
                        torch.sum(desired_exact_delta * tangent, dim=-1),
                        torch.sum(desired_exact_delta * bitangent, dim=-1),
                        torch.sum(desired_exact_delta * normal, dim=-1),
                    ],
                    dim=-1,
                )
                field.initial_residual_offset_local[atlas_sources] = (
                    new_local_offset[atlas_sources]
                    - desired_exact_delta_local[atlas_sources]
                )
                field.rest_surface_frame[atlas_sources] = new_frame[
                    atlas_sources
                ]
                # Preserve the already-estimated world axis across the rebind;
                # a fresh multi-view estimate below then adapts it to the atlas.
                rebound_local = torch.einsum(
                    "nij,ni->nj", new_frame, old_direction_world
                )
                field.direction_local_raw.copy_(
                    F.normalize(rebound_local, dim=-1, eps=1e-8)
                )
                neighbor_numpy = _surface_knn_indices(
                    new_root.detach().cpu().numpy(), neighbor_k
                )
                field.route_neighbor_index = torch.as_tensor(
                    neighbor_numpy, device=device, dtype=torch.long
                )
                if orientation_targets and any(
                    target is not None for target in orientation_targets
                ):
                    atlas_direction, _world, atlas_confidence, atlas_orientation = (
                        _estimate_multiview_orientation_directions(
                            new_root,
                            tangent,
                            bitangent,
                            normal,
                            cameras,
                            frame_indices,
                            targets,
                            orientation_targets,
                            min_views=int(
                                cfg.fiber_surface_propagation_min_views
                            ),
                            normal_bias=float(
                                cfg.fiber_surface_propagation_normal_bias
                            ),
                            confidence_floor=float(
                                cfg.fiber_surface_propagation_confidence_floor
                            ),
                        )
                    )
                    atlas_direction = _parallel_transport_surface_directions(
                        atlas_direction,
                        new_frame,
                        field.route_neighbor_index,
                        atlas_confidence,
                        steps=int(cfg.fiber_surface_propagation_steps),
                        observation_weight=float(
                            cfg.fiber_surface_propagation_observation_weight
                        ),
                    )
                    field.direction_local_raw.copy_(atlas_direction)
                    orientation_report = {
                        **orientation_report,
                        "atlas_reestimate": atlas_orientation,
                    }
            if bool(cfg.fiber_strand_outward_sign_enabled):
                signed_direction, strand_sign_report = (
                    _synchronize_outward_direction_signs(
                        field.direction_local.detach(),
                        new_frame,
                        field.route_neighbor_index,
                        anchor_threshold=float(
                            cfg.fiber_strand_outward_anchor_threshold
                        ),
                        steps=int(cfg.fiber_strand_sign_sync_steps),
                    )
                )
                current_direction = field.direction_local.detach().clone()
                current_direction[source_scalp] = signed_direction[source_scalp]
                field.direction_local_raw.copy_(current_direction)
            shell_index = ROUTE_NAMES.index("shell")
            strand_index = ROUTE_NAMES.index("strand")
            residual_index = ROUTE_NAMES.index("residual")
            residual_active = field.route_active_gate[:, residual_index].clone()
            field.route_active_gate[:, shell_index] = 1.0
            field.route_active_gate[:, strand_index] = 0.0
            field.route_active_gate[initial_strands, shell_index] = 0.0
            field.route_active_gate[initial_strands, strand_index] = 1.0
            field.route_active_gate[:, residual_index] = residual_active
            field.initial_route_probabilities.copy_(
                field.route_probabilities(
                    temperature=float(cfg.fiber_final_temperature)
                ).clamp_min(1e-6)
            )
            scalp_source_count = int(scalp_sources.numel())
            scalp_initial_strand_count = int(initial_strands.numel())

        final_quantized = torch.round(field.barycentric[:, :2] * 1023.0).long()
        unique_after = torch.unique(
            field.face_index * (1024 * 1024)
            + final_quantized[:, 0] * 1024
            + final_quantized[:, 1]
        ).numel()
        selected_roots = new_root[selected_slots]
        if selected_count > 0:
            _write_coverage_seed_ply(
                out_dir / "surface_propagation_roots.ply",
                selected_roots,
                support_fraction[selected_faces],
                field.strand_length[selected_slots],
            )
        report: dict[str, object] = {
            "requested_fraction": fraction,
            "duplicate_slot_count": int(duplicate_slots.numel()),
            "requested_count": requested,
            "eligible_face_count": int(eligible.numel()),
            "reassigned_count": selected_count,
            "unique_anchor_count_before": int(torch.unique(anchor_key).numel()),
            "unique_anchor_count_after": int(unique_after),
            "mean_selected_support": float(
                support_fraction[selected_faces].mean().cpu()
            )
            if selected_count > 0
            else 0.0,
            "direction_observed_fraction": float((confidence > 0.0).float().mean().cpu()),
            "direction_confidence_mean": float(confidence.mean().cpu()),
            "direction_steps": int(cfg.fiber_surface_propagation_steps),
            "neighbor_k": neighbor_k,
            "shell_propagated_direction_weight": float(
                cfg.fiber_shell_propagated_direction_weight
            ),
            "scalp_occupancy_enabled": bool(
                cfg.fiber_scalp_occupancy_enabled
            ),
            "scalp_face_count": int(scalp_face_eligible.sum().cpu()),
            "scalp_source_count": scalp_source_count,
            "scalp_initial_strand_count": scalp_initial_strand_count,
            "scalp_erosion_px": int(cfg.fiber_scalp_occupancy_erosion_px),
            "scalp_atlas": scalp_atlas_report,
            "strand_direction_sign": strand_sign_report,
            "orientation": orientation_report,
            "diagnostic_ply": str(out_dir / "surface_propagation_roots.ply"),
        }
        (out_dir / "surface_propagation_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        return report


def _estimate_multiview_orientation_directions(
    roots: torch.Tensor,
    tangent: torch.Tensor,
    bitangent: torch.Tensor,
    normal: torch.Tensor,
    cameras: list,
    frame_indices: list[int],
    targets: list[dict[str, torch.Tensor]],
    orientation_targets: list[dict[str, torch.Tensor] | None],
    *,
    min_views: int,
    normal_bias: float,
    confidence_floor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """Triangulate sign-invariant 3D hair directions from 2D orientations.

    For each calibrated view, the image orientation contributes one linear
    constraint: the projected 3D direction must have zero component along the
    perpendicular image direction.  The least-eigenvalue vector of the
    confidence-weighted normal matrix solves these constraints without using
    any 3D strand annotation.  A small outward-normal bias resolves the line
    sign/root ambiguity and prevents immediately inward-growing carriers.
    """

    if not (
        len(frame_indices)
        == len(targets)
        == len(orientation_targets)
    ):
        raise ValueError("Orientation initialization views must align")
    device, dtype = roots.device, roots.dtype
    covariance = torch.zeros(
        (roots.shape[0], 3, 3), device=device, dtype=dtype
    )
    weight_sum = torch.zeros(roots.shape[0], device=device, dtype=dtype)
    view_count = torch.zeros(roots.shape[0], device=device, dtype=torch.int32)

    for frame_index, target, orientation in zip(
        frame_indices, targets, orientation_targets, strict=True
    ):
        if orientation is None:
            continue
        camera = cameras[frame_index]
        target_mask = target["mask"].to(device=device, dtype=dtype)
        support, valid = _sample_mask_at_world_points(
            roots, camera, target_mask, margin_px=1
        )
        confidence, _ = _sample_mask_at_world_points(
            roots,
            camera,
            orientation["confidence"].to(device=device, dtype=dtype),
        )
        vector_x, _ = _sample_mask_at_world_points(
            roots,
            camera,
            orientation["vectors"][..., 0].to(device=device, dtype=dtype),
        )
        vector_y, _ = _sample_mask_at_world_points(
            roots,
            camera,
            orientation["vectors"][..., 1].to(device=device, dtype=dtype),
        )
        perpendicular = _hairgs_orientation_perpendicular(vector_x, vector_y)

        world_to_camera = torch.as_tensor(
            camera.world_to_camera, device=device, dtype=dtype
        )
        rotation = world_to_camera[:3, :3]
        camera_xyz = roots @ rotation.T + world_to_camera[:3, 3]
        z = camera_xyz[:, 2].clamp_min(1e-6)
        jacobian_x = torch.stack(
            [
                torch.full_like(z, float(camera.fx)) / z,
                torch.zeros_like(z),
                -float(camera.fx) * camera_xyz[:, 0] / z.square(),
            ],
            dim=-1,
        )
        y_sign = 1.0 if camera.image_y_down else -1.0
        jacobian_y = torch.stack(
            [
                torch.zeros_like(z),
                torch.full_like(z, y_sign * float(camera.fy)) / z,
                -y_sign * float(camera.fy) * camera_xyz[:, 1] / z.square(),
            ],
            dim=-1,
        )
        constraint_camera = (
            perpendicular[:, :1] * jacobian_x
            + perpendicular[:, 1:] * jacobian_y
        )
        constraint_world = F.normalize(
            constraint_camera @ rotation, dim=-1, eps=1e-8
        )
        weight = (
            confidence.clamp_min(float(confidence_floor))
            * support.clamp(0.0, 1.0)
            * valid.to(dtype)
        )
        covariance.add_(
            weight[:, None, None]
            * constraint_world[:, :, None]
            * constraint_world[:, None, :]
        )
        weight_sum.add_(weight)
        observation_threshold = max(0.5 * float(confidence_floor), 1e-6)
        view_count.add_((weight > observation_threshold).to(view_count.dtype))

    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    estimated = eigenvectors[..., 0]
    eigengap = (eigenvalues[:, 1] - eigenvalues[:, 0]) / eigenvalues[:, 2].clamp_min(
        1e-8
    )
    reliable = (
        (view_count >= max(int(min_views), 2))
        & (weight_sum > 1e-6)
        & (eigengap > 1e-4)
    )
    # Orient the undirected line toward gravity/downward, with the outward
    # normal as a secondary cue for nearly horizontal hair.
    sign_score = -estimated[:, 1] + 0.20 * torch.sum(estimated * normal, dim=-1)
    estimated = torch.where(
        (sign_score < 0.0)[:, None], -estimated, estimated
    )
    estimated = F.normalize(
        estimated + float(normal_bias) * normal, dim=-1, eps=1e-8
    )
    world_direction = torch.where(reliable[:, None], estimated, normal)
    local_direction = F.normalize(
        torch.stack(
            [
                torch.sum(world_direction * tangent, dim=-1),
                torch.sum(world_direction * bitangent, dim=-1),
                torch.sum(world_direction * normal, dim=-1),
            ],
            dim=-1,
        ),
        dim=-1,
        eps=1e-8,
    )
    report = {
        "reliable_fraction": float(reliable.float().mean().cpu()),
        "mean_supporting_views": float(view_count.float().mean().cpu()),
        "mean_eigengap": float(eigengap[reliable].mean().cpu())
        if torch.any(reliable)
        else 0.0,
        "normal_bias": float(normal_bias),
        "confidence_floor": float(confidence_floor),
    }
    confidence = torch.where(
        reliable,
        eigengap.clamp(0.0, 1.0)
        * (weight_sum / max(len(frame_indices), 1)).clamp(0.0, 1.0),
        torch.zeros_like(eigengap),
    )
    return local_direction, world_direction, confidence, report


def _hairgs_orientation_perpendicular(
    double_angle_cosine: torch.Tensor,
    double_angle_sine: torch.Tensor,
) -> torch.Tensor:
    """Decode HairGS' y-axis-clockwise axial orientation convention."""

    theta = 0.5 * torch.atan2(double_angle_sine, double_angle_cosine)
    # HairGS tangent is [sin(theta), cos(theta)] in image x/y coordinates.
    return torch.stack([torch.cos(theta), -torch.sin(theta)], dim=-1)


def _initialize_multiview_coverage_seeds(
    field: UnifiedFiberField,
    teacher_field: UnifiedFiberField,
    surface_faces: torch.Tensor,
    surface_vertices_per_view: list[torch.Tensor],
    cameras: list,
    frame_indices: list[int],
    targets: list[dict[str, torch.Tensor]],
    orientation_targets: list[dict[str, torch.Tensor] | None] | None,
    cfg: PipelineConfig,
    renderer_name: str,
    out_dir: Path,
) -> tuple[dict[str, object] | None, dict[int, torch.Tensor]]:
    """Activate structured roots that explain frozen-teacher mask deficits.

    Each surface root proposes a short normal-grown segment.  Candidate
    samples must lie inside a sufficient fraction of calibrated hair masks;
    among those visual-hull-consistent samples, the frozen residual teacher's
    alpha deficit provides the allocation score.  This changes the 3D support
    of the structured representation instead of asking another scalar loss to
    discover sparse carriers from an all-zero activation state.
    """

    requested_count = int(cfg.fiber_coverage_seed_count)
    if requested_count <= 0:
        return None, {}
    if not (
        len(surface_vertices_per_view)
        == len(frame_indices)
        == len(targets)
    ):
        raise ValueError("Coverage seed views, indices, and targets must align")
    if not frame_indices:
        raise ValueError("Coverage seeding requires at least one calibrated view")

    sample_count = int(cfg.fiber_coverage_seed_samples)
    min_views = int(cfg.fiber_coverage_seed_min_views)
    min_fraction = float(cfg.fiber_coverage_seed_min_fraction)
    min_deficit = float(cfg.fiber_coverage_seed_min_deficit)
    device = field.route_logits.device
    dtype = field.route_logits.dtype
    faces = surface_faces.to(device=device)
    scene_scale = field.scene_scale.detach().to(device=device, dtype=dtype)
    candidate_length = scene_scale * float(
        cfg.fiber_coverage_seed_strand_length_scale
    )
    candidate_t = (
        torch.arange(1, sample_count + 1, device=device, dtype=dtype)
        / float(sample_count)
    )

    with torch.no_grad():
        rest_vertices = surface_vertices_per_view[0].to(device=device)
        roots, tangent, bitangent, normals = field.surface_frame(
            rest_vertices, faces
        )
        candidate_local_direction = torch.zeros_like(roots)
        candidate_local_direction[:, 2] = 1.0
        candidate_world_direction = normals
        orientation_report: dict[str, float] | None = None
        if bool(cfg.fiber_coverage_seed_orientation_init):
            if orientation_targets is None:
                raise ValueError(
                    "Coverage orientation initialization requires orientation targets"
                )
            (
                candidate_local_direction,
                candidate_world_direction,
                _candidate_direction_confidence,
                orientation_report,
            ) = _estimate_multiview_orientation_directions(
                roots,
                tangent,
                bitangent,
                normals,
                cameras,
                frame_indices,
                targets,
                orientation_targets,
                min_views=min_views,
                normal_bias=float(
                    cfg.fiber_coverage_seed_orientation_normal_bias
                ),
                confidence_floor=float(
                    cfg.fiber_coverage_seed_orientation_confidence_floor
                ),
            )
            # Direction is structured-only, so initialize every reliable root.
            # Hard routing may later activate carriers beyond the explicit
            # deficit-seed budget; those roots must not fall back to radial
            # surface normals.
            field.direction_local_raw.copy_(candidate_local_direction)
        candidate_points = roots[:, None, :] + (
            candidate_length
            * candidate_t[None, :, None]
            * candidate_world_direction[:, None, :]
        )
        valid_count = torch.zeros(
            (field.point_count, sample_count), device=device, dtype=dtype
        )
        support_sum = torch.zeros_like(valid_count)
        deficit_sum = torch.zeros_like(valid_count)
        root_visible_count = torch.zeros(
            field.point_count, device=device, dtype=dtype
        )
        root_visible_hair_support = torch.zeros_like(root_visible_count)
        teacher_masks: dict[int, torch.Tensor] = {}
        teacher_deficit_mean: list[float] = []

        for vertices, frame_index, target in zip(
            surface_vertices_per_view, frame_indices, targets, strict=True
        ):
            camera = cameras[frame_index]
            vertices = vertices.to(device=device)
            teacher_prediction = _render(
                teacher_field.residual_primitives(vertices, faces),
                camera,
                cfg,
                renderer_name,
            )
            teacher_mask = teacher_prediction["mask"].detach().clamp(0.0, 1.0)
            teacher_masks[frame_index] = teacher_mask
            target_mask = target["mask"].to(
                device=device, dtype=dtype
            ).clamp(0.0, 1.0)
            deficit_image = F.relu(target_mask - teacher_mask)
            teacher_deficit_mean.append(
                float(
                    (deficit_image * target_mask).sum().cpu()
                    / target_mask.sum().clamp_min(1.0).cpu()
                )
            )
            root_support, root_valid = _sample_mask_at_world_points(
                roots,
                camera,
                target_mask,
                margin_px=int(cfg.fiber_visual_hull_margin_px),
            )
            root_visible = root_valid
            if bool(cfg.fiber_coverage_seed_visibility_cull):
                root_visible = root_visible & _front_surface_visibility(
                    roots,
                    camera,
                    bin_px=int(cfg.fiber_coverage_seed_visibility_bin_px),
                    depth_tolerance=float(scene_scale.cpu())
                    * float(cfg.fiber_coverage_seed_visibility_depth_scale),
                )
            root_visible_float = root_visible.to(dtype)
            root_visible_count.add_(root_visible_float)
            root_visible_hair_support.add_(root_support * root_visible_float)

            support, valid = _sample_mask_at_world_points(
                candidate_points,
                camera,
                target_mask,
                margin_px=int(cfg.fiber_visual_hull_margin_px),
            )
            deficit, _ = _sample_mask_at_world_points(
                candidate_points,
                camera,
                deficit_image,
            )
            valid_float = valid.to(dtype=dtype) * root_visible_float[:, None]
            valid_count.add_(valid_float)
            support_sum.add_(support * valid_float)
            deficit_sum.add_(deficit * valid_float)

        denominator = valid_count.clamp_min(1.0)
        support_fraction = support_sum / denominator
        deficit_mean = deficit_sum / denominator
        accepted_samples = (
            (valid_count >= float(min_views))
            & (support_fraction >= min_fraction)
            & (deficit_mean >= min_deficit)
        )
        sample_score = torch.where(
            accepted_samples,
            deficit_mean * support_fraction,
            torch.zeros_like(deficit_mean),
        )
        root_score, best_sample_index = sample_score.max(dim=1)
        root_occupancy_fraction = (
            root_visible_hair_support / root_visible_count.clamp_min(1.0)
        )
        eligible_mask = root_score > 0.0
        if bool(getattr(field, "structured_foreground_only", False)):
            eligible_mask &= field.source_foreground > 0.5
        eligible = torch.nonzero(eligible_mask, as_tuple=False).reshape(-1)
        if eligible.numel() == 0:
            report: dict[str, object] = {
                "requested_count": requested_count,
                "eligible_count": 0,
                "selected_count": 0,
                "teacher_mask_deficit_mean": teacher_deficit_mean,
                "visibility_cull": bool(cfg.fiber_coverage_seed_visibility_cull),
                "mean_visible_views": float(root_visible_count.mean().cpu()),
                "reason": "no visual-hull-consistent teacher-deficit candidates",
            }
            (out_dir / "coverage_seed_report.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )
            return report, teacher_masks

        sorted_eligible = eligible[
            torch.argsort(root_score[eligible], descending=True)
        ]
        voxel_size = max(
            float(scene_scale.cpu()) * float(cfg.fiber_coverage_seed_voxel_scale),
            1e-8,
        )
        roots_cpu = roots.detach().cpu()
        voxels = torch.floor(
            (roots_cpu - roots_cpu.amin(dim=0, keepdim=True)) / voxel_size
        ).to(torch.int64)
        ordered = sorted_eligible.detach().cpu().tolist()
        selected_cpu: list[int] = []
        selected_set: set[int] = set()
        occupied_voxels: set[tuple[int, int, int]] = set()
        limit = min(requested_count, len(ordered))
        for index in ordered:
            voxel = tuple(int(value) for value in voxels[index].tolist())
            if voxel in occupied_voxels:
                continue
            occupied_voxels.add(voxel)
            selected_cpu.append(index)
            selected_set.add(index)
            if len(selected_cpu) >= limit:
                break
        # If the requested budget is denser than the voxel grid, preserve the
        # score ordering when filling the remainder instead of silently using
        # fewer carriers.
        if len(selected_cpu) < limit:
            for index in ordered:
                if index in selected_set:
                    continue
                selected_cpu.append(index)
                if len(selected_cpu) >= limit:
                    break

        selected = torch.as_tensor(selected_cpu, device=device, dtype=torch.long)
        selected_best_t = candidate_t[best_sample_index[selected]].clamp_min(0.35)
        selected_strand_length = candidate_length * selected_best_t
        selected_shell_length = torch.minimum(
            selected_strand_length,
            scene_scale * float(cfg.fiber_coverage_seed_shell_length_scale),
        )

        field.direction_local_raw[selected] = candidate_local_direction[selected]
        field.bend_local[selected] = 0.0
        field.bend_cubic_local[selected] = 0.0
        field.strand_length_raw[selected, 0] = _positive_parameter_raw(
            selected_strand_length, eps=field.positive_eps
        )
        field.shell_length_raw[selected, 0] = _positive_parameter_raw(
            selected_shell_length, eps=field.positive_eps
        )
        field.structured_delta_raw[selected] = float(
            cfg.fiber_coverage_seed_geometry_gain
        )
        field.structured_opacity_raw[selected] = float(
            cfg.fiber_coverage_seed_structured_opacity
        )
        # Coverage seeds are real topology births when the structured pool is
        # initially dormant.  Both routes are made available; their learned
        # probabilities still decide whether a Fin or a strand is deployed.
        field.route_active_gate[selected, ROUTE_NAMES.index("shell")] = 1.0
        field.route_active_gate[selected, ROUTE_NAMES.index("strand")] = 1.0

        desired_route = torch.as_tensor(
            cfg.fiber_coverage_seed_route_mass, device=device, dtype=dtype
        ).reshape(-1)
        desired_route = desired_route / desired_route.sum()
        residual_index = ROUTE_NAMES.index("residual")
        seed_trust = min(0.02, 0.5 * float(desired_route[residual_index].cpu()))
        seed_trust = max(seed_trust, 1e-4)
        base_route = desired_route.clone()
        base_route[:residual_index] /= 1.0 - seed_trust
        base_route[residual_index] = (
            base_route[residual_index] - seed_trust
        ) / (1.0 - seed_trust)
        temperature = float(cfg.fiber_final_temperature)
        field.residual_trust_logits[selected, 0] = torch.logit(
            torch.tensor(seed_trust, device=device, dtype=dtype)
        )
        field.route_logits[selected] = temperature * torch.log(
            base_route.clamp_min(1e-6)
        )
        field.initial_route_probabilities[selected] = desired_route

        desired_carrier = torch.stack(
            [desired_route[2], desired_route[0], desired_route[1]]
        )
        desired_carrier = desired_carrier / desired_carrier.sum()
        field.carrier_logits[selected] = temperature * torch.log(
            desired_carrier.clamp_min(1e-6)
        )
        field.initial_carrier_probabilities[selected] = desired_carrier
        field.carrier_root_tip_raw[selected] = 0.0
        field.initial_carrier_root_tip[selected] = 0.0

        selected_scores = root_score[selected]
        selected_roots = roots[selected]
        _write_coverage_seed_ply(
            out_dir / "coverage_seed_roots.ply",
            selected_roots,
            selected_scores,
            selected_strand_length,
        )
        report = {
            "requested_count": requested_count,
            "eligible_count": int(eligible.numel()),
            "selected_count": int(selected.numel()),
            "sample_count": sample_count,
            "min_views": min_views,
            "min_fraction": min_fraction,
            "min_deficit": min_deficit,
            "voxel_size_world": voxel_size,
            "teacher_mask_deficit_mean": teacher_deficit_mean,
            "visibility_cull": bool(cfg.fiber_coverage_seed_visibility_cull),
            "visibility_bin_px": int(cfg.fiber_coverage_seed_visibility_bin_px),
            "visibility_depth_scale": float(
                cfg.fiber_coverage_seed_visibility_depth_scale
            ),
            "mean_visible_views": float(root_visible_count.mean().cpu()),
            "selected_visible_views_mean": float(
                root_visible_count[selected].mean().cpu()
            ),
            "selected_root_occupancy_mean": float(
                root_occupancy_fraction[selected].mean().cpu()
            ),
            "selected_score_min": float(selected_scores.min().cpu()),
            "selected_score_mean": float(selected_scores.mean().cpu()),
            "selected_score_max": float(selected_scores.max().cpu()),
            "selected_strand_length_min": float(selected_strand_length.min().cpu()),
            "selected_strand_length_mean": float(selected_strand_length.mean().cpu()),
            "selected_strand_length_max": float(selected_strand_length.max().cpu()),
            "route_mass": [float(value) for value in desired_route.cpu()],
            "structured_opacity": float(
                cfg.fiber_coverage_seed_structured_opacity
            ),
            "geometry_gain": float(cfg.fiber_coverage_seed_geometry_gain),
            "orientation_initialization": orientation_report,
            "diagnostic_ply": str(out_dir / "coverage_seed_roots.ply"),
        }
        (out_dir / "coverage_seed_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        return report, teacher_masks


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


def _masked_rgb_gradient_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    """Match foreground and silhouette image gradients to counter L1 blur."""

    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("RGB gradient inputs must be aligned HxWxC tensors")
    if tuple(target_mask.shape) != tuple(target.shape[:2]):
        raise ValueError("RGB gradient mask must match the image height and width")
    pred = prediction.permute(2, 0, 1)[None]
    truth = target.permute(2, 0, 1)[None]
    mask = target_mask.clamp(0.0, 1.0)[None, None]
    pred_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    truth_x = truth[:, :, :, 1:] - truth[:, :, :, :-1]
    pred_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    truth_y = truth[:, :, 1:, :] - truth[:, :, :-1, :]
    weight_x = torch.maximum(mask[:, :, :, 1:], mask[:, :, :, :-1])
    weight_y = torch.maximum(mask[:, :, 1:, :], mask[:, :, :-1, :])
    loss_x = (torch.abs(pred_x - truth_x) * weight_x).sum() / (
        weight_x.sum() * prediction.shape[-1]
    ).clamp_min(1.0)
    loss_y = (torch.abs(pred_y - truth_y) * weight_y).sum() / (
        weight_y.sum() * prediction.shape[-1]
    ).clamp_min(1.0)
    return 0.5 * (loss_x + loss_y)


def _structured_spill_loss(
    prediction_mask: torch.Tensor,
    teacher_mask: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    """Charge only structured opacity added outside the teacher silhouette."""

    if not (
        prediction_mask.shape == teacher_mask.shape == target_mask.shape
    ):
        raise ValueError("Structured spill masks must have identical shapes")
    background = 1.0 - target_mask.clamp(0.0, 1.0)
    excess = F.relu(prediction_mask - teacher_mask.detach())
    return (excess * background).sum() / background.sum().clamp_min(1.0)


def _bidirectional_mask_losses(
    prediction_mask: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    inside_alpha_target: float,
    outside_margin_px: int = 0,
    outside_reference_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return explicit mask-interior coverage and mask-exterior spill losses.

    Both terms are normalized by foreground area.  This makes a missing hair
    pixel and an equally opaque extra pixel comparable, instead of diluting
    exterior spill over the much larger image background.  When a frozen
    residual teacher is supplied, only opacity above that immutable baseline
    is charged; route-only calls omit the reference and remain absolute.  A
    small dilation may tolerate rasterization antialiasing at the boundary.
    """

    if prediction_mask.shape != target_mask.shape or prediction_mask.ndim != 2:
        raise ValueError("Prediction and target masks must be aligned HxW tensors")
    if not 0.0 <= float(inside_alpha_target) <= 1.0:
        raise ValueError("inside_alpha_target must be in [0, 1]")
    margin = int(outside_margin_px)
    if margin < 0:
        raise ValueError("outside_margin_px must be non-negative")
    target = target_mask.to(
        device=prediction_mask.device, dtype=prediction_mask.dtype
    ).clamp(0.0, 1.0)
    prediction = prediction_mask.clamp(0.0, 1.0)
    reference = prediction.new_zeros(prediction.shape)
    if outside_reference_mask is not None:
        if outside_reference_mask.shape != prediction_mask.shape:
            raise ValueError("Outside reference mask must match prediction")
        reference = outside_reference_mask.to(
            device=prediction.device, dtype=prediction.dtype
        ).detach().clamp(0.0, 1.0)
    foreground_area = target.sum().clamp_min(1.0)
    desired_alpha = target * float(inside_alpha_target)
    inside = (F.relu(desired_alpha - prediction) * target).sum()
    inside = inside / foreground_area

    allowed = target
    if margin > 0:
        kernel = 2 * margin + 1
        allowed = F.max_pool2d(
            target[None, None], kernel_size=kernel, stride=1, padding=margin
        )[0, 0]
    outside_excess = F.relu(prediction - reference)
    outside = (outside_excess * (1.0 - allowed)).sum() / foreground_area
    return inside, outside


def _maximum_hole_soft_loss(
    prediction_mask: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    kernel_size: int,
    topk_fraction: float,
) -> torch.Tensor:
    """Differentiable surrogate for the largest connected missing region.

    Exact connected-component labelling is discrete and cannot supervise the
    renderer.  Averaging the foreground deficit over a large sliding window
    suppresses isolated antialiasing errors and produces a high response only
    inside a spatially coherent hole.  The top-k pooled responses approximate
    the worst connected component while retaining gradients over an area,
    rather than through a single max pixel.  This term is always additive to
    the ordinary mean inside/outside mask objectives.
    """

    if prediction_mask.shape != target_mask.shape or prediction_mask.ndim != 2:
        raise ValueError("Prediction and target masks must be aligned HxW tensors")
    kernel = int(kernel_size)
    if kernel <= 0:
        raise ValueError("kernel_size must be positive")
    if kernel % 2 == 0:
        kernel += 1
    fraction = float(topk_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("topk_fraction must be in (0, 1]")
    target = target_mask.to(
        device=prediction_mask.device, dtype=prediction_mask.dtype
    ).clamp(0.0, 1.0)
    deficit = F.relu(target - prediction_mask.clamp(0.0, 1.0)) * target
    coherent = F.avg_pool2d(
        deficit[None, None], kernel, stride=1, padding=kernel // 2
    )[0, 0]
    foreground = target > 0.05
    values = coherent[foreground]
    if values.numel() == 0:
        return coherent.new_zeros(())
    count = max(int(math.ceil(fraction * int(values.numel()))), 1)
    return torch.topk(values, min(count, int(values.numel()))).values.mean()


def _rendered_route_spill_loss(
    route_mask: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    margin_px: int,
) -> torch.Tensor:
    """Penalize a route's rendered footprint outside a dilated hair mask."""

    if route_mask.shape != target_mask.shape:
        raise ValueError("Route and target masks must have identical shapes")
    target = target_mask.to(dtype=route_mask.dtype).clamp(0.0, 1.0)
    if margin_px > 0:
        kernel = 2 * int(margin_px) + 1
        target = F.max_pool2d(
            target[None, None], kernel_size=kernel, stride=1, padding=margin_px
        )[0, 0]
    background = 1.0 - target
    # Normalize by the route's total opacity as well as image area. A large
    # background must not make a small but entirely invalid Fin cheap.
    outside = (route_mask * background).sum()
    return outside / route_mask.sum().clamp_min(1e-6)


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
    *,
    strand_points: torch.Tensor | None = None,
) -> torch.Tensor:
    """Boot zero-opacity strands from projected foreground support.

    This only decides whether a strand is allowed to become visible.  The
    HairGS orientation render, RGB/mask objective, visual hull, and signed LOO
    evidence still decide its direction, geometry, and retained route mass.
    """

    if strand_points is None:
        strand_points = _strand_points_from_primitives(
            primitives,
            field.point_count,
            int(
                (primitives.route_id == ROUTE_NAMES.index("strand"))
                .sum()
                .item()
            )
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


def _shell_points_from_primitives(
    primitives, point_count: int, shell_samples: int
) -> torch.Tensor:
    shell = primitives.xyz[primitives.route_id == ROUTE_NAMES.index("shell")]
    expected = int(point_count) * int(shell_samples)
    if shell.shape[0] != expected:
        raise RuntimeError(
            f"Expected {expected} shell samples, found {shell.shape[0]}"
        )
    return shell.reshape(point_count, shell_samples, 3)


def _compute_visual_hull_gates(
    field: UnifiedFiberField,
    surface_vertices: list[torch.Tensor],
    surface_faces: torch.Tensor,
    cameras: list,
    ground_truth: list[dict[str, torch.Tensor]],
    cfg: PipelineConfig,
    temperature: float,
    geometry_blend: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Cull shell and strand samples unsupported by multi-view hair masks.

    Prefix connectivity is enforced root-to-tip: after the first rejected
    sample, more distal samples are rejected as well.  This avoids isolated
    floating tips and Fin segments even if a distal projection re-enters a
    mask. Shell gating is opt-in so legacy configurations remain unchanged.
    """

    if not surface_vertices or not (
        len(surface_vertices) == len(cameras) == len(ground_truth)
    ):
        raise ValueError("Visual-hull views, cameras, and masks must be aligned")
    shell_samples = int(cfg.fiber_shell_samples)
    strand_samples = int(cfg.fiber_strand_samples)
    shell_supported = field.route_logits.new_zeros(
        (field.point_count, shell_samples)
    )
    shell_valid_count = torch.zeros_like(shell_supported)
    strand_supported = field.route_logits.new_zeros(
        (field.point_count, strand_samples)
    )
    strand_valid_count = torch.zeros_like(strand_supported)
    supervise_shell = float(cfg.fiber_shell_visual_hull_weight) > 0.0
    occlusion_aware = bool(cfg.fiber_visual_hull_occlusion_aware)
    visible_shell_samples = 0.0
    visible_strand_samples = 0.0
    total_shell_samples = 0.0
    total_strand_samples = 0.0
    depth_tolerance = float(field.scene_scale.detach().cpu()) * float(
        cfg.fiber_visual_hull_occlusion_depth_scale
    )
    with torch.no_grad():
        for vertices, camera, target in zip(surface_vertices, cameras, ground_truth):
            if bool(cfg.fiber_visual_hull_target_geometry):
                shell_points = field.shell_target_geometry(
                    vertices,
                    surface_faces,
                    shell_samples=shell_samples,
                )
                strand_points, _directions = field.strand_target_geometry(
                    vertices,
                    surface_faces,
                    strand_samples=strand_samples,
                )
            else:
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
                    teacher_opacity_transfer=cfg.fiber_teacher_opacity_transfer,
                )
                shell_points = _shell_points_from_primitives(
                    primitives, field.point_count, shell_samples
                )
                strand_points = _strand_points_from_primitives(
                    primitives, field.point_count, strand_samples
                )
            shell_front_visible = torch.ones(
                shell_points.shape[:-1], dtype=torch.bool, device=shell_points.device
            )
            strand_front_visible = torch.ones(
                strand_points.shape[:-1], dtype=torch.bool, device=strand_points.device
            )
            if occlusion_aware:
                shell_flat = shell_points.reshape(-1, 3)
                strand_flat = strand_points.reshape(-1, 3)
                combined = torch.cat(
                    [vertices.detach().reshape(-1, 3), shell_flat, strand_flat],
                    dim=0,
                )
                combined_visible = _front_surface_visibility(
                    combined,
                    camera,
                    bin_px=int(cfg.fiber_visual_hull_occlusion_bin_px),
                    depth_tolerance=depth_tolerance,
                )
                start = int(vertices.numel() // 3)
                shell_count = int(shell_flat.shape[0])
                shell_front_visible = combined_visible[
                    start : start + shell_count
                ].reshape(shell_points.shape[:-1])
                strand_front_visible = combined_visible[
                    start + shell_count :
                ].reshape(strand_points.shape[:-1])
            visible_shell_samples += float(shell_front_visible.sum().cpu())
            visible_strand_samples += float(strand_front_visible.sum().cpu())
            total_shell_samples += float(shell_front_visible.numel())
            total_strand_samples += float(strand_front_visible.numel())
            if supervise_shell:
                shell_mask_support, shell_valid = _sample_mask_at_world_points(
                    shell_points,
                    camera,
                    target["mask"],
                    int(cfg.fiber_visual_hull_margin_px),
                )
                shell_valid = shell_valid & shell_front_visible
                shell_valid_float = shell_valid.to(shell_supported.dtype)
                shell_supported += (
                    (shell_mask_support >= 0.5).to(shell_supported.dtype)
                    * shell_valid_float
                )
                shell_valid_count += shell_valid_float
            strand_mask_support, strand_valid = _sample_mask_at_world_points(
                strand_points,
                camera,
                target["mask"],
                int(cfg.fiber_visual_hull_margin_px),
            )
            strand_valid = strand_valid & strand_front_visible
            strand_valid_float = strand_valid.to(strand_supported.dtype)
            strand_supported += (
                (strand_mask_support >= 0.5).to(strand_supported.dtype)
                * strand_valid_float
            )
            strand_valid_count += strand_valid_float
        strand_fraction = strand_supported / strand_valid_count.clamp_min(1.0)
        minimum_views = max(int(cfg.fiber_visual_hull_min_views), 1)
        strand_required_support = torch.full_like(
            strand_supported, float(minimum_views)
        )
        if occlusion_aware:
            # Occluded views are unknown.  Require agreement in every available
            # view up to min-k, while still rejecting samples never observed.
            strand_required_support = torch.minimum(
                strand_required_support, strand_valid_count
            ).clamp_min(1.0)
        strand_gate = (
            (strand_supported >= strand_required_support)
            & (strand_fraction >= float(cfg.fiber_visual_hull_min_fraction))
        ).to(strand_supported.dtype)
        strand_gate = torch.cumprod(strand_gate, dim=1)
        if supervise_shell:
            shell_fraction = shell_supported / shell_valid_count.clamp_min(1.0)
            shell_required_support = torch.full_like(
                shell_supported, float(minimum_views)
            )
            if occlusion_aware:
                shell_required_support = torch.minimum(
                    shell_required_support, shell_valid_count
                ).clamp_min(1.0)
            shell_gate = (
                (
                    shell_supported
                    >= shell_required_support
                )
                & (
                    shell_fraction
                    >= float(cfg.fiber_visual_hull_min_fraction)
                )
            ).to(shell_supported.dtype)
            shell_gate = torch.cumprod(shell_gate, dim=1)
        else:
            shell_fraction = torch.ones_like(shell_supported)
            shell_gate = torch.ones_like(shell_supported)
    report = {
        "kept_fraction": float(strand_gate.mean().cpu()),
        "fully_kept_strands": float(
            (strand_gate[:, -1] > 0.5).float().mean().cpu()
        ),
        "mean_support_fraction": float(strand_fraction.mean().cpu()),
        "shell_kept_fraction": float(shell_gate.mean().cpu()),
        "fully_kept_shells": float(
            (shell_gate[:, -1] > 0.5).float().mean().cpu()
        ),
        "shell_mean_support_fraction": float(shell_fraction.mean().cpu()),
        "shell_supervised": float(supervise_shell),
        "views": float(len(cameras)),
        "target_geometry": float(bool(cfg.fiber_visual_hull_target_geometry)),
        "occlusion_aware": float(occlusion_aware),
        "shell_front_visible_fraction": (
            visible_shell_samples / max(total_shell_samples, 1.0)
        ),
        "strand_front_visible_fraction": (
            visible_strand_samples / max(total_strand_samples, 1.0)
        ),
    }
    return shell_gate, strand_gate, report


def _visual_hull_soft_loss(
    strand_points: torch.Tensor,
    route_probabilities: torch.Tensor,
    camera,
    mask: torch.Tensor,
    margin_px: int,
    *,
    sample_gate: torch.Tensor | None = None,
    visibility_gate: torch.Tensor | None = None,
) -> torch.Tensor:
    return _route_visual_hull_soft_loss(
        strand_points,
        route_probabilities,
        ROUTE_NAMES.index("strand"),
        camera,
        mask,
        margin_px,
        sample_gate=sample_gate,
        visibility_gate=visibility_gate,
    )


def _route_visual_hull_soft_loss(
    points: torch.Tensor,
    route_probabilities: torch.Tensor,
    route_index: int,
    camera,
    mask: torch.Tensor,
    margin_px: int,
    *,
    sample_gate: torch.Tensor | None = None,
    visibility_gate: torch.Tensor | None = None,
) -> torch.Tensor:
    support, valid = _sample_mask_at_world_points(
        points, camera, mask, margin_px
    )
    route_mass = route_probabilities[:, int(route_index)][:, None]
    weights = route_mass.expand_as(support) * valid.to(route_mass.dtype)
    if visibility_gate is not None:
        if tuple(visibility_gate.shape) != tuple(support.shape):
            raise ValueError(
                "visibility_gate must match projected route points, got "
                f"{tuple(visibility_gate.shape)} != {tuple(support.shape)}"
            )
        weights = weights * visibility_gate.detach().to(weights.dtype)
    if sample_gate is not None:
        if tuple(sample_gate.shape) != tuple(support.shape):
            raise ValueError(
                "sample_gate must match projected route points, got "
                f"{tuple(sample_gate.shape)} != {tuple(support.shape)}"
            )
        # The persistent gate already encodes the multi-view OR/min-k rule.
        # Penalizing accepted samples in every individual camera silently
        # changes that rule to an AND over all cameras and collapses strands
        # that are merely back-facing or occluded in the current image.
        weights = weights * (1.0 - sample_gate.detach().to(weights.dtype))
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
                teacher_opacity_transfer=cfg.fiber_teacher_opacity_transfer,
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


def _scheduled_structure_detach_floor(
    completed_step: int,
    total_steps: int,
    cfg: PipelineConfig,
) -> float:
    """Anneal active structured routes away from teacher geometry."""

    final_gain = float(cfg.fiber_structure_detach_final_gain)
    if final_gain <= 0.0:
        return 0.0
    start = float(cfg.fiber_structure_detach_start_fraction)
    end = float(cfg.fiber_structure_detach_end_fraction)
    fraction = float(completed_step) / max(float(total_steps), 1.0)
    if end <= start:
        progress = 1.0 if fraction >= end else 0.0
    else:
        progress = (fraction - start) / (end - start)
    progress = min(max(progress, 0.0), 1.0)
    # Smoothstep avoids a derivative discontinuity in the geometry presented
    # to the renderer while still reaching an exact final floor.
    progress = progress * progress * (3.0 - 2.0 * progress)
    return min(max(final_gain * progress, 0.0), 1.0)


def _enforce_structured_deployment_floor(
    field: UnifiedFiberField,
    floor: float,
    *,
    min_support_fraction: float = 0.0,
) -> None:
    if float(floor) <= 0.0:
        return
    with torch.no_grad():
        active = field.route_active_gate[:, :2] > 0.5
        if float(min_support_fraction) > 0.0:
            active = active & _structured_support_mask(
                field, min_support_fraction
            )
        current = field.structured_delta_raw[:, :2]
        floor_tensor = torch.full_like(current, float(floor))
        current.copy_(torch.where(active, torch.maximum(current, floor_tensor), current))


def _structured_support_mask(
    field: UnifiedFiberField,
    min_support_fraction: float,
) -> torch.Tensor:
    threshold = min(max(float(min_support_fraction), 0.0), 1.0)
    if threshold <= 0.0:
        return torch.ones(
            (field.point_count, 2),
            dtype=torch.bool,
            device=field.route_logits.device,
        )
    masks = []
    for gate in (field.shell_visibility_gate, field.strand_visibility_gate):
        if gate.ndim == 2 and gate.shape[0] == field.point_count and gate.shape[1] > 0:
            masks.append(gate.mean(dim=1) >= threshold)
        else:
            masks.append(
                torch.ones(
                    field.point_count,
                    dtype=torch.bool,
                    device=field.route_logits.device,
                )
                if threshold <= 0.0
                else torch.zeros(
                    field.point_count,
                    dtype=torch.bool,
                    device=field.route_logits.device,
                )
            )
    return torch.stack(masks, dim=-1)


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
                "binding_mode": str(cfg.fiber_binding_mode),
                "source_mask_mode": (
                    "foreground"
                    if bool(cfg.fiber_split_fixed_base)
                    else str(cfg.fiber_source_mask_mode)
                ),
                "split_fixed_base": bool(cfg.fiber_split_fixed_base),
                "fixed_base_count": (
                    field.fixed_base.point_count
                    if field.fixed_base is not None
                    else 0
                ),
                "fixed_base_max_scale_fraction": float(
                    cfg.fiber_fixed_base_max_scale_fraction
                ),
                "source_mask_threshold": float(cfg.fiber_source_mask_threshold),
                "source_min_opacity": float(cfg.fiber_source_min_opacity),
                "residual_max_scale_fraction": float(
                    cfg.fiber_residual_max_scale_fraction
                ),
                "semantic_mask_from_source": bool(
                    cfg.fiber_semantic_mask_from_source
                ),
                "structured_foreground_only": bool(
                    cfg.fiber_structured_foreground_only
                ),
                "frame_indices": list(frame_indices),
                "shell_samples": cfg.fiber_shell_samples,
                "strand_samples": cfg.fiber_strand_samples,
                "shell_propagated_direction_weight": float(
                    cfg.fiber_shell_propagated_direction_weight
                ),
                "route_neighbor_k": int(cfg.fiber_route_neighbor_k),
                "surface_propagation_neighbor_k": int(
                    cfg.fiber_surface_propagation_neighbor_k
                ),
                "root_barycentric_max_delta": float(
                    cfg.fiber_root_barycentric_max_delta
                ),
                "expert_sh_max_delta": float(cfg.fiber_expert_sh_max_delta),
                "expert_sh_degree": int(cfg.fiber_expert_sh_degree),
                "additive_teacher_mode": bool(cfg.fiber_additive_teacher_mode),
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
        route_active_gate=cpu(field.route_active_gate),
        shell_visibility_gate=cpu(field.shell_visibility_gate),
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
    motion: DifferentiableSurfaceScaffold,
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
                        teacher_opacity_transfer=cfg.fiber_teacher_opacity_transfer,
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


def _measure_semantic_migration_render_equivalence(
    field: UnifiedFiberField,
    teacher_field: UnifiedFiberField,
    surface_vertices: torch.Tensor,
    surface_faces: torch.Tensor,
    camera,
    cfg: PipelineConfig,
    renderer_name: str,
) -> dict[str, object]:
    """Render one camera before any update and verify teacher equivalence."""

    with torch.no_grad():
        teacher_primitives = teacher_field.residual_primitives(
            surface_vertices, surface_faces
        )
        student_primitives = field.primitives(
            surface_vertices,
            surface_faces,
            shell_samples=int(cfg.fiber_shell_samples),
            strand_samples=int(cfg.fiber_strand_samples),
            temperature=float(cfg.fiber_final_temperature),
            route_blend=1.0,
            geometry_blend=1.0,
            route_hardening=(
                1.0
                if cfg.fiber_teacher_adaptive_migration_domain is not None
                else 0.0
            ),
            hard_route_policy=str(cfg.fiber_hard_route_policy),
            fin_aspect_ratio=float(cfg.fiber_fin_aspect_ratio),
            additive_teacher=bool(cfg.fiber_additive_teacher_mode),
            teacher_opacity_transfer=float(cfg.fiber_teacher_opacity_transfer),
        )
        teacher_prediction = _render(
            teacher_primitives, camera, cfg, renderer_name
        )
        student_prediction = _render(
            student_primitives, camera, cfg, renderer_name
        )
        rgb_error = torch.abs(
            student_prediction["rgb"] - teacher_prediction["rgb"]
        )
        mask_error = torch.abs(
            student_prediction["mask"] - teacher_prediction["mask"]
        )
        physical_mask_error = torch.abs(
            student_prediction.get("physical_mask", student_prediction["mask"])
            - teacher_prediction.get("physical_mask", teacher_prediction["mask"])
        )
        source_count = field.point_count
        active_route = torch.argmax(field.route_active_gate, dim=-1)
        source_index = torch.arange(source_count, device=active_route.device)
        selected_index = torch.where(
            active_route == ROUTE_NAMES.index("shell"),
            source_index * int(cfg.fiber_shell_samples),
            torch.where(
                active_route == ROUTE_NAMES.index("strand"),
                source_count * int(cfg.fiber_shell_samples)
                + source_index * int(cfg.fiber_strand_samples),
                source_count
                * (int(cfg.fiber_shell_samples) + int(cfg.fiber_strand_samples))
                + source_index,
            ),
        )
        selected_xyz_error = torch.abs(
            student_primitives.xyz[selected_index]
            - teacher_primitives.xyz[:source_count]
        )
        selected_scale_error = torch.abs(
            student_primitives.scaling[selected_index]
            - teacher_primitives.scaling[:source_count]
        )
        selected_opacity_error = torch.abs(
            student_primitives.opacity[selected_index]
            - teacher_primitives.opacity[:source_count]
        )
        report = {
            "rgb_mean_absolute_error": float(rgb_error.mean().cpu()),
            "rgb_max_absolute_error": float(rgb_error.max().cpu()),
            "mask_mean_absolute_error": float(mask_error.mean().cpu()),
            "mask_max_absolute_error": float(mask_error.max().cpu()),
            "physical_mask_mean_absolute_error": float(
                physical_mask_error.mean().cpu()
            ),
            "maximum_mean_absolute_error": float(
                torch.stack(
                    [
                        rgb_error.mean(),
                        mask_error.mean(),
                        physical_mask_error.mean(),
                    ]
                ).max().cpu()
            ),
            "teacher_active_primitives": int(
                (teacher_primitives.opacity > 1e-10).sum().cpu()
            ),
            "migrated_active_primitives": int(
                (student_primitives.opacity > 1e-10).sum().cpu()
            ),
            "structured_delta_max": float(
                field.structured_delta_gain.max().detach().cpu()
            ),
            "selected_xyz_max_error": float(selected_xyz_error.max().cpu()),
            "selected_xyz_mean_error": float(selected_xyz_error.mean().cpu()),
            "selected_scale_max_error": float(selected_scale_error.max().cpu()),
            "selected_opacity_max_error": float(selected_opacity_error.max().cpu()),
        }
    if field.route_active_gate.shape[0] > 0:
        capacity_per_source = (field.route_active_gate > 0.5).sum(dim=-1)
        forward_per_source = (
            student_primitives.route_probabilities > 0.5
        ).sum(dim=-1)
        report["single_active_route_per_source"] = bool(
            torch.all(forward_per_source == 1).detach().cpu()
        )
        report["single_active_capacity_route_per_source"] = bool(
            torch.all(capacity_per_source == 1).detach().cpu()
        )
    return report


def _load_orientation_targets(
    frame_paths: list[Path],
    frame_indices: list[int],
    width: int,
    height: int,
    device: str,
    orientation_dir: str | None,
    *,
    distribution_radius: int = 0,
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
        vectors = F.normalize(vectors, dim=-1, eps=1e-8)
        moment2, moment4, local_confidence = _local_axial_orientation_moments(
            vectors,
            confidence_tensor,
            radius=int(distribution_radius),
        )
        targets.append(
            {
                "vectors": vectors.to(device),
                "confidence": confidence_tensor.to(device),
                "moment2": moment2.to(device),
                "moment4": moment4.to(device),
                "distribution_confidence": local_confidence.to(device),
            }
        )
    return targets


def _local_axial_orientation_moments(
    vectors: torch.Tensor,
    confidence: torch.Tensor,
    *,
    radius: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build local second/fourth moments of a pi-periodic direction field.

    The fourth moment stays strong for two orthogonal modes even when their
    second moments cancel.  This gives curly/crossing regions an explicit
    multi-modal target while coherent regions retain their original tangent.
    """

    if vectors.ndim != 3 or vectors.shape[-1] != 2:
        raise ValueError("vectors must have shape [H, W, 2]")
    if tuple(confidence.shape) != tuple(vectors.shape[:2]):
        raise ValueError("confidence must match the orientation image")
    window_radius = int(radius)
    if window_radius < 0:
        raise ValueError("radius must be non-negative")
    moment2 = vectors
    moment4 = torch.stack(
        [
            vectors[..., 0].square() - vectors[..., 1].square(),
            2.0 * vectors[..., 0] * vectors[..., 1],
        ],
        dim=-1,
    )
    if window_radius == 0:
        return moment2, moment4, confidence
    kernel = 2 * window_radius + 1
    weights = confidence.clamp_min(0.0)[None, None]

    def weighted_pool(values: torch.Tensor) -> torch.Tensor:
        channels = values.permute(2, 0, 1)[None]
        numerator = F.avg_pool2d(
            channels * weights,
            kernel_size=kernel,
            stride=1,
            padding=window_radius,
        )
        denominator = F.avg_pool2d(
            weights,
            kernel_size=kernel,
            stride=1,
            padding=window_radius,
        ).clamp_min(1e-8)
        return (numerator / denominator).squeeze(0).permute(1, 2, 0)

    local_confidence = F.avg_pool2d(
        weights,
        kernel_size=kernel,
        stride=1,
        padding=window_radius,
    )[0, 0]
    return weighted_pool(moment2), weighted_pool(moment4), local_confidence


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


def _orientation_distribution_loss(
    predicted2: torch.Tensor,
    predicted4: torch.Tensor,
    target: dict[str, torch.Tensor],
    foreground_mask: torch.Tensor,
    physical_mask: torch.Tensor,
) -> torch.Tensor:
    """Match local axial distributions rather than one tangent per pixel."""

    if "moment2" not in target or "moment4" not in target:
        raise ValueError("Orientation target is missing axial distribution moments")
    alpha = physical_mask.to(predicted2.dtype).clamp_min(1e-4)[..., None]
    # Rasterized signed moment colors are premultiplied by opacity.  Dividing
    # by the physical alpha approximates the transmittance-weighted mixture.
    rendered2 = (predicted2[..., :2] / alpha).clamp(-1.0, 1.0)
    rendered4 = (predicted4[..., :2] / alpha).clamp(-1.0, 1.0)
    discrepancy = F.smooth_l1_loss(
        rendered2, target["moment2"], reduction="none"
    ).sum(dim=-1)
    discrepancy = discrepancy + F.smooth_l1_loss(
        rendered4, target["moment4"], reduction="none"
    ).sum(dim=-1)
    weights = (
        target.get("distribution_confidence", target["confidence"])
        * foreground_mask.to(target["confidence"].dtype)
        * (physical_mask > 1e-4).to(target["confidence"].dtype)
    )
    return (discrepancy * weights).sum() / weights.sum().clamp_min(1e-8)
