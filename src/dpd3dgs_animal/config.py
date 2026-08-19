from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class PipelineConfig:
    mask_loss_weight: float = 10.0
    # Class-balanced foreground/background alpha supervision.  This is
    # especially important for sparse hair silhouettes where a global mask
    # mean otherwise rewards an almost-transparent solution.
    mask_balance_weight: float = 0.0
    mask_boundary_weight: float = 0.0
    mask_boundary_radius: int = 1
    color_loss_weight: float = 1.0
    max_render_points: int = 120000
    device: str = "cuda"
    optimize_lr: float = 1e-3
    optimize_steps: int = 200
    render_sigma_px: float = 1.5
    render_radius_px: int = 4
    # Optimization-first unified fur/hair representation.
    # ``residual_only`` is the controlled skinned-3DGS baseline: one
    # anisotropic Gaussian per source point, with no shell/strand experts or
    # learned routing. ``unified`` enables the full mixture.
    fiber_representation: str = "unified"
    # Capacity may remain fixed for already well-initialized 3DGS, or scale
    # with the raster area for weak geometry-only seeds.  ``fiber_max_points``
    # is always a hard ceiling, so the adaptive policy cannot silently exceed
    # the memory budget recorded by an experiment config.
    fiber_capacity_mode: str = "fixed"
    fiber_min_points: int = 0
    fiber_max_points: int = 20000
    fiber_target_pixels_per_point: float = 8.0
    # PLY order is often topology/optimizer order rather than spatial order.
    # Morton-systematic sampling preserves 3D coverage when a cloud must be
    # truncated, while ``uniform_index`` reproduces previous checkpoints.
    fiber_point_sampling_mode: str = "uniform_index"
    fiber_exact_vertex_binding: bool = False
    # A trained Stage-I GS is not vertex-aligned. ``nearest_vertex`` keeps
    # every learned world-space Gaussian exact through its local offset while
    # assigning a stable carrier frame without an O(points x faces) search.
    fiber_binding_mode: str = "closest_surface"
    # HairGS stores a learned per-Gaussian foreground logit in ``mask``.
    # Filtering before routing prevents head/body splats from entering the
    # hair scaffold; the complementary background split remains the fixed
    # non-hair base used by the final compositor.
    fiber_source_mask_mode: str = "all"
    fiber_source_mask_threshold: float = 0.25
    fiber_source_min_opacity: float = 0.0
    # Physically remove head/body rows from the learnable fiber field. Their
    # filtered Stage-I GS remain an immutable depth/RGB compositor module and
    # are appended only immediately before rasterization.
    fiber_split_fixed_base: bool = False
    # Optional safety cap for a Stage-I residual teacher.  HairGS can hide a
    # few very large, opaque anisotropic splats in the fitted cameras; those
    # become broad ellipses from held-out views.  Zero preserves the source
    # covariance exactly.  Positive values cap every residual scale axis as a
    # fraction of the rest-surface scene diagonal.
    fiber_residual_max_scale_fraction: float = 0.0
    # Independent cap for the immutable head/body compositor.  Keeping this
    # separate avoids thinning valid hair splats merely to remove a few large
    # Stage-I head/background covariance outliers.
    fiber_fixed_base_max_scale_fraction: float = 0.0
    # Preserve the complete opaque Stage-I head/body compositor but render a
    # separate HairGS-style binary semantic alpha for hair supervision.
    fiber_semantic_mask_from_source: bool = False
    # Only semantic-foreground sources may enter shell/strand. Background
    # head/body sources remain a fixed residual occlusion and color base.
    fiber_structured_foreground_only: bool = False
    fiber_default_opacity: float = 0.5
    fiber_default_opacity_reference_points: int = 0
    fiber_binding_cache: str | None = None
    fiber_max_frames: int = 8
    fiber_renderer: str = "torch"
    fiber_shell_samples: int = 2
    fiber_strand_samples: int = 5
    fiber_warmup_steps: int = 50
    fiber_initial_temperature: float = 2.0
    fiber_final_temperature: float = 0.35
    fiber_sigma_scale: float = 1.0
    fiber_appearance_lr_scale: float = 0.25
    # Small, regularized route-specific appearance offsets let shell, strand,
    # and residual experts specialize without tripling the base color state.
    fiber_expert_appearance_weight: float = 0.0
    fiber_geometry_lr_scale: float = 0.5
    fiber_structure_activation_lr_scale: float = 1.0
    # A render-preserving semantic migration starts every structured source as
    # a co-located copy of its residual teacher.  This regularizer then asks
    # the selected shell/strand routes to acquire a real geometric extent;
    # otherwise a nominally structured checkpoint could remain a disguised
    # collection of collapsed residual Gaussians.
    fiber_structure_deployment_weight: float = 0.0
    fiber_structure_min_deployment_gain: float = 0.0
    fiber_route_lr_scale: float = 1.0
    fiber_initial_residual_trust: float = 0.95
    # Optional geometry-only initialization for sparse neutral scalp/head
    # seeds.  Values are relative to the rest-surface scene scale, so they
    # transfer across subjects without consuming strand annotations.
    fiber_initial_shell_length_scale: float | None = None
    fiber_initial_strand_length_scale: float | None = None
    fiber_initialize_direction_from_normal: bool = False
    # UnityFurURP-inspired silhouette fins inside the shell expert.  The
    # geometry is a thin anisotropic Gaussian ribbon and its opacity becomes
    # view-selective only as the zero-initialized structured delta unfolds.
    # A strength of zero is exactly backward compatible.
    fiber_fin_gate_strength: float = 0.0
    fiber_fin_grazing_threshold: float = 0.25
    fiber_fin_grazing_softness: float = 0.05
    fiber_fin_aspect_ratio: float = 1.0
    # Shell-only supervision in a narrow ground-truth silhouette band.  This
    # teaches fins their actual job instead of asking them to repaint the full
    # residual image whenever the teacher route is dropped.
    fiber_fin_silhouette_weight: float = 0.0
    fiber_fin_silhouette_radius: int = 3
    fiber_strand_support_weight: float = 0.0
    # Unlike the legacy Fin band loss, these terms supervise the complete
    # shell expert. The visual-hull term persists a multi-view hard gate in
    # the checkpoint, while render spill charges the actual Fin footprint.
    fiber_shell_visual_hull_weight: float = 0.0
    fiber_shell_render_spill_weight: float = 0.0
    # Hair detail needs an image-gradient objective; RGB L1 alone prefers a
    # smooth average.  In additive-teacher mode, structured opacity outside
    # the foreground is also charged only for the excess over the frozen
    # residual teacher, so shell/strand cannot worsen an existing halo.
    fiber_rgb_gradient_weight: float = 0.0
    fiber_structured_spill_weight: float = 0.0
    # Explicit two-sided silhouette supervision on the fully deployed render.
    # The ordinary image mask loss is dominated by the large background and
    # can be satisfied by lowering structured opacity.  These terms normalize
    # both missing foreground and excess background by the hair-mask area.
    fiber_mask_inside_coverage_weight: float = 0.0
    fiber_mask_outside_spill_weight: float = 0.0
    fiber_mask_inside_alpha_target: float = 0.9
    fiber_mask_outside_margin_px: int = 0
    # Apply the same contract to shell+strand with residual removed.  A modest
    # alpha floor forces useful structural ownership instead of allowing the
    # residual teacher to hide empty explicit routes.
    fiber_structured_mask_inside_coverage_weight: float = 0.0
    fiber_structured_mask_outside_spill_weight: float = 0.0
    fiber_structured_mask_inside_alpha_target: float = 0.0
    # Geometry-level coverage initialization.  A frozen residual teacher is
    # rendered in every calibrated view, candidate normal-grown segments are
    # intersected with the multi-view hair masks, and only roots that explain
    # teacher under-coverage are activated as shell/strand carriers.  A count
    # of zero preserves the legacy zero-initialized structured field exactly.
    fiber_coverage_seed_count: int = 0
    fiber_coverage_seed_samples: int = 7
    fiber_coverage_seed_min_views: int = 3
    fiber_coverage_seed_min_fraction: float = 0.25
    fiber_coverage_seed_min_deficit: float = 0.01
    fiber_coverage_seed_voxel_scale: float = 0.01
    fiber_coverage_seed_visibility_cull: bool = False
    fiber_coverage_seed_visibility_bin_px: int = 2
    fiber_coverage_seed_visibility_depth_scale: float = 0.01
    fiber_coverage_seed_structured_opacity: float = 0.25
    fiber_coverage_seed_geometry_gain: float = 1.0
    fiber_coverage_seed_shell_length_scale: float = 0.012
    fiber_coverage_seed_strand_length_scale: float = 0.08
    fiber_coverage_seed_orientation_init: bool = False
    fiber_coverage_seed_orientation_normal_bias: float = 0.15
    fiber_coverage_seed_orientation_confidence_floor: float = 0.0
    fiber_coverage_seed_route_mass: list[float] = field(
        default_factory=lambda: [0.15, 0.75, 0.10]
    )
    # Route-aware adaptive density control.  The field preallocates a
    # surface-bound candidate pool, while this controller changes the set of
    # *active* renderer primitives.  Residual outliers are disabled, mask
    # deficits activate new shell/strand groups, and silhouette detail
    # activates narrower Fin groups.  Keeping tensor shapes fixed makes the
    # topology updates safe for Adam and checkpoint loading; zero-gated
    # primitives are culled before rasterization.
    fiber_topology_update_every: int = 0
    fiber_topology_start_step: int = 0
    fiber_topology_stop_step: int = 0
    fiber_topology_prune_count: int = 0
    fiber_topology_grow_count: int = 0
    fiber_topology_densify_count: int = 0
    fiber_topology_min_views: int = 3
    fiber_topology_prune_max_support: float = 0.10
    fiber_topology_prune_footprint_sigma: float = 2.0
    fiber_topology_grow_min_support: float = 0.50
    fiber_topology_grow_min_deficit: float = 0.015
    fiber_topology_boundary_radius: int = 2
    fiber_topology_detail_radius_scale: float = 0.65
    fiber_topology_max_residual_prune_fraction: float = 0.12
    fiber_topology_initial_structured_off: bool = False
    # Validate every discrete topology event independently on multiple views.
    # An event is kept only when no view regresses beyond the margin and the
    # mean validation loss does not increase.
    fiber_topology_validate_events: bool = False
    fiber_topology_validation_margin: float = 0.0
    # Incremental births admit a nearly render-equivalent strand capacity and
    # let ordinary optimization unfold it afterwards.  This replaces the old
    # full-length, full-opacity route switch that was rejected as one batch.
    fiber_topology_incremental_birth: bool = False
    fiber_topology_birth_strand_mass: float = 0.01
    fiber_topology_birth_initial_delta: float = 0.0
    # A deficit birth is a surface-anchored 3D proposal.  Require the same
    # proposed strand volume to intersect missing foreground in several
    # calibrated views rather than reacting to an isolated image-space hole.
    fiber_topology_deficit_min_views: int = 2
    # Auxiliary supervision on the exact fully-deployed inference geometry.
    # This closes the continuation-schedule gap where an early checkpoint is
    # safe while partially blended during training but spills when evaluated
    # with geometry_blend=1.
    fiber_deployment_render_weight: float = 0.0
    # Hair growth is optimized on the uncollapsed analytic target rather than
    # the residual-blended render proxy.  This closes a loophole where a
    # zero-gain strand trivially passes every visual-hull test.
    fiber_visual_hull_target_geometry: bool = False
    # A shared, scalp-local orientation field and deployment constraints turn
    # independent short cubic splats into coherent, usable strand carriers.
    # All length values are normalized by the calibrated scene scale.
    fiber_strand_field_weight: float = 0.0
    fiber_strand_deployability_weight: float = 0.0
    fiber_strand_min_deployment_gain: float = 0.0
    fiber_strand_min_deployed_length_scale: float = 0.0
    fiber_strand_coverage_weight: float = 0.0
    fiber_strand_coverage_target: float = 0.0
    # Optional image-derived, confidence-weighted 2D orientation supervision.
    # This accepts the same Gabor fields used by HairGS, never 3D strand GT.
    fiber_orientation_dir: str | None = None
    fiber_orientation_weight: float = 0.0
    # Preserve crossing/curly hair evidence with local axial orientation
    # moments.  The second harmonic represents the usual pi-periodic tangent;
    # the fourth harmonic distinguishes a crossing distribution from an
    # uncertain single direction without requiring strand ground truth.
    fiber_orientation_distribution_weight: float = 0.0
    fiber_orientation_distribution_radius: int = 3
    # Optional effective [shell, strand, residual] routing mass assigned when
    # a residual-only checkpoint bootstraps a unified field.  This explicitly
    # decouples a conservative residual photometric scaffold from the initial
    # structural routing prior.
    fiber_bootstrap_route_mass: list[float] | None = None
    # Optional hard source allocation [shell, strand, residual].  The
    # residual teacher is used only as a photometric target: source Gaussians
    # are deterministically assigned to exactly one deployable family while
    # shell/strand geometry remains a zero-initialized, render-equivalent copy.
    # This makes the residual fraction a true capacity ceiling rather than a
    # soft loss that calibration can silently violate.
    fiber_teacher_semantic_migration_mass: list[float] | None = None
    # Residual-free alternative for clean hair/fur scaffolds.  ``hair`` adds
    # a strand prior, ``fur`` adds a shell prior, and ``auto`` uses only local
    # Gaussian/surface evidence.  These are log-odds priors, not fixed global
    # quotas: the learned per-source router remains free to choose either
    # structural family and residual is unavailable at deployment.
    fiber_teacher_adaptive_migration_domain: str | None = None
    fiber_teacher_adaptive_migration_bias: float = 0.65
    # Residual-free migration can either enforce one family per source with a
    # straight-through hard router, or learn a continuous shell/strand source
    # distribution.  In both modes every emitted primitive remains explicitly
    # typed and residual capacity stays disabled.
    fiber_adaptive_migration_hard_router: bool = True
    # Abort before training when the initial semantic migration changes the
    # teacher render by more than this mean absolute RGB/mask tolerance.  Zero
    # disables the runtime check (unit-level primitive checks still apply).
    fiber_teacher_semantic_migration_tolerance: float = 0.0
    # Keep teacher position/covariance fixed as the calibration anchor while
    # allowing the structured student to refine shared color/opacity.  Route-
    # specific DC color deltas and analytic length/radius/orientation remain
    # trainable as before.
    fiber_optimize_structured_base_appearance: bool = False
    fiber_residual_trust_weight: float = 0.0
    fiber_gradient_clip: float = 10.0
    fiber_log_every: int = 10
    fiber_checkpoint_every: int = 250
    fiber_ema_decay: float = 0.95
    fiber_route_continuation: bool = True
    fiber_route_hardening: bool = True
    fiber_hard_route_policy: str = "argmax"
    fiber_route_neighbor_k: int = 0
    fiber_route_neighbor_weight: float = 0.0
    # Complete sparse image observations into a surface-attached structural
    # field before photometric optimization. Duplicate source anchors become
    # candidate slots on visual-hull-supported faces; observed axial
    # directions are then parallel-transported over the root graph. This is
    # initialization/completion rather than a new residual rendering route.
    fiber_surface_propagation_enabled: bool = False
    fiber_surface_propagation_reassign_fraction: float = 0.0
    fiber_surface_propagation_min_views: int = 3
    fiber_surface_propagation_min_fraction: float = 0.5
    fiber_surface_propagation_margin_px: int = 2
    fiber_surface_propagation_neighbor_k: int = 8
    fiber_surface_propagation_steps: int = 12
    fiber_surface_propagation_observation_weight: float = 0.85
    fiber_surface_propagation_normal_bias: float = 0.10
    fiber_surface_propagation_confidence_floor: float = 0.05
    # Shell and strand share root-flow evidence but consume it differently.
    # 0 keeps shells surface-normal; 1 lets them follow the propagated flow.
    # Strands always use the completed direction field.
    fiber_shell_propagated_direction_weight: float = 1.0
    # Roots remain on their owning surface triangle, but can refine their
    # barycentric location after initialization. This exposes root placement
    # to the renderer without allowing unconstrained 3D drift.
    fiber_root_barycentric_lr_scale: float = 0.05
    fiber_root_barycentric_max_delta: float = 0.20
    fiber_root_barycentric_weight: float = 0.001
    # Hair-only scalp occupancy is reconstructed on the surface from eroded,
    # multi-view hair masks. It gates strand births; shell/Fur remains free to
    # cover the complete body surface.
    fiber_scalp_occupancy_enabled: bool = False
    fiber_scalp_occupancy_erosion_px: int = 8
    fiber_scalp_occupancy_min_views: int = 3
    fiber_scalp_occupancy_min_fraction: float = 0.60
    fiber_scalp_initial_strand_fraction: float = 0.70
    # Decouple Hair root placement from the lumpy Stage-I GS density by
    # rebinding strand sources to an area-stratified supported scalp atlas.
    fiber_scalp_atlas_enabled: bool = False
    fiber_scalp_atlas_min_roots_per_face: int = 1
    # Image orientation is axial: d and -d are observationally equivalent.
    # Resolve polarity before curves are deployed or visual-hull culled.
    fiber_strand_outward_sign_enabled: bool = False
    fiber_strand_outward_anchor_threshold: float = 0.05
    fiber_strand_sign_sync_steps: int = 8
    # Higher-order view-dependent appearance is route-specific. The source
    # SH stays the teacher prior and only a bounded delta is optimized.
    fiber_expert_sh_lr_scale: float = 0.05
    fiber_expert_sh_weight: float = 0.0001
    fiber_expert_sh_max_delta: float = 0.50
    # Minimum SH degree used by the route experts.  DC-only Stage-I PLYs are
    # zero-padded so shell/strand can still learn view-dependent appearance.
    fiber_expert_sh_degree: int = 0
    fiber_route_dropout_probability: float = 0.0
    # Keep expert-removal pressure early, then optionally anneal it away so
    # the final hard deployment receives uninterrupted joint refinement.
    fiber_route_dropout_final_fraction: float = 1.0
    fiber_route_dropout_residual_bias: float = 1.0 / 3.0
    # Optional aggregate [shell, strand, residual] route-mass floor for
    # contribution-aware calibration.  Remaining mass is assigned from
    # multi-view leave-one-route-out evidence.
    fiber_route_minimum_mass: list[float] | None = None
    fiber_route_prior_final_fraction: float = 0.0
    fiber_calibration_frames: int = 0
    fiber_risk_calibration_every: int = 0
    # Do not estimate leave-one-route-out risk while structured primitives are
    # still blended out of the renderer; such estimates are not identifiable.
    fiber_risk_calibration_start_geometry_blend: float = 0.0
    fiber_risk_calibration_weight: float = 0.0
    fiber_risk_calibration_ema: float = 0.8
    fiber_risk_target_prior_blend: float = 0.0
    fiber_risk_floor: float = 1e-4
    # Residual-only teacher and deployment-safety calibration.  When enabled,
    # the bootstrapped residual appearance/covariance/position are frozen and
    # only zero-initialized structured increments plus routing are optimized.
    fiber_freeze_residual_teacher: bool = False
    # Preserve the complete residual teacher and learn shell/strand as
    # zero-opacity structured increments.  This is the safe mode for Fin:
    # closing a view-conditioned fin never removes teacher opacity.
    fiber_additive_teacher_mode: bool = False
    # Fraction of structured opacity that is taken from the co-located
    # residual source instead of being added on top.  Zero preserves the
    # historical fully-additive teacher; one conserves the source opacity
    # budget and lets deployed shell/strand replace a blurred residual splat.
    fiber_teacher_opacity_transfer: float = 0.0
    fiber_teacher_nonregression_weight: float = 0.0
    fiber_teacher_nonregression_margin: float = 0.0
    fiber_teacher_nonregression_every: int = 1
    # Evaluate several withheld cameras in the same update and aggregate
    # their non-regression violations.  ``max`` is a robust worst-view gate;
    # the default one-view mean preserves the historical behavior.
    fiber_teacher_nonregression_views_per_step: int = 1
    fiber_teacher_nonregression_reduction: str = "mean"
    # LOO evidence is signed: positive ablation damage earns route mass;
    # negative contribution (ablation improves the held-out result) is
    # directly charged against that route's current aggregate mass.
    fiber_negative_contribution_weight: float = 0.0
    # Multi-view hair-mask visual hull.  The hard gate removes unsupported
    # strand samples; the soft loss supplies a geometric gradient before the
    # next gate refresh.
    fiber_visual_hull_weight: float = 0.0
    fiber_visual_hull_update_every: int = 0
    fiber_visual_hull_min_views: int = 2
    fiber_visual_hull_min_fraction: float = 0.15
    fiber_visual_hull_margin_px: int = 3
    # Mask disagreement is evidence only for front-visible samples.  Hidden
    # shell/strand samples are unknown, not negatives; a coarse point z-buffer
    # prevents back-side geometry from being pruned through the foreground.
    fiber_visual_hull_occlusion_aware: bool = False
    fiber_visual_hull_occlusion_bin_px: int = 2
    fiber_visual_hull_occlusion_depth_scale: float = 0.01
    fiber_random_seed: int = 20260809
    fiber_route_entropy_weight: float = 1e-3
    fiber_route_prior_weight: float = 2e-2
    fiber_shell_normal_weight: float = 2e-2
    fiber_shell_length_weight: float = 1e-3
    fiber_strand_thinness_weight: float = 1e-2
    fiber_height_weight: float = 1e-3
    fiber_bend_weight: float = 1e-3
    fiber_residual_drift_weight: float = 1e-2
    # Rendering ownership is not deformation ownership.  These terms learn a
    # separate surface/shell/strand carrier for every source Gaussian, so a
    # photometric residual can still move with the simulated asset.
    fiber_carrier_entropy_weight: float = 0.0
    fiber_carrier_prior_weight: float = 0.0
    fiber_carrier_neighbor_weight: float = 0.0
    fiber_carrier_tip_neighbor_weight: float = 0.0
    fiber_carrier_attachment_weight: float = 0.0
    fiber_carrier_tip_prior_weight: float = 0.0
    fiber_carrier_family_alignment_weight: float = 0.0
    fiber_carrier_structure_floor_weight: float = 0.0


def load_config(path: str | Path | None = None) -> PipelineConfig:
    """Load a strict UniFur YAML configuration.

    Silent acceptance of stale SAM3D/Mocap/FEM keys made historical configs
    look valid even though the fiber pipeline ignored them.  The public
    project now rejects unknown keys so obsolete experiment files fail early.
    """
    cfg = PipelineConfig()
    if path is None:
        return cfg

    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")

    valid = {item.name for item in fields(PipelineConfig)}
    unknown = sorted(set(raw) - valid)
    if unknown:
        raise ValueError(
            f"Unknown UniFur configuration keys in {path}: {', '.join(unknown)}"
        )
    for key, value in raw.items():
        setattr(cfg, key, value)
    return cfg
