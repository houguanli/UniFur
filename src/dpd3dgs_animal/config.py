from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass
class ExternalProjectPaths:
    sam3d_root: Path = field(default_factory=lambda: project_root() / "third_party/sam3d_objects")
    elastic_root: Path = field(default_factory=lambda: project_root() / "third_party/elastic_simulator")
    mocap_root: Path = field(default_factory=lambda: project_root() / "third_party/mocap_anything")
    sam3d_checkpoints: Path = field(default_factory=lambda: project_root() / "checkpoints/sam3d")
    mocap_checkpoints: Path = field(default_factory=lambda: project_root() / "checkpoints/mocap_anything")
    mocap_zoo: Path = field(default_factory=lambda: project_root() / "samples/mocap_anything/zoo")


@dataclass
class PipelineConfig:
    paths: ExternalProjectPaths = field(default_factory=ExternalProjectPaths)
    conda_env: str = "dpd3dgs-animal"
    canonical_frame: str = "x_right_y_up_z_forward"
    mocap_axis_transform: str = "swap_yz"
    skeleton_scale_mode: str = "height"
    skeleton_center_mode: str = "root"
    skeleton_fit_padding: float = 1.0
    sam3d_tag: str = "hf"
    reference_frame: int = 0
    mocap_ref_seq: str = "Dog#Dog-Galloping/y30"
    mocap_ref_idx: int = 0
    mask_loss_weight: float = 10.0
    mask_boundary_weight: float = 0.0
    mask_boundary_radius: int = 1
    color_loss_weight: float = 1.0
    max_render_points: int = 120000
    device: str = "cuda"
    optimize_lr: float = 1e-3
    optimize_steps: int = 200
    elastic_edge_weight: float = 1e-2
    elastic_volume_weight: float = 1e-2
    bone_length_weight: float = 1e-1
    temporal_weight: float = 1e-2
    mocap_prior_weight: float = 1e-2
    render_sigma_px: float = 1.5
    render_radius_px: int = 4
    gravity: float = 0.0
    camera_align_to_frame0: bool = True
    camera_alignment_points: int = 12000
    camera_alignment_padding: float = 0.92
    gaussian_binding_k: int = 8
    pull_gaussians_to_surface: bool = True
    tet_mesh_max_faces: int = 20000
    skinning_weight_mode: str = "tet_geodesic"
    skinning_weight_k: int = 4
    skinning_deformation_mode: str = "dqs"
    elastic_fem_dt: float = 5e-3
    elastic_fem_substeps: int = 1
    elastic_fem_youngs_modulus: float = 3e5
    elastic_fem_poisson_ratio: float = 0.4
    elastic_fem_density: float = 1000.0
    elastic_fem_damping: float = 0.0
    elastic_fem_radius_scale: float = 0.10
    elastic_fem_min_radius_scale: float = 0.012
    elastic_fem_max_radius_scale: float = 0.08
    elastic_fem_min_bone_length_scale: float = 0.015
    elastic_fem_reset_each_frame: bool = True
    fem_cg_iters: int = 32
    fem_elastic_stiffness: float = 1.0
    fem_handle_stiffness: float = 80.0
    fem_diagonal_reg: float = 1e-4
    fem_handle_support: float = 3.0
    fem_handle_weight_power: float = 1.0
    # Optimization-first unified fur/hair representation.
    # ``residual_only`` is the controlled skinned-3DGS baseline: one
    # anisotropic Gaussian per source point, with no shell/strand experts or
    # learned routing. ``unified`` enables the full mixture.
    fiber_representation: str = "unified"
    fiber_max_points: int = 20000
    fiber_max_frames: int = 8
    fiber_renderer: str = "torch"
    fiber_shell_samples: int = 2
    fiber_strand_samples: int = 5
    fiber_warmup_steps: int = 50
    fiber_initial_temperature: float = 2.0
    fiber_final_temperature: float = 0.35
    fiber_sigma_scale: float = 1.0
    fiber_appearance_lr_scale: float = 0.25
    fiber_geometry_lr_scale: float = 0.5
    fiber_route_lr_scale: float = 1.0
    fiber_initial_residual_trust: float = 0.95
    # Optional effective [shell, strand, residual] routing mass assigned when
    # a residual-only checkpoint bootstraps a unified field.  This explicitly
    # decouples a conservative residual photometric scaffold from the initial
    # structural routing prior.
    fiber_bootstrap_route_mass: list[float] | None = None
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
    fiber_route_dropout_probability: float = 0.0
    fiber_route_dropout_residual_bias: float = 1.0 / 3.0
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
    fiber_random_seed: int = 20260809
    fiber_route_entropy_weight: float = 1e-3
    fiber_route_prior_weight: float = 2e-2
    fiber_shell_normal_weight: float = 2e-2
    fiber_shell_length_weight: float = 1e-3
    fiber_strand_thinness_weight: float = 1e-2
    fiber_height_weight: float = 1e-3
    fiber_bend_weight: float = 1e-3
    fiber_residual_drift_weight: float = 1e-2


def _coerce_path(value: Any) -> Path:
    return value if isinstance(value, Path) else Path(str(value)).expanduser()


def _config_path(value: Any, base: Path) -> Path:
    path = _coerce_path(value)
    return path if path.is_absolute() else (base / path).resolve()


def load_config(path: str | Path | None = None) -> PipelineConfig:
    cfg = PipelineConfig()
    if path is None:
        return cfg

    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    paths = raw.get("paths", {})
    base = project_root()
    cfg.paths = ExternalProjectPaths(
        sam3d_root=_config_path(paths.get("sam3d_root", cfg.paths.sam3d_root), base),
        elastic_root=_config_path(paths.get("elastic_root", cfg.paths.elastic_root), base),
        mocap_root=_config_path(paths.get("mocap_root", cfg.paths.mocap_root), base),
        sam3d_checkpoints=_config_path(paths.get("sam3d_checkpoints", cfg.paths.sam3d_checkpoints), base),
        mocap_checkpoints=_config_path(paths.get("mocap_checkpoints", cfg.paths.mocap_checkpoints), base),
        mocap_zoo=_config_path(paths.get("mocap_zoo", cfg.paths.mocap_zoo), base),
    )
    for key in (
        "conda_env",
        "canonical_frame",
        "mocap_axis_transform",
        "skeleton_scale_mode",
        "skeleton_center_mode",
        "skeleton_fit_padding",
        "sam3d_tag",
        "reference_frame",
        "mocap_ref_seq",
        "mocap_ref_idx",
        "mask_loss_weight",
        "mask_boundary_weight",
        "mask_boundary_radius",
        "color_loss_weight",
        "max_render_points",
        "device",
        "optimize_lr",
        "optimize_steps",
        "elastic_edge_weight",
        "elastic_volume_weight",
        "bone_length_weight",
        "temporal_weight",
        "mocap_prior_weight",
        "render_sigma_px",
        "render_radius_px",
        "gravity",
        "camera_align_to_frame0",
        "camera_alignment_points",
        "camera_alignment_padding",
        "gaussian_binding_k",
        "pull_gaussians_to_surface",
        "tet_mesh_max_faces",
        "skinning_weight_mode",
        "skinning_weight_k",
        "skinning_deformation_mode",
        "elastic_fem_dt",
        "elastic_fem_substeps",
        "elastic_fem_youngs_modulus",
        "elastic_fem_poisson_ratio",
        "elastic_fem_density",
        "elastic_fem_damping",
        "elastic_fem_radius_scale",
        "elastic_fem_min_radius_scale",
        "elastic_fem_max_radius_scale",
        "elastic_fem_min_bone_length_scale",
        "elastic_fem_reset_each_frame",
        "fem_cg_iters",
        "fem_elastic_stiffness",
        "fem_handle_stiffness",
        "fem_diagonal_reg",
        "fem_handle_support",
        "fem_handle_weight_power",
        "fiber_representation",
        "fiber_max_points",
        "fiber_max_frames",
        "fiber_renderer",
        "fiber_shell_samples",
        "fiber_strand_samples",
        "fiber_warmup_steps",
        "fiber_initial_temperature",
        "fiber_final_temperature",
        "fiber_sigma_scale",
        "fiber_appearance_lr_scale",
        "fiber_geometry_lr_scale",
        "fiber_route_lr_scale",
        "fiber_initial_residual_trust",
        "fiber_bootstrap_route_mass",
        "fiber_residual_trust_weight",
        "fiber_gradient_clip",
        "fiber_log_every",
        "fiber_checkpoint_every",
        "fiber_ema_decay",
        "fiber_route_continuation",
        "fiber_route_hardening",
        "fiber_hard_route_policy",
        "fiber_route_neighbor_k",
        "fiber_route_neighbor_weight",
        "fiber_route_dropout_probability",
        "fiber_route_dropout_residual_bias",
        "fiber_route_prior_final_fraction",
        "fiber_calibration_frames",
        "fiber_risk_calibration_every",
        "fiber_risk_calibration_start_geometry_blend",
        "fiber_risk_calibration_weight",
        "fiber_risk_calibration_ema",
        "fiber_risk_target_prior_blend",
        "fiber_risk_floor",
        "fiber_random_seed",
        "fiber_route_entropy_weight",
        "fiber_route_prior_weight",
        "fiber_shell_normal_weight",
        "fiber_shell_length_weight",
        "fiber_strand_thinness_weight",
        "fiber_height_weight",
        "fiber_bend_weight",
        "fiber_residual_drift_weight",
    ):
        if key in raw:
            setattr(cfg, key, raw[key])
    return cfg


def dump_yaml(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
