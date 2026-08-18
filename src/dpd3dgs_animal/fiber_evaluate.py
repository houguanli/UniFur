from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from .config import PipelineConfig
from .fiber import (
    HARD_ROUTE_POLICIES,
    ROUTE_NAMES,
    attach_fixed_gaussian_base,
    create_unified_fiber_field,
    partition_binding_cache,
)
from .fiber_optimize import _render
from .optimize import (
    DifferentiableSkeletonTetModel,
    _frame_paths,
    _load_gt_frame_torch,
    _resolve_device,
    _resolve_render_size,
)
from .observations import resolve_observations


@dataclass
class FiberEvaluationArtifacts:
    out_dir: Path
    report_json: Path
    contact_sheet_png: Path


class _ImageQualityMetrics:
    def __init__(self, device: str) -> None:
        from torchmetrics.image import (
            LearnedPerceptualImagePatchSimilarity,
            StructuralSimilarityIndexMeasure,
        )

        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=True
        ).to(device)
        self.ssim.eval()
        self.lpips.eval()

    def compute(
        self,
        prediction: torch.Tensor,
        ground_truth: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, float]:
        valid = mask[..., None]
        pred = (prediction * valid).permute(2, 0, 1).unsqueeze(0)
        target = (ground_truth * valid).permute(2, 0, 1).unsqueeze(0)
        self.ssim.reset()
        masked_ssim = float(self.ssim(pred, target).detach().cpu())
        self.lpips.reset()
        masked_lpips = float(self.lpips(pred, target).detach().cpu())
        full_prediction = prediction.permute(2, 0, 1).unsqueeze(0)
        self.ssim.reset()
        full_ssim = float(self.ssim(full_prediction, target).detach().cpu())
        self.lpips.reset()
        full_lpips = float(self.lpips(full_prediction, target).detach().cpu())
        return {
            "masked_ssim": masked_ssim,
            "masked_lpips": masked_lpips,
            "full_ssim": full_ssim,
            "full_lpips": full_lpips,
        }


def evaluate_unified_fiber_stage2(
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
    route_mode: str = "hard",
    camera_manifest: str | Path | None = None,
    export_external_renders: bool = False,
    fixed_base_gaussian_ply: str | Path | None = None,
) -> FiberEvaluationArtifacts:
    """Evaluate a trained fiber field on a disjoint sequence slice."""

    out_dir = Path(out_dir)
    preview_dir = out_dir / "frames"
    preview_dir.mkdir(parents=True, exist_ok=True)
    external_render_dir = out_dir / "external_renders"
    if export_external_renders:
        external_render_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(cfg.device)
    renderer_name = str(renderer or cfg.fiber_renderer).lower()
    if renderer_name not in {"torch", "hairgs"}:
        raise ValueError("renderer must be 'torch' or 'hairgs'")
    valid_route_modes = {"hard", "soft", *ROUTE_NAMES}
    if route_mode not in valid_route_modes:
        raise ValueError(f"route_mode must be one of {sorted(valid_route_modes)}")

    payload = torch.load(checkpoint_pt, map_location=device)
    metadata = payload.get("metadata", {})
    representation = str(
        metadata.get("representation", cfg.fiber_representation)
    ).lower()
    if representation not in {"unified", "residual_only"}:
        raise ValueError(f"Unknown checkpoint representation {representation!r}")
    point_count = int(metadata.get("point_count", cfg.fiber_max_points))
    point_sampling_mode = str(
        metadata.get("point_sampling_mode", cfg.fiber_point_sampling_mode)
    )
    exact_vertex_binding = bool(
        metadata.get("exact_vertex_binding", cfg.fiber_exact_vertex_binding)
    )
    binding_mode = str(metadata.get("binding_mode", cfg.fiber_binding_mode))
    source_mask_mode = str(
        metadata.get("source_mask_mode", cfg.fiber_source_mask_mode)
    )
    source_mask_threshold = float(
        metadata.get("source_mask_threshold", cfg.fiber_source_mask_threshold)
    )
    source_min_opacity = float(
        metadata.get("source_min_opacity", cfg.fiber_source_min_opacity)
    )
    split_fixed_base = bool(
        metadata.get("split_fixed_base", cfg.fiber_split_fixed_base)
    )
    fixed_base_source = Path(
        fixed_base_gaussian_ply
        or metadata.get("fixed_base_gaussian_ply")
        or gaussian_ply
    )
    # A positive evaluation setting is an intentional runtime audit override.
    # Checkpoint metadata remains the default when no override is requested.
    residual_max_scale_fraction = float(cfg.fiber_residual_max_scale_fraction)
    if residual_max_scale_fraction <= 0.0:
        residual_max_scale_fraction = float(
            metadata.get("residual_max_scale_fraction", 0.0)
        )
    fixed_base_max_scale_fraction = float(
        cfg.fiber_fixed_base_max_scale_fraction
    )
    if fixed_base_max_scale_fraction <= 0.0:
        fixed_base_max_scale_fraction = float(
            metadata.get("fixed_base_max_scale_fraction", 0.0)
        )
    shell_samples = int(metadata.get("shell_samples", cfg.fiber_shell_samples))
    strand_samples = int(metadata.get("strand_samples", cfg.fiber_strand_samples))
    hard_route_policy = str(
        metadata.get("hard_route_policy", cfg.fiber_hard_route_policy)
    )
    if hard_route_policy not in HARD_ROUTE_POLICIES:
        raise ValueError(
            f"Unknown checkpoint hard-route policy {hard_route_policy!r}"
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
        max_points=point_count,
        point_sampling_mode=point_sampling_mode,
        exact_vertex_binding=exact_vertex_binding,
        binding_mode=binding_mode,
        source_mask_mode=source_mask_mode,
        source_mask_threshold=source_mask_threshold,
        source_min_opacity=source_min_opacity,
        residual_max_scale_fraction=residual_max_scale_fraction,
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
            binding_mode=binding_mode,
            source_mask_threshold=source_mask_threshold,
            source_min_opacity=source_min_opacity,
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
    allowed_missing = {
        "residual_log_scale_delta",
        "residual_rotation_raw",
        "residual_trust_logits",
        "expert_color_delta",
        "bend_cubic_local",
        "structured_delta_raw",
        "structured_opacity_raw",
        "shell_visibility_gate",
        "strand_visibility_gate",
        "route_active_gate",
        "carrier_logits",
        "carrier_root_tip_raw",
        "initial_carrier_probabilities",
        "initial_carrier_root_tip",
    }
    unexpected_missing = set(incompatible.missing_keys) - allowed_missing
    if unexpected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint state mismatch: "
            f"missing={sorted(unexpected_missing)}, "
            f"unexpected={sorted(incompatible.unexpected_keys)}"
        )
    if "structured_delta_raw" in incompatible.missing_keys:
        # Checkpoints produced before zero-initialized structured increments
        # already stored fully deployed shell/strand geometry.
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
    camera_source = observation_set.source
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
        raise ValueError("No evaluation frames selected")

    manifest_observations: dict[str, dict] = {}
    if camera_manifest is not None:
        with Path(camera_manifest).open("r", encoding="utf-8") as file:
            camera_payload = json.load(file)
        manifest_observations = {
            Path(str(item["image"])).name: item
            for item in camera_payload.get("observations", [])
            if isinstance(item, dict) and "image" in item
        }
    external_observations: list[dict[str, str | int]] = []

    palette = torch.tensor(
        [[0.15, 0.55, 1.0], [1.0, 0.25, 0.1], [0.4, 1.0, 0.2]],
        dtype=torch.float32,
        device=device,
    )
    per_frame: list[dict[str, float | int]] = []
    rows: list[Image.Image] = []
    quality_metrics = _ImageQualityMetrics(device)
    with torch.no_grad():
        for frame_index in frame_indices:
            camera = cameras[frame_index]
            _tet_nodes, surface_vertices, _joints = motion.driven_points(
                motion_indices[frame_index]
            )
            if representation == "residual_only":
                primitives = field.residual_primitives(
                    surface_vertices, motion.surface_faces
                )
            else:
                primitives = field.primitives(
                    surface_vertices,
                    motion.surface_faces,
                    shell_samples=shell_samples,
                    strand_samples=strand_samples,
                    temperature=cfg.fiber_final_temperature,
                    forced_route=route_mode if route_mode in ROUTE_NAMES else None,
                    hard_route=route_mode == "hard",
                    hard_route_policy=hard_route_policy,
                    fin_aspect_ratio=cfg.fiber_fin_aspect_ratio,
                    additive_teacher=cfg.fiber_additive_teacher_mode,
                    teacher_opacity_transfer=cfg.fiber_teacher_opacity_transfer,
                )
            prediction = _render(primitives, camera, cfg, renderer_name)
            ground_truth = _load_gt_frame_torch(
                frame_paths[frame_index], width, height, device
            )
            metrics = _frame_metrics(
                prediction, ground_truth, quality_metrics=quality_metrics
            )
            metrics["frame"] = frame_index
            per_frame.append(metrics)

            if export_external_renders:
                array_path = external_render_dir / f"{frame_index:05d}.npz"
                np.savez_compressed(
                    array_path,
                    rgb=prediction["rgb"].detach().cpu().numpy().astype(np.float32),
                    mask=prediction["mask"].detach().cpu().numpy().astype(np.float32),
                )
                observation = manifest_observations.get(frame_paths[frame_index].name, {})
                external_observations.append(
                    {
                        "image": frame_paths[frame_index].name,
                        "array": str(array_path.resolve()),
                        "frame_index": int(observation.get("frame_index", frame_index)),
                        "view_index": int(observation.get("view_index", 0)),
                    }
                )

            route_color = palette[primitives.route_id]
            if primitives.semantic_foreground is not None:
                # The immutable head/body base participates in depth
                # compositing but has no learnable hair route.  Paint it
                # neutral gray so diagnostics cannot misread it as residual.
                fixed_base = primitives.semantic_foreground <= 0.0
                neutral = torch.full_like(route_color, 0.32)
                route_color = torch.where(fixed_base[:, None], neutral, route_color)
            route_prediction = _render(
                replace(primitives, color=route_color),
                camera,
                cfg,
                renderer_name,
            )
            row = _preview_row(
                ground_truth,
                prediction,
                route_prediction,
                frame_index,
                metrics,
            )
            row.save(preview_dir / f"{frame_index:05d}_comparison.png")
            rows.append(row)

    aggregate = {
        key: float(np.mean([float(item[key]) for item in per_frame]))
        for key in (
            "foreground_l1",
            "foreground_psnr",
            "masked_full_psnr",
            "masked_ssim",
            "masked_lpips",
            "full_psnr",
            "full_ssim",
            "full_lpips",
            "mask_mae",
            "mask_iou",
            "mask_f1",
            "background_opacity_mean",
        )
    }
    probabilities = (
        field.route_probabilities(forced_route="residual").detach()
        if representation == "residual_only"
        else field.route_probabilities(
            temperature=cfg.fiber_final_temperature,
            hard=True,
            hard_policy=hard_route_policy,
        ).detach()
    )
    hard_counts = torch.bincount(
        probabilities.argmax(dim=-1), minlength=len(ROUTE_NAMES)
    ).float()
    hard_counts /= hard_counts.sum().clamp_min(1.0)
    hard_routes = {
        name: float(hard_counts[index].cpu())
        for index, name in enumerate(ROUTE_NAMES)
    }

    contact_sheet_png = out_dir / "evaluation_contact_sheet.png"
    contact_sheet = Image.new(
        "RGB", (max(row.width for row in rows), sum(row.height for row in rows)), "black"
    )
    y = 0
    for row in rows:
        contact_sheet.paste(row, (0, y))
        y += row.height
    contact_sheet.save(contact_sheet_png)

    report_json = out_dir / "evaluation.json"
    report = {
        "checkpoint": str(checkpoint_pt),
        "renderer": renderer_name,
        "representation": representation,
        "route_mode": route_mode,
        "camera_source": camera_source,
        "render_size": [width, height],
        "frame_indices": frame_indices,
        "motion_indices": [motion_indices[index] for index in frame_indices],
        "aggregate": aggregate,
        "soft_routes": (
            {"shell": 0.0, "strand": 0.0, "residual": 1.0}
            if representation == "residual_only"
            else field.route_summary(cfg.fiber_final_temperature)
        ),
        "hard_routes": hard_routes,
        "hard_route_policy": hard_route_policy,
        "fixed_base_count": (
            field.fixed_base.point_count if field.fixed_base is not None else 0
        ),
        "fixed_base_render_role": (
            "immutable_depth_occluder" if field.fixed_base is not None else "none"
        ),
        "per_frame": per_frame,
    }
    with open(report_json, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    if export_external_renders:
        external_manifest = {
            "schema": "dpd3dgs-external-render-manifest-v1",
            "status": "complete",
            "render_size": [width, height],
            "observations": external_observations,
        }
        (out_dir / "external_render_manifest.json").write_text(
            json.dumps(external_manifest, indent=2), encoding="utf-8"
        )
    return FiberEvaluationArtifacts(out_dir, report_json, contact_sheet_png)


def _select_frame_indices(
    available_frames: int,
    frame_start: int,
    frame_stride: int,
    max_frames: int | None,
) -> list[int]:
    if frame_start < 0:
        raise ValueError("frame_start must be non-negative")
    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    indices = list(range(frame_start, available_frames, frame_stride))
    if max_frames is not None and max_frames > 0:
        indices = indices[:max_frames]
    return indices


def _frame_metrics(
    prediction: dict[str, torch.Tensor],
    ground_truth: dict[str, torch.Tensor],
    quality_metrics: _ImageQualityMetrics | None = None,
) -> dict[str, float]:
    gt_mask = ground_truth["mask"].clamp(0.0, 1.0)
    valid = gt_mask[..., None]
    difference = prediction["rgb"] - ground_truth["rgb"]
    denominator = (valid.sum() * 3.0).clamp_min(1.0)
    foreground_l1 = (difference.abs() * valid).sum() / denominator
    foreground_mse = (difference.square() * valid).sum() / denominator
    psnr = -10.0 * torch.log10(foreground_mse.clamp_min(1e-10))
    masked_full_mse = torch.mean((difference * valid).square())
    masked_full_psnr = -10.0 * torch.log10(masked_full_mse.clamp_min(1e-10))
    composite_target = ground_truth["rgb"] * valid
    full_mse = torch.mean((prediction["rgb"] - composite_target).square())
    full_psnr = -10.0 * torch.log10(full_mse.clamp_min(1e-10))

    pred_mask_soft = prediction["mask"].clamp(0.0, 1.0)
    pred_mask = pred_mask_soft > 0.5
    target_mask = gt_mask > 0.5
    intersection = torch.logical_and(pred_mask, target_mask).sum().float()
    union = torch.logical_or(pred_mask, target_mask).sum().float()
    precision = intersection / pred_mask.sum().float().clamp_min(1.0)
    recall = intersection / target_mask.sum().float().clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-8)
    result = {
        "foreground_l1": float(foreground_l1.cpu()),
        "foreground_psnr": float(psnr.cpu()),
        "masked_full_psnr": float(masked_full_psnr.cpu()),
        "full_psnr": float(full_psnr.cpu()),
        "mask_mae": float(torch.mean(torch.abs(pred_mask_soft - gt_mask)).cpu()),
        "mask_iou": float((intersection / union.clamp_min(1.0)).cpu()),
        "mask_f1": float(f1.cpu()),
        "background_opacity_mean": float(
            (
                (pred_mask_soft * (1.0 - gt_mask)).sum()
                / (1.0 - gt_mask).sum().clamp_min(1.0)
            ).cpu()
        ),
    }
    if quality_metrics is not None:
        result.update(
            quality_metrics.compute(
                prediction["rgb"], ground_truth["rgb"], gt_mask
            )
        )
    return result


def _preview_row(
    ground_truth: dict[str, torch.Tensor],
    prediction: dict[str, torch.Tensor],
    route_prediction: dict[str, torch.Tensor],
    frame_index: int,
    metrics: dict[str, float | int],
) -> Image.Image:
    gt_mask = ground_truth["mask"][..., None]
    gt_rgb = ground_truth["rgb"] * gt_mask
    pred_rgb = prediction["rgb"]
    semantic_mask = prediction["mask"].clamp(0.0, 1.0)[..., None]
    hair_layer_rgb = pred_rgb * semantic_mask
    error = torch.abs(pred_rgb - ground_truth["rgb"]) * gt_mask
    images = [
        _tensor_image(gt_rgb),
        _tensor_image(hair_layer_rgb),
        _tensor_image(pred_rgb),
        _tensor_image(error * 3.0),
        _tensor_image(route_prediction["rgb"]),
    ]
    labels = [
        f"frame {frame_index} GT",
        f"hair layer diagnostic PSNR {float(metrics['foreground_psnr']):.2f}",
        "joint composite (fixed base is immutable)",
        "3x foreground error",
        "route map: shell/strand/residual; gray=fixed base",
    ]
    label_height = 24
    row = Image.new("RGB", (sum(image.width for image in images), images[0].height + label_height), "black")
    draw = ImageDraw.Draw(row)
    x = 0
    for image, label in zip(images, labels):
        row.paste(image, (x, label_height))
        draw.text((x + 5, 5), label, fill="white")
        x += image.width
    return row


def _tensor_image(tensor: torch.Tensor) -> Image.Image:
    array = (tensor.detach().cpu().numpy().clip(0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(array)
