#!/usr/bin/env python3
"""Train a hair-only HairGS Stage-1 with multi-view topology safeguards."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from plyfile import PlyData
from pytorch3d.renderer.mesh import rasterize_meshes
from pytorch3d.structures import Meshes
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from arguments import GeneralParams, ModelParams, OptimizationParams
from gaussian_renderer import render
from loss.losses import orientation_loss_rast, ssim
from scene import Scene
from utils import (
    build_rotation,
    enable_accelerated_rasterization,
    inverse_sigmoid,
    prepare_output_path,
)


_SH_C0 = 0.28209479177387814


def _clone_state(value):
    if torch.is_tensor(value):
        result = value.detach().clone()
        result.requires_grad_(value.requires_grad)
        return result
    if isinstance(value, dict):
        return {key: _clone_state(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_state(item) for item in value)
    if isinstance(value, list):
        return [_clone_state(item) for item in value]
    return copy.deepcopy(value)


_GAUSSIAN_PARAMETER_NAMES = (
    "_xyz",
    "_features_dc",
    "_features_rest",
    "_opacity",
    "_mask",
    "_scaling",
    "_rotation",
)


@torch.no_grad()
def _snapshot_parent_values(gaussians, count: int) -> dict[str, torch.Tensor]:
    return {
        name: getattr(gaussians, name)[:count].detach().clone()
        for name in _GAUSSIAN_PARAMETER_NAMES
    }


@torch.no_grad()
def _freeze_parent_gradients(gaussians, count: int) -> None:
    """Make probation optimize only newly appended children."""
    for name in _GAUSSIAN_PARAMETER_NAMES:
        gradient = getattr(gaussians, name).grad
        if gradient is not None:
            gradient[:count].zero_()


@torch.no_grad()
def _restore_parent_values(gaussians, values: dict[str, torch.Tensor]) -> None:
    # Adam momentum can move a parameter even after its current gradient was
    # zeroed.  Restoring the prefix makes the calibrated parent representation
    # bit-identical throughout the child-only probation interval.
    for name, value in values.items():
        getattr(gaussians, name)[: value.shape[0]].copy_(value)


@torch.no_grad()
def _project_scales_to_feasible_set(
    gaussians,
    hull: "HairVisualHull",
    args: argparse.Namespace,
    *,
    check_projected_radius: bool,
) -> dict[str, int]:
    """Prevent a few large ellipsoids from filling hair-mask deficits."""
    scales = gaussians.get_scaling
    major = scales.max(dim=1).values
    world_cap = args.clean_world_scale_cap * args.scene_extent
    factor = torch.clamp(world_cap / major.clamp_min(1e-8), max=1.0)
    world_clamped = int((factor < 1.0).sum())
    projected_clamped = 0
    if check_projected_radius:
        _support, projected = hull.support_and_projected_radius(
            gaussians.get_xyz, major
        )
        projected_factor = torch.clamp(
            args.clean_max_projected_radius / projected.clamp_min(1e-8),
            max=1.0,
        )
        projected_clamped = int((projected_factor < 1.0).sum())
        factor = torch.minimum(factor, projected_factor)
    constrained = scales * factor[:, None]

    ordered, order = torch.sort(constrained, dim=1, descending=True)
    hard_major = ordered[:, 1] * args.clean_hard_max_anisotropy
    anisotropy_clamped = int((ordered[:, 0] > hard_major).sum())
    ordered[:, 0] = torch.minimum(ordered[:, 0], hard_major)
    constrained.scatter_(1, order, ordered)
    gaussians._scaling.copy_(
        gaussians.scaling_inverse_activation(constrained.clamp_min(1e-8))
    )
    return {
        "world_scale_clamped": world_clamped,
        "projected_radius_clamped": projected_clamped,
        "anisotropy_clamped": anisotropy_clamped,
    }


@torch.no_grad()
def _project_appearance_to_feasible_set(
    gaussians, args: argparse.Namespace
) -> dict[str, int]:
    """Remove the low-opacity/high-radiance degeneracy of masked Stage-1.

    HairGS uses degree-zero SH in this stage, so its rendered base colour is
    ``0.5 + C0 * features_dc``.  An unconstrained optimiser can compensate a
    very small opacity with RGB values far above one.  Those points contribute
    little to the average training loss but become saturated streaks in other
    views.  Projection keeps the appearance physically displayable without
    changing opacity or geometry.
    """
    features = gaussians._features_dc
    rgb = 0.5 + _SH_C0 * features
    constrained = rgb.clamp(args.clean_rgb_min, args.clean_rgb_max)
    gamut_reordered = 0
    gamut_chroma_clamped = 0
    if args.clean_enforce_warm_hair_gamut:
        warm = torch.sort(constrained, dim=-1, descending=True).values
        gamut_reordered = int(
            torch.any(torch.abs(warm - constrained) > 1e-7, dim=-1).sum()
        )
        # A channel-order constraint alone still admits saturated red/orange
        # points such as (1, 0, 0).  Those are nearly invisible in the average
        # fit loss but become bright streaks from held-out views.  Learn robust
        # R-G and G-B limits from the actual training hair pixels, then project
        # every Gaussian onto that continuous warm-colour cone while preserving
        # its mean luminance as far as the display bounds permit.
        warm_flat = warm.reshape(-1, 3)
        rg = warm_flat[:, 0] - warm_flat[:, 1]
        gb = warm_flat[:, 1] - warm_flat[:, 2]
        rg_limited = rg.clamp_max(args.clean_warm_rg_gap_cap)
        gb_limited = gb.clamp_max(args.clean_warm_gb_gap_cap)
        gamut_chroma_clamped = int(
            ((rg > args.clean_warm_rg_gap_cap + 1e-7)
             | (gb > args.clean_warm_gb_gap_cap + 1e-7)).sum()
        )
        base = torch.stack(
            [
                warm_flat[:, 2] + gb_limited + rg_limited,
                warm_flat[:, 2] + gb_limited,
                warm_flat[:, 2],
            ],
            dim=-1,
        )
        shift = warm_flat.mean(dim=-1) - base.mean(dim=-1)
        shift = torch.maximum(shift, -base[:, 2])
        shift = torch.minimum(shift, 1.0 - base[:, 0])
        constrained = (base + shift[:, None]).clamp(
            args.clean_rgb_min, args.clean_rgb_max
        ).reshape_as(warm)
    changed = torch.any(torch.abs(constrained - rgb) > 1e-7, dim=-1)
    gaussians._features_dc.copy_((constrained - 0.5) / _SH_C0)
    return {
        "rgb_clamped": int(changed.sum()),
        "gamut_reordered": gamut_reordered,
        "gamut_chroma_clamped": gamut_chroma_clamped,
    }


@torch.no_grad()
def _estimate_warm_hair_gamut(cameras, args: argparse.Namespace) -> dict[str, float]:
    """Estimate robust chroma limits from visible training hair pixels only."""
    pixels = []
    for camera in cameras:
        mask = camera.mask.bool()
        if mask.any():
            pixels.append(camera.original_image[:, mask].transpose(0, 1))
    if not pixels:
        raise RuntimeError("Cannot calibrate warm hair gamut without hair-mask pixels")
    color = torch.cat(pixels, dim=0).clamp(0.0, 1.0)
    ordered = torch.sort(color, dim=-1, descending=True).values
    quantile = float(args.clean_warm_gamut_quantile)
    if not 0.5 <= quantile <= 1.0:
        raise ValueError("clean_warm_gamut_quantile must be in [0.5, 1.0]")
    empirical_rg = float(
        torch.quantile(ordered[:, 0] - ordered[:, 1], quantile).item()
    )
    empirical_gb = float(
        torch.quantile(ordered[:, 1] - ordered[:, 2], quantile).item()
    )
    if args.clean_warm_rg_gap_cap < 0.0:
        args.clean_warm_rg_gap_cap = empirical_rg
    if args.clean_warm_gb_gap_cap < 0.0:
        args.clean_warm_gb_gap_cap = empirical_gb
    if args.clean_warm_rg_gap_cap < 0.0 or args.clean_warm_gb_gap_cap < 0.0:
        raise ValueError("warm gamut channel-gap caps must be non-negative")
    return {
        "quantile": quantile,
        "empirical_rg_gap": empirical_rg,
        "empirical_gb_gap": empirical_gb,
        "applied_rg_gap_cap": float(args.clean_warm_rg_gap_cap),
        "applied_gb_gap_cap": float(args.clean_warm_gb_gap_cap),
        "sample_count": int(color.shape[0]),
    }


@torch.no_grad()
def _cap_probation_child_opacity(
    gaussians, parent_count: int, maximum_alpha: float
) -> int:
    """Keep incremental children residual-sized until calibration accepts them."""
    if parent_count >= gaussians._opacity.shape[0]:
        return 0
    logits = gaussians._opacity[parent_count:]
    alpha = torch.sigmoid(logits)
    clamped = alpha.clamp_max(maximum_alpha)
    changed = int((alpha > maximum_alpha).sum())
    logits.copy_(inverse_sigmoid(clamped.clamp(1e-5, 1.0 - 1e-5)))
    return changed


class HairVisualHull:
    def __init__(
        self,
        dataset: Path,
        provenance: dict,
        dilation: int,
        device: str,
        head_occlusion_margin: float,
    ):
        kernel = np.ones((2 * dilation + 1, 2 * dilation + 1), np.uint8)
        ply = PlyData.read(dataset / "head_mesh.ply")
        vertex = ply["vertex"].data
        head_vertices = np.stack(
            [vertex["x"], vertex["y"], vertex["z"]], axis=1
        ).astype(np.float32)
        head_faces = np.stack(ply["face"].data["vertex_indices"]).astype(np.int64)
        self.head_occlusion_margin = float(head_occlusion_margin)
        self.fit_entries = self._load_entries(
            dataset,
            provenance["fit_observations"],
            kernel,
            dilation,
            device,
            head_vertices,
            head_faces,
        )
        self.calibration_entries = self._load_entries(
            dataset,
            provenance.get("calibration_observations", []),
            kernel,
            dilation,
            device,
            head_vertices,
            head_faces,
        )
        # Scale projection and the persistent scaffold constraint use only the
        # fit-view hull.  Calibration masks are reserved for screening newly
        # proposed children and never become photometric training targets.
        self.entries = self.fit_entries

    @staticmethod
    def _load_entries(
        dataset,
        observations,
        kernel,
        dilation,
        device,
        head_vertices,
        head_faces,
    ):
        entries = []
        for observation in observations:
            mask = np.asarray(
                Image.open(dataset / "masks" / observation["image"]).convert("L")
            ) >= 128
            if dilation:
                mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) != 0
            width, height, fx, fy, cx, cy = observation["intrinsics"]
            transform = np.asarray(observation["world_to_camera"], np.float32)
            head_depth = HairVisualHull._rasterize_head_depth(
                head_vertices,
                head_faces,
                transform,
                observation["intrinsics"],
                device,
            )
            entries.append(
                {
                    "name": observation["image"],
                    "rotation": torch.as_tensor(
                        transform[:3, :3], dtype=torch.float32, device=device
                    ),
                    "translation": torch.as_tensor(
                        transform[:3, 3], dtype=torch.float32, device=device
                    ),
                    "mask": torch.as_tensor(mask, dtype=torch.bool, device=device),
                    "head_depth": head_depth,
                    "head_mask": head_depth > 0,
                    "width": int(width),
                    "height": int(height),
                    "fx": float(fx),
                    "fy": float(fy),
                    "cx": float(cx),
                    "cy": float(cy),
                }
            )
        return entries

    @staticmethod
    @torch.no_grad()
    def _rasterize_head_depth(vertices, faces, transform, intrinsics, device):
        width, height, fx, fy, cx, cy = intrinsics
        camera = vertices @ transform[:3, :3].T + transform[:3, 3]
        safe_depth = np.maximum(camera[:, 2], 1e-8)
        u = fx * camera[:, 0] / safe_depth + cx
        v = fy * camera[:, 1] / safe_depth + cy
        projected = np.stack(
            [
                2.0 * u / (width - 1.0) - 1.0,
                1.0 - 2.0 * v / (height - 1.0),
                camera[:, 2],
            ],
            axis=1,
        ).astype(np.float32)
        mesh = Meshes(
            verts=[torch.as_tensor(projected, device=device)],
            faces=[torch.as_tensor(faces, device=device)],
        )
        pix_to_face, zbuf, _, _ = rasterize_meshes(
            mesh,
            image_size=(int(height), int(width)),
            blur_radius=0.0,
            faces_per_pixel=1,
            perspective_correct=False,
            cull_backfaces=False,
            max_faces_per_bin=200_000,
        )
        depth = zbuf[0, ..., 0]
        return torch.where(
            pix_to_face[0, ..., 0] >= 0, depth, torch.zeros_like(depth)
        )

    @torch.no_grad()
    def support_and_projected_radius(
        self,
        xyz: torch.Tensor,
        major_scale: torch.Tensor,
        chunk: int = 200_000,
        entries=None,
        allow_head_occlusion: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        support = torch.zeros(xyz.shape[0], dtype=torch.int16, device=xyz.device)
        projected = torch.zeros(xyz.shape[0], dtype=torch.float32, device=xyz.device)
        for entry in self.entries if entries is None else entries:
            for start in range(0, xyz.shape[0], chunk):
                stop = min(start + chunk, xyz.shape[0])
                camera = (
                    xyz[start:stop] @ entry["rotation"].T + entry["translation"]
                )
                depth = camera[:, 2]
                safe_depth = depth.clamp_min(1e-8)
                u = torch.round(entry["fx"] * camera[:, 0] / safe_depth + entry["cx"]).long()
                v = torch.round(entry["fy"] * camera[:, 1] / safe_depth + entry["cy"]).long()
                valid = (
                    (depth > 1e-8)
                    & (u >= 0)
                    & (u < entry["width"])
                    & (v >= 0)
                    & (v < entry["height"])
                )
                in_mask = torch.zeros_like(valid)
                if valid.any():
                    in_mask[valid] = entry["mask"][v[valid], u[valid]]
                    if allow_head_occlusion:
                        head_depth = entry["head_depth"][v[valid], u[valid]]
                        occluded = (head_depth > 0) & (
                            depth[valid]
                            > head_depth + self.head_occlusion_margin
                        )
                        in_mask[valid] |= occluded
                support[start:stop] += in_mask.to(torch.int16)
                radius = max(entry["fx"], entry["fy"]) * major_scale[start:stop] / safe_depth
                radius = torch.where(valid, radius, torch.zeros_like(radius))
                projected[start:stop] = torch.maximum(projected[start:stop], radius)
        return support, projected


def _hair_bbox(mask: torch.Tensor, padding: int = 12) -> tuple[slice, slice]:
    indices = torch.nonzero(mask, as_tuple=False)
    if not indices.numel():
        return slice(0, mask.shape[0]), slice(0, mask.shape[1])
    y0, x0 = indices.min(dim=0).values.tolist()
    y1, x1 = indices.max(dim=0).values.tolist()
    return (
        slice(max(0, y0 - padding), min(mask.shape[0], y1 + padding + 1)),
        slice(max(0, x0 - padding), min(mask.shape[1], x1 + padding + 1)),
    )


def _hair_image_loss(
    gaussians,
    camera,
    background: torch.Tensor,
    args: argparse.Namespace,
    include_orientation: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict]:
    package = render(camera, gaussians, background)
    prediction = package["render"]
    occupancy = render(
        camera,
        gaussians,
        background,
        override_color=torch.ones_like(gaussians.get_xyz),
    )["render"][0].clamp(0.0, 1.0)
    mask = camera.mask.bool()
    inverse = ~mask
    head_mask = getattr(camera, "head_occlusion_mask", torch.zeros_like(mask))
    ignored_occlusion = head_mask & inverse
    supervised_inverse = inverse & ~ignored_occlusion
    target = camera.original_image
    mask3 = mask.unsqueeze(0).expand_as(target)

    inside_l1 = torch.abs(prediction[mask3] - target[mask3]).mean()
    outside_rgb_global = (
        torch.abs(prediction[:, supervised_inverse]).mean()
        if supervised_inverse.any()
        else prediction.new_zeros(())
    )
    coverage = (1.0 - occupancy[mask]).mean()
    spill_global = (
        occupancy[supervised_inverse].mean()
        if supervised_inverse.any()
        else occupancy.new_zeros(())
    )
    visible_occupancy = occupancy.masked_fill(ignored_occlusion, 0.0)
    intersection = visible_occupancy[mask].sum()
    dice = 1.0 - (2.0 * intersection + 1e-6) / (
        visible_occupancy.sum() + mask.float().sum() + 1e-6
    )
    ys, xs = _hair_bbox(mask, padding=args.clean_local_outside_padding)
    local_inverse = supervised_inverse[ys, xs]
    outside_rgb_local = (
        torch.abs(prediction[:, ys, xs][:, local_inverse]).mean()
        if local_inverse.any()
        else prediction.new_zeros(())
    )
    spill_local = (
        occupancy[ys, xs][local_inverse].mean()
        if local_inverse.any()
        else occupancy.new_zeros(())
    )
    outside_rgb = (
        args.clean_local_outside_blend * outside_rgb_local
        + (1.0 - args.clean_local_outside_blend) * outside_rgb_global
    )
    spill = (
        args.clean_local_outside_blend * spill_local
        + (1.0 - args.clean_local_outside_blend) * spill_global
    )
    masked_prediction = prediction[:, ys, xs] * mask3[:, ys, xs]
    masked_target = target[:, ys, xs] * mask3[:, ys, xs]
    dssim = 1.0 - ssim(masked_prediction, masked_target)

    scales = gaussians.get_scaling
    ordered = torch.sort(scales, dim=1, descending=True).values
    major, middle, minor = ordered.unbind(dim=1)
    ratio = major / middle.clamp_min(1e-8)
    cross_ratio = middle / minor.clamp_min(1e-8)
    scale_cap = args.clean_world_scale_cap * args.scene_extent
    scale_penalty = torch.square(F.relu(major / scale_cap - 1.0)).mean()
    anisotropy_low = torch.square(F.relu(args.clean_min_anisotropy - ratio)).mean()
    anisotropy_high = torch.square(F.relu(ratio - args.clean_max_anisotropy)).mean()
    cross_section = torch.square(torch.log(cross_ratio.clamp_min(1e-8))).mean()

    orientation = prediction.new_zeros(())
    if include_orientation and args.clean_orientation_weight > 0:
        orientation = orientation_loss_rast(gaussians, camera, args, background)

    terms = {
        "inside_l1": inside_l1,
        "outside_rgb": outside_rgb,
        "outside_rgb_global": outside_rgb_global,
        "outside_rgb_local": outside_rgb_local,
        "coverage": coverage,
        "spill": spill,
        "spill_global": spill_global,
        "spill_local": spill_local,
        "dice": dice,
        "dssim": dssim,
        "orientation": orientation,
        "scale": scale_penalty,
        "anisotropy_low": anisotropy_low,
        "anisotropy_high": anisotropy_high,
        "cross_section": cross_section,
    }
    total = (
        inside_l1
        + args.clean_outside_rgb_weight * outside_rgb
        + args.clean_coverage_weight * coverage
        + args.clean_spill_weight * spill
        + args.clean_dice_weight * dice
        + args.clean_dssim_weight * dssim
        + args.clean_orientation_weight * orientation
        + args.clean_scale_weight * scale_penalty
        + args.clean_anisotropy_weight * (anisotropy_low + anisotropy_high)
        + args.clean_cross_section_weight * cross_section
    )
    return total, terms, package


@torch.no_grad()
def _calibration_metrics(gaussians, cameras, background, args) -> list[dict]:
    rows = []
    for camera in cameras:
        _, terms, _ = _hair_image_loss(
            gaussians, camera, background, args, include_orientation=False
        )
        risk = (
            terms["inside_l1"]
            + 0.5 * terms["coverage"]
            + 0.5 * terms["spill"]
            + 0.25 * terms["dice"]
            + 0.2 * terms["dssim"]
        )
        rows.append(
            {
                "image": camera.image_name + ".png",
                "risk": float(risk.item()),
                **{
                    key: float(terms[key].item())
                    for key in ("inside_l1", "coverage", "spill", "dice", "dssim")
                },
            }
        )
    return rows


def _hard_prune(
    gaussians,
    hull: HairVisualHull,
    args: argparse.Namespace,
    protected_prefix: int = 0,
) -> dict:
    with torch.no_grad():
        scales = gaussians.get_scaling
        ordered = torch.sort(scales, dim=1, descending=True).values
        major = ordered[:, 0]
        ratio = major / ordered[:, 1].clamp_min(1e-8)
        support, projected = hull.support_and_projected_radius(
            gaussians.get_xyz, major
        )
        scale_cap = args.clean_world_scale_cap * args.scene_extent
        prune = (
            (support < args.clean_min_hull_support)
            | (major > scale_cap)
            | (ratio > args.clean_hard_max_anisotropy)
            | (projected > args.clean_max_projected_radius)
        )
        reason_counts = {
            "below_hull_support": int((support < args.clean_min_hull_support).sum()),
            "above_world_scale": int((major > scale_cap).sum()),
            "above_anisotropy": int((ratio > args.clean_hard_max_anisotropy).sum()),
            "above_projected_radius": int(
                (projected > args.clean_max_projected_radius).sum()
            ),
        }
        protected_violations = int(prune[:protected_prefix].sum())
        # A topology proposal is only allowed to add or reject children.  It
        # must not silently delete a previously calibrated parent and claim the
        # resulting image change as a successful densification event.
        if protected_prefix:
            prune[:protected_prefix] = False
        keep = ~prune
        if int(keep.sum()) > args.clean_max_gaussians:
            eligible_mask = keep.clone()
            eligible_mask[:protected_prefix] = False
            eligible = torch.nonzero(eligible_mask, as_tuple=False).squeeze(1)
            protected_count = min(protected_prefix, int(keep.sum()))
            available = max(args.clean_max_gaussians - protected_count, 0)
            if available < 1:
                raise RuntimeError("Clean-scaffold capacity is below parent count")
            quality = gaussians.get_opacity.squeeze(1)[eligible] * (
                support[eligible].float() / len(hull.entries)
            )
            order = torch.topk(
                quality, k=min(available, len(eligible)), sorted=False
            ).indices
            capacity_keep = torch.zeros_like(keep)
            capacity_keep[:protected_prefix] = True
            capacity_keep[eligible[order]] = True
            prune |= ~capacity_keep
            reason_counts["capacity"] = int((~capacity_keep & keep).sum())
        else:
            reason_counts["capacity"] = 0
        if int((~prune).sum()) < 32:
            raise RuntimeError("Geometric constraints would prune the entire scaffold")
        pruned = int(prune.sum())
        if pruned:
            gaussians.prune_points(prune)
        return {
            **reason_counts,
            "protected_parent_violations": protected_violations,
            "pruned_total": pruned,
            "remaining": int(gaussians.get_xyz.shape[0]),
            "support_mean": float(support.float().mean().item()),
            "projected_radius_p99": float(
                torch.quantile(projected.float(), 0.99).item()
            ),
        }


@torch.no_grad()
def _opacity_preserving_densification(
    gaussians, hull: HairVisualHull, extent: float, args
) -> dict:
    """Add low-opacity children without changing or deleting their parents.

    A stock 3DGS split deletes the parent and randomly displaces two children.
    That is appropriate for an unconstrained radiance field, but it can tear a
    hair silhouette in an unseen direction before the children have learned a
    useful placement.  Clean Stage-1 therefore treats topology as a residual:
    the calibrated parent stays bit-identical and a low-alpha child is added.
    The child starts close to its parent and has to earn opacity during the
    multi-view probation window.  A rejected event restores the exact snapshot.
    """
    point_count = gaussians.get_xyz.shape[0]
    gradients = gaussians.xyz_gradient_accum / gaussians.denom.clamp_min(1.0)
    gradients[~torch.isfinite(gradients)] = 0.0
    score = torch.norm(gradients, dim=-1)
    scale = gaussians.get_scaling
    split_threshold = gaussians.training_args.percent_dense * extent
    calibration_support, _ = hull.support_and_projected_radius(
        gaussians.get_xyz,
        scale.max(dim=1).values,
        entries=hull.calibration_entries,
        allow_head_occlusion=True,
    )
    minimum_calibration_support = min(
        args.clean_child_min_calibration_support,
        len(hull.calibration_entries),
    )
    candidate = (
        (score >= gaussians.training_args.densify_grad_threshold)
        & (calibration_support >= minimum_calibration_support)
    )
    clone_mask = candidate & (scale.max(dim=1).values <= split_threshold)
    split_mask = candidate & (scale.max(dim=1).values > split_threshold)

    # Every selected parent contributes one incremental child.  Bound the event
    # before allocating tensors so a single high-gradient view cannot dominate
    # the multi-view calibration decision.
    candidate_ids = torch.nonzero(clone_mask | split_mask, as_tuple=False).squeeze(1)
    if candidate_ids.numel() > args.clean_max_new_per_event:
        keep_ids = candidate_ids[
            torch.topk(
                score[candidate_ids],
                k=args.clean_max_new_per_event,
                sorted=False,
            ).indices
        ]
        allowed = torch.zeros(point_count, dtype=torch.bool, device=score.device)
        allowed[keep_ids] = True
        clone_mask &= allowed
        split_mask &= allowed

    clone_count = int(clone_mask.sum())
    split_count = int(split_mask.sum())
    if clone_count + split_count == 0:
        gaussians.xyz_gradient_accum.zero_()
        gaussians.denom.zero_()
        return {
            "clone": 0,
            "split": 0,
            "net_new": 0,
            "calibration_safe_candidates": int(candidate.sum()),
        }

    selected = clone_mask | split_mask
    selected_rotation = gaussians._rotation[selected]
    selected_scale = scale[selected]
    selected_xyz = gaussians._xyz[selected]

    # A small local-frame displacement breaks clone symmetry while remaining
    # far below the parent's support.  Large parents move only along their
    # major axis; compact parents remain coincident and diverge through their
    # independent Adam state.
    local_jitter = torch.zeros_like(selected_scale)
    selected_is_split = split_mask[selected]
    split_scales = selected_scale[selected_is_split]
    split_major_axis = split_scales.argmax(dim=1, keepdim=True)
    split_jitter = torch.zeros_like(split_scales)
    split_jitter.scatter_(
        1,
        split_major_axis,
        args.clean_child_jitter_scale
        * split_scales.gather(1, split_major_axis),
    )
    local_jitter[selected_is_split] = split_jitter
    world_jitter = torch.bmm(
        build_rotation(selected_rotation), local_jitter.unsqueeze(-1)
    ).squeeze(-1)
    child_alpha_value = min(
        args.clean_new_child_alpha,
        args.clean_child_total_alpha_budget / max(clone_count + split_count, 1),
    )
    child_alpha = torch.full_like(
        gaussians._opacity[selected], child_alpha_value
    )
    child_logit = inverse_sigmoid(child_alpha.clamp(1e-5, 1.0 - 1e-5))
    child_scale = (selected_scale * args.clean_child_scale_factor).clamp_min(1e-8)
    child_scaling = gaussians.scaling_inverse_activation(child_scale)

    gaussians.densification_postfix(
        selected_xyz + world_jitter,
        gaussians._features_dc[selected],
        gaussians._features_rest[selected],
        child_logit,
        gaussians._mask[selected],
        child_scaling,
        selected_rotation,
    )
    return {
        "clone": clone_count,
        "split": split_count,
        "net_new": clone_count + split_count,
        "calibration_safe_candidates": int(candidate.sum()),
        "minimum_calibration_support": int(minimum_calibration_support),
        "child_alpha": float(child_alpha_value),
        "child_total_alpha_budget": float(args.clean_child_total_alpha_budget),
        "child_scale_factor": float(args.clean_child_scale_factor),
        "parent_preserved": True,
    }


def _begin_topology_event(
    iteration: int,
    gaussians,
    hull: HairVisualHull,
    topology_fit_cameras,
    calibration_cameras,
    background,
    opt,
    args,
) -> dict:
    before_count = int(gaussians.get_xyz.shape[0])
    before_metrics = _calibration_metrics(
        gaussians, calibration_cameras, background, args
    )
    before_fit_metrics = _calibration_metrics(
        gaussians, topology_fit_cameras, background, args
    )
    snapshot = _clone_state(gaussians.capture())
    frozen_parent_values = _snapshot_parent_values(gaussians, before_count)
    densification = _opacity_preserving_densification(
        gaussians, hull, args.scene_extent, args
    )
    constraints = _hard_prune(
        gaussians, hull, args, protected_prefix=before_count
    )
    return {
        "iteration": iteration,
        "decision_iteration": iteration + args.clean_topology_probation,
        "snapshot": snapshot,
        "frozen_parent_values": frozen_parent_values,
        "before_count": before_count,
        "proposed_count": int(gaussians.get_xyz.shape[0]),
        "before": before_metrics,
        "before_fit": before_fit_metrics,
        "before_mean_fit_risk": float(
            np.mean([row["risk"] for row in before_fit_metrics])
        ),
        "before_mean_risk": float(
            np.mean([row["risk"] for row in before_metrics])
        ),
        "densification": densification,
        "initial_constraints": constraints,
    }


def _finalize_topology_event(
    iteration: int,
    pending: dict,
    gaussians,
    hull: HairVisualHull,
    topology_fit_cameras,
    calibration_cameras,
    background,
    opt,
    args,
) -> dict:
    final_constraints = _hard_prune(
        gaussians,
        hull,
        args,
        protected_prefix=pending["before_count"],
    )
    after_metrics = _calibration_metrics(
        gaussians, calibration_cameras, background, args
    )
    after_fit_metrics = _calibration_metrics(
        gaussians, topology_fit_cameras, background, args
    )
    per_view = [
        after["risk"] <= before["risk"] + args.clean_calibration_view_margin
        for before, after in zip(pending["before"], after_metrics)
    ]
    before_mean = pending["before_mean_risk"]
    after_mean = float(np.mean([row["risk"] for row in after_metrics]))
    before_mean_fit = pending["before_mean_fit_risk"]
    after_mean_fit = float(np.mean([row["risk"] for row in after_fit_metrics]))
    fit_gain = before_mean_fit - after_mean_fit
    fit_pass = fit_gain >= args.clean_topology_min_fit_gain
    accepted = bool(
        all(per_view)
        and after_mean <= before_mean + args.clean_calibration_mean_margin
        and fit_pass
    )
    trial_count = int(gaussians.get_xyz.shape[0])
    if not accepted:
        gaussians.restore(pending["snapshot"], opt)
        gaussians.xyz_gradient_accum.zero_()
        gaussians.denom.zero_()
    return {
        "iteration": pending["iteration"],
        "decision_iteration": iteration,
        "probation_steps": iteration - pending["iteration"],
        "before_count": pending["before_count"],
        "proposed_count": pending["proposed_count"],
        "trial_count": trial_count,
        "final_count": int(gaussians.get_xyz.shape[0]),
        "accepted": accepted,
        "per_view_pass": per_view,
        "before_mean_risk": before_mean,
        "after_mean_risk": after_mean,
        "before": pending["before"],
        "after": after_metrics,
        "before_fit": pending["before_fit"],
        "after_fit": after_fit_metrics,
        "before_mean_fit_risk": before_mean_fit,
        "after_mean_fit_risk": after_mean_fit,
        "fit_gain": fit_gain,
        "fit_pass": fit_pass,
        "densification": pending["densification"],
        "initial_constraints": pending["initial_constraints"],
        "final_constraints": final_constraints,
    }


@torch.no_grad()
def _save_review(path: Path, gaussians, cameras, background) -> None:
    tiles = []
    for camera in cameras:
        prediction = render(camera, gaussians, background)["render"]
        head_mask = getattr(
            camera, "head_occlusion_mask", torch.zeros_like(camera.mask.bool())
        )
        prediction = prediction.masked_fill(
            (head_mask & ~camera.mask.bool()).unsqueeze(0), 0.0
        )
        target = camera.original_image * camera.mask.unsqueeze(0)
        pair = []
        for label, tensor in (("GT hair", target), ("clean Stage-1", prediction)):
            array = (
                tensor.clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
            )
            image = Image.fromarray(array).resize((250, 250), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (250, 276), "black")
            canvas.paste(image, (0, 26))
            ImageDraw.Draw(canvas).text(
                (5, 5), f"{camera.image_name} | {label}", fill="white"
            )
            pair.append(canvas)
        column = Image.new("RGB", (250, 552), "black")
        column.paste(pair[0], (0, 0))
        column.paste(pair[1], (0, 276))
        tiles.append(column)
    columns = min(4, len(tiles))
    rows = math.ceil(len(tiles) / columns)
    sheet = Image.new("RGB", (columns * 250, rows * 552), (24, 24, 24))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * 250, (index // columns) * 552))
    sheet.save(path)


def train(mp, opt, args) -> None:
    scene = Scene(args, shuffle=False)
    args.scene_extent = float(scene.cameras_extent)
    gaussians = scene.gaussians
    if args.clean_resume_ply:
        resume_ply = Path(args.clean_resume_ply)
        if not resume_ply.is_file():
            raise FileNotFoundError(f"Clean Stage-1 resume PLY not found: {resume_ply}")
        gaussians.load_ply(str(resume_ply))
        print(f"Loaded clean Stage-1 refinement source: {resume_ply}")
    # Every initialization point was carved from hair masks; keep the semantic
    # channel explicitly hair-only instead of relearning a face/hair partition.
    with torch.no_grad():
        gaussians._mask.fill_(math.log(0.99 / 0.01))
    opt.mask_lr = 0.0
    gaussians.training_setup(opt)
    background = torch.zeros(3, dtype=torch.float32, device="cuda")

    provenance_path = Path(mp.source_path) / "clean_scaffold_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    calibration_names = {
        Path(name).stem for name in provenance["calibration_images"]
    }
    cameras = scene.getCameras()
    calibration_cameras = [
        camera for camera in cameras if camera.image_name in calibration_names
    ]
    topology_fit_cameras = [
        camera for camera in cameras if camera.image_name not in calibration_names
    ]
    if (
        len(calibration_cameras) != len(calibration_names)
        or not topology_fit_cameras
    ):
        raise RuntimeError("Clean-scaffold fit/calibration camera split is inconsistent")
    hull = HairVisualHull(
        Path(mp.source_path),
        provenance,
        args.clean_hull_dilation,
        "cuda",
        args.clean_head_occlusion_margin,
    )
    head_masks = {
        Path(entry["name"]).stem: entry["head_mask"]
        for entry in hull.fit_entries + hull.calibration_entries
    }
    for camera in cameras:
        camera.head_occlusion_mask = head_masks[camera.image_name]
    warm_gamut_calibration = None
    if args.clean_enforce_warm_hair_gamut:
        warm_gamut_calibration = _estimate_warm_hair_gamut(cameras, args)
        print(
            "Warm hair gamut calibration: "
            + json.dumps(warm_gamut_calibration, sort_keys=True)
        )

    output = Path(args.model_path)
    writer = SummaryWriter(log_dir=str(output / "clean_tensorboard"))
    log_path = output / "clean_training.jsonl"
    event_path = output / "topology_events.jsonl"
    fit_stack = []
    fit_stack_role = None
    events = []
    pending_topology = None
    progress = tqdm(range(1, opt.iterations + 1), desc="Clean hair Stage-1")
    started = time.time()
    for iteration in progress:
        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        # All input views supervise ordinary Stage-1 optimization.  The four
        # internal calibration views are withheld only during a pending
        # topology probation, so they measure the incremental children rather
        # than becoming a permanently unseen training subset.
        stack_role = "topology_fit8" if pending_topology is not None else "train12"
        camera_pool = topology_fit_cameras if pending_topology is not None else cameras
        if not fit_stack or fit_stack_role != stack_role:
            fit_stack = camera_pool.copy()
            random.shuffle(fit_stack)
            fit_stack_role = stack_role
        camera = fit_stack.pop()
        include_orientation = iteration % args.clean_orientation_every == 0
        total, terms, package = _hair_image_loss(
            gaussians, camera, background, args, include_orientation
        )
        total.backward()
        if pending_topology is not None:
            _freeze_parent_gradients(
                gaussians, pending_topology["before_count"]
            )
        with torch.no_grad():
            if iteration < opt.densify_until_iter:
                gaussians.update_densification_stats(
                    package["viewspace_points"],
                    package["radii"],
                    package["visibility_filter"],
                )
            gaussians.optimizer.step()
            appearance_projection = _project_appearance_to_feasible_set(
                gaussians, args
            )
            probation_opacity_clamped = 0
            if pending_topology is not None:
                probation_opacity_clamped = _cap_probation_child_opacity(
                    gaussians,
                    pending_topology["before_count"],
                    args.clean_child_probation_max_alpha,
                )
            scale_projection = _project_scales_to_feasible_set(
                gaussians,
                hull,
                args,
                check_projected_radius=(
                    iteration % max(args.clean_scale_clamp_every, 1) == 0
                ),
            )
            if pending_topology is not None:
                _restore_parent_values(
                    gaussians, pending_topology["frozen_parent_values"]
                )
            gaussians.optimizer.zero_grad(set_to_none=True)

            event = None
            if (
                pending_topology is not None
                and iteration >= pending_topology["decision_iteration"]
            ):
                event = _finalize_topology_event(
                    iteration,
                    pending_topology,
                    gaussians,
                    hull,
                    topology_fit_cameras,
                    calibration_cameras,
                    background,
                    opt,
                    args,
                )
                pending_topology = None
                events.append(event)
                with event_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(event) + "\n")
            if (
                pending_topology is None
                and iteration < opt.densify_until_iter
                and iteration > opt.densify_from_iter
                and iteration % opt.densification_interval == 0
            ):
                pending_topology = _begin_topology_event(
                    iteration,
                    gaussians,
                    hull,
                    topology_fit_cameras,
                    calibration_cameras,
                    background,
                    opt,
                    args,
                )

            row = {
                "iteration": iteration,
                "camera": camera.image_name,
                "loss": float(total.item()),
                "gaussians": int(gaussians.get_xyz.shape[0]),
                **appearance_projection,
                "probation_opacity_clamped": probation_opacity_clamped,
                **scale_projection,
                **{key: float(value.item()) for key, value in terms.items()},
            }
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            for key, value in row.items():
                if isinstance(value, (float, int)) and key != "iteration":
                    writer.add_scalar(f"train/{key}", value, iteration)
            if event is not None:
                writer.add_scalar(
                    "topology/accepted", int(event["accepted"]), iteration
                )
                writer.add_scalar(
                    "topology/calibration_risk", event["after_mean_risk"], iteration
                )
            if iteration % 20 == 0:
                progress.set_postfix(
                    loss=f"{total.item():.4f}", points=gaussians.get_xyz.shape[0]
                )
            # Never publish a checkpoint containing an uncalibrated topology
            # proposal.  The final post-loop save below handles a pending event
            # only after it has been accepted or rolled back.
            if (
                (iteration % args.save_frequency == 0 or iteration == opt.iterations)
                and pending_topology is None
            ):
                scene.save(iteration)

    if pending_topology is not None:
        event = _finalize_topology_event(
            opt.iterations,
            pending_topology,
            gaussians,
            hull,
            topology_fit_cameras,
            calibration_cameras,
            background,
            opt,
            args,
        )
        events.append(event)
        with event_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event) + "\n")
        # The end-of-loop save may have captured the uncalibrated proposal.
        scene.save(opt.iterations)

    final_calibration = _calibration_metrics(
        gaussians, calibration_cameras, background, args
    )
    scales = torch.sort(gaussians.get_scaling, dim=1, descending=True).values
    final_rgb = 0.5 + _SH_C0 * gaussians._features_dc
    final_rg_gap = final_rgb[..., 0] - final_rgb[..., 1]
    final_gb_gap = final_rgb[..., 1] - final_rgb[..., 2]
    metadata = {
        "schema": "unifur-clean-hair-stage1-training-v1",
        "source_dataset": str(Path(mp.source_path).resolve()),
        "fit_images": [camera.image_name + ".png" for camera in cameras],
        "topology_fit_images": [
            camera.image_name + ".png" for camera in topology_fit_cameras
        ],
        "calibration_images": [
            camera.image_name + ".png" for camera in calibration_cameras
        ],
        "final_gaussians": int(gaussians.get_xyz.shape[0]),
        "scene_extent": args.scene_extent,
        "world_scale_cap": args.clean_world_scale_cap * args.scene_extent,
        "major_scale_p99": float(torch.quantile(scales[:, 0], 0.99).item()),
        "anisotropy_p99": float(
            torch.quantile(scales[:, 0] / scales[:, 1].clamp_min(1e-8), 0.99).item()
        ),
        "rgb_min": float(final_rgb.min().item()),
        "rgb_max": float(final_rgb.max().item()),
        "rgb_out_of_bounds": int(
            ((final_rgb < args.clean_rgb_min) | (final_rgb > args.clean_rgb_max))
            .any(dim=-1)
            .sum()
            .item()
        ),
        "warm_gamut_violations": int(
            (
                (final_rgb[..., 0] < final_rgb[..., 1])
                | (final_rgb[..., 1] < final_rgb[..., 2])
            )
            .sum()
            .item()
        ),
        "warm_chroma_violations": int(
            (
                (final_rg_gap > args.clean_warm_rg_gap_cap + 1e-6)
                | (final_gb_gap > args.clean_warm_gb_gap_cap + 1e-6)
            )
            .sum()
            .item()
        ) if args.clean_enforce_warm_hair_gamut else 0,
        "warm_gamut_calibration": warm_gamut_calibration,
        "accepted_topology_events": int(sum(row["accepted"] for row in events)),
        "rejected_topology_events": int(sum(not row["accepted"] for row in events)),
        "final_calibration": final_calibration,
        "elapsed_seconds": time.time() - started,
        "test4_used_for_training_or_calibration": False,
        "ground_truth_hair_geometry_used": False,
        "head_occlusion_aware": True,
        "resume_ply": args.clean_resume_ply or None,
        "arguments": {
            key: value
            for key, value in vars(args).items()
            if isinstance(value, (str, int, float, bool, type(None)))
        },
    }
    (output / "clean_stage1_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    _save_review(output / "clean_review_fit.png", gaussians, cameras, background)
    _save_review(
        output / "clean_review_calibration.png",
        gaussians,
        calibration_cameras,
        background,
    )
    writer.close()
    print(json.dumps(metadata, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mp = ModelParams(parser)
    op = OptimizationParams(parser)
    GeneralParams(parser)
    parser.add_argument("--clean_outside_rgb_weight", type=float, default=0.5)
    parser.add_argument("--clean_local_outside_blend", type=float, default=0.75)
    parser.add_argument("--clean_local_outside_padding", type=int, default=12)
    parser.add_argument("--clean_rgb_min", type=float, default=0.0)
    parser.add_argument("--clean_rgb_max", type=float, default=1.0)
    parser.add_argument("--clean_resume_ply", type=str, default="")
    parser.add_argument("--clean_head_occlusion_margin", type=float, default=0.002)
    parser.add_argument(
        "--clean_enforce_warm_hair_gamut", action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--clean_warm_gamut_quantile", type=float, default=0.999)
    parser.add_argument("--clean_warm_rg_gap_cap", type=float, default=-1.0)
    parser.add_argument("--clean_warm_gb_gap_cap", type=float, default=-1.0)
    parser.add_argument("--clean_coverage_weight", type=float, default=0.6)
    parser.add_argument("--clean_spill_weight", type=float, default=0.8)
    parser.add_argument("--clean_dice_weight", type=float, default=0.5)
    parser.add_argument("--clean_dssim_weight", type=float, default=0.2)
    parser.add_argument("--clean_orientation_weight", type=float, default=2.0)
    parser.add_argument("--clean_orientation_every", type=int, default=2)
    parser.add_argument("--clean_scale_weight", type=float, default=0.05)
    parser.add_argument("--clean_anisotropy_weight", type=float, default=0.02)
    parser.add_argument("--clean_cross_section_weight", type=float, default=0.01)
    parser.add_argument("--clean_world_scale_cap", type=float, default=0.04)
    parser.add_argument("--clean_min_anisotropy", type=float, default=2.5)
    parser.add_argument("--clean_max_anisotropy", type=float, default=16.0)
    parser.add_argument("--clean_hard_max_anisotropy", type=float, default=24.0)
    parser.add_argument("--clean_max_projected_radius", type=float, default=48.0)
    parser.add_argument("--clean_scale_clamp_every", type=int, default=20)
    parser.add_argument("--clean_min_hull_support", type=int, default=4)
    parser.add_argument("--clean_hull_dilation", type=int, default=3)
    parser.add_argument("--clean_max_gaussians", type=int, default=350_000)
    parser.add_argument("--clean_max_new_per_event", type=int, default=2_000)
    parser.add_argument("--clean_new_child_alpha", type=float, default=0.005)
    parser.add_argument("--clean_child_total_alpha_budget", type=float, default=5.0)
    parser.add_argument("--clean_child_jitter_scale", type=float, default=0.10)
    parser.add_argument("--clean_child_scale_factor", type=float, default=0.5)
    parser.add_argument(
        "--clean_child_probation_max_alpha", type=float, default=0.02
    )
    parser.add_argument(
        "--clean_child_min_calibration_support", type=int, default=4
    )
    parser.add_argument("--clean_topology_probation", type=int, default=100)
    parser.add_argument("--clean_topology_min_fit_gain", type=float, default=0.000001)
    parser.add_argument("--clean_calibration_view_margin", type=float, default=0.002)
    parser.add_argument("--clean_calibration_mean_margin", type=float, default=0.0005)
    args = parser.parse_args(sys.argv[1:])

    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    torch.cuda.set_device(0)
    enable_accelerated_rasterization()
    output = Path(args.model_path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    prepare_output_path(args)
    train(mp.extract(args), op.extract(args), args)


if __name__ == "__main__":
    main()
