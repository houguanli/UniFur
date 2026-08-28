#!/usr/bin/env python3
"""Train the auditable HairGS shell/fin/strand covariance extension."""

from __future__ import annotations

import json
import math
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import scene as scene_module
import train as upstream_train
from arguments import GeneralParams, ModelParams, OptimizationParams
from gaussian_renderer import render
from loss.losses import bidirectional_angle_difference, create_window
from utils import enable_accelerated_rasterization, prepare_output_path, safe_state

from hairgs_hybrid.hybrid_model import HybridHairGaussianModel, HybridModelConfig


def add_hybrid_arguments(parser: ArgumentParser) -> None:
    parser.add_argument("--route-sidecar", type=Path, required=True)
    parser.add_argument(
        "--hybrid-init-mode",
        choices=("adaptive", "all_strand", "neutral"),
        default="adaptive",
    )
    parser.add_argument(
        "--hybrid-route-mode", choices=("soft", "hard"), default="soft"
    )
    parser.add_argument("--hybrid-temperature", type=float, default=0.75)
    parser.add_argument("--hybrid-temperature-final", type=float, default=0.75)
    parser.add_argument("--hybrid-temperature-anneal-steps", type=int, default=0)
    parser.add_argument("--hybrid-neutral-activation-steps", type=int, default=1000)
    parser.add_argument("--hybrid-topology-hard-forward-start", type=int, default=-1)
    parser.add_argument(
        "--hybrid-route-schedule", choices=("constant", "staged"), default="constant"
    )
    parser.add_argument("--hybrid-route-warmup-steps", type=int, default=0)
    parser.add_argument("--hybrid-shell-strength", type=float, default=0.16)
    parser.add_argument("--hybrid-fin-strength", type=float, default=0.20)
    parser.add_argument("--hybrid-strength-max", type=float, default=0.80)
    parser.add_argument("--hybrid-roll-max", type=float, default=0.35)
    parser.add_argument("--hybrid-route-lr", type=float, default=2.5e-3)
    parser.add_argument("--hybrid-shape-lr", type=float, default=1.0e-3)
    parser.add_argument("--hybrid-roll-lr", type=float, default=5.0e-4)
    parser.add_argument("--lambda-hybrid-mask-bce", type=float, default=0.02)
    parser.add_argument("--lambda-hybrid-mask-fp", type=float, default=0.04)
    parser.add_argument("--lambda-hybrid-mask-fn", type=float, default=0.06)
    parser.add_argument("--lambda-hybrid-hole", type=float, default=0.03)
    parser.add_argument("--lambda-hybrid-overflow-peak", type=float, default=0.10)
    parser.add_argument("--lambda-hybrid-background-rgb", type=float, default=0.0)
    parser.add_argument("--hybrid-mask-temperature", type=float, default=0.05)
    parser.add_argument("--lambda-hybrid-route-cost", type=float, default=2.0e-4)
    parser.add_argument("--lambda-hybrid-route-entropy", type=float, default=5.0e-5)
    parser.add_argument("--lambda-hybrid-route-explore", type=float, default=0.0)
    parser.add_argument("--lambda-hybrid-route-prior", type=float, default=0.0)
    parser.add_argument("--lambda-hybrid-budget", type=float, default=0.0)
    parser.add_argument("--hybrid-target-active-splats", type=int, default=0)
    parser.add_argument("--lambda-hybrid-shape-reg", type=float, default=2.0e-5)
    parser.add_argument("--lambda-hybrid-center-anchor", type=float, default=0.0)
    parser.add_argument("--lambda-hybrid-length-anchor", type=float, default=0.0)
    parser.add_argument("--lambda-hybrid-width-anchor", type=float, default=0.0)


def model_config(args) -> HybridModelConfig:
    return HybridModelConfig(
        route_sidecar=Path(args.route_sidecar),
        init_mode=args.hybrid_init_mode,
        route_mode=args.hybrid_route_mode,
        temperature=args.hybrid_temperature,
        temperature_final=args.hybrid_temperature_final,
        temperature_anneal_steps=args.hybrid_temperature_anneal_steps,
        neutral_activation_steps=args.hybrid_neutral_activation_steps,
        topology_hard_forward_start=args.hybrid_topology_hard_forward_start,
        shell_strength_init=args.hybrid_shell_strength,
        fin_strength_init=args.hybrid_fin_strength,
        strength_max=args.hybrid_strength_max,
        roll_residual_max=args.hybrid_roll_max,
        route_lr=args.hybrid_route_lr,
        shape_lr=args.hybrid_shape_lr,
        roll_lr=args.hybrid_roll_lr,
    )


def _route_schedule(
    iteration: int, total_steps: int, warmup_steps: int, mode: str
) -> tuple[float, float]:
    """Return (commitment, exploration) without a case-specific route quota."""

    if mode == "constant":
        return 1.0, 0.0
    if mode != "staged":
        raise ValueError(f"Unsupported route schedule: {mode}")
    if warmup_steps < 0 or total_steps <= 0:
        raise ValueError("Invalid staged route schedule")
    warmup = min(warmup_steps, max(total_steps - 1, 0))
    exploration = max(0.0, 1.0 - float(iteration) / float(max(warmup, 1)))
    commitment = max(
        0.0,
        min(
            1.0,
            float(iteration - warmup) / float(max(total_steps - warmup, 1)),
        ),
    )
    # A smooth ramp avoids making route cost, rather than reconstruction,
    # decide ownership at the first non-warmup step.
    commitment = commitment * commitment * (3.0 - 2.0 * commitment)
    return commitment, exploration


def _masked_l1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    weight = mask[None].to(dtype=prediction.dtype)
    # Preserve HairGS's native full-frame gradient scale.  Foreground
    # normalization would amplify a 4%-coverage frontal view by roughly 25x
    # while retaining the original appearance/geometry learning rates.
    return (torch.abs(prediction - target) * weight).sum() / float(
        prediction.numel()
    )


def _masked_dssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    window_size: int = 11,
) -> torch.Tensor:
    """Differentiable SSIM averaged only over windows touching GT hair."""
    image1 = prediction[None]
    image2 = target[None]
    channel = int(prediction.shape[0])
    window = create_window(window_size, channel).to(
        device=prediction.device, dtype=prediction.dtype
    )
    padding = window_size // 2
    mu1 = F.conv2d(image1, window, padding=padding, groups=channel)
    mu2 = F.conv2d(image2, window, padding=padding, groups=channel)
    mu1_sq = mu1.square()
    mu2_sq = mu2.square()
    mu12 = mu1 * mu2
    sigma1_sq = F.conv2d(
        image1.square(), window, padding=padding, groups=channel
    ) - mu1_sq
    sigma2_sq = F.conv2d(
        image2.square(), window, padding=padding, groups=channel
    ) - mu2_sq
    sigma12 = F.conv2d(
        image1 * image2, window, padding=padding, groups=channel
    ) - mu12
    c1 = 0.01**2
    c2 = 0.03**2
    ssim_map = ((2.0 * mu12 + c1) * (2.0 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    # Include local windows whose support touches hair, but do not let the large
    # black background or the face hole dominate the structural image loss.
    support = F.max_pool2d(
        mask[None, None].to(dtype=prediction.dtype),
        kernel_size=window_size,
        stride=1,
        padding=padding,
    )
    return ((1.0 - ssim_map) * support).sum() / float(ssim_map.numel())


def _hair_orientation_loss(gaussians, camera) -> torch.Tensor:
    """Orientation supervision with fixed head/body used only as occluders."""
    background = torch.zeros(3, device="cuda")
    orientation_color = gaussians.get_orientation * gaussians.hair_route_mask[:, None]
    orientation_world = render(
        camera, gaussians, background, override_color=orientation_color
    )["render"].permute(1, 2, 0)
    flattened = orientation_world.flatten(0, 1)
    orientation_view = flattened @ camera.world_view_transform[:3, :3]
    orientation_pixel = orientation_view[:, :2]
    orientation_pixel = orientation_pixel / (
        torch.linalg.vector_norm(orientation_pixel, dim=1, keepdim=True)
        + gaussians.min_val
    )
    x = orientation_pixel[:, 0]
    y = orientation_pixel[:, 1]
    y = torch.where(y < gaussians.min_val, y + gaussians.min_val, y)
    theta = torch.atan2(x, y)
    theta = torch.where(theta < 0, theta + np.pi, theta).reshape(
        orientation_world.shape[:2]
    )
    target_mask = camera.mask
    if target_mask is None or not torch.any(target_mask):
        return torch.zeros((), device=theta.device)
    difference = bidirectional_angle_difference(
        theta[target_mask], camera.orientation_field[target_mask]
    )
    confidence = camera.orientation_confidence[target_mask]
    # Match official HairGS normalization exactly; confidence is a weight, not
    # a second sampling distribution to normalize away.
    return (difference * confidence).mean()


def _hair_smoothness_loss(gaussians, threshold_degrees: float = 30.0):
    adjacent = gaussians.hair_adjacent_segment_pairs
    if adjacent.numel() == 0:
        return torch.zeros((), device=gaussians._endpoints.device)
    directions = gaussians.get_orientation[adjacent]
    cosine = torch.abs((directions[:, 0] * directions[:, 1]).sum(dim=1))
    angles = torch.acos(torch.clamp(cosine, -1.0 + 1.0e-6, 1.0 - 1.0e-6))
    excess = F.relu(angles - math.radians(threshold_degrees))
    return excess.square().mean()


def _balanced_visibility_loss(prediction: torch.Tensor, target: torch.Tensor):
    prediction = prediction.clamp(1.0e-5, 1.0 - 1.0e-5)
    foreground = target.sum().clamp_min(1.0)
    background = (1.0 - target).sum().clamp_min(1.0)
    positive_bce = -(target * torch.log(prediction)).sum() / foreground
    negative_bce = -((1.0 - target) * torch.log(1.0 - prediction)).sum() / background
    return 0.5 * (positive_bce + negative_bce)


def _base_reference_image(gaussians, camera, image: torch.Tensor) -> torch.Tensor:
    """Render and cache the frozen non-hair base for one training camera.

    Comparing against black outside a hair-only mask is correct for Panda but
    would incorrectly penalize the face/head in WCurly.  The base-only render
    makes the spill term measure *additional hair radiance or occlusion* in
    non-hair pixels for both protocols.
    """
    if torch.all(gaussians.hair_route_mask):
        return torch.zeros_like(image)
    key = (
        int(getattr(camera, "uid", -1)),
        str(getattr(camera, "image_name", "")),
        int(camera.image_height),
        int(camera.image_width),
    )
    cached = gaussians._base_reference_cache.get(key)
    if cached is None:
        with torch.no_grad(), gaussians.base_only_render_mode():
            cached = render(
                camera, gaussians, torch.zeros(3, device=image.device)
            )["render"].detach()
        gaussians._base_reference_cache[key] = cached
    return cached


def hybrid_loss(gaussians, image, camera, args):
    if not isinstance(gaussians, HybridHairGaussianModel):
        raise TypeError("Hybrid training constructed a non-hybrid HairGS model")

    if camera.mask is None:
        raise RuntimeError("Hair-only training requires a per-view hair mask")

    target_mask = camera.float_mask.float()
    target_image = camera.original_image
    # The full composite render is retained so the fixed head/body correctly
    # occludes back-side hair.  Only GT hair pixels contribute RGB gradients;
    # base gradients are independently zeroed by HybridHairGaussianModel.
    l1 = _masked_l1(image, target_image, target_mask)
    dssim = _masked_dssim(
        image * target_mask[None], target_image * target_mask[None], target_mask
    )
    loss = max(0.0, 1.0 - args.lambda_dssim) * l1 + args.lambda_dssim * dssim
    loss_dict = {"hair_l1": l1, "hair_dssim": dssim}

    if args.lambda_orientation > 0:
        orientation = _hair_orientation_loss(gaussians, camera)
        loss = loss + args.lambda_orientation * orientation
        loss_dict["hair_orientation"] = orientation
    if args.lambda_smooth > 0:
        smoothness = _hair_smoothness_loss(gaussians)
        loss = loss + args.lambda_smooth * smoothness
        loss_dict["hair_smooth"] = smoothness

    # Exact hair-route colors with all opacities retained: head/body emits zero
    # but still acts as a fixed occluder.  This distinguishes valid back hair
    # from visible strands that actually leak onto the face/background.
    visibility = render(
        camera,
        gaussians,
        torch.zeros(3, device="cuda"),
        override_color=gaussians.exact_hair_color,
    )["render"][0].clamp(1.0e-5, 1.0 - 1.0e-5)
    foreground = target_mask.sum().clamp_min(1.0)
    background = (1.0 - target_mask).sum().clamp_min(1.0)
    mask_bce = _balanced_visibility_loss(visibility, target_mask)
    mask_fp = (visibility * (1.0 - target_mask)).sum() / background
    mask_fn = ((1.0 - visibility) * target_mask).sum() / foreground
    deficit = ((1.0 - visibility) * target_mask)[None, None]
    local_hole = F.max_pool2d(deficit, kernel_size=31, stride=1, padding=15)
    soft_max_hole = (
        torch.logsumexp(12.0 * local_hole.flatten(), dim=0)
        - math.log(local_hole.numel())
    ) / 12.0
    hole = 0.5 * local_hole.mean() + 0.5 * soft_max_hole
    overflow = (visibility * (1.0 - target_mask))[None, None]
    local_overflow = F.max_pool2d(overflow, kernel_size=31, stride=1, padding=15)
    overflow_peak = (
        torch.logsumexp(12.0 * local_overflow.flatten(), dim=0)
        - math.log(local_overflow.numel())
    ) / 12.0
    loss = (
        loss
        + args.lambda_hybrid_mask_bce * mask_bce
        + args.lambda_hybrid_mask_fp * mask_fp
        + args.lambda_hybrid_mask_fn * mask_fn
        + args.lambda_hybrid_hole * hole
        + args.lambda_hybrid_overflow_peak * overflow_peak
    )
    loss_dict.update(
        {
            "hair_mask_bce": mask_bce,
            "hair_mask_fp": mask_fp,
            "hair_mask_fn": mask_fn,
            "hair_hole": hole,
            "hair_overflow_peak": overflow_peak,
        }
    )

    if args.lambda_hybrid_background_rgb > 0:
        base_reference = _base_reference_image(gaussians, camera, image)
        outside = (1.0 - target_mask)[None]
        spill_residual = (image - base_reference) * outside
        background_rgb_l1 = spill_residual.abs().sum() / float(image.numel())
        background_rgb_mse = spill_residual.square().sum() / float(image.numel())
        loss = loss + args.lambda_hybrid_background_rgb * background_rgb_l1
        loss_dict.update(
            {
                "hybrid_background_rgb_l1": background_rgb_l1,
                "hybrid_background_rgb_mse": background_rgb_mse,
            }
        )

    hair = gaussians.hair_route_mask
    probabilities = gaussians.routing_decision_probabilities
    if probabilities.numel() > 0:
        costs = gaussians.routing_route_costs.to(
            device=probabilities.device, dtype=probabilities.dtype
        )
        expected_active = (probabilities * costs).sum()
        route_cost = expected_active / float(probabilities.shape[0])
        if args.hybrid_target_active_splats > 0:
            target_active = torch.as_tensor(
                float(args.hybrid_target_active_splats),
                device=probabilities.device,
                dtype=probabilities.dtype,
            )
            budget_loss = ((expected_active - target_active) / target_active).square()
        else:
            budget_loss = torch.zeros_like(route_cost)
        shape_reg = gaussians.effective_shape_regularizer()
        if gaussians.runtime_config.route_mode == "hard":
            # Ownership is immutable and one-hot, so route cost/entropy cannot
            # influence optimization.  Keep their diagnostic values while
            # excluding constant graph work from the training objective.
            route_entropy = torch.zeros_like(route_cost)
            loss = loss + args.lambda_hybrid_shape_reg * shape_reg
        else:
            commitment, exploration = _route_schedule(
                int(gaussians._hybrid_iteration),
                int(args.iterations),
                int(args.hybrid_route_warmup_steps),
                args.hybrid_route_schedule,
            )
            route_entropy = -(
                probabilities * torch.log(probabilities.clamp_min(1.0e-8))
            ).sum(dim=1).mean()
            route_explore = (
                probabilities
                * torch.log((3.0 * probabilities).clamp_min(1.0e-8))
            ).sum(dim=1).mean()
            prior = gaussians.route_prior
            if prior is None:
                route_prior = torch.zeros_like(route_explore)
            else:
                prior = prior[hair].to(
                    device=probabilities.device, dtype=probabilities.dtype
                )
                route_prior = (
                    prior
                    * (
                        torch.log(prior.clamp_min(1.0e-8))
                        - torch.log(probabilities.clamp_min(1.0e-8))
                    )
                ).sum(dim=1).mean()
            loss = (
                loss
                + commitment * args.lambda_hybrid_route_cost * route_cost
                + commitment * args.lambda_hybrid_route_entropy * route_entropy
                + exploration * args.lambda_hybrid_route_explore * route_explore
                + commitment * args.lambda_hybrid_route_prior * route_prior
                + commitment * args.lambda_hybrid_budget * budget_loss
                + args.lambda_hybrid_shape_reg * shape_reg
            )
            loss_dict.update(
                {
                    "hybrid_route_explore": route_explore,
                    "hybrid_route_prior": route_prior,
                    "hybrid_route_commitment": torch.as_tensor(
                        commitment, device=probabilities.device
                    ),
                    "hybrid_route_exploration_weight": torch.as_tensor(
                        exploration, device=probabilities.device
                    ),
                    "hybrid_temperature": torch.as_tensor(
                        gaussians.current_temperature, device=probabilities.device
                    ),
                }
            )
        loss_dict.update(
            {
                "hybrid_route_cost": route_cost,
                "hybrid_route_entropy": route_entropy,
                "hybrid_shape_reg": shape_reg,
                "hybrid_expected_active_splats": expected_active,
                "hybrid_budget_loss": budget_loss,
            }
        )
    geometry = gaussians.geometry_anchor_losses()
    loss = (
        loss
        + args.lambda_hybrid_center_anchor * geometry["center"]
        + args.lambda_hybrid_length_anchor * geometry["length"]
        + args.lambda_hybrid_width_anchor * geometry["width"]
    )
    loss_dict.update(
        {
            "hybrid_center_anchor": geometry["center"],
            "hybrid_length_anchor": geometry["length"],
            "hybrid_width_anchor": geometry["width"],
        }
    )
    return loss, loss_dict


def main() -> None:
    parser = ArgumentParser(description="HairGS hybrid shell/fin/strand training")
    model = ModelParams(parser)
    optimization = OptimizationParams(parser)
    general = GeneralParams(parser)
    add_hybrid_arguments(parser)
    args = parser.parse_args(sys.argv[1:])

    config = model_config(args)
    HybridHairGaussianModel.configure(config)
    # Scene resolves HairGaussianModel from its module globals at construction.
    # The subclass still passes every upstream isinstance check.
    scene_module.HairGaussianModel = HybridHairGaussianModel
    upstream_train.loss_function = hybrid_loss

    route_sidecar = Path(args.route_sidecar).resolve()
    args.route_sidecar = str(route_sidecar)
    prepare_output_path(args)
    audit = Path(args.model_path) / "audit/hybrid_config.json"
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit.write_text(
        json.dumps(
            {
                "schema": "hairgs-haironly-maskbounded-training-config-v1",
                "route_sidecar": str(route_sidecar),
                "model": config.__dict__ | {"route_sidecar": str(route_sidecar)},
                "loss": {
                    key: value
                    for key, value in vars(args).items()
                    if key.startswith("lambda_hybrid_")
                },
                "mask_membership_temperature": args.hybrid_mask_temperature,
                "route_schedule": {
                    "mode": args.hybrid_route_schedule,
                    "warmup_steps": args.hybrid_route_warmup_steps,
                },
                "hair_membership": "exact route_id >= 0",
                "rgb_supervision": "GT hair-mask pixels only, native full-frame gradient scale",
                "head_body_policy": "strictly frozen, retained only as occluder/background",
                "visibility_policy": "exact hair color with fixed head/body occlusion",
                "background_rgb_policy": (
                    "penalize route-induced RGB/occlusion outside the hair mask "
                    "relative to a cached frozen-base render"
                ),
                "topology_frozen_required": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    safe_state(args.quiet)
    enable_accelerated_rasterization()
    model_args = model.extract(args)
    optimization_args = optimization.extract(args)
    general_args = general.extract(args)
    upstream_train.op = optimization_args
    upstream_train.network_gui.init(general_args.ip, general_args.port)
    upstream_train.training(model_args, optimization_args, general_args, args)


if __name__ == "__main__":
    main()
