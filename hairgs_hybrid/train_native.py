#!/usr/bin/env python3
"""Headless HairGS/3DGS training loop for datasets without strand-geometry GT.

This uses the pinned upstream Scene, models, renderer, losses, optimizer and
densification unchanged.  It replaces only upstream ``train.py`` orchestration,
whose final optional geometry-evaluation branch dereferences an undefined
``pred`` when ``hair_eval_data.npz`` is absent.
"""

from __future__ import annotations

import json
import math
import sys
from copy import copy
from argparse import ArgumentParser
from pathlib import Path
from random import randint

import numpy as np
import torch

from arguments import GeneralParams, ModelParams, OptimizationParams
from gaussian_renderer import render
from scene import HairGaussianModel, Scene
from utils import enable_accelerated_rasterization, prepare_output_path, safe_state

# Upstream loss defaults allocate CUDA tensors at import time. Load them only
# when native training actually begins; grouped callers install their own loss.
loss_function = None


def _native_crop_camera(camera, crop_size: int):
    """Return an off-axis native-pixel crop without resizing the source view.

    Full-frame 2K soft routing is dominated by rasterizer tile duplication,
    while downsampling erases exactly the thin residuals used for Strand
    birth.  This view keeps the original pixel pitch and changes only the
    projection/image window.  A foreground-bounding-box sampler covers the
    animal most of the time and a uniform branch retains background/spill
    supervision.
    """

    height = int(camera.image_height)
    width = int(camera.image_width)
    size = int(crop_size)
    if size <= 0 or size >= min(height, width):
        return camera

    bounds = getattr(camera, "_native_crop_mask_bounds", None)
    if bounds is None:
        if getattr(camera, "mask", None) is not None and torch.any(camera.mask):
            rows, columns = torch.where(camera.mask)
            bounds = (
                int(columns.min()),
                int(columns.max()),
                int(rows.min()),
                int(rows.max()),
            )
        else:
            bounds = (0, width - 1, 0, height - 1)
        camera._native_crop_mask_bounds = bounds

    if randint(0, 3) != 0:
        xmin, xmax, ymin, ymax = bounds
        center_x = randint(xmin, xmax)
        center_y = randint(ymin, ymax)
        x0 = min(max(center_x - size // 2, 0), width - size)
        y0 = min(max(center_y - size // 2, 0), height - size)
    else:
        x0 = randint(0, width - size)
        y0 = randint(0, height - size)

    cropped = copy(camera)
    cropped.original_image = camera.original_image[:, y0 : y0 + size, x0 : x0 + size]
    if getattr(camera, "mask", None) is not None:
        cropped.mask = camera.mask[y0 : y0 + size, x0 : x0 + size]
        cropped.float_mask = cropped.mask.to(torch.float32)
        cropped.masked_image = cropped.original_image.clone()
        cropped.masked_image[:, ~cropped.mask] = 0.0
    if getattr(camera, "orientation_field", None) is not None:
        cropped.orientation_field = camera.orientation_field[
            y0 : y0 + size, x0 : x0 + size
        ]
    if getattr(camera, "orientation_confidence", None) is not None:
        cropped.orientation_confidence = camera.orientation_confidence[
            y0 : y0 + size, x0 : x0 + size
        ]

    # Row-vector convention: [x,y,z,w] @ crop_transform.  The translation
    # term maps the selected full-frame NDC window back to [-1, 1].
    crop_transform = torch.eye(
        4, device=camera.full_proj_transform.device,
        dtype=camera.full_proj_transform.dtype,
    )
    crop_transform[0, 0] = float(width) / float(size)
    crop_transform[1, 1] = float(height) / float(size)
    crop_transform[3, 0] = float(width - 2 * x0 - size) / float(size)
    crop_transform[3, 1] = float(height - 2 * y0 - size) / float(size)
    cropped.projection_matrix = camera.projection_matrix @ crop_transform
    cropped.full_proj_transform = camera.full_proj_transform @ crop_transform
    cropped.FoVx = 2.0 * math.atan(
        math.tan(float(camera.FoVx) * 0.5) * float(size) / float(width)
    )
    cropped.FoVy = 2.0 * math.atan(
        math.tan(float(camera.FoVy) * 0.5) * float(size) / float(height)
    )
    cropped.image_width = size
    cropped.image_height = size
    cropped.image_name = f"{camera.image_name}@native_crop_{x0}_{y0}_{size}"
    cropped.native_crop_origin = (x0, y0)
    return cropped


def _update_learning_rate(gaussians, stage_iteration: int, total_iteration: int):
    """Keep optimizer stage time while preserving Hybrid curriculum time."""

    gaussians.update_learning_rate(stage_iteration)
    if hasattr(gaussians, "_hybrid_iteration"):
        gaussians._hybrid_iteration = int(total_iteration)
        if hasattr(gaussians, "_apply_handoff_learning_rates"):
            gaussians._apply_handoff_learning_rates()


def train(mp, opt, args, diagnostic=None) -> None:
    global loss_function
    if loss_function is None:
        from loss import loss_function
    scene = Scene(args, shuffle=True)
    gaussians = scene.gaussians
    gaussians.training_setup(opt)
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    metrics_path = Path(scene.model_path) / "training_metrics.jsonl"
    viewpoint_stack = None
    ema = 0.0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    log_frequency = int(getattr(args, "training_log_frequency", 100))
    if log_frequency <= 0:
        raise ValueError("training log frequency must be positive")
    if diagnostic is not None:
        diagnostic("initial", scene, total_iteration=scene.loaded_iter)

    with metrics_path.open("a", encoding="utf-8", buffering=1) as metrics:
        for iteration in range(1, opt.iterations + 1):
            total_iteration = scene.loaded_iter + iteration
            start.record()
            _update_learning_rate(gaussians, iteration, total_iteration)
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()
            if not viewpoint_stack:
                viewpoint_stack = scene.getCameras().copy()
            camera = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
            camera = _native_crop_camera(
                camera, int(getattr(args, "hybrid_native_crop_size", 0))
            )
            package = render(camera, gaussians, background)
            loss, loss_dict = loss_function(gaussians, package["render"], camera, args)
            if diagnostic is not None:
                diagnostic("loss", scene, total_iteration=total_iteration,
                           camera=camera, loss=loss, loss_terms=loss_dict)
            loss.backward()
            if diagnostic is not None:
                diagnostic("gradients", scene, total_iteration=total_iteration)
            end.record()

            with torch.no_grad():
                if iteration < opt.densify_until_iter:
                    gaussians.update_densification_stats(
                        package["viewspace_points"],
                        package["radii"],
                        package["visibility_filter"],
                    )
                    if (
                        iteration > opt.densify_from_iter
                        and iteration % opt.densification_interval == 0
                    ):
                        threshold = (
                            opt.prune_max_radii_2d
                            if iteration > opt.opacity_reset_interval
                            else None
                        )
                        gaussians.densification(scene.cameras_extent, threshold)
                    if iteration % opt.opacity_reset_interval == 0:
                        gaussians.reset_opacity()

                if isinstance(gaussians, HairGaussianModel):
                    if iteration % opt.merge_interval == 0:
                        gaussians.merging()
                    if iteration % opt.growth_interval == 0:
                        gaussians.growing()

                gaussians.optimizer.step()
                if diagnostic is not None:
                    diagnostic("step", scene, total_iteration=total_iteration)
                gaussians.optimizer.zero_grad(set_to_none=True)
                end.synchronize()
                ema = 0.4 * float(loss) + 0.6 * ema
                if iteration == 1 or iteration % log_frequency == 0 or iteration == opt.iterations:
                    row = {
                        "iteration": total_iteration,
                        "stage_iteration": iteration,
                        "loss": float(loss),
                        "ema_loss": ema,
                        "loss_terms": {
                            name: float(value) for name, value in loss_dict.items()
                        },
                        "splats": int(gaussians.get_xyz.shape[0]),
                        "iter_ms": float(start.elapsed_time(end)),
                        "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                    }
                    metrics.write(json.dumps(row) + "\n")
                    print(json.dumps(row), flush=True)

                if iteration % args.save_frequency == 0 or iteration == opt.iterations:
                    scene.gaussians = gaussians
                    scene.save(iteration)
                    if diagnostic is not None:
                        diagnostic("checkpoint", scene, total_iteration=total_iteration,
                                   camera=camera, loss=loss, loss_terms=loss_dict)

    print(
        json.dumps(
            {
                "status": "complete",
                "model_path": scene.model_path,
                "iteration": scene.loaded_iter + opt.iterations,
                "splats": int(gaussians.get_xyz.shape[0]),
            }
        )
    )


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    mp = ModelParams(parser)
    op = OptimizationParams(parser)
    GeneralParams(parser)
    parser.add_argument("--training-log-frequency", type=int, default=100)
    args = parser.parse_args(sys.argv[1:])
    prepare_output_path(args)
    safe_state(args.quiet)
    enable_accelerated_rasterization()
    train(mp.extract(args), op.extract(args), args)


if __name__ == "__main__":
    main()
