#!/usr/bin/env python3
"""Headless HairGS/3DGS training loop for datasets without strand-geometry GT.

This uses the pinned upstream Scene, models, renderer, losses, optimizer and
densification unchanged.  It replaces only upstream ``train.py`` orchestration,
whose final optional geometry-evaluation branch dereferences an undefined
``pred`` when ``hair_eval_data.npz`` is absent.
"""

from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path
from random import randint

import numpy as np
import torch

from arguments import GeneralParams, ModelParams, OptimizationParams
from gaussian_renderer import render
from loss import loss_function
from scene import HairGaussianModel, Scene
from utils import enable_accelerated_rasterization, prepare_output_path, safe_state


def train(mp, opt, args) -> None:
    scene = Scene(args, shuffle=True)
    gaussians = scene.gaussians
    gaussians.training_setup(opt)
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    metrics_path = Path(scene.model_path) / "training_metrics.jsonl"
    viewpoint_stack = None
    ema = 0.0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with metrics_path.open("a", encoding="utf-8", buffering=1) as metrics:
        for iteration in range(1, opt.iterations + 1):
            total_iteration = scene.loaded_iter + iteration
            start.record()
            gaussians.update_learning_rate(iteration)
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()
            if not viewpoint_stack:
                viewpoint_stack = scene.getCameras().copy()
            camera = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
            package = render(camera, gaussians, background)
            loss, loss_dict = loss_function(gaussians, package["render"], camera, args)
            loss.backward()
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
                gaussians.optimizer.zero_grad(set_to_none=True)
                end.synchronize()
                ema = 0.4 * float(loss) + 0.6 * ema
                if iteration == 1 or iteration % 100 == 0 or iteration == opt.iterations:
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
    args = parser.parse_args(sys.argv[1:])
    prepare_output_path(args)
    safe_state(args.quiet)
    enable_accelerated_rasterization()
    train(mp.extract(args), op.extract(args), args)


if __name__ == "__main__":
    main()
