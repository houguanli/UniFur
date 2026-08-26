#!/usr/bin/env python3
"""Measure steady-state HairGS RGB render throughput without process startup."""

from __future__ import annotations

import argparse
import json
import statistics
from argparse import Namespace
from pathlib import Path

import torch


def load_args(model_dir: Path, dataset: Path) -> Namespace:
    args = eval(
        (model_dir / "cfg_args").read_text(encoding="utf-8"),
        {"Namespace": Namespace},
    )
    args.model_path = str(model_dir)
    args.source_path = str(dataset)
    args.resolution = 1
    args.data_device = "cuda"
    return args


def configure_hybrid(args: Namespace) -> None:
    import scene as scene_module

    from hairgs_hybrid.hybrid_model import HybridHairGaussianModel, HybridModelConfig

    HybridHairGaussianModel.configure(
        HybridModelConfig(
            route_sidecar=Path(args.route_sidecar) if args.route_sidecar else None,
            init_mode=getattr(args, "hybrid_init_mode", "adaptive"),
            temperature=getattr(args, "hybrid_temperature", 0.75),
            shell_strength_init=getattr(args, "hybrid_shell_strength", 0.16),
            fin_strength_init=getattr(args, "hybrid_fin_strength", 0.20),
            strength_max=getattr(args, "hybrid_strength_max", 0.80),
            roll_residual_max=getattr(args, "hybrid_roll_max", 0.35),
        )
    )
    scene_module.HairGaussianModel = HybridHairGaussianModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--kind", choices=("native", "hybrid"), required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    cli = parser.parse_args()

    args = load_args(cli.model_dir, cli.dataset)
    if cli.kind == "hybrid":
        configure_hybrid(args)

    from gaussian_renderer import render
    from scene import Scene

    with torch.no_grad():
        scene = Scene(args, shuffle=False)
        gaussians = scene.gaussians
        if hasattr(gaussians, "force_exact_hair_membership"):
            gaussians.force_exact_hair_membership(False)
        cameras = scene.getCameras()
        background = torch.zeros(3, dtype=torch.float32, device="cuda")

        for _ in range(cli.warmup):
            for camera in cameras:
                render(camera, gaussians, background)["render"]
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        timings = []
        for _ in range(cli.repeats):
            for camera in cameras:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                render(camera, gaussians, background)["render"]
                end.record()
                end.synchronize()
                timings.append(float(start.elapsed_time(end)))

        report = {
            "schema": "hairgs-render-benchmark-v1",
            "kind": cli.kind,
            "model_dir": str(cli.model_dir.resolve()),
            "dataset": str(cli.dataset.resolve()),
            "camera_count": len(cameras),
            "warmup_passes": cli.warmup,
            "measured_passes": cli.repeats,
            "frame_count": len(timings),
            "mean_ms_per_frame": statistics.fmean(timings),
            "median_ms_per_frame": statistics.median(timings),
            "p95_ms_per_frame": sorted(timings)[int(0.95 * (len(timings) - 1))],
            "fps_from_mean": 1000.0 / statistics.fmean(timings),
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        }
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
