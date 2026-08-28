#!/usr/bin/env python3
"""Pixel-level gate for HairGS-compatible route initializations."""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

import torch

from gaussian_renderer import render
from scene import Scene
from scene.hair_gaussian_model import HairGaussianModel

from hairgs_hybrid.hybrid_model import HybridHairGaussianModel, HybridModelConfig


def load_scene_args(model_dir: Path, dataset: Path) -> Namespace:
    text = (model_dir / "cfg_args").read_text(encoding="utf-8")
    args = eval(text, {"Namespace": Namespace})
    args.model_path = str(model_dir)
    args.source_path = str(dataset)
    args.resolution = 1
    args.data_device = "cuda"
    return args


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--input-ply", type=Path, required=True)
    parser.add_argument("--route-sidecar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1.0e-5)
    parser.add_argument(
        "--init-mode", choices=("all_strand", "neutral", "adaptive"), default="all_strand"
    )
    args = parser.parse_args()

    scene = Scene(load_scene_args(args.baseline_model, args.dataset), shuffle=False)
    cameras = scene.getCameras()
    extent = scene.cameras_extent
    sh_degree = scene.gaussians.max_sh_degree
    scene.gaussians = None
    torch.cuda.empty_cache()

    native = HairGaussianModel(sh_degree, extent, device="cuda")
    native.load_ply(args.input_ply)
    HybridHairGaussianModel.configure(
        HybridModelConfig(
            route_sidecar=args.route_sidecar,
            init_mode=args.init_mode,
            strength_max=0.0 if args.init_mode == "all_strand" else 0.8,
            shell_strength_init=0.0 if args.init_mode == "all_strand" else 0.16,
            fin_strength_init=0.0 if args.init_mode == "all_strand" else 0.20,
        )
    )
    hybrid = HybridHairGaussianModel(sh_degree, extent, device="cuda")
    hybrid.load_ply(args.input_ply)
    background = torch.zeros(3, device="cuda")

    rows = []
    maximum = 0.0
    scaling_max = float((native.get_scaling - hybrid.get_scaling).abs().max().item())
    rotation_max = float((native.get_rotation - hybrid.get_rotation).abs().max().item())
    with torch.no_grad():
        for index, camera in enumerate(cameras):
            native_rgb = render(camera, native, background)["render"]
            hybrid_rgb = render(camera, hybrid, background)["render"]
            difference = (native_rgb - hybrid_rgb).abs()
            value = float(difference.max().item())
            maximum = max(maximum, value)
            rows.append(
                {
                    "camera_index": index,
                    "camera_uid": int(camera.uid),
                    "max_abs_pixel": value,
                    "mean_abs_pixel": float(difference.mean().item()),
                }
            )

    report = {
        "schema": "hairgs-hybrid-equivalence-v1",
        "status": "pass" if maximum <= args.tolerance else "fail",
        "tolerance": args.tolerance,
        "input_ply": str(args.input_ply.resolve()),
        "route_sidecar": str(args.route_sidecar.resolve()),
        "init_mode": args.init_mode,
        "camera_count": len(rows),
        "max_abs_scaling": scaling_max,
        "max_abs_rotation": rotation_max,
        "max_abs_pixel": maximum,
        "per_camera": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
