#!/usr/bin/env python3
"""Compare a neutral topology candidate bank against its dense HairGS teacher."""

from __future__ import annotations

import argparse
import json
import math
from argparse import Namespace
from pathlib import Path

import torch

from gaussian_renderer import render
from scene import Scene
from scene.hair_gaussian_model import HairGaussianModel

from hairgs_hybrid.hybrid_model import HybridHairGaussianModel, HybridModelConfig


def scene_args(model_dir: Path, dataset: Path) -> Namespace:
    args = eval((model_dir / "cfg_args").read_text(), {"Namespace": Namespace})
    args.model_path = str(model_dir)
    args.source_path = str(dataset)
    args.resolution = 1
    args.data_device = "cuda"
    return args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dense-ply", type=Path, required=True)
    parser.add_argument("--candidate-ply", type=Path, required=True)
    parser.add_argument("--route-sidecar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    scene = Scene(scene_args(args.baseline_model, args.dataset), shuffle=False)
    cameras = scene.getCameras()
    extent = scene.cameras_extent
    sh_degree = scene.gaussians.max_sh_degree
    scene.gaussians = None
    torch.cuda.empty_cache()
    dense = HairGaussianModel(sh_degree, extent, device="cuda")
    dense.load_ply(args.dense_ply)
    HybridHairGaussianModel.configure(
        HybridModelConfig(
            route_sidecar=args.route_sidecar,
            init_mode="neutral",
            route_mode="soft",
            temperature=1.5,
            neutral_activation_steps=1000,
        )
    )
    topology = HybridHairGaussianModel(sh_degree, extent, device="cuda")
    topology.load_ply(args.candidate_ply)
    background = torch.zeros(3, device="cuda")
    rows = []
    with torch.no_grad():
        for camera in cameras:
            target = render(camera, dense, background)["render"]
            prediction = render(camera, topology, background)["render"]
            difference = prediction - target
            mse = float(difference.square().mean())
            rows.append(
                {
                    "image": str(camera.image_name),
                    "psnr_to_dense_db": -10.0 * math.log10(max(mse, 1.0e-12)),
                    "mean_abs": float(difference.abs().mean()),
                    "max_abs": float(difference.abs().max()),
                }
            )
    report = {
        "schema": "hairgs-topology-initialization-audit-v1",
        "dense_segments": int(dense.endpoint_pairs.shape[0]),
        "candidate_segments": int(topology.endpoint_pairs.shape[0]),
        "routing_groups": int(topology.routing_decision_probabilities.shape[0]),
        "initial_expected_active_splats": float(topology.expected_active_splats),
        "camera_count": len(rows),
        "mean_psnr_to_dense_db": float(
            sum(row["psnr_to_dense_db"] for row in rows) / len(rows)
        ),
        "mean_abs": float(sum(row["mean_abs"] for row in rows) / len(rows)),
        "max_abs": float(max(row["max_abs"] for row in rows)),
        "per_camera": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
