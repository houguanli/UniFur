#!/usr/bin/env python3
"""Render a NeuralFur checkpoint on its official held-out Panda cameras."""

from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render_hair
from scene import GaussianModel, GaussianModelHair, Scene


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    model_params = ModelParams(parser)
    optimization_params = OptimizationParams(parser)
    pipeline_params = PipelineParams(parser)
    parser.add_argument("--hair_conf_path", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--pointcloud_path_head", required=True)
    parser.add_argument("--checkpoint_hair", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--iteration", type=int, default=20000)
    parser.add_argument("--scene_suffix", default="")
    parser.add_argument("--scale_factor", type=int, default=1)
    parser.add_argument("--resolution_val", type=int, nargs=2, default=[512, 288])
    parser.add_argument("--num_views", type=int, default=-1)
    parser.add_argument("--body_scale", type=float, default=0.0)
    parser.add_argument("--use_test_split", action="store_true")
    parser.add_argument("--max_observations", type=int, default=-1)
    args = parser.parse_args()

    with open(args.hair_conf_path, "r", encoding="utf-8") as file:
        replaced = str(yaml.load(file, Loader=yaml.Loader))
    replaced = replaced.replace("DATASET_TYPE", "monocular")
    replaced = replaced.replace("DATA_ROOT", args.data_root.rstrip("/"))
    hair_config = yaml.load(replaced, Loader=yaml.Loader)

    dataset = model_params.extract(args)
    opt = optimization_params.extract(args)
    pipe = pipeline_params.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    gaussians_hair = GaussianModelHair(dataset.source_path, hair_config, dataset.sh_degree)
    scene = Scene(
        dataset,
        gaussians,
        pointcloud_path=args.pointcloud_path_head,
        scene_suffix=args.scene_suffix,
        load_iteration=None,
        scale_factor=args.scale_factor,
        resolution=args.resolution_val,
        use_test_split=args.use_test_split,
        num_views=args.num_views,
    )
    gaussians.training_setup(opt)
    gaussians_hair.create_from_pcd(dataset.source_path, dataset.strand_scale)
    gaussians_hair.training_setup(opt, hair_config)
    model_parameters, checkpoint_iteration = torch.load(args.checkpoint_hair)
    gaussians_hair.restore(model_parameters, opt, hair_config)

    with torch.no_grad():
        gaussians.mask_precomp = gaussians.get_label2[..., 0] <= 0.5
        gaussians.xyz_precomp = gaussians.get_xyz[gaussians.mask_precomp].detach()
        gaussians.opacity_precomp = gaussians.get_opacity[gaussians.mask_precomp].detach()
        gaussians.scaling_precomp = gaussians.get_scaling[gaussians.mask_precomp].detach()
        gaussians.rotation_precomp = gaussians.get_rotation[gaussians.mask_precomp].detach()
        gaussians.cov3D_precomp = gaussians.get_covariance(1.0)[gaussians.mask_precomp].detach()
        gaussians.shs_view = (
            gaussians.get_features[gaussians.mask_precomp]
            .detach()
            .transpose(1, 2)
            .view(-1, 3, (gaussians.max_sh_degree + 1) ** 2)
        )
        gaussians_hair.initialize_gaussians_hair(args.iteration)

    background_values = (
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 100]
        if dataset.white_background
        else [0, 0, 0, 0, 0, 0, 0, 0, 0, 100]
    )
    background = torch.tensor(background_values, dtype=torch.float32, device="cuda")
    output = Path(args.output_dir).resolve()
    arrays_dir = output / "arrays"
    preview_dir = output / "preview"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    cameras = scene.getTestCameras()
    if args.max_observations > 0:
        cameras = cameras[: args.max_observations]
    report = {
        "schema": "neuralfur-heldout-render-v1",
        "method": "NeuralFur",
        "checkpoint": str(Path(args.checkpoint_hair).resolve()),
        "checkpoint_iteration": int(checkpoint_iteration),
        "render_size": list(map(int, args.resolution_val)),
        "status": "running",
        "completed": 0,
        "observations": [],
    }
    report_path = output / "render_manifest.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for index, camera in enumerate(cameras):
        with torch.no_grad():
            rendered = render_hair(
                camera,
                gaussians,
                gaussians_hair,
                pipe,
                background,
                body_scale=args.body_scale,
            )
        rgb_tensor = rendered["render"][:3].clamp(0.0, 1.0)[None]
        # NeuralFur supervises its foreground/hair mask with channel zero
        # (see train_latent_fur.py).  The second channel is an auxiliary
        # training signal rather than a comparable foreground alpha.
        mask_tensor = rendered["mask"][:1].clamp(0.0, 1.0)[None]
        target_height, target_width = args.resolution_val[1], args.resolution_val[0]
        if tuple(rgb_tensor.shape[-2:]) != (target_height, target_width):
            # NeuralFur's Scene camera loader may preserve the native image
            # size despite its resolution argument.  Emit the frozen shared
            # evaluation resolution explicitly for both color and alpha.
            rgb_tensor = F.interpolate(
                rgb_tensor, size=(target_height, target_width), mode="bilinear", align_corners=False
            )
            mask_tensor = F.interpolate(
                mask_tensor, size=(target_height, target_width), mode="bilinear", align_corners=False
            )
        rgb = rgb_tensor[0].permute(1, 2, 0).detach().cpu().numpy()
        mask = mask_tensor[0, 0].detach().cpu().numpy()
        name = str(camera.image_name)
        array_path = arrays_dir / f"{name}.npz"
        np.savez_compressed(
            array_path,
            rgb=rgb.astype(np.float16),
            mask=mask.astype(np.float16),
        )
        Image.fromarray(np.round(rgb * 255.0).astype(np.uint8)).save(
            preview_dir / f"{name}_rgb.png"
        )
        Image.fromarray(np.round(mask * 255.0).astype(np.uint8)).save(
            preview_dir / f"{name}_mask.png"
        )
        report["observations"].append(
            {
                "image": f"{name}.png",
                "frame_index": index,
                "view_index": int(name),
                "array": str(array_path),
            }
        )
        report["completed"] = index + 1
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["status"] = "complete"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "observations"}, indent=2))


if __name__ == "__main__":
    main()
