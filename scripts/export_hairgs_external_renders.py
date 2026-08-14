#!/usr/bin/env python3
"""Convert official Hair-GS RGB/mask renders to the common evaluator schema."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--camera-dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument(
        "--camera-manifest",
        type=Path,
        help="Frozen protocol manifest used to preserve semantic view indices.",
    )
    parser.add_argument(
        "--output-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        help="Resize official renders to the frozen evaluation raster.",
    )
    parser.add_argument(
        "--hairgs-root",
        type=Path,
        default=Path("/home/aoki/fur_hair_baselines/hair-gs"),
    )
    args = parser.parse_args()

    sys.path.insert(0, str(args.hairgs_root.resolve()))
    from data.colmap import (  # type: ignore[import-not-found]
        read_extrinsics_binary,
        read_extrinsics_text,
    )

    binary = args.camera_dataset / "sparse/0/images.bin"
    cameras = (
        read_extrinsics_binary(binary)
        if binary.is_file()
        else read_extrinsics_text(args.camera_dataset / "sparse/0/images.txt")
    )
    image_names = sorted(Path(camera.name).name for camera in cameras.values())
    # Hair-GS render.py calls safe_state() and then constructs Scene with
    # shuffle=True.  Scene first sorts names, then Python's seeded shuffle
    # determines the 00000.png order.  Reproduce that exact order instead of
    # pairing numbered renders with the pre-shuffle lexical list.
    random.Random(0).shuffle(image_names)
    view_indices = {name: index for index, name in enumerate(image_names)}
    if args.camera_manifest is not None:
        protocol = json.loads(args.camera_manifest.read_text(encoding="utf-8"))
        view_indices = {
            str(item["image"]): int(item["view_index"])
            for item in protocol["observations"]
        }
        if set(view_indices) != set(image_names):
            raise ValueError(
                "Hair-GS camera dataset and frozen protocol manifest names differ"
            )
    render_root = args.model_dir / "render/train" / f"iteration_{args.iteration}"
    rgb_dir = render_root / "renders/rgb"
    mask_dir = render_root / "renders/mask_foreground"
    output = args.output_dir.resolve()
    arrays = output / "arrays"
    arrays.mkdir(parents=True, exist_ok=True)

    observations = []
    render_size = None
    for index, image_name in enumerate(image_names):
        rendered_name = f"{index:05d}.png"
        rgb_path = rgb_dir / rendered_name
        mask_path = mask_dir / rendered_name
        if not rgb_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(f"Missing Hair-GS render pair for {rendered_name}")
        rgb_image = Image.open(rgb_path).convert("RGB")
        mask_image = Image.open(mask_path).convert("L")
        if args.output_size is not None:
            target_size = tuple(args.output_size)
            rgb_image = rgb_image.resize(target_size, Image.Resampling.LANCZOS)
            # Alpha is a continuous rasterized coverage map, not a categorical
            # segmentation label; bilinear downsampling preserves that coverage.
            mask_image = mask_image.resize(target_size, Image.Resampling.BILINEAR)
        rgb = np.asarray(rgb_image, dtype=np.float32) / 255.0
        mask = np.asarray(mask_image, dtype=np.float32) / 255.0
        if render_size is None:
            render_size = [int(rgb.shape[1]), int(rgb.shape[0])]
        array_path = arrays / f"{Path(image_name).stem}.npz"
        np.savez_compressed(array_path, rgb=rgb, mask=mask)
        observations.append(
            {
                "image": image_name,
                "frame_index": 0,
                "view_index": view_indices[image_name],
                "array": str(array_path),
                "source_rgb": str(rgb_path.resolve()),
                "source_mask": str(mask_path.resolve()),
            }
        )

    manifest = {
        "schema": "external-render-manifest-v1",
        "status": "complete",
        "method": "Hair-GS",
        "camera_dataset": str(args.camera_dataset.resolve()),
        "model_dir": str(args.model_dir.resolve()),
        "iteration": args.iteration,
        "renderer_camera_order": image_names,
        "source_render_size": [
            int(Image.open(rgb_dir / "00000.png").width),
            int(Image.open(rgb_dir / "00000.png").height),
        ],
        "render_size": render_size,
        "image_count": len(observations),
        "observations": observations,
    }
    manifest_path = output / "render_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
