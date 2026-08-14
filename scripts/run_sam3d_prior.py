#!/usr/bin/env python3
"""Build a SAM3D prior from an RGBA reference image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from dpd3dgs_animal.sam3d_adapter import Sam3DObjectAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-rgba", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sam3d-root", default="/home/aoki/sam3d-obj")
    parser.add_argument("--checkpoint-root", default=None)
    parser.add_argument("--tag", default="hf")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    input_dir = out_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    rgba = np.asarray(Image.open(args.reference_rgba).convert("RGBA"))
    rgb_path = input_dir / "reference_rgb.png"
    mask_path = input_dir / "reference_mask.png"
    Image.fromarray(rgba[..., :3]).save(rgb_path)
    Image.fromarray(rgba[..., 3]).save(mask_path)

    adapter = Sam3DObjectAdapter(
        sam3d_root=args.sam3d_root,
        tag=args.tag,
        checkpoint_root=args.checkpoint_root,
    )
    result = adapter.reconstruct(
        rgb_path,
        mask_path,
        out_dir / "reconstruction",
        seed=args.seed,
    )
    report = {
        "schema": "dpd3dgs-sam3d-prior-v1",
        "reference_rgba": str(Path(args.reference_rgba).resolve()),
        "reference_rgb": str(rgb_path.resolve()),
        "reference_mask": str(mask_path.resolve()),
        "seed": args.seed,
        "camera_ply": str(result.camera_ply_path) if result.camera_ply_path else None,
        "camera_mesh": str(result.camera_mesh_path) if result.camera_mesh_path else None,
        "camera_metadata": str(result.metadata_path) if result.metadata_path else None,
    }
    report_path = out_dir / "sam3d_prior_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
