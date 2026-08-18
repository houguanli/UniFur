#!/usr/bin/env python3
"""Build an auditable image set by applying same-name binary masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    images = sorted(args.image_dir.resolve().glob("*.png"))
    if not images:
        raise FileNotFoundError(f"No PNG images in {args.image_dir}")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    observations = []
    for image_path in images:
        mask_path = args.mask_dir.resolve() / image_path.name
        if not mask_path.is_file():
            raise FileNotFoundError(f"Missing same-name mask: {mask_path}")
        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
        if image.shape[:2] != mask.shape:
            raise ValueError(f"Image/mask shape mismatch for {image_path.name}")
        masked = image.copy()
        masked[mask < 128] = 0
        destination = output / image_path.name
        Image.fromarray(masked).save(destination)
        observations.append(
            {
                "image": image_path.name,
                "source_image": str(image_path),
                "source_mask": str(mask_path),
                "foreground_fraction": float((mask >= 128).mean()),
            }
        )

    manifest = {
        "schema": "masked-image-set-v1",
        "image_dir": str(args.image_dir.resolve()),
        "mask_dir": str(args.mask_dir.resolve()),
        "output_dir": str(output),
        "image_count": len(observations),
        "observations": observations,
    }
    (output / "masked_image_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
