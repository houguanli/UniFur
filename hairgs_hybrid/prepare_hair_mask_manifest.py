#!/usr/bin/env python3
"""Validate immutable hair-only masks used by mask-bounded training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_dataset(dataset: Path) -> dict:
    images = {path.name: path for path in (dataset / "images").glob("*.png")}
    masks = {path.name: path for path in (dataset / "masks").glob("*.png")}
    if not images or set(images) != set(masks):
        raise ValueError(f"Image/mask names differ or are empty in {dataset}")
    observations = []
    for name in sorted(images):
        image = Image.open(images[name]).convert("RGB")
        mask_image = Image.open(masks[name]).convert("L")
        mask = np.asarray(mask_image, dtype=np.uint8)
        unique = np.unique(mask)
        if image.size != mask_image.size:
            raise ValueError(f"Image/mask size mismatch for {name}")
        if not np.all(np.isin(unique, [0, 255])):
            raise ValueError(f"Mask is not binary for {name}: {unique.tolist()}")
        fraction = float((mask == 255).mean())
        if not 0.001 < fraction < 0.75:
            raise ValueError(f"Implausible hair-only support for {name}: {fraction}")
        observations.append(
            {
                "image": name,
                "size": list(image.size),
                "hair_fraction": fraction,
                "image_sha256": sha256(images[name]),
                "mask_sha256": sha256(masks[name]),
            }
        )
    return {
        "dataset": str(dataset.resolve()),
        "image_count": len(observations),
        "semantic_label": "hair-only; face, scalp interior, neck and bust excluded",
        "mask_rule": "binary source mask, no synthesized head/body foreground",
        "observations": observations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "schema": "hair-only-mask-manifest-v1",
        "datasets": [inspect_dataset(path) for path in args.dataset],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
