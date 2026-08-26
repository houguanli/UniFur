#!/usr/bin/env python3
"""Create compact GT/native/route-aware/hybrid visual comparison sheets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def rgb_image(array: np.ndarray) -> Image.Image:
    return Image.fromarray(
        np.clip(np.asarray(array) * 255.0 + 0.5, 0, 255).astype(np.uint8), "RGB"
    )


def prediction(root: Path, name: str) -> tuple[Image.Image, Image.Image]:
    data = np.load(root / "external_renders/arrays" / f"{Path(name).stem}.npz")
    alpha = np.repeat(np.asarray(data["mask"])[..., None], 3, axis=2)
    return rgb_image(data["rgb"]), rgb_image(alpha)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--route-aware", type=Path, required=True)
    parser.add_argument("--hybrid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=4)
    args = parser.parse_args()

    result = json.loads(
        (args.hybrid / "external_evaluation/evaluation.json").read_text(
            encoding="utf-8"
        )
    )
    names = [row["image"] for row in result["per_frame"]]
    if len(names) > args.max_rows:
        indices = np.linspace(0, len(names) - 1, args.max_rows).round().astype(int)
        names = [names[index] for index in indices]
    labels = (
        "GT",
        "HairGS 15k",
        "native 5k",
        "old route 5k",
        "hybrid 5k",
        "hybrid alpha",
    )
    cell, header = 500, 48
    canvas = Image.new("RGB", (cell * len(labels), header + cell * len(names)), "black")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default(size=20)
    except TypeError:
        font = ImageFont.load_default()
    for column, label in enumerate(labels):
        draw.text((column * cell + 12, 13), label, fill="white", font=font)
    for row, name in enumerate(names):
        gt = Image.open(args.ground_truth / name).convert("RGB").resize((cell, cell))
        reference, _ = prediction(args.reference, name)
        native, _ = prediction(args.native, name)
        route, _ = prediction(args.route_aware, name)
        hybrid, alpha = prediction(args.hybrid, name)
        for column, image in enumerate((gt, reference, native, route, hybrid, alpha)):
            canvas.paste(image, (column * cell, header + row * cell))
        draw.text((8, header + row * cell + 8), name, fill="white", font=font)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, compress_level=6)
    print(args.output)


if __name__ == "__main__":
    main()
