#!/usr/bin/env python3
"""Build a protocol-matched HairGS/soft-route/hard-route visual audit sheet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _load_prediction(root: Path, image_name: str) -> tuple[Image.Image, Image.Image]:
    array = root / "evaluation_test/external_renders/arrays" / f"{Path(image_name).stem}.npz"
    with np.load(array) as payload:
        rgb = np.clip(payload["rgb"] * 255.0 + 0.5, 0, 255).astype(np.uint8)
        alpha = np.clip(payload["mask"] * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb), Image.fromarray(alpha).convert("RGB")


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--hairgs", type=Path, required=True)
    parser.add_argument("--soft", type=Path, required=True)
    parser.add_argument("--hard", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=4)
    parser.add_argument("--budget-label", default="22k")
    args = parser.parse_args()

    evaluation = json.loads(
        (args.hard / "evaluation_test/external_evaluation/evaluation.json").read_text(
            encoding="utf-8"
        )
    )
    names = [frame["image"] for frame in evaluation["per_frame"]]
    if len(names) > args.max_rows:
        indices = np.linspace(0, len(names) - 1, args.max_rows).round().astype(int)
        names = [names[index] for index in indices]

    labels = (
        "GT",
        f"HairGS {args.budget_label}",
        f"UniFur soft {args.budget_label}",
        f"UniFur hard {args.budget_label}",
        "hard alpha",
    )
    width, height = Image.open(args.ground_truth / names[0]).size
    header = 34
    canvas = Image.new("RGB", (width * len(labels), header + height * len(names)), "black")
    draw = ImageDraw.Draw(canvas)
    font = _font(16)
    for column, label in enumerate(labels):
        draw.text((column * width + 8, 8), label, fill="white", font=font)

    for row, name in enumerate(names):
        gt = Image.open(args.ground_truth / name).convert("RGB")
        hairgs, _ = _load_prediction(args.hairgs, name)
        soft, _ = _load_prediction(args.soft, name)
        hard, hard_alpha = _load_prediction(args.hard, name)
        for column, panel in enumerate((gt, hairgs, soft, hard, hard_alpha)):
            canvas.paste(panel, (column * width, header + row * height))
        draw.text((5, header + row * height + 5), name, fill="white", font=font)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, compress_level=6)
    print(args.output)


if __name__ == "__main__":
    main()
