#!/usr/bin/env python3
"""Build a compact GT / HairGS / native-5k / adaptive-5k comparison sheet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def to_image(array: np.ndarray) -> Image.Image:
    return Image.fromarray(
        np.clip(np.asarray(array) * 255.0 + 0.5, 0, 255).astype(np.uint8), "RGB"
    )


def load_prediction(root: Path, name: str) -> tuple[Image.Image, Image.Image]:
    data = np.load(root / "external_renders/arrays" / f"{Path(name).stem}.npz")
    rgb = to_image(data["rgb"])
    alpha = np.repeat(data["mask"][..., None], 3, axis=2)
    return rgb, to_image(alpha)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--baseline-eval", type=Path, required=True)
    parser.add_argument("--native-eval", type=Path, required=True)
    parser.add_argument("--adaptive-eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    evaluation = load_json(args.adaptive_eval / "external_evaluation/evaluation.json")
    names = [row["image"] for row in evaluation["per_frame"]]
    labels = ("GT", "HairGS 15k", "HairGS native 5k", "adaptive 5k", "adaptive alpha")
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
        baseline, _ = load_prediction(args.baseline_eval, name)
        native, _ = load_prediction(args.native_eval, name)
        adaptive, alpha = load_prediction(args.adaptive_eval, name)
        for column, image in enumerate((gt, baseline, native, adaptive, alpha)):
            canvas.paste(image, (column * cell, header + row * cell))
        draw.text((8, header + row * cell + 8), name, fill="white", font=font)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, compress_level=6)
    print(args.output)


if __name__ == "__main__":
    main()
