#!/usr/bin/env python3
"""Compare two HairGS variants under exact hair-only mask semantics.

The RGB columns show the full composite so frozen-head artifacts remain visible.
The mask/error columns evaluate only exact route-sidecar hair primitives, with the
frozen head retained as an occluder by the renderer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _uint8_rgb(array: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(array) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _load_prediction(root: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    path = root / "external_renders" / "arrays" / f"{Path(name).stem}.npz"
    data = np.load(path)
    return np.asarray(data["rgb"], dtype=np.float32), np.asarray(
        data["mask"], dtype=np.float32
    )


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def _mask_rgb(mask: np.ndarray) -> np.ndarray:
    return np.repeat(mask[..., None], 3, axis=2)


def _error_overlay(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """White=TP, red=FP, cyan=FN, dark gray=correct background."""
    pred = prediction >= 0.5
    truth = target >= 0.5
    image = np.full((*truth.shape, 3), 0.04, dtype=np.float32)
    image[pred & truth] = (0.92, 0.92, 0.92)
    image[pred & ~truth] = (1.0, 0.08, 0.08)
    image[~pred & truth] = (0.0, 0.88, 1.0)
    return image


def _image(array: np.ndarray, size: int) -> Image.Image:
    image = Image.fromarray(_uint8_rgb(array), "RGB")
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.LANCZOS)
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--hair-masks", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference-label", default="Previous exact-route")
    parser.add_argument("--candidate-label", default="Hair-only frozen-base")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=4)
    parser.add_argument("--cell-size", type=int, default=360)
    args = parser.parse_args()

    evaluation = json.loads(
        (args.candidate / "external_evaluation" / "evaluation.json").read_text(
            encoding="utf-8"
        )
    )
    names = [row["image"] for row in evaluation["per_frame"]]
    if len(names) > args.max_rows:
        indices = np.linspace(0, len(names) - 1, args.max_rows).round().astype(int)
        names = [names[index] for index in indices]

    labels = (
        "GT composite",
        args.reference_label,
        args.candidate_label,
        "GT hair mask",
        "Previous hair alpha",
        "New hair alpha",
        "New error: FP red / FN cyan",
    )
    size, header = args.cell_size, 52
    canvas = Image.new("RGB", (size * len(labels), header + size * len(names)), "black")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default(size=18)
    except TypeError:
        font = ImageFont.load_default()
    for column, label in enumerate(labels):
        draw.text((column * size + 9, 15), label, fill="white", font=font)

    for row, name in enumerate(names):
        gt = _load_rgb(args.ground_truth / name)
        target = _load_mask(args.hair_masks / name)
        reference_rgb, reference_mask = _load_prediction(args.reference, name)
        candidate_rgb, candidate_mask = _load_prediction(args.candidate, name)
        columns = (
            gt,
            reference_rgb,
            candidate_rgb,
            _mask_rgb(target),
            _mask_rgb(reference_mask),
            _mask_rgb(candidate_mask),
            _error_overlay(candidate_mask, target),
        )
        for column, array in enumerate(columns):
            canvas.paste(_image(array, size), (column * size, header + row * size))
        draw.rectangle(
            (0, header + row * size, 122, header + row * size + 29), fill="black"
        )
        draw.text((8, header + row * size + 7), name, fill="white", font=font)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, compress_level=6)
    print(args.output)


if __name__ == "__main__":
    main()
