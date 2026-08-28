#!/usr/bin/env python3
"""Build a protocol-locked GT/method comparison sheet for training views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _parse_method(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("method must be LABEL=EVALUATION_ROOT")
    label, root = value.split("=", 1)
    if not label.strip():
        raise argparse.ArgumentTypeError("method label must not be empty")
    return label.strip(), Path(root)


def _rgb(array: np.ndarray) -> Image.Image:
    pixels = np.asarray(array)
    if pixels.dtype != np.uint8:
        pixels = np.clip(pixels * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return Image.fromarray(pixels, "RGB")


def _fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return image.convert("RGB").resize(size, Image.Resampling.LANCZOS)


def _load_evaluation(root: Path) -> dict:
    path = root / "external_evaluation" / "evaluation.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _prediction(root: Path, name: str) -> Image.Image:
    path = root / "external_renders" / "arrays" / f"{Path(name).stem}.npz"
    with np.load(path) as data:
        return _rgb(data["rgb"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--method", action="append", type=_parse_method, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--columns", type=int, default=7)
    parser.add_argument("--tile-width", type=int, default=300)
    args = parser.parse_args()

    methods = args.method
    evaluations = [(label, root, _load_evaluation(root)) for label, root in methods]
    reference = evaluations[0][2]
    protocol = reference["protocol"]
    render_size = tuple(reference["render_size"])
    names = [row["image"] for row in reference["per_frame"]]
    view_indices = {row["image"]: row.get("view_index") for row in reference["per_frame"]}

    for label, _, evaluation in evaluations[1:]:
        other_names = [row["image"] for row in evaluation["per_frame"]]
        if evaluation["protocol"] != protocol:
            raise ValueError(f"protocol mismatch for {label}: {evaluation['protocol']} != {protocol}")
        if tuple(evaluation["render_size"]) != render_size:
            raise ValueError(f"render-size mismatch for {label}")
        if other_names != names:
            raise ValueError(f"ordered view mismatch for {label}")

    metric_maps = {
        label: {row["image"]: row for row in evaluation["per_frame"]}
        for label, _, evaluation in evaluations
    }
    tile_width = args.tile_width
    tile_height = round(tile_width * render_size[1] / render_size[0])
    label_width = 260
    header_height = 46
    row_height = tile_height
    block_gap = 22
    rows_per_block = 1 + len(methods)
    blocks = [names[index : index + args.columns] for index in range(0, len(names), args.columns)]
    canvas_width = label_width + args.columns * tile_width
    block_height = header_height + rows_per_block * row_height
    canvas_height = len(blocks) * block_height + max(0, len(blocks) - 1) * block_gap
    canvas = Image.new("RGB", (canvas_width, canvas_height), (12, 12, 12))
    draw = ImageDraw.Draw(canvas)
    title_font = _font(20)
    label_font = _font(18)
    metric_font = _font(15)

    y0 = 0
    for block in blocks:
        draw.rectangle((0, y0, canvas_width, y0 + header_height), fill=(25, 25, 25))
        draw.text((12, y0 + 12), protocol, fill=(225, 225, 225), font=metric_font)
        for column, name in enumerate(block):
            x = label_width + column * tile_width
            view = view_indices.get(name)
            suffix = f" | view {view}" if view is not None else ""
            draw.text((x + 8, y0 + 11), f"{name}{suffix}", fill="white", font=metric_font)

        row_specs: list[tuple[str, Path | None]] = [("GT", None)] + [
            (label, root) for label, root, _ in evaluations
        ]
        for row_index, (label, root) in enumerate(row_specs):
            y = y0 + header_height + row_index * row_height
            fill = (32, 45, 55) if row_index == 0 else (25, 25, 25)
            draw.rectangle((0, y, label_width, y + row_height), fill=fill)
            draw.text((12, y + 14), label, fill="white", font=label_font)
            if root is not None:
                aggregate = next(item[2]["aggregate"] for item in evaluations if item[0] == label)
                draw.text(
                    (12, y + 43),
                    f"mean FG {aggregate['foreground_psnr']:.2f} dB",
                    fill=(180, 205, 220),
                    font=metric_font,
                )
            for column, name in enumerate(block):
                x = label_width + column * tile_width
                if root is None:
                    image = Image.open(args.ground_truth / name)
                else:
                    image = _prediction(root, name)
                canvas.paste(_fit(image, (tile_width, tile_height)), (x, y))
                if root is not None:
                    value = metric_maps[label][name]["foreground_psnr"]
                    text = f"FG {value:.2f} dB"
                    draw.rectangle((x + 4, y + 4, x + 108, y + 25), fill=(0, 0, 0))
                    draw.text((x + 8, y + 6), text, fill="white", font=metric_font)
        y0 += block_height + block_gap

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, compress_level=6)
    print(args.output)
    print(f"protocol={protocol} views={len(names)} render_size={render_size}")


if __name__ == "__main__":
    main()
