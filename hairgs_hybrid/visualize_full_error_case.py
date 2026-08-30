#!/usr/bin/env python3
"""Render unmasked full-image RGB error sheets for one ratio-ladder case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_for(case: Path, compression: float) -> Path:
    prefix = f"c{compression * 100:g}_"
    values = [path for path in case.iterdir() if path.name.startswith(prefix)]
    if len(values) != 1:
        raise RuntimeError(f"Expected one {prefix} run under {case}")
    return values[0]


def frame_map(report: dict) -> dict[str, dict]:
    return {str(row["image"]): row for row in report["per_frame"]}


def predictions(report: dict) -> dict[str, np.ndarray]:
    manifest = load(Path(report["render_manifest"]))
    result = {}
    for observation in manifest["observations"]:
        with np.load(observation["array"]) as payload:
            result[str(observation["image"])] = payload["rgb"].astype(np.float32)
    return result


def ground_truth(path: Path, size: tuple[int, int]) -> np.ndarray:
    bgra = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if bgra is None or bgra.ndim != 3 or bgra.shape[2] != 4:
        raise ValueError(f"Expected RGBA ground truth: {path}")
    maximum = float(np.iinfo(bgra.dtype).max)
    if (bgra.shape[1], bgra.shape[0]) != size:
        bgra = cv2.resize(bgra, size, interpolation=cv2.INTER_AREA)
    rgba = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA).astype(np.float32) / maximum
    return rgba[..., :3] * rgba[..., 3:4]


def absolute_heatmap(error: np.ndarray, vmax: float) -> np.ndarray:
    value = np.round(np.clip(error / vmax, 0.0, 1.0) * 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(value, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)


def signed_heatmap(delta: np.ndarray, vmax: float) -> np.ndarray:
    value = np.clip(delta / vmax, -1.0, 1.0)
    magnitude = np.abs(value)[..., None]
    neutral = np.asarray([12.0, 12.0, 12.0], dtype=np.float32)
    positive = np.asarray([255.0, 35.0, 35.0], dtype=np.float32)
    negative = np.asarray([35.0, 95.0, 255.0], dtype=np.float32)
    endpoint = np.where((value >= 0)[..., None], positive, negative)
    return np.round(neutral * (1.0 - magnitude) + endpoint * magnitude).astype(np.uint8)


def tile(array: np.ndarray, label: str, size: int) -> Image.Image:
    image = Image.fromarray(np.round(np.clip(array, 0, 255)).astype(np.uint8), "RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    output = Image.new("RGB", (size, size + 28), "black")
    output.paste(image, ((size - image.width) // 2, 28))
    ImageDraw.Draw(output).text((5, 7), label, fill="white", font=ImageFont.load_default())
    return output


def make_row(
    run: Path,
    split: str,
    image_name: str,
    role: str,
    tile_size: int,
    error_vmax: float,
    delta_vmax: float,
) -> Image.Image:
    reports = {
        method: load(run / method / f"evaluation_{split}/external_evaluation/evaluation.json")
        for method in ("hairgs_native", "unifur")
    }
    pred = {method: predictions(report)[image_name] for method, report in reports.items()}
    metrics = {method: frame_map(report)[image_name] for method, report in reports.items()}
    size = (pred["hairgs_native"].shape[1], pred["hairgs_native"].shape[0])
    gt = ground_truth(Path(reports["hairgs_native"]["ground_truth_dir"]) / image_name, size)
    native_error = np.abs(pred["hairgs_native"] - gt).mean(axis=-1)
    unifur_error = np.abs(pred["unifur"] - gt).mean(axis=-1)
    values = (
        (gt * 255.0, f"GT {image_name} ({role})"),
        (pred["hairgs_native"] * 255.0, f"native full {metrics['hairgs_native']['full_psnr']:.2f} dB"),
        (pred["unifur"] * 255.0, f"UniFur full {metrics['unifur']['full_psnr']:.2f} dB"),
        (absolute_heatmap(native_error, error_vmax), f"native |err| [0,{error_vmax:g}]"),
        (absolute_heatmap(unifur_error, error_vmax), f"UniFur |err| [0,{error_vmax:g}]"),
        (signed_heatmap(unifur_error - native_error, delta_vmax), f"delta [-{delta_vmax:g},+{delta_vmax:g}]"),
    )
    tiles = [tile(array, label, tile_size) for array, label in values]
    output = Image.new("RGB", (sum(item.width for item in tiles), max(item.height for item in tiles)), "black")
    x = 0
    for item in tiles:
        output.paste(item, (x, 0)); x += item.width
    return output


def ranked_views(run: Path, split: str) -> list[tuple[str, str]]:
    reports = {
        method: load(run / method / f"evaluation_{split}/external_evaluation/evaluation.json")
        for method in ("hairgs_native", "unifur")
    }
    metrics = {method: frame_map(report) for method, report in reports.items()}
    common = sorted(set(metrics["hairgs_native"]) & set(metrics["unifur"]))
    ranked = sorted(
        common,
        key=lambda name: metrics["unifur"][name]["full_psnr"] - metrics["hairgs_native"][name]["full_psnr"],
    )
    return [(ranked[0], "worst"), (ranked[len(ranked) // 2], "median"), (ranked[-1], "best")]


def stack(rows: list[Image.Image], header: str, output: Path) -> None:
    top = 36
    image = Image.new("RGB", (max(row.width for row in rows), top + sum(row.height for row in rows)), "black")
    ImageDraw.Draw(image).text((8, 10), header, fill="white", font=ImageFont.load_default())
    y = top
    for row in rows:
        image.paste(row, (0, y)); y += row.height
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratio-root", type=Path, required=True)
    parser.add_argument("--case", default="A037_horse")
    parser.add_argument("--compression", type=float, default=0.75)
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--error-vmax", type=float, default=0.20)
    parser.add_argument("--delta-vmax", type=float, default=0.10)
    args = parser.parse_args()
    case = args.ratio_root / "cases" / args.case
    run = run_for(case, args.compression)
    output = args.ratio_root / "audit/full_image_error" / args.case
    result = {}
    for split in ("train_v48", "test_v12"):
        selected = ranked_views(run, split)
        rows = [
            make_row(run, split, name, role, args.tile, args.error_vmax, args.delta_vmax)
            for name, role in selected
        ]
        path = output / f"{split}_c{args.compression*100:g}_full_error.png"
        stack(
            rows,
            f"{args.case} | {split} | compression {args.compression*100:g}% | full-image RGB error | red=UniFur worse, blue=better",
            path,
        )
        result[split] = {"image": str(path.resolve()), "selected": selected}
    (output / "selection.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
