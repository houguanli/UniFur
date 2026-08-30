#!/usr/bin/env python3
"""Create same-view train/test visual sheets for the ratio ladder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


COMPRESSIONS = (0.75, 0.875, 0.9375)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_for(case: Path, compression: float) -> Path:
    prefix = f"c{compression * 100:g}_"
    return next(path for path in case.iterdir() if path.name.startswith(prefix))


def predictions(report: dict) -> dict[str, np.ndarray]:
    manifest = load(Path(report["render_manifest"]))
    output = {}
    for item in manifest["observations"]:
        with np.load(item["array"]) as payload:
            output[str(item["image"])] = payload["rgb"].astype(np.float32)
    return output


def frame_map(report: dict) -> dict[str, dict]:
    return {str(item["image"]): item for item in report["per_frame"]}


def gt_rgb(path: Path, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    bgra = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    scale = float(np.iinfo(bgra.dtype).max)
    if (bgra.shape[1], bgra.shape[0]) != size:
        bgra = cv2.resize(bgra, size, interpolation=cv2.INTER_AREA)
    rgba = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA).astype(np.float32) / scale
    return rgba[..., :3], rgba[..., 3] > 0.5


def signed_delta(value: np.ndarray, mask: np.ndarray, vmax: float = 0.1) -> np.ndarray:
    value = np.clip(value / vmax, -1.0, 1.0)
    magnitude = np.abs(value)[..., None]
    positive = np.asarray([255.0, 35.0, 35.0])
    negative = np.asarray([35.0, 95.0, 255.0])
    endpoint = np.where((value >= 0)[..., None], positive, negative)
    rgb = 255.0 * (1.0 - magnitude) + endpoint * magnitude
    rgb[~mask] = 0
    return np.round(rgb).astype(np.uint8)


def tile(array: np.ndarray, label: str, size: int = 256) -> Image.Image:
    image = Image.fromarray(np.round(np.clip(array, 0, 255)).astype(np.uint8), "RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    output = Image.new("RGB", (size, size + 28), "black")
    output.paste(image, ((size - image.width) // 2, 28))
    ImageDraw.Draw(output).text((5, 7), label, fill="white", font=ImageFont.load_default())
    return output


def row(case: Path, compression: float, split: str, image_name: str) -> Image.Image:
    run = run_for(case, compression)
    reports = {
        method: load(run / method / f"evaluation_{split}/external_evaluation/evaluation.json")
        for method in ("hairgs_native", "unifur")
    }
    pred = {method: predictions(report)[image_name] for method, report in reports.items()}
    size = (pred["hairgs_native"].shape[1], pred["hairgs_native"].shape[0])
    gt, mask = gt_rgb(Path(reports["hairgs_native"]["ground_truth_dir"]) / image_name, size)
    native_error = np.abs(pred["hairgs_native"] - gt).mean(axis=-1)
    unifur_error = np.abs(pred["unifur"] - gt).mean(axis=-1)
    metrics = {method: frame_map(report)[image_name] for method, report in reports.items()}
    values = (
        tile(gt * mask[..., None] * 255, f"GT {image_name}"),
        tile(pred["hairgs_native"] * 255, f"native {metrics['hairgs_native']['foreground_psnr']:.2f} dB"),
        tile(pred["unifur"] * 255, f"UniFur {metrics['unifur']['foreground_psnr']:.2f} dB"),
        tile(signed_delta(unifur_error - native_error, mask), "delta red=worse blue=better"),
    )
    output = Image.new("RGB", (sum(item.width for item in values), max(item.height for item in values)), "black")
    x = 0
    for value in values:
        output.paste(value, (x, 0)); x += value.width
    return output


def worst_view(case: Path, split: str) -> str:
    run = run_for(case, 0.75)
    reports = {
        method: load(run / method / f"evaluation_{split}/external_evaluation/evaluation.json")
        for method in ("hairgs_native", "unifur")
    }
    frames = {method: frame_map(report) for method, report in reports.items()}
    common = set(frames["hairgs_native"]) & set(frames["unifur"])
    return min(
        common,
        key=lambda name: frames["unifur"][name]["foreground_psnr"] - frames["hairgs_native"][name]["foreground_psnr"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratio-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.ratio_root / "audit/ratio_visuals"
    output.mkdir(parents=True, exist_ok=True)
    for split in ("train_v48", "test_v12"):
        overview = []
        for case in sorted(path for path in (args.ratio_root / "cases").iterdir() if path.is_dir()):
            image_name = worst_view(case, split)
            rows = []
            for compression in COMPRESSIONS:
                item = row(case, compression, split, image_name)
                header = Image.new("RGB", (item.width, 24), "black")
                ImageDraw.Draw(header).text((6, 6), f"{case.name} | compression {compression*100:g}%", fill="white", font=ImageFont.load_default())
                block = Image.new("RGB", (item.width, item.height + 24), "black")
                block.paste(header, (0, 0)); block.paste(item, (0, 24)); rows.append(block)
            sheet = Image.new("RGB", (max(item.width for item in rows), sum(item.height for item in rows)), "black")
            y = 0
            for item in rows:
                sheet.paste(item, (0, y)); y += item.height
            sheet.save(output / f"{case.name}_{split}_ratio_ladder.png")
            overview.append(rows[0])
        combined = Image.new("RGB", (max(item.width for item in overview), sum(item.height for item in overview)), "black")
        y = 0
        for item in overview:
            combined.paste(item, (0, y)); y += item.height
        combined.save(args.ratio_root / "audit" / f"ratio_visual_overview_{split}.png")
    print(json.dumps({"status": "complete", "case_sheets": 16, "output": str(output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
