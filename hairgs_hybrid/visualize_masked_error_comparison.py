#!/usr/bin/env python3
"""Visualize foreground RGB error for unrestricted HairGS versus UniFur-25k.

The absolute-error panels use one fixed range across every case/view.  The
signed panel is ``UniFur error - unrestricted error``: red means UniFur is
worse and blue means UniFur is better.  Only pixels inside the GT alpha mask
are visualized, so background leakage is intentionally excluded here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


METHODS = ("hairgs_unrestricted", "unifur_25k")
SPLITS = ("train_v48", "test_v12")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_gt(path: Path, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    bgra = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if bgra is None or bgra.ndim != 3 or bgra.shape[2] != 4:
        raise ValueError(f"Expected RGBA ground truth: {path}")
    scale = float(np.iinfo(bgra.dtype).max)
    if (bgra.shape[1], bgra.shape[0]) != size:
        bgra = cv2.resize(bgra, size, interpolation=cv2.INTER_AREA)
    rgba = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA).astype(np.float32) / scale
    return rgba[..., :3], rgba[..., 3] > 0.5


def load_predictions(report: dict) -> dict[str, dict[str, np.ndarray]]:
    manifest = load_json(Path(report["render_manifest"]))
    result = {}
    for observation in manifest["observations"]:
        with np.load(observation["array"]) as payload:
            result[str(observation["image"])] = {
                "rgb": payload["rgb"].astype(np.float32),
                "mask": payload["mask"].astype(np.float32),
            }
    return result


def turbo_error(error: np.ndarray, mask: np.ndarray, vmax: float) -> np.ndarray:
    normalized = np.round(np.clip(error / vmax, 0.0, 1.0) * 255).astype(np.uint8)
    colored = cv2.cvtColor(cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    colored[~mask] = 0
    return colored


def signed_error(delta: np.ndarray, mask: np.ndarray, vmax: float) -> np.ndarray:
    value = np.clip(delta / vmax, -1.0, 1.0)
    magnitude = np.abs(value)[..., None]
    positive = np.array([255.0, 35.0, 35.0], dtype=np.float32)
    negative = np.array([35.0, 95.0, 255.0], dtype=np.float32)
    endpoint = np.where((value >= 0)[..., None], positive, negative)
    rgb = 255.0 * (1.0 - magnitude) + endpoint * magnitude
    rgb[~mask] = 0
    return np.round(rgb).astype(np.uint8)


def panel(array: np.ndarray, label: str, tile: int) -> Image.Image:
    image = Image.fromarray(np.round(np.clip(array, 0, 255)).astype(np.uint8), "RGB")
    image.thumbnail((tile, tile), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (tile, tile + 28), "black")
    canvas.paste(image, ((tile - image.width) // 2, 28))
    ImageDraw.Draw(canvas).text((5, 7), label, fill="white", font=ImageFont.load_default())
    return canvas


def frame_metrics(report: dict) -> dict[str, dict]:
    return {str(row["image"]): row for row in report["per_frame"]}


def choose_views(native: dict, unifur: dict, count: int) -> list[tuple[str, str]]:
    native_frames = frame_metrics(native)
    unifur_frames = frame_metrics(unifur)
    common = sorted(set(native_frames) & set(unifur_frames))
    ranked = sorted(
        common,
        key=lambda name: (
            unifur_frames[name]["foreground_psnr"]
            - native_frames[name]["foreground_psnr"]
        ),
    )
    if not ranked:
        raise RuntimeError("No common rendered views")
    if count <= 1 or len(ranked) == 1:
        return [(ranked[0], "worst")]
    indices = np.linspace(0, len(ranked) - 1, count, dtype=int)
    roles = ["worst", *(["median"] * max(count - 2, 0)), "best"]
    return [(ranked[index], roles[offset]) for offset, index in enumerate(indices)]


def make_case_sheet(
    case: Path,
    split: str,
    output: Path,
    tile: int,
    views: int,
    error_vmax: float,
    delta_vmax: float,
) -> tuple[Image.Image, list[dict]]:
    reports = {
        method: load_json(
            case / method / f"evaluation_{split}/external_evaluation/evaluation.json"
        )
        for method in METHODS
    }
    predictions = {method: load_predictions(report) for method, report in reports.items()}
    metrics = {method: frame_metrics(report) for method, report in reports.items()}
    selected = choose_views(reports[METHODS[0]], reports[METHODS[1]], views)
    rows = []
    selection_report = []
    for image_name, role in selected:
        native = predictions["hairgs_unrestricted"][image_name]["rgb"]
        unifur = predictions["unifur_25k"][image_name]["rgb"]
        size = (int(native.shape[1]), int(native.shape[0]))
        gt_dir = Path(reports["hairgs_unrestricted"]["ground_truth_dir"])
        gt, mask = load_gt(gt_dir / image_name, size)
        native_error = np.abs(native - gt).mean(axis=-1)
        unifur_error = np.abs(unifur - gt).mean(axis=-1)
        delta = unifur_error - native_error
        native_psnr = metrics["hairgs_unrestricted"][image_name]["foreground_psnr"]
        unifur_psnr = metrics["unifur_25k"][image_name]["foreground_psnr"]
        labels = (
            f"GT {image_name} ({role})",
            f"unrestricted {native_psnr:.2f} dB",
            f"UniFur-25k {unifur_psnr:.2f} dB",
            f"unrestricted |err| [0,{error_vmax:g}]",
            f"UniFur |err| [0,{error_vmax:g}]",
            f"delta [-{delta_vmax:g},+{delta_vmax:g}]",
        )
        arrays = (
            gt * mask[..., None] * 255.0,
            native * 255.0,
            unifur * 255.0,
            turbo_error(native_error, mask, error_vmax),
            turbo_error(unifur_error, mask, error_vmax),
            signed_error(delta, mask, delta_vmax),
        )
        tiles = [panel(array, label, tile) for array, label in zip(arrays, labels)]
        row = Image.new("RGB", (sum(item.width for item in tiles), max(item.height for item in tiles)), "black")
        x = 0
        for item in tiles:
            row.paste(item, (x, 0))
            x += item.width
        rows.append(row)
        selection_report.append(
            {
                "image": image_name,
                "role": role,
                "unrestricted_foreground_psnr": native_psnr,
                "unifur_foreground_psnr": unifur_psnr,
                "delta_psnr": unifur_psnr - native_psnr,
                "masked_mean_abs_error_unrestricted": float(native_error[mask].mean()),
                "masked_mean_abs_error_unifur": float(unifur_error[mask].mean()),
            }
        )
    header = 42
    sheet = Image.new("RGB", (max(row.width for row in rows), header + sum(row.height for row in rows)), "black")
    ImageDraw.Draw(sheet).text(
        (8, 8),
        f"{case.name} | {split} | signed delta: blue=UniFur better, red=UniFur worse",
        fill="white",
        font=ImageFont.load_default(),
    )
    y = header
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    return sheet, selection_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cases", nargs="*")
    parser.add_argument("--views", type=int, default=3)
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--error-vmax", type=float, default=0.20)
    parser.add_argument("--delta-vmax", type=float, default=0.10)
    args = parser.parse_args()
    output = args.output_dir or args.benchmark_root / "audit/masked_error_comparison"
    cases_root = args.benchmark_root / "cases"
    cases = [cases_root / name for name in args.cases] if args.cases else [
        case for case in sorted(cases_root.iterdir()) if (case / "COMPLETE").is_file()
    ]
    all_reports = {}
    overview_rows = {split: [] for split in SPLITS}
    for case in cases:
        all_reports[case.name] = {}
        for split in SPLITS:
            sheet, report = make_case_sheet(
                case,
                split,
                output / f"{case.name}_{split}_masked_error.png",
                args.tile,
                args.views,
                args.error_vmax,
                args.delta_vmax,
            )
            all_reports[case.name][split] = report
            # The first selected row is the largest UniFur-vs-unrestricted PSNR loss.
            row_height = args.tile + 28
            overview_rows[split].append(sheet.crop((0, 42, sheet.width, 42 + row_height)))
    for split, rows in overview_rows.items():
        if not rows:
            continue
        case_header = 24
        overview = Image.new(
            "RGB",
            (max(row.width for row in rows), sum(row.height + case_header for row in rows)),
            "black",
        )
        y = 0
        for case, row in zip(cases, rows):
            ImageDraw.Draw(overview).text(
                (8, y + 6),
                f"{case.name} | {split} | worst foreground-PSNR gap",
                fill="white",
                font=ImageFont.load_default(),
            )
            y += case_header
            overview.paste(row, (0, y))
            y += row.height
        overview.save(output / f"overview_{split}_worst_views.png")
    (output / "selection_report.json").write_text(
        json.dumps(
            {
                "schema": "unifur-masked-error-comparison-v1",
                "cases": all_reports,
                "absolute_error_vmax": args.error_vmax,
                "signed_delta_vmax": args.delta_vmax,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"cases": len(cases), "output": str(output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
