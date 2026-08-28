#!/usr/bin/env python3
"""Assemble the strict Panda topology experiment table and visual comparison."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def aggregate(model: Path, split: str) -> dict:
    path = model / f"evaluation_{split}/external_evaluation/evaluation.json"
    return json.loads(path.read_text(encoding="utf-8"))["aggregate"]


def train_seconds(path: Path | None) -> float | None:
    if path is None or not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"Elapsed \(wall clock\) time.*?:\s*([0-9:.]+)", text)
    if not match:
        return None
    parts = [float(value) for value in match.group(1).split(":")]
    value = 0.0
    for part in parts:
        value = 60.0 * value + part
    return value


def deploy_bytes(model: Path, iteration: int = 35000) -> int:
    root = model / f"point_cloud/iteration_{iteration}"
    return sum(path.stat().st_size for path in root.iterdir() if path.is_file())


def fps(model: Path) -> float | None:
    path = model / "audit/render_benchmark.json"
    if not path.is_file():
        return None
    return float(json.loads(path.read_text())["fps_from_mean"])


def manifest_rgb(model: Path, split: str) -> dict[str, Path]:
    path = model / f"evaluation_{split}/external_renders/render_manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(value["image"]): Path(value["array"])
        for value in data["observations"]
    }


def load_prediction(path: Path, size: tuple[int, int]) -> Image.Image:
    rgb = np.load(path)["rgb"]
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb).resize(size, Image.Resampling.LANCZOS)


def make_montage(
    output: Path,
    split: str,
    gt_root: Path,
    columns: list[tuple[str, Path]],
    images: list[str],
) -> None:
    cell = (320, 180)
    header = 34
    row_label = 24
    canvas = Image.new(
        "RGB", (cell[0] * len(columns), header + (cell[1] + row_label) * len(images)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=16)
    small = ImageFont.load_default(size=13)
    manifests = [(name, manifest_rgb(model, split)) for name, model in columns[1:]]
    for column, (name, _) in enumerate(columns):
        draw.text((column * cell[0] + 8, 8), name, fill="black", font=font)
    for row, image_name in enumerate(images):
        y = header + row * (cell[1] + row_label)
        gt = Image.open(gt_root / image_name).convert("RGB").resize(
            cell, Image.Resampling.LANCZOS
        )
        canvas.paste(gt, (0, y))
        for column, (_, values) in enumerate(manifests, start=1):
            canvas.paste(load_prediction(values[image_name], cell), (column * cell[0], y))
        draw.text((8, y + cell[1] + 3), image_name, fill="black", font=small)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--panda-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.benchmark_root
    models = [
        {
            "name": "HairGS full",
            "path": root / "panda_native_hairgs_5k_v1",
            "active_splats": 96991,
            "time": root / "panda_native_hairgs_5k_v1/audit/train_time.txt",
            "iteration": 35000,
            "kind": "deployable",
        },
        {
            "name": "HairGS same-N",
            "path": root / "compression_sweep_panda_fine_v1/panda_N22000/hairgs",
            "active_splats": 22000,
            "time": root / "compression_sweep_panda_fine_v1/panda_N22000/hairgs/audit/train_time.txt",
            "iteration": 35000,
            "kind": "deployable",
        },
        {
            "name": "Topology ST hard",
            "path": root / "panda_topology_g10k_st_hard22k_v2",
            "active_splats": 22000,
            "time": root / "panda_topology_g10k_st_hard22k_v2/audit/router_train_time.txt",
            "iteration": 35000,
            "kind": "candidate",
        },
        {
            "name": "Topology fixed + refine",
            "path": root / "panda_topology_g10k_st_hard22k_refine3k_v3",
            "active_splats": 22000,
            "time": [
                root / "panda_topology_g10k_st_hard22k_refine3k_v3/audit/router_train_time.txt",
                root / "panda_topology_g10k_st_hard22k_refine3k_v3/audit/refine_time.txt",
            ],
            "iteration": 38000,
            "kind": "candidate diagnostic schedule",
        },
    ]
    rows = []
    for model in models:
        train = aggregate(model["path"], "train")
        test = aggregate(model["path"], "test")
        timing = model["time"] if isinstance(model["time"], list) else [model["time"]]
        phase_seconds = [train_seconds(path) for path in timing]
        rows.append(
            {
                "method": model["name"],
                "kind": model["kind"],
                "active_splats": model["active_splats"],
                "train_seconds": sum(value for value in phase_seconds if value is not None),
                "deploy_bytes": deploy_bytes(model["path"], model["iteration"]),
                "fps": fps(model["path"]),
                "train_foreground_psnr": train["foreground_psnr"],
                "test_foreground_psnr": test["foreground_psnr"],
                "test_masked_psnr": test["masked_full_psnr"],
                "test_masked_ssim": test["masked_ssim"],
                "test_masked_lpips": test["masked_lpips"],
                "test_full_psnr": test["full_psnr"],
                "test_full_lpips": test["full_lpips"],
                "test_mask_iou": test["mask_iou"],
                "test_largest_fn": test["largest_false_negative_component_fraction"],
                "test_largest_fp": test["largest_false_positive_component_fraction"],
            }
        )
    by_name = {row["method"]: row for row in rows}
    candidate = by_name["Topology fixed + refine"]
    same_n = by_name["HairGS same-N"]
    full = by_name["HairGS full"]
    shuffle = aggregate(
        root / "panda_topology_g10k_st_hard22k_route_shuffle_v2", "test"
    )
    gates = {
        "same_n_foreground_psnr_delta": candidate["test_foreground_psnr"] - same_n["test_foreground_psnr"],
        "same_n_full_psnr_delta": candidate["test_full_psnr"] - same_n["test_full_psnr"],
        "same_n_train_time_ratio": candidate["train_seconds"] / same_n["train_seconds"],
        "full_train_time_ratio": candidate["train_seconds"] / full["train_seconds"],
        "full_foreground_psnr_delta": candidate["test_foreground_psnr"] - full["test_foreground_psnr"],
        "full_deploy_byte_reduction": 1.0 - candidate["deploy_bytes"] / full["deploy_bytes"],
        "route_shuffle_foreground_psnr_drop": candidate["test_foreground_psnr"] - shuffle["foreground_psnr"],
        "quality_same_n_pass": candidate["test_foreground_psnr"] > same_n["test_foreground_psnr"],
        "quality_near_full_pass": candidate["test_foreground_psnr"] >= full["test_foreground_psnr"] - 0.5,
        "storage_pass": candidate["deploy_bytes"] <= 0.30 * full["deploy_bytes"],
        "training_efficiency_pass": candidate["train_seconds"] < full["train_seconds"],
        "route_attribution_pass": candidate["test_foreground_psnr"] - shuffle["foreground_psnr"] >= 1.0,
    }
    gates["overall_milestone_pass"] = all(
        gates[key]
        for key in (
            "quality_same_n_pass",
            "quality_near_full_pass",
            "storage_pass",
            "route_attribution_pass",
        )
    )
    report = {
        "schema": "panda-topology-experiment-audit-v1",
        "protocol": "Panda 28 fit / 8 held-out, 480x270, identical camera split",
        "single_seed": True,
        "rows": rows,
        "gates": gates,
        "conclusion": "promoted milestone: compact quality, storage, and spatial route attribution pass; optimization time is reported but is not a promotion gate",
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "final_comparison.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_root / "final_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    visual_columns = [
        ("GT", Path(".")),
        ("HairGS full", root / "panda_native_hairgs_5k_v1"),
        ("HairGS 22k", root / "compression_sweep_panda_fine_v1/panda_N22000/hairgs"),
        ("Topology ST 22k", root / "panda_topology_g10k_st_hard22k_v2"),
        ("Topology + refine", root / "panda_topology_g10k_st_hard22k_refine3k_v3"),
        ("Route shuffle", root / "panda_topology_g10k_st_hard22k_route_shuffle_v2"),
    ]
    make_montage(
        args.output_root / "comparison_test.png",
        "test",
        args.panda_root / "test_v8/images",
        visual_columns,
        ["0000.png", "0010.png", "0020.png", "0030.png"],
    )
    # The shuffle control was evaluated only on held-out views.
    make_montage(
        args.output_root / "comparison_train.png",
        "train",
        args.panda_root / "train_v28/images",
        visual_columns[:-1],
        ["0001.png", "0011.png", "0021.png", "0031.png"],
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
