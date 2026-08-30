#!/usr/bin/env python3
"""Audit the 48-fit/12-held-out Animals60 UniFur/HairGS benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from plyfile import PlyData


METHODS = ("hairgs_unrestricted", "hairgs_25k", "unifur_25k")
METRICS = (
    "foreground_psnr", "masked_full_psnr", "masked_ssim", "masked_lpips",
    "full_psnr", "full_ssim", "full_lpips", "mask_iou", "mask_f1",
    "false_negative_fraction", "false_positive_fraction",
    "largest_false_negative_component_fraction",
    "largest_false_positive_component_fraction",
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def elapsed(path: Path) -> float:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(
        r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*([0-9:.]+)", text
    )
    if match is None:
        raise ValueError(f"No elapsed wall time in {path}")
    values = [float(value) for value in match.group(1).split(":")]
    return values[0] * 60 + values[1] if len(values) == 2 else (
        values[0] * 3600 + values[1] * 60 + values[2]
    )


def peak_gpu(path: Path) -> float:
    values = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) >= 2:
                values.append(float(row[1]))
    return max(values, default=float("nan"))


def peak_cuda_allocated(paths: list[Path]) -> float:
    values = []
    for path in paths:
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                values.append(float(row["cuda_peak_allocated_mib"]))
    if not values:
        raise RuntimeError(f"No CUDA allocation metrics in {paths}")
    return max(values)


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def latest_iteration(model: Path) -> Path:
    paths = list((model / "point_cloud").glob("iteration_*"))
    if not paths:
        raise FileNotFoundError(model / "point_cloud")
    return max(paths, key=lambda path: int(path.name.rsplit("_", 1)[1]))


def deploy(root: Path) -> dict:
    iteration = latest_iteration(root / "model")
    ply = iteration / "point_cloud.ply"
    point_cloud = PlyData.read(ply)
    splats = point_cloud["edge"].count
    files = [path for path in iteration.iterdir() if path.is_file()]
    return {
        "iteration": int(iteration.name.rsplit("_", 1)[1]),
        "splats": int(splats),
        "deploy_bytes": int(sum(path.stat().st_size for path in files)),
        "ply_bytes": int(ply.stat().st_size),
        "files": [path.name for path in files],
    }


def evaluations(root: Path) -> dict:
    result = {}
    expected_counts = {"train_v48": 48, "test_v12": 12}
    for split, expected_count in expected_counts.items():
        report = load(root / f"evaluation_{split}/external_evaluation/evaluation.json")
        if report["image_count"] != expected_count or report["render_size"] != [512, 512]:
            raise RuntimeError(
                f"Invalid {split} raster protocol under {root}: "
                f"count={report['image_count']}, size={report['render_size']}"
            )
        result[split] = {
            "protocol": report["protocol"],
            "image_count": report["image_count"],
            "metrics": {key: report["aggregate"][key] for key in METRICS},
            "worst_case": report["worst_case"],
        }
    return result


def time_and_memory(case: Path, method: str) -> dict:
    root = case / method
    if method.startswith("hairgs"):
        own = elapsed(root / "audit/stage1_time.txt") + elapsed(root / "audit/stage3_time.txt")
        additional = own
        e2e = own
        own_peak = peak_cuda_allocated([
            root / "stage1/training_metrics.jsonl",
            root / "model/training_metrics.jsonl",
        ])
        e2e_peak = own_peak
    else:
        additional = elapsed(root / "audit/soft_time.txt") + elapsed(root / "audit/refine_time.txt")
        teacher = time_and_memory(case, "hairgs_unrestricted")
        e2e = teacher["end_to_end_train_seconds"] + additional
        own_peak = peak_cuda_allocated([
            root / "soft/training_metrics.jsonl",
            root / "model/training_metrics.jsonl",
        ])
        e2e_peak = float(np.nanmax([teacher["end_to_end_peak_gpu_mib"], own_peak]))
    sampled_device_peak = peak_gpu(root / "audit/train_gpu.csv")
    return {
        "incremental_train_seconds": float(additional),
        "end_to_end_train_seconds": float(e2e),
        "incremental_peak_gpu_mib": float(own_peak),
        "end_to_end_peak_gpu_mib": float(e2e_peak),
        "sampled_device_peak_gpu_mib": float(sampled_device_peak),
    }


def method_report(case: Path, method: str) -> dict:
    root = case / method
    benchmark = load(root / "audit/render_benchmark.json")
    result = {
        "method": method,
        "root": str(root.resolve()),
        **deploy(root),
        **time_and_memory(case, method),
        "working_directory_bytes": directory_bytes(root),
        "render_mean_ms": benchmark["mean_ms_per_frame"],
        "render_fps": benchmark["fps_from_mean"],
        "render_peak_allocated_mib": benchmark["peak_allocated_mib"],
        "evaluation": evaluations(root),
    }
    if method == "unifur_25k":
        projection = load(root / "hard/audit/topology_hard_projection.json")
        result["route_group_counts"] = projection["exported_group_counts"]
        result["route_splat_counts"] = projection["exported_splat_counts"]
        result["route_mean_max_probability"] = projection["mean_max_probability"]
        result["route_mean_entropy"] = projection["mean_entropy"]
    return result


def contact_sheet(case: Path, split: str, output: Path) -> None:
    panels = []
    for method in METHODS:
        source = case / method / f"evaluation_{split}/external_evaluation/evaluation_contact_sheet.png"
        image = Image.open(source).convert("RGB")
        target_width = 520
        image.thumbnail((target_width, 1600), Image.Resampling.LANCZOS)
        panel = Image.new("RGB", (target_width, image.height + 32), "black")
        ImageDraw.Draw(panel).text((6, 8), method, fill="white")
        panel.paste(image, ((target_width - image.width) // 2, 32))
        panels.append(panel)
    canvas = Image.new(
        "RGB", (sum(panel.width for panel in panels), max(panel.height for panel in panels)), "black"
    )
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def audit_case(case: Path) -> dict:
    methods = {name: method_report(case, name) for name in METHODS}
    protocols = {
        split: {methods[name]["evaluation"][split]["protocol"] for name in METHODS}
        for split in ("train_v48", "test_v12")
    }
    if any(len(value) != 1 for value in protocols.values()):
        raise RuntimeError(f"Mixed evaluation protocol under {case}")
    if methods["hairgs_25k"]["splats"] != 25000 or methods["unifur_25k"]["splats"] != 25000:
        raise RuntimeError(f"Fixed-budget method is not exactly 25k under {case}")
    initialization_hashes = {
        method: sha256(case / method / "stage1/input.ply")
        for method in ("hairgs_unrestricted", "hairgs_25k")
    }
    if len(set(initialization_hashes.values())) != 1:
        raise RuntimeError(f"HairGS baselines do not share initialization under {case}")
    report = {
        "schema": "unifur-animals60-case-audit-v1",
        "status": "complete",
        "case_id": case.name,
        "fit_views": 48,
        "heldout_views": 12,
        "render_size": [512, 512],
        "baseline_initialization_sha256": next(iter(initialization_hashes.values())),
        "methods": methods,
    }
    audit = case / "audit"
    audit.mkdir(parents=True, exist_ok=True)
    (audit / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    contact_sheet(case, "train_v48", audit / "comparison_train_v48.png")
    contact_sheet(case, "test_v12", audit / "comparison_test_v12.png")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--case")
    args = parser.parse_args()
    cases_root = args.benchmark_root / "cases"
    selected = [cases_root / args.case] if args.case else sorted(cases_root.iterdir())
    reports = []
    for case in selected:
        if all((case / method / "evaluation_test_v12/external_evaluation/evaluation.json").is_file() for method in METHODS):
            reports.append(audit_case(case))
    rows = []
    for report in reports:
        for method, result in report["methods"].items():
            rows.append({
                "case_id": report["case_id"],
                "method": method,
                "splats": result["splats"],
                "deploy_mib": result["deploy_bytes"] / 2**20,
                "end_to_end_train_seconds": result["end_to_end_train_seconds"],
                "peak_gpu_mib": result["end_to_end_peak_gpu_mib"],
                "render_fps": result["render_fps"],
                **{f"train_{key}": value for key, value in result["evaluation"]["train_v48"]["metrics"].items()},
                **{f"test_{key}": value for key, value in result["evaluation"]["test_v12"]["metrics"].items()},
            })
    audit_root = args.benchmark_root / "audit"
    audit_root.mkdir(parents=True, exist_ok=True)
    aggregate = {
        "schema": "unifur-animals60-benchmark-audit-v1",
        "status": "complete" if len(reports) == 8 else "partial",
        "completed_cases": len(reports),
        "target_cases": 8,
        "fixed_splat_budget": 25000,
        "fit_views": 48,
        "heldout_views": 12,
        "cases": reports,
    }
    (audit_root / "benchmark.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    if rows:
        with (audit_root / "benchmark.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({"status": aggregate["status"], "completed_cases": len(reports), "output": str(audit_root)}, indent=2))


if __name__ == "__main__":
    main()
