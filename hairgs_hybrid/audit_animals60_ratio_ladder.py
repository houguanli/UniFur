#!/usr/bin/env python3
"""Audit the complete Animals60 percentage-compression ladder."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from plyfile import PlyData


CASES = (
    "A003_silver_fox", "A008_gray_wolf", "A027_polar_bear", "A029_goat",
    "A037_horse", "A045_rabbit", "A050_horned_cow", "A063_wild_ass",
)
COMPRESSIONS = (0.75, 0.875, 0.9375)
METHODS = ("hairgs_native", "unifur")
METRICS = (
    "foreground_psnr", "full_psnr", "masked_ssim", "masked_lpips",
    "mask_iou", "false_negative_fraction", "false_positive_fraction",
    "largest_false_negative_component_fraction",
    "largest_false_positive_component_fraction",
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def final_ply(model: Path) -> Path:
    values = []
    for path in model.glob("point_cloud/iteration_*/point_cloud.ply"):
        values.append((int(path.parent.name.removeprefix("iteration_")), path))
    if not values:
        raise FileNotFoundError(model)
    return max(values)[1]


def elapsed_seconds(path: Path) -> float:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(
        r"Elapsed \(wall clock\) time.*?\):\s*([0-9]+(?::[0-9]+)*(?:\.[0-9]+)?)",
        text,
    )
    if not match:
        raise ValueError(f"No elapsed time in {path}")
    parts = [float(value) for value in match.group(1).split(":")]
    total = 0.0
    for value in parts:
        total = 60.0 * total + value
    return total


def peak_gpu_mib(audit: Path) -> float:
    values = []
    for path in audit.glob("*gpu.csv"):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            fields = line.split(",")
            if len(fields) >= 2:
                try:
                    values.append(float(fields[1].strip()))
                except ValueError:
                    pass
    return max(values) if values else float("nan")


def find_run(case: Path, compression: float) -> Path:
    prefix = f"c{compression * 100:g}_"
    values = [path for path in case.iterdir() if path.name.startswith(prefix)]
    if len(values) != 1 or not (values[0] / "COMPLETE").is_file():
        raise RuntimeError(f"Expected one complete {prefix} run under {case}")
    return values[0]


def plot_curve(summary: dict, output: Path) -> None:
    x = np.asarray([100.0 * value for value in COMPRESSIONS])
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), dpi=150)
    specs = (
        ("foreground_psnr", "Foreground PSNR (dB)", True),
        ("full_psnr", "Full PSNR (dB)", True),
        ("mask_iou", "Mask IoU", True),
    )
    for axis, (metric, label, _) in zip(axes, specs):
        for method, title, marker in (
            ("hairgs_native", "HairGS native", "o"),
            ("unifur", "UniFur", "s"),
        ):
            y = [summary[f"{value:g}"][method][metric] for value in COMPRESSIONS]
            axis.plot(x, y, marker=marker, linewidth=2, label=title)
        axis.set_xlabel("Compression removed (%)")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Animals60: quality versus unrestricted-relative compression")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratio-root", type=Path, required=True)
    parser.add_argument("--fixed-root", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for case_name in CASES:
        case = args.ratio_root / "cases" / case_name
        fixed = args.fixed_root / "cases" / case_name
        input_report = load(fixed / "protocol/input_report.json")
        if input_report["heldout_geometry_used"] is not False:
            raise RuntimeError(f"Held-out leakage in {case_name}")
        dense = final_ply(fixed / "hairgs_unrestricted/model")
        dense_count = int(PlyData.read(dense)["edge"].count)
        dense_hash = sha256(dense)
        for compression in COMPRESSIONS:
            run = find_run(case, compression)
            comparison = load(run / "comparison.json")
            target = int(comparison["target_segments"])
            if int(comparison["source_segments"]) != dense_count:
                raise RuntimeError(f"Dense count mismatch: {run}")
            expected = int(round(dense_count * (1.0 - compression)))
            if target != expected:
                raise RuntimeError(f"Ratio budget mismatch: {run}")
            bank = load(run / "unifur/bank/candidate_bank_report.json")
            native_compression = load(run / "hairgs_native/model/compression_report.json")
            if bank["input_sha256"] != dense_hash or native_compression["input_sha256"] != dense_hash:
                raise RuntimeError(f"Teacher provenance mismatch: {run}")
            routes = load(run / "unifur/hard/audit/topology_hard_projection.json")
            for method in METHODS:
                method_root = run / method
                ply = final_ply(method_root / "model")
                count = int(PlyData.read(ply)["edge"].count)
                if count != target:
                    raise RuntimeError(f"Non-exact budget {count}!={target}: {ply}")
                test = load(method_root / "evaluation_test_v12/external_evaluation/evaluation.json")
                train = load(method_root / "evaluation_train_v48/external_evaluation/evaluation.json")
                if len(test["per_frame"]) != 12 or len(train["per_frame"]) != 48:
                    raise RuntimeError(f"Split mismatch: {method_root}")
                benchmark = load(method_root / "audit/render_benchmark.json")
                state_files = [ply]
                if method == "unifur":
                    state_files.extend(
                        path for path in ply.parent.glob("*.npz") if path.is_file()
                    )
                    time_files = (method_root / "audit/soft_time.txt", method_root / "audit/refine_time.txt")
                else:
                    time_files = (method_root / "audit/train_time.txt",)
                row = {
                    "case": case_name,
                    "compression": compression,
                    "retention": 1.0 - compression,
                    "method": method,
                    "source_segments": dense_count,
                    "target_segments": target,
                    "compression_ratio_x": dense_count / target,
                    "deployment_bytes": sum(path.stat().st_size for path in state_files),
                    "training_seconds": sum(elapsed_seconds(path) for path in time_files),
                    "peak_gpu_mib": peak_gpu_mib(method_root / "audit"),
                    "fps": benchmark["fps_from_mean"],
                    **{f"test_{key}": test["aggregate"][key] for key in METRICS},
                    **{f"train_{key}": train["aggregate"][key] for key in METRICS},
                }
                if method == "unifur":
                    counts = routes["exported_splat_counts"]
                    row.update(
                        route_shell=counts["shell"] / target,
                        route_fin=counts["fin"] / target,
                        route_strand=counts["strand"] / target,
                        route_confidence=routes["mean_max_probability"],
                        route_entropy=routes["mean_entropy"],
                    )
                rows.append(row)

    summary = {}
    for compression in COMPRESSIONS:
        by_method = {}
        for method in METHODS:
            selected = [row for row in rows if row["compression"] == compression and row["method"] == method]
            aggregate = {
                key: float(np.mean([row[f"test_{key}"] for row in selected]))
                for key in METRICS
            }
            aggregate.update(
                target_segments=float(np.mean([row["target_segments"] for row in selected])),
                deployment_mib=float(np.mean([row["deployment_bytes"] for row in selected])) / 2**20,
                training_seconds=float(np.mean([row["training_seconds"] for row in selected])),
                peak_gpu_mib=float(np.mean([row["peak_gpu_mib"] for row in selected])),
                fps=float(np.mean([row["fps"] for row in selected])),
            )
            if method == "unifur":
                aggregate.update(
                    route_shell=float(np.mean([row["route_shell"] for row in selected])),
                    route_fin=float(np.mean([row["route_fin"] for row in selected])),
                    route_strand=float(np.mean([row["route_strand"] for row in selected])),
                    route_confidence=float(np.mean([row["route_confidence"] for row in selected])),
                )
            by_method[method] = aggregate
        summary[f"{compression:g}"] = by_method
    report = {
        "schema": "unifur-animals60-ratio-ladder-audit-v1",
        "status": "complete",
        "case_count": len(CASES),
        "compression_levels": list(COMPRESSIONS),
        "protocol": {"fit_views": 48, "heldout_views": 12, "resolution": [512, 512]},
        "summary": summary,
        "rows": rows,
    }
    audit = args.ratio_root / "audit"
    audit.mkdir(parents=True, exist_ok=True)
    json_path = audit / "ratio_benchmark.json"
    csv_path = audit / "ratio_benchmark.csv"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    fields = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    plot_curve(summary, audit / "ratio_quality_curve.png")
    print(json.dumps({"status": "complete", "rows": len(rows), "json": str(json_path.resolve()), "csv": str(csv_path.resolve())}, indent=2))


if __name__ == "__main__":
    main()
