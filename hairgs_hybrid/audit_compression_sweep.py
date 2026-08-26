#!/usr/bin/env python3
"""Aggregate a same-budget Panda/WCurly compression sweep."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


METRICS = (
    "foreground_psnr",
    "masked_full_psnr",
    "masked_ssim",
    "masked_lpips",
    "full_psnr",
    "full_ssim",
    "full_lpips",
    "mask_iou",
    "false_negative_fraction",
    "false_positive_fraction",
    "largest_false_negative_component_fraction",
    "largest_false_positive_component_fraction",
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def elapsed(path: Path) -> float:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(
        r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*([0-9:.]+)", text
    )
    if match is None:
        raise ValueError(f"No elapsed time in {path}")
    fields = [float(value) for value in match.group(1).split(":")]
    if len(fields) == 2:
        return fields[0] * 60.0 + fields[1]
    return fields[0] * 3600.0 + fields[1] * 60.0 + fields[2]


def peak_gpu(path: Path) -> float:
    values = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) >= 2:
                values.append(float(row[1]))
    if not values:
        raise ValueError(f"No GPU samples in {path}")
    return max(values)


def artifact(root: Path) -> dict:
    iteration_dirs = sorted(
        (root / "point_cloud").glob("iteration_*"),
        key=lambda path: int(path.name.split("_")[-1]),
    )
    latest = iteration_dirs[-1]
    ply = latest / "point_cloud.ply"
    state = latest / "hybrid_state.npz"
    state_bytes = state.stat().st_size if state.is_file() else 0
    return {
        "ply_bytes": ply.stat().st_size,
        "sidecar_bytes": state_bytes,
        "total_bytes": ply.stat().st_size + state_bytes,
    }


def method(root: Path, benchmark_name: str = "render_benchmark.json") -> dict:
    evaluation = load(root / "evaluation_test/external_evaluation/evaluation.json")
    benchmark_path = root / "audit" / benchmark_name
    benchmark = load(benchmark_path) if benchmark_path.is_file() else None
    result = {
        "root": str(root.resolve()),
        "protocol": evaluation["protocol"],
        "image_count": evaluation["image_count"],
        "metrics": {name: evaluation["aggregate"][name] for name in METRICS},
        "efficiency": {
            "train_seconds": elapsed(root / "audit/train_time.txt"),
            "observed_gpu_memory_peak_mib": peak_gpu(root / "audit/train_gpu.csv"),
            **artifact(root),
        },
    }
    if benchmark is not None:
        result["efficiency"].update(
            {
                "render_mean_ms": benchmark["mean_ms_per_frame"],
                "render_fps": benchmark["fps_from_mean"],
                "render_peak_allocated_mib": benchmark["peak_allocated_mib"],
            }
        )
    return result


def delta(candidate: dict, control: dict) -> dict:
    metrics = {
        name: candidate["metrics"][name] - control["metrics"][name]
        for name in METRICS
    }
    # Set iteration depends on Python's hash seed. Sort the intersection so
    # repeated audits are byte reproducible as well as numerically identical.
    common = sorted(candidate["efficiency"].keys() & control["efficiency"].keys())
    efficiency = {
        name: candidate["efficiency"][name] - control["efficiency"][name]
        for name in common
        if isinstance(candidate["efficiency"][name], (int, float))
    }
    return {"metrics": metrics, "efficiency": efficiency}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = load(args.manifest)
    rows = []
    for entry in manifest["entries"]:
        candidate = method(Path(entry["candidate"]))
        control = method(Path(entry["control"]))
        rows.append(
            {
                "case": entry["case"],
                "active_budget": entry["active_budget"],
                "source_active": entry["source_active"],
                "active_fraction": entry["active_budget"] / entry["source_active"],
                "allocation": load(Path(entry["allocation"])),
                "candidate": candidate,
                "same_budget_hairgs": control,
                "candidate_minus_hairgs": delta(candidate, control),
            }
        )
    report = {
        "schema": "hairgs-hybrid-compression-sweep-v1",
        "status": "complete",
        "same_geometry_per_budget": True,
        "heldout_used_for_allocation": False,
        "entries": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
