#!/usr/bin/env python3
"""Build one reproducible quality/efficiency audit for Panda and WCurly."""

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
    fields = [float(value) for value in match.group(1).split(":" )]
    if len(fields) == 2:
        return fields[0] * 60.0 + fields[1]
    return fields[0] * 3600.0 + fields[1] * 60.0 + fields[2]


def gpu_peak(path: Path) -> float:
    values = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) == 3:
                values.append(float(row[1]))
    if not values:
        raise ValueError(f"No GPU samples in {path}")
    return max(values)


def artifact(root: Path) -> dict:
    ply = sorted((root / "point_cloud").glob("iteration_*/point_cloud.ply"))[-1]
    states = sorted((root / "point_cloud").glob("iteration_*/hybrid_state.npz"))
    state_bytes = states[-1].stat().st_size if states else 0
    return {
        "ply": str(ply.resolve()),
        "ply_bytes": ply.stat().st_size,
        "sidecar_bytes": state_bytes,
        "total_bytes": ply.stat().st_size + state_bytes,
    }


def method(root: Path) -> dict:
    splits = {}
    for split in ("train", "test"):
        evaluation = load(
            root / f"evaluation_{split}/external_evaluation/evaluation.json"
        )
        splits[split] = {
            "protocol": evaluation["protocol"],
            "image_count": evaluation["image_count"],
            **{name: evaluation["aggregate"][name] for name in METRICS},
        }
    benchmark = load(root / "audit/render_benchmark.json")
    return {
        "root": str(root.resolve()),
        "splits": splits,
        "efficiency": {
            "train_seconds": elapsed(root / "audit/train_time.txt"),
            "observed_gpu_memory_peak_mib": gpu_peak(root / "audit/train_gpu.csv"),
            "render_mean_ms": benchmark["mean_ms_per_frame"],
            "render_fps": benchmark["fps_from_mean"],
            **artifact(root),
        },
    }


def delta(candidate: dict, control: dict) -> dict:
    return {
        name: candidate["splits"]["test"][name] - control["splits"]["test"][name]
        for name in METRICS
    } | {
        "train_seconds": candidate["efficiency"]["train_seconds"]
        - control["efficiency"]["train_seconds"],
        "artifact_bytes": candidate["efficiency"]["total_bytes"]
        - control["efficiency"]["total_bytes"],
        "render_fps": candidate["efficiency"]["render_fps"]
        - control["efficiency"]["render_fps"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "panda_candidate",
        "panda_control",
        "hair_candidate",
        "hair_strict_control",
        "hair_standard_control",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--panda-allocation", type=Path, required=True)
    parser.add_argument("--hair-allocation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    methods = {
        "panda_candidate": method(args.panda_candidate),
        "panda_control": method(args.panda_control),
        "hair_candidate": method(args.hair_candidate),
        "hair_strict_control": method(args.hair_strict_control),
        "hair_standard_control": method(args.hair_standard_control),
    }
    report = {
        "schema": "adaptive-shell-fin-strand-crosscase-audit-v1",
        "status": "complete",
        "allocation_uses_heldout": False,
        "cases": {
            "panda": {
                "allocation": load(args.panda_allocation),
                "candidate": methods["panda_candidate"],
                "same_budget_hairgs": methods["panda_control"],
                "delta_candidate_minus_control": delta(
                    methods["panda_candidate"], methods["panda_control"]
                ),
            },
            "wcurly": {
                "allocation": load(args.hair_allocation),
                "candidate": methods["hair_candidate"],
                "strict_hair_only_hairgs": methods["hair_strict_control"],
                "standard_trainable_head_hairgs": methods["hair_standard_control"],
                "delta_candidate_minus_strict_control": delta(
                    methods["hair_candidate"], methods["hair_strict_control"]
                ),
                "delta_candidate_minus_standard_control": delta(
                    methods["hair_candidate"], methods["hair_standard_control"]
                ),
                "control_protocol_note": (
                    "The strict control freezes the 29,278 head/base segments and "
                    "trains only 12,000 hair segments with the same hair-only loss. "
                    "The standard control updates all 41,278 segments and uses the "
                    "upstream full-frame HairGS loss."
                ),
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
