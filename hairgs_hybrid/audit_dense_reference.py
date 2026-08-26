#!/usr/bin/env python3
"""Compare compressed UniFur Hard with an uncompressed dense HairGS reference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from plyfile import PlyData

from hairgs_hybrid.audit_compression_sweep import METRICS, delta, load, method


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _segments(path: Path) -> int:
    return int(PlyData.read(path)["edge"].count)


def _train_metrics(root: Path) -> dict[str, float]:
    evaluation = load(root / "evaluation_train/external_evaluation/evaluation.json")
    return {name: float(evaluation["aggregate"][name]) for name in METRICS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hard", type=Path, required=True)
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--hard-source-ply", type=Path, required=True)
    parser.add_argument("--dense-source-ply", type=Path, required=True)
    parser.add_argument("--hard-initial-ply", type=Path, required=True)
    parser.add_argument("--dense-initial-ply", type=Path, required=True)
    parser.add_argument("--benchmark-name", default="render_benchmark_dense_compare.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    hard = method(args.hard, args.benchmark_name)
    dense = method(args.dense, args.benchmark_name)
    protocol_keys = {(row["protocol"], row["image_count"]) for row in (hard, dense)}
    if len(protocol_keys) != 1:
        raise ValueError(f"Evaluation protocol mismatch: {protocol_keys}")

    hashes = {
        "hard_source": _digest(args.hard_source_ply),
        "hard_initial": _digest(args.hard_initial_ply),
        "dense_source": _digest(args.dense_source_ply),
        "dense_initial": _digest(args.dense_initial_ply),
    }
    if hashes["hard_source"] != hashes["hard_initial"]:
        raise ValueError("Compressed UniFur initialization does not match its source PLY")
    if hashes["dense_source"] != hashes["dense_initial"]:
        raise ValueError("Dense HairGS initialization does not match the full source PLY")

    hard_count = _segments(args.hard_initial_ply)
    dense_count = _segments(args.dense_initial_ply)
    if not 0 < hard_count < dense_count:
        raise ValueError(f"Expected compressed count < dense count, got {hard_count}/{dense_count}")

    hard_efficiency = hard["efficiency"]
    dense_efficiency = dense["efficiency"]
    hard_train = _train_metrics(args.hard)
    dense_train = _train_metrics(args.dense)
    report = {
        "schema": "hairgs-dense-reference-audit-v1",
        "status": "complete",
        "protocol": hard["protocol"],
        "image_count": hard["image_count"],
        "optimization_steps": 5000,
        "segment_counts": {"unifur_hard": hard_count, "dense_hairgs": dense_count},
        "active_fraction": hard_count / dense_count,
        "initialization_sha256": hashes,
        "methods": {"unifur_hard": hard, "dense_hairgs": dense},
        "training_view_metrics": {
            "unifur_hard": hard_train,
            "dense_hairgs": dense_train,
            "unifur_minus_dense": {
                name: hard_train[name] - dense_train[name] for name in METRICS
            },
        },
        "unifur_minus_dense": delta(hard, dense),
        "relative_efficiency": {
            "training_time_reduction": 1.0
            - hard_efficiency["train_seconds"] / dense_efficiency["train_seconds"],
            "render_latency_reduction": 1.0
            - hard_efficiency["render_mean_ms"] / dense_efficiency["render_mean_ms"],
            "render_fps_increase": hard_efficiency["render_fps"]
            / dense_efficiency["render_fps"]
            - 1.0,
            "model_storage_reduction": 1.0
            - hard_efficiency["total_bytes"] / dense_efficiency["total_bytes"],
            "render_peak_allocated_reduction": 1.0
            - hard_efficiency["render_peak_allocated_mib"]
            / dense_efficiency["render_peak_allocated_mib"],
            "observed_train_gpu_peak_reduction": 1.0
            - hard_efficiency["observed_gpu_memory_peak_mib"]
            / dense_efficiency["observed_gpu_memory_peak_mib"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
