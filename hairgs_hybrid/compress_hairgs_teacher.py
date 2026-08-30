#!/usr/bin/env python3
"""Aggregate a dense HairGS teacher to an exact native segment budget."""

from __future__ import annotations

import argparse
import hashlib
import json
from argparse import Namespace
from pathlib import Path

import numpy as np

from hairgs_hybrid.build_topology_candidate_bank import (
    aggregate_labeled,
    balanced_spatial_partition,
    groups_to_labels,
)
from hairgs_hybrid.scaffold_io import load_scaffold, write_scaffold


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-ply", type=Path, required=True)
    parser.add_argument("--source-cfg", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--output-iteration", type=int, default=0)
    parser.add_argument("--target-active", type=int, required=True)
    args = parser.parse_args()

    source = load_scaffold(args.input_ply)
    source_count = len(source["centers"])
    if not 1 <= args.target_active <= source_count:
        raise ValueError(
            f"Target {args.target_active} outside dense source [1,{source_count}]"
        )
    groups = balanced_spatial_partition(
        np.asarray(source["centers"]),
        np.arange(source_count, dtype=np.int64),
        args.target_active,
    )
    labels = groups_to_labels(groups, source_count)
    compressed = aggregate_labeled(source, labels)
    output = (
        args.output_model
        / "point_cloud"
        / f"iteration_{args.output_iteration}"
        / "point_cloud.ply"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_scaffold(output, compressed)

    saved = eval(args.source_cfg.read_text(encoding="utf-8"), {"Namespace": Namespace})
    saved.model_path = str(args.output_model.resolve())
    saved.source_path = str(args.dataset.resolve())
    args.output_model.mkdir(parents=True, exist_ok=True)
    (args.output_model / "cfg_args").write_text(repr(saved), encoding="utf-8")
    report = {
        "schema": "hairgs-native-ratio-compression-v1",
        "input_ply": str(args.input_ply.resolve()),
        "input_sha256": sha256(args.input_ply),
        "source_segments": source_count,
        "target_segments": args.target_active,
        "actual_segments": len(compressed["centers"]),
        "compression": 1.0 - args.target_active / source_count,
        "mean_source_segments_per_output": source_count / args.target_active,
        "output_ply": str(output.resolve()),
        "output_sha256": sha256(output),
    }
    (args.output_model / "compression_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
