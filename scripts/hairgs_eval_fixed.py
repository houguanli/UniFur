#!/usr/bin/env python3
"""Compatibility wrapper for HairGS eval.py at commit 1658865.

That commit's eval.py asks compute_metrics(return_table=True), while the bundled
compute_metrics implementation returns only (metric_dict, thresholds). This
wrapper uses the unchanged loading callbacks and metric implementation, then
serializes their actual return value.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data import HairEvalData, eval_data_loading_callbacks
from loss import compute_metrics


def _load_hair_eval_npz(path: str) -> HairEvalData:
    """Load the common HairGS metric representation from an NPZ bundle.

    Unlike HairGS ground truth, an unstructured Gaussian baseline has no
    strand IDs or edges. Keeping those fields optional prevents us from
    inventing topology merely to obtain a strand-consistency score.
    """

    with np.load(path, allow_pickle=False) as data:
        points = np.asarray(data["points"], dtype=np.float64)
        directions = np.asarray(data["directions"], dtype=np.float64)
        norm = np.linalg.norm(directions, axis=1, keepdims=True)
        if np.any(norm <= 1e-12):
            raise ValueError("Prediction contains zero-length directions")
        directions = directions / norm
        strand_ids = (
            np.asarray(data["points_id_to_strand_id"], dtype=np.int64)
            if "points_id_to_strand_id" in data.files
            else None
        )
        edges = (
            np.asarray(data["edges"], dtype=np.int64)
            if "edges" in data.files
            else None
        )
    return HairEvalData(points, directions, strand_ids, edges)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate HairGS geometry with the bundled metric implementation.")
    parser.add_argument("--source_data_path", "-s", required=True)
    parser.add_argument("--pred_data_path", "-p", required=True)
    parser.add_argument("--pred_data_type", "-pt", default="gs")
    parser.add_argument("--out_json", required=True)
    args = parser.parse_args()

    gt_path = Path(args.source_data_path) / "hair_eval_data.npz"
    gt = eval_data_loading_callbacks["gt"](str(gt_path))
    pred = (
        _load_hair_eval_npz(args.pred_data_path)
        if args.pred_data_type == "hair_eval_npz"
        else eval_data_loading_callbacks[args.pred_data_type](args.pred_data_path)
    )
    metrics, thresholds = compute_metrics(pred, gt, bidirectional=True)
    serialized_metrics = {
        name: [float(value) for value in values]
        for name, values in metrics.items()
        if len(values) == len(thresholds)
    }
    payload = {
        "gt": str(gt_path),
        "prediction": args.pred_data_path,
        "prediction_type": args.pred_data_type,
        "bidirectional_orientation": True,
        "thresholds": thresholds,
        "metrics": serialized_metrics,
        "strand_consistency_applicable": pred.points_id_to_strand_id is not None,
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    header = ["metric", *thresholds]
    rows = [header]
    for name, values in payload["metrics"].items():
        rows.append([name, *[f"{value:.6f}" for value in values]])
    widths = [max(len(str(row[column])) for row in rows) for column in range(len(header))]
    for row in rows:
        print("  ".join(str(value).ljust(widths[index]) for index, value in enumerate(row)))
    print(f"json={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
