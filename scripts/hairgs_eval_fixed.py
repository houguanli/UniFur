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

from data import eval_data_loading_callbacks
from loss import compute_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate HairGS geometry with the bundled metric implementation.")
    parser.add_argument("--source_data_path", "-s", required=True)
    parser.add_argument("--pred_data_path", "-p", required=True)
    parser.add_argument("--pred_data_type", "-pt", default="gs")
    parser.add_argument("--out_json", required=True)
    args = parser.parse_args()

    gt_path = Path(args.source_data_path) / "hair_eval_data.npz"
    gt = eval_data_loading_callbacks["gt"](str(gt_path))
    pred = eval_data_loading_callbacks[args.pred_data_type](args.pred_data_path)
    metrics, thresholds = compute_metrics(pred, gt, bidirectional=True)
    payload = {
        "gt": str(gt_path),
        "prediction": args.pred_data_path,
        "prediction_type": args.pred_data_type,
        "bidirectional_orientation": True,
        "thresholds": thresholds,
        "metrics": {name: [float(value) for value in values] for name, values in metrics.items()},
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

