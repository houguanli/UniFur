#!/usr/bin/env python3
"""Summarize train-view leave-one-route-out utility for adaptive allocation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROUTE_NAMES = ("shell", "fin", "strand")


def read(path: Path) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != "external-render-evaluation-v1":
        raise ValueError(f"Unsupported evaluation schema: {path}")
    return document


def keyed_frames(document: dict) -> dict[str, dict]:
    return {frame["image"]: frame for frame in document["per_frame"]}


def frame_delta(full: dict, dropped: dict) -> float:
    """Positive means that retaining the route improves training-view quality.

    One score unit is approximately one dB.  Foreground appearance is primary;
    masked appearance, silhouette IoU, and LPIPS prevent allocating capacity to
    a route that only improves a single RGB statistic while damaging structure.
    """

    return float(
        0.60 * (full["foreground_psnr"] - dropped["foreground_psnr"])
        + 0.20 * (full["masked_full_psnr"] - dropped["masked_full_psnr"])
        + 0.15 * 100.0 * (full["mask_iou"] - dropped["mask_iou"])
        + 0.05 * 100.0 * (dropped["masked_lpips"] - full["masked_lpips"])
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", type=Path, required=True)
    for route in ROUTE_NAMES:
        parser.add_argument(f"--drop-{route}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--floor-fraction", type=float, default=0.03)
    args = parser.parse_args()

    full = read(args.full)
    if not full["protocol"].endswith("-train"):
        raise ValueError("Full evaluation is not a training split")
    full_frames = keyed_frames(full)
    routes: dict[str, dict] = {}
    raw_weights = []
    for route in ROUTE_NAMES:
        dropped_path = getattr(args, f"drop_{route}")
        dropped = read(dropped_path)
        if not dropped["protocol"].endswith("-train"):
            raise ValueError(f"{route} ablation is not a training split")
        dropped_frames = keyed_frames(dropped)
        if full_frames.keys() != dropped_frames.keys():
            raise ValueError(f"Frame mismatch for {route}")
        delta = np.asarray(
            [frame_delta(full_frames[key], dropped_frames[key]) for key in full_frames],
            dtype=np.float64,
        )
        positive = np.maximum(delta, 0.0)
        positive_fraction = float(np.mean(delta > 0.0))
        # A route must help across views, not just win on a few easy cameras.
        raw_weight = float(positive.mean() * positive_fraction**2)
        raw_weights.append(raw_weight)
        routes[route] = {
            "drop_evaluation": str(dropped_path.resolve()),
            "image_count": int(len(delta)),
            "mean_signed_score": float(delta.mean()),
            "mean_positive_score": float(positive.mean()),
            "positive_view_fraction": positive_fraction,
            "negative_view_fraction": float(np.mean(delta < 0.0)),
            "minimum_view_score": float(delta.min()),
            "maximum_view_score": float(delta.max()),
            "raw_allocation_weight": raw_weight,
        }

    raw = np.asarray(raw_weights, dtype=np.float64)
    floor = max(float(raw.max()) * args.floor_fraction, 1.0e-6)
    weight = np.maximum(raw, floor)
    weight /= weight.mean()
    for route, value in zip(ROUTE_NAMES, weight, strict=True):
        routes[route]["allocation_weight"] = float(value)

    report = {
        "schema": "train-route-contribution-v1",
        "protocol_id": args.protocol_id,
        "split": "train",
        "heldout_used": False,
        "full_evaluation": str(args.full.resolve()),
        "score": (
            "0.60*dFG-PSNR + 0.20*dMasked-PSNR + "
            "0.15*100*dIoU + 0.05*100*dLPIPS"
        ),
        "cross_view_rule": "mean(positive score) * positive_view_fraction^2",
        "floor_fraction": args.floor_fraction,
        "routes": routes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
