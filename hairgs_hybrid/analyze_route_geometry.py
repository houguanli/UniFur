#!/usr/bin/env python3
"""Summarize and export learned route ownership in world space."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree


ROUTE_NAMES = ("shell", "fin", "strand")
ROUTE_COLORS = np.asarray(((70, 130, 255), (80, 220, 120), (255, 90, 70)), dtype=np.uint8)


def quantiles(value: np.ndarray) -> list[float]:
    return [float(item) for item in np.quantile(value, [0, 0.25, 0.5, 0.75, 1])]


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--output-ply", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    ply = PlyData.read(args.ply)
    vertex = ply["vertex"].data
    endpoints = np.stack((vertex["x"], vertex["y"], vertex["z"]), axis=1)
    edge = ply["edge"].data
    pairs = np.stack((edge["vertex1"], edge["vertex2"]), axis=1).astype(np.int64)
    segment = ply["segment"].data
    ref = ply["ref_strand_root"].data
    scalp = np.stack((ref["x"], ref["y"], ref["z"]), axis=1)
    state = np.load(args.state)
    route_id = np.asarray(state["route_id"], dtype=np.int64)
    logits = np.asarray(state["route_logits"], dtype=np.float64) / args.temperature
    logits -= logits.max(axis=1, keepdims=True)
    probability = np.exp(logits)
    probability /= probability.sum(axis=1, keepdims=True)
    hard = probability.argmax(axis=1)
    confidence = probability.max(axis=1)
    hair = route_id >= 0
    if route_id.shape != (pairs.shape[0],):
        raise ValueError("State/PLY segment mismatch")

    points = endpoints[pairs]
    centers = points.mean(axis=1)
    lengths = np.linalg.norm(points[:, 1] - points[:, 0], axis=1)
    scalp_distance, _ = cKDTree(scalp).query(centers, k=1)
    opacity = sigmoid(np.asarray(segment["opacity"], dtype=np.float64))
    rows = {}
    for index, name in enumerate(ROUTE_NAMES):
        selected = np.logical_and(hair, hard == index)
        rows[name] = {
            "count": int(selected.sum()),
            "probability_mass": float(probability[hair, index].sum()),
            "mean_confidence": float(confidence[selected].mean()),
            "opacity_mass": float(opacity[selected].sum()),
            "segment_length_quantiles": quantiles(lengths[selected]),
            "scalp_distance_quantiles": quantiles(scalp_distance[selected]),
        }
    report = {
        "schema": "hairgs-hybrid-route-geometry-v1",
        "hair_segment_count": int(hair.sum()),
        "routes": rows,
        "semantic_order_checks": {
            "shell_median_scalp_distance_le_fin": rows["shell"][
                "scalp_distance_quantiles"
            ][2]
            <= rows["fin"]["scalp_distance_quantiles"][2],
            "fin_median_scalp_distance_le_strand": rows["fin"][
                "scalp_distance_quantiles"
            ][2]
            <= rows["strand"]["scalp_distance_quantiles"][2],
            "shell_median_length_le_strand": rows["shell"][
                "segment_length_quantiles"
            ][2]
            <= rows["strand"]["segment_length_quantiles"][2],
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    selected_centers = centers[hair].astype(np.float32)
    selected_hard = hard[hair].astype(np.uint8)
    selected_confidence = confidence[hair].astype(np.float32)
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
        ("route", "u1"),
        ("confidence", "f4"),
    ]
    output = np.empty(selected_centers.shape[0], dtype=dtype)
    output["x"], output["y"], output["z"] = selected_centers.T
    colors = ROUTE_COLORS[selected_hard]
    output["red"], output["green"], output["blue"] = colors.T
    output["route"] = selected_hard
    output["confidence"] = selected_confidence
    args.output_ply.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(output, "vertex")], text=False).write(args.output_ply)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
