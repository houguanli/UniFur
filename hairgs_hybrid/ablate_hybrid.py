#!/usr/bin/env python3
"""Disable all primitives whose learned hard ownership is one hybrid route."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


ROUTE_NAMES = ("shell", "fin", "strand")


def softmax(value: np.ndarray, temperature: float) -> np.ndarray:
    shifted = value / temperature
    shifted -= shifted.max(axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=1, keepdims=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-ply", type=Path, required=True)
    parser.add_argument("--hybrid-state", type=Path, required=True)
    parser.add_argument("--drop", choices=ROUTE_NAMES, required=True)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--output-ply", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    ply = PlyData.read(args.input_ply)
    state = np.load(args.hybrid_state)
    route_ids = np.asarray(state["route_id"], dtype=np.int64)
    probabilities = softmax(
        np.asarray(state["route_logits"], dtype=np.float64), args.temperature
    )
    hard = probabilities.argmax(axis=1)
    target = ROUTE_NAMES.index(args.drop)
    removed = np.logical_and(route_ids >= 0, hard == target)
    segment_element = ply["segment"]
    if removed.shape != (segment_element.count,):
        raise ValueError("Hybrid state and PLY segment counts differ")
    segment = segment_element.data.copy()
    before = 1.0 / (1.0 + np.exp(-np.clip(segment["opacity"], -30.0, 30.0)))
    segment["opacity"][removed] = -30.0

    output_elements = [
        PlyElement.describe(segment, element.name)
        if element.name == "segment"
        else PlyElement.describe(element.data, element.name)
        for element in ply.elements
    ]
    args.output_ply.parent.mkdir(parents=True, exist_ok=True)
    PlyData(output_elements, text=False).write(args.output_ply)
    shutil.copy2(args.hybrid_state, args.output_ply.with_name("hybrid_state.npz"))
    report = {
        "schema": "hairgs-hybrid-hard-route-ablation-v1",
        "status": "complete",
        "dropped_route": args.drop,
        "temperature": args.temperature,
        "removed_segments": int(removed.sum()),
        "removed_effective_opacity_sum": float(before[removed].sum()),
        "removed_opacity_mass_fraction": float(
            before[removed].sum() / max(float(before.sum()), 1.0e-12)
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
