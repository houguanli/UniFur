#!/usr/bin/env python3
"""Project a learned soft router to compact hard ownership without retraining."""

from __future__ import annotations

import argparse
import json
import shutil
from argparse import Namespace
from pathlib import Path

import numpy as np

from hairgs_hybrid.compact_state import save_compact_hard_state


ROUTE_NAMES = ("shell", "fin", "strand")


def softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    values = np.exp(scaled)
    return values / values.sum(axis=1, keepdims=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--source-iteration", type=int, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--output-iteration", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    args = parser.parse_args()
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")

    source = args.source_model / f"point_cloud/iteration_{args.source_iteration}"
    output = args.output_model / f"point_cloud/iteration_{args.output_iteration}"
    state_path = source / "hybrid_state.npz"
    ply_path = source / "point_cloud.ply"
    if not state_path.is_file() or not ply_path.is_file():
        raise FileNotFoundError("source soft checkpoint is incomplete")
    with np.load(state_path, allow_pickle=False) as state:
        route_id = np.asarray(state["route_id"], dtype=np.int8)
        logits = np.asarray(state["route_logits"], dtype=np.float32)
        strengths = np.asarray(state["shape_strength_raw"], dtype=np.float32)
        rolls = np.asarray(state["roll_residual"], dtype=np.float32).reshape(-1)
    if logits.shape != (len(route_id), 3) or strengths.shape != (len(route_id), 2):
        raise ValueError("soft checkpoint arrays are inconsistent")

    hair = route_id >= 0
    probability = softmax(logits.astype(np.float64), args.temperature)
    hard = route_id.copy()
    hard[hair] = probability[hair].argmax(axis=1).astype(np.int8)
    shell = hard == 0
    fin = hard == 1
    non_strand = np.logical_or(shell, fin)

    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ply_path, output / "point_cloud.ply")
    save_compact_hard_state(
        output / "hybrid_state.npz",
        hard,
        strengths[shell, 0],
        strengths[fin, 1],
        rolls[non_strand],
    )
    cfg = args.source_model / "cfg_args"
    if cfg.is_file():
        saved_args = eval(cfg.read_text(encoding="utf-8"), {"Namespace": Namespace})
        saved_args.model_path = str(args.output_model)
        saved_args.hybrid_route_mode = "hard"
        (args.output_model / "cfg_args").write_text(
            repr(saved_args), encoding="utf-8"
        )
    entropy = -(
        probability[hair] * np.log(np.maximum(probability[hair], 1.0e-12))
    ).sum(axis=1)
    report = {
        "schema": "soft-to-compact-hard-projection-v1",
        "source_model": str(args.source_model.resolve()),
        "source_iteration": args.source_iteration,
        "output_iteration": args.output_iteration,
        "temperature": args.temperature,
        "hair_segment_count": int(hair.sum()),
        "hard_route_counts": {
            name: int(np.count_nonzero(np.logical_and(hair, hard == index)))
            for index, name in enumerate(ROUTE_NAMES)
        },
        "soft_route_mass": {
            name: float(probability[hair, index].sum())
            for index, name in enumerate(ROUTE_NAMES)
        },
        "mean_entropy": float(entropy.mean()),
        "mean_max_probability": float(probability[hair].max(axis=1).mean()),
    }
    (args.output_model / "audit").mkdir(parents=True, exist_ok=True)
    (args.output_model / "audit/hard_projection.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
