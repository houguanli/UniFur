#!/usr/bin/env python3
"""Physically prune a soft topology bank to an exact active-splat budget."""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

import numpy as np

from hairgs_hybrid.compact_state import save_compact_hard_state
from hairgs_hybrid.scaffold_io import (
    ROUTE_NAMES,
    load_scaffold,
    subset,
    write_scaffold,
)
from hairgs_hybrid.hybrid_model import TOPOLOGY_STATE_SCHEMA


def select_exact_budget(
    logits: np.ndarray, costs: np.ndarray, target: int
) -> np.ndarray:
    """Lagrangian assignment followed by an exact minimum-regret repair."""

    logits = np.asarray(logits, dtype=np.float64)
    costs = np.asarray(costs, dtype=np.int64)
    if logits.shape != costs.shape or logits.ndim != 2 or logits.shape[1] != 3:
        raise ValueError("logits/costs must both have shape (groups, 3)")
    minimum = int(costs.min(axis=1).sum())
    maximum = int(costs.max(axis=1).sum())
    if not minimum <= target <= maximum:
        raise ValueError(f"Target {target} is outside [{minimum}, {maximum}]")

    low, high = -100.0, 100.0
    candidates: list[np.ndarray] = []
    for _ in range(80):
        value = 0.5 * (low + high)
        assignment = np.argmax(logits - value * costs, axis=1)
        candidates.append(assignment)
        active = int(costs[np.arange(len(costs)), assignment].sum())
        if active > target:
            low = value
        else:
            high = value
    assignment = min(
        candidates,
        key=lambda value: abs(
            int(costs[np.arange(len(costs)), value].sum()) - target
        ),
    ).copy()

    rows = np.arange(len(costs))
    while True:
        current_cost = costs[rows, assignment]
        active = int(current_cost.sum())
        remaining = target - active
        if remaining == 0:
            return assignment.astype(np.int8)
        delta = costs - current_cost[:, None]
        if remaining > 0:
            valid = (delta > 0) & (delta <= remaining)
        else:
            valid = (delta < 0) & (-delta <= -remaining)
        valid[rows, assignment] = False
        if not np.any(valid):
            raise RuntimeError("Could not repair assignment to the exact budget")
        current_score = logits[rows, assignment]
        regret = current_score[:, None] - logits
        # Prefer the smallest score loss per changed active splat.  A stable
        # flat index tie-break makes projection exactly reproducible.
        objective = np.where(valid, regret / np.maximum(np.abs(delta), 1), np.inf)
        flat = int(np.argmin(objective))
        group, route = np.unravel_index(flat, objective.shape)
        assignment[group] = route


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--source-iteration", type=int, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--output-iteration", type=int, required=True)
    parser.add_argument("--target-active", type=int, required=True)
    parser.add_argument(
        "--shuffle-assignment",
        action="store_true",
        help="Deterministic route-shuffle control preserving route counts and budget",
    )
    args = parser.parse_args()

    source = args.source_model / f"point_cloud/iteration_{args.source_iteration}"
    output = args.output_model / f"point_cloud/iteration_{args.output_iteration}"
    with np.load(source / "hybrid_state.npz", allow_pickle=False) as state:
        schema = str(np.asarray(state["schema"]).item())
        if schema != TOPOLOGY_STATE_SCHEMA:
            raise ValueError(f"Expected {TOPOLOGY_STATE_SCHEMA}, got {schema}")
        group_id = np.asarray(state["topology_group_id"], dtype=np.int64)
        candidate_route = np.asarray(
            state["topology_candidate_route"], dtype=np.int64
        )
        logits = np.asarray(state["topology_group_logits"], dtype=np.float64)
        costs = np.asarray(state["topology_route_counts"], dtype=np.int64)
        strengths = np.asarray(state["shape_strength_raw"], dtype=np.float32)
        rolls = np.asarray(state["roll_residual"], dtype=np.float32).reshape(-1)

    selected = select_exact_budget(logits, costs, args.target_active)
    learned_selected = selected.copy()
    if args.shuffle_assignment:
        selected = np.random.default_rng(0).permutation(selected)
    selected_cost = costs[np.arange(len(costs)), selected]
    if int(selected_cost.sum()) != args.target_active:
        raise RuntimeError("Projected assignment missed the requested budget")
    keep = candidate_route == selected[group_id]
    if int(keep.sum()) != args.target_active:
        raise RuntimeError("Candidate pruning and cost accounting disagree")

    scaffold = load_scaffold(source / "point_cloud.ply")
    compact = subset(scaffold, np.flatnonzero(keep))
    route_ids = candidate_route[keep].astype(np.int8)
    kept_strengths = strengths[keep]
    kept_rolls = rolls[keep]
    output.mkdir(parents=True, exist_ok=True)
    write_scaffold(output / "point_cloud.ply", compact)
    np.savez_compressed(
        output / "routes.npz",
        route_names=np.asarray(ROUTE_NAMES),
        route_id=route_ids,
    )
    shell = route_ids == 0
    fin = route_ids == 1
    non_strand = np.logical_or(shell, fin)
    save_compact_hard_state(
        output / "hybrid_state.npz",
        route_ids,
        kept_strengths[shell, 0],
        kept_strengths[fin, 1],
        kept_rolls[non_strand],
    )

    cfg = args.source_model / "cfg_args"
    if cfg.is_file():
        saved_args = eval(cfg.read_text(encoding="utf-8"), {"Namespace": Namespace})
        saved_args.model_path = str(args.output_model)
        saved_args.hybrid_route_mode = "hard"
        saved_args.route_sidecar = str((output / "routes.npz").resolve())
        (args.output_model / "cfg_args").write_text(repr(saved_args), encoding="utf-8")

    probability = np.exp(logits - logits.max(axis=1, keepdims=True))
    probability /= probability.sum(axis=1, keepdims=True)
    report = {
        "schema": "hairgs-topology-hard-projection-v1",
        "source_model": str(args.source_model.resolve()),
        "source_iteration": args.source_iteration,
        "output_iteration": args.output_iteration,
        "target_active_splats": args.target_active,
        "actual_active_splats": int(keep.sum()),
        "candidate_splats": int(len(keep)),
        "group_count": int(len(logits)),
        "shuffle_assignment": args.shuffle_assignment,
        "learned_group_counts": {
            name: int(np.count_nonzero(learned_selected == route))
            for route, name in enumerate(ROUTE_NAMES)
        },
        "exported_group_counts": {
            name: int(np.count_nonzero(selected == route))
            for route, name in enumerate(ROUTE_NAMES)
        },
        "exported_splat_counts": {
            name: int(np.count_nonzero(route_ids == route))
            for route, name in enumerate(ROUTE_NAMES)
        },
        "soft_route_mass": {
            name: float(probability[:, route].sum())
            for route, name in enumerate(ROUTE_NAMES)
        },
        "mean_max_probability": float(probability.max(axis=1).mean()),
        "mean_entropy": float(
            -(probability * np.log(np.maximum(probability, 1.0e-12))).sum(axis=1).mean()
        ),
    }
    (args.output_model / "audit").mkdir(parents=True, exist_ok=True)
    (args.output_model / "audit/topology_hard_projection.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
