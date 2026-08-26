#!/usr/bin/env python3
"""Compress a learned HairGS hybrid scaffold under one global point budget.

The allocator is shared by fur and human-hair cases.  It uses only a hybrid
teacher fitted on the training views, the training masks/cameras, and geometric
features of the source scaffold.  No species label and no held-out image enters
the allocation.

Base/head segments (route ``-1``) are copied exactly and do not consume the
active hair/fur budget.  The active budget is inferred from learned route
probabilities and route-specific evidence, rather than fixed percentages:

* shell: learned shell probability, local density, and mask-interior support;
* fin: learned fin probability and multi-view silhouette-boundary support;
* strand: learned strand probability, length, and local direction frequency.

The default shared-coverage policy first performs route-neutral optical
aggregation, exactly matching the native control's compressed PLY, and then
assigns routes on that common support.  This prevents route allocation itself
from creating spatial holes.  The earlier route-specialized top-k policy is
retained only as an explicit ablation.  A route-unaware exact-budget native
control is emitted from the same source PLY for a fair capacity comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from hairgs_hybrid.compress_panda_scaffold import (
    ROUTE_NAMES,
    aggregate,
    concatenate,
    load_scaffold,
    multi_view_boundary_score,
    rank01,
    sha256,
    sigmoid,
    strand_score,
    subset,
    write_scaffold,
)


def softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    exponential = np.exp(scaled)
    return exponential / exponential.sum(axis=1, keepdims=True)


def largest_remainder(mass: np.ndarray, total: int, minimum: int) -> np.ndarray:
    if total < minimum * len(mass):
        raise ValueError("Target is too small for the requested route minimum")
    residual = total - minimum * len(mass)
    normalized = mass / np.maximum(mass.sum(), 1.0e-12)
    real = normalized * residual
    result = np.floor(real).astype(np.int64) + minimum
    remainder = total - int(result.sum())
    if remainder:
        order = np.argsort(real - np.floor(real), kind="mergesort")
        result[order[-remainder:]] += 1
    return result


def local_density(centers: np.ndarray, neighbors: int) -> np.ndarray:
    k = min(neighbors + 1, len(centers))
    distances = cKDTree(centers).query(centers, k=k, workers=-1)[0]
    radius = distances[:, -1] if distances.ndim == 2 else distances
    return 1.0 - rank01(radius)


def one_hot_logits(route_ids: np.ndarray) -> np.ndarray:
    logits = np.full((len(route_ids), 3), -2.0, dtype=np.float32)
    for route in range(3):
        logits[route_ids == route, route] = 2.0
    logits[route_ids == -1, 2] = 2.0
    return logits


def select_without_replacement(
    scores: np.ndarray, available: np.ndarray, count: int
) -> np.ndarray:
    candidates = np.flatnonzero(available)
    if len(candidates) < count:
        raise RuntimeError(f"Only {len(candidates)} candidates for a budget of {count}")
    order = np.argsort(scores[candidates], kind="mergesort")
    return candidates[order[-count:]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-ply", type=Path, required=True)
    parser.add_argument("--teacher-state", type=Path, required=True)
    parser.add_argument("--camera-manifest", type=Path, required=True)
    parser.add_argument("--mask-dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target-active", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--minimum-route-fraction", type=float, default=0.03)
    parser.add_argument("--boundary-sigma", type=float, default=4.0)
    parser.add_argument("--neighbors", type=int, default=12)
    parser.add_argument(
        "--geometry-policy",
        choices=("shared-coverage", "route-specialized"),
        default="shared-coverage",
        help=(
            "shared-coverage first builds the same uniform compressed geometry "
            "as the native control and assigns routes on that support; "
            "route-specialized retains the earlier route-specific top-k geometry"
        ),
    )
    parser.add_argument(
        "--route-contribution-report",
        type=Path,
        help=(
            "Optional train-view leave-one-route-out report. Positive "
            "multi-view contributions reweight the learned route mass; "
            "held-out reports are rejected."
        ),
    )
    args = parser.parse_args()

    source = load_scaffold(args.input_ply)
    state = np.load(args.teacher_state)
    source_count = len(np.asarray(source["centers"]))
    route_id = np.asarray(state["route_id"], dtype=np.int8)
    logits = np.asarray(state["route_logits"], dtype=np.float64)
    if route_id.shape != (source_count,) or logits.shape != (source_count, 3):
        raise ValueError("Teacher state and source PLY do not share point ordering")
    if not np.all(np.isin(route_id, [-1, 0, 1, 2])):
        raise ValueError("Unsupported teacher route id")

    base_ids = np.flatnonzero(route_id == -1)
    active_ids = np.flatnonzero(route_id >= 0)
    if not 3 <= args.target_active < len(active_ids):
        raise ValueError("target-active must compress the active scaffold")

    active = subset(source, active_ids)
    probability = softmax(logits[active_ids], args.temperature)
    entropy = -(probability * np.log(np.maximum(probability, 1.0e-12))).sum(axis=1)
    confidence = 1.0 - entropy / np.log(3.0)

    fields = np.asarray(active["fields"])
    names = tuple(active["field_names"])
    field_index = {name: index for index, name in enumerate(names)}
    opacity = sigmoid(fields[:, field_index["opacity"]])
    half = np.asarray(active["half_length"])
    width = np.asarray(active["width"])
    optical_mass = -np.log(np.maximum(1.0 - opacity, 1.0e-8)) * np.maximum(
        half * width**2, 1.0e-12
    )
    importance = 0.20 + 0.80 * rank01(np.log(np.maximum(optical_mass, 1.0e-20)))

    boundary, visible = multi_view_boundary_score(
        np.asarray(active["centers"]),
        [args.camera_manifest],
        [args.mask_dataset],
        args.boundary_sigma,
    )
    strand, frequency = strand_score(active, args.neighbors)
    density = local_density(np.asarray(active["centers"]), args.neighbors)
    visibility = rank01(visible.astype(np.float64))

    evidence = np.column_stack(
        [
            0.20 + 0.80 * density * (1.0 - boundary),
            0.20 + 0.80 * boundary * (0.50 + 0.50 * visibility),
            0.20 + 0.80 * strand,
        ]
    )
    point_route_utility = probability * evidence * importance[:, None]
    point_route_utility *= (0.50 + 0.50 * confidence[:, None])
    learned_mass_before_contribution = point_route_utility.sum(axis=0)
    contribution_weight = np.ones(3, dtype=np.float64)
    contribution_report = None
    if args.route_contribution_report is not None:
        contribution_report = json.loads(
            args.route_contribution_report.read_text(encoding="utf-8")
        )
        if contribution_report.get("schema") != "train-route-contribution-v1":
            raise ValueError("Unsupported route contribution report schema")
        if contribution_report.get("split") != "train":
            raise ValueError("Route allocation may only use a train-view report")
        if contribution_report.get("heldout_used") is not False:
            raise ValueError("Held-out views must not enter route allocation")
        contribution_weight = np.asarray(
            [
                contribution_report["routes"][name]["allocation_weight"]
                for name in ROUTE_NAMES
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(contribution_weight)) or np.any(
            contribution_weight <= 0.0
        ):
            raise ValueError("Contribution weights must be finite and positive")
        contribution_weight /= contribution_weight.mean()
    learned_mass = learned_mass_before_contribution * contribution_weight
    minimum = max(1, int(round(args.minimum_route_fraction * args.target_active)))
    budget = largest_remainder(learned_mass, args.target_active, minimum)

    base = subset(source, base_ids) if len(base_ids) else None
    native_active = aggregate(active, np.arange(len(active_ids)), args.target_active)
    native_parts = ([base] if base is not None else []) + [native_active]
    native = concatenate(native_parts)
    native_route_id = np.concatenate(
        [
            np.full(len(base_ids), -1, dtype=np.int8),
            np.full(args.target_active, 2, dtype=np.int8),
        ]
    )

    if args.geometry_policy == "shared-coverage":
        # Keep the exact same compressed spatial support as the native control.
        # Route utility is interpolated onto the compressed centers, so route
        # assignment cannot introduce holes merely by changing which source
        # primitives survive compression.
        compressed_centers = np.asarray(native_active["centers"])
        k = min(8, len(active_ids))
        distance, neighbor = cKDTree(np.asarray(active["centers"])).query(
            compressed_centers, k=k, workers=-1
        )
        if k == 1:
            distance = distance[:, None]
            neighbor = neighbor[:, None]
        interpolation_weight = 1.0 / np.maximum(distance, 1.0e-8)
        interpolation_weight /= interpolation_weight.sum(axis=1, keepdims=True)
        compressed_utility = (
            point_route_utility[neighbor] * interpolation_weight[:, :, None]
        ).sum(axis=1)
        available = np.ones(args.target_active, dtype=bool)
        strand_output = select_without_replacement(
            compressed_utility[:, 2], available, int(budget[2])
        )
        available[strand_output] = False
        fin_output = select_without_replacement(
            compressed_utility[:, 1], available, int(budget[1])
        )
        available[fin_output] = False
        active_route_id = np.zeros(args.target_active, dtype=np.int8)
        active_route_id[fin_output] = 1
        active_route_id[strand_output] = 2
        hybrid = native
        hybrid_route_id = np.concatenate(
            [np.full(len(base_ids), -1, dtype=np.int8), active_route_id]
        )
        shell_source_local = np.empty(0, dtype=np.int64)
        fin_local = np.empty(0, dtype=np.int64)
        strand_local = np.empty(0, dtype=np.int64)
    else:
        available = np.ones(len(active_ids), dtype=bool)
        # Preserve sparse/high-frequency strands first, then silhouette fins.
        # Remaining source support is optically merged into shell.
        strand_local = select_without_replacement(
            point_route_utility[:, 2], available, int(budget[2])
        )
        available[strand_local] = False
        fin_local = select_without_replacement(
            point_route_utility[:, 1], available, int(budget[1])
        )
        available[fin_local] = False
        shell_source_local = np.flatnonzero(available)
        if len(shell_source_local) < int(budget[0]):
            raise RuntimeError(
                "Adaptive fin/strand allocation leaves too little shell support"
            )
        shell = aggregate(active, shell_source_local, int(budget[0]))
        fin = subset(active, fin_local)
        strand_part = subset(active, strand_local)
        hybrid_parts = ([base] if base is not None else []) + [
            shell,
            fin,
            strand_part,
        ]
        hybrid = concatenate(hybrid_parts)
        hybrid_route_id = np.concatenate(
            [
                np.full(len(base_ids), -1, dtype=np.int8),
                np.zeros(int(budget[0]), dtype=np.int8),
                np.ones(int(budget[1]), dtype=np.int8),
                np.full(int(budget[2]), 2, dtype=np.int8),
            ]
        )

    hybrid_ply = args.output_root / "hybrid" / "point_cloud.ply"
    native_ply = args.output_root / "native" / "point_cloud.ply"
    write_scaffold(hybrid_ply, hybrid)
    write_scaffold(native_ply, native)
    args.output_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_root / "hybrid" / "routes.npz",
        route_names=ROUTE_NAMES,
        route_id=hybrid_route_id,
        route_logits_init=one_hot_logits(hybrid_route_id),
    )
    np.savez_compressed(
        args.output_root / "native" / "routes.npz",
        route_names=ROUTE_NAMES,
        route_id=native_route_id,
    )
    np.savez_compressed(
        args.output_root / "allocation_scores.npz",
        teacher_probability=probability.astype(np.float32),
        utility=point_route_utility.astype(np.float32),
        boundary=boundary.astype(np.float32),
        strand=strand.astype(np.float32),
        direction_frequency=frequency.astype(np.float32),
        density=density.astype(np.float32),
        visibility=visible,
        active_source_ids=active_ids,
        fin_source_ids=active_ids[fin_local],
        strand_source_ids=active_ids[strand_local],
        shell_source_ids=active_ids[shell_source_local],
        geometry_policy=np.asarray(args.geometry_policy),
        compressed_utility=(
            compressed_utility.astype(np.float32)
            if args.geometry_policy == "shared-coverage"
            else np.empty((0, 3), dtype=np.float32)
        ),
    )

    report = {
        "schema": "learned-route-fixed-budget-compression-v1",
        "input_ply": str(args.input_ply.resolve()),
        "input_sha256": sha256(args.input_ply),
        "teacher_state": str(args.teacher_state.resolve()),
        "teacher_sha256": sha256(args.teacher_state),
        "camera_manifest": str(args.camera_manifest.resolve()),
        "source": {
            "total": source_count,
            "base": int(len(base_ids)),
            "active": int(len(active_ids)),
        },
        "output": {
            "total": int(len(base_ids) + args.target_active),
            "base": int(len(base_ids)),
            "active": int(args.target_active),
            "compression_ratio_active": args.target_active / len(active_ids),
            "budget": {
                name: int(value) for name, value in zip(ROUTE_NAMES, budget, strict=True)
            },
            "budget_fraction": {
                name: float(value / args.target_active)
                for name, value in zip(ROUTE_NAMES, budget, strict=True)
            },
            "geometry_policy": args.geometry_policy,
            "hybrid_native_initial_ply_equal": bool(
                args.geometry_policy == "shared-coverage"
            ),
        },
        "learned_mass": {
            name: float(value)
            for name, value in zip(ROUTE_NAMES, learned_mass, strict=True)
        },
        "learned_mass_before_contribution": {
            name: float(value)
            for name, value in zip(
                ROUTE_NAMES, learned_mass_before_contribution, strict=True
            )
        },
        "train_route_contribution": {
            "report": (
                str(args.route_contribution_report.resolve())
                if args.route_contribution_report is not None
                else None
            ),
            "report_sha256": (
                sha256(args.route_contribution_report)
                if args.route_contribution_report is not None
                else None
            ),
            "normalized_weight": {
                name: float(value)
                for name, value in zip(
                    ROUTE_NAMES, contribution_weight, strict=True
                )
            },
        },
        "rules": {
            "species_label_used": False,
            "heldout_used": False,
            "allocation": (
                "learned probability/evidence mass x positive multi-view "
                "train leave-one-route-out contribution"
                if contribution_report is not None
                else "learned probability/evidence mass"
            ),
            "shell": "learned probability x density x mask interior; optical merge",
            "fin": "learned probability x multi-view training silhouette boundary",
            "strand": "learned probability x length/direction-frequency evidence",
            "coverage": (
                "route-neutral optical aggregation first; routes assigned on the "
                "shared compressed support"
                if args.geometry_policy == "shared-coverage"
                else "route-specialized source selection"
            ),
        },
        "outputs": {
            "hybrid_ply": str(hybrid_ply.resolve()),
            "hybrid_sha256": sha256(hybrid_ply),
            "native_ply": str(native_ply.resolve()),
            "native_sha256": sha256(native_ply),
        },
    }
    (args.output_root / "allocation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
