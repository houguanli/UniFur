#!/usr/bin/env python3
"""Prove that head/body native HairGS state is unchanged by hair-only training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData


def digest(arrays: list[np.ndarray]) -> str:
    value = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        value.update(str(contiguous.dtype).encode("ascii"))
        value.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        value.update(contiguous.tobytes())
    return value.hexdigest()


def native_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    data = PlyData.read(path)
    endpoints = np.stack(
        [np.asarray(data["vertex"][axis]) for axis in ("x", "y", "z")], axis=1
    ).astype(np.float32)
    pairs = np.stack(
        [np.asarray(data["edge"][name]) for name in ("vertex1", "vertex2")], axis=1
    ).astype(np.int64)
    segment = data["segment"]
    fields = {
        prop.name: np.asarray(segment[prop.name]).copy()
        for prop in segment.properties
    }
    return endpoints, pairs, fields


def max_abs(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0:
        return 0.0
    return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-ply", type=Path, required=True)
    parser.add_argument("--final-ply", type=Path, required=True)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    initial_endpoints, initial_pairs, initial_fields = native_arrays(args.initial_ply)
    final_endpoints, final_pairs, final_fields = native_arrays(args.final_ply)
    route_ids = np.asarray(np.load(args.routes)["route_id"], dtype=np.int64)
    if not np.array_equal(initial_pairs, final_pairs):
        raise ValueError("Topology changed in a topology-frozen experiment")
    if route_ids.shape != (initial_pairs.shape[0],):
        raise ValueError("Route sidecar does not match PLY topology")
    hair_segments = route_ids >= 0
    base_segments = ~hair_segments
    hair_endpoints = np.zeros(initial_endpoints.shape[0], dtype=bool)
    base_endpoints = np.zeros_like(hair_endpoints)
    hair_endpoints[initial_pairs[hair_segments].reshape(-1)] = True
    base_endpoints[initial_pairs[base_segments].reshape(-1)] = True
    shared = hair_endpoints & base_endpoints
    if np.any(shared):
        raise ValueError(f"Hair/base share {int(shared.sum())} endpoints")

    base_arrays_initial = [initial_endpoints[base_endpoints]]
    base_arrays_final = [final_endpoints[base_endpoints]]
    field_errors = {}
    hair_field_changes = {}
    for name in sorted(initial_fields):
        if name not in final_fields:
            raise ValueError(f"Final PLY lacks segment field {name}")
        field_errors[name] = max_abs(
            initial_fields[name][base_segments], final_fields[name][base_segments]
        )
        hair_field_changes[name] = max_abs(
            initial_fields[name][hair_segments], final_fields[name][hair_segments]
        )
        base_arrays_initial.append(initial_fields[name][base_segments])
        base_arrays_final.append(final_fields[name][base_segments])

    report = {
        "schema": "hairgs-frozen-base-audit-v1",
        "initial_ply": str(args.initial_ply.resolve()),
        "final_ply": str(args.final_ply.resolve()),
        "segment_count": int(route_ids.size),
        "hair_segment_count": int(hair_segments.sum()),
        "base_segment_count": int(base_segments.sum()),
        "hair_endpoint_count": int(hair_endpoints.sum()),
        "base_endpoint_count": int(base_endpoints.sum()),
        "shared_endpoint_count": int(shared.sum()),
        "base_endpoint_max_abs_error": max_abs(
            initial_endpoints[base_endpoints], final_endpoints[base_endpoints]
        ),
        "base_segment_field_max_abs_error": field_errors,
        "base_initial_digest": digest(base_arrays_initial),
        "base_final_digest": digest(base_arrays_final),
        "base_bitwise_unchanged": all(value == 0.0 for value in field_errors.values())
        and max_abs(initial_endpoints[base_endpoints], final_endpoints[base_endpoints])
        == 0.0,
        "hair_endpoint_max_abs_change": max_abs(
            initial_endpoints[hair_endpoints], final_endpoints[hair_endpoints]
        ),
        "hair_segment_field_max_abs_change": hair_field_changes,
    }
    if not report["base_bitwise_unchanged"]:
        raise RuntimeError("Head/body state changed during hair-only training")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
