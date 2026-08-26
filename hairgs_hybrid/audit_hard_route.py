#!/usr/bin/env python3
"""Audit hard deployment against soft routing and a same-budget HairGS control."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from hairgs_hybrid.audit_compression_sweep import delta, method


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--hard", type=Path, required=True)
    parser.add_argument("--soft", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--hard-initial-ply", type=Path, required=True)
    parser.add_argument("--soft-initial-ply", type=Path, required=True)
    parser.add_argument("--control-initial-ply", type=Path, required=True)
    parser.add_argument("--hard-routes", type=Path, required=True)
    parser.add_argument("--soft-routes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = {
        "hard_fused_compact": method(args.hard),
        "soft_unifur": method(args.soft),
        "same_budget_hairgs": method(args.control),
    }
    protocol_keys = {(row["protocol"], row["image_count"]) for row in rows.values()}
    if len(protocol_keys) != 1:
        raise ValueError(f"Evaluation protocol mismatch: {protocol_keys}")

    ply_digests = {
        "hard": _digest(args.hard_initial_ply),
        "soft": _digest(args.soft_initial_ply),
        "control": _digest(args.control_initial_ply),
    }
    route_digests = {
        "hard": _digest(args.hard_routes),
        "soft": _digest(args.soft_routes),
    }
    if len(set(ply_digests.values())) != 1:
        raise ValueError(f"Initial PLY mismatch: {ply_digests}")
    if len(set(route_digests.values())) != 1:
        raise ValueError(f"Hard/soft route allocation mismatch: {route_digests}")

    hard = rows["hard_fused_compact"]
    report = {
        "schema": "hairgs-hard-route-audit-v1",
        "status": "complete",
        "case": args.case,
        "protocol": hard["protocol"],
        "image_count": hard["image_count"],
        "same_initial_ply": True,
        "same_hard_soft_route_allocation": True,
        "initial_ply_sha256": ply_digests["hard"],
        "route_sidecar_sha256": route_digests["hard"],
        "methods": rows,
        "deltas": {
            "hard_minus_soft": delta(hard, rows["soft_unifur"]),
            "hard_minus_hairgs": delta(hard, rows["same_budget_hairgs"]),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
