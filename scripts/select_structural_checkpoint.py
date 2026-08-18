#!/usr/bin/env python3
"""Select a UniFur checkpoint without consulting 3D hair ground truth.

The selector maximizes deployed structured coverage only after the analytic
strand target passes the configured multi-view hull checks.  A selected model
still has to pass the residual-teacher held-out calibration before it can be
reported; old runs that predate persistent teacher logging are marked as
requiring that external verification.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


_CHECKPOINT = re.compile(r"step_(\d+)\.pt$")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-geometry-blend", type=float, default=0.95)
    parser.add_argument("--min-kept-fraction", type=float, default=0.90)
    parser.add_argument("--min-support-fraction", type=float, default=0.25)
    parser.add_argument("--teacher-margin", type=float, default=0.0)
    parser.add_argument("--require-risk-calibration", action="store_true")
    parser.add_argument("--coverage-retention", type=float, default=0.95)
    return parser.parse_args()


def _candidate(row: dict, args: argparse.Namespace) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if row.get("phase") == "gaussian_scaffold":
        reasons.append("gaussian_scaffold")
    if float(row.get("geometry_blend", 0.0)) < args.min_geometry_blend:
        reasons.append("geometry_blend")
    visual = row.get("visual_hull_report") or {}
    if float(visual.get("kept_fraction", 0.0)) < args.min_kept_fraction:
        reasons.append("visual_hull_kept_fraction")
    if float(visual.get("mean_support_fraction", 0.0)) < args.min_support_fraction:
        reasons.append("visual_hull_support_fraction")
    if float(row.get("strand_effective_coverage", 0.0)) <= 0.0:
        reasons.append("zero_structural_coverage")
    if args.require_risk_calibration and not row.get("latest_calibration_frames"):
        reasons.append("missing_fully_deployed_risk_calibration")
    student = row.get("teacher_calibration_student")
    teacher = row.get("teacher_calibration_residual")
    if student is not None and teacher is not None:
        if float(student) > float(teacher) + args.teacher_margin:
            reasons.append("teacher_nonregression")
    return not reasons, reasons


def main() -> int:
    args = _arguments()
    metrics_path = Path(args.metrics)
    checkpoint_dir = Path(args.checkpoint_dir)
    rows = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows_by_step = {int(row["step"]) + 1: row for row in rows}
    candidates: list[dict] = []
    rejected: list[dict] = []
    for checkpoint in sorted(checkpoint_dir.glob("step_*.pt")):
        match = _CHECKPOINT.match(checkpoint.name)
        if match is None:
            continue
        step = int(match.group(1))
        row = rows_by_step.get(step)
        if row is None:
            rejected.append(
                {"step": step, "checkpoint": str(checkpoint), "reasons": ["missing_metric"]}
            )
            continue
        accepted, reasons = _candidate(row, args)
        record = {
            "step": step,
            "checkpoint": str(checkpoint),
            "strand_effective_coverage": float(row["strand_effective_coverage"]),
            "geometry_blend": float(row.get("geometry_blend", 0.0)),
            "risk_calibrated": bool(row.get("latest_calibration_frames")),
            "route_risk_l1": (
                sum(
                    abs(float((row.get("routes") or {}).get(name, 0.0))
                        - float((row.get("risk_target") or {}).get(name, 0.0)))
                    for name in ("shell", "strand", "residual")
                )
                if row.get("risk_target")
                else None
            ),
            "visual_hull_kept_fraction": float(
                (row.get("visual_hull_report") or {}).get("kept_fraction", 0.0)
            ),
            "visual_hull_support_fraction": float(
                (row.get("visual_hull_report") or {}).get(
                    "mean_support_fraction", 0.0
                )
            ),
            "teacher_calibration_student": row.get("teacher_calibration_student"),
            "teacher_calibration_residual": row.get("teacher_calibration_residual"),
            "teacher_verification_required": row.get(
                "teacher_calibration_student"
            )
            is None,
        }
        if accepted:
            candidates.append(record)
        else:
            record["reasons"] = reasons
            rejected.append(record)
    if not candidates:
        raise RuntimeError("No checkpoint passed the structural selection gates")
    if not 0.0 < args.coverage_retention <= 1.0:
        raise ValueError("coverage-retention must be in (0, 1]")
    maximum_coverage = max(
        item["strand_effective_coverage"] for item in candidates
    )
    retained = [
        item
        for item in candidates
        if item["strand_effective_coverage"]
        >= args.coverage_retention * maximum_coverage
    ]
    selected = min(
        retained,
        key=lambda item: (
            item["route_risk_l1"]
            if item["route_risk_l1"] is not None
            else float("inf"),
            -item["visual_hull_support_fraction"],
            -item["step"],
        ),
    )
    payload = {
        "schema": "unifur-structural-checkpoint-selection-v1",
        "uses_3d_ground_truth": False,
        "criterion": (
            "minimum route-risk mismatch while retaining near-maximum "
            "deployed strand coverage"
        ),
        "thresholds": {
            "min_geometry_blend": args.min_geometry_blend,
            "min_kept_fraction": args.min_kept_fraction,
            "min_support_fraction": args.min_support_fraction,
            "teacher_margin": args.teacher_margin,
            "require_risk_calibration": args.require_risk_calibration,
            "coverage_retention": args.coverage_retention,
        },
        "selected": selected,
        "candidates": candidates,
        "rejected": rejected,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(selected, indent=2))
    print(f"selection={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
