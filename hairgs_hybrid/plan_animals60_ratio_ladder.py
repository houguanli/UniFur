#!/usr/bin/env python3
"""Plan per-case Shell/Fin/Strand budgets from an unrestricted HairGS teacher.

Compression is defined on renderable HairGS segments (PLY ``edge`` elements),
not on the two endpoint vertices stored for every segment.  A compression of
75 percent therefore retains exactly 25 percent of the unrestricted teacher.
The neutral 1/2/4 topology bank uses ``round(3 * target / 7)`` groups so an
equal 1/3 Shell/Fin/Strand prior has an expected active count near the target.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from plyfile import PlyData


DEFAULT_COMPRESSIONS = (0.75, 0.875, 0.9375)


def ratio_budget(source_segments: int, compression: float) -> int:
    if source_segments < 4:
        raise ValueError("The unrestricted teacher must contain at least four segments")
    if not 0.0 < compression < 1.0:
        raise ValueError("Compression must be strictly between zero and one")
    return max(1, int(round(source_segments * (1.0 - compression))))


def neutral_group_count(source_segments: int, target_segments: int) -> int:
    """Choose a feasible group count centered on the neutral 1/2/4 prior."""

    minimum = int(math.ceil(target_segments / 4.0))
    maximum = min(target_segments, source_segments // 4)
    if minimum > maximum:
        raise ValueError(
            f"Cannot expose a 1/2/4 bank for source={source_segments}, "
            f"target={target_segments}"
        )
    desired = int(round(3.0 * target_segments / 7.0))
    return max(minimum, min(desired, maximum))


def final_dense_ply(case_root: Path) -> Path:
    point_root = case_root / "hairgs_unrestricted/model/point_cloud"
    candidates = []
    for path in point_root.glob("iteration_*/point_cloud.ply"):
        try:
            iteration = int(path.parent.name.removeprefix("iteration_"))
        except ValueError:
            continue
        candidates.append((iteration, path))
    if not candidates:
        raise FileNotFoundError(f"No unrestricted checkpoint under {point_root}")
    return max(candidates)[1]


def edge_count(path: Path) -> int:
    ply = PlyData.read(path)
    if "edge" not in ply:
        raise ValueError(f"HairGS checkpoint has no edge segments: {path}")
    return int(ply["edge"].count)


def build_plan(
    benchmark_root: Path,
    compressions: tuple[float, ...],
    require_complete: bool = True,
) -> dict:
    cases_root = benchmark_root / "cases"
    rows = []
    for case in sorted(path for path in cases_root.iterdir() if path.is_dir()):
        if require_complete and not (case / "COMPLETE").is_file():
            continue
        dense_ply = final_dense_ply(case)
        source = edge_count(dense_ply)
        for compression in compressions:
            target = ratio_budget(source, compression)
            groups = neutral_group_count(source, target)
            rows.append(
                {
                    "case": case.name,
                    "unrestricted_segments": source,
                    "compression": compression,
                    "retention": 1.0 - compression,
                    "target_active_segments": target,
                    "actual_compression": 1.0 - target / source,
                    "group_count": groups,
                    "candidate_segments": 7 * groups,
                    "neutral_expected_active_segments": 7.0 * groups / 3.0,
                    "mean_source_segments_per_group": source / groups,
                    "dense_teacher_ply": str(dense_ply.resolve()),
                    "run_tag": (
                        f"c{compression * 100:g}_r{(1.0 - compression) * 100:g}"
                        f"_b{target}"
                    ),
                }
            )
    return {
        "schema": "unifur-animals60-ratio-ladder-plan-v1",
        "budget_unit": "renderable HairGS edge segments",
        "endpoint_vertices_per_segment": 2,
        "compressions": list(compressions),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument(
        "--compressions",
        type=float,
        nargs="+",
        default=DEFAULT_COMPRESSIONS,
        help="Fractions removed from the unrestricted teacher",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    compressions = tuple(sorted(set(args.compressions)))
    plan = build_plan(args.benchmark_root, compressions, not args.allow_incomplete)
    output = args.output or args.benchmark_root / "audit/ratio_ladder_plan.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    fields = [key for key in plan["rows"][0] if key != "dense_teacher_ply"] if plan["rows"] else []
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        if fields:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(plan["rows"])
    print(json.dumps({"cases": len({row['case'] for row in plan['rows']}), "rows": len(plan["rows"]), "json": str(output.resolve()), "csv": str(csv_path.resolve())}, indent=2))


if __name__ == "__main__":
    main()
