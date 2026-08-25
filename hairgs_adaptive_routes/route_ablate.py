#!/usr/bin/env python3
"""Create a HairGS PLY with one adaptive hair route disabled."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


ROUTE_NAMES = ("shell", "fin", "strand")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--drop", choices=ROUTE_NAMES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    ply = PlyData.read(args.input)
    route_id = np.load(args.routes)["route_id"].astype(np.int64)
    segment_element = ply["segment"]
    if route_id.size != segment_element.count:
        raise ValueError(
            f"route/segment mismatch: {route_id.size} vs {segment_element.count}"
        )
    target = ROUTE_NAMES.index(args.drop)
    removed = route_id == target
    segment = segment_element.data.copy()
    before = 1.0 / (1.0 + np.exp(-np.clip(segment["opacity"], -30.0, 30.0)))
    segment["opacity"][removed] = -30.0

    elements = []
    for element in ply.elements:
        if element.name == "segment":
            elements.append(PlyElement.describe(segment, "segment"))
        else:
            elements.append(PlyElement.describe(element.data, element.name))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    PlyData(elements, text=False).write(args.output)

    report = {
        "status": "complete",
        "dropped_route": args.drop,
        "removed_segments": int(np.count_nonzero(removed)),
        "removed_effective_opacity_sum": float(before[removed].sum()),
        "total_effective_opacity_sum": float(before.sum()),
        "removed_opacity_mass_fraction": float(
            before[removed].sum() / max(float(before.sum()), 1e-12)
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
