#!/usr/bin/env python3
"""Build protocol-safe Fur/Hair comparison tables from strict evaluations."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any


METRICS = (
    ("FG PSNR", "foreground_psnr", True),
    ("Masked PSNR", "masked_full_psnr", True),
    ("Masked SSIM", "masked_ssim", True),
    ("Masked LPIPS", "masked_lpips", False),
    ("Mask IoU", "mask_iou", True),
    ("BG opacity", "background_opacity_mean", False),
)


def _load_evaluation(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "external-render-evaluation-v1":
        raise ValueError(f"{path} was not produced by the strict external evaluator")
    required = {key for _, key, _ in METRICS}
    missing = required - set(payload.get("aggregate", {}))
    if missing:
        raise ValueError(f"{path} is missing aggregate metrics: {sorted(missing)}")
    return payload


def _protocol_identity(payload: dict[str, Any]) -> tuple[Any, ...]:
    per_frame = payload.get("per_frame", [])
    # Exporters may preserve a model's native camera order; comparison only
    # requires the held-out image/view set to be identical.
    views = tuple(
        sorted((item.get("image"), item.get("view_index")) for item in per_frame)
    )
    return (
        payload.get("protocol"),
        tuple(payload.get("render_size", [])),
        payload.get("image_count"),
        str(Path(payload.get("ground_truth_dir", "")).resolve()),
        views,
    )


def _validate_group(group: str, rows: list[dict[str, Any]]) -> None:
    reference = _protocol_identity(rows[0]["evaluation"])
    for row in rows[1:]:
        candidate = _protocol_identity(row["evaluation"])
        if candidate != reference:
            raise ValueError(
                f"Group {group!r} mixes incompatible evaluation protocols: "
                f"{rows[0]['method']!r} != {row['method']!r}"
            )


def _markdown(groups: OrderedDict[str, list[dict[str, Any]]]) -> str:
    chunks: list[str] = []
    for group, rows in groups.items():
        evaluation = rows[0]["evaluation"]
        size = "x".join(str(value) for value in evaluation["render_size"])
        chunks.extend(
            [
                f"## {group}",
                "",
                f"Protocol: `{evaluation['protocol']}`; "
                f"{evaluation['image_count']} held-out views; {size}.",
                "",
                "| Method | "
                + " | ".join(
                    label + (" ↑" if higher else " ↓")
                    for label, _, higher in METRICS
                )
                + " |",
                "|---|" + "---:|" * len(METRICS),
            ]
        )
        for row in rows:
            aggregate = row["evaluation"]["aggregate"]
            values = [f"{float(aggregate[key]):.4f}" for _, key, _ in METRICS]
            chunks.append(f"| {row['method']} | " + " | ".join(values) + " |")
        chunks.append("")
    return "\n".join(chunks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--entry",
        nargs=3,
        action="append",
        metavar=("GROUP", "METHOD", "EVALUATION_JSON"),
        required=True,
    )
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for group, method, path_text in args.entry:
        path = Path(path_text).resolve()
        evaluation = _load_evaluation(path)
        groups.setdefault(group, []).append(
            {"method": method, "evaluation_path": str(path), "evaluation": evaluation}
        )
    for group, rows in groups.items():
        _validate_group(group, rows)

    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(_markdown(groups), encoding="utf-8")
    serializable = {
        group: [
            {
                "method": row["method"],
                "evaluation_path": row["evaluation_path"],
                "protocol": row["evaluation"]["protocol"],
                "render_size": row["evaluation"]["render_size"],
                "image_count": row["evaluation"]["image_count"],
                "aggregate": row["evaluation"]["aggregate"],
            }
            for row in rows
        ]
        for group, rows in groups.items()
    }
    args.output_json.write_text(
        json.dumps(serializable, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
