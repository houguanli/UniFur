#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export TensorBoard scalars to JSON and a compact training plot."
    )
    parser.add_argument("event_file")
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-png", default=None)
    args = parser.parse_args()

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    event_file = Path(args.event_file)
    accumulator = EventAccumulator(
        str(event_file), size_guidance={"scalars": 0, "images": 0}
    )
    accumulator.Reload()
    payload = {
        "event_file": str(event_file),
        "scalars": {
            tag: [
                {"step": item.step, "value": item.value, "wall_time": item.wall_time}
                for item in accumulator.Scalars(tag)
            ]
            for tag in accumulator.Tags().get("scalars", [])
        },
    }
    summary = {
        tag: {
            "first_step": values[0]["step"],
            "first_value": values[0]["value"],
            "last_step": values[-1]["step"],
            "last_value": values[-1]["value"],
            "minimum": min(value["value"] for value in values),
        }
        for tag, values in payload["scalars"].items()
        if values
    }
    payload["summary"] = summary
    print(json.dumps(summary, indent=2))

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)
    if args.out_png:
        _plot(payload["scalars"], Path(args.out_png))
    return 0


def _plot(scalars: dict[str, list[dict[str, float]]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for tag in ("train/loss", "train/l1", "train/dssim"):
        if tag in scalars:
            axes[0].plot(
                [item["step"] for item in scalars[tag]],
                [item["value"] for item in scalars[tag]],
                label=tag.removeprefix("train/"),
                alpha=0.85,
            )
    axes[0].set_ylabel("photometric loss")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    for tag in ("train/orientation", "train/mask"):
        if tag in scalars:
            axes[1].plot(
                [item["step"] for item in scalars[tag]],
                [item["value"] for item in scalars[tag]],
                label=tag.removeprefix("train/"),
            )
    axes[1].set_ylabel("structure losses")
    axes[1].legend()
    axes[1].grid(alpha=0.2)

    if "general/total_gaussians" in scalars:
        values = scalars["general/total_gaussians"]
        axes[2].plot(
            [item["step"] for item in values],
            [item["value"] for item in values],
            label="total Gaussians",
        )
    axes[2].set_ylabel("primitive count")
    axes[2].set_xlabel("iteration")
    axes[2].legend()
    axes[2].grid(alpha=0.2)
    figure.suptitle("HairGS TensorBoard diagnostics")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(main())
