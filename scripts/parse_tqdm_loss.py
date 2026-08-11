#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


LOSS_RE = re.compile(r"\|\s*(\d+)/(\d+).*?Loss=([0-9.eE+-]+)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract tqdm EMA loss from a redirected training log.")
    parser.add_argument("log", nargs="+")
    parser.add_argument(
        "--step-offsets",
        default=None,
        help="Comma-separated offset for each log segment (default: all zero).",
    )
    parser.add_argument(
        "--segment-max-steps",
        default=None,
        help="Comma-separated inclusive global max step per segment; use 'none' for no limit.",
    )
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-png", default=None)
    args = parser.parse_args()

    paths = [Path(value) for value in args.log]
    offsets = (
        [int(value) for value in args.step_offsets.split(",")]
        if args.step_offsets
        else [0] * len(paths)
    )
    max_steps = (
        [None if value.lower() == "none" else int(value) for value in args.segment_max_steps.split(",")]
        if args.segment_max_steps
        else [None] * len(paths)
    )
    if len(offsets) != len(paths) or len(max_steps) != len(paths):
        raise ValueError("log, step-offset, and segment-max-step counts must match")
    by_step: dict[int, float] = {}
    total = None
    for path, offset, max_step in zip(paths, offsets, max_steps, strict=True):
        text = path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
        for match in LOSS_RE.finditer(text):
            step, total_value, loss = match.groups()
            global_step = int(step) + offset
            if max_step is None or global_step <= max_step:
                by_step[global_step] = float(loss)
            total = max(total or 0, int(total_value) + offset)
    values = [{"step": step, "ema_loss": loss} for step, loss in sorted(by_step.items())]
    payload = {
        "logs": [str(path) for path in paths],
        "total_iterations": total,
        "samples": values,
        "summary": {
            "first_step": values[0]["step"] if values else None,
            "first_loss": values[0]["ema_loss"] if values else None,
            "last_step": values[-1]["step"] if values else None,
            "last_loss": values[-1]["ema_loss"] if values else None,
            "minimum": min((item["ema_loss"] for item in values), default=None),
        },
    }
    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)
    if args.out_png and values:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(10, 4.8))
        axis.plot([item["step"] for item in values], [item["ema_loss"] for item in values], linewidth=1.1)
        axis.set_xlabel("Stage-III iteration")
        axis.set_ylabel("EMA training loss")
        axis.set_title("HairGS strand-refinement loss")
        axis.grid(alpha=0.25)
        figure.tight_layout()
        out_png = Path(args.out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(out_png, dpi=170)
        plt.close(figure)
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
