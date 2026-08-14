#!/usr/bin/env python3
"""Calibrate shell-vs-strand motion ownership without changing rendering."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from dpd3dgs_animal.config import load_config


def _summary(probabilities: torch.Tensor) -> dict[str, float]:
    mean = probabilities.mean(dim=0)
    return {
        "surface": float(mean[0]),
        "shell": float(mean[1]),
        "strand": float(mean[2]),
        "confidence": float(probabilities.max(dim=-1).values.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--blend", type=float, default=1.0)
    args = parser.parse_args()
    if not 0.0 <= float(args.blend) <= 1.0:
        raise ValueError("blend must be in [0, 1]")

    cfg = load_config(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    required = {"route_logits", "residual_trust_logits", "carrier_logits"}
    missing = required - set(state)
    if missing:
        raise ValueError(f"Checkpoint lacks carrier calibration state: {sorted(missing)}")
    temperature = max(float(cfg.fiber_final_temperature), 1e-4)
    base_route = torch.softmax(state["route_logits"] / temperature, dim=-1)
    trust = torch.sigmoid(state["residual_trust_logits"])
    routes = (1.0 - trust) * base_route
    routes[:, 2] += trust[:, 0]
    route_family = routes[:, :2]
    route_family = route_family / route_family.sum(
        dim=-1, keepdim=True
    ).clamp_min(1e-8)

    before = torch.softmax(state["carrier_logits"] / temperature, dim=-1)
    original_fiber_mass = 1.0 - before[:, :1]
    route_fiber_floor = routes[:, :2].sum(dim=-1, keepdim=True)
    fiber_mass = torch.maximum(original_fiber_mass, route_fiber_floor)
    surface_mass = 1.0 - fiber_mass
    target = torch.cat([surface_mass, fiber_mass * route_family], dim=-1)
    calibrated = (
        (1.0 - float(args.blend)) * before + float(args.blend) * target
    )
    calibrated = calibrated / calibrated.sum(dim=-1, keepdim=True)
    calibrated_logits = temperature * torch.log(calibrated.clamp_min(1e-8))
    calibrated_logits -= calibrated_logits.mean(dim=-1, keepdim=True)

    output = copy.deepcopy(checkpoint)
    output["state_dict"]["carrier_logits"] = calibrated_logits.to(
        state["carrier_logits"].dtype
    )
    metadata = dict(output.get("metadata", {}))
    metadata["carrier_calibration"] = {
        "method": "route-structure-floor-family-v2",
        "source_checkpoint": str(Path(args.checkpoint).resolve()),
        "temperature": temperature,
        "blend": float(args.blend),
        "before": _summary(before),
        "after": _summary(calibrated),
        "route_family": {
            "shell": float(route_family[:, 0].mean()),
            "strand": float(route_family[:, 1].mean()),
        },
        "fiber_mass": {
            "before": float(original_fiber_mass.mean()),
            "route_floor": float(route_fiber_floor.mean()),
            "after": float(fiber_mass.mean()),
        },
    }
    output["metadata"] = metadata
    output_path = Path(args.output_checkpoint)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    report_path = output_path.with_suffix(".json")
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(metadata["carrier_calibration"], f, indent=2)
    print(json.dumps(metadata["carrier_calibration"], indent=2))
    print(f"checkpoint={output_path}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
