#!/usr/bin/env python3
"""Frozen-protocol quality, structure, contribution, and efficiency audit."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


METRICS = (
    "foreground_psnr",
    "masked_ssim",
    "masked_lpips",
    "mask_iou",
    "false_negative_fraction",
    "false_positive_fraction",
    "largest_false_negative_component_fraction",
    "largest_false_positive_component_fraction",
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluation(root: Path, split: str, suffix: str = "") -> dict:
    folder = f"evaluation_{split}{suffix}"
    return load(root / folder / "external_evaluation/evaluation.json")


def validate(items: list[dict], split: str) -> None:
    expected_count = 12 if split == "train" else 4
    protocol = items[0]["protocol"]
    names = [row["image"] for row in items[0]["per_frame"]]
    for item in items:
        if item["schema"] != "external-render-evaluation-v1":
            raise ValueError("Unexpected evaluation schema")
        if item["protocol"] != protocol or item["render_size"] != [500, 500]:
            raise ValueError("Protocol or raster mismatch")
        if item["image_count"] != expected_count:
            raise ValueError("Split size mismatch")
        if [row["image"] for row in item["per_frame"]] != names:
            raise ValueError("Frame order mismatch")


def elapsed(path: Path) -> float:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(
        r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*([0-9:.]+)", text
    )
    if not match:
        raise ValueError(f"No elapsed time in {path}")
    fields = [float(value) for value in match.group(1).split(":")]
    if len(fields) == 2:
        return fields[0] * 60 + fields[1]
    return fields[0] * 3600 + fields[1] * 60 + fields[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hairgs-15k", type=Path, required=True)
    parser.add_argument("--native-5k", type=Path, required=True)
    parser.add_argument("--route-aware-5k", type=Path, required=True)
    parser.add_argument("--hybrid-5k", type=Path, required=True)
    parser.add_argument("--hybrid-label", default="anisotropic hybrid 5k")
    parser.add_argument("--hybrid-iteration", type=int, default=20007)
    parser.add_argument("--hybrid-time", type=Path, action="append", default=[])
    parser.add_argument("--ablations", type=Path, required=True)
    parser.add_argument("--reference-audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    methods = {
        "HairGS 15k": args.hairgs_15k,
        "HairGS native 5k": args.native_5k,
        "route-aware 5k": args.route_aware_5k,
        args.hybrid_label: args.hybrid_5k,
    }
    split_data = {
        split: {
            name: evaluation(root, split, "_5k" if name == "route-aware 5k" else "")
            for name, root in methods.items()
        }
        for split in ("train", "test")
    }
    for split in ("train", "test"):
        validate(list(split_data[split].values()), split)

    ablations = {
        route: evaluation(args.ablations / f"drop_{route}", "test")
        for route in ("shell", "fin", "strand")
    }
    validate([split_data["test"][args.hybrid_label], *ablations.values()], "test")
    full_test = split_data["test"][args.hybrid_label]["aggregate"]
    native_test = split_data["test"]["HairGS native 5k"]["aggregate"]

    final_iteration = args.hybrid_iteration
    hybrid_ply = args.hybrid_5k / f"point_cloud/iteration_{final_iteration}/point_cloud.ply"
    hybrid_state = args.hybrid_5k / f"point_cloud/iteration_{final_iteration}/hybrid_state.npz"
    reference_ply = args.hairgs_15k / "point_cloud/iteration_30007/point_cloud.ply"
    native_ply = args.native_5k / "point_cloud/iteration_20007/point_cloud.ply"
    hybrid_report = load(
        args.hybrid_5k / f"point_cloud/iteration_{final_iteration}/hybrid_report.json"
    )

    quality_gates = {
        "native5_reference_psnr_not_worse_0p10db": full_test["foreground_psnr"]
        >= native_test["foreground_psnr"] - 0.10,
        "native5_reference_iou_not_worse_0p005": full_test["mask_iou"]
        >= native_test["mask_iou"] - 0.005,
        "native5_reference_max_fn_not_worse_0p005": full_test[
            "largest_false_negative_component_fraction"
        ]
        <= native_test["largest_false_negative_component_fraction"] + 0.005,
    }
    route_contribution = {
        route: {
            "delta_foreground_psnr_after_drop": float(
                ablation["aggregate"]["foreground_psnr"]
                - full_test["foreground_psnr"]
            ),
            "delta_mask_iou_after_drop": float(
                ablation["aggregate"]["mask_iou"] - full_test["mask_iou"]
            ),
            "positive": bool(
                ablation["aggregate"]["foreground_psnr"]
                < full_test["foreground_psnr"]
                and ablation["aggregate"]["mask_iou"] < full_test["mask_iou"]
            ),
        }
        for route, ablation in ablations.items()
    }
    hybrid_time_paths = args.hybrid_time or [args.hybrid_5k / "audit/train_time.txt"]
    hybrid_seconds = sum(elapsed(path) for path in hybrid_time_paths)
    reference_seconds = None
    if args.reference_audit is not None:
        reference_seconds = load(args.reference_audit)["efficiency"]["baseline"][
            "stage3_checkpoint_span_seconds"
        ]
    report = {
        "schema": "hairgs-hybrid-frozen-audit-v1",
        "status": "pass" if all(quality_gates.values()) else "fail",
        "protocol": split_data["test"][args.hybrid_label]["protocol"],
        "metrics": {
            split: {
                name: {metric: data["aggregate"][metric] for metric in METRICS}
                for name, data in split_data[split].items()
            }
            for split in ("train", "test")
        },
        "delta_hybrid_minus_native_5k": {
            metric: float(full_test[metric] - native_test[metric]) for metric in METRICS
        },
        "quality_gates": quality_gates,
        "route_contribution": route_contribution,
        "structure": hybrid_report,
        "efficiency": {
            "hairgs_15k_final_bytes": reference_ply.stat().st_size,
            "native_5k_final_bytes": native_ply.stat().st_size,
            "hybrid_5k_final_bytes": hybrid_ply.stat().st_size + hybrid_state.stat().st_size,
            "hybrid_reduction_vs_15k": 1.0
            - (hybrid_ply.stat().st_size + hybrid_state.stat().st_size)
            / reference_ply.stat().st_size,
            "hybrid_train_seconds": hybrid_seconds,
            "native_5k_train_seconds": elapsed(args.native_5k / "audit/native4k_time.txt")
            + elapsed(args.native_5k / "audit/native1k_time.txt"),
            "hairgs_15k_stage3_seconds": reference_seconds,
            "hybrid_train_time_reduction_vs_15k": (
                1.0 - hybrid_seconds / reference_seconds
                if reference_seconds is not None
                else None
            ),
        },
        "claim_limit": (
            "This experiment validates a route-conditioned covariance extension on fixed "
            "HairGS topology. It does not yet validate independent shell/fin topology or topology birth."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
