from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROUTES = ("shell", "strand", "residual")


def _read(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _load_run(path: Path) -> dict:
    soft = _read(path / "validation_soft" / "evaluation.json")
    hard = _read(path / "validation_hard" / "evaluation.json")
    audit = _read(path / "route_audit" / "route_audit.json")
    train = _read(path / "unified_fiber_report.json")
    return {"soft": soft, "hard": hard, "audit": audit, "train": train}


def _summary(run: dict) -> dict:
    soft = run["soft"]["aggregate"]
    hard = run["hard"]["aggregate"]
    audit = run["audit"]
    return {
        "soft_psnr": soft["foreground_psnr"],
        "soft_l1": soft["foreground_l1"],
        "soft_iou": soft["mask_iou"],
        "soft_f1": soft["mask_f1"],
        "hard_psnr": hard["foreground_psnr"],
        "hard_iou": hard["mask_iou"],
        "hard_psnr_drop": audit["hard_minus_soft_gap"]["psnr_drop"],
        "probability_contribution_tv": audit[
            "probability_contribution_total_variation"
        ],
        "neighbor_agreement": audit["spatial_coherence"][
            "same_hard_route_fraction"
        ],
        "neighbor_excess": audit["spatial_coherence"]["excess_over_random"],
        "neighbor_probability_l1": audit["spatial_coherence"][
            "neighbor_probability_l1_mean"
        ],
        "normalized_entropy": audit["confidence"]["normalized_entropy_mean"],
        "max_probability": audit["confidence"]["max_probability_mean"],
        "route_mass": audit["route_probability_mean"],
        "route_impact": {
            route: audit["route_contribution"][route][
                "normalized_positive_impact"
            ]
            for route in ROUTES
        },
    }


def _plot(baseline: dict, candidate: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    labels = ["baseline-28", "calibrated soft"]
    colors = ["#667085", "#0ea5a8"]

    axes[0, 0].bar(labels, [baseline["soft_psnr"], candidate["soft_psnr"]], color=colors)
    axes[0, 0].set_title("Held-out foreground PSNR")
    axes[0, 0].set_ylabel("dB (higher is better)")
    axes[0, 0].set_ylim(13.5, 15.0)

    x = np.arange(2)
    width = 0.36
    axes[0, 1].bar(
        x - width / 2,
        [baseline["probability_contribution_tv"], baseline["neighbor_probability_l1"]],
        width,
        label="baseline-28",
        color=colors[0],
    )
    axes[0, 1].bar(
        x + width / 2,
        [candidate["probability_contribution_tv"], candidate["neighbor_probability_l1"]],
        width,
        label="calibrated soft",
        color=colors[1],
    )
    axes[0, 1].set_xticks(x, ["prob.-impact TV", "neighbor prob. L1"])
    axes[0, 1].set_title("Routing mismatch (lower is better)")
    axes[0, 1].legend()

    axes[1, 0].bar(
        labels,
        [baseline["neighbor_agreement"], candidate["neighbor_agreement"]],
        color=colors,
    )
    axes[1, 0].set_title("8-NN hard-route agreement")
    axes[1, 0].set_ylim(0.0, 0.8)
    axes[1, 0].set_ylabel("fraction (higher is better)")

    route_x = np.arange(len(ROUTES))
    axes[1, 1].bar(
        route_x - width / 2,
        [candidate["route_mass"][route] for route in ROUTES],
        width,
        label="soft mass",
        color="#2563eb",
    )
    axes[1, 1].bar(
        route_x + width / 2,
        [candidate["route_impact"][route] for route in ROUTES],
        width,
        label="held-out LOO impact",
        color="#16a34a",
    )
    axes[1, 1].set_xticks(route_x, ROUTES)
    axes[1, 1].set_title("Candidate probability vs contribution")
    axes[1, 1].set_ylim(0.0, 0.7)
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Cat route-training A/B: 1k sources, 28 fit + 4 calibration, frames 32-39 test")
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    baseline = _summary(_load_run(args.baseline))
    candidate = _summary(_load_run(args.candidate))
    delta = {
        key: candidate[key] - baseline[key]
        for key in (
            "soft_psnr",
            "soft_l1",
            "soft_iou",
            "soft_f1",
            "hard_psnr_drop",
            "probability_contribution_tv",
            "neighbor_agreement",
            "neighbor_excess",
            "neighbor_probability_l1",
            "normalized_entropy",
            "max_probability",
        )
    }
    payload = {"baseline": baseline, "candidate": candidate, "delta": delta}
    with open(args.out_dir / "comparison.json", "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    _plot(baseline, candidate, args.out_dir / "comparison.png")
    tv_reduction = 1.0 - (
        candidate["probability_contribution_tv"]
        / baseline["probability_contribution_tv"]
    )
    neighbor_reduction = 1.0 - (
        candidate["neighbor_probability_l1"]
        / baseline["neighbor_probability_l1"]
    )
    markdown = f"""# Contribution-calibrated route A/B

Protocol: 1,000 source Gaussians, 256x144 HairGS rasterization, 1,200 steps,
frames 0--27 for fitting, 28--31 reserved for calibration, and 32--39 for final
held-out evaluation.

| Metric | Matched baseline | Calibrated soft | Delta |
|---|---:|---:|---:|
| Held-out PSNR | {baseline['soft_psnr']:.4f} | {candidate['soft_psnr']:.4f} | {delta['soft_psnr']:+.4f} dB |
| Mask IoU | {baseline['soft_iou']:.4f} | {candidate['soft_iou']:.4f} | {delta['soft_iou']:+.4f} |
| Probability-impact TV | {baseline['probability_contribution_tv']:.4f} | {candidate['probability_contribution_tv']:.4f} | {tv_reduction:.1%} lower |
| 8-NN hard agreement | {baseline['neighbor_agreement']:.4f} | {candidate['neighbor_agreement']:.4f} | {delta['neighbor_agreement']:+.4f} |
| Neighbor probability L1 | {baseline['neighbor_probability_l1']:.4f} | {candidate['neighbor_probability_l1']:.4f} | {neighbor_reduction:.1%} lower |
| Hard-vs-soft PSNR drop | {baseline['hard_psnr_drop']:.4f} | {candidate['hard_psnr_drop']:.4f} | {delta['hard_psnr_drop']:+.4f} dB |

Conclusion: calibration and surface regularization materially improve route
coherence and probability/contribution agreement for a small held-out rendering
cost. Hard deployment becomes worse, which supports retaining the soft mixture.
"""
    with open(args.out_dir / "comparison.md", "w", encoding="utf-8") as file:
        file.write(markdown)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
