#!/usr/bin/env python3
"""Summarize physical scale and anisotropy outliers in a Gaussian PLY."""

from __future__ import annotations

import argparse

import numpy as np

from dpd3dgs_animal.gaussian import load_gaussian_ply


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ply")
    args = parser.parse_args()
    gaussian = load_gaussian_ply(args.ply)
    if gaussian.scaling is None or gaussian.opacity is None:
        raise ValueError("PLY must contain 3DGS scaling and opacity properties")
    scales = gaussian.scaling
    foreground = (
        gaussian.foreground_probability
        if gaussian.foreground_probability is not None
        else np.ones((len(scales),), dtype=np.float32)
    )
    opacity = gaussian.opacity
    maximum = scales.max(axis=1)
    minimum = scales.min(axis=1)
    percentiles = [0, 1, 5, 25, 50, 75, 90, 95, 97, 98, 99, 99.5, 99.9, 100]
    print(f"points={len(scales)}")
    print("scale_per_axis=")
    print(np.percentile(scales, percentiles, axis=0))
    print("max_scale=")
    print(np.percentile(maximum, percentiles))
    print("anisotropy=")
    print(np.percentile(maximum / np.maximum(minimum, 1e-12), percentiles))
    for threshold in (0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1):
        selected = maximum > threshold
        print(
            f"max>{threshold:g}: count={int(selected.sum())} "
            f"foreground={float(foreground[selected].mean()) if selected.any() else 0:.4f} "
            f"opacity={float(opacity[selected].mean()) if selected.any() else 0:.4f}"
        )


if __name__ == "__main__":
    main()
