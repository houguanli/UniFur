#!/usr/bin/env python3
"""Render a persisted hybrid model through the native HairGS rasterizer."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import torch

import scene as scene_module
from arguments import GeneralParams, ModelParams, OptimizationParams, get_combined_args
from utils import safe_state

from hairgs_hybrid.hybrid_model import HybridHairGaussianModel, HybridModelConfig


def main() -> None:
    parser = ArgumentParser(description="Render HairGS hybrid model")
    model = ModelParams(parser, sentinel=True)
    optimization = OptimizationParams(parser)
    GeneralParams(parser)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--type", "-t", type=int, default=0)
    parser.add_argument("--route-sidecar", type=str, default=None)
    parser.add_argument("--hybrid-init-mode", default=None)
    parser.add_argument("--hybrid-temperature", type=float, default=None)
    parser.add_argument("--hybrid-shell-strength", type=float, default=None)
    parser.add_argument("--hybrid-fin-strength", type=float, default=None)
    parser.add_argument("--hybrid-strength-max", type=float, default=None)
    parser.add_argument("--hybrid-roll-max", type=float, default=None)
    parser.add_argument("--hybrid-route-lr", type=float, default=None)
    parser.add_argument("--hybrid-shape-lr", type=float, default=None)
    parser.add_argument("--hybrid-roll-lr", type=float, default=None)
    parser.add_argument("--hybrid-mask-temperature", type=float, default=None)
    # These values are persisted in cfg_args by training.  Rendering does not
    # use them, but argparse must accept command-line overrides cleanly.
    for name, default in (
        ("lambda-hybrid-mask-bce", 0.02),
        ("lambda-hybrid-mask-fp", 0.04),
        ("lambda-hybrid-mask-fn", 0.06),
        ("lambda-hybrid-hole", 0.03),
        ("lambda-hybrid-overflow-peak", 0.10),
        ("lambda-hybrid-route-cost", 2.0e-4),
        ("lambda-hybrid-route-entropy", 5.0e-5),
        ("lambda-hybrid-shape-reg", 2.0e-5),
        ("lambda-hybrid-center-anchor", 0.0),
        ("lambda-hybrid-length-anchor", 0.0),
        ("lambda-hybrid-width-anchor", 0.0),
    ):
        parser.add_argument(f"--{name}", type=float, default=None)
    args = get_combined_args(parser)

    HybridHairGaussianModel.configure(
        HybridModelConfig(
            route_sidecar=Path(args.route_sidecar) if args.route_sidecar else None,
            init_mode=getattr(args, "hybrid_init_mode", None) or "adaptive",
            temperature=getattr(args, "hybrid_temperature", None) or 0.75,
            shell_strength_init=getattr(args, "hybrid_shell_strength", None)
            if getattr(args, "hybrid_shell_strength", None) is not None
            else 0.16,
            fin_strength_init=getattr(args, "hybrid_fin_strength", None)
            if getattr(args, "hybrid_fin_strength", None) is not None
            else 0.20,
            strength_max=getattr(args, "hybrid_strength_max", None)
            if getattr(args, "hybrid_strength_max", None) is not None
            else 0.80,
            roll_residual_max=getattr(args, "hybrid_roll_max", None)
            if getattr(args, "hybrid_roll_max", None) is not None
            else 0.35,
            route_lr=getattr(args, "hybrid_route_lr", None)
            if getattr(args, "hybrid_route_lr", None) is not None
            else 2.5e-3,
            shape_lr=getattr(args, "hybrid_shape_lr", None)
            if getattr(args, "hybrid_shape_lr", None) is not None
            else 1.0e-3,
            roll_lr=getattr(args, "hybrid_roll_lr", None)
            if getattr(args, "hybrid_roll_lr", None) is not None
            else 5.0e-4,
        )
    )
    scene_module.HairGaussianModel = HybridHairGaussianModel
    # Import after patching so render.py resolves the patched Scene model.
    import render as upstream_render
    from scene import Scene

    safe_state(args.quiet)
    with torch.no_grad():
        scene = Scene(args)
        gaussians = scene.gaussians
        if not isinstance(gaussians, HybridHairGaussianModel):
            raise TypeError("The requested checkpoint did not load as hybrid HairGS")
        types = [args.type] if args.type != -1 else [0, 2, 4]
        for render_type in types:
            # Frozen evaluation uses the audited route-sidecar split.  This
            # prevents low-level head/body mask logits from being counted as
            # hair while retaining their opacity as a physically correct
            # occluder for the visible hair silhouette.
            gaussians.force_exact_hair_membership(render_type == 2)
            if not args.skip_train:
                upstream_render.render_set(
                    args,
                    "train",
                    scene.loaded_iter,
                    scene.getCameras(),
                    gaussians,
                    optimization.extract(args),
                    render_type,
                )


if __name__ == "__main__":
    main()
