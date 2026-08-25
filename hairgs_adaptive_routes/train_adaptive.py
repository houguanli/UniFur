#!/usr/bin/env python3
"""HairGS training entry point with route-aware gradient allocation.

Topology stays frozen for the efficiency experiment.  A compact route sidecar
aligned with the loaded PLY assigns each hair segment to shell, fin, or strand;
the non-hair head/base is route -1.  Route-dependent gradient bandwidth spends
most update capacity on explicit strand detail while keeping the shell and base
stable.  The native HairGS renderer and objective are otherwise unchanged.
"""

from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

import train as upstream_train
from arguments import GeneralParams, ModelParams, OptimizationParams
from scene import HairGaussianModel
from utils import enable_accelerated_rasterization, prepare_output_path, safe_state


ROUTE_NAMES = ("shell", "fin", "strand")
ROUTE_GRADIENT_SCALE = {-1: 0.20, 0: 0.55, 1: 0.80, 2: 1.00}


class RouteGradientController:
    def __init__(self, route_ids: np.ndarray, audit_path: Path):
        self.route_ids = route_ids.astype(np.int64, copy=False)
        self.audit_path = audit_path
        self.registered = False
        self.handles: list[torch.utils.hooks.RemovableHandle] = []

    def register(self, gaussians: HairGaussianModel) -> None:
        if self.registered:
            return
        if not isinstance(gaussians, HairGaussianModel):
            raise TypeError("Adaptive route training requires HairGaussianModel")
        if gaussians.endpoint_pairs.shape[0] != self.route_ids.size:
            raise ValueError(
                "Route sidecar/Ply segment mismatch: "
                f"{self.route_ids.size} vs {gaussians.endpoint_pairs.shape[0]}"
            )
        route = torch.as_tensor(self.route_ids, device=gaussians.device)
        segment_scale = torch.ones(route.shape[0], device=gaussians.device)
        for route_id, value in ROUTE_GRADIENT_SCALE.items():
            segment_scale[route == route_id] = float(value)

        endpoint_scale = torch.full(
            (gaussians._endpoints.shape[0],),
            float(ROUTE_GRADIENT_SCALE[-1]),
            device=gaussians.device,
        )
        pairs = gaussians.endpoint_pairs
        for column in range(2):
            endpoint_scale.scatter_reduce_(
                0,
                pairs[:, column],
                segment_scale,
                reduce="amax",
                include_self=True,
            )

        self.handles.append(
            gaussians._endpoints.register_hook(
                lambda gradient: gradient * endpoint_scale[:, None]
            )
        )
        for parameter in (gaussians._features_dc, gaussians._features_rest):
            self.handles.append(
                parameter.register_hook(
                    lambda gradient, scale=segment_scale: gradient
                    * scale[:, None, None]
                )
            )
        for parameter in (gaussians._opacity, gaussians._mask, gaussians._width):
            self.handles.append(
                parameter.register_hook(
                    lambda gradient, scale=segment_scale: gradient * scale[:, None]
                )
            )
        self.registered = True
        counts = {
            "base": int(np.count_nonzero(self.route_ids == -1)),
            **{
                name: int(np.count_nonzero(self.route_ids == index))
                for index, name in enumerate(ROUTE_NAMES)
            },
        }
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self.audit_path.write_text(
            json.dumps(
                {
                    "status": "registered",
                    "route_counts": counts,
                    "gradient_scale": {
                        "base": ROUTE_GRADIENT_SCALE[-1],
                        **{
                            name: ROUTE_GRADIENT_SCALE[index]
                            for index, name in enumerate(ROUTE_NAMES)
                        },
                    },
                    "topology_frozen": True,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def main() -> None:
    parser = ArgumentParser(description="Route-aware HairGS training")
    model = ModelParams(parser)
    optimization = OptimizationParams(parser)
    general = GeneralParams(parser)
    parser.add_argument("--route-sidecar", type=Path, required=True)
    args = parser.parse_args(sys.argv[1:])

    route_data = np.load(args.route_sidecar)
    controller = RouteGradientController(
        route_data["route_id"], Path(args.model_path) / "audit/route_training.json"
    )
    # HairGS persists the full argparse Namespace and later reconstructs it via
    # eval().  PosixPath is not imported by its renderer, so serialize this new
    # argument as a plain string before prepare_output_path writes cfg_args.
    args.route_sidecar = str(args.route_sidecar)
    native_loss = upstream_train.loss_function

    def controlled_loss(gaussians, image, camera, loss_args):
        controller.register(gaussians)
        return native_loss(gaussians, image, camera, loss_args)

    upstream_train.loss_function = controlled_loss
    prepare_output_path(args)
    safe_state(args.quiet)
    enable_accelerated_rasterization()
    model_args = model.extract(args)
    optimization_args = optimization.extract(args)
    general_args = general.extract(args)
    # Upstream train.py reads a module-global `op` at its evaluation sites even
    # though the same object is already passed as `opt`.  The stock __main__
    # happens to create that global; an imported entry point must bind it.
    upstream_train.op = optimization_args
    upstream_train.network_gui.init(general_args.ip, general_args.port)
    upstream_train.training(model_args, optimization_args, general_args, args)


if __name__ == "__main__":
    main()
