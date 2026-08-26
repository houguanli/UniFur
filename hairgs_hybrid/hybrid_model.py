#!/usr/bin/env python3
"""HairGS-compatible anisotropic shell/fin/strand parameterization.

The upstream HairGS CUDA rasterizer already accepts a full 3-axis scale and a
quaternion.  HairGS restricts that interface to a circular cross section
``(length, width, width)``.  This subclass keeps the native topology, SH,
opacity, mask, and endpoints, but adds a differentiable route mixture that can
turn the same primitive into a line (strand), ribbon (fin), or tangent plane
(shell).

No topology operation is supported in this first auditable increment.  Runs
must disable densification, merging, and growth.  Extra state is saved next to
the native PLY, so the PLY remains readable by unmodified HairGS.
"""

from __future__ import annotations

import json
import hashlib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from pytorch3d.transforms import quaternion_multiply, quaternion_to_matrix
from scipy.spatial import cKDTree
from torch import nn

from scene.hair_gaussian_model import HairGaussianModel


ROUTE_NAMES = ("shell", "fin", "strand")


@dataclass(frozen=True)
class HybridModelConfig:
    route_sidecar: Path | None = None
    init_mode: str = "adaptive"
    temperature: float = 0.75
    shell_strength_init: float = 0.16
    fin_strength_init: float = 0.20
    strength_max: float = 0.80
    roll_residual_max: float = 0.35
    route_lr: float = 2.5e-3
    shape_lr: float = 1.0e-3
    roll_lr: float = 5.0e-4

    def validate(self) -> None:
        if self.init_mode not in {"adaptive", "all_strand"}:
            raise ValueError(f"Unsupported hybrid init mode: {self.init_mode}")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.strength_max < 0:
            raise ValueError("strength_max must be non-negative")
        for name, value in (
            ("shell_strength_init", self.shell_strength_init),
            ("fin_strength_init", self.fin_strength_init),
        ):
            if not 0 <= value <= self.strength_max:
                raise ValueError(f"{name} must be inside [0, strength_max]")


def _logit(value: float) -> float:
    value = min(max(value, 1.0e-6), 1.0 - 1.0e-6)
    return float(np.log(value / (1.0 - value)))


class HybridHairGaussianModel(HairGaussianModel):
    """HairGS model with a soft shell/fin/strand covariance router."""

    runtime_config = HybridModelConfig()

    @classmethod
    def configure(cls, config: HybridModelConfig) -> None:
        config.validate()
        cls.runtime_config = config

    def load_ply(self, path):
        super().load_ply(path)
        source = Path(path)
        persisted = source.with_name("hybrid_state.npz")
        if persisted.is_file():
            self._load_hybrid_state(persisted)
        else:
            self._initialize_hybrid_state(self.runtime_config.route_sidecar)
        self._target_roll = self._compute_scalp_target_roll()
        self._force_exact_hair_membership = False
        self._opacity_render_mode = "full"
        self._base_reference_cache = {}
        self._freeze_hook_handles = []
        self._initialize_hair_topology_masks()
        self._initialize_geometry_reference()

    def _initialize_geometry_reference(self) -> None:
        """Freeze the loaded HairGS scaffold as a local geometry trust region.

        These tensors are intentionally not saved as model parameters. Fresh
        training anchors to the audited input PLY; rendering a saved model does
        not consume the references.
        """
        pairs = self._endpoints[self.endpoint_pairs].detach()
        self._reference_centers = pairs.mean(dim=1).clone()
        self._reference_lengths = torch.linalg.vector_norm(
            pairs[:, 1] - pairs[:, 0], dim=1
        ).clone()
        self._reference_log_width = self._width.detach().clone()

    def geometry_anchor_losses(self) -> dict[str, torch.Tensor]:
        hair = self.hair_route_mask
        zero = torch.zeros((), device=self._endpoints.device)
        if not torch.any(hair):
            return {"center": zero, "length": zero, "width": zero}
        pairs = self._endpoints[self.endpoint_pairs]
        centers = pairs.mean(dim=1)
        lengths = torch.linalg.vector_norm(pairs[:, 1] - pairs[:, 0], dim=1)
        reference_lengths = self._reference_lengths
        floor = 0.25 * torch.median(reference_lengths[hair]).clamp_min(self.min_val)
        normalizer = torch.clamp(reference_lengths, min=floor)
        center = (
            (centers - self._reference_centers).square().sum(dim=1)
            / normalizer.square()
        )[hair].mean()
        length = torch.log(
            torch.clamp(lengths, min=self.min_val)
            / torch.clamp(reference_lengths, min=self.min_val)
        )[hair].square().mean()
        width = (self._width - self._reference_log_width)[hair].square().mean()
        return {"center": center, "length": length, "width": width}

    def _initialize_hybrid_state(self, route_sidecar: Path | None) -> None:
        count = int(self.endpoint_pairs.shape[0])
        if route_sidecar is None:
            raise ValueError("A route sidecar is required for a fresh hybrid model")
        route_data = np.load(Path(route_sidecar))
        route_ids = np.asarray(route_data["route_id"], dtype=np.int64)
        if route_ids.shape != (count,):
            raise ValueError(
                f"Route sidecar/PLY mismatch: {route_ids.shape} versus {(count,)}"
            )
        if not np.all(np.isin(route_ids, [-1, 0, 1, 2])):
            raise ValueError("Route sidecar contains an unsupported route id")

        provided_logits = (
            np.asarray(route_data["route_logits_init"], dtype=np.float32)
            if "route_logits_init" in route_data.files
            else None
        )
        if provided_logits is not None and provided_logits.shape != (count, 3):
            raise ValueError(
                "route_logits_init must have shape "
                f"{(count, 3)}, got {provided_logits.shape}"
            )
        logits = np.full((count, 3), -2.0, dtype=np.float32)
        if self.runtime_config.init_mode == "all_strand":
            logits[:, 2] = 2.0
        elif provided_logits is not None:
            if not np.all(np.isfinite(provided_logits)):
                raise ValueError("route_logits_init contains non-finite values")
            logits = provided_logits.copy()
        else:
            for route_id in range(3):
                logits[route_ids == route_id, route_id] = 2.0
            logits[route_ids == -1, 2] = 2.0

        strength_max = self.runtime_config.strength_max
        if strength_max == 0:
            strength_raw = np.zeros((count, 2), dtype=np.float32)
        else:
            shell_fraction = self.runtime_config.shell_strength_init / strength_max
            fin_fraction = self.runtime_config.fin_strength_init / strength_max
            strength_raw = np.empty((count, 2), dtype=np.float32)
            strength_raw[:, 0] = _logit(shell_fraction)
            strength_raw[:, 1] = _logit(fin_fraction)

        self._route_ids = torch.as_tensor(route_ids, device=self.device)
        self._route_logits = nn.Parameter(
            torch.as_tensor(logits, device=self.device).requires_grad_(True)
        )
        self._shape_strength_raw = nn.Parameter(
            torch.as_tensor(strength_raw, device=self.device).requires_grad_(True)
        )
        self._roll_residual = nn.Parameter(
            torch.zeros((count, 1), device=self.device).requires_grad_(True)
        )
        self._loaded_hybrid_state = None

    def _load_hybrid_state(self, path: Path) -> None:
        state = np.load(path)
        count = int(self.endpoint_pairs.shape[0])
        route_ids = np.asarray(state["route_id"], dtype=np.int64)
        logits = np.asarray(state["route_logits"], dtype=np.float32)
        strengths = np.asarray(state["shape_strength_raw"], dtype=np.float32)
        rolls = np.asarray(state["roll_residual"], dtype=np.float32)
        expected = {
            "route_id": (count,),
            "route_logits": (count, 3),
            "shape_strength_raw": (count, 2),
            "roll_residual": (count, 1),
        }
        actual = {
            "route_id": route_ids.shape,
            "route_logits": logits.shape,
            "shape_strength_raw": strengths.shape,
            "roll_residual": rolls.shape,
        }
        if actual != expected:
            raise ValueError(f"Invalid hybrid state shapes: {actual}, expected {expected}")
        self._route_ids = torch.as_tensor(route_ids, device=self.device)
        self._route_logits = nn.Parameter(
            torch.as_tensor(logits, device=self.device).requires_grad_(True)
        )
        self._shape_strength_raw = nn.Parameter(
            torch.as_tensor(strengths, device=self.device).requires_grad_(True)
        )
        self._roll_residual = nn.Parameter(
            torch.as_tensor(rolls, device=self.device).requires_grad_(True)
        )
        self._loaded_hybrid_state = str(path.resolve())

    @property
    def hair_route_mask(self) -> torch.Tensor:
        return self._route_ids >= 0

    @property
    def get_opacity(self):
        """Expose a base-only render mode without mutating native parameters.

        The frozen head/body is a valid occluder and appearance reference for
        human-hair cases.  A temporary opacity view lets the loss render that
        reference once per camera while keeping every hair opacity at exactly
        zero.  Panda has no base segments, so its reference is black.
        """
        opacity = super().get_opacity
        if getattr(self, "_opacity_render_mode", "full") == "base_only":
            return torch.where(
                self.hair_route_mask[:, None], torch.zeros_like(opacity), opacity
            )
        return opacity

    @contextmanager
    def base_only_render_mode(self):
        previous = getattr(self, "_opacity_render_mode", "full")
        self._opacity_render_mode = "base_only"
        try:
            yield
        finally:
            self._opacity_render_mode = previous

    @property
    def get_mask(self):
        """Return native soft membership, or exact audited hair membership.

        HairGS stores head/body and hair segments in one PLY.  The route sidecar
        is the frozen semantic split for this experiment.  Evaluation of the
        hair silhouette must therefore not re-introduce head primitives through
        a learned ``get_mask >= 0.25`` threshold.
        """
        if getattr(self, "_force_exact_hair_membership", False):
            return self.hair_route_mask[:, None].to(dtype=self._mask.dtype)
        return super().get_mask

    def force_exact_hair_membership(self, enabled: bool = True) -> None:
        self._force_exact_hair_membership = bool(enabled)

    @property
    def exact_hair_color(self) -> torch.Tensor:
        return self.hair_route_mask[:, None].to(dtype=self._opacity.dtype).repeat(1, 3)

    def _initialize_hair_topology_masks(self) -> None:
        hair_segments = self.hair_route_mask
        base_segments = ~hair_segments
        endpoint_count = int(self._endpoints.shape[0])
        hair_endpoints = torch.zeros(endpoint_count, dtype=torch.bool, device=self.device)
        base_endpoints = torch.zeros_like(hair_endpoints)
        if torch.any(hair_segments):
            hair_endpoints[self.endpoint_pairs[hair_segments].reshape(-1)] = True
        if torch.any(base_segments):
            base_endpoints[self.endpoint_pairs[base_segments].reshape(-1)] = True
        shared = hair_endpoints & base_endpoints
        if torch.any(shared):
            raise RuntimeError(
                "Hair/head endpoint sharing prevents strict base freezing: "
                f"{int(shared.sum())} shared endpoints"
            )
        self._hair_endpoint_mask = hair_endpoints
        self._base_endpoint_mask = base_endpoints
        self._hair_adjacent_segment_pairs = self._build_hair_adjacency()

    def _build_hair_adjacency(self) -> torch.Tensor:
        """Build pairs of consecutive hair segments without head/body edges."""
        endpoint_pairs = self.endpoint_pairs.detach().cpu().numpy()
        hair = self.hair_route_mask.detach().cpu().numpy()
        incident: dict[int, list[int]] = {}
        for segment_id in np.flatnonzero(hair):
            for endpoint_id in endpoint_pairs[segment_id]:
                incident.setdefault(int(endpoint_id), []).append(int(segment_id))
        adjacent = []
        for segment_ids in incident.values():
            unique = sorted(set(segment_ids))
            if len(unique) == 2:
                adjacent.append(unique)
        if not adjacent:
            return torch.empty((0, 2), dtype=torch.long, device=self.device)
        return torch.as_tensor(np.asarray(adjacent), dtype=torch.long, device=self.device)

    @property
    def hair_adjacent_segment_pairs(self) -> torch.Tensor:
        return self._hair_adjacent_segment_pairs

    def _remove_freeze_hooks(self) -> None:
        for handle in getattr(self, "_freeze_hook_handles", []):
            handle.remove()
        self._freeze_hook_handles = []

    def _register_base_freeze_hooks(self) -> None:
        """Zero every gradient owned by head/body primitives.

        The hair and head endpoint sets are asserted disjoint at load time, so
        this freezes geometry as well as per-segment appearance exactly.
        """
        self._remove_freeze_hooks()

        def register(parameter: nn.Parameter, mask: torch.Tensor) -> None:
            expanded = mask
            while expanded.ndim < parameter.ndim:
                expanded = expanded.unsqueeze(-1)
            expanded = expanded.to(device=parameter.device, dtype=parameter.dtype)
            self._freeze_hook_handles.append(
                parameter.register_hook(lambda gradient, keep=expanded: gradient * keep)
            )

        register(self._endpoints, self._hair_endpoint_mask)
        for parameter in (
            self._features_dc,
            self._features_rest,
            self._opacity,
            self._mask,
            self._width,
            self._route_logits,
            self._shape_strength_raw,
            self._roll_residual,
        ):
            register(parameter, self.hair_route_mask)

    def base_state_digest(self) -> dict[str, str]:
        """Hash the exact frozen native state for audit before/after training."""
        fields = {
            "endpoints": self._endpoints[self._base_endpoint_mask],
            "features_dc": self._features_dc[~self.hair_route_mask],
            "features_rest": self._features_rest[~self.hair_route_mask],
            "opacity": self._opacity[~self.hair_route_mask],
            "mask": self._mask[~self.hair_route_mask],
            "width": self._width[~self.hair_route_mask],
        }
        digests = {}
        combined = hashlib.sha256()
        for name, value in fields.items():
            array = value.detach().cpu().contiguous().numpy()
            payload = array.tobytes(order="C")
            digest = hashlib.sha256(payload).hexdigest()
            digests[name] = digest
            combined.update(name.encode("utf-8"))
            combined.update(payload)
        digests["combined"] = combined.hexdigest()
        return digests

    @property
    def route_probabilities(self) -> torch.Tensor:
        probabilities = torch.softmax(
            self._route_logits / self.runtime_config.temperature, dim=1
        )
        if torch.any(~self.hair_route_mask):
            base = torch.zeros_like(probabilities)
            base[:, 2] = 1.0
            probabilities = torch.where(
                self.hair_route_mask[:, None], probabilities, base
            )
        return probabilities

    @property
    def shape_strengths(self) -> torch.Tensor:
        if self.runtime_config.strength_max == 0:
            return torch.zeros_like(self._shape_strength_raw)
        return self.runtime_config.strength_max * torch.sigmoid(
            self._shape_strength_raw
        )

    @property
    def get_scaling(self):
        native = super().get_scaling
        if self.runtime_config.init_mode == "all_strand":
            return native
        probabilities = self.route_probabilities
        strengths = self.shape_strengths
        shell = probabilities[:, 0:1] * strengths[:, 0:1]
        fin = probabilities[:, 1:2] * strengths[:, 1:2]
        # Volume-preserving log-scale bases:
        # shell: broad tangent/binormal, thin normal
        # fin:   native tangent, broad binormal, thin normal
        delta = torch.cat(
            (
                0.35 * shell,
                0.35 * shell + 0.50 * fin,
                -0.70 * shell - 0.50 * fin,
            ),
            dim=1,
        )
        delta = torch.where(self.hair_route_mask[:, None], delta, 0.0 * delta)
        return torch.clamp(native * torch.exp(delta), min=self.min_val)

    @property
    def get_rotation(self):
        native = super().get_rotation
        if self.runtime_config.init_mode == "all_strand":
            return native
        non_strand = self.route_probabilities[:, 0:2].sum(dim=1, keepdim=True)
        residual = self.runtime_config.roll_residual_max * torch.tanh(
            self._roll_residual
        )
        angle = non_strand * (self._target_roll + residual)
        angle = torch.where(self.hair_route_mask[:, None], angle, 0.0 * angle)
        half = 0.5 * angle
        local_roll = torch.cat(
            (
                torch.cos(half),
                torch.sin(half),
                torch.zeros_like(half),
                torch.zeros_like(half),
            ),
            dim=1,
        )
        return quaternion_multiply(native, local_roll)

    def _compute_scalp_target_roll(self) -> torch.Tensor:
        count = int(self.endpoint_pairs.shape[0])
        if self.ref_strand_root is None or len(self.ref_strand_root) == 0:
            return torch.zeros((count, 1), device=self.device)
        centers = self.get_xyz.detach().cpu().numpy()
        roots = np.asarray(self.ref_strand_root, dtype=np.float64)
        tree = cKDTree(roots)
        _, nearest_ids = tree.query(centers, k=1)
        nearest = torch.as_tensor(
            roots[np.asarray(nearest_ids)], dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            tangent = self.get_orientation
            desired_normal = self.get_xyz - nearest
            desired_normal = desired_normal - (
                desired_normal * tangent
            ).sum(dim=1, keepdim=True) * tangent
            native_matrix = quaternion_to_matrix(super().get_rotation)
            native_y = native_matrix[:, :, 1]
            native_z = native_matrix[:, :, 2]
            norm = torch.linalg.vector_norm(desired_normal, dim=1, keepdim=True)
            desired_normal = torch.where(
                norm > self.min_val,
                desired_normal / torch.clamp(norm, min=self.min_val),
                native_z,
            )
            sin_term = -(desired_normal * native_y).sum(dim=1, keepdim=True)
            cos_term = (desired_normal * native_z).sum(dim=1, keepdim=True)
            return torch.atan2(sin_term, cos_term)

    def training_setup(self, training_args):
        super().training_setup(training_args)
        if self.runtime_config.init_mode != "all_strand":
            self.optimizer.add_param_group(
                {
                    "params": [self._route_logits],
                    "lr": self.runtime_config.route_lr,
                    "name": "hybrid_route_logits",
                }
            )
            self.optimizer.add_param_group(
                {
                    "params": [self._shape_strength_raw],
                    "lr": self.runtime_config.shape_lr,
                    "name": "hybrid_shape_strength",
                }
            )
            self.optimizer.add_param_group(
                {
                    "params": [self._roll_residual],
                    "lr": self.runtime_config.roll_lr,
                    "name": "hybrid_roll_residual",
                }
            )
        self._register_base_freeze_hooks()

    def save_ply(self, path):
        super().save_ply(path)
        state_path = Path(path).with_name("hybrid_state.npz")
        np.savez_compressed(
            state_path,
            schema=np.asarray("hairgs-hybrid-state-v1"),
            route_names=np.asarray(ROUTE_NAMES),
            route_id=self._route_ids.detach().cpu().numpy().astype(np.int8),
            route_logits=self._route_logits.detach().cpu().numpy().astype(np.float32),
            shape_strength_raw=self._shape_strength_raw.detach()
            .cpu()
            .numpy()
            .astype(np.float32),
            roll_residual=self._roll_residual.detach()
            .cpu()
            .numpy()
            .astype(np.float32),
        )
        report = self.hybrid_report()
        state_path.with_name("hybrid_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )

    def hybrid_report(self) -> dict:
        with torch.no_grad():
            probabilities = self.route_probabilities.detach().cpu().numpy()
            hard = probabilities.argmax(axis=1)
            hair = self.hair_route_mask.detach().cpu().numpy()
            strengths = self.shape_strengths.detach().cpu().numpy()
            scaling = self.get_scaling.detach()
            ratios = (scaling / torch.clamp(super().get_scaling, min=self.min_val)).cpu().numpy()
        return {
            "schema": "hairgs-hybrid-report-v1",
            "init_mode": self.runtime_config.init_mode,
            "temperature": self.runtime_config.temperature,
            "loaded_hybrid_state": self._loaded_hybrid_state,
            "segment_count": int(hard.size),
            "hair_segment_count": int(np.count_nonzero(hair)),
            "base_segment_count": int(np.count_nonzero(~hair)),
            "hair_endpoint_count": int(self._hair_endpoint_mask.sum()),
            "base_endpoint_count": int(self._base_endpoint_mask.sum()),
            "shared_endpoint_count": int(
                (self._hair_endpoint_mask & self._base_endpoint_mask).sum()
            ),
            "hair_adjacency_pair_count": int(self.hair_adjacent_segment_pairs.shape[0]),
            "base_state_digest": self.base_state_digest(),
            "hard_route_counts": {
                name: int(np.count_nonzero(np.logical_and(hair, hard == index)))
                for index, name in enumerate(ROUTE_NAMES)
            },
            "route_mass": {
                name: float(probabilities[hair, index].sum())
                for index, name in enumerate(ROUTE_NAMES)
            },
            "mean_strength": {
                "shell": float(strengths[hair, 0].mean()) if np.any(hair) else 0.0,
                "fin": float(strengths[hair, 1].mean()) if np.any(hair) else 0.0,
            },
            "scale_ratio_quantiles": {
                axis: [float(value) for value in np.quantile(ratios[hair, index], [0, 0.25, 0.5, 0.75, 1])]
                if np.any(hair)
                else [1.0] * 5
                for index, axis in enumerate(("tangent", "binormal", "normal"))
            },
        }
