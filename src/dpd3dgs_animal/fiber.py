from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .gaussian import (
    GaussianSurfaceBinding,
    bind_gaussians_to_surface,
    load_gaussian_ply,
)
from .render import PinholeCamera


ROUTE_NAMES = ("shell", "strand", "residual")
CARRIER_NAMES = ("surface", "shell", "strand")
HARD_ROUTE_POLICIES = ("argmax", "mass_preserving")


def mass_preserving_route_ids(probabilities: torch.Tensor) -> torch.Tensor:
    """Assign one route per source while preserving aggregate soft route mass.

    Ordinary argmax discretisation is locally optimal but can collapse almost
    every source into residual.  This deterministic allocator turns summed soft
    probabilities into integer capacities, then moves the least-confident
    argmax assignments until those capacities are met.  It only changes the
    hard forward path; straight-through gradients stay attached to the soft
    probabilities.
    """
    if probabilities.ndim != 2 or probabilities.shape[1] != len(ROUTE_NAMES):
        raise ValueError(
            "probabilities must have shape (points, "
            f"{len(ROUTE_NAMES)}), got {tuple(probabilities.shape)}"
        )
    point_count = int(probabilities.shape[0])
    if point_count == 0:
        return torch.empty((0,), dtype=torch.long, device=probabilities.device)

    detached = probabilities.detach()
    if not torch.isfinite(detached).all():
        raise RuntimeError(
            "Mass-preserving route allocation received non-finite probabilities"
        )
    row_mass = detached.sum(dim=-1, keepdim=True)
    if torch.any(row_mass <= 0.0):
        raise RuntimeError(
            "Mass-preserving route allocation received a zero-mass probability row"
        )
    # A straight-through mixture can differ from unit row mass by a few ulps.
    # Normalize before converting aggregate mass to integer capacities so a
    # floating-point deficit cannot become an unfillable point quota.
    detached = detached / row_mass
    expected_counts = detached.sum(dim=0)
    capacities = torch.floor(expected_counts).to(torch.long)
    remaining = point_count - int(capacities.sum().item())
    fractional = expected_counts - capacities.to(expected_counts.dtype)
    if remaining > 0:
        capacities[torch.argsort(fractional, descending=True)[:remaining]] += 1
    elif remaining < 0:
        # Numerical protection: remove an over-allocation without allowing a
        # negative capacity.
        for target in torch.argsort(fractional, descending=False).tolist():
            removable = min(-remaining, int(capacities[target].item()))
            capacities[target] -= removable
            remaining += removable
            if remaining == 0:
                break
    if int(capacities.sum().item()) != point_count:
        raise RuntimeError(
            "Mass-preserving route capacity rounding did not conserve point count"
        )

    route_ids = detached.argmax(dim=-1)
    counts = torch.bincount(route_ids, minlength=len(ROUTE_NAMES))
    log_probabilities = detached.clamp_min(1e-8).log()
    tie_scale = torch.finfo(log_probabilities.dtype).eps / float(max(point_count, 1))

    # Greedily reroute sources with the smallest log-probability penalty.  A
    # source route only loses points while it has surplus and a destination only
    # receives points while it has capacity, so the final histogram is exact.
    for source in range(len(ROUTE_NAMES)):
        while int(counts[source].item()) > int(capacities[source].item()):
            destinations = torch.nonzero(counts < capacities, as_tuple=False).flatten()
            if destinations.numel() == 0:
                break
            destination = int(destinations[0].item())
            move_count = min(
                int(counts[source].item() - capacities[source].item()),
                int(capacities[destination].item() - counts[destination].item()),
            )
            candidates = torch.nonzero(route_ids == source, as_tuple=False).flatten()
            penalty = (
                log_probabilities[candidates, source]
                - log_probabilities[candidates, destination]
            )
            penalty = penalty + candidates.to(penalty.dtype) * tie_scale
            selected = candidates[torch.topk(-penalty, k=move_count).indices]
            route_ids[selected] = destination
            counts[source] -= move_count
            counts[destination] += move_count
    if not torch.equal(counts, capacities):
        raise RuntimeError("Mass-preserving route allocation failed to meet capacities")
    return route_ids


@dataclass
class FiberPrimitives:
    """Renderer-facing Gaussian primitives produced by the unified field.

    ``scaling`` and ``rotation`` follow the original 3DGS convention: three
    world-space standard deviations and a wxyz quaternion.  The lightweight
    Torch renderer below uses the transverse scale only, while the same
    tensors can be passed to a CUDA Gaussian rasterizer.
    """

    xyz: torch.Tensor
    color: torch.Tensor
    opacity: torch.Tensor
    scaling: torch.Tensor
    rotation: torch.Tensor
    route_id: torch.Tensor
    source_id: torch.Tensor
    route_probabilities: torch.Tensor
    # Renderer-side structural metadata.  Surface normals make the shell
    # expert view-selective at grazing angles; root_tip and structure_weight
    # are also used by downstream edit/animation audits.  Optional defaults
    # keep older call sites and serialized diagnostic objects compatible.
    surface_normal: torch.Tensor | None = None
    root_tip: torch.Tensor | None = None
    structure_weight: torch.Tensor | None = None
    root_xyz: torch.Tensor | None = None
    # Rendering ownership and deformation ownership are deliberately
    # separate.  A Gaussian may remain residual for photometric safety while
    # being driven by a surface, shell or strand carrier downstream.
    carrier_probabilities: torch.Tensor | None = None
    carrier_root_tip: torch.Tensor | None = None


class UnifiedFiberField(nn.Module):
    """Per-sequence surface-anchored shell/strand/residual Gaussian field.

    This module is intentionally *not* a feed-forward reconstructor.  Its
    parameters are initialized from an ordinary Gaussian reconstruction and
    optimized for one video through differentiable rendering.
    """

    def __init__(
        self,
        *,
        face_index: torch.Tensor,
        barycentric: torch.Tensor,
        color: torch.Tensor,
        opacity: torch.Tensor,
        original_scaling: torch.Tensor,
        original_rotation: torch.Tensor,
        rest_surface_frame: torch.Tensor,
        residual_offset_local: torch.Tensor,
        direction_local: torch.Tensor,
        height: torch.Tensor,
        shell_length: torch.Tensor,
        strand_length: torch.Tensor,
        radius: torch.Tensor,
        route_logits: torch.Tensor,
        scene_scale: float,
        carrier_logits: torch.Tensor | None = None,
        carrier_root_tip: torch.Tensor | None = None,
        initial_residual_trust: float = 0.95,
        route_neighbor_index: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        eps = max(float(scene_scale) * 1e-7, 1e-8)
        self.positive_eps = eps
        self.register_buffer("face_index", face_index.long())
        self.register_buffer("barycentric", barycentric.float())
        self.register_buffer("original_scaling", original_scaling.float())
        self.register_buffer("original_rotation", _normalize_quaternion(original_rotation.float()))
        # Columns are the rest-pose tangent, bitangent and normal.  This is
        # intentionally non-persistent: old checkpoints contain the learned
        # rest-world residual quaternion, while the frame is reconstructed
        # deterministically from the input surface on every load.
        self.register_buffer(
            "rest_surface_frame", rest_surface_frame.float(), persistent=False
        )
        self.register_buffer("initial_residual_offset_local", residual_offset_local.float().clone())
        self.register_buffer("scene_scale", torch.tensor(float(scene_scale), dtype=torch.float32))
        if route_neighbor_index is None:
            route_neighbor_index = torch.empty(
                (color.shape[0], 0), dtype=torch.long, device=color.device
            )
        self.register_buffer(
            "route_neighbor_index",
            route_neighbor_index.long(),
            persistent=False,
        )
        self.register_buffer(
            "strand_visibility_gate",
            torch.empty(
                (color.shape[0], 0), dtype=torch.float32, device=color.device
            ),
        )

        color = color.clamp(1e-4, 1.0 - 1e-4)
        opacity = opacity.clamp(1e-4, 1.0 - 1e-4)
        self.color_logits = nn.Parameter(torch.logit(color))
        # The shared logit remains the compact appearance scaffold.  Each
        # route learns only a zero-initialized correction, so the unified
        # model is exactly render-equivalent to the previous model at step 0
        # while shell/strand/residual can specialize afterwards.
        self.expert_color_delta = nn.Parameter(
            torch.zeros(
                (color.shape[0], len(ROUTE_NAMES), color.shape[1]),
                dtype=torch.float32,
                device=color.device,
            )
        )
        self.opacity_logits = nn.Parameter(torch.logit(opacity.reshape(-1, 1)))
        self.residual_offset_local = nn.Parameter(residual_offset_local.float())
        # A residual-only comparison must retain the actual anisotropic 3DGS
        # degrees of freedom. Earlier scaffold experiments optimized only
        # position/color/opacity and kept covariance frozen, which is too weak
        # to call a 3DGS baseline.
        self.residual_log_scale_delta = nn.Parameter(
            torch.zeros_like(original_scaling.float())
        )
        self.residual_rotation_raw = nn.Parameter(
            _normalize_quaternion(original_rotation.float()).clone()
        )
        self.direction_local_raw = nn.Parameter(direction_local.float())
        self.bend_local = nn.Parameter(torch.zeros((color.shape[0], 2), dtype=torch.float32, device=color.device))
        # A second tangent-plane coefficient turns the previous parabolic
        # strand into a cubic curve.  Zero initialization is exactly backward
        # compatible, while opposite quadratic/cubic signs can represent an
        # inflection (important for wavy and curly human hair).
        self.bend_cubic_local = nn.Parameter(
            torch.zeros(
                (color.shape[0], 2), dtype=torch.float32, device=color.device
            )
        )
        # A structured expert starts as an *exact* copy of the residual
        # teacher.  The two gains below are zero-initialized learned
        # increments (shell, strand), rather than a non-zero geometric branch
        # that is merely hidden by a training schedule.  The straight-through
        # clamp in ``structured_delta_gain`` keeps the forward value in [0, 1]
        # while retaining a useful derivative at the exact-zero boundary.
        self.structured_delta_raw = nn.Parameter(
            torch.zeros(
                (color.shape[0], 2), dtype=torch.float32, device=color.device
            )
        )
        # Visibility is separated from geometry unfolding.  Boundary evidence
        # can first activate an exact teacher-position copy and only then move
        # it toward a Fin/strand target, avoiding a product of two tiny gains.
        self.structured_opacity_raw = nn.Parameter(
            torch.zeros(
                (color.shape[0], 2), dtype=torch.float32, device=color.device
            )
        )
        self.height_raw = nn.Parameter(_inverse_softplus(height.reshape(-1, 1), eps))
        self.shell_length_raw = nn.Parameter(_inverse_softplus(shell_length.reshape(-1, 1), eps))
        self.strand_length_raw = nn.Parameter(_inverse_softplus(strand_length.reshape(-1, 1), eps))
        self.radius_raw = nn.Parameter(_inverse_softplus(radius.reshape(-1, 1), eps))
        self.route_logits = nn.Parameter(route_logits.float())
        if carrier_logits is None:
            carrier_logits = torch.stack(
                [route_logits[:, 2], route_logits[:, 0], route_logits[:, 1]],
                dim=-1,
            )
        if tuple(carrier_logits.shape) != (color.shape[0], len(CARRIER_NAMES)):
            raise ValueError(
                "carrier_logits must have shape "
                f"({color.shape[0]}, {len(CARRIER_NAMES)})"
            )
        self.carrier_logits = nn.Parameter(carrier_logits.float())
        if carrier_root_tip is None:
            carrier_root_tip = torch.zeros(
                (color.shape[0],), dtype=torch.float32, device=color.device
            )
        carrier_root_tip = carrier_root_tip.float().reshape(-1)
        if carrier_root_tip.shape[0] != color.shape[0]:
            raise ValueError("carrier_root_tip must contain one value per source")
        self.carrier_root_tip_raw = nn.Parameter(carrier_root_tip.clamp(0.0, 1.0))
        self.register_buffer(
            "initial_carrier_probabilities",
            torch.softmax(carrier_logits.float(), dim=-1).clamp_min(1e-6),
        )
        self.register_buffer(
            "initial_carrier_root_tip", carrier_root_tip.clamp(0.0, 1.0).clone()
        )
        initial_trust = min(max(float(initial_residual_trust), 1e-4), 1.0 - 1e-4)
        self.residual_trust_logits = nn.Parameter(
            torch.full(
                (color.shape[0], 1),
                torch.logit(torch.tensor(initial_trust)).item(),
                dtype=torch.float32,
                device=color.device,
            )
        )
        base_initial_probabilities = torch.softmax(route_logits.float(), dim=-1)
        initial_route_probabilities = (1.0 - initial_trust) * base_initial_probabilities
        initial_route_probabilities[:, ROUTE_NAMES.index("residual")] += initial_trust
        self.register_buffer(
            "initial_route_probabilities",
            initial_route_probabilities.clamp_min(1e-6),
        )

    @property
    def point_count(self) -> int:
        return int(self.face_index.shape[0])

    @property
    def color(self) -> torch.Tensor:
        return torch.sigmoid(self.color_logits)

    @property
    def expert_color(self) -> torch.Tensor:
        return torch.sigmoid(self.color_logits[:, None, :] + self.expert_color_delta)

    @property
    def opacity(self) -> torch.Tensor:
        return torch.sigmoid(self.opacity_logits).reshape(-1)

    @property
    def height(self) -> torch.Tensor:
        return F.softplus(self.height_raw).reshape(-1) + self.positive_eps

    @property
    def shell_length(self) -> torch.Tensor:
        return F.softplus(self.shell_length_raw).reshape(-1) + self.positive_eps

    @property
    def strand_length(self) -> torch.Tensor:
        return F.softplus(self.strand_length_raw).reshape(-1) + self.positive_eps

    @property
    def radius(self) -> torch.Tensor:
        return F.softplus(self.radius_raw).reshape(-1) + self.positive_eps

    @property
    def direction_local(self) -> torch.Tensor:
        return F.normalize(self.direction_local_raw, dim=-1, eps=1e-8)

    @property
    def residual_trust(self) -> torch.Tensor:
        return torch.sigmoid(self.residual_trust_logits).reshape(-1)

    @property
    def structured_delta_gain(self) -> torch.Tensor:
        clipped = self.structured_delta_raw.clamp(0.0, 1.0)
        return self.structured_delta_raw + (clipped - self.structured_delta_raw).detach()

    @property
    def structured_opacity_gain(self) -> torch.Tensor:
        clipped = self.structured_opacity_raw.clamp(0.0, 1.0)
        return self.structured_opacity_raw + (
            clipped - self.structured_opacity_raw
        ).detach()

    @property
    def carrier_root_tip(self) -> torch.Tensor:
        clipped = self.carrier_root_tip_raw.clamp(0.0, 1.0)
        return self.carrier_root_tip_raw + (
            clipped - self.carrier_root_tip_raw
        ).detach()

    def carrier_probabilities(
        self, temperature: float = 1.0, hard: bool = False
    ) -> torch.Tensor:
        probabilities = torch.softmax(
            self.carrier_logits / max(float(temperature), 1e-4), dim=-1
        )
        if hard:
            ids = probabilities.argmax(dim=-1)
            hard_probabilities = F.one_hot(
                ids, num_classes=len(CARRIER_NAMES)
            ).to(probabilities.dtype)
            probabilities = (
                hard_probabilities.detach()
                - probabilities.detach()
                + probabilities
            )
        return probabilities

    @property
    def residual_scaling(self) -> torch.Tensor:
        scale_multiplier = torch.exp(
            self.residual_log_scale_delta.clamp(min=-4.0, max=4.0)
        )
        return (self.original_scaling * scale_multiplier).clamp_min(
            self.positive_eps
        )

    @property
    def residual_rotation(self) -> torch.Tensor:
        return _normalize_quaternion(self.residual_rotation_raw)

    def transported_residual_rotation(
        self,
        tangent: torch.Tensor,
        bitangent: torch.Tensor,
        normal: torch.Tensor,
    ) -> torch.Tensor:
        """Transport residual covariance from its rest frame to the surface.

        ``residual_rotation_raw`` remains a trainable rest-world orientation.
        The current face frame supplies the rigid frame delta, so an
        anisotropic residual Gaussian follows articulated surface rotation in
        addition to the already transported local center offset.
        """

        current_surface_frame = torch.stack(
            [tangent, bitangent, normal], dim=-1
        )
        frame_delta = current_surface_frame @ self.rest_surface_frame.transpose(
            -1, -2
        )
        residual_matrix = _quaternion_to_matrix_torch(self.residual_rotation)
        transported_matrix = frame_delta @ residual_matrix
        return _matrix_to_quaternion_torch(transported_matrix)

    def route_probabilities(
        self,
        temperature: float = 1.0,
        forced_route: str | None = None,
        hard: bool = False,
        route_blend: float = 1.0,
        hardening: float = 0.0,
        dropped_route: str | None = None,
        hard_policy: str = "argmax",
    ) -> torch.Tensor:
        if hard_policy not in HARD_ROUTE_POLICIES:
            raise ValueError(
                f"Unknown hard route policy {hard_policy!r}; expected one of {HARD_ROUTE_POLICIES}"
            )
        if forced_route is not None:
            if forced_route not in ROUTE_NAMES:
                raise ValueError(f"Unknown route {forced_route!r}; expected one of {ROUTE_NAMES}")
            probabilities = torch.zeros_like(self.route_logits)
            probabilities[:, ROUTE_NAMES.index(forced_route)] = 1.0
            return probabilities
        probabilities = torch.softmax(self.route_logits / max(float(temperature), 1e-4), dim=-1)
        trust = self.residual_trust[:, None]
        probabilities = (1.0 - trust) * probabilities
        probabilities[:, ROUTE_NAMES.index("residual")] += trust[:, 0]
        blend = min(max(float(route_blend), 0.0), 1.0)
        if blend < 1.0:
            residual = torch.zeros_like(probabilities)
            residual[:, ROUTE_NAMES.index("residual")] = 1.0
            probabilities = blend * probabilities + (1.0 - blend) * residual
        if dropped_route is not None:
            if dropped_route not in ROUTE_NAMES:
                raise ValueError(
                    f"Unknown dropped route {dropped_route!r}; expected one of {ROUTE_NAMES}"
                )
            keep = torch.ones_like(probabilities)
            keep[:, ROUTE_NAMES.index(dropped_route)] = 0.0
            probabilities = probabilities * keep
            probabilities = probabilities / probabilities.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)
        hardness = 1.0 if hard else min(max(float(hardening), 0.0), 1.0)
        if hardness > 0.0:
            route_ids = (
                probabilities.argmax(dim=-1)
                if hard_policy == "argmax"
                else mass_preserving_route_ids(probabilities)
            )
            hard_probabilities = F.one_hot(
                route_ids, num_classes=len(ROUTE_NAMES)
            ).to(probabilities.dtype)
            straight_through = (
                hard_probabilities.detach() - probabilities.detach() + probabilities
            )
            probabilities = (
                (1.0 - hardness) * probabilities + hardness * straight_through
            )
        return probabilities

    def surface_frame(
        self,
        surface_vertices: torch.Tensor,
        surface_faces: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        triangles = surface_vertices[surface_faces[self.face_index]]
        root = (self.barycentric[..., None] * triangles).sum(dim=1)
        edge0 = triangles[:, 1] - triangles[:, 0]
        edge1 = triangles[:, 2] - triangles[:, 0]
        tangent = F.normalize(edge0, dim=-1, eps=1e-8)
        normal = F.normalize(torch.linalg.cross(edge0, edge1, dim=-1), dim=-1, eps=1e-8)
        bitangent = F.normalize(torch.linalg.cross(normal, tangent, dim=-1), dim=-1, eps=1e-8)
        return root, tangent, bitangent, normal

    def primitives(
        self,
        surface_vertices: torch.Tensor,
        surface_faces: torch.Tensor,
        *,
        shell_samples: int = 2,
        strand_samples: int = 5,
        temperature: float = 1.0,
        forced_route: str | None = None,
        hard_route: bool = False,
        route_blend: float = 1.0,
        geometry_blend: float = 1.0,
        route_hardening: float = 0.0,
        dropped_route: str | None = None,
        hard_route_policy: str = "argmax",
        strand_visibility: torch.Tensor | None = None,
        fin_aspect_ratio: float = 1.0,
        additive_teacher: bool = False,
    ) -> FiberPrimitives:
        if shell_samples < 1 or strand_samples < 1:
            raise ValueError("shell_samples and strand_samples must be positive")
        if float(fin_aspect_ratio) < 1.0:
            raise ValueError("fin_aspect_ratio must be at least one")
        root, tangent, bitangent, normal = self.surface_frame(surface_vertices, surface_faces)
        direction = _local_to_world(self.direction_local, tangent, bitangent, normal)
        direction = F.normalize(direction, dim=-1, eps=1e-8)
        probabilities = self.route_probabilities(
            temperature=temperature,
            forced_route=forced_route,
            hard=hard_route,
            route_blend=route_blend,
            hardening=route_hardening,
            dropped_route=dropped_route,
            hard_policy=hard_route_policy,
        )
        source_id = torch.arange(self.point_count, device=root.device, dtype=torch.long)

        shell_t = (
            torch.arange(shell_samples, device=root.device, dtype=root.dtype) + 0.5
        ) / float(shell_samples)
        shell_origin = root + self.height[:, None] * normal
        shell_xyz = shell_origin[:, None, :] + (
            self.shell_length[:, None, None] * shell_t[None, :, None] * direction[:, None, :]
        )
        shell_direction = direction[:, None, :].expand(-1, shell_samples, -1)
        shell_axis_scale = self.shell_length[:, None] / (2.0 * shell_samples)
        fin_sqrt_aspect = math.sqrt(float(fin_aspect_ratio))
        shell_scaling = torch.stack(
            [
                shell_axis_scale.expand(-1, shell_samples),
                fin_sqrt_aspect
                * self.radius[:, None].expand(-1, shell_samples),
                self.radius[:, None].expand(-1, shell_samples)
                / fin_sqrt_aspect,
            ],
            dim=-1,
        )

        strand_t = (
            torch.arange(strand_samples, device=root.device, dtype=root.dtype) + 0.5
        ) / float(strand_samples)
        bend_world = (
            self.bend_local[:, :1] * tangent + self.bend_local[:, 1:] * bitangent
        )
        bend_cubic_world = (
            self.bend_cubic_local[:, :1] * tangent
            + self.bend_cubic_local[:, 1:] * bitangent
        )
        strand_xyz = shell_origin[:, None, :] + (
            self.strand_length[:, None, None]
            * (
                strand_t[None, :, None] * direction[:, None, :]
                + strand_t[None, :, None].square() * bend_world[:, None, :]
                + strand_t[None, :, None].pow(3) * bend_cubic_world[:, None, :]
            )
        )
        strand_direction = F.normalize(
            direction[:, None, :]
            + 2.0 * strand_t[None, :, None] * bend_world[:, None, :]
            + 3.0 * strand_t[None, :, None].square() * bend_cubic_world[:, None, :],
            dim=-1,
            eps=1e-8,
        )
        strand_axis_scale = self.strand_length[:, None] / (2.0 * strand_samples)
        strand_scaling = torch.stack(
            [
                strand_axis_scale.expand(-1, strand_samples),
                self.radius[:, None].expand(-1, strand_samples),
                self.radius[:, None].expand(-1, strand_samples),
            ],
            dim=-1,
        )

        residual_xyz = root + _local_to_world(
            self.residual_offset_local, tangent, bitangent, normal
        )
        residual_rotation = self.transported_residual_rotation(
            tangent, bitangent, normal
        )
        # At the start of structured routing, every shell/strand expert is an
        # exact split copy of its residual Gaussian.  In addition to the global
        # continuation schedule, each source owns a zero-initialized learned
        # delta gain.  This prevents a scheduled switch from exposing a large,
        # uncalibrated strand displacement all at once.
        geometry_mix = min(max(float(geometry_blend), 0.0), 1.0)
        # A Fin is a thin ribbon spanning the fur direction and a surface
        # tangent.  Unlike a round strand Gaussian, this keeps a meaningful
        # cross-section that can carry the silhouette with fewer primitives.
        shell_side = tangent[:, None, :].expand(-1, shell_samples, -1)
        shell_side = shell_side - torch.sum(
            shell_side * shell_direction, dim=-1, keepdim=True
        ) * shell_direction
        fallback_side = bitangent[:, None, :].expand(-1, shell_samples, -1)
        fallback_side = fallback_side - torch.sum(
            fallback_side * shell_direction, dim=-1, keepdim=True
        ) * shell_direction
        shell_side = torch.where(
            torch.linalg.vector_norm(shell_side, dim=-1, keepdim=True) > 1e-5,
            shell_side,
            fallback_side,
        )
        shell_side = F.normalize(shell_side, dim=-1, eps=1e-8)
        shell_fin_normal = F.normalize(
            torch.linalg.cross(shell_direction, shell_side, dim=-1),
            dim=-1,
            eps=1e-8,
        )
        shell_frame = torch.stack(
            [shell_direction, shell_side, shell_fin_normal], dim=-1
        )
        shell_rotation = _matrix_to_quaternion_torch(
            shell_frame.reshape(-1, 3, 3)
        )
        strand_rotation = _quaternion_from_x_axis(strand_direction.reshape(-1, 3))
        delta_gain = self.structured_delta_gain
        shell_mix = geometry_mix * delta_gain[:, 0]
        strand_mix = geometry_mix * delta_gain[:, 1]
        shell_teacher_xyz = residual_xyz[:, None, :].expand(
            -1, shell_samples, -1
        )
        strand_teacher_xyz = residual_xyz[:, None, :].expand(
            -1, strand_samples, -1
        )
        shell_xyz = torch.lerp(
            shell_teacher_xyz, shell_xyz, shell_mix[:, None, None]
        )
        strand_xyz = torch.lerp(
            strand_teacher_xyz, strand_xyz, strand_mix[:, None, None]
        )
        shell_teacher_scaling = self.residual_scaling[:, None, :].expand(
            -1, shell_samples, -1
        )
        strand_teacher_scaling = self.residual_scaling[:, None, :].expand(
            -1, strand_samples, -1
        )
        shell_scaling = torch.lerp(
            shell_teacher_scaling, shell_scaling, shell_mix[:, None, None]
        )
        strand_scaling = torch.lerp(
            strand_teacher_scaling, strand_scaling, strand_mix[:, None, None]
        )
        shell_teacher_rotation = residual_rotation[:, None, :].expand(
            -1, shell_samples, -1
        ).reshape(-1, 4)
        strand_teacher_rotation = residual_rotation[:, None, :].expand(
            -1, strand_samples, -1
        ).reshape(-1, 4)
        shell_rotation = _quaternion_nlerp(
            shell_teacher_rotation,
            shell_rotation,
            shell_mix[:, None].expand(-1, shell_samples).reshape(-1, 1),
        )
        strand_rotation = _quaternion_nlerp(
            strand_teacher_rotation,
            strand_rotation,
            strand_mix[:, None].expand(-1, strand_samples).reshape(-1, 1),
        )
        shell_source_opacity = self.opacity * probabilities[:, 0]
        strand_source_opacity = self.opacity * probabilities[:, 1]
        if additive_teacher:
            # Geometry gain doubles as a zero-initialized structured opacity
            # gate.  The residual teacher remains complete; shell/strand can
            # only add evidence after their geometry begins to unfold.
            opacity_gain = self.structured_opacity_gain
            shell_source_opacity = shell_source_opacity * opacity_gain[:, 0]
            strand_source_opacity = strand_source_opacity * opacity_gain[:, 1]
        shell_opacity = _split_opacity(
            shell_source_opacity, shell_samples
        )[:, None].expand(-1, shell_samples)
        strand_opacity = _split_opacity(
            strand_source_opacity, strand_samples
        )[:, None].expand(-1, strand_samples)
        if strand_visibility is None and tuple(self.strand_visibility_gate.shape) == (
            self.point_count,
            strand_samples,
        ):
            strand_visibility = self.strand_visibility_gate
        if strand_visibility is not None:
            if tuple(strand_visibility.shape) != (self.point_count, strand_samples):
                raise ValueError(
                    "strand_visibility must have shape "
                    f"({self.point_count}, {strand_samples}), got "
                    f"{tuple(strand_visibility.shape)}"
                )
            strand_opacity = strand_opacity * strand_visibility.to(
                device=strand_opacity.device, dtype=strand_opacity.dtype
            )
        if additive_teacher:
            residual_visible = (
                forced_route in (None, "residual") and dropped_route != "residual"
            )
            residual_opacity = (
                self.opacity if residual_visible else torch.zeros_like(self.opacity)
            )
        else:
            residual_opacity = self.opacity * probabilities[:, 2]

        xyz = torch.cat(
            [
                shell_xyz.reshape(-1, 3),
                strand_xyz.reshape(-1, 3),
                residual_xyz,
            ],
            dim=0,
        )
        color = torch.cat(
            [
                self.expert_color[:, 0, :][:, None, :]
                .expand(-1, shell_samples, -1)
                .reshape(-1, 3),
                self.expert_color[:, 1, :][:, None, :]
                .expand(-1, strand_samples, -1)
                .reshape(-1, 3),
                (
                    self.color
                    if getattr(self, "freeze_residual_teacher", False)
                    else self.expert_color[:, 2, :]
                ),
            ],
            dim=0,
        )
        opacity = torch.cat(
            [shell_opacity.reshape(-1), strand_opacity.reshape(-1), residual_opacity],
            dim=0,
        )
        scaling = torch.cat(
            [
                shell_scaling.reshape(-1, 3),
                strand_scaling.reshape(-1, 3),
                self.residual_scaling,
            ],
            dim=0,
        ).clamp_min(self.positive_eps)
        rotation = torch.cat(
            [
                shell_rotation,
                strand_rotation,
                residual_rotation,
            ],
            dim=0,
        )
        route_id = torch.cat(
            [
                torch.zeros(self.point_count * shell_samples, dtype=torch.long, device=root.device),
                torch.ones(self.point_count * strand_samples, dtype=torch.long, device=root.device),
                torch.full((self.point_count,), 2, dtype=torch.long, device=root.device),
            ],
            dim=0,
        )
        expanded_source_id = torch.cat(
            [
                source_id[:, None].expand(-1, shell_samples).reshape(-1),
                source_id[:, None].expand(-1, strand_samples).reshape(-1),
                source_id,
            ],
            dim=0,
        )
        surface_normal = torch.cat(
            [
                normal[:, None, :].expand(-1, shell_samples, -1).reshape(-1, 3),
                normal[:, None, :].expand(-1, strand_samples, -1).reshape(-1, 3),
                normal,
            ],
            dim=0,
        )
        root_tip = torch.cat(
            [
                shell_t[None, :].expand(self.point_count, -1).reshape(-1),
                strand_t[None, :].expand(self.point_count, -1).reshape(-1),
                torch.zeros(self.point_count, dtype=root.dtype, device=root.device),
            ],
            dim=0,
        )
        structure_weight = torch.cat(
            [
                (
                    torch.ones_like(shell_mix)
                    if additive_teacher
                    else shell_mix
                )[:, None].expand(-1, shell_samples).reshape(-1),
                (
                    torch.ones_like(strand_mix)
                    if additive_teacher
                    else strand_mix
                )[:, None].expand(-1, strand_samples).reshape(-1),
                torch.zeros(self.point_count, dtype=root.dtype, device=root.device),
            ],
            dim=0,
        )
        primitive_root_xyz = torch.cat(
            [
                shell_origin[:, None, :]
                .expand(-1, shell_samples, -1)
                .reshape(-1, 3),
                shell_origin[:, None, :]
                .expand(-1, strand_samples, -1)
                .reshape(-1, 3),
                shell_origin,
            ],
            dim=0,
        )
        carrier_probabilities = self.carrier_probabilities(temperature)
        shell_carrier = F.one_hot(
            torch.full(
                (self.point_count * shell_samples,),
                CARRIER_NAMES.index("shell"),
                dtype=torch.long,
                device=root.device,
            ),
            num_classes=len(CARRIER_NAMES),
        ).to(root.dtype)
        strand_carrier = F.one_hot(
            torch.full(
                (self.point_count * strand_samples,),
                CARRIER_NAMES.index("strand"),
                dtype=torch.long,
                device=root.device,
            ),
            num_classes=len(CARRIER_NAMES),
        ).to(root.dtype)
        expanded_carrier_probabilities = torch.cat(
            [shell_carrier, strand_carrier, carrier_probabilities], dim=0
        )
        carrier_root_tip = torch.cat(
            [
                shell_t[None, :].expand(self.point_count, -1).reshape(-1),
                strand_t[None, :].expand(self.point_count, -1).reshape(-1),
                self.carrier_root_tip,
            ],
            dim=0,
        )
        return FiberPrimitives(
            xyz=xyz,
            color=color,
            opacity=opacity,
            scaling=scaling,
            rotation=rotation,
            route_id=route_id,
            source_id=expanded_source_id,
            route_probabilities=probabilities,
            surface_normal=surface_normal,
            root_tip=root_tip,
            structure_weight=structure_weight,
            root_xyz=primitive_root_xyz,
            carrier_probabilities=expanded_carrier_probabilities,
            carrier_root_tip=carrier_root_tip,
        )

    def residual_primitives(
        self,
        surface_vertices: torch.Tensor,
        surface_faces: torch.Tensor,
    ) -> FiberPrimitives:
        """Return the compact residual-only skinned 3DGS representation.

        This path creates exactly one Gaussian per source point. It avoids
        sending zero-opacity shell/strand primitives to the rasterizer and is
        therefore both a clean ablation and a useful compute control.
        """

        root, tangent, bitangent, normal = self.surface_frame(
            surface_vertices, surface_faces
        )
        xyz = root + _local_to_world(
            self.residual_offset_local, tangent, bitangent, normal
        )
        rotation = self.transported_residual_rotation(
            tangent, bitangent, normal
        )
        probabilities = self.route_probabilities(forced_route="residual")
        source_id = torch.arange(
            self.point_count, device=root.device, dtype=torch.long
        )
        return FiberPrimitives(
            xyz=xyz,
            color=self.color,
            opacity=self.opacity,
            scaling=self.residual_scaling,
            rotation=rotation,
            route_id=torch.full(
                (self.point_count,),
                ROUTE_NAMES.index("residual"),
                dtype=torch.long,
                device=root.device,
            ),
            source_id=source_id,
            route_probabilities=probabilities,
            surface_normal=normal,
            root_tip=torch.zeros_like(self.opacity),
            structure_weight=torch.zeros_like(self.opacity),
            root_xyz=xyz,
            carrier_probabilities=F.one_hot(
                torch.full(
                    (self.point_count,),
                    CARRIER_NAMES.index("surface"),
                    dtype=torch.long,
                    device=root.device,
                ),
                num_classes=len(CARRIER_NAMES),
            ).to(root.dtype),
            carrier_root_tip=torch.zeros_like(self.opacity),
        )

    def residual_drift_regularizer(self) -> torch.Tensor:
        scale = self.scene_scale.clamp_min(1e-8)
        return (
            (self.residual_offset_local - self.initial_residual_offset_local)
            .square()
            .sum(dim=-1)
            / scale.square()
        ).mean()

    def regularizers(
        self,
        surface_vertices: torch.Tensor,
        surface_faces: torch.Tensor,
        *,
        temperature: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        probabilities = self.route_probabilities(temperature)
        _root, tangent, bitangent, normal = self.surface_frame(surface_vertices, surface_faces)
        direction = F.normalize(
            _local_to_world(self.direction_local, tangent, bitangent, normal),
            dim=-1,
            eps=1e-8,
        )
        scale = self.scene_scale.clamp_min(1e-8)
        entropy = -torch.sum(
            probabilities * torch.log(probabilities.clamp_min(1e-8)), dim=-1
        ).mean()
        route_prior = torch.sum(
            probabilities
            * (
                torch.log(probabilities.clamp_min(1e-8))
                - torch.log(self.initial_route_probabilities)
            ),
            dim=-1,
        ).mean()
        shell_normal = (
            probabilities[:, 0] * (1.0 - torch.abs(torch.sum(direction * normal, dim=-1)))
        ).mean()
        shell_length = (
            probabilities[:, 0] * (self.shell_length / (0.02 * scale)).square()
        ).mean()
        strand_thinness = (
            probabilities[:, 1]
            * (self.radius / self.strand_length.clamp_min(self.positive_eps)).square()
        ).mean()
        height = (
            (probabilities[:, 0] + probabilities[:, 1]) * self.height / scale
        ).mean()
        bend = (
            probabilities[:, 1]
            * (
                self.bend_local.square().sum(dim=-1)
                + self.bend_cubic_local.square().sum(dim=-1)
            )
        ).mean()
        residual_drift = (
            probabilities[:, 2]
            * (
                (self.residual_offset_local - self.initial_residual_offset_local)
                .square()
                .sum(dim=-1)
                / scale.square()
            )
        ).mean()
        if self.route_neighbor_index.numel() > 0:
            neighbor_probabilities = probabilities[self.route_neighbor_index]
            route_neighbor = (
                probabilities[:, None, :] - neighbor_probabilities
            ).square().sum(dim=-1).mean()
        else:
            route_neighbor = probabilities.new_zeros(())
        carrier_probabilities = self.carrier_probabilities(temperature)
        carrier_entropy = -torch.sum(
            carrier_probabilities
            * torch.log(carrier_probabilities.clamp_min(1e-8)),
            dim=-1,
        ).mean()
        carrier_prior = torch.sum(
            carrier_probabilities
            * (
                torch.log(carrier_probabilities.clamp_min(1e-8))
                - torch.log(self.initial_carrier_probabilities)
            ),
            dim=-1,
        ).mean()
        if self.route_neighbor_index.numel() > 0:
            neighbor_carriers = carrier_probabilities[self.route_neighbor_index]
            carrier_neighbor = (
                carrier_probabilities[:, None, :] - neighbor_carriers
            ).square().sum(dim=-1).mean()
            carrier_tip_neighbor = (
                self.carrier_root_tip[:, None]
                - self.carrier_root_tip[self.route_neighbor_index]
            ).square().mean()
        else:
            carrier_neighbor = probabilities.new_zeros(())
            carrier_tip_neighbor = probabilities.new_zeros(())

        # Geometry-grounded deformation ownership.  Surface carriers prefer
        # points near the carrier surface, while shell/strand carriers prefer
        # a point close to the corresponding root-to-tip centerline.  The
        # renderer route is intentionally absent: photometric ownership and
        # downstream motion ownership need not be identical.
        root, tangent, bitangent, normal = self.surface_frame(
            surface_vertices, surface_faces
        )
        shell_origin = root + self.height[:, None] * normal
        residual_xyz = root + _local_to_world(
            self.residual_offset_local, tangent, bitangent, normal
        )
        carrier_t = self.carrier_root_tip[:, None]
        shell_carrier_xyz = shell_origin + (
            self.shell_length[:, None] * carrier_t * direction
        )
        bend_world = (
            self.bend_local[:, :1] * tangent
            + self.bend_local[:, 1:] * bitangent
        )
        bend_cubic_world = (
            self.bend_cubic_local[:, :1] * tangent
            + self.bend_cubic_local[:, 1:] * bitangent
        )
        strand_carrier_xyz = shell_origin + self.strand_length[:, None] * (
            carrier_t * direction
            + carrier_t.square() * bend_world
            + carrier_t.pow(3) * bend_cubic_world
        )
        surface_distance = self.residual_offset_local[:, 2].square()
        shell_distance = (residual_xyz - shell_carrier_xyz).square().sum(dim=-1)
        strand_distance = (residual_xyz - strand_carrier_xyz).square().sum(dim=-1)
        carrier_distances = torch.stack(
            [surface_distance, shell_distance, strand_distance], dim=-1
        ) / scale.square()
        carrier_attachment = (
            carrier_probabilities * carrier_distances.clamp_max(100.0)
        ).sum(dim=-1).mean()
        carrier_tip_prior = (
            self.carrier_root_tip - self.initial_carrier_root_tip
        ).square().mean()
        carrier_confidence = carrier_probabilities.max(dim=-1).values.mean()
        carrier_structure_mass = (
            carrier_probabilities[:, CARRIER_NAMES.index("shell")]
            + carrier_probabilities[:, CARRIER_NAMES.index("strand")]
        ).mean()
        route_family = probabilities[:, :2].detach()
        route_family = route_family / route_family.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        carrier_family = carrier_probabilities[:, 1:]
        carrier_family = carrier_family / carrier_family.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        carrier_family_alignment = -torch.sum(
            route_family * torch.log(carrier_family.clamp_min(1e-8)), dim=-1
        ).mean()
        route_structure = probabilities[:, :2].detach().sum(dim=-1)
        carrier_structure = carrier_probabilities[:, 1:].sum(dim=-1)
        carrier_structure_floor = F.relu(
            route_structure - carrier_structure
        ).square().mean()
        return {
            "route_entropy": entropy,
            "route_prior": route_prior,
            "route_neighbor": route_neighbor,
            "shell_normal": shell_normal,
            "shell_length": shell_length,
            "strand_thinness": strand_thinness,
            "height": height,
            "bend": bend,
            "structured_delta": self.structured_delta_gain.mean(),
            "structured_opacity": self.structured_opacity_gain.mean(),
            "residual_drift": residual_drift,
            "residual_trust": self.residual_trust.mean(),
            "expert_appearance": self.expert_color_delta.square().mean(),
            "carrier_entropy": carrier_entropy,
            "carrier_prior": carrier_prior,
            "carrier_neighbor": carrier_neighbor,
            "carrier_tip_neighbor": carrier_tip_neighbor,
            "carrier_attachment": carrier_attachment,
            "carrier_tip_prior": carrier_tip_prior,
            "carrier_confidence": carrier_confidence,
            "carrier_structure_mass": carrier_structure_mass,
            "carrier_family_alignment": carrier_family_alignment,
            "carrier_structure_floor": carrier_structure_floor,
        }

    def route_summary(self, temperature: float = 1.0) -> dict[str, float]:
        probabilities = self.route_probabilities(temperature).detach().mean(dim=0).cpu()
        return {name: float(probabilities[index]) for index, name in enumerate(ROUTE_NAMES)}

    def carrier_summary(self, temperature: float = 1.0) -> dict[str, float]:
        probabilities = (
            self.carrier_probabilities(temperature).detach().mean(dim=0).cpu()
        )
        summary = {
            name: float(probabilities[index])
            for index, name in enumerate(CARRIER_NAMES)
        }
        summary["confidence"] = float(
            self.carrier_probabilities(temperature)
            .detach()
            .max(dim=-1)
            .values.mean()
            .cpu()
        )
        return summary


def create_unified_fiber_field(
    gaussian_ply: str | Path,
    rest_vertices: np.ndarray,
    faces: np.ndarray,
    *,
    device: str = "cuda",
    max_points: int = 20_000,
    neighbor_k: int = 0,
    initial_residual_trust: float = 0.95,
    initial_shell_length_scale: float | None = None,
    initial_strand_length_scale: float | None = None,
    initialize_direction_from_normal: bool = False,
    scalp_face_indices: np.ndarray | None = None,
    binding_cache: str | Path | None = None,
) -> UnifiedFiberField:
    device = device if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu"
    cloud = load_gaussian_ply(str(gaussian_ply))
    xyz = np.asarray(cloud.xyz, dtype=np.float32)
    color = np.asarray(cloud.color, dtype=np.float32)
    opacity = (
        np.asarray(cloud.opacity, dtype=np.float32)
        if cloud.opacity is not None
        else np.full((xyz.shape[0],), 0.5, dtype=np.float32)
    )
    scaling = (
        np.asarray(cloud.scaling, dtype=np.float32)
        if cloud.scaling is not None
        else None
    )
    rotation = (
        np.asarray(cloud.rotation, dtype=np.float32)
        if cloud.rotation is not None
        else None
    )
    if max_points > 0 and xyz.shape[0] > max_points:
        indices = np.linspace(0, xyz.shape[0] - 1, max_points).astype(np.int64)
        xyz, color, opacity = xyz[indices], color[indices], opacity[indices]
        if scaling is not None:
            scaling = scaling[indices]
        if rotation is not None:
            rotation = rotation[indices]

    vertices = np.asarray(rest_vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    scene_scale = max(float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))), 1e-5)
    default_radius = scene_scale * 0.0025
    if scaling is None:
        scaling = np.full((xyz.shape[0], 3), default_radius, dtype=np.float32)
    scaling = np.maximum(scaling, scene_scale * 1e-7)
    if rotation is None:
        rotation = np.zeros((xyz.shape[0], 4), dtype=np.float32)
        rotation[:, 0] = 1.0

    binding_faces = faces
    scalp_faces = None
    if scalp_face_indices is not None:
        scalp_faces = np.asarray(scalp_face_indices, dtype=np.int64).reshape(-1)
        if scalp_faces.size == 0:
            raise ValueError("scalp_face_indices must not be empty")
        if scalp_faces.min() < 0 or scalp_faces.max() >= len(faces):
            raise ValueError("scalp_face_indices contains an out-of-range face")
        binding_faces = faces[scalp_faces]
    cache_key = _binding_cache_key(xyz, vertices, faces, scalp_faces)
    binding = None
    cache_path = Path(binding_cache) if binding_cache is not None else None
    if cache_path is not None and cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as payload:
            stored_key = str(payload["cache_key"].reshape(-1)[0])
            if stored_key == cache_key:
                cached_faces = payload["face_index"].astype(np.int64)
                cached_barycentric = payload["barycentric"].astype(np.float32)
                if cached_faces.shape == (xyz.shape[0],) and cached_barycentric.shape == (
                    xyz.shape[0],
                    3,
                ):
                    binding = GaussianSurfaceBinding(
                        face_index=cached_faces,
                        barycentric=cached_barycentric,
                        local_offset=np.zeros_like(xyz, dtype=np.float32),
                    )
    if binding is None:
        binding = bind_gaussians_to_surface(
            xyz,
            vertices,
            binding_faces,
            device=device,
            vertex_k=0,
            pull_to_surface=True,
        )
        if scalp_faces is not None:
            binding.face_index = scalp_faces[binding.face_index]
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                cache_key=np.asarray([cache_key]),
                face_index=np.asarray(binding.face_index, dtype=np.int64),
                barycentric=np.asarray(binding.barycentric, dtype=np.float32),
            )
    triangles = vertices[faces[binding.face_index]]
    roots = (binding.barycentric[..., None] * triangles).sum(axis=1)
    tangent, bitangent, normal = _surface_frames_numpy(triangles)
    rest_surface_frame = np.stack(
        [tangent, bitangent, normal], axis=-1
    ).astype(np.float32)
    offset = xyz - roots
    residual_offset_local = np.stack(
        [
            np.sum(offset * tangent, axis=-1),
            np.sum(offset * bitangent, axis=-1),
            np.sum(offset * normal, axis=-1),
        ],
        axis=-1,
    ).astype(np.float32)

    rotation_matrix = _quaternion_to_matrix_numpy(rotation)
    longest_axis = np.argmax(scaling, axis=-1)
    axis = rotation_matrix[np.arange(xyz.shape[0]), :, longest_axis]
    direction_local = np.stack(
        [
            np.sum(axis * tangent, axis=-1),
            np.sum(axis * bitangent, axis=-1),
            np.sum(axis * normal, axis=-1),
        ],
        axis=-1,
    )
    direction_local[direction_local[:, 2] < 0.0] *= -1.0
    direction_local /= np.maximum(
        np.linalg.norm(direction_local, axis=-1, keepdims=True), 1e-8
    )
    if initialize_direction_from_normal:
        # The local z-axis is the outward rest-surface normal.  This is a
        # generic scalp/surface prior rather than any strand supervision.
        direction_local = np.zeros_like(direction_local)
        direction_local[:, 2] = 1.0

    sorted_scaling = np.sort(scaling, axis=-1)
    radius = np.maximum(sorted_scaling[:, :2].mean(axis=-1), scene_scale * 2e-5)
    source_length = np.maximum(2.0 * sorted_scaling[:, 2], 2.0 * radius)
    shell_length = np.minimum(source_length, scene_scale * 0.02)
    strand_length = source_length
    if initial_shell_length_scale is not None:
        shell_length = np.full(
            xyz.shape[0],
            max(float(initial_shell_length_scale), 0.0) * scene_scale,
            dtype=np.float32,
        )
    if initial_strand_length_scale is not None:
        strand_length = np.full(
            xyz.shape[0],
            max(float(initial_strand_length_scale), 0.0) * scene_scale,
            dtype=np.float32,
        )
    height = np.maximum(np.abs(residual_offset_local[:, 2]), radius * 0.25)

    anisotropy = np.log(
        np.maximum(sorted_scaling[:, 2] / np.maximum(sorted_scaling[:, 0], 1e-8), 1.0)
    )
    distance_relative = np.linalg.norm(offset, axis=-1) / scene_scale
    length_relative = source_length / scene_scale
    shell_score = 1.25 - 35.0 * distance_relative - 10.0 * length_relative
    strand_score = anisotropy + 15.0 * length_relative - 20.0 * distance_relative - 0.4
    residual_score = 30.0 * distance_relative + 0.25 * (1.0 - anisotropy)
    route_logits = np.stack([shell_score, strand_score, residual_score], axis=-1)
    route_logits -= route_logits.mean(axis=-1, keepdims=True)
    route_logits = np.clip(route_logits, -8.0, 8.0).astype(np.float32)
    direction_world = (
        direction_local[:, :1] * tangent
        + direction_local[:, 1:2] * bitangent
        + direction_local[:, 2:3] * normal
    )
    carrier_root_tip = np.clip(
        np.sum(offset * direction_world, axis=-1)
        / np.maximum(strand_length, scene_scale * 1e-6),
        0.0,
        1.0,
    ).astype(np.float32)
    # Motion ownership uses the opposite off-surface prior to the photometric
    # residual route: a point far from the carrier surface is more likely to
    # be hair that should follow a shell/strand than a free body residual.
    carrier_surface_score = (
        2.0 - 6.0 * carrier_root_tip - 20.0 * distance_relative
    )
    carrier_shell_score = 1.25 - 4.0 * np.abs(carrier_root_tip - 0.25)
    carrier_strand_score = -0.5 + 4.0 * carrier_root_tip + anisotropy
    carrier_logits = np.stack(
        [carrier_surface_score, carrier_shell_score, carrier_strand_score],
        axis=-1,
    )
    carrier_logits -= carrier_logits.mean(axis=-1, keepdims=True)
    carrier_logits = np.clip(carrier_logits, -8.0, 8.0).astype(np.float32)
    route_neighbor_index = _surface_knn_indices(roots, int(neighbor_k))

    tensor = lambda value: torch.as_tensor(value, dtype=torch.float32, device=device)
    return UnifiedFiberField(
        face_index=torch.as_tensor(binding.face_index, dtype=torch.long, device=device),
        barycentric=tensor(binding.barycentric),
        color=tensor(color),
        opacity=tensor(opacity),
        original_scaling=tensor(scaling),
        original_rotation=tensor(rotation),
        rest_surface_frame=tensor(rest_surface_frame),
        residual_offset_local=tensor(residual_offset_local),
        direction_local=tensor(direction_local),
        height=tensor(height),
        shell_length=tensor(shell_length),
        strand_length=tensor(strand_length),
        radius=tensor(radius),
        route_logits=tensor(route_logits),
        carrier_logits=tensor(carrier_logits),
        carrier_root_tip=tensor(carrier_root_tip),
        scene_scale=scene_scale,
        initial_residual_trust=initial_residual_trust,
        route_neighbor_index=torch.as_tensor(
            route_neighbor_index, dtype=torch.long, device=device
        ),
    )


def _binding_cache_key(
    xyz: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    scalp_faces: np.ndarray | None,
) -> str:
    digest = hashlib.sha256()
    for value in (xyz, vertices, faces):
        array = np.ascontiguousarray(value)
        digest.update(str(array.shape).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(array.view(np.uint8))
    if scalp_faces is not None:
        array = np.ascontiguousarray(scalp_faces)
        digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _surface_knn_indices(points: np.ndarray, neighbor_k: int) -> np.ndarray:
    """Return rest-surface KNN indices without constructing an NxN matrix."""

    points = np.asarray(points, dtype=np.float32)
    count = int(points.shape[0])
    k = min(max(int(neighbor_k), 0), max(count - 1, 0))
    if k == 0:
        return np.empty((count, 0), dtype=np.int64)

    from scipy.spatial import cKDTree

    _distance, candidates = cKDTree(points).query(points, k=k + 1, workers=-1)
    candidates = np.asarray(candidates, dtype=np.int64).reshape(count, k + 1)
    neighbors = np.empty((count, k), dtype=np.int64)
    for index, row in enumerate(candidates):
        without_self = row[row != index]
        if without_self.shape[0] < k:
            raise RuntimeError("KNN query did not return enough non-self neighbors")
        neighbors[index] = without_self[:k]
    return neighbors


def render_fiber_primitives(
    primitives: FiberPrimitives,
    camera: PinholeCamera,
    *,
    radius_px: int = 4,
    sigma_scale: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Differentiable integration renderer used for smoke tests and ablations.

    It preserves gradients to position, opacity, radius and routing.  Full
    experiments should use the HairGS/3DGS CUDA rasterizer with the exported
    scaling and quaternion tensors.
    """

    xyz = primitives.xyz
    device, dtype = xyz.device, xyz.dtype
    world_to_camera = torch.as_tensor(camera.world_to_camera, dtype=dtype, device=device)
    homogeneous = torch.cat(
        [xyz, torch.ones((xyz.shape[0], 1), dtype=dtype, device=device)], dim=-1
    )
    camera_xyz = homogeneous @ world_to_camera.T
    z = camera_xyz[:, 2]
    safe_z = z.clamp_min(1e-6)
    x = camera.fx * camera_xyz[:, 0] / safe_z + camera.cx
    y_sign = 1.0 if camera.image_y_down else -1.0
    y = camera.cy + y_sign * camera.fy * camera_xyz[:, 1] / safe_z
    valid = (
        (z > 1e-5)
        & (x >= -radius_px)
        & (x < camera.width + radius_px)
        & (y >= -radius_px)
        & (y < camera.height + radius_px)
        & (primitives.opacity > 1e-8)
    )
    x, y, safe_z = x[valid], y[valid], safe_z[valid]
    color = primitives.color[valid]
    opacity = primitives.opacity[valid]
    transverse_scale = primitives.scaling[valid, 1:].mean(dim=-1)
    sigma_px = (
        float(sigma_scale)
        * 0.5
        * (float(camera.fx) + float(camera.fy))
        * transverse_scale
        / safe_z
    ).clamp(0.35, max(float(radius_px), 0.35))

    if x.numel() == 0:
        rgb = torch.zeros((camera.height, camera.width, 3), dtype=dtype, device=device)
        mask = torch.zeros((camera.height, camera.width), dtype=dtype, device=device)
        return {"rgb": rgb, "mask": mask}

    offsets = torch.arange(-radius_px, radius_px + 1, device=device)
    oy, ox = torch.meshgrid(offsets, offsets, indexing="ij")
    ox = ox.reshape(1, -1)
    oy = oy.reshape(1, -1)
    base_x = torch.round(x.detach()).long().reshape(-1, 1)
    base_y = torch.round(y.detach()).long().reshape(-1, 1)
    px = base_x + ox
    py = base_y + oy
    pixel_valid = (px >= 0) & (px < camera.width) & (py >= 0) & (py < camera.height)
    distance_squared = (
        (px.to(dtype) + 0.5 - x[:, None]).square()
        + (py.to(dtype) + 0.5 - y[:, None]).square()
    )
    weights = torch.exp(-0.5 * distance_squared / sigma_px[:, None].square())
    weights = weights * opacity[:, None]
    weights = torch.where(pixel_valid, weights, torch.zeros_like(weights))

    flat_index = (
        py.clamp(0, camera.height - 1) * camera.width
        + px.clamp(0, camera.width - 1)
    ).reshape(-1)
    flat_weights = weights.reshape(-1)
    rgb_sum = torch.zeros((camera.height * camera.width, 3), dtype=dtype, device=device)
    weight_sum = torch.zeros((camera.height * camera.width, 1), dtype=dtype, device=device)
    rgb_sum.index_add_(
        0,
        flat_index,
        (weights[..., None] * color[:, None, :]).reshape(-1, 3),
    )
    weight_sum.index_add_(0, flat_index, flat_weights[:, None])
    rgb = rgb_sum / weight_sum.clamp_min(1e-6)
    mask = 1.0 - torch.exp(-weight_sum[:, 0])
    return {
        "rgb": rgb.reshape(camera.height, camera.width, 3).clamp(0.0, 1.0),
        "mask": mask.reshape(camera.height, camera.width).clamp(0.0, 1.0),
    }


def fin_grazing_gate(
    xyz: torch.Tensor,
    surface_normal: torch.Tensor,
    camera: PinholeCamera,
    *,
    threshold: float,
    softness: float,
) -> torch.Tensor:
    """Return a soft Fin activation for surface points near the silhouette.

    UnityFurURP emits a fin only when ``abs(dot(view, face_normal))`` is below
    a threshold.  The sigmoid used here is the differentiable analogue: zero
    is front-facing and one is grazing.  Geometry remains world-space; only
    its renderer-side opacity is view conditioned.
    """

    if xyz.shape != surface_normal.shape or xyz.shape[-1] != 3:
        raise ValueError("xyz and surface_normal must both have shape (N, 3)")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("Fin grazing threshold must be in [0, 1]")
    if float(softness) <= 0.0:
        raise ValueError("Fin grazing softness must be positive")
    world_to_camera = torch.as_tensor(
        camera.world_to_camera, dtype=xyz.dtype, device=xyz.device
    )
    camera_center = torch.linalg.inv(world_to_camera)[:3, 3]
    view_direction = F.normalize(
        camera_center[None, :] - xyz, dim=-1, eps=1e-8
    )
    normal = F.normalize(surface_normal, dim=-1, eps=1e-8)
    face_view_product = torch.abs(
        torch.sum(view_direction * normal, dim=-1)
    )
    return torch.sigmoid(
        (float(threshold) - face_view_product) / float(softness)
    )


def apply_fin_view_gate(
    primitives: FiberPrimitives,
    camera: PinholeCamera,
    *,
    strength: float,
    threshold: float,
    softness: float,
) -> FiberPrimitives:
    """Apply Fin visibility to the shell expert without changing route mass.

    ``structure_weight`` is the same zero-initialized continuation factor used
    to unfold shell geometry.  Therefore enabling Fin settings cannot change
    the residual-teacher render at initialization.  At full activation, shell
    opacity is concentrated at grazing views while residual and strand opacity
    remain untouched.
    """

    strength = float(strength)
    if not 0.0 <= strength <= 1.0:
        raise ValueError("Fin gate strength must be in [0, 1]")
    if strength == 0.0:
        return primitives
    if primitives.surface_normal is None or primitives.structure_weight is None:
        raise ValueError(
            "Fin gating requires surface_normal and structure_weight metadata"
        )
    gate = fin_grazing_gate(
        primitives.xyz,
        primitives.surface_normal,
        camera,
        threshold=threshold,
        softness=softness,
    )
    shell = (
        primitives.route_id == ROUTE_NAMES.index("shell")
    ).to(primitives.opacity.dtype)
    activation = (
        shell
        * primitives.structure_weight.to(
            device=primitives.opacity.device, dtype=primitives.opacity.dtype
        ).clamp(0.0, 1.0)
        * strength
    )
    multiplier = 1.0 - activation * (1.0 - gate)
    return replace(primitives, opacity=primitives.opacity * multiplier)


def edit_structured_fibers(
    primitives: FiberPrimitives,
    *,
    length_scale: float = 1.0,
    wind_displacement: torch.Tensor | None = None,
    wind_power: float = 2.0,
) -> FiberPrimitives:
    """Apply root-preserving semantic edits to shell/strand primitives.

    Residual Gaussians are byte-for-byte unchanged.  Length edits scale the
    structured root-relative offset; wind displacement grows root-to-tip as
    ``t**wind_power``.  This is a downstream capability of the distribution,
    not a reconstruction-time photometric degree of freedom.
    """

    if primitives.root_xyz is None or primitives.root_tip is None:
        raise ValueError("Structured editing requires root_xyz and root_tip metadata")
    length_scale = float(length_scale)
    if length_scale <= 0.0:
        raise ValueError("length_scale must be positive")
    if float(wind_power) <= 0.0:
        raise ValueError("wind_power must be positive")
    structured = primitives.route_id != ROUTE_NAMES.index("residual")
    structured_weight = structured.to(primitives.xyz.dtype)[:, None]
    relative = primitives.xyz - primitives.root_xyz
    edited_xyz = primitives.xyz + structured_weight * (
        (length_scale - 1.0) * relative
    )
    if wind_displacement is not None:
        wind = torch.as_tensor(
            wind_displacement,
            dtype=primitives.xyz.dtype,
            device=primitives.xyz.device,
        ).reshape(-1)
        if wind.numel() != 3:
            raise ValueError("wind_displacement must contain three values")
        tip_weight = primitives.root_tip.clamp(0.0, 1.0).pow(float(wind_power))
        edited_xyz = edited_xyz + structured_weight * tip_weight[:, None] * wind
    edited_scaling = primitives.scaling.clone()
    edited_scaling[structured, 0] = (
        edited_scaling[structured, 0] * length_scale
    )
    return replace(primitives, xyz=edited_xyz, scaling=edited_scaling)


def deform_simulation_asset(
    primitives: FiberPrimitives,
    *,
    length_scale: float = 1.0,
    wind_displacement: torch.Tensor | None = None,
    wind_power: float = 2.0,
    shell_wind_response: float = 0.45,
    strand_wind_response: float = 1.0,
    hard_carriers: bool = False,
) -> FiberPrimitives:
    """Deform every Gaussian through its learned simulation carrier.

    Unlike :func:`edit_structured_fibers`, this includes residual-rendering
    Gaussians.  A residual can remain photometrically useful while following
    the surface, shell or strand motion selected by ``carrier_probabilities``.
    Surface-bound mass receives no fiber edit because its body/scalp motion is
    already supplied when ``primitives`` is built from deformed vertices.
    """

    if (
        primitives.root_xyz is None
        or primitives.carrier_probabilities is None
        or primitives.carrier_root_tip is None
    ):
        raise ValueError(
            "Simulation deformation requires root and carrier metadata"
        )
    if float(length_scale) <= 0.0:
        raise ValueError("length_scale must be positive")
    if float(wind_power) <= 0.0:
        raise ValueError("wind_power must be positive")
    if float(shell_wind_response) < 0.0 or float(strand_wind_response) < 0.0:
        raise ValueError("wind responses must be non-negative")

    carriers = primitives.carrier_probabilities.to(
        device=primitives.xyz.device, dtype=primitives.xyz.dtype
    )
    if hard_carriers:
        carrier_ids = carriers.argmax(dim=-1)
        carriers = F.one_hot(
            carrier_ids, num_classes=len(CARRIER_NAMES)
        ).to(primitives.xyz.dtype)
    shell_mass = carriers[:, CARRIER_NAMES.index("shell")]
    strand_mass = carriers[:, CARRIER_NAMES.index("strand")]
    fiber_mass = (shell_mass + strand_mass).clamp(0.0, 1.0)
    relative = primitives.xyz - primitives.root_xyz
    edited_xyz = primitives.xyz + (
        (float(length_scale) - 1.0) * fiber_mass[:, None] * relative
    )
    if wind_displacement is not None:
        wind = torch.as_tensor(
            wind_displacement,
            dtype=primitives.xyz.dtype,
            device=primitives.xyz.device,
        ).reshape(-1)
        if wind.numel() != 3:
            raise ValueError("wind_displacement must contain three values")
        response = (
            float(shell_wind_response) * shell_mass
            + float(strand_wind_response) * strand_mass
        )
        tip = primitives.carrier_root_tip.clamp(0.0, 1.0).pow(
            float(wind_power)
        )
        edited_xyz = edited_xyz + (response * tip)[:, None] * wind

    edited_scaling = primitives.scaling.clone()
    edited_scaling[:, 0] = edited_scaling[:, 0] * (
        1.0 + fiber_mass * (float(length_scale) - 1.0)
    )
    return replace(primitives, xyz=edited_xyz, scaling=edited_scaling)


def simulation_asset_summary(primitives: FiberPrimitives) -> dict[str, float]:
    """Return deformation-ownership diagnostics for an exported asset."""

    if primitives.carrier_probabilities is None:
        raise ValueError("Carrier probabilities are unavailable")
    probabilities = primitives.carrier_probabilities.detach()
    opacity = primitives.opacity.detach().clamp_min(0.0)
    weights = opacity / opacity.sum().clamp_min(1e-8)
    mean = torch.sum(weights[:, None] * probabilities, dim=0)
    summary = {
        name: float(mean[index].cpu())
        for index, name in enumerate(CARRIER_NAMES)
    }
    summary["confidence"] = float(
        torch.sum(weights * probabilities.max(dim=-1).values).cpu()
    )
    residual = primitives.route_id == ROUTE_NAMES.index("residual")
    if torch.any(residual):
        residual_probabilities = probabilities[residual]
        residual_opacity = opacity[residual]
        residual_weights = residual_opacity / residual_opacity.sum().clamp_min(1e-8)
        summary["residual_fiber_bound"] = float(
            torch.sum(
                residual_weights
                * residual_probabilities[:, CARRIER_NAMES.index("shell") :]
                .sum(dim=-1)
            ).cpu()
        )
    else:
        summary["residual_fiber_bound"] = 0.0
    return summary


def _local_to_world(
    value: torch.Tensor,
    tangent: torch.Tensor,
    bitangent: torch.Tensor,
    normal: torch.Tensor,
) -> torch.Tensor:
    return (
        value[:, :1] * tangent
        + value[:, 1:2] * bitangent
        + value[:, 2:3] * normal
    )


def _split_opacity(opacity: torch.Tensor, samples: int) -> torch.Tensor:
    return 1.0 - torch.pow((1.0 - opacity).clamp_min(1e-6), 1.0 / float(samples))


def _inverse_softplus(value: torch.Tensor, eps: float) -> torch.Tensor:
    value = (value - eps).clamp_min(1e-8)
    return value + torch.log(-torch.expm1(-value))


def _normalize_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    return quaternion / torch.linalg.norm(quaternion, dim=-1, keepdim=True).clamp_min(1e-8)


def _quaternion_nlerp(
    start: torch.Tensor, target: torch.Tensor, blend: float | torch.Tensor
) -> torch.Tensor:
    """Shortest-path normalized quaternion interpolation."""

    aligned_target = torch.where(
        torch.sum(start * target, dim=-1, keepdim=True) < 0.0,
        -target,
        target,
    )
    if isinstance(blend, torch.Tensor):
        weight = blend.to(device=start.device, dtype=start.dtype)
    else:
        weight = float(blend)
    return _normalize_quaternion(torch.lerp(start, aligned_target, weight))


def _quaternion_to_matrix_torch(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert normalized wxyz quaternions to differentiable matrices."""

    quaternion = _normalize_quaternion(quaternion)
    w, x, y, z = quaternion.unbind(dim=-1)
    return torch.stack(
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))


def _matrix_to_quaternion_torch(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to wxyz quaternions without CPU detours."""

    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Expected (..., 3, 3) matrices, got {matrix.shape}")
    m00, m01, m02 = matrix[..., 0, :].unbind(dim=-1)
    m10, m11, m12 = matrix[..., 1, :].unbind(dim=-1)
    m20, m21, m22 = matrix[..., 2, :].unbind(dim=-1)
    q_abs = torch.sqrt(
        torch.clamp(
            torch.stack(
                [
                    1.0 + m00 + m11 + m22,
                    1.0 + m00 - m11 - m22,
                    1.0 - m00 + m11 - m22,
                    1.0 - m00 - m11 + m22,
                ],
                dim=-1,
            ),
            # The non-selected quaternion candidates are exactly zero for
            # common rotations such as identity.  A literal sqrt(0) has an
            # infinite derivative and can poison backward through the gather
            # with 0 * inf.  Clamping gives those inactive branches a finite,
            # zero gradient while leaving the selected branch unchanged.
            min=1e-12,
        )
    )
    quaternion_candidates = torch.stack(
        [
            torch.stack([q_abs[..., 0].square(), m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1].square(), m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2].square(), m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3].square()], dim=-1),
        ],
        dim=-2,
    )
    quaternion_candidates = quaternion_candidates / (
        2.0 * q_abs[..., None].clamp_min(0.1)
    )
    best = q_abs.argmax(dim=-1)
    gather_index = best[..., None, None].expand(
        best.shape + (1, 4)
    )
    quaternion = quaternion_candidates.gather(-2, gather_index).squeeze(-2)
    return _normalize_quaternion(quaternion)


def _quaternion_from_x_axis(direction: torch.Tensor) -> torch.Tensor:
    direction = F.normalize(direction, dim=-1, eps=1e-8)
    cross = torch.stack(
        [torch.zeros_like(direction[:, 0]), -direction[:, 2], direction[:, 1]], dim=-1
    )
    quaternion = torch.cat([(1.0 + direction[:, :1]), cross], dim=-1)
    opposite = torch.linalg.norm(quaternion, dim=-1) < 1e-6
    fallback = torch.zeros_like(quaternion)
    fallback[:, 2] = 1.0
    quaternion = torch.where(opposite[:, None], fallback, quaternion)
    return _normalize_quaternion(quaternion)


def _surface_frames_numpy(
    triangles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edge0 = triangles[:, 1] - triangles[:, 0]
    edge1 = triangles[:, 2] - triangles[:, 0]
    tangent = edge0 / np.maximum(np.linalg.norm(edge0, axis=-1, keepdims=True), 1e-8)
    normal = np.cross(edge0, edge1)
    normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-8)
    bitangent = np.cross(normal, tangent)
    bitangent /= np.maximum(np.linalg.norm(bitangent, axis=-1, keepdims=True), 1e-8)
    return tangent.astype(np.float32), bitangent.astype(np.float32), normal.astype(np.float32)


def _quaternion_to_matrix_numpy(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float32)
    quaternion /= np.maximum(np.linalg.norm(quaternion, axis=-1, keepdims=True), 1e-8)
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    return np.stack(
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(-1, 3, 3)
