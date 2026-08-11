from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .gaussian import bind_gaussians_to_surface, load_gaussian_ply
from .render import PinholeCamera


ROUTE_NAMES = ("shell", "strand", "residual")
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

        color = color.clamp(1e-4, 1.0 - 1e-4)
        opacity = opacity.clamp(1e-4, 1.0 - 1e-4)
        self.color_logits = nn.Parameter(torch.logit(color))
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
        self.height_raw = nn.Parameter(_inverse_softplus(height.reshape(-1, 1), eps))
        self.shell_length_raw = nn.Parameter(_inverse_softplus(shell_length.reshape(-1, 1), eps))
        self.strand_length_raw = nn.Parameter(_inverse_softplus(strand_length.reshape(-1, 1), eps))
        self.radius_raw = nn.Parameter(_inverse_softplus(radius.reshape(-1, 1), eps))
        self.route_logits = nn.Parameter(route_logits.float())
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
    ) -> FiberPrimitives:
        if shell_samples < 1 or strand_samples < 1:
            raise ValueError("shell_samples and strand_samples must be positive")
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
        shell_scaling = torch.stack(
            [
                shell_axis_scale.expand(-1, shell_samples),
                self.radius[:, None].expand(-1, shell_samples),
                self.radius[:, None].expand(-1, shell_samples),
            ],
            dim=-1,
        )

        strand_t = (
            torch.arange(strand_samples, device=root.device, dtype=root.dtype) + 0.5
        ) / float(strand_samples)
        bend_world = (
            self.bend_local[:, :1] * tangent + self.bend_local[:, 1:] * bitangent
        )
        strand_xyz = shell_origin[:, None, :] + (
            self.strand_length[:, None, None]
            * (
                strand_t[None, :, None] * direction[:, None, :]
                + strand_t[None, :, None].square() * bend_world[:, None, :]
            )
        )
        strand_direction = F.normalize(
            direction[:, None, :]
            + 2.0 * strand_t[None, :, None] * bend_world[:, None, :],
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
        # At the start of structured routing, every shell/strand expert is a
        # split copy of its residual Gaussian.  This makes expert substitution
        # approximately render-preserving.  Geometry then unfolds toward the
        # shell/strand target after routing has begun, avoiding the abrupt loss
        # spike caused by moving mass to untrained primitives in one step.
        geometry_mix = min(max(float(geometry_blend), 0.0), 1.0)
        shell_rotation = _quaternion_from_x_axis(shell_direction.reshape(-1, 3))
        strand_rotation = _quaternion_from_x_axis(strand_direction.reshape(-1, 3))
        if geometry_mix < 1.0:
            shell_teacher_xyz = residual_xyz[:, None, :].expand(
                -1, shell_samples, -1
            )
            strand_teacher_xyz = residual_xyz[:, None, :].expand(
                -1, strand_samples, -1
            )
            shell_xyz = torch.lerp(shell_teacher_xyz, shell_xyz, geometry_mix)
            strand_xyz = torch.lerp(strand_teacher_xyz, strand_xyz, geometry_mix)
            shell_teacher_scaling = self.residual_scaling[:, None, :].expand(
                -1, shell_samples, -1
            )
            strand_teacher_scaling = self.residual_scaling[:, None, :].expand(
                -1, strand_samples, -1
            )
            shell_scaling = torch.lerp(
                shell_teacher_scaling, shell_scaling, geometry_mix
            )
            strand_scaling = torch.lerp(
                strand_teacher_scaling, strand_scaling, geometry_mix
            )
            shell_teacher_rotation = residual_rotation[:, None, :].expand(
                -1, shell_samples, -1
            ).reshape(-1, 4)
            strand_teacher_rotation = residual_rotation[:, None, :].expand(
                -1, strand_samples, -1
            ).reshape(-1, 4)
            shell_rotation = _quaternion_nlerp(
                shell_teacher_rotation, shell_rotation, geometry_mix
            )
            strand_rotation = _quaternion_nlerp(
                strand_teacher_rotation, strand_rotation, geometry_mix
            )
        shell_opacity = _split_opacity(
            self.opacity * probabilities[:, 0], shell_samples
        )[:, None].expand(-1, shell_samples)
        strand_opacity = _split_opacity(
            self.opacity * probabilities[:, 1], strand_samples
        )[:, None].expand(-1, strand_samples)
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
                self.color[:, None, :].expand(-1, shell_samples, -1).reshape(-1, 3),
                self.color[:, None, :].expand(-1, strand_samples, -1).reshape(-1, 3),
                self.color,
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
        return FiberPrimitives(
            xyz=xyz,
            color=color,
            opacity=opacity,
            scaling=scaling,
            rotation=rotation,
            route_id=route_id,
            source_id=expanded_source_id,
            route_probabilities=probabilities,
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
        bend = (probabilities[:, 1] * self.bend_local.square().sum(dim=-1)).mean()
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
        return {
            "route_entropy": entropy,
            "route_prior": route_prior,
            "route_neighbor": route_neighbor,
            "shell_normal": shell_normal,
            "shell_length": shell_length,
            "strand_thinness": strand_thinness,
            "height": height,
            "bend": bend,
            "residual_drift": residual_drift,
            "residual_trust": self.residual_trust.mean(),
        }

    def route_summary(self, temperature: float = 1.0) -> dict[str, float]:
        probabilities = self.route_probabilities(temperature).detach().mean(dim=0).cpu()
        return {name: float(probabilities[index]) for index, name in enumerate(ROUTE_NAMES)}


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

    binding = bind_gaussians_to_surface(
        xyz,
        vertices,
        faces,
        device=device,
        vertex_k=0,
        pull_to_surface=True,
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
        scene_scale=scene_scale,
        initial_residual_trust=initial_residual_trust,
        route_neighbor_index=torch.as_tensor(
            route_neighbor_index, dtype=torch.long, device=device
        ),
    )


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
    start: torch.Tensor, target: torch.Tensor, blend: float
) -> torch.Tensor:
    """Shortest-path normalized quaternion interpolation."""

    aligned_target = torch.where(
        torch.sum(start * target, dim=-1, keepdim=True) < 0.0,
        -target,
        target,
    )
    return _normalize_quaternion(torch.lerp(start, aligned_target, float(blend)))


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
