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
    # Optional per-primitive semantic hair class.  This is deliberately
    # independent of physical opacity: opaque head/body Gaussians must still
    # occlude hair behind the face while contributing zero to the rendered
    # hair mask, matching HairGS' Stage-I compositor.
    semantic_foreground: torch.Tensor | None = None
    # Shared source-level SH state avoids expanding 16 coefficients across
    # every preallocated shell/strand sample. ``source_id`` indexes both.
    source_sh_coefficients: torch.Tensor | None = None
    source_base_color: torch.Tensor | None = None
    # Logical source ids stay stable for routing/topology.  Route-specific SH
    # tables use a separate compact appearance index.
    appearance_source_id: torch.Tensor | None = None


class FixedGaussianBase(nn.Module):
    """Non-optimizable head/body GS used only for joint depth compositing.

    The unified hair field must not spend route, carrier, KNN, or optimizer
    capacity on a known head/body scaffold.  These persistent=False buffers
    are reconstructed from the Stage-I PLY at load time, so checkpoints store
    only the learnable hair field.
    """

    def __init__(self, source: "UnifiedFiberField") -> None:
        super().__init__()

        def fixed(name: str, value: torch.Tensor | None) -> None:
            self.register_buffer(
                name,
                None if value is None else value.detach().clone(),
                persistent=False,
            )

        fixed("face_index", source.face_index)
        fixed("barycentric", source.barycentric)
        fixed("scaling", source.residual_scaling)
        fixed("rotation", source.residual_rotation)
        fixed("rest_surface_frame", source.rest_surface_frame)
        fixed("initial_residual_offset_local", source.initial_residual_offset_local)
        fixed("original_xyz", source.original_xyz)
        fixed("color", source.color)
        fixed("opacity", source.opacity)
        fixed("source_sh_coefficients", source.source_sh_coefficients)
        fixed("scene_scale", source.scene_scale)
        self.residual_max_scale_fraction = float(
            source.residual_max_scale_fraction
        )
        self.has_exact_original_xyz = bool(source.has_exact_original_xyz)

    @property
    def point_count(self) -> int:
        return int(self.color.shape[0])

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
        normal = F.normalize(
            torch.linalg.cross(edge0, edge1, dim=-1), dim=-1, eps=1e-8
        )
        bitangent = F.normalize(
            torch.linalg.cross(normal, tangent, dim=-1), dim=-1, eps=1e-8
        )
        return root, tangent, bitangent, normal

    def primitives(
        self,
        surface_vertices: torch.Tensor,
        surface_faces: torch.Tensor,
    ) -> FiberPrimitives:
        root, tangent, bitangent, normal = self.surface_frame(
            surface_vertices, surface_faces
        )
        reconstructed = root + _local_to_world(
            self.initial_residual_offset_local, tangent, bitangent, normal
        )
        current_frame = torch.stack([tangent, bitangent, normal], dim=-1)
        same_frame = (
            torch.amax(
                torch.abs(current_frame - self.rest_surface_frame), dim=(-2, -1)
            )
            < 1e-6
        ) & bool(self.has_exact_original_xyz)
        xyz = torch.where(same_frame[:, None], self.original_xyz, reconstructed)

        frame_delta = current_frame @ self.rest_surface_frame.transpose(-1, -2)
        transported_matrix = frame_delta @ _quaternion_to_matrix_torch(self.rotation)
        transported_rotation = _matrix_to_quaternion_torch(transported_matrix)
        rotation = torch.where(
            same_frame[:, None], self.rotation, transported_rotation
        )
        source_id = torch.arange(
            self.point_count, device=root.device, dtype=torch.long
        )
        probabilities = torch.zeros(
            (self.point_count, len(ROUTE_NAMES)),
            device=root.device,
            dtype=root.dtype,
        )
        probabilities[:, ROUTE_NAMES.index("residual")] = 1.0
        surface_carrier = F.one_hot(
            torch.full(
                (self.point_count,),
                CARRIER_NAMES.index("surface"),
                device=root.device,
                dtype=torch.long,
            ),
            num_classes=len(CARRIER_NAMES),
        ).to(root.dtype)
        zeros = torch.zeros(self.point_count, device=root.device, dtype=root.dtype)
        scaling = self.scaling
        if self.residual_max_scale_fraction > 0.0:
            maximum = self.scene_scale.to(
                device=scaling.device, dtype=scaling.dtype
            ) * self.residual_max_scale_fraction
            scaling = scaling.clamp_max(maximum)
        return FiberPrimitives(
            xyz=xyz,
            color=self.color,
            opacity=self.opacity,
            scaling=scaling,
            rotation=rotation,
            route_id=torch.full(
                (self.point_count,),
                ROUTE_NAMES.index("residual"),
                device=root.device,
                dtype=torch.long,
            ),
            source_id=source_id,
            route_probabilities=probabilities,
            surface_normal=normal,
            root_tip=zeros,
            structure_weight=zeros,
            root_xyz=xyz,
            carrier_probabilities=surface_carrier,
            carrier_root_tip=zeros,
            semantic_foreground=zeros,
            source_sh_coefficients=self.source_sh_coefficients,
            source_base_color=self.color,
            appearance_source_id=source_id,
        )


def append_fixed_base_primitives(
    hair: FiberPrimitives, base: FiberPrimitives
) -> FiberPrimitives:
    """Append fixed base primitives without changing hair source indexing.

    Hair primitives stay first because topology/visual-hull helpers slice the
    structured sample blocks using ``field.point_count``.  Base source IDs are
    offset only for the renderer's shared SH table.
    """

    hair_source_count = int(hair.route_probabilities.shape[0])
    hair_appearance_count = (
        int(hair.source_base_color.shape[0])
        if hair.source_base_color is not None
        else hair_source_count
    )

    def required(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return torch.cat([left, right], dim=0)

    def optional(
        left: torch.Tensor | None, right: torch.Tensor | None, name: str
    ) -> torch.Tensor | None:
        if left is None and right is None:
            return None
        if left is None or right is None:
            raise ValueError(f"Split compositor requires both {name} tensors")
        return required(left, right)

    def shared_sh_table(
        left: torch.Tensor | None, right: torch.Tensor | None
    ) -> torch.Tensor | None:
        if left is None and right is None:
            return None
        if left is None or right is None:
            raise ValueError(
                "Split compositor requires both source_sh_coefficients tensors"
            )
        coefficient_count = max(int(left.shape[1]), int(right.shape[1]))

        def pad(value: torch.Tensor) -> torch.Tensor:
            if int(value.shape[1]) == coefficient_count:
                return value
            return F.pad(value, (0, 0, 0, coefficient_count - value.shape[1]))

        return required(pad(left), pad(right))

    shared_sh = shared_sh_table(
        hair.source_sh_coefficients, base.source_sh_coefficients
    )
    shared_color = optional(
        hair.source_base_color, base.source_base_color, "source_base_color"
    )
    return FiberPrimitives(
        xyz=required(hair.xyz, base.xyz),
        color=required(hair.color, base.color),
        opacity=required(hair.opacity, base.opacity),
        scaling=required(hair.scaling, base.scaling),
        rotation=required(hair.rotation, base.rotation),
        route_id=required(hair.route_id, base.route_id),
        source_id=required(hair.source_id, base.source_id + hair_source_count),
        # Source-level routing belongs exclusively to the learnable hair field.
        route_probabilities=hair.route_probabilities,
        surface_normal=optional(
            hair.surface_normal, base.surface_normal, "surface_normal"
        ),
        root_tip=optional(hair.root_tip, base.root_tip, "root_tip"),
        structure_weight=optional(
            hair.structure_weight, base.structure_weight, "structure_weight"
        ),
        root_xyz=optional(hair.root_xyz, base.root_xyz, "root_xyz"),
        carrier_probabilities=optional(
            hair.carrier_probabilities,
            base.carrier_probabilities,
            "carrier_probabilities",
        ),
        carrier_root_tip=optional(
            hair.carrier_root_tip, base.carrier_root_tip, "carrier_root_tip"
        ),
        semantic_foreground=optional(
            hair.semantic_foreground,
            base.semantic_foreground,
            "semantic_foreground",
        ),
        source_sh_coefficients=shared_sh,
        source_base_color=shared_color,
        appearance_source_id=required(
            (
                hair.appearance_source_id
                if hair.appearance_source_id is not None
                else hair.source_id
            ),
            (
                base.appearance_source_id
                if base.appearance_source_id is not None
                else base.source_id
            )
            + hair_appearance_count,
        ),
    )


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
        residual_max_scale_fraction: float = 0.0,
        carrier_logits: torch.Tensor | None = None,
        carrier_root_tip: torch.Tensor | None = None,
        initial_residual_trust: float = 0.95,
        route_neighbor_index: torch.Tensor | None = None,
        shell_propagated_direction_weight: float = 1.0,
        root_barycentric_max_delta: float = 0.0,
        expert_sh_max_delta: float = 0.5,
        source_foreground_probability: torch.Tensor | None = None,
        semantic_mask_from_source: bool = False,
        structured_foreground_only: bool = False,
        source_mask_threshold: float = 0.25,
        source_sh_coefficients: torch.Tensor | None = None,
        original_xyz: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        eps = max(float(scene_scale) * 1e-7, 1e-8)
        self.positive_eps = eps
        self.register_buffer("face_index", face_index.long())
        self.register_buffer("barycentric", barycentric.float())
        if not 0.0 <= float(root_barycentric_max_delta) <= 1.0:
            raise ValueError("root_barycentric_max_delta must be in [0, 1]")
        self.root_barycentric_max_delta = float(root_barycentric_max_delta)
        self.barycentric_offset_raw = nn.Parameter(
            torch.zeros(
                (barycentric.shape[0], 2),
                dtype=torch.float32,
                device=barycentric.device,
            )
        )
        self.register_buffer("original_scaling", original_scaling.float())
        self.register_buffer("original_rotation", _normalize_quaternion(original_rotation.float()))
        self.has_exact_original_xyz = original_xyz is not None
        if original_xyz is None:
            original_xyz = torch.zeros(
                (color.shape[0], 3), dtype=torch.float32, device=color.device
            )
        self.register_buffer("original_xyz", original_xyz.float(), persistent=False)
        # Columns are the rest-pose tangent, bitangent and normal.  This is
        # intentionally non-persistent: old checkpoints contain the learned
        # rest-world residual quaternion, while the frame is reconstructed
        # deterministically from the input surface on every load.
        self.register_buffer(
            "rest_surface_frame", rest_surface_frame.float(), persistent=True
        )
        self.register_buffer("initial_residual_offset_local", residual_offset_local.float().clone())
        self.register_buffer("scene_scale", torch.tensor(float(scene_scale), dtype=torch.float32))
        self.fixed_base: FixedGaussianBase | None = None
        if float(residual_max_scale_fraction) < 0.0:
            raise ValueError("residual_max_scale_fraction must be non-negative")
        # This is deliberately runtime policy rather than checkpoint state.
        # It lets an existing Stage-I checkpoint be audited under a stricter
        # covariance envelope without rewriting its learned source PLY.
        self.residual_max_scale_fraction = float(residual_max_scale_fraction)
        if route_neighbor_index is None:
            route_neighbor_index = torch.empty(
                (color.shape[0], 0), dtype=torch.long, device=color.device
            )
        self.register_buffer(
            "route_neighbor_index",
            route_neighbor_index.long(),
            # The exact graph is part of the learned topology. Rebuilding KNN
            # after root optimization can change signed continuity and makes
            # checkpoint audits disagree with training-time regularization.
            persistent=True,
        )
        self.register_buffer(
            "strand_root_occupancy",
            torch.ones(
                color.shape[0], dtype=torch.float32, device=color.device
            ),
        )
        if not 0.0 <= float(shell_propagated_direction_weight) <= 1.0:
            raise ValueError("shell_propagated_direction_weight must be in [0, 1]")
        self.shell_propagated_direction_weight = float(
            shell_propagated_direction_weight
        )
        self.register_buffer(
            "shell_visibility_gate",
            torch.empty(
                (color.shape[0], 0), dtype=torch.float32, device=color.device
            ),
        )
        self.register_buffer(
            "strand_visibility_gate",
            torch.empty(
                (color.shape[0], 0), dtype=torch.float32, device=color.device
            ),
        )
        # A persistent topology mask separates preallocated tensor capacity
        # from active renderer primitives.  Columns follow ROUTE_NAMES.  Old
        # checkpoints default to an all-active field; adaptive runs explicitly
        # start with residual-only capacity and activate structured groups from
        # multi-view evidence.
        self.register_buffer(
            "route_active_gate",
            torch.ones(
                (color.shape[0], len(ROUTE_NAMES)),
                dtype=torch.float32,
                device=color.device,
            ),
        )
        if source_foreground_probability is None:
            source_foreground_probability = torch.ones(
                (color.shape[0],), dtype=torch.float32, device=color.device
            )
        source_foreground_probability = source_foreground_probability.float().reshape(-1)
        if source_foreground_probability.shape[0] != color.shape[0]:
            raise ValueError(
                "source_foreground_probability must contain one value per source"
            )
        if not 0.0 <= float(source_mask_threshold) <= 1.0:
            raise ValueError("source_mask_threshold must be in [0, 1]")
        # These are reconstructed deterministically from the Stage-I PLY, so
        # old checkpoints remain loadable without missing-buffer errors.
        self.register_buffer(
            "source_foreground_probability",
            source_foreground_probability.clamp(0.0, 1.0),
            persistent=False,
        )
        self.register_buffer(
            "source_foreground",
            (
                source_foreground_probability >= float(source_mask_threshold)
            ).float(),
            persistent=False,
        )
        self.semantic_mask_from_source = bool(semantic_mask_from_source)
        self.structured_foreground_only = bool(structured_foreground_only)
        if self.structured_foreground_only:
            with torch.no_grad():
                background = self.source_foreground < 0.5
                self.route_active_gate[background, :2] = 0.0
                self.route_active_gate[background, 2] = 1.0
        if source_sh_coefficients is not None:
            source_sh_coefficients = source_sh_coefficients.float()
            if source_sh_coefficients.ndim != 3 or tuple(
                source_sh_coefficients.shape[::2]
            ) != (color.shape[0], 3):
                raise ValueError(
                    "source_sh_coefficients must have shape [sources, K, 3]"
                )
        self.register_buffer(
            "source_sh_coefficients",
            source_sh_coefficients,
            persistent=False,
        )
        if float(expert_sh_max_delta) < 0.0:
            raise ValueError("expert_sh_max_delta must be non-negative")
        self.expert_sh_max_delta = float(expert_sh_max_delta)
        if source_sh_coefficients is None or source_sh_coefficients.shape[1] <= 1:
            self.register_parameter("expert_sh_delta_raw", None)
        else:
            self.expert_sh_delta_raw = nn.Parameter(
                torch.zeros(
                    (
                        color.shape[0],
                        len(ROUTE_NAMES),
                        source_sh_coefficients.shape[1] - 1,
                        3,
                    ),
                    dtype=torch.float32,
                    device=color.device,
                )
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
    def current_barycentric(self) -> torch.Tensor:
        """Surface-constrained learnable root coordinate on the owning face."""

        if self.root_barycentric_max_delta <= 0.0:
            return self.barycentric
        uv_delta = self.root_barycentric_max_delta * torch.tanh(
            self.barycentric_offset_raw
        )
        delta = torch.cat(
            [uv_delta, -uv_delta.sum(dim=-1, keepdim=True)], dim=-1
        )
        barycentric = (self.barycentric + delta).clamp_min(0.0)
        return barycentric / barycentric.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    @property
    def expert_sh_delta(self) -> torch.Tensor | None:
        if self.expert_sh_delta_raw is None:
            return None
        return self.expert_sh_max_delta * torch.tanh(self.expert_sh_delta_raw)

    def route_source_appearance(
        self,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Return route-major SH/base-color tables indexed by route*N+source."""

        base_color = (
            self.color[None, :, :]
            .expand(len(ROUTE_NAMES), -1, -1)
            .reshape(-1, 3)
        )
        if self.source_sh_coefficients is None:
            return None, base_color
        source = self.source_sh_coefficients[None, :, :, :].expand(
            len(ROUTE_NAMES), -1, -1, -1
        )
        delta = self.expert_sh_delta
        if delta is not None:
            higher = source[:, :, 1:, :] + delta.permute(1, 0, 2, 3)
            source = torch.cat([source[:, :, :1, :], higher], dim=2)
        return source.reshape(-1, source.shape[2], 3), base_color

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

    def shell_direction_world(
        self,
        tangent: torch.Tensor,
        bitangent: torch.Tensor,
        normal: torch.Tensor,
    ) -> torch.Tensor:
        """Blend propagated root flow with the surface-normal shell prior."""

        propagated = F.normalize(
            _local_to_world(self.direction_local, tangent, bitangent, normal),
            dim=-1,
            eps=1e-8,
        )
        weight = self.shell_propagated_direction_weight
        return F.normalize(
            weight * propagated + (1.0 - weight) * normal,
            dim=-1,
            eps=1e-8,
        )

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
        # HairGS uses transverse scales down to about 2e-9 for line splats.
        # A scene-relative epsilon inflated them into visible black/color rods.
        scaling = (self.original_scaling * scale_multiplier).clamp_min(1e-12)
        if self.residual_max_scale_fraction > 0.0:
            maximum = self.scene_scale.to(
                device=scaling.device, dtype=scaling.dtype
            ) * self.residual_max_scale_fraction
            scaling = scaling.clamp_max(maximum)
        return scaling

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
        transported = _matrix_to_quaternion_torch(transported_matrix)
        same_frame = (
            torch.amax(
                torch.abs(current_surface_frame - self.rest_surface_frame),
                dim=(-2, -1),
            )
            < 1e-6
        )
        same_frame = same_frame & bool(self.has_exact_original_xyz)
        # At the rest pose, bypass the frame->matrix->quaternion round trip.
        # Degenerate head-mesh faces do not define an orthonormal frame and
        # previously rotated a few opaque base Gaussians even with zero motion.
        return torch.where(
            same_frame[:, None], self.residual_rotation, transported
        )

    def transported_residual_xyz(
        self,
        root: torch.Tensor,
        tangent: torch.Tensor,
        bitangent: torch.Tensor,
        normal: torch.Tensor,
    ) -> torch.Tensor:
        reconstructed = root + _local_to_world(
            self.residual_offset_local, tangent, bitangent, normal
        )
        current_frame = torch.stack([tangent, bitangent, normal], dim=-1)
        same_frame = (
            torch.amax(
                torch.abs(current_frame - self.rest_surface_frame),
                dim=(-2, -1),
            )
            < 1e-6
        )
        same_frame = same_frame & bool(self.has_exact_original_xyz)
        rest_exact = self.original_xyz + _local_to_world(
            self.residual_offset_local - self.initial_residual_offset_local,
            tangent,
            bitangent,
            normal,
        )
        return torch.where(same_frame[:, None], rest_exact, reconstructed)

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
        # Inactive routes must not consume soft or hard routing mass.  A fully
        # pruned source keeps a numerical residual fallback here, while its
        # opacity is still exactly zeroed below by route_active_gate.
        active = self.route_active_gate.to(
            device=probabilities.device, dtype=probabilities.dtype
        )
        probabilities = probabilities * active
        active_mass = probabilities.sum(dim=-1, keepdim=True)
        fallback = torch.zeros_like(probabilities)
        fallback[:, ROUTE_NAMES.index("residual")] = 1.0
        probabilities = torch.where(
            active_mass > 1e-8,
            probabilities / active_mass.clamp_min(1e-8),
            fallback,
        )
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
        root = (self.current_barycentric[..., None] * triangles).sum(dim=1)
        edge0 = triangles[:, 1] - triangles[:, 0]
        edge1 = triangles[:, 2] - triangles[:, 0]
        tangent = F.normalize(edge0, dim=-1, eps=1e-8)
        normal = F.normalize(torch.linalg.cross(edge0, edge1, dim=-1), dim=-1, eps=1e-8)
        bitangent = F.normalize(torch.linalg.cross(normal, tangent, dim=-1), dim=-1, eps=1e-8)
        return root, tangent, bitangent, normal

    def _strand_target_geometry(
        self,
        root: torch.Tensor,
        tangent: torch.Tensor,
        bitangent: torch.Tensor,
        normal: torch.Tensor,
        strand_samples: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the uncollapsed root-to-tip strand target and tangents.

        Rendering may blend this target back to the residual teacher through
        ``structured_delta_gain``.  Geometry supervision must nevertheless see
        this target directly; otherwise a zero-gain curve is indistinguishable
        from a valid short strand to visual-hull losses.
        """

        if strand_samples < 1:
            raise ValueError("strand_samples must be positive")
        direction = F.normalize(
            _local_to_world(self.direction_local, tangent, bitangent, normal),
            dim=-1,
            eps=1e-8,
        )
        bend_world = (
            self.bend_local[:, :1] * tangent
            + self.bend_local[:, 1:] * bitangent
        )
        bend_cubic_world = (
            self.bend_cubic_local[:, :1] * tangent
            + self.bend_cubic_local[:, 1:] * bitangent
        )
        strand_t = (
            torch.arange(
                strand_samples, device=root.device, dtype=root.dtype
            )
            + 0.5
        ) / float(strand_samples)
        origin = root + self.height[:, None] * normal
        xyz = origin[:, None, :] + self.strand_length[:, None, None] * (
            strand_t[None, :, None] * direction[:, None, :]
            + strand_t[None, :, None].square() * bend_world[:, None, :]
            + strand_t[None, :, None].pow(3) * bend_cubic_world[:, None, :]
        )
        strand_direction = F.normalize(
            direction[:, None, :]
            + 2.0 * strand_t[None, :, None] * bend_world[:, None, :]
            + 3.0
            * strand_t[None, :, None].square()
            * bend_cubic_world[:, None, :],
            dim=-1,
            eps=1e-8,
        )
        return xyz, strand_direction

    def strand_target_geometry(
        self,
        surface_vertices: torch.Tensor,
        surface_faces: torch.Tensor,
        *,
        strand_samples: int = 5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Public lightweight target geometry for multi-view supervision."""

        root, tangent, bitangent, normal = self.surface_frame(
            surface_vertices, surface_faces
        )
        return self._strand_target_geometry(
            root, tangent, bitangent, normal, strand_samples
        )

    def shell_target_geometry(
        self,
        surface_vertices: torch.Tensor,
        surface_faces: torch.Tensor,
        *,
        shell_samples: int = 2,
    ) -> torch.Tensor:
        """Return the fully deployed analytic shell centres for supervision."""

        if shell_samples < 1:
            raise ValueError("shell_samples must be positive")
        root, tangent, bitangent, normal = self.surface_frame(
            surface_vertices, surface_faces
        )
        direction = self.shell_direction_world(tangent, bitangent, normal)
        shell_t = (
            torch.arange(
                shell_samples, device=root.device, dtype=root.dtype
            )
            + 0.5
        ) / float(shell_samples)
        shell_origin = root + self.height[:, None] * normal
        return shell_origin[:, None, :] + (
            self.shell_length[:, None, None]
            * shell_t[None, :, None]
            * direction[:, None, :]
        )

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
        shell_visibility: torch.Tensor | None = None,
        strand_visibility: torch.Tensor | None = None,
        fin_aspect_ratio: float = 1.0,
        additive_teacher: bool = False,
        teacher_opacity_transfer: float = 0.0,
        structured_delta_override: float | torch.Tensor | None = None,
    ) -> FiberPrimitives:
        if shell_samples < 1 or strand_samples < 1:
            raise ValueError("shell_samples and strand_samples must be positive")
        if float(fin_aspect_ratio) < 1.0:
            raise ValueError("fin_aspect_ratio must be at least one")
        if not 0.0 <= float(teacher_opacity_transfer) <= 1.0:
            raise ValueError("teacher_opacity_transfer must be in [0, 1]")
        root, tangent, bitangent, normal = self.surface_frame(surface_vertices, surface_faces)
        shell_root_direction = self.shell_direction_world(
            tangent, bitangent, normal
        )
        probabilities = self.route_probabilities(
            temperature=temperature,
            forced_route=forced_route,
            hard=hard_route,
            route_blend=route_blend,
            hardening=route_hardening,
            dropped_route=dropped_route,
            hard_policy=hard_route_policy,
        )
        if self.structured_foreground_only:
            # Head/body sources form an immutable occluding base.  They remain
            # residual even for forced-route and leave-one-route-out renders;
            # otherwise a shell-only diagnostic would expose hair splats that
            # correctly live behind the face in the full Stage-I composite.
            foreground = self.source_foreground[:, None].to(
                device=probabilities.device, dtype=probabilities.dtype
            )
            base_probabilities = torch.zeros_like(probabilities)
            base_probabilities[:, ROUTE_NAMES.index("residual")] = 1.0
            probabilities = foreground * probabilities + (
                1.0 - foreground
            ) * base_probabilities
        source_id = torch.arange(self.point_count, device=root.device, dtype=torch.long)

        shell_t = (
            torch.arange(shell_samples, device=root.device, dtype=root.dtype) + 0.5
        ) / float(shell_samples)
        shell_origin = root + self.height[:, None] * normal
        shell_xyz = shell_origin[:, None, :] + (
            self.shell_length[:, None, None]
            * shell_t[None, :, None]
            * shell_root_direction[:, None, :]
        )
        shell_direction = shell_root_direction[:, None, :].expand(
            -1, shell_samples, -1
        )
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
        strand_xyz, strand_direction = self._strand_target_geometry(
            root, tangent, bitangent, normal, strand_samples
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

        residual_xyz = self.transported_residual_xyz(
            root, tangent, bitangent, normal
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
        if structured_delta_override is not None:
            override = torch.as_tensor(
                structured_delta_override,
                device=delta_gain.device,
                dtype=delta_gain.dtype,
            )
            if override.numel() == 1:
                override = override.expand_as(delta_gain)
            if tuple(override.shape) != tuple(delta_gain.shape):
                raise ValueError(
                    "structured_delta_override must be scalar or have shape "
                    f"{tuple(delta_gain.shape)}, got {tuple(override.shape)}"
                )
            # Used by the target-geometry auxiliary render: optimize the
            # analytic primitive itself while the primary render remains an
            # exact/near-exact teacher migration. This avoids the old loophole
            # where a "deployment" loss still saw a collapsed learned gain.
            delta_gain = override.clamp(0.0, 1.0)
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
        if additive_teacher:
            shell_source_opacity = self.opacity * probabilities[:, 0]
            strand_source_opacity = self.opacity * probabilities[:, 1]
            # Geometry gain doubles as a zero-initialized structured opacity
            # gate.  The residual teacher remains complete; shell/strand can
            # only add evidence after their geometry begins to unfold.
            opacity_gain = self.structured_opacity_gain
            shell_source_opacity = shell_source_opacity * opacity_gain[:, 0]
            strand_source_opacity = strand_source_opacity * opacity_gain[:, 1]
        # A zero-displacement structured expert must be one exact teacher
        # splat, not several co-located transparent splats whose CUDA sorting
        # can change color compositing.  As geometry unfolds, distribute the
        # source transmittance across samples.  The weights sum to one for
        # every gain, so prod(1 - sample_alpha) == 1 - source_alpha.
        shell_sample_weight = (
            shell_mix[:, None] / float(shell_samples)
        ).expand(-1, shell_samples).clone()
        shell_sample_weight[:, 0] += 1.0 - shell_mix
        strand_sample_weight = (
            strand_mix[:, None] / float(strand_samples)
        ).expand(-1, strand_samples).clone()
        strand_sample_weight[:, 0] += 1.0 - strand_mix
        if additive_teacher:
            shell_opacity = _weighted_split_opacity(
                shell_source_opacity, shell_sample_weight
            )
            strand_opacity = _weighted_split_opacity(
                strand_source_opacity, strand_sample_weight
            )
        else:
            # The router owns *optical thickness*, not alpha itself.  Splitting
            # alpha linearly as ``alpha * p(route)`` makes two co-located soft
            # routes more transparent than their source teacher.  Using route
            # probabilities as transmittance exponents instead preserves
            # ``prod(1 - alpha_route_sample) == 1 - alpha_teacher`` exactly
            # whenever the structured copies are co-located and share color.
            shell_opacity = _weighted_split_opacity(
                self.opacity,
                probabilities[:, 0, None] * shell_sample_weight,
            )
            strand_opacity = _weighted_split_opacity(
                self.opacity,
                probabilities[:, 1, None] * strand_sample_weight,
            )
        if shell_visibility is None and tuple(self.shell_visibility_gate.shape) == (
            self.point_count,
            shell_samples,
        ):
            shell_visibility = self.shell_visibility_gate
        if shell_visibility is not None:
            if tuple(shell_visibility.shape) != (self.point_count, shell_samples):
                raise ValueError(
                    "shell_visibility must have shape "
                    f"({self.point_count}, {shell_samples}), got "
                    f"{tuple(shell_visibility.shape)}"
                )
            shell_visibility = shell_visibility.to(
                device=shell_opacity.device, dtype=shell_opacity.dtype
            )
            # At zero structured displacement these samples are exact split
            # copies of the residual teacher.  A visual-hull decision made on
            # the *target* Fin must therefore not punch holes in that safe
            # initialization.  Fade the hard gate in with the same geometric
            # deployment gain; it becomes exact at full deployment.
            shell_visibility_mix = shell_mix[:, None]
            effective_shell_visibility = (
                1.0 - shell_visibility_mix
                + shell_visibility_mix * shell_visibility
            )
            shell_opacity = shell_opacity * effective_shell_visibility
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
            strand_visibility = strand_visibility.to(
                device=strand_opacity.device, dtype=strand_opacity.dtype
            )
            strand_visibility_mix = strand_mix[:, None]
            effective_strand_visibility = (
                1.0 - strand_visibility_mix
                + strand_visibility_mix * strand_visibility
            )
            strand_opacity = strand_opacity * effective_strand_visibility
        if additive_teacher:
            residual_visible = (
                forced_route in (None, "residual") and dropped_route != "residual"
            )
            structured_fraction = (
                probabilities[:, 0] * self.structured_opacity_gain[:, 0]
                + probabilities[:, 1] * self.structured_opacity_gain[:, 1]
            ).clamp(0.0, 1.0)
            residual_budget = 1.0 - (
                float(teacher_opacity_transfer) * structured_fraction
            )
            residual_opacity = (
                self.opacity * residual_budget
                if residual_visible
                else torch.zeros_like(self.opacity)
            )
            if self.structured_foreground_only:
                background = 1.0 - self.source_foreground.to(
                    device=self.opacity.device, dtype=self.opacity.dtype
                )
                residual_opacity = torch.maximum(
                    residual_opacity, self.opacity * background
                )
        else:
            residual_opacity = _weighted_split_opacity(
                self.opacity, probabilities[:, 2, None]
            )[:, 0]
        active = self.route_active_gate.to(
            device=self.opacity.device, dtype=self.opacity.dtype
        )
        shell_opacity = shell_opacity * active[:, 0, None]
        strand_opacity = strand_opacity * active[:, 1, None]
        residual_opacity = residual_opacity * active[:, 2]
        if self.structured_foreground_only:
            foreground = self.source_foreground.to(
                device=self.opacity.device, dtype=self.opacity.dtype
            )
            shell_opacity = shell_opacity * foreground[:, None]
            strand_opacity = strand_opacity * foreground[:, None]

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
        appearance_source_id = torch.cat(
            [
                source_id[:, None].expand(-1, shell_samples).reshape(-1),
                (source_id + self.point_count)[:, None]
                .expand(-1, strand_samples)
                .reshape(-1),
                source_id + 2 * self.point_count,
            ],
            dim=0,
        )
        route_source_sh, route_source_base_color = self.route_source_appearance()
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
        semantic_foreground = None
        if self.semantic_mask_from_source:
            semantic_source = self.source_foreground.to(
                device=root.device, dtype=root.dtype
            )
            semantic_foreground = torch.cat(
                [
                    semantic_source[:, None]
                    .expand(-1, shell_samples)
                    .reshape(-1),
                    semantic_source[:, None]
                    .expand(-1, strand_samples)
                    .reshape(-1),
                    semantic_source,
                ],
                dim=0,
            )
        primitives = FiberPrimitives(
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
            semantic_foreground=semantic_foreground,
            source_sh_coefficients=route_source_sh,
            source_base_color=route_source_base_color,
            appearance_source_id=appearance_source_id,
        )
        if self.fixed_base is not None:
            primitives = append_fixed_base_primitives(
                primitives,
                self.fixed_base.primitives(surface_vertices, surface_faces),
            )
        return primitives

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
        xyz = self.transported_residual_xyz(root, tangent, bitangent, normal)
        rotation = self.transported_residual_rotation(
            tangent, bitangent, normal
        )
        probabilities = self.route_probabilities(forced_route="residual")
        source_id = torch.arange(
            self.point_count, device=root.device, dtype=torch.long
        )
        primitives = FiberPrimitives(
            xyz=xyz,
            color=self.color,
            opacity=(
                self.opacity
                * self.route_active_gate[:, ROUTE_NAMES.index("residual")].to(
                    device=self.opacity.device, dtype=self.opacity.dtype
                )
            ),
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
            semantic_foreground=(
                self.source_foreground.to(device=root.device, dtype=root.dtype)
                if self.semantic_mask_from_source
                else None
            ),
            source_sh_coefficients=self.source_sh_coefficients,
            source_base_color=self.color,
        )
        if self.fixed_base is not None:
            primitives = append_fixed_base_primitives(
                primitives,
                self.fixed_base.primitives(surface_vertices, surface_faces),
            )
        return primitives

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
        structure_min_deployment_gain: float = 0.0,
        strand_min_deployment_gain: float = 0.0,
        strand_min_deployed_length_scale: float = 0.0,
        strand_coverage_target: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        probabilities = self.route_probabilities(temperature)
        _root, tangent, bitangent, normal = self.surface_frame(surface_vertices, surface_faces)
        strand_direction = F.normalize(
            _local_to_world(self.direction_local, tangent, bitangent, normal),
            dim=-1,
            eps=1e-8,
        )
        shell_direction = self.shell_direction_world(tangent, bitangent, normal)
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
            probabilities[:, 0]
            * (1.0 - torch.abs(torch.sum(shell_direction * normal, dim=-1)))
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
        root_barycentric = (
            self.current_barycentric - self.barycentric
        ).square().sum(dim=-1).mean()
        expert_sh = (
            probabilities.new_zeros(())
            if self.expert_sh_delta is None
            else self.expert_sh_delta.square().mean()
        )
        structure_mass = probabilities[:, :2]
        structure_gain_deficit = F.relu(
            float(structure_min_deployment_gain) - self.structured_delta_gain
        ).square()
        structure_deployment = (
            structure_mass * structure_gain_deficit
        ).sum() / structure_mass.sum().clamp_min(1e-8)
        if self.route_neighbor_index.numel() > 0:
            neighbor_probabilities = probabilities[self.route_neighbor_index]
            route_neighbor = (
                probabilities[:, None, :] - neighbor_probabilities
            ).square().sum(dim=-1).mean()
        else:
            route_neighbor = probabilities.new_zeros(())

        # Shared orientation-field regularization.  Root-to-tip signs are
        # meaningful because every curve is scalp anchored, so opposite
        # neighboring tangents are penalized instead of being treated as an
        # equivalent unoriented image line.
        bend_world = (
            self.bend_local[:, :1] * tangent
            + self.bend_local[:, 1:] * bitangent
        )
        bend_cubic_world = (
            self.bend_cubic_local[:, :1] * tangent
            + self.bend_cubic_local[:, 1:] * bitangent
        )
        tip_direction = F.normalize(
            strand_direction + 2.0 * bend_world + 3.0 * bend_cubic_world,
            dim=-1,
            eps=1e-8,
        )
        strand_probability = probabilities[:, ROUTE_NAMES.index("strand")]
        if self.route_neighbor_index.numel() > 0:
            neighbor = self.route_neighbor_index
            pair_weight = (
                strand_probability[:, None] * strand_probability[neighbor]
            )
            root_dot = torch.sum(
                strand_direction[:, None, :]
                * strand_direction[neighbor],
                dim=-1,
            ).clamp(-1.0, 1.0)
            tip_dot = torch.sum(
                tip_direction[:, None, :] * tip_direction[neighbor], dim=-1
            ).clamp(-1.0, 1.0)
            strand_field = (
                pair_weight * (1.0 - 0.5 * (root_dot + tip_dot))
            ).sum() / pair_weight.sum().clamp_min(1e-8)
        else:
            strand_field = probabilities.new_zeros(())

        # The deployed curve is a linear interpolation from one residual point
        # to the analytic strand.  Its entire arc length is therefore scaled
        # by the learned deployment gain.  Five-point quadrature handles curls
        # without confusing a small endpoint displacement with a collapsed
        # curve.
        arc_t = torch.linspace(
            0.0,
            1.0,
            5,
            device=strand_direction.device,
            dtype=strand_direction.dtype,
        )
        derivative = (
            strand_direction[:, None, :]
            + 2.0 * arc_t[None, :, None] * bend_world[:, None, :]
            + 3.0
            * arc_t[None, :, None].square()
            * bend_cubic_world[:, None, :]
        )
        target_arc_length = self.strand_length * torch.linalg.vector_norm(
            derivative, dim=-1
        ).mean(dim=1)
        deployment_gain = self.structured_delta_gain[:, 1]
        deployed_length_scale = (
            deployment_gain * target_arc_length / scale
        )
        if (
            self.strand_visibility_gate.ndim == 2
            and self.strand_visibility_gate.shape[0] == self.point_count
            and self.strand_visibility_gate.shape[1] > 0
        ):
            support_fraction = self.strand_visibility_gate.mean(dim=1).to(
                device=probabilities.device, dtype=probabilities.dtype
            )
        else:
            support_fraction = torch.ones_like(strand_probability)
        supported_mass = strand_probability * support_fraction.detach()
        gain_deficit = F.relu(
            float(strand_min_deployment_gain) - deployment_gain
        ).square()
        length_deficit = F.relu(
            float(strand_min_deployed_length_scale) - deployed_length_scale
        ).square()
        strand_deployability = (
            supported_mass * (gain_deficit + length_deficit)
        ).sum() / supported_mass.sum().clamp_min(1e-8)
        # Unsupported analytic targets must not retain strand route mass.  The
        # target-geometry visual-hull loss can later move them into support and
        # make that route available again.
        strand_deployability = strand_deployability + (
            strand_probability * (1.0 - support_fraction.detach())
        ).mean()
        strand_effective_coverage = (
            supported_mass * deployed_length_scale.clamp_max(2.0)
        ).mean()
        # A squared hinge made small coverage deficits effectively invisible
        # (for example 0.001**2 even with a weight of 20).  The linear hinge
        # keeps a usable gradient until the declared deployment floor is met.
        strand_coverage_deficit = F.relu(
            float(strand_coverage_target) - strand_effective_coverage
        )
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
            self.shell_length[:, None] * carrier_t * shell_direction
        )
        strand_carrier_xyz = shell_origin + self.strand_length[:, None] * (
            carrier_t * strand_direction
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
            "structure_deployment": structure_deployment,
            "strand_field": strand_field,
            "strand_deployability": strand_deployability,
            "strand_effective_coverage": strand_effective_coverage,
            "strand_coverage_deficit": strand_coverage_deficit,
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
            "expert_sh": expert_sh,
            "root_barycentric": root_barycentric,
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

    def active_topology_summary(
        self, *, shell_samples: int, strand_samples: int
    ) -> dict[str, int]:
        """Return source and effective Gaussian counts after topology gating."""

        active = self.route_active_gate.detach() > 0.5
        shell_sources = int(active[:, ROUTE_NAMES.index("shell")].sum().cpu())
        strand_sources = int(active[:, ROUTE_NAMES.index("strand")].sum().cpu())
        residual_sources = int(active[:, ROUTE_NAMES.index("residual")].sum().cpu())
        return {
            "shell_sources": shell_sources,
            "strand_sources": strand_sources,
            "residual_sources": residual_sources,
            "dead_sources": int((~active.any(dim=1)).sum().cpu()),
            "active_gaussians": (
                shell_sources * int(shell_samples)
                + strand_sources * int(strand_samples)
                + residual_sources
            ),
        }

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
    point_sampling_mode: str = "uniform_index",
    exact_vertex_binding: bool = False,
    binding_mode: str = "closest_surface",
    source_mask_mode: str = "all",
    source_mask_threshold: float = 0.25,
    source_min_opacity: float = 0.0,
    residual_max_scale_fraction: float = 0.0,
    semantic_mask_from_source: bool = False,
    structured_foreground_only: bool = False,
    default_opacity: float = 0.5,
    default_opacity_reference_points: int = 0,
    neighbor_k: int = 0,
    shell_propagated_direction_weight: float = 1.0,
    root_barycentric_max_delta: float = 0.0,
    expert_sh_max_delta: float = 0.5,
    expert_sh_degree: int = 0,
    initial_residual_trust: float = 0.95,
    initial_shell_length_scale: float | None = None,
    initial_strand_length_scale: float | None = None,
    initialize_direction_from_normal: bool = False,
    scalp_face_indices: np.ndarray | None = None,
    binding_cache: str | Path | None = None,
) -> UnifiedFiberField:
    device = device if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu"
    cloud = load_gaussian_ply(str(gaussian_ply))
    if (
        bool(semantic_mask_from_source) or bool(structured_foreground_only)
    ) and cloud.foreground_probability is None:
        raise ValueError(
            "semantic/source-restricted hair routing requires a Stage-I PLY "
            "with a learned 'mask' property"
        )
    xyz = np.asarray(cloud.xyz, dtype=np.float32)
    unfiltered_xyz = xyz
    color = np.asarray(cloud.color, dtype=np.float32)
    uses_default_opacity = cloud.opacity is None
    if not 0.0 < float(default_opacity) < 1.0:
        raise ValueError("default_opacity must be in (0, 1)")
    opacity = (
        np.asarray(cloud.opacity, dtype=np.float32)
        if cloud.opacity is not None
        else np.full((xyz.shape[0],), float(default_opacity), dtype=np.float32)
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
    source_indices = np.arange(xyz.shape[0], dtype=np.int64)
    foreground_probability = (
        np.asarray(cloud.foreground_probability, dtype=np.float32).reshape(-1)
        if cloud.foreground_probability is not None
        else np.ones((xyz.shape[0],), dtype=np.float32)
    )
    sh_coefficients = (
        np.asarray(cloud.sh_coefficients, dtype=np.float32)
        if cloud.sh_coefficients is not None
        else None
    )
    if not 0 <= int(expert_sh_degree) <= 3:
        raise ValueError("expert_sh_degree must be in [0, 3]")
    target_sh_coefficients = (int(expert_sh_degree) + 1) ** 2
    if target_sh_coefficients > 1:
        if sh_coefficients is None:
            sh_coefficients = np.zeros(
                (xyz.shape[0], target_sh_coefficients, 3), dtype=np.float32
            )
            # Convert the RGB initialization to the native 3DGS DC basis.
            sh_coefficients[:, 0, :] = (color - 0.5) / 0.28209479177387814
        elif sh_coefficients.shape[1] < target_sh_coefficients:
            padded = np.zeros(
                (xyz.shape[0], target_sh_coefficients, 3), dtype=np.float32
            )
            padded[:, : sh_coefficients.shape[1], :] = sh_coefficients
            sh_coefficients = padded
    normalized_mask_mode = str(source_mask_mode).lower()
    if normalized_mask_mode not in {"all", "foreground", "background"}:
        raise ValueError(
            "source_mask_mode must be 'all', 'foreground', or 'background'"
        )
    source_filter = np.ones((xyz.shape[0],), dtype=bool)
    if normalized_mask_mode != "all":
        if cloud.foreground_probability is None:
            raise ValueError(
                f"source_mask_mode={normalized_mask_mode!r} requires a PLY "
                "'mask' property"
            )
        if not 0.0 <= float(source_mask_threshold) <= 1.0:
            raise ValueError("source_mask_threshold must be in [0, 1]")
        is_foreground = foreground_probability >= float(source_mask_threshold)
        source_filter &= (
            is_foreground
            if normalized_mask_mode == "foreground"
            else ~is_foreground
        )
    if float(source_min_opacity) > 0.0:
        if not 0.0 <= float(source_min_opacity) < 1.0:
            raise ValueError("source_min_opacity must be in [0, 1)")
        source_filter &= opacity >= float(source_min_opacity)
    if not bool(source_filter.all()):
        if not bool(source_filter.any()):
            raise ValueError("Gaussian source filter removed every point")
        source_indices = source_indices[source_filter]
        xyz, color, opacity = (
            xyz[source_filter],
            color[source_filter],
            opacity[source_filter],
        )
        foreground_probability = foreground_probability[source_filter]
        if sh_coefficients is not None:
            sh_coefficients = sh_coefficients[source_filter]
        if scaling is not None:
            scaling = scaling[source_filter]
        if rotation is not None:
            rotation = rotation[source_filter]
    if max_points > 0 and xyz.shape[0] > max_points:
        indices = _select_gaussian_indices(
            xyz,
            max_points,
            mode=point_sampling_mode,
            opacity=opacity,
            scaling=scaling,
        )
        source_indices = source_indices[indices]
        xyz, color, opacity = xyz[indices], color[indices], opacity[indices]
        foreground_probability = foreground_probability[indices]
        if sh_coefficients is not None:
            sh_coefficients = sh_coefficients[indices]
        if scaling is not None:
            scaling = scaling[indices]
        if rotation is not None:
            rotation = rotation[indices]
    if uses_default_opacity and int(default_opacity_reference_points) > 0:
        # Preserve approximate accumulated transmittance when adaptive
        # capacity changes the number of initially overlapping neutral seeds.
        exponent = float(default_opacity_reference_points) / max(xyz.shape[0], 1)
        effective_opacity = 1.0 - (1.0 - float(default_opacity)) ** exponent
        opacity.fill(effective_opacity)

    vertices = np.asarray(rest_vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    scene_scale = max(float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))), 1e-5)
    default_radius = scene_scale * 0.0025
    if scaling is None:
        scaling = np.full((xyz.shape[0], 3), default_radius, dtype=np.float32)
    # Preserve trained HairGS anisotropy; only reject literal non-positive
    # values instead of applying the shell parameter epsilon to residual GS.
    scaling = np.maximum(scaling, 1e-12)
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
    # Hair datasets commonly provide the neutral head/hair mesh vertices as
    # the initial PLY (same order, same coordinates).  Binding those points by
    # an all-pairs point/triangle search is both wasteful and numerically less
    # stable.  The exact-vertex path is O(V+F), and is especially important
    # when adaptive capacity keeps the complete 40k+ vertex scaffold.
    if (
        binding is None
        and bool(exact_vertex_binding)
        and unfiltered_xyz.shape == vertices.shape
        and np.array_equal(unfiltered_xyz, vertices)
    ):
        binding = _bind_exact_surface_vertices(
            source_indices,
            faces,
            scalp_faces,
            vertices,
        )
    if binding is None:
        normalized_binding_mode = str(binding_mode).lower()
        if normalized_binding_mode == "nearest_vertex":
            binding = _bind_nearest_surface_vertices(
                xyz,
                vertices,
                faces,
                scalp_faces,
            )
        elif normalized_binding_mode == "closest_surface":
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
        else:
            raise ValueError(
                "binding_mode must be 'closest_surface' or 'nearest_vertex'"
            )
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
        original_xyz=tensor(xyz),
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
        residual_max_scale_fraction=residual_max_scale_fraction,
        initial_residual_trust=initial_residual_trust,
        route_neighbor_index=torch.as_tensor(
            route_neighbor_index, dtype=torch.long, device=device
        ),
        shell_propagated_direction_weight=shell_propagated_direction_weight,
        root_barycentric_max_delta=root_barycentric_max_delta,
        expert_sh_max_delta=expert_sh_max_delta,
        source_foreground_probability=tensor(foreground_probability),
        semantic_mask_from_source=semantic_mask_from_source,
        structured_foreground_only=structured_foreground_only,
        source_mask_threshold=source_mask_threshold,
        source_sh_coefficients=(
            tensor(sh_coefficients) if sh_coefficients is not None else None
        ),
    )


def partition_binding_cache(
    binding_cache: str | Path | None, partition: str
) -> Path | None:
    if binding_cache is None:
        return None
    path = Path(binding_cache)
    return path.with_name(f"{path.stem}.{partition}{path.suffix}")


def create_fixed_gaussian_base(
    gaussian_ply: str | Path,
    rest_vertices: np.ndarray,
    faces: np.ndarray,
    *,
    device: str = "cuda",
    point_sampling_mode: str = "uniform_index",
    exact_vertex_binding: bool = False,
    binding_mode: str = "closest_surface",
    source_mask_threshold: float = 0.25,
    source_min_opacity: float = 0.0,
    residual_max_scale_fraction: float = 0.0,
    scalp_face_indices: np.ndarray | None = None,
    binding_cache: str | Path | None = None,
) -> FixedGaussianBase:
    """Load only Stage-I head/body GS as an immutable compositor module."""

    temporary = create_unified_fiber_field(
        gaussian_ply,
        rest_vertices,
        faces,
        device=device,
        max_points=0,
        point_sampling_mode=point_sampling_mode,
        exact_vertex_binding=exact_vertex_binding,
        binding_mode=binding_mode,
        source_mask_mode="background",
        source_mask_threshold=source_mask_threshold,
        source_min_opacity=source_min_opacity,
        residual_max_scale_fraction=residual_max_scale_fraction,
        semantic_mask_from_source=True,
        structured_foreground_only=False,
        neighbor_k=0,
        scalp_face_indices=scalp_face_indices,
        binding_cache=partition_binding_cache(binding_cache, "background"),
    )
    base = FixedGaussianBase(temporary).eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    return base


def attach_fixed_gaussian_base(
    field: UnifiedFiberField,
    gaussian_ply: str | Path,
    rest_vertices: np.ndarray,
    faces: np.ndarray,
    *,
    device: str,
    point_sampling_mode: str,
    exact_vertex_binding: bool,
    binding_mode: str,
    source_mask_threshold: float,
    source_min_opacity: float,
    residual_max_scale_fraction: float,
    scalp_face_indices: np.ndarray | None,
    binding_cache: str | Path | None,
) -> FixedGaussianBase:
    base = create_fixed_gaussian_base(
        gaussian_ply,
        rest_vertices,
        faces,
        device=device,
        point_sampling_mode=point_sampling_mode,
        exact_vertex_binding=exact_vertex_binding,
        binding_mode=binding_mode,
        source_mask_threshold=source_mask_threshold,
        source_min_opacity=source_min_opacity,
        residual_max_scale_fraction=residual_max_scale_fraction,
        scalp_face_indices=scalp_face_indices,
        binding_cache=binding_cache,
    )
    field.fixed_base = base
    return base


def _bind_nearest_surface_vertices(
    points: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    scalp_face_indices: np.ndarray | None = None,
) -> GaussianSurfaceBinding:
    """Bind a dense trained GS to its nearest carrier vertex in O(N log V).

    HairGS Stage-I produces many off-surface Gaussians.  The legacy exact
    point/triangle search is quadratic for this case and is unnecessary: the
    residual local offset reconstructs the original point exactly, while the
    nearest carrier vertex supplies a stable scalp/head frame for deformation.
    """

    from scipy.spatial import cKDTree

    mesh_vertices = np.asarray(vertices, dtype=np.float32)
    mesh_faces = np.asarray(faces, dtype=np.int64)
    query_points = np.asarray(points, dtype=np.float32)
    if scalp_face_indices is None:
        candidate_vertices = np.arange(mesh_vertices.shape[0], dtype=np.int64)
    else:
        scalp_faces = np.asarray(scalp_face_indices, dtype=np.int64).reshape(-1)
        candidate_vertices = np.unique(mesh_faces[scalp_faces].reshape(-1))
    if candidate_vertices.size == 0:
        raise ValueError("No carrier vertices are available for nearest binding")
    tree = cKDTree(mesh_vertices[candidate_vertices])
    _distance, local_indices = tree.query(query_points, k=1, workers=-1)
    nearest_vertices = candidate_vertices[np.asarray(local_indices, dtype=np.int64)]
    binding = _bind_exact_surface_vertices(
        nearest_vertices,
        mesh_faces,
        scalp_face_indices,
        mesh_vertices,
    )
    if binding is None:
        raise RuntimeError("Could not bind nearest carrier vertices to incident faces")
    return binding


def _select_gaussian_indices(
    xyz: np.ndarray,
    max_points: int,
    *,
    mode: str = "uniform_index",
    opacity: np.ndarray | None = None,
    scaling: np.ndarray | None = None,
) -> np.ndarray:
    """Select a deterministic, coverage-preserving subset of a source cloud."""

    points = np.asarray(xyz, dtype=np.float32)
    count = int(points.shape[0])
    budget = int(max_points)
    if budget <= 0 or budget >= count:
        return np.arange(count, dtype=np.int64)
    if budget < 1:
        raise ValueError("max_points must be positive or zero for all points")
    normalized_mode = str(mode).lower()
    if normalized_mode == "uniform_index":
        return np.linspace(0, count - 1, budget).astype(np.int64)
    if normalized_mode != "spatial_morton":
        raise ValueError(
            "point_sampling_mode must be 'uniform_index' or 'spatial_morton'"
        )

    lower = points.min(axis=0)
    extent = np.maximum(points.max(axis=0) - lower, 1e-12)
    quantized = np.clip(
        np.floor((points - lower) / extent * 1023.0), 0.0, 1023.0
    ).astype(np.uint64)
    morton = (
        _morton_part_1by2(quantized[:, 0])
        | (_morton_part_1by2(quantized[:, 1]) << np.uint64(1))
        | (_morton_part_1by2(quantized[:, 2]) << np.uint64(2))
    )
    quality = np.ones((count,), dtype=np.float64)
    if opacity is not None:
        quality *= np.maximum(np.asarray(opacity).reshape(-1), 1e-6)
    if scaling is not None:
        scale = np.maximum(np.asarray(scaling, dtype=np.float64), 1e-12)
        quality *= np.cbrt(np.prod(scale, axis=-1))
    # Morton order supplies spatial locality; descending quality only breaks
    # ties inside a quantized cell.  Systematic positions then retain density
    # while covering the complete 3D extent instead of depending on PLY order.
    order = np.lexsort((np.arange(count), -quality, morton))
    positions = np.floor(
        (np.arange(budget, dtype=np.float64) + 0.5) * count / budget
    ).astype(np.int64)
    return order[positions].astype(np.int64)


def _morton_part_1by2(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.uint64) & np.uint64(0x3FF)
    value = (value | (value << np.uint64(16))) & np.uint64(0x30000FF)
    value = (value | (value << np.uint64(8))) & np.uint64(0x300F00F)
    value = (value | (value << np.uint64(4))) & np.uint64(0x30C30C3)
    value = (value | (value << np.uint64(2))) & np.uint64(0x9249249)
    return value


def _bind_exact_surface_vertices(
    vertex_indices: np.ndarray,
    faces: np.ndarray,
    scalp_face_indices: np.ndarray | None,
    vertices: np.ndarray | None = None,
) -> GaussianSurfaceBinding | None:
    """Bind exact mesh vertices to one incident face without a distance search."""

    mesh_faces = np.asarray(faces, dtype=np.int64)
    if scalp_face_indices is None:
        candidate_global = np.arange(mesh_faces.shape[0], dtype=np.int64)
    else:
        candidate_global = np.asarray(scalp_face_indices, dtype=np.int64)
    candidate_faces = mesh_faces[candidate_global]
    vertex_count = int(mesh_faces.max()) + 1
    incident_face = np.full((vertex_count,), -1, dtype=np.int64)
    incident_corner = np.full((vertex_count,), -1, dtype=np.int64)
    best_area = np.full((vertex_count,), -1.0, dtype=np.float64)
    if vertices is None:
        face_quality = np.ones(candidate_faces.shape[0], dtype=np.float64)
    else:
        mesh_vertices = np.asarray(vertices, dtype=np.float64)
        triangles = mesh_vertices[candidate_faces]
        face_quality = np.linalg.norm(
            np.cross(
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0],
            ),
            axis=-1,
        )
    for local_face, face in enumerate(candidate_faces):
        score = float(face_quality[local_face])
        global_face = int(candidate_global[local_face])
        for corner, vertex in enumerate(face):
            vertex = int(vertex)
            if score > best_area[vertex] or (
                score == best_area[vertex]
                and (incident_face[vertex] < 0 or global_face < incident_face[vertex])
            ):
                best_area[vertex] = score
                incident_face[vertex] = global_face
                incident_corner[vertex] = corner
    selected = np.asarray(vertex_indices, dtype=np.int64)
    if selected.size and (
        selected.min() < 0
        or selected.max() >= vertex_count
        or np.any(incident_face[selected] < 0)
    ):
        return None
    barycentric = np.zeros((selected.shape[0], 3), dtype=np.float32)
    barycentric[np.arange(selected.shape[0]), incident_corner[selected]] = 1.0
    return GaussianSurfaceBinding(
        face_index=incident_face[selected],
        barycentric=barycentric,
        local_offset=np.zeros((selected.shape[0], 3), dtype=np.float32),
    )


def _binding_cache_key(
    xyz: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    scalp_faces: np.ndarray | None,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"stable-nearest-vertex-binding-v2")
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
        return {"rgb": rgb, "mask": mask, "physical_mask": mask}

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
    physical_mask = 1.0 - torch.exp(-weight_sum[:, 0])
    mask = physical_mask
    if primitives.semantic_foreground is not None:
        semantic = primitives.semantic_foreground[valid].to(dtype=dtype)
        semantic_sum = torch.zeros(
            (camera.height * camera.width, 1), dtype=dtype, device=device
        )
        semantic_sum.index_add_(
            0,
            flat_index,
            (weights * semantic[:, None]).reshape(-1, 1),
        )
        semantic_fraction = semantic_sum[:, 0] / weight_sum[:, 0].clamp_min(1e-6)
        mask = physical_mask * semantic_fraction
    return {
        "rgb": rgb.reshape(camera.height, camera.width, 3).clamp(0.0, 1.0),
        "mask": mask.reshape(camera.height, camera.width).clamp(0.0, 1.0),
        "physical_mask": physical_mask.reshape(
            camera.height, camera.width
        ).clamp(0.0, 1.0),
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


def _weighted_split_opacity(
    opacity: torch.Tensor, sample_weight: torch.Tensor
) -> torch.Tensor:
    if sample_weight.ndim != 2 or sample_weight.shape[0] != opacity.shape[0]:
        raise ValueError("sample_weight must have shape [sources, samples]")
    return 1.0 - torch.pow(
        (1.0 - opacity).clamp_min(1e-6)[:, None],
        sample_weight.clamp_min(0.0),
    )


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
