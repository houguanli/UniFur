from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class DifferentiableSurfaceScaffold(torch.nn.Module):
    """Load the protocol surface and evaluate its optional frame deformation."""

    def __init__(self, stage1_npz: str | Path, device: str = "cuda") -> None:
        super().__init__()
        device = _resolve_device(device)
        data = np.load(stage1_npz)
        self.device_name = device
        self.register_buffer("rest_tet_nodes", _tensor(data["rest_tet_nodes"], device))
        self.register_buffer("tets", torch.as_tensor(data["tets"], dtype=torch.long, device=device))
        self.register_buffer("rest_surface_vertices", _tensor(data["rest_surface_vertices"], device))
        self.register_buffer("surface_faces", torch.as_tensor(data["surface_faces"], dtype=torch.long, device=device))
        self.register_buffer(
            "surface_node_indices",
            torch.as_tensor(
                data["surface_node_indices"]
                if "surface_node_indices" in data
                else np.arange(data["rest_surface_vertices"].shape[0]),
                dtype=torch.long,
                device=device,
            ),
        )
        self.register_buffer("rest_joints", _tensor(data["skeleton_joints"][0], device))
        self.register_buffer("initial_joints", _tensor(data["skeleton_joints"], device))
        self.register_buffer("parents", torch.as_tensor(data["parents"], dtype=torch.long, device=device))
        self.register_buffer("tet_weights", _tensor(data["tet_weights"], device))
        self.register_buffer("surface_weights", _tensor(data["surface_weights"], device))
        self.deformation_mode = (
            str(np.asarray(data["skinning_deformation_mode"]).reshape(-1)[0])
            if "skinning_deformation_mode" in data
            else "lbs"
        )
        self.has_bone_transforms = "bone_transforms" in data
        if self.has_bone_transforms:
            bone_transforms = _tensor(data["bone_transforms"], device)
            expected = (
                int(self.initial_joints.shape[0]),
                int(self.initial_joints.shape[1]),
                4,
                4,
            )
            if tuple(bone_transforms.shape) != expected:
                raise ValueError(
                    "bone_transforms must have shape "
                    f"{expected}, found {tuple(bone_transforms.shape)}"
                )
            self.register_buffer("rest_bone_transforms", bone_transforms[0].clone())
            self.register_buffer("bone_transforms", bone_transforms)
        elif self.deformation_mode == "matrix_lbs":
            raise ValueError(
                "skinning_deformation_mode='matrix_lbs' requires bone_transforms"
            )
        self.joints = torch.nn.Parameter(self.initial_joints.clone())

    def driven_points(self, frame_index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        joints = self.joints[frame_index]
        if self.deformation_mode == "matrix_lbs":
            tet_nodes = skin_points_by_bone_transforms_torch(
                self.rest_tet_nodes,
                self.rest_bone_transforms,
                self.bone_transforms[frame_index],
                self.tet_weights,
            )
        elif self.deformation_mode == "dqs":
            tet_nodes = skin_points_by_bone_dqs_torch(
                self.rest_tet_nodes,
                self.rest_joints,
                joints,
                self.parents,
                self.tet_weights,
            )
        elif self.deformation_mode == "lbs":
            tet_nodes = skin_points_by_bone_lbs_torch(
                self.rest_tet_nodes,
                self.rest_joints,
                joints,
                self.parents,
                self.tet_weights,
            )
        else:
            raise ValueError(f"Unknown skinning deformation mode: {self.deformation_mode}")
        surface_vertices = tet_nodes[self.surface_node_indices]
        return tet_nodes, surface_vertices, joints


def differentiable_render_loss(
    pred: dict[str, torch.Tensor],
    gt_rgb: torch.Tensor,
    gt_mask: torch.Tensor,
    color_weight: float,
    mask_weight: float,
    mask_boundary_weight: float = 0.0,
    mask_boundary_radius: int = 1,
    mask_balance_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    valid = gt_mask[..., None].clamp(0.0, 1.0)
    color = (torch.abs(pred["rgb"] - gt_rgb) * valid).sum() / (valid.sum() * 3.0).clamp_min(1.0)
    mask_soft = torch.mean(torch.abs(pred["mask"] - gt_mask))
    mask_hard = (pred["mask"] > 0.5).float()
    mask_01 = torch.mean(torch.abs(mask_hard - (gt_mask > 0.5).float()))
    # A straight-through hard threshold gives zero loss and zero gradient for
    # background alpha below 0.5. Broad, faint Gaussian halos can therefore
    # improve foreground color while evading silhouette supervision. Optimize
    # continuous alpha and retain mask_01 as a diagnostic only.
    mask_loss = mask_soft
    target_mask = gt_mask.clamp(0.0, 1.0)
    foreground = (
        torch.abs(pred["mask"] - target_mask) * target_mask
    ).sum() / target_mask.sum().clamp_min(1.0)
    background_mask = 1.0 - target_mask
    background = (
        torch.abs(pred["mask"] - target_mask) * background_mask
    ).sum() / background_mask.sum().clamp_min(1.0)
    mask_balanced = 0.5 * (foreground + background)
    mask_boundary = pred["mask"].new_zeros(())
    if mask_boundary_weight > 0.0:
        radius = max(int(mask_boundary_radius), 0)
        kernel = 2 * radius + 1
        target = gt_mask.clamp(0.0, 1.0)[None, None]
        dilated = F.max_pool2d(target, kernel, stride=1, padding=radius)
        eroded = 1.0 - F.max_pool2d(1.0 - target, kernel, stride=1, padding=radius)
        boundary = (dilated - eroded).squeeze(0).squeeze(0).clamp(0.0, 1.0)
        mask_boundary = (
            torch.abs(pred["mask"] - gt_mask) * boundary
        ).sum() / boundary.sum().clamp_min(1.0)
    total = (
        color_weight * color
        + mask_weight * mask_loss
        + mask_balance_weight * mask_balanced
        + mask_boundary_weight * mask_boundary
    )
    return total, {
        "color": color,
        "mask_loss": mask_loss,
        "mask_soft": mask_soft,
        "mask_01": mask_01,
        "mask_balanced": mask_balanced,
        "mask_foreground": foreground,
        "mask_background": background,
        "mask_boundary": mask_boundary,
    }


def _frame_paths(frame_dir: str | Path) -> list[Path]:
    root = Path(frame_dir)
    if not root.exists():
        raise FileNotFoundError(root)
    return sorted(p for p in root.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})


def _resolve_render_size(
    render_size: tuple[int, int] | None,
    frame_paths: list[Path],
    stage1_npz: str | Path,
) -> tuple[int, int]:
    if render_size is not None:
        return int(render_size[0]), int(render_size[1])
    if frame_paths:
        with Image.open(frame_paths[0]) as image:
            return int(image.width), int(image.height)
    data = np.load(stage1_npz)
    if "native_frame_size" in data:
        size = data["native_frame_size"].astype(np.int32)
        return int(size[0]), int(size[1])
    if "render_size" in data:
        size = data["render_size"].astype(np.int32)
        return int(size[0]), int(size[1])
    raise ValueError("Cannot resolve render size without frames or stage1 size metadata")


def _load_gt_frame_torch(path: Path, width: int, height: int, device: str) -> dict[str, torch.Tensor]:
    bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise FileNotFoundError(path)
    if bgr.shape[1] != width or bgr.shape[0] != height:
        bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
    if bgr.ndim == 3 and bgr.shape[2] == 4:
        alpha = bgr[..., 3].astype(np.float32) / 255.0
        rgb = cv2.cvtColor(bgr[..., :3], cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mask = (alpha > 0.5).astype(np.float32)
    else:
        rgb = cv2.cvtColor(bgr[..., :3], cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        gray = cv2.cvtColor(bgr[..., :3], cv2.COLOR_BGR2GRAY)
        mask = (gray > 5).astype(np.float32)
    return {
        "rgb": torch.as_tensor(rgb, dtype=torch.float32, device=device),
        "mask": torch.as_tensor(mask, dtype=torch.float32, device=device),
    }


def skin_points_by_bone_lbs_torch(
    rest_points: torch.Tensor,
    rest_joints: torch.Tensor,
    posed_joints: torch.Tensor,
    parents: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    joint_count = rest_joints.shape[0]
    safe_parents = parents.clamp(0, joint_count - 1)
    valid_parent = parents >= 0
    rest_anchor = torch.where(
        valid_parent[:, None],
        rest_joints[safe_parents],
        rest_joints,
    )
    posed_anchor = torch.where(
        valid_parent[:, None],
        posed_joints[safe_parents],
        posed_joints,
    )
    rotations = _rotations_between_vectors_torch(
        rest_joints - rest_anchor,
        posed_joints - posed_anchor,
    )
    centered = rest_points[:, None, :] - rest_anchor[None, :, :]
    transformed = torch.einsum("njc,jdc->njd", centered, rotations)
    transformed = transformed + posed_anchor[None, :, :]
    return torch.sum(weights[..., None] * transformed, dim=1)


def skin_points_by_bone_transforms_torch(
    rest_points: torch.Tensor,
    rest_bone_transforms: torch.Tensor,
    posed_bone_transforms: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Apply exact per-bone world transforms with linear blend skinning.

    DFA/Artemis provides full 3x4 bone transforms.  Reconstructing rotations
    only from parent-to-child directions discards axial twist and is therefore
    an avoidable source of dynamic fur error.  Both transform arrays map bone
    coordinates to world coordinates, so ``T_pose @ inv(T_rest)`` maps a
    rest-world point into the posed world frame for that bone.
    """

    if rest_bone_transforms.ndim != 3 or rest_bone_transforms.shape[-2:] != (4, 4):
        raise ValueError("rest_bone_transforms must have shape [J, 4, 4]")
    if posed_bone_transforms.shape != rest_bone_transforms.shape:
        raise ValueError("posed and rest bone transforms must have matching shapes")
    if weights.shape != (rest_points.shape[0], rest_bone_transforms.shape[0]):
        raise ValueError("weights must have shape [N, J]")

    relative = posed_bone_transforms @ torch.linalg.inv(rest_bone_transforms)
    homogeneous = torch.cat(
        [rest_points, torch.ones_like(rest_points[:, :1])],
        dim=-1,
    )
    transformed = torch.einsum("jab,nb->nja", relative, homogeneous)[..., :3]
    return torch.sum(weights[..., None] * transformed, dim=1)


def skin_points_by_bone_dqs_torch(
    rest_points: torch.Tensor,
    rest_joints: torch.Tensor,
    posed_joints: torch.Tensor,
    parents: torch.Tensor,
    weights: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    joint_count = rest_joints.shape[0]
    safe_parents = parents.clamp(0, joint_count - 1)
    valid_parent = parents >= 0
    rest_anchor = torch.where(
        valid_parent[:, None],
        rest_joints[safe_parents],
        rest_joints,
    )
    posed_anchor = torch.where(
        valid_parent[:, None],
        posed_joints[safe_parents],
        posed_joints,
    )
    rotation_quaternion = _quaternions_between_vectors_torch(
        rest_joints - rest_anchor,
        posed_joints - posed_anchor,
    )
    rotations = _quaternion_to_matrix_torch(rotation_quaternion)
    translation = posed_anchor - torch.einsum(
        "jdc,jc->jd",
        rotations,
        rest_anchor,
    )
    translation_quaternion = torch.cat(
        [torch.zeros_like(translation[:, :1]), translation],
        dim=-1,
    )
    dual_quaternion = 0.5 * _quaternion_multiply_torch(
        translation_quaternion,
        rotation_quaternion,
    )
    blended_rotation = weights @ rotation_quaternion
    blended_dual = weights @ dual_quaternion
    norm = torch.linalg.norm(
        blended_rotation,
        dim=-1,
        keepdim=True,
    ).clamp_min(eps)
    blended_rotation = blended_rotation / norm
    blended_dual = blended_dual / norm
    blended_dual = blended_dual - blended_rotation * torch.sum(
        blended_rotation * blended_dual,
        dim=-1,
        keepdim=True,
    )
    rotated = _quaternion_rotate_torch(blended_rotation, rest_points)
    translation_blend = 2.0 * _quaternion_multiply_torch(
        blended_dual,
        _quaternion_conjugate_torch(blended_rotation),
    )[:, 1:]
    return rotated + translation_blend


def _rotations_between_vectors_torch(
    source: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    return _quaternion_to_matrix_torch(
        _quaternions_between_vectors_torch(source, target, eps=eps)
    )


def _quaternions_between_vectors_torch(
    source: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    source_norm = torch.linalg.norm(source, dim=-1, keepdim=True)
    target_norm = torch.linalg.norm(target, dim=-1, keepdim=True)
    valid = (source_norm[:, 0] > eps) & (target_norm[:, 0] > eps)
    a = source / source_norm.clamp_min(eps)
    b = target / target_norm.clamp_min(eps)
    xyz = torch.linalg.cross(a, b, dim=-1)
    dot = torch.sum(a * b, dim=-1, keepdim=True)
    quaternion = torch.cat([1.0 + dot, xyz], dim=-1)

    basis_index = torch.argmin(torch.abs(a), dim=-1)
    basis = torch.eye(3, dtype=a.dtype, device=a.device)[basis_index]
    opposite_axis = torch.linalg.cross(a, basis, dim=-1)
    opposite_axis = opposite_axis / torch.linalg.norm(
        opposite_axis,
        dim=-1,
        keepdim=True,
    ).clamp_min(eps)
    opposite_quaternion = torch.cat(
        [torch.zeros_like(dot), opposite_axis],
        dim=-1,
    )
    opposite = valid & (torch.linalg.norm(quaternion, dim=-1) <= eps)
    quaternion = torch.where(opposite[:, None], opposite_quaternion, quaternion)
    identity_quaternion = torch.zeros_like(quaternion)
    identity_quaternion[:, 0] = 1.0
    quaternion = torch.where(valid[:, None], quaternion, identity_quaternion)
    quaternion = quaternion / torch.linalg.norm(
        quaternion,
        dim=-1,
        keepdim=True,
    ).clamp_min(eps)
    return quaternion


def _quaternion_to_matrix_torch(quaternion: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quaternion.unbind(dim=-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(-1, 3, 3)


def _quaternion_multiply_torch(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(dim=-1)
    bw, bx, by, bz = b.unbind(dim=-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def _quaternion_conjugate_torch(quaternion: torch.Tensor) -> torch.Tensor:
    return torch.cat([quaternion[..., :1], -quaternion[..., 1:]], dim=-1)


def _quaternion_rotate_torch(
    quaternion: torch.Tensor,
    points: torch.Tensor,
) -> torch.Tensor:
    point_quaternion = torch.cat(
        [torch.zeros_like(points[:, :1]), points],
        dim=-1,
    )
    return _quaternion_multiply_torch(
        _quaternion_multiply_torch(quaternion, point_quaternion),
        _quaternion_conjugate_torch(quaternion),
    )[:, 1:]


def _tensor(array: np.ndarray, device: str) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def _resolve_device(device: str) -> str:
    return device if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu"
