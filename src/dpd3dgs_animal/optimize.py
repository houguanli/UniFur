from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from .config import PipelineConfig
from .gaussian import bind_gaussians_to_surface, load_gaussian_ply
from .render import PinholeCamera, camera_from_stage1_npz, default_camera_for_vertices
from .video import image_size


@dataclass
class OptimizationArtifacts:
    out_dir: Path
    optimized_npz: Path
    losses_json: Path
    joints_npy: Path


class DifferentiableSkeletonTetModel(torch.nn.Module):
    """Torch version of the Stage 1 skeleton -> tet -> surface driver.

    ElasticSimulator is still responsible for tetrahedralizing the mesh. This
    module adds the missing backward path by expressing the driven tet and
    surface state as differentiable functions of per-frame skeleton nodes.
    """

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
        edges = _tet_edges(data["tets"])
        self.register_buffer("tet_edges", torch.as_tensor(edges, dtype=torch.long, device=device))
        self.register_buffer("rest_edge_lengths", _edge_lengths(self.rest_tet_nodes, self.tet_edges).detach())
        self.register_buffer("rest_volumes", _tet_volumes(self.rest_tet_nodes, self.tets).detach())
        self.register_buffer("rest_bone_lengths", _bone_lengths(self.initial_joints, self.parents).detach())

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

    def all_tet_nodes(self) -> torch.Tensor:
        nodes = []
        for frame_index, joints in enumerate(self.joints):
            if self.deformation_mode == "matrix_lbs":
                current = skin_points_by_bone_transforms_torch(
                    self.rest_tet_nodes,
                    self.rest_bone_transforms,
                    self.bone_transforms[frame_index],
                    self.tet_weights,
                )
            elif self.deformation_mode == "dqs":
                current = skin_points_by_bone_dqs_torch(
                        self.rest_tet_nodes,
                        self.rest_joints,
                        joints,
                        self.parents,
                        self.tet_weights,
                )
            elif self.deformation_mode == "lbs":
                current = skin_points_by_bone_lbs_torch(
                        self.rest_tet_nodes,
                        self.rest_joints,
                        joints,
                        self.parents,
                        self.tet_weights,
                )
            else:
                raise ValueError(f"Unknown skinning deformation mode: {self.deformation_mode}")
            nodes.append(current)
        return torch.stack(nodes, dim=0)

    def elastic_losses_for_nodes(self, tet_nodes: torch.Tensor) -> dict[str, torch.Tensor]:
        edges = tet_nodes[self.tet_edges[:, 0]] - tet_nodes[self.tet_edges[:, 1]]
        lengths = torch.linalg.norm(edges, dim=-1).clamp_min(1e-8)
        edge = torch.mean((lengths - self.rest_edge_lengths) ** 2)
        volumes = _tet_volumes(tet_nodes, self.tets)
        volume = torch.mean((volumes - self.rest_volumes) ** 2)
        return {"elastic_edge": edge, "elastic_volume": volume}

    def elastic_losses(self) -> dict[str, torch.Tensor]:
        frame_losses = [
            self.elastic_losses_for_nodes(self.driven_points(frame_index)[0])
            for frame_index in range(self.joints.shape[0])
        ]
        return {
            name: torch.stack([loss[name] for loss in frame_losses]).mean()
            for name in ("elastic_edge", "elastic_volume")
        }

    def skeleton_regularizers(self) -> dict[str, torch.Tensor]:
        bone = torch.mean((_bone_lengths(self.joints, self.parents) - self.rest_bone_lengths) ** 2)
        prior = torch.mean((self.joints - self.initial_joints) ** 2)
        if self.joints.shape[0] > 2:
            accel = self.joints[2:] - 2.0 * self.joints[1:-1] + self.joints[:-2]
            temporal = torch.mean(accel * accel)
        elif self.joints.shape[0] > 1:
            temporal = torch.mean((self.joints[1:] - self.joints[:-1]) ** 2)
        else:
            temporal = self.joints.sum() * 0.0
        return {"bone_length": bone, "temporal": temporal, "mocap_prior": prior}


@dataclass
class TorchGaussianBinding:
    face_index: torch.Tensor
    barycentric: torch.Tensor
    local_offset: torch.Tensor
    vertex_indices: torch.Tensor | None
    vertex_weights: torch.Tensor | None
    rest_position: torch.Tensor | None
    rest_blended_position: torch.Tensor | None
    color: torch.Tensor
    opacity: torch.Tensor


def make_torch_gaussian_binding(
    gaussian_ply: str | Path,
    rest_vertices: np.ndarray,
    faces: np.ndarray,
    device: str = "cuda",
    max_points: int = 120000,
    vertex_k: int = 8,
    pull_to_surface: bool = True,
) -> TorchGaussianBinding:
    device = _resolve_device(device)
    cloud = load_gaussian_ply(str(gaussian_ply))
    xyz = cloud.xyz
    color = cloud.color
    opacity = cloud.opacity if cloud.opacity is not None else np.ones((xyz.shape[0],), dtype=np.float32)
    if xyz.shape[0] > max_points:
        idx = np.linspace(0, xyz.shape[0] - 1, max_points).astype(np.int64)
        xyz = xyz[idx]
        color = color[idx]
        opacity = opacity[idx]
    binding = bind_gaussians_to_surface(
        xyz,
        rest_vertices,
        faces,
        device=device,
        vertex_k=vertex_k,
        pull_to_surface=pull_to_surface,
    )
    return TorchGaussianBinding(
        face_index=torch.as_tensor(binding.face_index, dtype=torch.long, device=device),
        barycentric=_tensor(binding.barycentric, device),
        local_offset=_tensor(binding.local_offset, device),
        vertex_indices=(
            torch.as_tensor(binding.vertex_indices, dtype=torch.long, device=device)
            if binding.vertex_indices is not None
            else None
        ),
        vertex_weights=(
            _tensor(binding.vertex_weights, device) if binding.vertex_weights is not None else None
        ),
        rest_position=(
            _tensor(binding.rest_position, device) if binding.rest_position is not None else None
        ),
        rest_blended_position=(
            _tensor(binding.rest_blended_position, device)
            if binding.rest_blended_position is not None
            else None
        ),
        color=_tensor(color, device),
        opacity=_tensor(opacity, device).clamp(0.0, 1.0),
    )


def gaussian_centers_from_surface(
    binding: TorchGaussianBinding,
    surface_vertices: torch.Tensor,
    surface_faces: torch.Tensor,
) -> torch.Tensor:
    if binding.vertex_indices is not None and binding.vertex_weights is not None:
        gathered = surface_vertices[binding.vertex_indices]
        driven_blend = (binding.vertex_weights[..., None] * gathered).sum(dim=1)
        if binding.rest_position is not None and binding.rest_blended_position is not None:
            return binding.rest_position + driven_blend - binding.rest_blended_position
        return driven_blend + binding.local_offset
    tri = surface_vertices[surface_faces[binding.face_index]]
    return (binding.barycentric[..., None] * tri).sum(dim=1) + binding.local_offset


def soft_splat_render(
    xyz: torch.Tensor,
    color: torch.Tensor,
    opacity: torch.Tensor,
    camera: PinholeCamera,
    sigma_px: float = 1.5,
    radius_px: int = 4,
) -> dict[str, torch.Tensor]:
    device = xyz.device
    dtype = xyz.dtype
    w2c = torch.as_tensor(camera.world_to_camera, dtype=dtype, device=device)
    ones = torch.ones((xyz.shape[0], 1), dtype=dtype, device=device)
    cam = torch.cat([xyz, ones], dim=1) @ w2c.T
    z = cam[:, 2].clamp_min(1e-6)
    x = camera.fx * (cam[:, 0] / z) + camera.cx
    y_sign = 1.0 if camera.image_y_down else -1.0
    y = camera.cy + y_sign * camera.fy * (cam[:, 1] / z)
    valid = (z > 1e-5) & (x >= -radius_px) & (x < camera.width + radius_px) & (y >= -radius_px) & (y < camera.height + radius_px)
    x = x[valid]
    y = y[valid]
    color = color[valid]
    opacity = opacity[valid]

    if x.numel() == 0:
        rgb = torch.zeros((camera.height, camera.width, 3), dtype=dtype, device=device)
        mask = torch.zeros((camera.height, camera.width), dtype=dtype, device=device)
        return {"rgb": rgb, "mask": mask}

    offsets = torch.arange(-radius_px, radius_px + 1, device=device)
    oy, ox = torch.meshgrid(offsets, offsets, indexing="ij")
    ox = ox.reshape(1, -1).to(dtype)
    oy = oy.reshape(1, -1).to(dtype)
    base_x = torch.round(x.detach()).to(torch.long).reshape(-1, 1)
    base_y = torch.round(y.detach()).to(torch.long).reshape(-1, 1)
    px = base_x + ox.to(torch.long)
    py = base_y + oy.to(torch.long)
    pixel_valid = (px >= 0) & (px < camera.width) & (py >= 0) & (py < camera.height)

    center_x = px.to(dtype) + 0.5
    center_y = py.to(dtype) + 0.5
    dist2 = (center_x - x.reshape(-1, 1)) ** 2 + (center_y - y.reshape(-1, 1)) ** 2
    weights = torch.exp(-0.5 * dist2 / max(sigma_px * sigma_px, 1e-6)) * opacity.reshape(-1, 1)
    weights = torch.where(pixel_valid, weights, torch.zeros_like(weights))

    flat = (py.clamp(0, camera.height - 1) * camera.width + px.clamp(0, camera.width - 1)).reshape(-1)
    weights_flat = weights.reshape(-1)
    rgb_sum = torch.zeros((camera.height * camera.width, 3), dtype=dtype, device=device)
    weight_sum = torch.zeros((camera.height * camera.width, 1), dtype=dtype, device=device)
    rgb_values = (weights[..., None] * color[:, None, :]).reshape(-1, 3)
    rgb_sum.index_add_(0, flat, rgb_values)
    weight_sum.index_add_(0, flat, weights_flat[:, None])
    rgb = rgb_sum / weight_sum.clamp_min(1e-6)
    mask = 1.0 - torch.exp(-weight_sum[:, 0])
    return {
        "rgb": rgb.reshape(camera.height, camera.width, 3).clamp(0.0, 1.0),
        "mask": mask.reshape(camera.height, camera.width).clamp(0.0, 1.0),
    }


def differentiable_render_loss(
    pred: dict[str, torch.Tensor],
    gt_rgb: torch.Tensor,
    gt_mask: torch.Tensor,
    color_weight: float,
    mask_weight: float,
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
    total = color_weight * color + mask_weight * mask_loss
    return total, {"color": color, "mask_loss": mask_loss, "mask_soft": mask_soft, "mask_01": mask_01}


def optimize_stage2(
    stage1_npz: str | Path,
    gaussian_ply: str | Path,
    frame_dir: str | Path,
    out_dir: str | Path,
    cfg: PipelineConfig,
    steps: int | None = None,
    lr: float | None = None,
    render_size: tuple[int, int] | None = None,
    max_frames: int | None = None,
) -> OptimizationArtifacts:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(cfg.device)
    model = DifferentiableSkeletonTetModel(stage1_npz, device=device)
    binding = make_torch_gaussian_binding(
        gaussian_ply,
        model.rest_surface_vertices.detach().cpu().numpy(),
        model.surface_faces.detach().cpu().numpy(),
        device=device,
        max_points=cfg.max_render_points,
        vertex_k=cfg.gaussian_binding_k,
        pull_to_surface=cfg.pull_gaussians_to_surface,
    )
    frame_paths = _frame_paths(frame_dir)
    width, height = _resolve_render_size(render_size, frame_paths, stage1_npz)
    camera = camera_from_stage1_npz(stage1_npz, width, height)
    camera_source = "stage1_npz"
    if camera is None:
        camera = default_camera_for_vertices(model.rest_surface_vertices.detach().cpu().numpy(), width, height)
        camera_source = "default"
    n_frames = min(len(frame_paths), model.joints.shape[0])
    if max_frames is not None:
        n_frames = min(n_frames, max_frames)
    if n_frames <= 0:
        raise ValueError(f"No frames available in {frame_dir}")

    gt = [_load_gt_frame_torch(path, width, height, device) for path in frame_paths[:n_frames]]
    opt = torch.optim.Adam([model.joints], lr=float(lr or cfg.optimize_lr))
    total_steps = int(steps or cfg.optimize_steps)
    history: list[dict[str, float]] = []

    for step in range(total_steps):
        opt.zero_grad(set_to_none=True)
        render_total_value = 0.0
        color_total_value = 0.0
        mask_total_value = 0.0
        mask_soft_total_value = 0.0
        hard_mask_total_value = 0.0
        elastic_edge_value = 0.0
        elastic_volume_value = 0.0
        for frame_index in range(n_frames):
            tet_nodes, surface_vertices, _ = model.driven_points(frame_index)
            xyz = gaussian_centers_from_surface(binding, surface_vertices, model.surface_faces)
            pred = soft_splat_render(
                xyz,
                binding.color,
                binding.opacity,
                camera,
                sigma_px=cfg.render_sigma_px,
                radius_px=cfg.render_radius_px,
            )
            loss, parts = differentiable_render_loss(
                pred,
                gt[frame_index]["rgb"],
                gt[frame_index]["mask"],
                cfg.color_loss_weight,
                cfg.mask_loss_weight,
            )
            elastic_frame = model.elastic_losses_for_nodes(tet_nodes)
            frame_objective = (
                loss
                + cfg.elastic_edge_weight * elastic_frame["elastic_edge"]
                + cfg.elastic_volume_weight * elastic_frame["elastic_volume"]
            )
            (frame_objective / n_frames).backward()
            render_total_value += float(loss.detach().cpu()) / n_frames
            color_total_value += float(parts["color"].detach().cpu()) / n_frames
            mask_total_value += float(parts["mask_loss"].detach().cpu()) / n_frames
            mask_soft_total_value += float(parts["mask_soft"].detach().cpu()) / n_frames
            hard_mask_total_value += float(parts["mask_01"].detach().cpu()) / n_frames
            elastic_edge_value += float(elastic_frame["elastic_edge"].detach().cpu()) / n_frames
            elastic_volume_value += float(elastic_frame["elastic_volume"].detach().cpu()) / n_frames

        regs = model.skeleton_regularizers()
        regularizer = (
            cfg.bone_length_weight * regs["bone_length"]
            + cfg.temporal_weight * regs["temporal"]
            + cfg.mocap_prior_weight * regs["mocap_prior"]
        )
        regularizer.backward()
        regularizer_value = (
            cfg.elastic_edge_weight * elastic_edge_value
            + cfg.elastic_volume_weight * elastic_volume_value
            + float(regularizer.detach().cpu())
        )
        total_value = render_total_value + regularizer_value
        grad_norm = float(model.joints.grad.detach().norm().cpu()) if model.joints.grad is not None else 0.0
        opt.step()

        if step == 0 or step == total_steps - 1 or (step + 1) % max(1, total_steps // 10) == 0:
            history.append(
                {
                    "step": step,
                    "total": total_value,
                    "render": render_total_value,
                    "color": color_total_value,
                    "mask_loss": mask_total_value,
                    "mask_soft": mask_soft_total_value,
                    "mask_01": hard_mask_total_value,
                    "elastic_edge": elastic_edge_value,
                    "elastic_volume": elastic_volume_value,
                    "bone_length": float(regs["bone_length"].detach().cpu()),
                    "temporal": float(regs["temporal"].detach().cpu()),
                    "mocap_prior": float(regs["mocap_prior"].detach().cpu()),
                    "grad_norm": grad_norm,
                }
            )

    optimized_npz = out_dir / "stage2_optimized_state.npz"
    joints_np = model.joints.detach().cpu().numpy()
    np.savez_compressed(
        optimized_npz,
        optimized_joints=joints_np,
        initial_joints=model.initial_joints.detach().cpu().numpy(),
        rest_tet_nodes=model.rest_tet_nodes.detach().cpu().numpy(),
        tets=model.tets.detach().cpu().numpy(),
        rest_surface_vertices=model.rest_surface_vertices.detach().cpu().numpy(),
        surface_faces=model.surface_faces.detach().cpu().numpy(),
        surface_node_indices=model.surface_node_indices.detach().cpu().numpy(),
        tet_weights=model.tet_weights.detach().cpu().numpy(),
        surface_weights=model.surface_weights.detach().cpu().numpy(),
    )
    joints_npy = out_dir / "optimized_joints.npy"
    np.save(joints_npy, joints_np)
    losses_json = out_dir / "stage2_losses.json"
    with open(losses_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "steps": total_steps,
                "frames": n_frames,
                "lr": float(lr or cfg.optimize_lr),
                "weights": {
                    "color": cfg.color_loss_weight,
                    "mask": cfg.mask_loss_weight,
                    "elastic_edge": cfg.elastic_edge_weight,
                    "elastic_volume": cfg.elastic_volume_weight,
                    "bone_length": cfg.bone_length_weight,
                    "temporal": cfg.temporal_weight,
                    "mocap_prior": cfg.mocap_prior_weight,
                },
                "gravity": cfg.gravity,
                "camera_source": camera_source,
                "render_size": [width, height],
                "render_resolution_source": "override" if render_size is not None else "input_frame",
                "max_render_points": cfg.max_render_points,
                "gaussian_binding_k": cfg.gaussian_binding_k,
                "pull_gaussians_to_surface": cfg.pull_gaussians_to_surface,
                "history": history,
            },
            f,
            indent=2,
        )
    _save_stage2_previews(out_dir, model, binding, camera, min(n_frames, 4), cfg)
    return OptimizationArtifacts(out_dir, optimized_npz, losses_json, joints_npy)


def _save_stage2_previews(
    out_dir: Path,
    model: DifferentiableSkeletonTetModel,
    binding: TorchGaussianBinding,
    camera: PinholeCamera,
    count: int,
    cfg: PipelineConfig,
) -> None:
    preview_dir = out_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for frame_index in range(count):
            _, surface_vertices, _ = model.driven_points(frame_index)
            xyz = gaussian_centers_from_surface(binding, surface_vertices, model.surface_faces)
            pred = soft_splat_render(
                xyz,
                binding.color,
                binding.opacity,
                camera,
                sigma_px=cfg.render_sigma_px,
                radius_px=cfg.render_radius_px,
            )
            image = (pred["rgb"].detach().cpu().numpy().clip(0.0, 1.0) * 255).astype(np.uint8)
            Image.fromarray(image).save(preview_dir / f"optimized_{frame_index:05d}.png")


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
        return image_size(frame_paths[0])
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


def _tet_edges(tets: np.ndarray) -> np.ndarray:
    edges: set[tuple[int, int]] = set()
    for tet in np.asarray(tets, dtype=np.int64):
        ids = [int(x) for x in tet]
        for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
            a, b = sorted((ids[i], ids[j]))
            edges.add((a, b))
    return np.asarray(sorted(edges), dtype=np.int64)


def _edge_lengths(vertices: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], dim=-1)


def _tet_volumes(vertices: torch.Tensor, tets: torch.Tensor) -> torch.Tensor:
    tet = vertices[tets]
    return torch.det(torch.stack([tet[:, 1] - tet[:, 0], tet[:, 2] - tet[:, 0], tet[:, 3] - tet[:, 0]], dim=-1)) / 6.0


def _batched_tet_volumes(vertices: torch.Tensor, tets: torch.Tensor) -> torch.Tensor:
    tet = vertices[:, tets]
    mat = torch.stack([tet[:, :, 1] - tet[:, :, 0], tet[:, :, 2] - tet[:, :, 0], tet[:, :, 3] - tet[:, :, 0]], dim=-1)
    return torch.linalg.det(mat) / 6.0


def _bone_lengths(joints: torch.Tensor, parents: torch.Tensor) -> torch.Tensor:
    valid = parents >= 0
    child = torch.arange(parents.shape[0], device=parents.device)[valid]
    parent = parents[valid]
    if child.numel() == 0:
        return joints.new_zeros((0,))
    if joints.ndim == 2:
        return torch.linalg.norm(joints[child] - joints[parent], dim=-1)
    return torch.linalg.norm(joints[:, child] - joints[:, parent], dim=-1)
