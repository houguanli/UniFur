from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .config import PipelineConfig
from .optimize import (
    OptimizationArtifacts,
    _bone_lengths,
    _frame_paths,
    _load_gt_frame_torch,
    _quaternions_between_vectors_torch,
    _quaternion_to_matrix_torch,
    _resolve_device,
    _resolve_render_size,
    _tensor,
    differentiable_render_loss,
    gaussian_centers_from_surface,
    make_torch_gaussian_binding,
    soft_splat_render,
)
from .render import camera_from_stage1_npz, default_camera_for_vertices


class DifferentiableConstrainedFEMModel(torch.nn.Module):
    """Quasi-static constrained tetrahedral FEM with differentiable CG solve.

    ElasticSimulator remains the source of the production CUDA constrained FEM
    forward path. This model mirrors the same skeleton-cylinder constraint idea
    in PyTorch and solves a linear tetrahedral FEM system so render losses can
    back-propagate to per-frame skeleton nodes.
    """

    def __init__(self, stage1_npz: str | Path, cfg: PipelineConfig, device: str = "cuda") -> None:
        super().__init__()
        device = _resolve_device(device)
        data = np.load(stage1_npz)
        self.device_name = device
        self.cg_iters = int(cfg.fem_cg_iters)
        self.elastic_stiffness = float(cfg.fem_elastic_stiffness)
        self.handle_stiffness = float(cfg.fem_handle_stiffness)
        self.diagonal_reg = float(cfg.fem_diagonal_reg)

        rest_tet_nodes_np = np.asarray(data["rest_tet_nodes"], dtype=np.float32)
        tets_np = np.asarray(data["tets"], dtype=np.int64)
        stiffness, stiffness_diag = _assemble_linear_tet_stiffness(
            rest_tet_nodes_np,
            tets_np,
            poisson_ratio=float(cfg.elastic_fem_poisson_ratio),
        )

        self.register_buffer("rest_tet_nodes", _tensor(rest_tet_nodes_np, device))
        self.register_buffer("tets", torch.as_tensor(tets_np, dtype=torch.long, device=device))
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
        self.register_buffer(
            "stiffness",
            torch.sparse_coo_tensor(
                torch.as_tensor(stiffness["indices"], dtype=torch.long, device=device),
                torch.as_tensor(stiffness["values"], dtype=torch.float32, device=device),
                size=tuple(stiffness["shape"]),
                device=device,
            ).coalesce(),
        )
        self.register_buffer("stiffness_diag", _tensor(stiffness_diag, device).reshape(-1, 3))
        self.register_buffer("k_rest", self._stiffness_apply(self.rest_tet_nodes).detach())
        self.register_buffer("rest_bone_lengths", _bone_lengths(self.initial_joints, self.parents).detach())

        handle_data = _build_bone_handle_weights(
            self.rest_tet_nodes,
            self.rest_joints,
            self.parents,
            cfg,
        )
        self.register_buffer("handle_children", handle_data["children"])
        self.register_buffer("handle_parents", handle_data["parents"])
        self.register_buffer("handle_weights", handle_data["weights"])
        self.register_buffer("handle_strength", handle_data["strength"])
        self.handle_count = int(self.handle_children.numel())

        self.joints = torch.nn.Parameter(self.initial_joints.clone())

    def solve_frame(self, frame_index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        joints = self.joints[frame_index]
        target = self._handle_target(joints)
        rhs = (
            self.elastic_stiffness * self.k_rest
            + self.handle_stiffness * self.handle_strength[:, None] * target
            + self.diagonal_reg * self.rest_tet_nodes
        )

        def matvec(x: torch.Tensor) -> torch.Tensor:
            return (
                self.elastic_stiffness * self._stiffness_apply(x)
                + self.handle_stiffness * self.handle_strength[:, None] * x
                + self.diagonal_reg * x
            )

        blend = self.handle_strength[:, None].clamp(0.0, 1.0)
        x0 = self.rest_tet_nodes * (1.0 - blend) + target * blend
        tet_nodes = _conjugate_gradient(matvec, rhs, x0, self.cg_iters)
        return tet_nodes, tet_nodes[self.surface_node_indices], joints

    def fem_residuals_for_nodes(
        self,
        tet_nodes: torch.Tensor,
        joints: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        target = self._handle_target(joints)
        displacement = tet_nodes - self.rest_tet_nodes
        elastic = torch.mean(displacement * self._stiffness_apply(displacement))
        constraint = torch.mean(self.handle_strength[:, None] * (tet_nodes - target) ** 2)
        return {
            "fem_elastic_energy": elastic,
            "fem_constraint_residual": constraint,
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

    def _stiffness_apply(self, vertices: torch.Tensor) -> torch.Tensor:
        flat = vertices.reshape(-1, 1)
        return torch.sparse.mm(self.stiffness, flat).reshape_as(vertices)

    def _handle_target(self, posed_joints: torch.Tensor) -> torch.Tensor:
        rest_start = self.rest_joints[self.handle_parents]
        rest_end = self.rest_joints[self.handle_children]
        posed_start = posed_joints[self.handle_parents]
        posed_end = posed_joints[self.handle_children]
        rotations = _quaternion_to_matrix_torch(
            _quaternions_between_vectors_torch(rest_end - rest_start, posed_end - posed_start)
        )
        centered = self.rest_tet_nodes[:, None, :] - rest_start[None, :, :]
        targets = torch.einsum("nhc,hdc->nhd", centered, rotations) + posed_start[None, :, :]
        weighted = torch.sum(self.handle_weights[..., None] * targets, dim=1)
        normalizer = torch.sum(self.handle_weights, dim=1, keepdim=True).clamp_min(1e-8)
        blended = weighted / normalizer
        return torch.where(self.handle_strength[:, None] > 1e-8, blended, self.rest_tet_nodes)


def optimize_constrained_fem_stage2(
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
    model = DifferentiableConstrainedFEMModel(stage1_npz, cfg, device=device)
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
        accum = _empty_metric_accumulator()
        for frame_index in range(n_frames):
            tet_nodes, surface_vertices, joints = model.solve_frame(frame_index)
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
            fem_parts = model.fem_residuals_for_nodes(tet_nodes, joints)
            frame_objective = loss
            (frame_objective / n_frames).backward()
            _accumulate_metrics(accum, loss, parts, fem_parts, scale=1.0 / n_frames)

        regs = model.skeleton_regularizers()
        regularizer = (
            cfg.bone_length_weight * regs["bone_length"]
            + cfg.temporal_weight * regs["temporal"]
            + cfg.mocap_prior_weight * regs["mocap_prior"]
        )
        regularizer.backward()
        grad_norm = float(model.joints.grad.detach().norm().cpu()) if model.joints.grad is not None else 0.0
        opt.step()

        if step == 0 or step == total_steps - 1 or (step + 1) % max(1, total_steps // 10) == 0:
            history.append(
                {
                    "step": step,
                    "total": accum["render"] + float(regularizer.detach().cpu()),
                    "render": accum["render"],
                    "color": accum["color"],
                    "mask_loss": accum["mask_loss"],
                    "mask_soft": accum["mask_soft"],
                    "mask_01": accum["mask_01"],
                    "fem_elastic_energy": accum["fem_elastic_energy"],
                    "fem_constraint_residual": accum["fem_constraint_residual"],
                    "bone_length": float(regs["bone_length"].detach().cpu()),
                    "temporal": float(regs["temporal"].detach().cpu()),
                    "mocap_prior": float(regs["mocap_prior"].detach().cpu()),
                    "grad_norm": grad_norm,
                }
            )

    final_eval = _evaluate_model(model, binding, gt, camera, cfg, n_frames)
    optimized_npz = out_dir / "elastic_stage2_optimized_state.npz"
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
        handle_children=model.handle_children.detach().cpu().numpy(),
        handle_parents=model.handle_parents.detach().cpu().numpy(),
        handle_strength=model.handle_strength.detach().cpu().numpy(),
    )
    joints_npy = out_dir / "elastic_stage2_optimized_joints.npy"
    np.save(joints_npy, joints_np)
    losses_json = out_dir / "elastic_stage2_losses.json"
    with open(losses_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "steps": total_steps,
                "frames": n_frames,
                "lr": float(lr or cfg.optimize_lr),
                "weights": {
                    "color": cfg.color_loss_weight,
                    "mask": cfg.mask_loss_weight,
                    "bone_length": cfg.bone_length_weight,
                    "temporal": cfg.temporal_weight,
                    "mocap_prior": cfg.mocap_prior_weight,
                },
                "constrained_fem": {
                    "cg_iters": cfg.fem_cg_iters,
                    "elastic_stiffness": cfg.fem_elastic_stiffness,
                    "handle_stiffness": cfg.fem_handle_stiffness,
                    "diagonal_reg": cfg.fem_diagonal_reg,
                    "handle_count": model.handle_count,
                    "constrained_nodes": int((model.handle_strength > 1e-6).sum().detach().cpu()),
                    "radius_scale": cfg.elastic_fem_radius_scale,
                    "min_radius_scale": cfg.elastic_fem_min_radius_scale,
                    "max_radius_scale": cfg.elastic_fem_max_radius_scale,
                    "min_bone_length_scale": cfg.elastic_fem_min_bone_length_scale,
                    "gravity": cfg.gravity,
                },
                "camera_source": camera_source,
                "render_size": [width, height],
                "render_resolution_source": "override" if render_size is not None else "input_frame",
                "max_render_points": cfg.max_render_points,
                "gaussian_binding_k": cfg.gaussian_binding_k,
                "pull_gaussians_to_surface": cfg.pull_gaussians_to_surface,
                "history": history,
                "final_eval": final_eval,
            },
            f,
            indent=2,
        )
    _save_previews(out_dir, model, binding, camera, n_frames, cfg)
    return OptimizationArtifacts(out_dir, optimized_npz, losses_json, joints_npy)


def _assemble_linear_tet_stiffness(
    vertices: np.ndarray,
    tets: np.ndarray,
    poisson_ratio: float,
) -> tuple[dict[str, np.ndarray | tuple[int, int]], np.ndarray]:
    vertices = np.asarray(vertices, dtype=np.float32)
    tets = np.asarray(tets, dtype=np.int64)
    tet_vertices = vertices[tets]
    dm = np.stack(
        [
            tet_vertices[:, 1] - tet_vertices[:, 0],
            tet_vertices[:, 2] - tet_vertices[:, 0],
            tet_vertices[:, 3] - tet_vertices[:, 0],
        ],
        axis=-1,
    )
    det = np.linalg.det(dm)
    valid = np.abs(det) > 1e-12
    if not np.any(valid):
        raise ValueError("No non-degenerate tetrahedra for FEM stiffness assembly")
    tets = tets[valid]
    dm = dm[valid]
    volume = (np.abs(det[valid]) / 6.0).astype(np.float32)
    inv_dm = np.linalg.inv(dm).astype(np.float32)

    grads = np.zeros((tets.shape[0], 4, 3), dtype=np.float32)
    grads[:, 1, :] = inv_dm[:, 0, :]
    grads[:, 2, :] = inv_dm[:, 1, :]
    grads[:, 3, :] = inv_dm[:, 2, :]
    grads[:, 0, :] = -np.sum(grads[:, 1:, :], axis=1)

    b = np.zeros((tets.shape[0], 6, 12), dtype=np.float32)
    for local in range(4):
        gx = grads[:, local, 0]
        gy = grads[:, local, 1]
        gz = grads[:, local, 2]
        col = 3 * local
        b[:, 0, col + 0] = gx
        b[:, 1, col + 1] = gy
        b[:, 2, col + 2] = gz
        b[:, 3, col + 0] = gy
        b[:, 3, col + 1] = gx
        b[:, 4, col + 1] = gz
        b[:, 4, col + 2] = gy
        b[:, 5, col + 0] = gz
        b[:, 5, col + 2] = gx

    nu = float(np.clip(poisson_ratio, -0.95, 0.49))
    young = 1.0
    lam = young * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = young / (2.0 * (1.0 + nu))
    d = np.asarray(
        [
            [lam + 2.0 * mu, lam, lam, 0.0, 0.0, 0.0],
            [lam, lam + 2.0 * mu, lam, 0.0, 0.0, 0.0],
            [lam, lam, lam + 2.0 * mu, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, mu, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, mu, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, mu],
        ],
        dtype=np.float32,
    )
    ke = np.einsum("tki,kl,tlj,t->tij", b, d, b, volume, optimize=True).astype(np.float32)
    dofs = np.empty((tets.shape[0], 12), dtype=np.int64)
    for local in range(4):
        dofs[:, 3 * local + 0] = 3 * tets[:, local] + 0
        dofs[:, 3 * local + 1] = 3 * tets[:, local] + 1
        dofs[:, 3 * local + 2] = 3 * tets[:, local] + 2
    rows = np.repeat(dofs, 12, axis=1).reshape(-1)
    cols = np.tile(dofs, (1, 12)).reshape(-1)
    values = ke.reshape(-1)
    keep = np.abs(values) > 1e-12
    rows = rows[keep]
    cols = cols[keep]
    values = values[keep]
    shape = (vertices.shape[0] * 3, vertices.shape[0] * 3)
    diag = np.zeros(shape[0], dtype=np.float32)
    diag_mask = rows == cols
    np.add.at(diag, rows[diag_mask], values[diag_mask])
    return {
        "indices": np.stack([rows, cols], axis=0),
        "values": values.astype(np.float32, copy=False),
        "shape": shape,
    }, diag


def _build_bone_handle_weights(
    rest_nodes: torch.Tensor,
    rest_joints: torch.Tensor,
    parents: torch.Tensor,
    cfg: PipelineConfig,
) -> dict[str, torch.Tensor]:
    device = rest_nodes.device
    dtype = rest_nodes.dtype
    bbox_diag = torch.linalg.norm(rest_nodes.max(dim=0).values - rest_nodes.min(dim=0).values).clamp_min(1e-6)
    children: list[int] = []
    parent_indices: list[int] = []
    weights: list[torch.Tensor] = []
    support = float(cfg.fem_handle_support)

    for child in range(int(parents.shape[0])):
        parent = int(parents[child].detach().cpu())
        if parent < 0:
            continue
        start = rest_joints[parent]
        end = rest_joints[child]
        segment = end - start
        length = torch.linalg.norm(segment).clamp_min(1e-8)
        if float(length.detach().cpu()) < float(cfg.elastic_fem_min_bone_length_scale) * float(bbox_diag.detach().cpu()):
            continue
        radius = torch.clamp(
            length * float(cfg.elastic_fem_radius_scale),
            min=float(cfg.elastic_fem_min_radius_scale) * float(bbox_diag.detach().cpu()),
            max=float(cfg.elastic_fem_max_radius_scale) * float(bbox_diag.detach().cpu()),
        )
        rel = rest_nodes - start
        t = torch.sum(rel * segment[None, :], dim=-1) / torch.sum(segment * segment).clamp_min(1e-12)
        t = t.clamp(0.0, 1.0)
        closest = start + t[:, None] * segment[None, :]
        dist = torch.linalg.norm(rest_nodes - closest, dim=-1)
        weight = torch.exp(-0.5 * (dist / radius.clamp_min(1e-8)) ** 2)
        weight = torch.where(dist <= support * radius, weight, torch.zeros_like(weight))
        if float(weight.max().detach().cpu()) <= 1e-8:
            nearest = torch.argmin(dist)
            weight = torch.zeros_like(weight)
            weight[nearest] = 1.0
        children.append(child)
        parent_indices.append(parent)
        weights.append(weight.to(dtype=dtype, device=device))

    if not weights:
        raise ValueError("No valid skeleton bone handles were produced for constrained FEM")
    weight_tensor = torch.stack(weights, dim=1)
    if float(cfg.fem_handle_weight_power) != 1.0:
        weight_tensor = weight_tensor.clamp_min(0.0) ** float(cfg.fem_handle_weight_power)
    strength = weight_tensor.sum(dim=1).clamp(0.0, 1.0)
    return {
        "children": torch.as_tensor(children, dtype=torch.long, device=device),
        "parents": torch.as_tensor(parent_indices, dtype=torch.long, device=device),
        "weights": weight_tensor,
        "strength": strength.to(dtype=dtype, device=device),
    }


def _conjugate_gradient(
    matvec,
    rhs: torch.Tensor,
    x0: torch.Tensor,
    iterations: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    x = x0
    residual = rhs - matvec(x)
    direction = residual
    residual_norm = torch.sum(residual * residual)
    for _ in range(max(1, int(iterations))):
        ad = matvec(direction)
        denom = torch.sum(direction * ad).clamp_min(eps)
        alpha = residual_norm / denom
        x = x + alpha * direction
        residual = residual - alpha * ad
        next_norm = torch.sum(residual * residual)
        beta = next_norm / residual_norm.clamp_min(eps)
        direction = residual + beta * direction
        residual_norm = next_norm
    return x


def _empty_metric_accumulator() -> dict[str, float]:
    return {
        "render": 0.0,
        "color": 0.0,
        "mask_loss": 0.0,
        "mask_soft": 0.0,
        "mask_01": 0.0,
        "fem_elastic_energy": 0.0,
        "fem_constraint_residual": 0.0,
    }


def _accumulate_metrics(
    accum: dict[str, float],
    loss: torch.Tensor,
    parts: dict[str, torch.Tensor],
    fem_parts: dict[str, torch.Tensor],
    scale: float,
) -> None:
    accum["render"] += float(loss.detach().cpu()) * scale
    accum["color"] += float(parts["color"].detach().cpu()) * scale
    accum["mask_loss"] += float(parts["mask_loss"].detach().cpu()) * scale
    accum["mask_soft"] += float(parts["mask_soft"].detach().cpu()) * scale
    accum["mask_01"] += float(parts["mask_01"].detach().cpu()) * scale
    accum["fem_elastic_energy"] += float(fem_parts["fem_elastic_energy"].detach().cpu()) * scale
    accum["fem_constraint_residual"] += float(fem_parts["fem_constraint_residual"].detach().cpu()) * scale


def _evaluate_model(
    model: DifferentiableConstrainedFEMModel,
    binding,
    gt: list[dict[str, torch.Tensor]],
    camera,
    cfg: PipelineConfig,
    n_frames: int,
) -> dict[str, float]:
    accum = _empty_metric_accumulator()
    with torch.no_grad():
        for frame_index in range(n_frames):
            tet_nodes, surface_vertices, joints = model.solve_frame(frame_index)
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
            fem_parts = model.fem_residuals_for_nodes(tet_nodes, joints)
            _accumulate_metrics(accum, loss, parts, fem_parts, scale=1.0 / n_frames)
    return accum


def _save_previews(
    out_dir: Path,
    model: DifferentiableConstrainedFEMModel,
    binding,
    camera,
    n_frames: int,
    cfg: PipelineConfig,
) -> None:
    preview_dir = out_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    indices = _preview_indices(n_frames)
    with torch.no_grad():
        for frame_index in indices:
            _, surface_vertices, _ = model.solve_frame(frame_index)
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
            Image.fromarray(image).save(preview_dir / f"elastic_optimized_{frame_index:05d}.png")


def _preview_indices(n_frames: int) -> list[int]:
    if n_frames <= 0:
        return []
    return sorted({0, min(n_frames - 1, 30), min(n_frames - 1, 60), n_frames - 1})
