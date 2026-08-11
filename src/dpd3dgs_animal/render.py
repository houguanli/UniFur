from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

from .gaussian import GaussianCloud


@dataclass
class PinholeCamera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    world_to_camera: np.ndarray
    image_y_down: bool = False


def camera_to_arrays(camera: PinholeCamera) -> tuple[np.ndarray, np.ndarray]:
    intrinsics = np.array(
        [camera.width, camera.height, camera.fx, camera.fy, camera.cx, camera.cy],
        dtype=np.float32,
    )
    return intrinsics, np.asarray(camera.world_to_camera, dtype=np.float32)


def camera_from_arrays(
    intrinsics: np.ndarray,
    world_to_camera: np.ndarray,
    width: int | None = None,
    height: int | None = None,
    image_y_down: bool = False,
) -> PinholeCamera:
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    src_width = float(intrinsics[0])
    src_height = float(intrinsics[1])
    dst_width = int(width if width is not None else src_width)
    dst_height = int(height if height is not None else src_height)
    sx = dst_width / max(src_width, 1.0)
    sy = dst_height / max(src_height, 1.0)
    return PinholeCamera(
        dst_width,
        dst_height,
        float(intrinsics[2] * sx),
        float(intrinsics[3] * sy),
        float(intrinsics[4] * sx),
        float(intrinsics[5] * sy),
        np.asarray(world_to_camera, dtype=np.float32),
        image_y_down=image_y_down,
    )


def camera_from_stage1_npz(
    stage1_npz: str,
    width: int,
    height: int,
) -> PinholeCamera | None:
    data = np.load(stage1_npz)
    if "camera_intrinsics" not in data or "camera_world_to_camera" not in data:
        return None
    image_y_down = bool(
        int(np.asarray(data["camera_image_y_down"]).reshape(-1)[0])
        if "camera_image_y_down" in data
        else 0
    )
    return camera_from_arrays(
        data["camera_intrinsics"],
        data["camera_world_to_camera"],
        width,
        height,
        image_y_down=image_y_down,
    )


def default_camera_for_vertices(vertices: np.ndarray, width: int, height: int) -> PinholeCamera:
    vertices = np.asarray(vertices, dtype=np.float32)
    lo = vertices.min(axis=0)
    hi = vertices.max(axis=0)
    center = (lo + hi) * 0.5
    radius = max(float(np.linalg.norm(hi - lo) * 0.5), 1e-3)
    eye = center + np.array([0.0, 0.0, radius * 3.0], dtype=np.float32)
    w2c = look_at_world_to_camera(eye, center, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    focal = 0.8 * max(width, height)
    return PinholeCamera(width, height, focal, focal, width * 0.5, height * 0.5, w2c)


def look_at_world_to_camera(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = target - eye
    forward = forward / max(float(np.linalg.norm(forward)), 1e-8)
    right = np.cross(forward, up)
    right = right / max(float(np.linalg.norm(right)), 1e-8)
    true_up = np.cross(right, forward)
    rot = np.stack([right, true_up, forward], axis=0)
    trans = -rot @ eye
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = rot
    mat[:3, 3] = trans
    return mat


def point_splat_render(
    cloud: GaussianCloud,
    camera: PinholeCamera,
    xyz: np.ndarray | None = None,
    radius_px: int = 1,
    max_points: int = 120000,
    device: str = "cuda",
) -> dict[str, np.ndarray]:
    device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
    pts_np = np.asarray(xyz if xyz is not None else cloud.xyz, dtype=np.float32)
    color_np = np.asarray(cloud.color, dtype=np.float32)
    opacity_np = (
        np.asarray(cloud.opacity, dtype=np.float32)
        if cloud.opacity is not None
        else np.ones((color_np.shape[0],), dtype=np.float32)
    )
    if pts_np.shape[0] > max_points:
        idx = np.linspace(0, pts_np.shape[0] - 1, max_points).astype(np.int64)
        pts_np = pts_np[idx]
        color_np = color_np[idx]
        opacity_np = opacity_np[idx]

    pts = torch.as_tensor(pts_np, dtype=torch.float32, device=device)
    color = torch.as_tensor(color_np, dtype=torch.float32, device=device)
    opacity = torch.as_tensor(opacity_np, dtype=torch.float32, device=device).clamp(0.0, 1.0)
    w2c = torch.as_tensor(camera.world_to_camera, dtype=torch.float32, device=device)
    ones = torch.ones((pts.shape[0], 1), dtype=torch.float32, device=device)
    cam = torch.cat([pts, ones], dim=1) @ w2c.T
    z = cam[:, 2].clamp_min(1e-6)
    x = camera.fx * (cam[:, 0] / z) + camera.cx
    y_sign = 1.0 if camera.image_y_down else -1.0
    y = camera.cy + y_sign * camera.fy * (cam[:, 1] / z)
    valid = (z > 1e-5) & (x >= -radius_px) & (x < camera.width + radius_px) & (y >= -radius_px) & (y < camera.height + radius_px)
    x = x[valid]
    y = y[valid]
    color = color[valid]
    opacity = opacity[valid]

    image = torch.zeros((camera.height, camera.width, 3), dtype=torch.float32, device=device)
    weight = torch.zeros((camera.height, camera.width, 1), dtype=torch.float32, device=device)
    if x.numel() == 0:
        return {"rgb": image.cpu().numpy(), "mask": weight[..., 0].cpu().numpy()}

    offsets = torch.arange(-radius_px, radius_px + 1, device=device)
    oy, ox = torch.meshgrid(offsets, offsets, indexing="ij")
    ox = ox.reshape(1, -1).float()
    oy = oy.reshape(1, -1).float()
    base_x = torch.round(x).long().reshape(-1, 1)
    base_y = torch.round(y).long().reshape(-1, 1)
    px = base_x + ox.long()
    py = base_y + oy.long()
    valid_px = (px >= 0) & (px < camera.width) & (py >= 0) & (py < camera.height)
    center_x = px.float() + 0.5
    center_y = py.float() + 0.5
    sigma = max(float(radius_px) * 0.5, 1e-6)
    dist2 = (center_x - x.reshape(-1, 1)) ** 2 + (center_y - y.reshape(-1, 1)) ** 2
    splat_weight = torch.exp(-0.5 * dist2 / (sigma * sigma)) * opacity.reshape(-1, 1)
    splat_weight = torch.where(valid_px, splat_weight, torch.zeros_like(splat_weight))
    flat = (py.clamp(0, camera.height - 1) * camera.width + px.clamp(0, camera.width - 1)).reshape(-1)
    weights_flat = splat_weight.reshape(-1)
    image_flat = image.reshape(-1, 3)
    weight_flat = weight.reshape(-1, 1)
    rgb_values = (splat_weight[..., None] * color[:, None, :]).reshape(-1, 3)
    image_flat.index_add_(0, flat, rgb_values)
    weight_flat.index_add_(0, flat, weights_flat[:, None])
    image = image_flat.reshape(camera.height, camera.width, 3)
    weight = weight_flat.reshape(camera.height, camera.width, 1)
    mask = 1.0 - torch.exp(-weight)
    image = image / weight.clamp_min(1e-6)
    return {"rgb": image.cpu().numpy(), "mask": mask[..., 0].cpu().numpy()}


def load_gt_frame(path: str, width: int, height: int) -> dict[str, np.ndarray]:
    bgr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
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
        mask = np.ones((height, width), dtype=np.float32)
    return {"rgb": rgb, "mask": mask}


def render_losses(
    pred_rgb: np.ndarray,
    pred_mask: np.ndarray,
    gt_rgb: np.ndarray,
    gt_mask: np.ndarray,
    color_weight: float = 1.0,
    mask_weight: float = 10.0,
) -> dict[str, float]:
    pred_rgb_t = torch.as_tensor(pred_rgb, dtype=torch.float32)
    pred_mask_t = torch.as_tensor(pred_mask, dtype=torch.float32)
    gt_rgb_t = torch.as_tensor(gt_rgb, dtype=torch.float32)
    gt_mask_t = torch.as_tensor(gt_mask, dtype=torch.float32)
    valid = gt_mask_t[..., None].clamp(0.0, 1.0)
    color_denom = (valid.sum() * pred_rgb_t.shape[-1]).clamp_min(1.0)
    color = (torch.abs(pred_rgb_t - gt_rgb_t) * valid).sum() / color_denom
    mask = torch.mean(torch.abs((pred_mask_t > 0.5).float() - (gt_mask_t > 0.5).float()))
    total = color_weight * color + mask_weight * mask
    return {"color": float(color), "mask": float(mask), "total": float(total)}
