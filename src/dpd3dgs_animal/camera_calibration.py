from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .gaussian import GaussianCloud
from .render import PinholeCamera, camera_to_arrays, look_at_world_to_camera


@dataclass
class CameraFitResult:
    camera: PinholeCamera
    score: float
    mask_iou: float
    mask_01_error: float
    view_direction: list[float]
    up: list[float]
    zoom: float
    shift: list[float]
    target_bbox: list[int]
    projected_bbox: list[int]
    valid_points: int

    def to_json_dict(self) -> dict:
        intrinsics, world_to_camera = camera_to_arrays(self.camera)
        payload = asdict(self)
        payload["camera"] = {
            "intrinsics": intrinsics.tolist(),
            "world_to_camera": world_to_camera.tolist(),
        }
        return payload


def fit_camera_to_frame0_mask(
    cloud: GaussianCloud,
    frame_path: str | Path,
    width: int,
    height: int,
    max_points: int = 12000,
    fit_padding: float = 0.92,
    radius_px: int = 3,
) -> CameraFitResult:
    target_mask = load_frame_mask(frame_path, width, height)
    target_bbox = mask_bbox(target_mask)
    points = np.asarray(cloud.xyz, dtype=np.float32)
    if points.shape[0] > max_points:
        idx = np.linspace(0, points.shape[0] - 1, max_points).astype(np.int64)
        points = points[idx]

    center = (points.min(axis=0) + points.max(axis=0)) * 0.5
    radius = max(float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)) * 0.5), 1e-3)
    target_cx = (target_bbox[0] + target_bbox[2]) * 0.5
    target_cy = (target_bbox[1] + target_bbox[3]) * 0.5
    target_w = max(float(target_bbox[2] - target_bbox[0] + 1), 1.0)
    target_h = max(float(target_bbox[3] - target_bbox[1] + 1), 1.0)

    best: CameraFitResult | None = None
    for view_direction in _view_directions():
        for up in _up_vectors(view_direction):
            w2c = look_at_world_to_camera(
                center + view_direction * radius * 3.0,
                center,
                up,
            )
            normalized = _normalized_projection(points, w2c)
            if normalized is None:
                continue
            u, v, valid_count = normalized
            proj_bbox_norm = _percentile_bbox(u, v)
            proj_w = max(float(proj_bbox_norm[2] - proj_bbox_norm[0]), 1e-6)
            proj_h = max(float(proj_bbox_norm[3] - proj_bbox_norm[1]), 1e-6)
            proj_cx = (proj_bbox_norm[0] + proj_bbox_norm[2]) * 0.5
            proj_cy = (proj_bbox_norm[1] + proj_bbox_norm[3]) * 0.5
            base_focal = min(target_w / proj_w, target_h / proj_h) * float(fit_padding)

            for zoom in (0.8, 0.9, 1.0, 1.1, 1.25):
                focal = base_focal * zoom
                cx = target_cx - focal * proj_cx
                cy = target_cy - focal * proj_cy
                for shift_x in (-0.08, 0.0, 0.08):
                    for shift_y in (-0.08, 0.0, 0.08):
                        camera = PinholeCamera(
                            width,
                            height,
                            float(focal),
                            float(focal),
                            float(cx + shift_x * target_w),
                            float(cy + shift_y * target_h),
                            w2c.astype(np.float32),
                        )
                        pred_mask = render_point_mask(points, camera, radius_px=radius_px)
                        iou, error = mask_metrics(pred_mask, target_mask)
                        score = iou - 0.25 * error
                        projected_bbox = mask_bbox(pred_mask)
                        result = CameraFitResult(
                            camera=camera,
                            score=float(score),
                            mask_iou=float(iou),
                            mask_01_error=float(error),
                            view_direction=view_direction.astype(float).tolist(),
                            up=up.astype(float).tolist(),
                            zoom=float(zoom),
                            shift=[float(shift_x), float(shift_y)],
                            target_bbox=[int(x) for x in target_bbox],
                            projected_bbox=[int(x) for x in projected_bbox],
                            valid_points=int(valid_count),
                        )
                        if best is None or result.score > best.score:
                            best = result

    if best is None:
        raise RuntimeError("Unable to fit camera: no valid projections found")
    return best


def write_camera_fit(path: str | Path, fit: CameraFitResult) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fit.to_json_dict(), f, indent=2)
    return path


def load_frame_mask(frame_path: str | Path, width: int, height: int) -> np.ndarray:
    image = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(frame_path)
    if image.shape[1] != width or image.shape[0] != height:
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    if image.ndim == 3 and image.shape[2] == 4:
        return (image[..., 3] > 127).astype(np.uint8)
    gray = cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2GRAY)
    return (gray > 5).astype(np.uint8)


def render_point_mask(points: np.ndarray, camera: PinholeCamera, radius_px: int = 3) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    cam = np.concatenate([points, ones], axis=1) @ np.asarray(camera.world_to_camera, dtype=np.float32).T
    z = np.maximum(cam[:, 2], 1e-6)
    x = camera.fx * (cam[:, 0] / z) + camera.cx
    y = camera.cy - camera.fy * (cam[:, 1] / z)
    valid = (
        (cam[:, 2] > 1e-5)
        & (x >= -radius_px)
        & (x < camera.width + radius_px)
        & (y >= -radius_px)
        & (y < camera.height + radius_px)
    )
    xi = np.rint(x[valid]).astype(np.int32)
    yi = np.rint(y[valid]).astype(np.int32)
    mask = np.zeros((camera.height, camera.width), dtype=np.uint8)
    for dy in range(-radius_px, radius_px + 1):
        yy = np.clip(yi + dy, 0, camera.height - 1)
        for dx in range(-radius_px, radius_px + 1):
            xx = np.clip(xi + dx, 0, camera.width - 1)
            mask[yy, xx] = 1
    return mask


def mask_metrics(pred_mask: np.ndarray, target_mask: np.ndarray) -> tuple[float, float]:
    pred = np.asarray(pred_mask, dtype=bool)
    target = np.asarray(target_mask, dtype=bool)
    inter = np.logical_and(pred, target).sum()
    union = max(int(np.logical_or(pred, target).sum()), 1)
    iou = float(inter / union)
    error = float(np.mean(pred != target))
    return iou, error


def mask_bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0 or ys.size == 0:
        h, w = mask.shape[:2]
        return [0, 0, w - 1, h - 1]
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _normalized_projection(points: np.ndarray, world_to_camera: np.ndarray) -> tuple[np.ndarray, np.ndarray, int] | None:
    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    cam = np.concatenate([points, ones], axis=1) @ world_to_camera.T
    valid = cam[:, 2] > 1e-5
    if int(valid.sum()) < 16:
        return None
    z = cam[valid, 2]
    u = cam[valid, 0] / z
    v = -cam[valid, 1] / z
    return u, v, int(valid.sum())


def _percentile_bbox(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.array(
        [
            np.percentile(u, 1.0),
            np.percentile(v, 1.0),
            np.percentile(u, 99.0),
            np.percentile(v, 99.0),
        ],
        dtype=np.float32,
    )


def _view_directions() -> tuple[np.ndarray, ...]:
    return (
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([-1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        np.array([0.0, -1.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
        np.array([0.0, 0.0, -1.0], dtype=np.float32),
    )


def _up_vectors(view_direction: np.ndarray) -> tuple[np.ndarray, ...]:
    axes = (
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
    )
    up: list[np.ndarray] = []
    for axis in axes:
        if abs(float(np.dot(axis, view_direction))) > 0.25:
            continue
        up.append(axis)
        up.append(-axis)
    return tuple(up)
