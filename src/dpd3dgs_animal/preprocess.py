from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import resize

from .vendor.briarmbg import BriaRMBG


@dataclass
class TransparentFrameArtifacts:
    raw_dir: Path
    mask_dir: Path
    rgba_dir: Path
    ref_rgb: Path
    ref_mask: Path
    ref_rgba: Path
    ref_transform: Path
    crop_box_xyxy: tuple[int, int, int, int]
    frame_count: int
    size: tuple[int, int]
    background_rgb: list[int]


def video_to_transparent_frames(
    video_path: str | Path,
    out_root: str | Path,
    method: str = "rmbg",
    rmbg_weights_dir: str | Path | None = None,
    device: str = "cuda",
    background_threshold: float = 10.0,
    min_component_area: int = 1000,
    ref_padding: int = 40,
) -> TransparentFrameArtifacts:
    video_path = Path(video_path)
    out_root = Path(out_root)
    raw_dir = out_root / "raw_frames"
    mask_dir = out_root / "masks"
    rgba_dir = out_root / "rgba_frames"
    ref_dir = out_root / "sam3d_ref"
    for directory in (raw_dir, mask_dir, rgba_dir, ref_dir):
        directory.mkdir(parents=True, exist_ok=True)

    frames = _read_video_rgb(video_path)
    if not frames:
        raise ValueError(f"No frames found in {video_path}")
    bg = _estimate_background(frames[0])
    method = method.lower()
    if method not in {"rmbg", "connected_background"}:
        raise ValueError(f"Unknown foreground method: {method}")
    rmbg_model = None
    if method == "rmbg":
        weights_dir = Path(rmbg_weights_dir) if rmbg_weights_dir else _default_rmbg_weights_dir()
        resolved_device = device if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu"
        rmbg_model = BriaRMBG.from_pretrained(str(weights_dir)).to(resolved_device).eval()

    counts = []
    for index, rgb in enumerate(frames):
        if rmbg_model is not None:
            alpha, mask = rmbg_alpha(
                rgb,
                rmbg_model,
                device=resolved_device,
                min_component_area=min_component_area,
            )
        else:
            mask = connected_background_alpha(rgb, bg, background_threshold, min_component_area)
            alpha = mask
        counts.append(int((mask > 0).sum()))
        Image.fromarray(rgb).save(raw_dir / f"{index:05d}.png")
        Image.fromarray(mask).save(mask_dir / f"{index:05d}.png")
        Image.fromarray(np.dstack([rgb, alpha])).save(rgba_dir / f"{index:05d}.png")

    ref_rgb, ref_mask, crop_box = _reference_crop(
        frames[0],
        np.array(Image.open(mask_dir / "00000.png")),
        ref_padding,
    )
    ref_rgb_path = ref_dir / "cat_ref_rgb.png"
    ref_mask_path = ref_dir / "cat_ref_mask.png"
    ref_rgba_path = ref_dir / "cat_ref_rgba.png"
    ref_transform_path = ref_dir / "reference_transform.json"
    Image.fromarray(ref_rgb).save(ref_rgb_path)
    Image.fromarray(ref_mask).save(ref_mask_path)
    Image.fromarray(np.dstack([ref_rgb, ref_mask])).save(ref_rgba_path)

    h, w = frames[0].shape[:2]
    with open(ref_transform_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "coordinate_convention": "full_frame_pixels_x_right_y_down",
                "full_frame_size": [w, h],
                "crop_box_xyxy": list(crop_box),
                "crop_size": [ref_rgb.shape[1], ref_rgb.shape[0]],
                "mapping": {
                    "crop_to_full": "u_full = u_crop + x0; v_full = v_crop + y0",
                    "full_to_crop": "u_crop = u_full - x0; v_crop = v_full - y0",
                },
            },
            f,
            indent=2,
        )
    return TransparentFrameArtifacts(
        raw_dir,
        mask_dir,
        rgba_dir,
        ref_rgb_path,
        ref_mask_path,
        ref_rgba_path,
        ref_transform_path,
        crop_box,
        len(frames),
        (w, h),
        [int(x) for x in bg],
    )


def rmbg_alpha(
    rgb: np.ndarray,
    model: BriaRMBG,
    device: str = "cuda",
    min_component_area: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Infer soft alpha with RMBG-1.4 and return soft alpha plus a clean binary mask."""

    tensor = torch.from_numpy(np.asarray(rgb)).to(device=device, dtype=torch.float32)
    tensor = tensor.permute(2, 0, 1) / 255.0
    tensor = resize(tensor, [1024, 1024], antialias=True)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    tensor = (tensor / tensor.max().clamp_min(1e-6) - mean).unsqueeze(0)
    with torch.no_grad():
        prediction = model(tensor)[0][0]
        prediction = torch.nn.functional.interpolate(
            prediction,
            size=rgb.shape[:2],
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        prediction = (prediction - prediction.min()) / (
            prediction.max() - prediction.min()
        ).clamp_min(1e-6)

    alpha = (prediction.cpu().numpy() * 255.0).astype(np.uint8)
    _threshold, binary = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    num, labels, stats, _ = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), 8)
    mask = np.zeros_like(binary)
    if num > 1:
        valid = [
            component
            for component in range(1, num)
            if stats[component, cv2.CC_STAT_AREA] >= int(min_component_area)
        ]
        if valid:
            largest = max(valid, key=lambda component: stats[component, cv2.CC_STAT_AREA])
            mask[labels == largest] = 255
    alpha = np.where(mask > 0, alpha, 0).astype(np.uint8)
    return alpha, mask


def connected_background_alpha(
    rgb: np.ndarray,
    background_rgb: np.ndarray,
    background_threshold: float = 10.0,
    min_component_area: int = 1000,
) -> np.ndarray:
    """Return alpha where only border-connected background is transparent."""

    rgb = np.asarray(rgb, dtype=np.uint8)
    bg = np.asarray(background_rgb, dtype=np.float32)
    dist = np.linalg.norm(rgb.astype(np.float32) - bg[None, None, :], axis=-1)
    near_background = (dist <= float(background_threshold)).astype(np.uint8)
    num, labels, _stats, _centroids = cv2.connectedComponentsWithStats(near_background, connectivity=8)
    if num <= 1:
        return np.full(rgb.shape[:2], 255, dtype=np.uint8)

    border_labels = np.unique(
        np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    )
    background = np.isin(labels, border_labels)
    foreground = (~background).astype(np.uint8)

    num_fg, fg_labels, fg_stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)
    kept = np.zeros_like(foreground, dtype=np.uint8)
    for component in range(1, num_fg):
        if fg_stats[component, cv2.CC_STAT_AREA] >= int(min_component_area):
            kept[fg_labels == component] = 1

    kernel = np.ones((5, 5), dtype=np.uint8)
    kept = cv2.morphologyEx(kept, cv2.MORPH_CLOSE, kernel, iterations=1)
    kept = _fill_internal_holes(kept)
    return (kept * 255).astype(np.uint8)


def _fill_internal_holes(mask: np.ndarray) -> np.ndarray:
    inv = (mask == 0).astype(np.uint8)
    num, labels, _stats, _centroids = cv2.connectedComponentsWithStats(inv, connectivity=8)
    if num <= 1:
        return mask.astype(np.uint8)
    border_labels = np.unique(
        np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    )
    outside = np.isin(labels, border_labels)
    holes = ~outside
    return np.maximum(mask.astype(np.uint8), holes.astype(np.uint8))


def _read_video_rgb(video_path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    frames: list[np.ndarray] = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def _estimate_background(frame_rgb: np.ndarray, border: int = 20) -> np.ndarray:
    pixels = np.concatenate(
        [
            frame_rgb[:border].reshape(-1, 3),
            frame_rgb[-border:].reshape(-1, 3),
            frame_rgb[:, :border].reshape(-1, 3),
            frame_rgb[:, -border:].reshape(-1, 3),
        ],
        axis=0,
    )
    return np.median(pixels, axis=0).astype(np.uint8)


def _reference_crop(
    rgb: np.ndarray,
    mask: np.ndarray,
    padding: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    ys, xs = np.nonzero(mask > 0)
    if xs.size == 0 or ys.size == 0:
        raise ValueError("Cannot build reference crop from empty mask")
    h, w = mask.shape
    x0 = max(0, int(xs.min()) - padding)
    y0 = max(0, int(ys.min()) - padding)
    x1 = min(w, int(xs.max()) + padding + 1)
    y1 = min(h, int(ys.max()) + padding + 1)
    side = max(x1 - x0, y1 - y0)
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    x0 = max(0, min(w - side, cx - side // 2))
    y0 = max(0, min(h - side, cy - side // 2))
    x1 = x0 + side
    y1 = y0 + side
    crop_box = (x0, y0, x1, y1)
    return rgb[y0:y1, x0:x1], mask[y0:y1, x0:x1], crop_box


def _default_rmbg_weights_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "checkpoints" / "mocap_anything" / "RMBG-1.4"
