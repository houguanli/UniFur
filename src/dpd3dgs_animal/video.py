from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


@dataclass
class FrameSequence:
    image_root: Path
    sequence_dir: Path
    frame_paths: list[Path]
    fps: float
    seq_name: str

    @property
    def width(self) -> int:
        return image_size(self.frame_paths[0])[0]

    @property
    def height(self) -> int:
        return image_size(self.frame_paths[0])[1]

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height


def extract_frames(
    video_path: str | Path,
    output_root: str | Path,
    seq_name: str | None = None,
    stride: int = 1,
    max_frames: int | None = None,
    resize_long_edge: int | None = None,
    overwrite: bool = False,
) -> FrameSequence:
    video_path = Path(video_path)
    output_root = Path(output_root)
    seq_name = seq_name or video_path.stem
    sequence_dir = output_root / seq_name
    sequence_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(sequence_dir.glob("*.png"))
    if existing and not overwrite:
        fps = _read_fps(video_path)
        return FrameSequence(output_root, sequence_dir, existing, fps, seq_name)

    for old in existing:
        old.unlink()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_paths: list[Path] = []
    input_index = 0
    output_index = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if input_index % stride != 0:
            input_index += 1
            continue
        if resize_long_edge:
            frame_bgr = _resize_long_edge(frame_bgr, resize_long_edge)
        path = sequence_dir / f"{output_index:05d}.png"
        cv2.imwrite(str(path), frame_bgr)
        frame_paths.append(path)
        output_index += 1
        input_index += 1
        if max_frames is not None and output_index >= max_frames:
            break
    cap.release()

    if len(frame_paths) < 2:
        raise ValueError(f"Need at least two frames for MocapAnything, got {len(frame_paths)}")
    return FrameSequence(output_root, sequence_dir, frame_paths, fps, seq_name)


def ensure_frame_sequence(
    input_path: str | Path,
    output_root: str | Path,
    seq_name: str | None = None,
    **kwargs,
) -> FrameSequence:
    input_path = Path(input_path)
    if input_path.is_dir():
        frames = sorted(
            p for p in input_path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        if len(frames) < 2:
            raise ValueError(f"Image directory must contain at least two frames: {input_path}")
        return FrameSequence(input_path.parent, input_path, frames, 30.0, seq_name or input_path.name)
    return extract_frames(input_path, output_root, seq_name=seq_name, **kwargs)


def make_reference_mask(
    image_path: str | Path,
    mask_path: str | Path,
    threshold: int = 8,
) -> Path:
    """Create a SAM3D binary mask from alpha if available, otherwise full frame."""

    image_path = Path(image_path)
    mask_path = Path(mask_path)
    mask_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path)
    if image.mode in {"RGBA", "LA"}:
        alpha = np.array(image.getchannel("A"))
        mask = (alpha > threshold).astype(np.uint8) * 255
    else:
        w, h = image.size
        mask = np.full((h, w), 255, dtype=np.uint8)
    Image.fromarray(mask).save(mask_path)
    return mask_path


def image_size(path: str | Path) -> tuple[int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    h, w = image.shape[:2]
    return int(w), int(h)


def _read_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return float(fps)


def _resize_long_edge(frame_bgr: np.ndarray, long_edge: int) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    scale = long_edge / float(max(h, w))
    if scale >= 1.0:
        return frame_bgr
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return cv2.resize(frame_bgr, new_size, interpolation=cv2.INTER_AREA)
