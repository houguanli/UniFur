from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from scripts.evaluate_external_renders import (
    _connected_mask_errors,
    _frame_metrics,
    _load_gt_frame_torch,
)


def test_load_gt_preserves_uint16_rgba_dynamic_range(tmp_path) -> None:
    rgba = np.zeros((2, 2, 4), dtype=np.uint16)
    rgba[0, 0] = [16384, 32768, 49152, 65535]  # BGRA
    path = tmp_path / "frame.png"
    assert cv2.imwrite(str(path), rgba)

    loaded = _load_gt_frame_torch(path, 2, 2, "cpu")

    assert loaded["mask"][0, 0].item() == pytest.approx(1.0)
    assert loaded["rgb"][0, 0].tolist() == pytest.approx(
        [49152 / 65535, 32768 / 65535, 16384 / 65535], abs=1.0e-6
    )


def test_perfect_frame_metrics() -> None:
    rgb = torch.rand(6, 8, 3)
    mask = torch.zeros(6, 8)
    mask[1:5, 2:7] = 1.0
    metrics = _frame_metrics(
        {"rgb": rgb * mask[..., None], "mask": mask},
        {"rgb": rgb, "mask": mask},
    )
    assert metrics["foreground_psnr"] == 100.0
    assert metrics["full_psnr"] == 100.0
    assert metrics["mask_iou"] == 1.0


def test_connected_mask_errors_detect_hole_and_overflow() -> None:
    target = torch.zeros(8, 8)
    target[1:7, 1:7] = 1.0
    prediction = target.clone()
    prediction[2:4, 2:4] = 0.0
    prediction[0, 0] = 1.0
    prediction[0, 7] = 1.0
    errors = _connected_mask_errors(prediction, target)
    assert errors["largest_false_negative_component_fraction"] == 4 / 36
    assert errors["largest_false_positive_component_fraction"] == 1 / 36
