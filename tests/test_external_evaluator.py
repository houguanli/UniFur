from __future__ import annotations

import torch

from scripts.evaluate_external_renders import _connected_mask_errors, _frame_metrics


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
