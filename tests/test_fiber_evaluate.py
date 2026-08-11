from __future__ import annotations

import pytest
import torch

from dpd3dgs_animal.fiber_evaluate import _frame_metrics, _select_frame_indices
from dpd3dgs_animal.fiber_route_audit import _route_contribution
from dpd3dgs_animal.optimize import differentiable_render_loss


def test_select_frame_indices_honors_holdout_slice() -> None:
    assert _select_frame_indices(40, 32, 1, 8) == list(range(32, 40))
    assert _select_frame_indices(40, 3, 4, 3) == [3, 7, 11]


def test_select_frame_indices_validates_arguments() -> None:
    with pytest.raises(ValueError):
        _select_frame_indices(40, -1, 1, None)
    with pytest.raises(ValueError):
        _select_frame_indices(40, 0, 0, None)


def test_frame_metrics_reports_foreground_and_official_masked_psnr() -> None:
    prediction = {
        "rgb": torch.zeros((2, 2, 3)),
        "mask": torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
    }
    ground_truth = {
        "rgb": torch.ones((2, 2, 3)),
        "mask": torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
    }
    metrics = _frame_metrics(prediction, ground_truth)
    assert metrics["foreground_psnr"] == pytest.approx(0.0)
    assert metrics["masked_full_psnr"] == pytest.approx(6.0206, abs=1e-4)


def test_soft_mask_loss_penalizes_subthreshold_background_alpha() -> None:
    prediction = {
        "rgb": torch.zeros((2, 2, 3)),
        "mask": torch.full((2, 2), 0.25, requires_grad=True),
    }
    loss, parts = differentiable_render_loss(
        prediction,
        torch.zeros((2, 2, 3)),
        torch.zeros((2, 2)),
        1.0,
        1.0,
    )
    assert float(parts["mask_loss"]) == pytest.approx(0.25)
    loss.backward()
    assert prediction["mask"].grad is not None
    assert prediction["mask"].grad.abs().sum() > 0


def test_route_contribution_separates_appearance_and_silhouette() -> None:
    full = {
        "foreground_l1": 0.1,
        "foreground_psnr": 20.0,
        "mask_mae": 0.02,
        "mask_iou": 0.8,
        "mask_f1": 0.9,
    }
    ablated = {
        "shell": {**full, "foreground_l1": 0.2, "foreground_psnr": 19.0},
        "strand": {**full, "mask_mae": 0.04, "mask_iou": 0.7},
        "residual": {**full, "foreground_l1": 0.08, "foreground_psnr": 21.0},
    }
    contribution = _route_contribution(full, ablated)
    assert contribution["shell"]["objective_proxy_increase"] == pytest.approx(0.1)
    assert contribution["strand"]["objective_proxy_increase"] == pytest.approx(0.2)
    assert contribution["shell"]["normalized_appearance_impact"] == 1.0
    assert contribution["strand"]["normalized_silhouette_impact"] == 1.0
    assert contribution["residual"]["normalized_appearance_impact"] == 0.0
    assert sum(
        contribution[route]["normalized_appearance_impact"]
        for route in ("shell", "strand", "residual")
    ) == pytest.approx(1.0)
