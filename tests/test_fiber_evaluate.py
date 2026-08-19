from __future__ import annotations

import pytest
import torch

from dpd3dgs_animal.fiber_evaluate import _frame_metrics, _select_frame_indices
from dpd3dgs_animal.fiber_route_audit import _route_contribution
from dpd3dgs_animal.scaffold import differentiable_render_loss


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


def test_boundary_loss_targets_alpha_at_the_silhouette() -> None:
    prediction = {
        "rgb": torch.zeros((5, 5, 3)),
        "mask": torch.zeros((5, 5), requires_grad=True),
    }
    target_mask = torch.zeros((5, 5))
    target_mask[1:4, 1:4] = 1.0
    loss, parts = differentiable_render_loss(
        prediction,
        torch.zeros((5, 5, 3)),
        target_mask,
        0.0,
        0.0,
        mask_boundary_weight=1.0,
        mask_boundary_radius=1,
    )
    assert float(parts["mask_boundary"]) > 0.0
    loss.backward()
    assert prediction["mask"].grad is not None
    assert prediction["mask"].grad[1:4, 1:4].abs().sum() > 0


def test_balanced_mask_loss_does_not_dilute_sparse_foreground() -> None:
    prediction = {
        "rgb": torch.zeros((10, 10, 3)),
        "mask": torch.zeros((10, 10), requires_grad=True),
    }
    target_mask = torch.zeros((10, 10))
    target_mask[0, 0] = 1.0
    loss, parts = differentiable_render_loss(
        prediction,
        torch.zeros((10, 10, 3)),
        target_mask,
        0.0,
        0.0,
        mask_balance_weight=1.0,
    )
    assert float(parts["mask_soft"]) == pytest.approx(0.01)
    assert float(parts["mask_foreground"]) == pytest.approx(1.0)
    assert float(parts["mask_balanced"]) == pytest.approx(0.5)
    loss.backward()
    assert prediction["mask"].grad is not None
    assert prediction["mask"].grad[0, 0].abs() > 0.1


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
