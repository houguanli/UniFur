import numpy as np
import pytest
import torch

from dpd3dgs_animal.fiber_optimize import (
    _normalize_positive_risk,
    _risk_calibration_kl,
    _sample_dropped_route,
    _split_training_and_calibration_frames,
)


def test_calibration_frames_are_disjoint_tail_holdouts() -> None:
    train, calibration = _split_training_and_calibration_frames(list(range(10)), 3)
    assert train == list(range(7))
    assert calibration == [7, 8, 9]
    assert set(train).isdisjoint(calibration)


def test_calibration_split_must_leave_a_training_frame() -> None:
    with pytest.raises(ValueError):
        _split_training_and_calibration_frames([0, 1], 2)


def test_positive_risk_target_and_kl_follow_ablation_contribution() -> None:
    target = _normalize_positive_risk(
        torch.tensor([0.4, 0.2, -0.1]), floor=1e-4
    )
    assert int(target.argmax()) == 0
    assert target[2] < 1e-3
    torch.testing.assert_close(target.sum(), torch.tensor(1.0))

    matched = target.repeat(5, 1)
    mismatched = torch.full((5, 3), 1.0 / 3.0)
    assert float(_risk_calibration_kl(matched, target)) < 1e-6
    assert float(_risk_calibration_kl(mismatched, target)) > 0.1


def test_route_dropout_sampling_is_seeded_and_skips_scaffold() -> None:
    rng_a = np.random.default_rng(7)
    rng_b = np.random.default_rng(7)
    sequence_a = [
        _sample_dropped_route(rng_a, "soft_routing", 0.5) for _ in range(20)
    ]
    sequence_b = [
        _sample_dropped_route(rng_b, "soft_routing", 0.5) for _ in range(20)
    ]
    assert sequence_a == sequence_b
    assert any(route is not None for route in sequence_a)
    assert _sample_dropped_route(rng_a, "gaussian_scaffold", 1.0) is None
    assert (
        _sample_dropped_route(
            np.random.default_rng(1),
            "structured_refinement",
            1.0,
            residual_bias=1.0,
        )
        == "residual"
    )
