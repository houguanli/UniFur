import numpy as np
import pytest
import torch
from PIL import Image

from dpd3dgs_animal.fiber_optimize import (
    _local_axial_orientation_moments,
    _load_orientation_targets,
    _negative_contribution_penalty,
    _normalize_positive_risk,
    _risk_calibration_kl,
    _sample_dropped_route,
    _scheduled_route_dropout_probability,
    _silhouette_band_loss,
    _split_training_and_calibration_frames,
)


def test_fourth_axial_moment_preserves_orthogonal_crossing() -> None:
    # In double-angle space, horizontal and vertical directions cancel.  Their
    # fourth harmonics agree, preserving explicit evidence for two modes.
    vectors = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]])
    confidence = torch.ones(1, 2)
    moment2, moment4, _ = _local_axial_orientation_moments(
        vectors, confidence, radius=1
    )
    assert float(moment2[0, 0].norm()) < 1e-6
    torch.testing.assert_close(moment4[0, 0], torch.tensor([1.0, 0.0]))


def test_orientation_loader_accepts_gaussian_haircut_variance(tmp_path) -> None:
    frame = tmp_path / "0001.png"
    Image.new("RGB", (2, 2)).save(frame)
    orientation = np.asarray([[0, 64], [128, 255]], dtype=np.uint8)
    Image.fromarray(orientation).save(tmp_path / "0001_orientation.png")
    variance = np.asarray([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
    np.save(tmp_path / "0001_orientation_var.npy", variance)

    target = _load_orientation_targets(
        [frame], [0], 2, 2, "cpu", str(tmp_path)
    )[0]
    assert target is not None
    expected = 1.0 / ((torch.from_numpy(variance) / np.pi**2) ** 2 + 1e-7)
    torch.testing.assert_close(target["confidence"], expected)
    torch.testing.assert_close(
        torch.linalg.vector_norm(target["vectors"], dim=-1), torch.ones(2, 2)
    )


def test_calibration_frames_are_disjoint_orbit_distributed_holdouts() -> None:
    train, calibration = _split_training_and_calibration_frames(list(range(10)), 3)
    assert train == [1, 2, 3, 5, 6, 7, 8]
    assert calibration == [0, 4, 9]
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


def test_zero_positive_risk_falls_back_only_to_residual_teacher() -> None:
    target = _normalize_positive_risk(torch.tensor([-0.2, 0.0, -0.1]), 1e-4)
    torch.testing.assert_close(target, torch.tensor([0.0, 0.0, 1.0]))


def test_negative_contribution_directly_penalizes_bad_route_mass() -> None:
    probabilities = torch.tensor([[0.6, 0.3, 0.1], [0.4, 0.5, 0.1]])
    penalty = _negative_contribution_penalty(
        probabilities, torch.tensor([0.2, 0.0, 0.0])
    )
    assert float(penalty) == pytest.approx(0.1)


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


def test_route_dropout_anneals_only_during_structured_refinement() -> None:
    assert _scheduled_route_dropout_probability(
        100, 1000, 400, "gaussian_scaffold", 0.6, 0.0
    ) == 0.0
    assert _scheduled_route_dropout_probability(
        300, 1000, 400, "soft_routing", 0.6, 0.0
    ) == pytest.approx(0.6)
    assert _scheduled_route_dropout_probability(
        400, 1000, 400, "structured_refinement", 0.6, 0.0
    ) == pytest.approx(0.6)
    assert _scheduled_route_dropout_probability(
        999, 1000, 400, "structured_refinement", 0.6, 0.0
    ) == pytest.approx(0.0)
    assert _scheduled_route_dropout_probability(
        999, 1000, 400, "structured_refinement", 0.6, 0.25
    ) == pytest.approx(0.15)


def test_fin_silhouette_band_rewards_boundary_and_rejects_interior_fill() -> None:
    target = torch.zeros((17, 17))
    target[5:12, 5:12] = 1.0
    empty = torch.zeros_like(target)
    filled = target.clone()
    # A two-pixel morphological band around the square is the ideal Fin mask.
    image = target[None, None]
    dilated = torch.nn.functional.max_pool2d(image, 5, stride=1, padding=2)[0, 0]
    eroded = -torch.nn.functional.max_pool2d(-image, 5, stride=1, padding=2)[0, 0]
    band = (dilated - eroded).clamp(0.0, 1.0)
    ideal_loss = _silhouette_band_loss(band, target, radius=2)
    assert float(ideal_loss) < float(_silhouette_band_loss(empty, target, radius=2))
    assert float(ideal_loss) < float(_silhouette_band_loss(filled, target, radius=2))
