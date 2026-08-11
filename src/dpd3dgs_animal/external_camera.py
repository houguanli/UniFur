from __future__ import annotations

import numpy as np


def as_homogeneous(matrix: np.ndarray) -> np.ndarray:
    """Return a float64 4x4 transform from a 3x4 or 4x4 matrix."""

    value = np.asarray(matrix, dtype=np.float64)
    if value.shape == (4, 4):
        return value.copy()
    if value.shape != (3, 4):
        raise ValueError(f"Expected a 3x4 or 4x4 transform, got {value.shape}")
    result = np.eye(4, dtype=np.float64)
    result[:3, :4] = value
    return result


def estimate_camera_unit_scale(
    learned_reference_w2c: np.ndarray,
    official_reference_w2c: np.ndarray,
) -> float:
    """Estimate learned-units per official-unit from reference camera depth.

    Monocular reconstructions have an arbitrary global scale.  The official
    target/reference relative camera translation therefore cannot be applied
    directly to a learned field.  This ratio maps the official baseline into
    the learned camera units while leaving relative rotation unchanged.
    """

    learned = as_homogeneous(learned_reference_w2c)
    official = as_homogeneous(official_reference_w2c)
    learned_distance = float(np.linalg.norm(learned[:3, 3]))
    official_distance = float(np.linalg.norm(official[:3, 3]))
    if official_distance <= 1e-8 or learned_distance <= 1e-8:
        raise ValueError("Camera translation norm is too small to estimate scale")
    return learned_distance / official_distance


def transfer_relative_camera(
    learned_reference_w2c: np.ndarray,
    official_reference_w2c: np.ndarray,
    official_target_w2c: np.ndarray,
    unit_scale: float,
) -> np.ndarray:
    """Transfer an official relative camera to a learned monocular field.

    If learned camera coordinates equal ``unit_scale`` times official camera
    coordinates for the reference view, the target camera is obtained by
    applying the official reference-to-target rotation and a correspondingly
    scaled translation to the learned reference field-to-camera transform.
    """

    if not np.isfinite(unit_scale) or unit_scale <= 0:
        raise ValueError("unit_scale must be finite and positive")
    learned = as_homogeneous(learned_reference_w2c)
    reference = as_homogeneous(official_reference_w2c)
    target = as_homogeneous(official_target_w2c)
    relative = target @ np.linalg.inv(reference)
    relative[:3, 3] *= float(unit_scale)
    return relative @ learned
