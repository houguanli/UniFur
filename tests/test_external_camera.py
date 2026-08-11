import numpy as np

from dpd3dgs_animal.external_camera import (
    as_homogeneous,
    estimate_camera_unit_scale,
    transfer_relative_camera,
)


def test_as_homogeneous_accepts_3x4() -> None:
    matrix = np.concatenate([np.eye(3), np.ones((3, 1))], axis=1)
    result = as_homogeneous(matrix)
    assert result.shape == (4, 4)
    np.testing.assert_allclose(result[:3], matrix)
    np.testing.assert_allclose(result[3], [0, 0, 0, 1])


def test_relative_camera_transfer_preserves_scaled_camera_coordinates() -> None:
    angle = np.deg2rad(35.0)
    target_rotation = np.asarray(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )
    official_reference = np.eye(4)
    official_reference[:3, 3] = [0.2, -0.1, 4.0]
    official_target = np.eye(4)
    official_target[:3, :3] = target_rotation
    official_target[:3, 3] = [-1.1, 0.2, 3.8]
    unit_scale = 0.37

    learned_reference = np.eye(4)
    learned_reference[:3, 3] = unit_scale * official_reference[:3, 3]
    learned_target = transfer_relative_camera(
        learned_reference,
        official_reference,
        official_target,
        unit_scale,
    )
    points = np.asarray(
        [[0.1, 0.2, 0.3, 1.0], [-0.4, 0.5, 0.1, 1.0], [0.7, -0.2, 0.0, 1.0]]
    )
    official_camera_points = (official_target @ points.T).T[:, :3]
    learned_points = points.copy()
    learned_points[:, :3] *= unit_scale
    learned_camera_points = (learned_target @ learned_points.T).T[:, :3]
    np.testing.assert_allclose(
        learned_camera_points,
        unit_scale * official_camera_points,
        atol=1e-7,
    )


def test_camera_unit_scale_uses_translation_norm() -> None:
    official = np.eye(4)
    official[:3, 3] = [0.0, 0.0, 4.0]
    learned = np.eye(4)
    learned[:3, 3] = [0.0, 0.0, 1.0]
    assert estimate_camera_unit_scale(learned, official) == 0.25
