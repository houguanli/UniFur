from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_hairgs_static_protocol.py"
SPEC = importlib.util.spec_from_file_location("prepare_hairgs_static_protocol", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_hairgs_json_camera_is_inverted_from_camera_to_world() -> None:
    angle = np.deg2rad(90.0)
    camera_to_world_rotation = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angle), -np.sin(angle)],
            [0.0, np.sin(angle), np.cos(angle)],
        ]
    )
    position = np.asarray([0.0, 0.3, 0.0])
    matrix = np.asarray(
        MODULE._world_to_camera(
            {"rotation": camera_to_world_rotation.tolist(), "position": position.tolist()}
        )
    )

    expected = np.eye(4)
    expected[:3, :3] = camera_to_world_rotation.T
    expected[:3, 3] = -camera_to_world_rotation.T @ position
    np.testing.assert_allclose(matrix, expected, atol=1e-12)

    camera_origin = np.concatenate([position, [1.0]])
    np.testing.assert_allclose(matrix @ camera_origin, [0.0, 0.0, 0.0, 1.0], atol=1e-12)


def test_hairgs_overhead_camera_places_world_origin_in_front() -> None:
    camera = {
        "position": [0.0, 0.3171228218078613, 0.0],
        "rotation": [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
    }
    world_to_camera = np.asarray(MODULE._world_to_camera(camera))
    camera_space_origin = world_to_camera @ np.asarray([0.0, 0.0, 0.0, 1.0])
    assert camera_space_origin[2] > 0.0
