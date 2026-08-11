from __future__ import annotations

import json

import pytest

from dpd3dgs_animal.observations import load_observation_manifest


def test_manifest_maps_per_image_cameras_and_shared_motion(tmp_path) -> None:
    frame_paths = [tmp_path / "0000.png", tmp_path / "0001.png"]
    payload = {
        "observations": [
            {
                "image": "0001.png",
                "motion_index": 0,
                "intrinsics": [200, 100, 120, 110, 100, 50],
                "world_to_camera": [
                    [1, 0, 0, 1],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ],
            },
            {
                "image": "0000.png",
                "motion_index": 0,
                "intrinsics": [200, 100, 100, 100, 100, 50],
                "world_to_camera": [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ],
            },
        ]
    }
    manifest = tmp_path / "cameras.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    observations = load_observation_manifest(
        manifest, frame_paths=frame_paths, width=100, height=50
    )

    assert observations.motion_indices == (0, 0)
    assert observations.cameras[0].fx == pytest.approx(50.0)
    assert observations.cameras[1].fx == pytest.approx(60.0)
    assert observations.cameras[0].world_to_camera[0, 3] == pytest.approx(0.0)
    assert observations.cameras[1].world_to_camera[0, 3] == pytest.approx(1.0)


def test_manifest_rejects_missing_image_observation(tmp_path) -> None:
    manifest = tmp_path / "cameras.json"
    manifest.write_text(
        json.dumps(
            {
                "observations": [
                    {
                        "image": "0000.png",
                        "intrinsics": [100, 50, 50, 50, 50, 25],
                        "world_to_camera": [
                            [1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1],
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="0001.png"):
        load_observation_manifest(
            manifest,
            frame_paths=[tmp_path / "0000.png", tmp_path / "0001.png"],
            width=100,
            height=50,
        )
