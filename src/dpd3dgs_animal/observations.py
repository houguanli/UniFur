from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .render import (
    PinholeCamera,
    camera_from_arrays,
    camera_from_stage1_npz,
    default_camera_for_vertices,
)


@dataclass(frozen=True)
class ObservationSet:
    """Per-image camera and deformation-state assignments.

    A monocular dynamic sequence uses one camera for every image and maps image
    ``i`` to motion state ``i``.  A static multi-view capture instead assigns a
    different camera to every image while mapping every image to motion state
    zero.  The same structure also covers synchronized dynamic multi-view data.
    """

    cameras: tuple[PinholeCamera, ...]
    motion_indices: tuple[int, ...]
    source: str


def resolve_observations(
    frame_paths: list[Path],
    stage1_npz: str | Path,
    rest_vertices: np.ndarray,
    width: int,
    height: int,
    camera_manifest: str | Path | None = None,
) -> ObservationSet:
    if camera_manifest is not None:
        return load_observation_manifest(
            camera_manifest,
            frame_paths=frame_paths,
            width=width,
            height=height,
        )

    camera = camera_from_stage1_npz(str(stage1_npz), width, height)
    source = "stage1_npz"
    if camera is None:
        camera = default_camera_for_vertices(rest_vertices, width, height)
        source = "default"
    return ObservationSet(
        cameras=tuple(camera for _ in frame_paths),
        motion_indices=tuple(range(len(frame_paths))),
        source=source,
    )


def load_observation_manifest(
    path: str | Path,
    *,
    frame_paths: list[Path],
    width: int,
    height: int,
) -> ObservationSet:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    observations = payload.get("observations")
    if not isinstance(observations, list) or not observations:
        raise ValueError("Camera manifest must contain a non-empty observations list")

    by_image: dict[str, dict] = {}
    for item in observations:
        if not isinstance(item, dict) or "image" not in item:
            raise ValueError("Every manifest observation must contain an image field")
        name = Path(str(item["image"])).name
        if name in by_image:
            raise ValueError(f"Duplicate observation for image {name!r}")
        by_image[name] = item

    cameras: list[PinholeCamera] = []
    motion_indices: list[int] = []
    for frame_path in frame_paths:
        item = by_image.get(frame_path.name)
        if item is None:
            raise ValueError(
                f"Camera manifest {manifest_path} has no observation for "
                f"{frame_path.name!r}"
            )
        intrinsics = np.asarray(item["intrinsics"], dtype=np.float32)
        world_to_camera = np.asarray(item["world_to_camera"], dtype=np.float32)
        if intrinsics.shape != (6,):
            raise ValueError(
                f"Observation {frame_path.name!r} intrinsics must have shape (6,)"
            )
        if world_to_camera.shape != (4, 4):
            raise ValueError(
                f"Observation {frame_path.name!r} world_to_camera must have shape (4, 4)"
            )
        cameras.append(
            camera_from_arrays(
                intrinsics,
                world_to_camera,
                width,
                height,
                image_y_down=bool(item.get("image_y_down", True)),
            )
        )
        motion_index = int(item.get("motion_index", 0))
        if motion_index < 0:
            raise ValueError("motion_index must be non-negative")
        motion_indices.append(motion_index)

    return ObservationSet(
        cameras=tuple(cameras),
        motion_indices=tuple(motion_indices),
        source=f"manifest:{manifest_path}",
    )
