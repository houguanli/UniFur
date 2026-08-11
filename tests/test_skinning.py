from pathlib import Path

import numpy as np
import torch

from dpd3dgs_animal.mesh import (
    skin_points_by_bone_dqs,
    skin_points_by_bone_lbs,
)
from dpd3dgs_animal.mocap_adapter import parse_bvh_hierarchy
from dpd3dgs_animal.optimize import (
    skin_points_by_bone_dqs_torch,
    skin_points_by_bone_transforms_torch,
)


def test_cat_bvh_parents_are_hierarchical() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "samples/mocap_anything/zoo/bvh/Cat#Cat-Walk/y30.bvh"
    names, parents = parse_bvh_hierarchy(path)
    assert names[:8] == [
        "Hips",
        "Bip01_Pelvis",
        "BN_Tail_01",
        "BN_Tail_02",
        "BN_Tail_03",
        "BN_Tail_04",
        "Bip01_Spine",
        "Bip01_R_Thigh",
    ]
    assert parents[:8] == [-1, 0, 1, 2, 3, 4, 1, 6]


def test_skinning_rest_pose_is_exact() -> None:
    points, rest, _posed, parents, weights = _quarter_turn_fixture()
    np.testing.assert_allclose(
        skin_points_by_bone_lbs(points, rest, rest, parents, weights),
        points,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        skin_points_by_bone_dqs(points, rest, rest, parents, weights),
        points,
        atol=1e-6,
    )


def test_dqs_matches_single_bone_rigid_rotation_in_numpy_and_torch() -> None:
    points, rest, posed, parents, weights = _quarter_turn_fixture()
    expected = np.asarray([[-0.2, 0.5, 0.0], [-0.2, 1.0, 0.0]], dtype=np.float32)
    actual = skin_points_by_bone_dqs(points, rest, posed, parents, weights)
    torch_actual = skin_points_by_bone_dqs_torch(
        torch.from_numpy(points),
        torch.from_numpy(rest),
        torch.from_numpy(posed),
        torch.from_numpy(parents),
        torch.from_numpy(weights),
    ).numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    np.testing.assert_allclose(torch_actual, expected, atol=1e-6)


def test_matrix_lbs_uses_full_rigid_transform_including_axial_rotation() -> None:
    points = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    rest = torch.eye(4).reshape(1, 4, 4)
    posed = rest.clone()
    posed[0, :3, :3] = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    posed[0, :3, 3] = torch.tensor([0.5, -0.25, 1.0])
    weights = torch.ones((2, 1))

    actual = skin_points_by_bone_transforms_torch(points, rest, posed, weights)
    expected = torch.tensor([[0.5, 0.75, 1.0], [-1.5, -0.25, 1.0]])
    torch.testing.assert_close(actual, expected)


def test_matrix_lbs_blends_two_exact_bone_transforms() -> None:
    points = torch.tensor([[0.0, 0.0, 0.0]])
    rest = torch.eye(4).repeat(2, 1, 1)
    posed = rest.clone()
    posed[0, 0, 3] = 2.0
    posed[1, 1, 3] = 4.0
    weights = torch.tensor([[0.25, 0.75]])

    actual = skin_points_by_bone_transforms_torch(points, rest, posed, weights)
    torch.testing.assert_close(actual, torch.tensor([[0.5, 3.0, 0.0]]))


def _quarter_turn_fixture():
    rest = np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    posed = np.asarray([[0, 0, 0], [0, 1, 0]], dtype=np.float32)
    parents = np.asarray([-1, 0], dtype=np.int64)
    points = np.asarray([[0.5, 0.2, 0], [1.0, 0.2, 0]], dtype=np.float32)
    weights = np.asarray([[0, 1], [0, 1]], dtype=np.float32)
    return points, rest, posed, parents, weights
