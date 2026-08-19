import torch

from dpd3dgs_animal.scaffold import (
    skin_points_by_bone_dqs_torch,
    skin_points_by_bone_transforms_torch,
)


def test_dqs_matches_single_bone_quarter_turn() -> None:
    rest = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    posed = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    parents = torch.tensor([-1, 0])
    points = torch.tensor([[0.5, 0.2, 0.0], [1.0, 0.2, 0.0]])
    weights = torch.tensor([[0.0, 1.0], [0.0, 1.0]])

    actual = skin_points_by_bone_dqs_torch(
        points, rest, posed, parents, weights
    )
    expected = torch.tensor([[-0.2, 0.5, 0.0], [-0.2, 1.0, 0.0]])
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_matrix_lbs_uses_full_rigid_transform() -> None:
    points = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    rest = torch.eye(4).reshape(1, 4, 4)
    posed = rest.clone()
    posed[0, :3, :3] = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    posed[0, :3, 3] = torch.tensor([0.5, -0.25, 1.0])

    actual = skin_points_by_bone_transforms_torch(
        points, rest, posed, torch.ones((2, 1))
    )
    expected = torch.tensor([[0.5, 0.75, 1.0], [-1.5, -0.25, 1.0]])
    torch.testing.assert_close(actual, expected)


def test_matrix_lbs_blends_two_bone_transforms() -> None:
    points = torch.tensor([[0.0, 0.0, 0.0]])
    rest = torch.eye(4).repeat(2, 1, 1)
    posed = rest.clone()
    posed[0, 0, 3] = 2.0
    posed[1, 1, 3] = 4.0

    actual = skin_points_by_bone_transforms_torch(
        points, rest, posed, torch.tensor([[0.25, 0.75]])
    )
    torch.testing.assert_close(actual, torch.tensor([[0.5, 3.0, 0.0]]))
