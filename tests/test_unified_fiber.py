import numpy as np
import torch

from dpd3dgs_animal.fiber_optimize import (
    _apply_route_mass_floor,
    _load_residual_bootstrap_checkpoint,
)

from dpd3dgs_animal.fiber import (
    UnifiedFiberField,
    _quaternion_to_matrix_torch,
    mass_preserving_route_ids,
    render_fiber_primitives,
)
from dpd3dgs_animal.render import PinholeCamera


def _toy_field() -> tuple[UnifiedFiberField, torch.Tensor, torch.Tensor]:
    vertices = torch.tensor(
        [[-0.5, -0.5, 2.0], [0.5, -0.5, 2.0], [-0.5, 0.5, 2.0]],
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
    field = UnifiedFiberField(
        face_index=torch.tensor([0, 0]),
        barycentric=torch.tensor([[0.3, 0.4, 0.3], [0.2, 0.2, 0.6]]),
        color=torch.tensor([[0.7, 0.4, 0.2], [0.2, 0.3, 0.7]]),
        opacity=torch.tensor([0.6, 0.7]),
        original_scaling=torch.full((2, 3), 0.015),
        original_rotation=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
        rest_surface_frame=torch.eye(3).repeat(2, 1, 1),
        residual_offset_local=torch.tensor([[0.0, 0.0, 0.02], [0.02, 0.0, 0.01]]),
        direction_local=torch.tensor([[0.0, 0.0, 1.0], [0.5, 0.0, 1.0]]),
        height=torch.tensor([0.005, 0.01]),
        shell_length=torch.tensor([0.02, 0.025]),
        strand_length=torch.tensor([0.08, 0.1]),
        radius=torch.tensor([0.005, 0.006]),
        route_logits=torch.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, -1.0]]),
        scene_scale=1.0,
        route_neighbor_index=torch.tensor([[1], [0]]),
    )
    return field, vertices, faces


def test_unified_routes_form_a_differentiable_partition() -> None:
    field, vertices, faces = _toy_field()
    primitives = field.primitives(
        vertices, faces, shell_samples=2, strand_samples=3, temperature=0.8
    )
    assert primitives.xyz.shape == (2 * (2 + 3 + 1), 3)
    torch.testing.assert_close(
        primitives.route_probabilities.sum(dim=-1), torch.ones(2)
    )
    assert torch.all(primitives.scaling > 0)
    assert torch.isfinite(primitives.rotation).all()

    route_weighted_loss = (
        primitives.opacity * (1.0 + primitives.route_id.to(torch.float32))
    ).mean() + 0.01 * primitives.xyz.square().mean()
    route_weighted_loss.backward()
    assert field.route_logits.grad is not None
    assert float(field.route_logits.grad.abs().sum()) > 0.0
    assert field.residual_trust_logits.grad is not None
    assert torch.isfinite(field.residual_trust_logits.grad).all()
    assert field.direction_local_raw.grad is not None
    assert torch.isfinite(field.direction_local_raw.grad).all()


def test_surface_anchor_moves_all_three_representations_with_the_animal() -> None:
    field, vertices, faces = _toy_field()
    translation = torch.tensor([0.25, -0.1, 0.4])
    before = field.primitives(vertices, faces, shell_samples=2, strand_samples=3)
    after = field.primitives(
        vertices + translation, faces, shell_samples=2, strand_samples=3
    )
    torch.testing.assert_close(
        after.xyz - before.xyz,
        translation.expand_as(before.xyz),
        atol=1e-6,
        rtol=1e-6,
    )


def test_residual_covariance_follows_rigid_surface_rotation() -> None:
    field, vertices, faces = _toy_field()
    angle = torch.tensor(0.7)
    cosine, sine = torch.cos(angle), torch.sin(angle)
    rotation = torch.stack(
        [
            torch.stack([cosine, -sine, torch.tensor(0.0)]),
            torch.stack([sine, cosine, torch.tensor(0.0)]),
            torch.tensor([0.0, 0.0, 1.0]),
        ]
    )
    before = field.residual_primitives(vertices, faces)
    after = field.residual_primitives(vertices @ rotation.T, faces)
    before_matrix = _quaternion_to_matrix_torch(before.rotation)
    after_matrix = _quaternion_to_matrix_torch(after.rotation)
    expected_matrix = rotation[None] @ before_matrix
    torch.testing.assert_close(
        after_matrix, expected_matrix, atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        after.xyz, before.xyz @ rotation.T, atol=1e-6, rtol=1e-6
    )


def test_identity_covariance_transport_has_finite_backward() -> None:
    field, vertices, faces = _toy_field()
    primitives = field.residual_primitives(vertices, faces)
    loss = primitives.rotation[:, 1:].square().sum()
    loss.backward()
    assert field.residual_rotation_raw.grad is not None
    assert torch.isfinite(field.residual_rotation_raw.grad).all()


def test_route_continuation_is_residual_at_start_and_hard_at_end() -> None:
    field, _vertices, _faces = _toy_field()
    start = field.route_probabilities(route_blend=0.0)
    torch.testing.assert_close(start[:, 2], torch.ones(2))
    torch.testing.assert_close(start[:, :2], torch.zeros(2, 2))

    soft = field.route_probabilities(temperature=0.8)
    hard = field.route_probabilities(temperature=0.8, hardening=1.0)
    torch.testing.assert_close(hard.sum(dim=-1), torch.ones(2))
    torch.testing.assert_close(hard, torch.nn.functional.one_hot(
        soft.argmax(dim=-1), num_classes=3
    ).to(hard.dtype))


def test_mass_preserving_hard_routes_match_soft_expected_counts() -> None:
    probabilities = torch.tensor(
        [
            [0.60, 0.30, 0.10],
            [0.59, 0.31, 0.10],
            [0.61, 0.29, 0.10],
            [0.30, 0.40, 0.30],
            [0.31, 0.39, 0.30],
            [0.29, 0.41, 0.30],
            [0.10, 0.20, 0.70],
            [0.20, 0.10, 0.70],
            [0.25, 0.05, 0.70],
            [0.15, 0.15, 0.70],
        ]
    )
    route_ids = mass_preserving_route_ids(probabilities)
    expected = torch.floor(probabilities.sum(dim=0)).to(torch.long)
    expected[torch.argmax(probabilities.sum(dim=0) - expected)] += 1
    actual = torch.bincount(route_ids, minlength=3)
    torch.testing.assert_close(actual, expected)
    # The hard policy produces a one-hot primal path while preserving gradients.
    field, _vertices, _faces = _toy_field()
    hard = field.route_probabilities(hard=True, hard_policy="mass_preserving")
    torch.testing.assert_close(hard.sum(dim=-1), torch.ones(2))
    hard[:, 0].sum().backward()
    assert field.route_logits.grad is not None
    assert float(field.route_logits.grad.abs().sum()) > 0.0


def test_residual_bootstrap_transfers_only_residual_scaffold(tmp_path) -> None:
    source, _vertices, _faces = _toy_field()
    with torch.no_grad():
        source.color_logits.fill_(0.3)
        source.opacity_logits.fill_(-0.7)
        source.residual_offset_local.fill_(0.04)
        source.residual_log_scale_delta.fill_(0.02)
        source.residual_rotation_raw.fill_(0.01)
        source.route_logits.fill_(4.0)
    checkpoint = tmp_path / "residual.pt"
    torch.save(
        {
            "state_dict": source.state_dict(),
            "metadata": {
                "representation": "residual_only",
                "point_count": source.point_count,
            },
        },
        checkpoint,
    )
    target, _vertices, _faces = _toy_field()
    original_routes = target.route_logits.detach().clone()
    loaded = _load_residual_bootstrap_checkpoint(target, checkpoint)
    torch.testing.assert_close(target.color_logits, source.color_logits)
    torch.testing.assert_close(target.opacity_logits, source.opacity_logits)
    torch.testing.assert_close(target.residual_offset_local, source.residual_offset_local)
    torch.testing.assert_close(target.route_logits, original_routes)
    assert loaded["source_representation"] == "residual_only"


def test_residual_bootstrap_can_seed_structured_effective_route_mass(tmp_path) -> None:
    source, _vertices, _faces = _toy_field()
    checkpoint = tmp_path / "residual.pt"
    torch.save(
        {
            "state_dict": source.state_dict(),
            "metadata": {
                "representation": "residual_only",
                "point_count": source.point_count,
            },
        },
        checkpoint,
    )
    target, _vertices, _faces = _toy_field()
    with torch.no_grad():
        target.residual_trust_logits.fill_(torch.logit(torch.tensor(0.2)))
    loaded = _load_residual_bootstrap_checkpoint(
        target,
        checkpoint,
        bootstrap_route_mass=[0.45, 0.20, 0.35],
        bootstrap_route_temperature=0.35,
    )
    expected = torch.tensor([0.45, 0.20, 0.35])
    torch.testing.assert_close(
        target.route_probabilities(temperature=0.35).mean(dim=0), expected
    )
    torch.testing.assert_close(target.initial_route_probabilities.mean(dim=0), expected)
    np.testing.assert_allclose(
        loaded["routing_bootstrap_mass"], [0.45, 0.2, 0.35], atol=1e-6
    )


def test_contribution_target_preserves_minimum_route_mass() -> None:
    target = torch.tensor([0.0, 0.0, 1.0])
    result = _apply_route_mass_floor(target, [0.25, 0.05, 0.20])
    torch.testing.assert_close(result, torch.tensor([0.25, 0.05, 0.70]))


def test_learned_residual_trust_prevents_early_structured_takeover() -> None:
    field, _vertices, _faces = _toy_field()
    probabilities = field.route_probabilities(temperature=1.0)
    assert torch.all(probabilities[:, 2] >= 0.95)
    probabilities[:, 2].mean().backward()
    assert field.residual_trust_logits.grad is not None
    assert float(field.residual_trust_logits.grad.abs().sum()) > 0.0


def test_structured_geometry_starts_as_render_preserving_residual_copies() -> None:
    field, vertices, faces = _toy_field()
    residual = field.residual_primitives(vertices, faces)
    primitives = field.primitives(
        vertices,
        faces,
        shell_samples=2,
        strand_samples=3,
        forced_route="shell",
        geometry_blend=0.0,
    )

    shell_count = field.point_count * 2
    expected_xyz = residual.xyz.repeat_interleave(2, dim=0)
    expected_scaling = residual.scaling.repeat_interleave(2, dim=0)
    expected_rotation = residual.rotation.repeat_interleave(2, dim=0)
    torch.testing.assert_close(primitives.xyz[:shell_count], expected_xyz)
    torch.testing.assert_close(primitives.scaling[:shell_count], expected_scaling)
    torch.testing.assert_close(primitives.rotation[:shell_count], expected_rotation)


def test_route_dropout_removes_one_expert_and_renormalizes_soft_mass() -> None:
    field, _vertices, _faces = _toy_field()
    probabilities = field.route_probabilities(
        temperature=0.8, dropped_route="residual"
    )
    torch.testing.assert_close(probabilities[:, 2], torch.zeros(2))
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(2))
    assert torch.all(probabilities[:, :2] > 0.0)


def test_residual_only_is_compact_anisotropic_3dgs() -> None:
    field, vertices, faces = _toy_field()
    primitives = field.residual_primitives(vertices, faces)
    assert primitives.xyz.shape == (field.point_count, 3)
    assert primitives.scaling.shape == (field.point_count, 3)
    assert primitives.rotation.shape == (field.point_count, 4)
    torch.testing.assert_close(
        primitives.route_probabilities[:, 2], torch.ones(field.point_count)
    )
    assert torch.all(primitives.route_id == 2)

    objective = (
        primitives.xyz.square().mean()
        + primitives.scaling.square().mean()
        + primitives.rotation.square().mean()
    )
    objective.backward()
    assert field.residual_offset_local.grad is not None
    assert field.residual_log_scale_delta.grad is not None
    assert field.residual_rotation_raw.grad is not None


def test_lightweight_fiber_renderer_backpropagates_to_geometry_and_routing() -> None:
    field, vertices, faces = _toy_field()
    camera = PinholeCamera(
        width=48,
        height=48,
        fx=45.0,
        fy=45.0,
        cx=24.0,
        cy=24.0,
        world_to_camera=np.eye(4, dtype=np.float32),
        image_y_down=True,
    )
    primitives = field.primitives(vertices, faces, shell_samples=2, strand_samples=3)
    prediction = render_fiber_primitives(primitives, camera, radius_px=3)
    objective = prediction["mask"].square().mean() + prediction["rgb"].mean()
    objective.backward()
    assert field.route_logits.grad is not None
    assert field.radius_raw.grad is not None
    assert torch.isfinite(field.route_logits.grad).all()
    assert torch.isfinite(field.radius_raw.grad).all()

    regularizers = field.regularizers(vertices, faces)
    assert set(regularizers) == {
        "route_entropy",
        "route_prior",
        "route_neighbor",
        "shell_normal",
        "shell_length",
        "strand_thinness",
        "height",
        "bend",
        "residual_drift",
        "residual_trust",
    }
    assert all(torch.isfinite(value) for value in regularizers.values())
    assert float(regularizers["route_neighbor"]) > 0.0
