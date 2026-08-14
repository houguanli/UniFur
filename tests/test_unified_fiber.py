import numpy as np
import torch

from dpd3dgs_animal.fiber_optimize import (
    _apply_route_mass_floor,
    _load_residual_bootstrap_checkpoint,
    _resolve_fiber_point_budget,
)
from dpd3dgs_animal.config import PipelineConfig

from dpd3dgs_animal.fiber import (
    CARRIER_NAMES,
    UnifiedFiberField,
    _bind_exact_surface_vertices,
    _quaternion_to_matrix_torch,
    _select_gaussian_indices,
    apply_fin_view_gate,
    deform_simulation_asset,
    edit_structured_fibers,
    fin_grazing_gate,
    mass_preserving_route_ids,
    render_fiber_primitives,
    simulation_asset_summary,
)
from dpd3dgs_animal.render import PinholeCamera


def test_pixel_adaptive_capacity_uses_resolution_and_hard_cap() -> None:
    cfg = PipelineConfig(
        fiber_capacity_mode="pixel_adaptive",
        fiber_min_points=20_000,
        fiber_max_points=43_662,
        fiber_target_pixels_per_point=6.0,
    )
    assert _resolve_fiber_point_budget(cfg, (512, 512)) == 43_662
    assert _resolve_fiber_point_budget(cfg, (320, 180)) == 20_000
    assert (
        _resolve_fiber_point_budget(
            cfg, (512, 512), explicit_max_points=12_345
        )
        == 12_345
    )


def test_spatial_morton_sampling_is_deterministic_and_covers_extent() -> None:
    grid = np.stack(
        np.meshgrid(
            np.linspace(-1.0, 1.0, 10),
            np.linspace(-2.0, 2.0, 8),
            np.linspace(-3.0, 3.0, 6),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)
    # Deliberately scramble topology/order; the selected subset must depend on
    # geometry, not this arbitrary PLY ordering.
    grid = grid[np.random.default_rng(7).permutation(grid.shape[0])]
    first = _select_gaussian_indices(grid, 96, mode="spatial_morton")
    second = _select_gaussian_indices(grid, 96, mode="spatial_morton")
    np.testing.assert_array_equal(first, second)
    assert np.unique(first).shape[0] == 96
    selected = grid[first]
    assert np.all(selected.min(axis=0) < np.array([-0.7, -1.4, -2.0]))
    assert np.all(selected.max(axis=0) > np.array([0.7, 1.4, 2.0]))


def test_exact_vertex_binding_reconstructs_selected_mesh_vertices() -> None:
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    selected = np.asarray([3, 1, 0], dtype=np.int64)
    binding = _bind_exact_surface_vertices(selected, faces, None)
    assert binding is not None
    triangles = vertices[faces[binding.face_index]]
    roots = (binding.barycentric[..., None] * triangles).sum(axis=1)
    np.testing.assert_allclose(roots, vertices[selected])


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


def test_expert_appearance_is_render_preserving_at_initialization() -> None:
    field, vertices, faces = _toy_field()
    primitives = field.primitives(
        vertices, faces, shell_samples=2, strand_samples=3, temperature=0.8
    )
    expected = field.color[primitives.source_id]
    torch.testing.assert_close(primitives.color, expected)


def test_expert_appearance_receives_route_specific_gradients() -> None:
    field, vertices, faces = _toy_field()
    primitives = field.primitives(
        vertices, faces, shell_samples=2, strand_samples=3, temperature=0.8
    )
    primitives.color[primitives.route_id == 1].sum().backward()
    gradient = field.expert_color_delta.grad
    assert gradient is not None
    assert float(gradient[:, 1].abs().sum()) > 0.0
    torch.testing.assert_close(gradient[:, 0], torch.zeros_like(gradient[:, 0]))
    torch.testing.assert_close(gradient[:, 2], torch.zeros_like(gradient[:, 2]))


def test_cubic_strand_bend_adds_inflection_and_receives_gradients() -> None:
    field, vertices, faces = _toy_field()
    baseline = field.primitives(
        vertices,
        faces,
        shell_samples=1,
        strand_samples=5,
        forced_route="strand",
    )
    strand_slice = slice(field.point_count, field.point_count * 6)
    with torch.no_grad():
        field.structured_delta_raw[:, 1] = 1.0
        field.bend_local[:, 0] = 0.25
        field.bend_cubic_local[:, 0] = -0.35
    curved = field.primitives(
        vertices,
        faces,
        shell_samples=1,
        strand_samples=5,
        forced_route="strand",
    )
    assert not torch.allclose(
        curved.xyz[strand_slice], baseline.xyz[strand_slice]
    )
    curved.xyz[strand_slice].square().mean().backward()
    assert field.bend_cubic_local.grad is not None
    assert float(field.bend_cubic_local.grad.abs().sum()) > 0.0
    assert torch.isfinite(field.bend_cubic_local.grad).all()


def test_structured_geometry_is_an_exact_zero_initialized_residual_increment() -> None:
    field, vertices, faces = _toy_field()
    primitives = field.primitives(
        vertices, faces, shell_samples=2, strand_samples=3
    )
    shell = primitives.xyz[: field.point_count * 2].reshape(
        field.point_count, 2, 3
    )
    strand = primitives.xyz[
        field.point_count * 2 : field.point_count * 5
    ].reshape(field.point_count, 3, 3)
    residual = primitives.xyz[-field.point_count :]
    torch.testing.assert_close(
        shell, residual[:, None, :].expand_as(shell)
    )
    torch.testing.assert_close(
        strand, residual[:, None, :].expand_as(strand)
    )


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


def test_mass_preserving_routes_normalize_small_row_mass_drift() -> None:
    probabilities = torch.tensor(
        [[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.2, 0.2, 0.6]],
        dtype=torch.float32,
    )
    route_ids = mass_preserving_route_ids(probabilities * 0.999)
    assert route_ids.shape == (3,)
    assert int(torch.bincount(route_ids, minlength=3).sum()) == 3


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
            "structured_delta",
            "structured_opacity",
                "residual_drift",
            "residual_trust",
            "expert_appearance",
            "carrier_entropy",
            "carrier_prior",
            "carrier_neighbor",
            "carrier_tip_neighbor",
            "carrier_attachment",
            "carrier_tip_prior",
            "carrier_confidence",
            "carrier_structure_mass",
            "carrier_family_alignment",
            "carrier_structure_floor",
        }
    assert all(torch.isfinite(value) for value in regularizers.values())
    assert float(regularizers["route_neighbor"]) > 0.0


def test_fin_gate_prefers_grazing_views_and_preserves_zero_delta_teacher() -> None:
    field, vertices, faces = _toy_field()
    front_camera = PinholeCamera(
        width=48,
        height=48,
        fx=45.0,
        fy=45.0,
        cx=24.0,
        cy=24.0,
        world_to_camera=np.eye(4, dtype=np.float32),
        image_y_down=True,
    )
    grazing_world_to_camera = np.eye(4, dtype=np.float32)
    grazing_world_to_camera[:3, 3] = [-5.0, 0.0, -2.0]
    grazing_camera = PinholeCamera(
        width=48,
        height=48,
        fx=45.0,
        fy=45.0,
        cx=24.0,
        cy=24.0,
        world_to_camera=grazing_world_to_camera,
        image_y_down=True,
    )
    point = torch.tensor([[0.0, 0.0, 2.0]])
    normal = torch.tensor([[0.0, 0.0, 1.0]])
    front_gate = fin_grazing_gate(
        point, normal, front_camera, threshold=0.25, softness=0.05
    )
    grazing_gate = fin_grazing_gate(
        point, normal, grazing_camera, threshold=0.25, softness=0.05
    )
    assert float(grazing_gate) > 0.99
    assert float(front_gate) < 1e-5

    # Enabling Fin cannot change the zero-initialized residual teacher.
    primitives = field.primitives(
        vertices, faces, shell_samples=2, strand_samples=3
    )
    gated = apply_fin_view_gate(
        primitives,
        front_camera,
        strength=1.0,
        threshold=0.25,
        softness=0.05,
    )
    torch.testing.assert_close(gated.opacity, primitives.opacity)


def test_fin_ribbon_is_anisotropic_and_only_gates_shell_opacity() -> None:
    field, vertices, faces = _toy_field()
    with torch.no_grad():
        field.structured_delta_raw[:, 0] = 1.0
    primitives = field.primitives(
        vertices,
        faces,
        shell_samples=2,
        strand_samples=3,
        geometry_blend=1.0,
        fin_aspect_ratio=9.0,
    )
    shell = primitives.route_id == 0
    residual_or_strand = ~shell
    torch.testing.assert_close(
        primitives.scaling[shell, 1] / primitives.scaling[shell, 2],
        torch.full_like(primitives.scaling[shell, 1], 9.0),
    )
    assert primitives.root_tip is not None
    assert primitives.surface_normal is not None
    assert primitives.structure_weight is not None

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
    gated = apply_fin_view_gate(
        primitives,
        camera,
        strength=1.0,
        threshold=0.25,
        softness=0.05,
    )
    assert torch.all(gated.opacity[shell] < primitives.opacity[shell])
    torch.testing.assert_close(
        gated.opacity[residual_or_strand], primitives.opacity[residual_or_strand]
    )


def test_additive_teacher_is_exact_at_zero_and_structured_gain_is_trainable() -> None:
    field, vertices, faces = _toy_field()
    residual = field.residual_primitives(vertices, faces)
    primitives = field.primitives(
        vertices,
        faces,
        shell_samples=2,
        strand_samples=3,
        additive_teacher=True,
    )
    shell_or_strand = primitives.route_id != 2
    residual_slice = primitives.route_id == 2
    torch.testing.assert_close(
        primitives.opacity[shell_or_strand],
        torch.zeros_like(primitives.opacity[shell_or_strand]),
    )
    torch.testing.assert_close(primitives.opacity[residual_slice], residual.opacity)

    # Although its primal opacity is zero, the straight-through gain supplies
    # a finite gradient that can grow a positively contributing Fin/strand.
    primitives.opacity[shell_or_strand].sum().backward()
    assert field.structured_opacity_raw.grad is not None
    assert float(field.structured_opacity_raw.grad.abs().sum()) > 0.0
    assert torch.isfinite(field.structured_opacity_raw.grad).all()


def test_downstream_length_and_wind_edits_preserve_residual_and_roots() -> None:
    field, vertices, faces = _toy_field()
    with torch.no_grad():
        field.structured_delta_raw.fill_(1.0)
        field.structured_opacity_raw.fill_(1.0)
    primitives = field.primitives(
        vertices, faces, shell_samples=2, strand_samples=3
    )
    edited = edit_structured_fibers(
        primitives,
        length_scale=1.25,
        wind_displacement=torch.tensor([0.05, 0.0, 0.0]),
        wind_power=2.0,
    )
    residual = primitives.route_id == 2
    structured = ~residual
    torch.testing.assert_close(edited.xyz[residual], primitives.xyz[residual])
    torch.testing.assert_close(
        edited.scaling[residual], primitives.scaling[residual]
    )
    assert float((edited.xyz[structured] - primitives.xyz[structured]).abs().sum()) > 0

    # Pure wind is exactly root-fixed and moves distal samples more.
    wind_only = edit_structured_fibers(
        primitives,
        wind_displacement=torch.tensor([0.05, 0.0, 0.0]),
        wind_power=2.0,
    )
    displacement = torch.linalg.vector_norm(
        wind_only.xyz - primitives.xyz, dim=-1
    )
    root_tip = primitives.root_tip
    assert root_tip is not None
    route_tip = structured & (root_tip > 0.7)
    route_root = structured & (root_tip < 0.3)
    assert float(displacement[route_tip].mean()) > float(displacement[route_root].mean())


def test_simulation_carrier_moves_residual_without_changing_render_route() -> None:
    field, vertices, faces = _toy_field()
    with torch.no_grad():
        field.carrier_logits.fill_(-10.0)
        field.carrier_logits[:, CARRIER_NAMES.index("strand")] = 10.0
        field.carrier_root_tip_raw.copy_(torch.tensor([0.0, 1.0]))
    primitives = field.primitives(vertices, faces, shell_samples=1, strand_samples=2)
    edited = deform_simulation_asset(
        primitives,
        wind_displacement=torch.tensor([0.1, 0.0, 0.0]),
        wind_power=2.0,
        hard_carriers=True,
    )
    residual = primitives.route_id == 2
    residual_displacement = edited.xyz[residual] - primitives.xyz[residual]
    torch.testing.assert_close(
        residual_displacement[0], torch.zeros(3), atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(
        residual_displacement[1], torch.tensor([0.1, 0.0, 0.0])
    )
    summary = simulation_asset_summary(primitives)
    assert summary["residual_fiber_bound"] > 0.999

    # Rendering ownership is unchanged: there is still exactly one residual
    # primitive per source even though its deformation owner is a strand.
    assert int(residual.sum()) == field.point_count


def test_surface_carrier_keeps_residual_fixed_under_fiber_edit() -> None:
    field, vertices, faces = _toy_field()
    with torch.no_grad():
        field.carrier_logits.fill_(-10.0)
        field.carrier_logits[:, CARRIER_NAMES.index("surface")] = 10.0
        field.carrier_root_tip_raw.fill_(1.0)
    primitives = field.primitives(vertices, faces, shell_samples=1, strand_samples=2)
    edited = deform_simulation_asset(
        primitives,
        length_scale=1.5,
        wind_displacement=torch.tensor([0.1, 0.0, 0.0]),
        hard_carriers=True,
    )
    residual = primitives.route_id == 2
    torch.testing.assert_close(edited.xyz[residual], primitives.xyz[residual])
    torch.testing.assert_close(
        edited.scaling[residual], primitives.scaling[residual]
    )


def test_carrier_regularizers_train_assignment_and_attachment() -> None:
    field, vertices, faces = _toy_field()
    regularizers = field.regularizers(vertices, faces, temperature=0.8)
    loss = (
        regularizers["carrier_entropy"]
        + regularizers["carrier_attachment"]
        + regularizers["carrier_tip_prior"]
    )
    loss.backward()
    assert field.carrier_logits.grad is not None
    assert torch.isfinite(field.carrier_logits.grad).all()
    assert field.carrier_root_tip_raw.grad is not None
    assert torch.isfinite(field.carrier_root_tip_raw.grad).all()
