import copy

import numpy as np
import torch

from dpd3dgs_animal.fiber_optimize import (
    _area_stratified_surface_samples,
    _apply_adaptive_topology_scores,
    _apply_route_mass_floor,
    _bidirectional_mask_losses,
    _front_surface_visibility,
    _front_visible_sample_gate,
    _initialize_multiview_coverage_seeds,
    _initialize_render_preserving_adaptive_migration,
    _initialize_render_preserving_semantic_migration,
    _load_residual_bootstrap_checkpoint,
    _masked_rgb_gradient_loss,
    _parallel_transport_surface_directions,
    _residual_footprint_probe_points,
    _route_visual_hull_soft_loss,
    _resolve_fiber_point_budget,
    _freeze_residual_teacher_scaffold,
    _structured_spill_loss,
    _synchronize_outward_direction_signs,
    _topology_event_is_accepted,
)
from dpd3dgs_animal.config import PipelineConfig, load_config

from dpd3dgs_animal.fiber import (
    CARRIER_NAMES,
    FixedGaussianBase,
    UnifiedFiberField,
    _bind_exact_surface_vertices,
    _bind_nearest_surface_vertices,
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


def test_semantic_migration_config_fields_are_loaded(tmp_path) -> None:
    path = tmp_path / "semantic.yaml"
    path.write_text(
        "fiber_teacher_semantic_migration_mass: [0.25, 0.40, 0.35]\n"
        "fiber_teacher_adaptive_migration_domain: hair\n"
        "fiber_teacher_adaptive_migration_bias: 0.7\n"
        "fiber_adaptive_migration_hard_router: false\n"
        "fiber_teacher_semantic_migration_tolerance: 0.005\n"
        "fiber_optimize_structured_base_appearance: true\n"
        "fiber_teacher_nonregression_views_per_step: 2\n"
        "fiber_teacher_nonregression_reduction: max\n"
        "fiber_structure_deployment_weight: 0.12\n"
        "fiber_structure_min_deployment_gain: 0.25\n",
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.fiber_teacher_semantic_migration_mass == [0.25, 0.40, 0.35]
    assert cfg.fiber_teacher_adaptive_migration_domain == "hair"
    assert cfg.fiber_teacher_adaptive_migration_bias == 0.7
    assert cfg.fiber_adaptive_migration_hard_router is False
    assert cfg.fiber_teacher_semantic_migration_tolerance == 0.005
    assert cfg.fiber_optimize_structured_base_appearance is True
    assert cfg.fiber_teacher_nonregression_views_per_step == 2
    assert cfg.fiber_teacher_nonregression_reduction == "max"
    assert cfg.fiber_structure_deployment_weight == 0.12
    assert cfg.fiber_structure_min_deployment_gain == 0.25


def test_nearest_vertex_binding_preserves_a_valid_carrier_face() -> None:
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    points = np.asarray([[0.02, 0.01, 0.4], [0.95, 0.02, -0.2]], dtype=np.float32)
    binding = _bind_nearest_surface_vertices(points, vertices, faces)
    assert binding.face_index.tolist() == [0, 0]
    np.testing.assert_allclose(binding.barycentric.sum(axis=1), 1.0)


def test_topology_event_rejects_single_view_regression_despite_better_mean() -> None:
    accepted, deltas = _topology_event_is_accepted(
        [1.0, 1.0, 1.0],
        [0.8, 0.8, 1.05],
        margin=0.01,
    )
    assert not accepted
    assert deltas[-1] > 0.01
    accepted, _ = _topology_event_is_accepted(
        [1.0, 1.0, 1.0],
        [0.99, 1.0, 0.98],
        margin=0.01,
    )
    assert accepted


def test_area_stratified_scalp_atlas_covers_faces_with_interior_roots() -> None:
    vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
         [1.0, 1.0, 0.0], [2.0, 0.0, 0.0], [2.0, 1.0, 0.0]]
    )
    faces = torch.tensor([[0, 1, 2], [1, 3, 2], [1, 4, 3], [4, 5, 3]])
    selected, barycentric, report = _area_stratified_surface_samples(
        vertices, faces, torch.arange(4), 12, min_roots_per_face=1
    )
    assert torch.unique(selected).numel() == 4
    assert report["atlas_covered_face_count"] == 4
    assert report["atlas_max_roots_per_face"] <= 4
    torch.testing.assert_close(barycentric.sum(dim=-1), torch.ones(12))
    assert torch.all(barycentric > 0.0)


def test_signed_direction_field_flips_inward_and_reduces_axis_disagreement() -> None:
    direction = torch.tensor(
        [[1.0, 0.0, 0.01], [-1.0, 0.0, -0.01],
         [0.9, 0.1, 0.02], [-0.9, -0.1, -0.02]]
    )
    frame = torch.eye(3).repeat(4, 1, 1)
    neighbors = torch.tensor([[1, 2], [0, 3], [0, 3], [1, 2]])
    signed, report = _synchronize_outward_direction_signs(
        direction, frame, neighbors, anchor_threshold=0.05, steps=4
    )
    assert report["inward_fraction_before"] == 0.5
    assert report["inward_fraction_after"] == 0.0
    assert report["neighbor_sign_disagreement_after"] == 0.0
    assert torch.all(signed[:, 2] >= 0.0)


def test_rgb_gradient_loss_penalizes_blur_and_matches_exact_image() -> None:
    target = torch.zeros(8, 8, 3)
    target[:, 4:] = 1.0
    mask = torch.ones(8, 8)
    exact = _masked_rgb_gradient_loss(target, target, mask)
    blurred = torch.full_like(target, 0.5)
    blur_loss = _masked_rgb_gradient_loss(blurred, target, mask)
    torch.testing.assert_close(exact, torch.zeros_like(exact))
    assert float(blur_loss) > 0.0


def test_structured_spill_only_penalizes_excess_background_opacity() -> None:
    target = torch.zeros(4, 4)
    target[1:3, 1:3] = 1.0
    teacher = torch.full((4, 4), 0.1)
    no_excess = _structured_spill_loss(teacher, teacher, target)
    prediction = teacher.clone()
    prediction[0, 0] = 0.5
    excess = _structured_spill_loss(prediction, teacher, target)
    torch.testing.assert_close(no_excess, torch.zeros_like(no_excess))
    assert float(excess) > 0.0


def test_bidirectional_mask_losses_separate_holes_from_spill() -> None:
    target = torch.zeros(6, 6)
    target[2:4, 2:4] = 1.0
    exact = target.clone()
    inside, outside = _bidirectional_mask_losses(
        exact, target, inside_alpha_target=1.0
    )
    torch.testing.assert_close(inside, torch.zeros_like(inside))
    torch.testing.assert_close(outside, torch.zeros_like(outside))

    missing = exact.clone()
    missing[2, 2] = 0.0
    inside, outside = _bidirectional_mask_losses(
        missing, target, inside_alpha_target=1.0
    )
    assert float(inside) > 0.0
    torch.testing.assert_close(outside, torch.zeros_like(outside))

    spilling = exact.clone()
    spilling[0, 0] = 0.5
    inside, outside = _bidirectional_mask_losses(
        spilling, target, inside_alpha_target=1.0
    )
    torch.testing.assert_close(inside, torch.zeros_like(inside))
    assert float(outside) > 0.0

    reference = spilling.clone()
    inside, outside = _bidirectional_mask_losses(
        spilling,
        target,
        inside_alpha_target=1.0,
        outside_reference_mask=reference,
    )
    torch.testing.assert_close(inside, torch.zeros_like(inside))
    torch.testing.assert_close(outside, torch.zeros_like(outside))


def test_multiview_coverage_seed_activates_deficit_supported_root(tmp_path) -> None:
    field, vertices, faces = _toy_field()
    teacher = copy.deepcopy(field)
    with torch.no_grad():
        teacher.opacity_logits.fill_(-20.0)
    camera = PinholeCamera(
        width=32,
        height=32,
        fx=24.0,
        fy=24.0,
        cx=16.0,
        cy=16.0,
        world_to_camera=np.eye(4, dtype=np.float32),
        image_y_down=True,
    )
    cfg = PipelineConfig(
        fiber_renderer="torch",
        fiber_coverage_seed_count=1,
        fiber_coverage_seed_samples=3,
        fiber_coverage_seed_min_views=1,
        fiber_coverage_seed_min_fraction=0.5,
        fiber_coverage_seed_min_deficit=0.01,
        fiber_coverage_seed_voxel_scale=0.01,
        fiber_coverage_seed_structured_opacity=0.4,
        fiber_coverage_seed_geometry_gain=1.0,
        fiber_coverage_seed_shell_length_scale=0.02,
        fiber_coverage_seed_strand_length_scale=0.2,
        fiber_coverage_seed_route_mass=[0.1, 0.8, 0.1],
        fiber_visual_hull_margin_px=0,
    )
    report, teacher_masks = _initialize_multiview_coverage_seeds(
        field,
        teacher,
        faces,
        [vertices],
        [camera],
        [0],
        [{"mask": torch.ones(32, 32)}],
        None,
        cfg,
        "torch",
        tmp_path,
    )
    assert report is not None
    assert report["selected_count"] == 1
    assert set(teacher_masks) == {0}
    assert (tmp_path / "coverage_seed_roots.ply").is_file()
    activated = field.structured_opacity_gain[:, 1] > 0.0
    assert int(activated.sum()) == 1
    selected = torch.nonzero(activated, as_tuple=False).reshape(-1)
    assert torch.all(field.structured_delta_gain[selected] == 1.0)
    assert torch.all(field.route_active_gate[selected, :2] == 1.0)
    probabilities = field.route_probabilities(
        temperature=cfg.fiber_final_temperature
    )
    torch.testing.assert_close(
        probabilities[selected][0], torch.tensor([0.1, 0.8, 0.1])
    )


def test_front_surface_visibility_rejects_far_same_pixel_point() -> None:
    camera = PinholeCamera(
        width=32,
        height=32,
        fx=24.0,
        fy=24.0,
        cx=16.0,
        cy=16.0,
        world_to_camera=np.eye(4, dtype=np.float32),
        image_y_down=True,
    )
    points = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.5, 0.0, 2.0]]
    )
    visible = _front_surface_visibility(
        points, camera, bin_px=1, depth_tolerance=0.01
    )
    assert visible.tolist() == [True, False, True]


def test_front_visible_sample_gate_uses_shared_occluder_geometry() -> None:
    camera = PinholeCamera(
        width=32,
        height=32,
        fx=24.0,
        fy=24.0,
        cx=16.0,
        cy=16.0,
        world_to_camera=np.eye(4, dtype=np.float32),
        image_y_down=True,
    )
    samples = torch.tensor([[[0.0, 0.0, 2.0], [0.5, 0.0, 2.0]]])
    occluders = torch.tensor([[0.0, 0.0, 1.0]])
    visible = _front_visible_sample_gate(
        samples,
        occluders,
        camera,
        bin_px=1,
        depth_tolerance=0.01,
    )
    assert visible.tolist() == [[False, True]]

def test_soft_hull_does_not_repenalize_multiview_accepted_samples() -> None:
    camera = PinholeCamera(
        width=16,
        height=16,
        fx=12.0,
        fy=12.0,
        cx=8.0,
        cy=8.0,
        world_to_camera=np.eye(4, dtype=np.float32),
        image_y_down=True,
    )
    points = torch.tensor([[[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]]])
    probabilities = torch.tensor([[0.0, 1.0, 0.0]])
    mask = torch.zeros(16, 16)
    rejected = _route_visual_hull_soft_loss(
        points,
        probabilities,
        1,
        camera,
        mask,
        0,
        sample_gate=torch.zeros(1, 2),
    )
    accepted = _route_visual_hull_soft_loss(
        points,
        probabilities,
        1,
        camera,
        mask,
        0,
        sample_gate=torch.ones(1, 2),
    )
    assert float(rejected) > 0.0
    torch.testing.assert_close(accepted, torch.zeros_like(accepted))


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


def test_surface_direction_propagation_fills_an_unobserved_root() -> None:
    local = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
    )
    frames = torch.eye(3).repeat(3, 1, 1)
    neighbors = torch.tensor([[1], [0], [1]], dtype=torch.long)
    confidence = torch.tensor([1.0, 0.0, 1.0])
    propagated = _parallel_transport_surface_directions(
        local,
        frames,
        neighbors,
        confidence,
        steps=4,
        observation_weight=1.0,
    )
    assert float(propagated[1, 0]) > 0.99
    torch.testing.assert_close(propagated[[0, 2]], local[[0, 2]])


def test_shell_direction_can_be_decoupled_from_propagated_strand_flow() -> None:
    field, vertices, faces = _toy_field()
    field.shell_propagated_direction_weight = 0.0
    roots, _tangent, _bitangent, normal = field.surface_frame(vertices, faces)
    shell = field.shell_target_geometry(vertices, faces, shell_samples=1)[:, 0]
    displacement = torch.nn.functional.normalize(
        shell - roots - field.height[:, None] * normal, dim=-1
    )
    torch.testing.assert_close(displacement, normal, atol=1e-5, rtol=1e-5)


def test_fixed_base_is_not_learnable_and_does_not_expand_hair_routes() -> None:
    field, vertices, faces = _toy_field()
    base_source = copy.deepcopy(field)
    base = FixedGaussianBase(base_source)
    assert list(base.parameters()) == []
    field.semantic_mask_from_source = True
    field.fixed_base = base
    primitives = field.primitives(
        vertices, faces, shell_samples=2, strand_samples=3, temperature=0.8
    )
    hair_primitive_count = field.point_count * (2 + 3 + 1)
    assert primitives.xyz.shape[0] == hair_primitive_count + base.point_count
    assert primitives.route_probabilities.shape == (field.point_count, 3)
    torch.testing.assert_close(
        primitives.source_id[-base.point_count :],
        torch.arange(base.point_count) + field.point_count,
    )
    torch.testing.assert_close(
        primitives.semantic_foreground[-base.point_count :],
        torch.zeros(base.point_count),
    )
    assert not any(key.startswith("fixed_base.") for key in field.state_dict())


def test_residual_scale_safety_cap_is_scene_relative_and_optional() -> None:
    field, _, _ = _toy_field()
    original = field.residual_scaling.detach().clone()
    torch.testing.assert_close(original, torch.full_like(original, 0.015))
    field.residual_max_scale_fraction = 0.01
    torch.testing.assert_close(
        field.residual_scaling,
        torch.full_like(field.residual_scaling, 0.01),
    )


def test_fixed_base_uses_the_same_scene_relative_scale_cap() -> None:
    field, vertices, faces = _toy_field()
    field.residual_max_scale_fraction = 0.01
    base = FixedGaussianBase(field)
    primitives = base.primitives(vertices, faces)
    torch.testing.assert_close(
        primitives.scaling,
        torch.full_like(primitives.scaling, 0.01),
    )


def test_route_topology_gate_removes_and_activates_renderer_primitives() -> None:
    field, vertices, faces = _toy_field()
    with torch.no_grad():
        field.route_active_gate[:, :2] = 0.0
        field.route_active_gate[:, 2] = 1.0
    primitives = field.primitives(
        vertices, faces, shell_samples=2, strand_samples=3, temperature=0.8
    )
    shell = primitives.opacity[: field.point_count * 2]
    strand = primitives.opacity[field.point_count * 2 : field.point_count * 5]
    residual = primitives.opacity[-field.point_count :]
    torch.testing.assert_close(shell, torch.zeros_like(shell))
    torch.testing.assert_close(strand, torch.zeros_like(strand))
    assert torch.all(residual > 0.0)
    torch.testing.assert_close(
        primitives.route_probabilities[:, 2], torch.ones(field.point_count)
    )

    with torch.no_grad():
        field.route_active_gate[0, 2] = 0.0
    residual_only = field.residual_primitives(vertices, faces)
    assert float(residual_only.opacity[0]) == 0.0
    assert float(residual_only.opacity[1]) > 0.0


def test_adaptive_topology_scores_prune_outlier_and_grow_hole() -> None:
    field, _vertices, _faces = _toy_field()
    with torch.no_grad():
        field.route_active_gate[:, :2] = 0.0
        field.route_active_gate[:, 2] = 1.0
    optimizer = torch.optim.Adam(field.parameters(), lr=1e-3)
    cfg = PipelineConfig(
        fiber_shell_samples=2,
        fiber_strand_samples=3,
        fiber_topology_prune_count=1,
        fiber_topology_grow_count=1,
        fiber_topology_densify_count=0,
        fiber_topology_min_views=1,
        fiber_topology_prune_max_support=0.1,
        fiber_topology_grow_min_support=0.5,
        fiber_topology_grow_min_deficit=0.1,
        fiber_topology_max_residual_prune_fraction=0.5,
        fiber_coverage_seed_route_mass=[0.15, 0.75, 0.10],
        fiber_topology_deficit_min_views=2,
    )
    event = _apply_adaptive_topology_scores(
        field,
        optimizer,
        cfg,
        visible_views=torch.tensor([3.0, 3.0]),
        residual_support=torch.tensor([0.0, 1.0]),
        structured_support=torch.tensor([0.0, 1.0]),
        deficit_score=torch.tensor([0.0, 0.5]),
        boundary_score=torch.zeros(2),
        step=100,
        deficit_views=torch.tensor([0.0, 3.0]),
    )
    assert event["pruned_count"] == 1
    assert event["grown_count"] == 1
    assert float(field.route_active_gate[0, 2]) == 0.0
    assert torch.all(field.route_active_gate[1, :2] == 1.0)
    assert event["topology"]["active_gaussians"] == 6


def test_incremental_3d_deficit_birth_keeps_shell_and_starts_nearly_zero() -> None:
    field, _vertices, _faces = _toy_field()
    with torch.no_grad():
        field.route_active_gate.zero_()
        field.route_active_gate[:, 0] = 1.0
    optimizer = torch.optim.Adam(field.parameters(), lr=1e-3)
    cfg = PipelineConfig(
        fiber_topology_incremental_birth=True,
        fiber_topology_birth_strand_mass=0.01,
        fiber_topology_birth_initial_delta=0.0,
        fiber_topology_deficit_min_views=2,
        fiber_topology_grow_count=1,
        fiber_topology_densify_count=0,
        fiber_topology_prune_count=0,
        fiber_topology_min_views=2,
        fiber_topology_grow_min_support=0.5,
        fiber_topology_grow_min_deficit=0.1,
    )
    event = _apply_adaptive_topology_scores(
        field,
        optimizer,
        cfg,
        visible_views=torch.tensor([3.0, 3.0]),
        residual_support=torch.zeros(2),
        structured_support=torch.ones(2),
        deficit_score=torch.tensor([0.8, 0.1]),
        boundary_score=torch.zeros(2),
        deficit_views=torch.tensor([3.0, 1.0]),
        step=100,
    )
    assert event["grown_count"] == 1
    assert torch.all(field.route_active_gate[0, :2] == 1.0)
    assert float(field.structured_delta_raw[0, 1]) == 0.0
    probabilities = field.route_probabilities(cfg.fiber_final_temperature)
    assert 0.005 < float(probabilities[0, 1]) < 0.02


def test_residual_footprint_probes_cover_center_and_local_axes() -> None:
    center = torch.tensor([[1.0, 2.0, 3.0]])
    tangent = torch.tensor([[1.0, 0.0, 0.0]])
    bitangent = torch.tensor([[0.0, 1.0, 0.0]])
    normal = torch.tensor([[0.0, 0.0, 1.0]])
    probes = _residual_footprint_probe_points(
        center,
        tangent,
        bitangent,
        normal,
        torch.tensor([[0.20, 0.10, 0.05]]),
        sigma=2.0,
    )
    assert probes.shape == (1, 7, 3)
    torch.testing.assert_close(probes[0, 0], center[0])
    torch.testing.assert_close(probes[0, 1], torch.tensor([1.4, 2.0, 3.0]))
    torch.testing.assert_close(probes[0, 2], torch.tensor([0.6, 2.0, 3.0]))
    torch.testing.assert_close(probes[0, 5], torch.tensor([1.0, 2.0, 3.4]))
    torch.testing.assert_close(probes[0, 6], torch.tensor([1.0, 2.0, 2.6]))


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


def test_target_geometry_remains_visible_to_geometry_losses_at_zero_gain() -> None:
    field, vertices, faces = _toy_field()
    primitives = field.primitives(
        vertices, faces, shell_samples=2, strand_samples=3
    )
    deployed = primitives.xyz[
        field.point_count * 2 : field.point_count * 5
    ].reshape(field.point_count, 3, 3)
    target, target_direction = field.strand_target_geometry(
        vertices, faces, strand_samples=3
    )
    assert not torch.allclose(target, deployed)
    assert target.shape == deployed.shape
    torch.testing.assert_close(
        torch.linalg.vector_norm(target_direction, dim=-1),
        torch.ones(field.point_count, 3),
    )


def test_shell_visibility_gate_removes_unsupported_fin_samples() -> None:
    field, vertices, faces = _toy_field()
    shell_gate = torch.ones(field.point_count, 2)
    shell_gate[:, 1] = 0.0
    with torch.no_grad():
        # Visual-hull culling is exact only for a deployed Fin.  At zero
        # deployment the samples remain the render-preserving teacher copy.
        field.structured_delta_raw[:, 0] = 1.0
    primitives = field.primitives(
        vertices,
        faces,
        shell_samples=2,
        strand_samples=3,
        shell_visibility=shell_gate,
    )
    shell_opacity = primitives.opacity[: field.point_count * 2].reshape(
        field.point_count, 2
    )
    assert torch.all(shell_opacity[:, 0] > 0.0)
    torch.testing.assert_close(shell_opacity[:, 1], torch.zeros_like(shell_opacity[:, 1]))


def test_shell_target_geometry_is_fully_deployed_at_zero_gain() -> None:
    field, vertices, faces = _toy_field()
    primitives = field.primitives(
        vertices, faces, shell_samples=2, strand_samples=3
    )
    rendered = primitives.xyz[: field.point_count * 2].reshape(
        field.point_count, 2, 3
    )
    target = field.shell_target_geometry(vertices, faces, shell_samples=2)
    assert target.shape == rendered.shape
    assert not torch.allclose(target, rendered)


def test_strand_deployment_and_shared_field_regularizers_have_gradients() -> None:
    field, vertices, faces = _toy_field()
    field.strand_visibility_gate = torch.ones(field.point_count, 3)
    regularizers = field.regularizers(
        vertices,
        faces,
        temperature=0.8,
        strand_min_deployment_gain=0.4,
        strand_min_deployed_length_scale=0.05,
        strand_coverage_target=0.01,
    )
    objective = (
        regularizers["strand_field"]
        + regularizers["strand_deployability"]
        + regularizers["strand_coverage_deficit"]
    )
    objective.backward()
    assert float(regularizers["strand_deployability"]) > 0.0
    assert float(regularizers["strand_coverage_deficit"]) > 0.0
    assert field.structured_delta_raw.grad is not None
    assert float(field.structured_delta_raw.grad[:, 1].abs().sum()) > 0.0
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


def test_semantic_migration_is_a_hard_capacity_and_opacity_reparameterization() -> None:
    field, vertices, faces = _toy_field()
    report = _initialize_render_preserving_semantic_migration(
        field, [0.5, 0.5, 0.0], temperature=0.35
    )
    assert report["capacity_counts"] == {"shell": 1, "strand": 1, "residual": 0}
    torch.testing.assert_close(
        field.route_active_gate.sum(dim=-1), torch.ones(field.point_count)
    )
    probabilities = field.route_probabilities(temperature=0.35)
    torch.testing.assert_close(
        probabilities, field.route_active_gate, rtol=0.0, atol=0.0
    )

    shell_samples, strand_samples = 2, 3
    primitives = field.primitives(
        vertices,
        faces,
        shell_samples=shell_samples,
        strand_samples=strand_samples,
        temperature=0.35,
        geometry_blend=1.0,
    )
    n = field.point_count
    shell_opacity = primitives.opacity[: n * shell_samples].reshape(
        n, shell_samples
    )
    strand_opacity = primitives.opacity[
        n * shell_samples : n * (shell_samples + strand_samples)
    ].reshape(n, strand_samples)
    residual_opacity = primitives.opacity[-n:]
    combined_alpha = 1.0 - (
        torch.prod(1.0 - shell_opacity, dim=-1)
        * torch.prod(1.0 - strand_opacity, dim=-1)
        * (1.0 - residual_opacity)
    )
    torch.testing.assert_close(combined_alpha, field.opacity)
    assert int((primitives.opacity > 1e-10).sum()) == field.point_count
    torch.testing.assert_close(
        field.structured_delta_gain, torch.zeros_like(field.structured_delta_gain)
    )


def test_adaptive_migration_removes_residual_and_preserves_soft_transmittance() -> None:
    field, vertices, faces = _toy_field()
    report = _initialize_render_preserving_adaptive_migration(
        field, "hair", domain_bias=0.65, temperature=0.35
    )
    assert report["fixed_global_quota"] is False
    assert report["residual_source_capacity_ceiling"] == 0.0
    assert report["initial_soft_mass"]["strand"] > report["initial_soft_mass"]["shell"]
    torch.testing.assert_close(
        field.route_active_gate[:, :2], torch.ones(field.point_count, 2)
    )
    torch.testing.assert_close(
        field.route_active_gate[:, 2], torch.zeros(field.point_count)
    )

    shell_samples, strand_samples = 2, 3
    primitives = field.primitives(
        vertices,
        faces,
        shell_samples=shell_samples,
        strand_samples=strand_samples,
        temperature=0.35,
        geometry_blend=1.0,
    )
    n = field.point_count
    shell_opacity = primitives.opacity[: n * shell_samples].reshape(
        n, shell_samples
    )
    strand_opacity = primitives.opacity[
        n * shell_samples : n * (shell_samples + strand_samples)
    ].reshape(n, strand_samples)
    residual_opacity = primitives.opacity[-n:]
    combined_alpha = 1.0 - (
        torch.prod(1.0 - shell_opacity, dim=-1)
        * torch.prod(1.0 - strand_opacity, dim=-1)
        * (1.0 - residual_opacity)
    )
    torch.testing.assert_close(combined_alpha, field.opacity)
    torch.testing.assert_close(residual_opacity, torch.zeros_like(residual_opacity))


def test_structured_student_can_refit_appearance_without_moving_teacher_geometry() -> None:
    field, _vertices, _faces = _toy_field()
    _freeze_residual_teacher_scaffold(
        field, optimize_structured_base_appearance=True
    )
    assert field.color_logits.requires_grad
    assert field.opacity_logits.requires_grad
    assert not field.residual_offset_local.requires_grad
    assert not field.residual_log_scale_delta.requires_grad
    assert not field.residual_rotation_raw.requires_grad


def test_visual_hull_culling_fades_in_with_structured_deployment() -> None:
    field, vertices, faces = _toy_field()
    _initialize_render_preserving_semantic_migration(
        field, [0.0, 1.0, 0.0], temperature=0.35
    )
    visibility = torch.zeros(field.point_count, 3)
    collapsed = field.primitives(
        vertices,
        faces,
        shell_samples=2,
        strand_samples=3,
        strand_visibility=visibility,
        geometry_blend=1.0,
    )
    n = field.point_count
    collapsed_strand = collapsed.opacity[n * 2 : n * 5].reshape(n, 3)
    collapsed_alpha = 1.0 - torch.prod(1.0 - collapsed_strand, dim=-1)
    torch.testing.assert_close(collapsed_alpha, field.opacity)

    with torch.no_grad():
        field.structured_delta_raw[:, 1] = 1.0
    deployed = field.primitives(
        vertices,
        faces,
        shell_samples=2,
        strand_samples=3,
        strand_visibility=visibility,
        geometry_blend=1.0,
    )
    deployed_strand = deployed.opacity[n * 2 : n * 5]
    torch.testing.assert_close(deployed_strand, torch.zeros_like(deployed_strand))


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
        "structure_deployment",
        "strand_field",
        "strand_deployability",
        "strand_effective_coverage",
        "strand_coverage_deficit",
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
            "expert_sh",
            "root_barycentric",
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


def test_teacher_opacity_transfer_replaces_residual_without_adding_structure() -> None:
    field, vertices, faces = _toy_field()
    with torch.no_grad():
        field.structured_opacity_raw.fill_(1.0)
    additive = field.primitives(
        vertices,
        faces,
        shell_samples=2,
        strand_samples=3,
        additive_teacher=True,
        teacher_opacity_transfer=0.0,
    )
    transferred = field.primitives(
        vertices,
        faces,
        shell_samples=2,
        strand_samples=3,
        additive_teacher=True,
        teacher_opacity_transfer=0.5,
    )
    structured = additive.route_id != 2
    residual = additive.route_id == 2
    torch.testing.assert_close(
        transferred.opacity[structured], additive.opacity[structured]
    )
    assert torch.all(transferred.opacity[residual] < additive.opacity[residual])
    assert torch.all(transferred.opacity[residual] > 0.0)


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
