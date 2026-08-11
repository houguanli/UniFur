from pathlib import Path

import numpy as np
import torch

from dpd3dgs_animal.config import PipelineConfig
from dpd3dgs_animal.fem_optimize import DifferentiableConstrainedFEMModel


def test_constrained_fem_solve_backpropagates_to_skeleton(tmp_path: Path) -> None:
    stage1 = tmp_path / "stage1_toy.npz"
    rest_nodes = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    np.savez_compressed(
        stage1,
        rest_tet_nodes=rest_nodes,
        tets=np.asarray([[0, 1, 2, 3]], dtype=np.int64),
        rest_surface_vertices=rest_nodes,
        surface_faces=np.asarray([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int64),
        surface_node_indices=np.arange(4, dtype=np.int64),
        skeleton_joints=np.asarray(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.0, 0.25, 0.0]],
            ],
            dtype=np.float32,
        ),
        parents=np.asarray([-1, 0], dtype=np.int64),
    )
    cfg = PipelineConfig(device="cpu")
    cfg.elastic_fem_radius_scale = 1.0
    cfg.elastic_fem_min_radius_scale = 0.01
    cfg.elastic_fem_max_radius_scale = 2.0
    cfg.elastic_fem_min_bone_length_scale = 0.0
    cfg.fem_cg_iters = 12
    cfg.fem_handle_stiffness = 20.0
    model = DifferentiableConstrainedFEMModel(stage1, cfg, device="cpu")

    tet_nodes, surface_vertices, _joints = model.solve_frame(1)
    assert model.handle_count == 1
    assert torch.isfinite(tet_nodes).all()
    assert torch.isfinite(surface_vertices).all()

    loss = surface_vertices[:, 1].mean()
    loss.backward()
    assert model.joints.grad is not None
    assert torch.isfinite(model.joints.grad).all()
    assert float(model.joints.grad[1].abs().sum()) > 0.0
