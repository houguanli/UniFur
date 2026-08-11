import numpy as np
import pytest
import torch

from dpd3dgs_animal.fiber import UnifiedFiberField
from dpd3dgs_animal.hairgs_renderer import render_fiber_primitives_hairgs
from dpd3dgs_animal.render import PinholeCamera


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="HairGS integration requires CUDA"
)


def test_hairgs_cuda_rasterizer_accepts_unified_primitives_and_backpropagates() -> None:
    pytest.importorskip("diff_gaussian_rasterization")
    device = "cuda"
    vertices = torch.tensor(
        [[-0.4, -0.4, 2.0], [0.4, -0.4, 2.0], [-0.4, 0.4, 2.0]],
        dtype=torch.float32,
        device=device,
    )
    faces = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)
    field = UnifiedFiberField(
        face_index=torch.tensor([0, 0, 0], device=device),
        barycentric=torch.tensor(
            [[0.34, 0.33, 0.33], [0.2, 0.6, 0.2], [0.6, 0.2, 0.2]],
            device=device,
        ),
        color=torch.tensor(
            [[0.9, 0.2, 0.1], [0.1, 0.9, 0.2], [0.2, 0.1, 0.9]],
            device=device,
        ),
        opacity=torch.full((3,), 0.8, device=device),
        original_scaling=torch.full((3, 3), 0.03, device=device),
        original_rotation=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]], device=device
        ).repeat(3, 1),
        rest_surface_frame=torch.eye(3, device=device).repeat(3, 1, 1),
        residual_offset_local=torch.tensor(
            [[0.0, 0.0, 0.02], [0.01, 0.0, 0.02], [-0.01, 0.0, 0.02]],
            device=device,
        ),
        direction_local=torch.tensor(
            [[0.0, 0.0, 1.0], [0.3, 0.0, 1.0], [-0.3, 0.0, 1.0]],
            device=device,
        ),
        height=torch.full((3,), 0.01, device=device),
        shell_length=torch.full((3,), 0.04, device=device),
        strand_length=torch.full((3,), 0.12, device=device),
        radius=torch.full((3,), 0.01, device=device),
        route_logits=torch.tensor(
            [[1.0, 0.0, -1.0], [0.0, 1.0, -1.0], [-1.0, 0.0, 1.0]],
            device=device,
        ),
        scene_scale=1.0,
    )
    camera = PinholeCamera(
        width=64,
        height=64,
        fx=60.0,
        fy=60.0,
        cx=32.0,
        cy=32.0,
        world_to_camera=np.eye(4, dtype=np.float32),
        image_y_down=True,
    )
    primitives = field.primitives(vertices, faces, shell_samples=2, strand_samples=3)
    rendered = render_fiber_primitives_hairgs(primitives, camera)
    assert rendered["rgb"].shape == (64, 64, 3)
    assert rendered["mask"].shape == (64, 64)
    assert bool(torch.isfinite(rendered["rgb"]).all())
    assert int(rendered["visibility_filter"].sum()) > 0

    loss = rendered["rgb"].mean() + rendered["mask"].mean()
    loss.backward()
    assert field.route_logits.grad is not None
    assert field.direction_local_raw.grad is not None
    assert torch.isfinite(field.route_logits.grad).all()
    assert torch.isfinite(field.direction_local_raw.grad).all()
