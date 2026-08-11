from __future__ import annotations

import torch

from .fiber import FiberPrimitives, _quaternion_to_matrix_torch
from .render import PinholeCamera


def render_fiber_primitives_hairgs(
    primitives: FiberPrimitives,
    camera: PinholeCamera,
    *,
    background: torch.Tensor | None = None,
    scale_modifier: float = 1.0,
    znear: float = 0.01,
    zfar: float = 100.0,
    debug: bool = False,
    render_orientation: bool = False,
) -> dict[str, torch.Tensor]:
    """Render unified primitives with the CUDA rasterizer shipped by HairGS.

    The import is deliberately lazy: the main animal environment can keep
    using the Torch integration renderer, while the pinned ``hair-gs`` Conda
    environment supplies the compiled extension for full experiments.
    """

    try:
        from diff_gaussian_rasterization import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )
    except ImportError as error:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "HairGS CUDA rasterizer is unavailable. Activate the hair-gs "
            "environment or run scripts/setup_hairgs_baseline.sh."
        ) from error

    if not primitives.xyz.is_cuda:
        raise ValueError("HairGS rasterization requires CUDA tensors")
    device, dtype = primitives.xyz.device, primitives.xyz.dtype
    if background is None:
        background = torch.zeros(3, dtype=dtype, device=device)
    else:
        background = background.to(device=device, dtype=dtype)

    tan_fov_x = float(camera.width) / (2.0 * float(camera.fx))
    tan_fov_y = float(camera.height) / (2.0 * float(camera.fy))
    view = torch.as_tensor(camera.world_to_camera, dtype=dtype, device=device).T
    projection = _projection_matrix(
        camera,
        znear=znear,
        zfar=zfar,
        dtype=dtype,
        device=device,
    ).T
    full_projection = view.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0)
    camera_center = torch.linalg.inv(view)[3, :3]
    settings = GaussianRasterizationSettings(
        image_height=int(camera.height),
        image_width=int(camera.width),
        tanfovx=tan_fov_x,
        tanfovy=tan_fov_y,
        bg=background,
        scale_modifier=float(scale_modifier),
        viewmatrix=view,
        projmatrix=full_projection,
        sh_degree=0,
        campos=camera_center,
        prefiltered=False,
        debug=debug,
    )

    def rasterize(colors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        means_2d = torch.zeros_like(
            primitives.xyz, dtype=dtype, device=device, requires_grad=True
        )
        means_2d.retain_grad()
        rasterizer = GaussianRasterizer(raster_settings=settings)
        image, radii = rasterizer(
            means3D=primitives.xyz,
            means2D=means_2d,
            colors_precomp=colors,
            opacities=primitives.opacity[:, None],
            scales=primitives.scaling,
            rotations=primitives.rotation,
            shs=None,
            cov3D_precomp=None,
        )
        return image, radii, means_2d

    rendered_rgb, radii, viewspace_points = rasterize(primitives.color)
    rendered_alpha_rgb, _alpha_radii, _alpha_viewspace = rasterize(
        torch.ones_like(primitives.color)
    )
    output = {
        "rgb": rendered_rgb.permute(1, 2, 0).contiguous().clamp(0.0, 1.0),
        "mask": rendered_alpha_rgb.mean(dim=0).clamp(0.0, 1.0),
        "radii": radii,
        "visibility_filter": radii > 0,
        "viewspace_points": viewspace_points,
    }
    if render_orientation:
        # Match HairGS: use the projected, sign-invariant tangent direction.
        # A double-angle encoding lets Gaussian alpha compositing average
        # directions without the pi-periodic sign ambiguity of hair.
        tangent_world = _quaternion_to_matrix_torch(primitives.rotation)[..., :, 0]
        world_to_camera = torch.as_tensor(
            camera.world_to_camera[:3, :3], dtype=dtype, device=device
        )
        tangent_camera = tangent_world @ world_to_camera.T
        x, y = tangent_camera[:, 0], tangent_camera[:, 1]
        denominator = (x.square() + y.square()).clamp_min(1e-8)
        orientation_color = torch.stack(
            [(y.square() - x.square()) / denominator, 2.0 * x * y / denominator,
             torch.zeros_like(x)],
            dim=-1,
        )
        rendered_orientation, _orientation_radii, _orientation_viewspace = rasterize(
            orientation_color
        )
        output["orientation"] = rendered_orientation.permute(1, 2, 0).contiguous()
    return output


def _projection_matrix(
    camera: PinholeCamera,
    *,
    znear: float,
    zfar: float,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if znear <= 0.0 or zfar <= znear:
        raise ValueError("Expected 0 < znear < zfar")
    matrix = torch.zeros((4, 4), dtype=dtype, device=device)
    matrix[0, 0] = 2.0 * float(camera.fx) / float(camera.width)
    matrix[1, 1] = 2.0 * float(camera.fy) / float(camera.height)
    matrix[0, 2] = 2.0 * float(camera.cx) / float(camera.width) - 1.0
    y_offset = 2.0 * float(camera.cy) / float(camera.height) - 1.0
    matrix[1, 2] = y_offset if camera.image_y_down else -y_offset
    matrix[3, 2] = 1.0
    matrix[2, 2] = float(zfar) / float(zfar - znear)
    matrix[2, 3] = -(float(zfar) * float(znear)) / float(zfar - znear)
    return matrix
