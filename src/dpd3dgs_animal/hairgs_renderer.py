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

    # Adaptive topology preallocates route capacity with exactly zero opacity.
    # Sending those dormant primitives to the CUDA rasterizer wastes most of
    # the memory for a 124k Stage-I scaffold (and previously made the clean
    # base configuration look infeasible). The detached predicate is safe:
    # topology events explicitly switch a primitive above zero before it is
    # expected to receive differentiable rendering gradients.
    active_indices = torch.nonzero(
        primitives.opacity.detach() > 1e-10, as_tuple=False
    ).reshape(-1)
    xyz = primitives.xyz[active_indices]
    color = primitives.color[active_indices]
    opacity = primitives.opacity[active_indices]
    scaling = primitives.scaling[active_indices]
    rotation = primitives.rotation[active_indices]

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
    source_sh = None
    sh_degree = 0
    if primitives.source_sh_coefficients is not None:
        source_id = primitives.source_id[active_indices]
        source_sh = primitives.source_sh_coefficients[source_id].to(
            device=device, dtype=dtype
        )
        root = int(round(float(source_sh.shape[1]) ** 0.5))
        if root * root != int(source_sh.shape[1]) or root - 1 > 3:
            raise ValueError("Stage-I SH tensor must encode degree 0--3")
        sh_degree = root - 1
        if primitives.source_base_color is not None:
            base_color = primitives.source_base_color[source_id].to(
                device=device, dtype=dtype
            )
            source_sh = source_sh.clone()
            # RGB correction maps exactly to the DC coefficient. This keeps
            # the native rasterizer SH path bit-compatible with HairGS while
            # retaining trainable route-specific appearance deltas.
            source_sh[:, 0, :] = source_sh[:, 0, :] + (
                color - base_color
            ) / 0.28209479177387814
    settings = GaussianRasterizationSettings(
        image_height=int(camera.height),
        image_width=int(camera.width),
        tanfovx=tan_fov_x,
        tanfovy=tan_fov_y,
        bg=background,
        scale_modifier=float(scale_modifier),
        viewmatrix=view,
        projmatrix=full_projection,
        sh_degree=sh_degree,
        campos=camera_center,
        prefiltered=False,
        debug=debug,
    )

    def rasterize(
        colors: torch.Tensor | None = None,
        shs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        means_2d = torch.zeros_like(
            xyz, dtype=dtype, device=device, requires_grad=True
        )
        means_2d.retain_grad()
        rasterizer = GaussianRasterizer(raster_settings=settings)
        image, radii = rasterizer(
            means3D=xyz,
            means2D=means_2d,
            colors_precomp=colors,
            opacities=opacity[:, None],
            scales=scaling,
            rotations=rotation,
            shs=shs,
            cov3D_precomp=None,
        )
        return image, radii, means_2d

    rendered_rgb, radii, viewspace_points = (
        rasterize(shs=source_sh) if source_sh is not None else rasterize(color)
    )
    rendered_alpha_rgb, _alpha_radii, _alpha_viewspace = rasterize(
        torch.ones_like(color)
    )
    physical_mask = rendered_alpha_rgb.mean(dim=0).clamp(0.0, 1.0)
    semantic_mask = physical_mask
    if primitives.semantic_foreground is not None:
        semantic_color = primitives.semantic_foreground.to(
            device=device, dtype=dtype
        )[active_indices].reshape(-1, 1).expand(-1, 3)
        rendered_semantic, _semantic_radii, _semantic_viewspace = rasterize(
            semantic_color
        )
        semantic_mask = rendered_semantic.mean(dim=0).clamp(0.0, 1.0)
    output = {
        "rgb": rendered_rgb.permute(1, 2, 0).contiguous().clamp(0.0, 1.0),
        "mask": semantic_mask,
        "physical_mask": physical_mask,
        "radii": radii,
        "visibility_filter": radii > 0,
        "viewspace_points": viewspace_points,
        "primitive_indices": active_indices,
    }
    if render_orientation:
        # Match HairGS: use the projected, sign-invariant tangent direction.
        # A double-angle encoding lets Gaussian alpha compositing average
        # directions without the pi-periodic sign ambiguity of hair.
        tangent_world = _quaternion_to_matrix_torch(rotation)[..., :, 0]
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
        # A second rasterized axial moment retains evidence from crossing
        # directions.  A single double-angle vector cannot distinguish a
        # two-mode crossing from low-confidence orientation, while the fourth
        # harmonic remains coherent for orthogonal strands.
        c2, s2 = orientation_color[:, 0], orientation_color[:, 1]
        orientation4_color = torch.stack(
            [c2.square() - s2.square(), 2.0 * c2 * s2, torch.zeros_like(c2)],
            dim=-1,
        )
        rendered_orientation4, _orientation4_radii, _orientation4_viewspace = (
            rasterize(orientation4_color)
        )
        output["orientation4"] = (
            rendered_orientation4.permute(1, 2, 0).contiguous()
        )
    return output


def _eval_sh_color(sh: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Evaluate real 3DGS spherical harmonics (degrees 0--3)."""

    if sh.ndim != 3 or sh.shape[-1] != 3 or direction.shape != (sh.shape[0], 3):
        raise ValueError("Expected SH [N,K,3] and directions [N,3]")
    coefficient_count = int(sh.shape[1])
    if coefficient_count not in {1, 4, 9, 16}:
        raise ValueError(f"Unsupported SH coefficient count: {coefficient_count}")
    c0 = 0.28209479177387814
    c1 = 0.4886025119029199
    c2 = (
        1.0925484305920792,
        -1.0925484305920792,
        0.31539156525252005,
        -1.0925484305920792,
        0.5462742152960396,
    )
    c3 = (
        -0.5900435899266435,
        2.890611442640554,
        -0.4570457994644658,
        0.3731763325901154,
        -0.4570457994644658,
        1.445305721320277,
        -0.5900435899266435,
    )
    result = c0 * sh[:, 0]
    if coefficient_count > 1:
        x, y, z = direction[:, 0:1], direction[:, 1:2], direction[:, 2:3]
        result = result - c1 * y * sh[:, 1] + c1 * z * sh[:, 2] - c1 * x * sh[:, 3]
    if coefficient_count > 4:
        xx, yy, zz = x * x, y * y, z * z
        xy, yz, xz = x * y, y * z, x * z
        result = (
            result
            + c2[0] * xy * sh[:, 4]
            + c2[1] * yz * sh[:, 5]
            + c2[2] * (2.0 * zz - xx - yy) * sh[:, 6]
            + c2[3] * xz * sh[:, 7]
            + c2[4] * (xx - yy) * sh[:, 8]
        )
    if coefficient_count > 9:
        result = (
            result
            + c3[0] * y * (3.0 * xx - yy) * sh[:, 9]
            + c3[1] * xy * z * sh[:, 10]
            + c3[2] * y * (4.0 * zz - xx - yy) * sh[:, 11]
            + c3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh[:, 12]
            + c3[4] * x * (4.0 * zz - xx - yy) * sh[:, 13]
            + c3[5] * z * (xx - yy) * sh[:, 14]
            + c3[6] * x * (xx - 3.0 * yy) * sh[:, 15]
        )
    return (result + 0.5).clamp_min(0.0)


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
