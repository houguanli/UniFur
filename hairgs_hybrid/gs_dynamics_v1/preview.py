"""Small, independent Gaussian preview renderer for the dynamics prototype.

The CUDA backend uses the installed Graphdeco rasterizer (all splats and degree-3
SH).  The CPU fallback is explicitly a diagnostic approximation: perspective
projected anisotropic covariances, DC colour, and depth-sorted alpha compositing.
It never replaces Gaussians with fixed-size point sprites.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import radians, tan
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class CameraSpec:
    width: int
    height: int
    eye: tuple[float, float, float]
    target: tuple[float, float, float]
    fov_y_degrees: float = 35.0
    near: float = 0.01
    far: float = 100.0
    world_up: tuple[float, float, float] = (0., 0., 1.)

    def to_dict(self) -> dict:
        return asdict(self)

    def matrices(self):
        eye, target = np.asarray(self.eye), np.asarray(self.target)
        forward = target - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, self.world_up)
        if np.linalg.norm(right) < 1e-8:
            right = np.cross(forward, np.eye(3)[np.abs(forward).argmin()])
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        view = np.eye(4, dtype=np.float32)
        view[:3, :3] = np.stack((right, down, forward))
        view[:3, 3] = -view[:3, :3] @ eye
        ty = tan(radians(self.fov_y_degrees) / 2)
        tx = ty * self.width / self.height
        projection = np.zeros((4, 4), dtype=np.float32)
        projection[0, 0], projection[1, 1] = 1 / tx, 1 / ty
        projection[2, 2] = self.far / (self.far - self.near)
        projection[2, 3] = -self.far * self.near / (self.far - self.near)
        projection[3, 2] = 1
        return view, projection, tx, ty


def make_camera(xyz, width=768, height=768, view="three_quarter", target=None,
                margin=1.14, fov_y_degrees=35.0) -> CameraSpec:
    """Fit a fixed Z-up perspective camera to the *rest* pose.

    Pass this same object for every frame so motion is not cancelled by an
    animated crop.  ``side`` looks from +X, and the lion faces approximately -Y.
    """
    xyz = np.asarray(xyz)
    lo, hi = np.quantile(xyz, (0.001, 0.999), axis=0)
    target = (lo + hi) / 2 if target is None else np.asarray(target, dtype=float)
    offsets = {"three_quarter": (1.5, -2.5, 0.7),
               "side": (3.0, -0.12, 0.3),
               "rear": (1.6, 2.5, 0.6),
               "front": (0.15, -3.0, 0.35)}
    if view not in offsets:
        raise ValueError(f"Unknown camera view {view!r}; choose {tuple(offsets)}")
    direction = np.asarray(offsets[view], dtype=float)
    direction /= np.linalg.norm(direction)
    forward = -direction
    right = np.cross(forward, (0., 0., 1.))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    # Fit the eight robust-bound corners with explicit image margin.
    corners = np.asarray([[x, y, z] for x in (lo[0], hi[0])
                          for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]) - target
    px, py, pz = corners @ right, corners @ down, corners @ forward
    ty = tan(radians(fov_y_degrees) / 2)
    distance = max(np.max(margin * np.abs(px) / (ty * width / height) - pz),
                   np.max(margin * np.abs(py) / ty - pz))
    distance = max(float(distance), 0.1)
    return CameraSpec(int(width), int(height), tuple(target + direction * distance),
                      tuple(target), float(fov_y_degrees), 0.01,
                      max(100., distance * 10))


def renderer_info() -> dict:
    """Describe availability without starting a render or changing the system."""
    try:
        import torch
        import diff_gaussian_rasterization as dgr
        if torch.cuda.is_available():
            return {"backend": "native_cuda", "torch": torch.__version__,
                    "device": torch.cuda.get_device_name(), "sh_degree": 3,
                    "implementation": dgr.__file__, "all_gaussians": True}
        reason = "CUDA unavailable"
    except (ImportError, OSError) as error:
        reason = str(error)
    return {"backend": "cpu_covariance_diagnostic", "reason": reason,
            "sh_degree": 0, "all_gaussians": False,
            "default_point_limit": 24000}


def render_frame(xyz, log_scale, rotation_wxyz, dc, rest, opacity_logits,
                 camera: CameraSpec, *, backend="auto", max_points=None,
                 background=(0.94, 0.95, 0.97)) -> Image.Image:
    """Render standard 3DGS arrays; inputs and camera are never modified.

    ``backend='auto'`` selects native CUDA when available, otherwise the labeled
    CPU diagnostic.  ``backend='native_cuda'`` never silently falls back after a
    CUDA failure.  Native rendering uses all points unless max_points is given.
    CPU rendering defaults to deterministic 24k subsampling, DC-only appearance,
    and records this limitation in ``image.info['renderer']``.
    """
    if backend == "auto":
        backend = renderer_info()["backend"]
    if backend in ("cuda", "native_cuda"):
        return _render_cuda(xyz, log_scale, rotation_wxyz, dc, rest,
                            opacity_logits, camera, max_points, background)
    if backend in ("cpu", "cpu_covariance_diagnostic"):
        return _render_cpu(xyz, log_scale, rotation_wxyz, dc, opacity_logits,
                           camera, max_points, background)
    raise ValueError(f"Unknown render backend: {backend}")


def _indices(count, limit):
    if limit is None or count <= limit:
        return slice(None)
    if int(limit) < 1:
        raise ValueError("max_points must be positive")
    return np.linspace(0, count - 1, int(limit), dtype=np.int64)


def _render_cuda(xyz, log_scale, rotation, dc, rest, opacity, camera, limit, bg):
    import torch
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

    idx = _indices(len(xyz), limit)

    def tensor(array):
        # CompleteAsset freezes its NumPy storage. Give the Torch bridge owned,
        # writable memory even though transfer to CUDA also copies its data.
        return torch.as_tensor(np.array(array, copy=True, order="C"), device="cuda", dtype=torch.float32)

    view, projection, tx, ty = camera.matrices()
    kwargs = dict(image_height=camera.height, image_width=camera.width,
                  tanfovx=tx, tanfovy=ty, bg=tensor(bg), scale_modifier=1.0,
                  viewmatrix=tensor(view.T), projmatrix=tensor((projection @ view).T),
                  sh_degree=3, campos=tensor(camera.eye), prefiltered=False, debug=False)
    if "antialiasing" in GaussianRasterizationSettings._fields:
        kwargs["antialiasing"] = False  # match the original, unfiltered 3DGS model
    with torch.inference_mode():
        means = tensor(np.asarray(xyz)[idx])
        rotations = tensor(np.asarray(rotation)[idx])
        rotations = torch.nn.functional.normalize(rotations, dim=-1)
        sh = tensor(np.concatenate((np.asarray(dc)[idx], np.asarray(rest)[idx]), axis=1))
        rasterizer = GaussianRasterizer(GaussianRasterizationSettings(**kwargs))
        output = rasterizer(means3D=means, means2D=torch.zeros_like(means),
                            shs=sh, opacities=torch.sigmoid(tensor(np.asarray(opacity)[idx])),
                            scales=torch.exp(tensor(np.asarray(log_scale)[idx])),
                            rotations=rotations)
        rgb = output[0].clamp(0, 1).permute(1, 2, 0).mul(255).round().to(torch.uint8).cpu().numpy()
    image = Image.fromarray(rgb, "RGB")
    image.info["renderer"] = {"backend": "native_cuda", "gaussians": len(means),
                              "total_gaussians": len(xyz), "sh_degree": 3}
    return image


def _quat_matrix(q):
    q = np.asarray(q, dtype=np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    w, x, y, z = q.T
    return np.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                     2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                     2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), axis=-1).reshape(-1,3,3)


def _render_cpu(xyz, log_scale, rotation, dc, opacity, camera, limit, bg):
    # Approximate splatting only: centre-depth sorting, DC colour and bounded
    # support; covariance propagation and alpha still represent the Gaussians.
    idx = _indices(len(xyz), 24000 if limit is None else limit)
    xyz, logs = np.asarray(xyz)[idx], np.asarray(log_scale)[idx]
    q = _quat_matrix(np.asarray(rotation)[idx])
    colour = np.clip(0.5 + 0.28209479177387814 * np.asarray(dc)[idx, 0], 0, 1)
    alpha = 1 / (1 + np.exp(-np.clip(np.asarray(opacity)[idx, 0], -40, 40)))
    view, _, tx, ty = camera.matrices()
    p = xyz @ view[:3, :3].T + view[:3, 3]
    cov = (q * np.exp(2 * logs)[:, None, :]) @ q.transpose(0, 2, 1)
    cov = view[:3, :3] @ cov @ view[:3, :3].T
    fx, fy = camera.width / (2*tx), camera.height / (2*ty)
    z = np.maximum(p[:, 2], camera.near)
    u, v = fx*p[:,0]/z + (camera.width-1)/2, fy*p[:,1]/z + (camera.height-1)/2
    jac = np.zeros((len(p),2,3))
    jac[:,0,0], jac[:,1,1] = fx/z, fy/z
    jac[:,0,2], jac[:,1,2] = -fx*p[:,0]/z**2, -fy*p[:,1]/z**2
    cov2 = jac @ cov @ jac.transpose(0,2,1) + 0.3*np.eye(2)
    # Back-to-front over is equivalent to front-to-back transmittance for a
    # centre-depth ordering, with the usual ambiguity for intersecting splats.
    image = np.empty((camera.height,camera.width,3),np.float32)
    image[:] = bg
    for i in np.argsort(p[:,2])[::-1]:
        if p[i,2] < camera.near or p[i,2] > camera.far or alpha[i] < 1/255:
            continue
        radius = min(int(np.ceil(3*np.sqrt(np.linalg.eigvalsh(cov2[i])[-1]))), max(camera.width,camera.height))
        x0,x1 = max(0,int(u[i])-radius),min(camera.width,int(u[i])+radius+1)
        y0,y1 = max(0,int(v[i])-radius),min(camera.height,int(v[i])+radius+1)
        if x0>=x1 or y0>=y1:
            continue
        yy,xx = np.mgrid[y0:y1,x0:x1]
        delta = np.stack((xx-u[i],yy-v[i]),axis=-1)
        inv = np.linalg.inv(cov2[i])
        exponent = np.einsum("...i,ij,...j->...",delta,inv,delta)
        a = np.minimum(0.99,alpha[i]*np.exp(-0.5*exponent))
        a[exponent > 9] = 0
        patch=image[y0:y1,x0:x1]
        patch[:] = colour[i]*a[...,None]+patch*(1-a[...,None])
    result=Image.fromarray(np.uint8(np.clip(image,0,1)*255),"RGB")
    result.info["renderer"]={"backend":"cpu_covariance_diagnostic", "gaussians":len(p),
                              "total_gaussians":len(xyz), "sh_degree":0}
    return result
