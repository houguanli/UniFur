"""Persistent CUDA deformation and native 3DGS rendering for the live viewer.

Only the small controller array and changed camera parameters cross to the GPU
per frame. All trained Gaussian parameters, the graph binding, and animation
buffers remain resident. A CUDA graph captures active-only Shell transport and
rooted Strand FK; the native rasterizer remains unchanged.

``deform`` returns borrowed CUDA tensors. Their active elements are overwritten
by the next call; use ``snapshot`` for owned NumPy arrays, e.g. before export.
The viewer creates and uses this object on its render worker thread.
"""
from __future__ import annotations

import math
import time
import warnings

import numpy as np
from PIL import Image
import torch

from .preview import CameraSpec


def _quat_from_rotvec(v):
    angle = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    # sinc is continuous at zero; torch.sinc(x) = sin(pi*x)/(pi*x).
    return torch.cat((torch.cos(angle * .5), v * (.5 * torch.sinc(angle / (2 * math.pi)))), dim=-1)


def _quat_apply(q, v):
    t = 2 * torch.linalg.cross(q[..., 1:], v, dim=-1)
    return v + q[..., :1] * t + torch.linalg.cross(q[..., 1:], t, dim=-1)


def _quat_multiply(a, b):
    w, xyz, v, uvw = a[..., :1], a[..., 1:], b[..., :1], b[..., 1:]
    return torch.cat((w * v - (xyz * uvw).sum(-1, keepdim=True),
                      w * uvw + v * xyz + torch.linalg.cross(xyz, uvw, dim=-1)), dim=-1)


class RealtimeGSRenderer:
    """All-Gaussian native renderer, optionally with rooted fur deformation.

    Args:
        asset: a validated GaussianAsset, or CompleteAsset for dynamics.
        bindings: optional Bindings shared with the CPU driver. None selects
            static rendering with no guide graph or deformation allocations.
        device: CUDA device identifier. No diagnostic CPU fallback is implicit.
        capture_graph: disable only for debugging or unsupported CUDA runtimes.

    Modes are ``appearance`` (trained SH), ``motion_mask`` (frozen/animated
    regions), and ``routes`` (Shell/Strand). Every mode uses the original
    Gaussian shapes, opacity, and native alpha compositor, with every splat.
    """

    def __init__(self, asset, bindings=None, device="cuda", *, capture_graph=True):
        from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("The realtime preview requires a CUDA-capable PyTorch runtime.")
        self.asset, self.bindings = asset, bindings
        self._Settings, self._Rasterizer = GaussianRasterizationSettings, GaussianRasterizer
        self.controller_count = len(bindings.controller_guide_ids) if bindings is not None else 0
        self.gaussian_count = len(asset.xyz)
        self.max_sh_degree = min(3, int(math.isqrt(1 + asset.rest.shape[1])) - 1)
        self.last_timings = {}
        self._graph = None
        self.graph_error = None
        self._closed = False
        self._camera_key = None
        self._rasterizer = None
        self._has_motion = False
        self.collision = None
        self._collision_pose = None

        def tensor(array, dtype=torch.float32):
            return torch.as_tensor(np.array(array, copy=True, order="C"), device=self.device, dtype=dtype)

        self._tensor = tensor
        with torch.cuda.device(self.device), torch.inference_mode():
            self.rest_xyz = tensor(asset.xyz)
            self.rest_rotation = tensor(asset.rotation)
            self.rest_nodes = tensor(asset.nodes if bindings is not None else np.empty((0, 3), np.float32))
            self._rest_rotation_render = torch.nn.functional.normalize(self.rest_rotation, dim=-1)
            self.xyz = self.rest_xyz.clone() if bindings is not None else self.rest_xyz
            self.rotation = self.rest_rotation.clone() if bindings is not None else self.rest_rotation
            self.nodes = self.rest_nodes.clone()
            self._rotation_render = self._rest_rotation_render.clone() if bindings is not None else self._rest_rotation_render
            self._means2d = torch.zeros_like(self.rest_xyz)
            self._sh = tensor(np.concatenate((asset.dc, asset.rest), axis=1))
            self._scales = torch.exp(tensor(asset.scale))
            self._opacity = torch.sigmoid(tensor(asset.opacity))
            self._angles = torch.zeros((self.controller_count, 3), dtype=torch.float32, device=self.device)
            self._angles_cpu = torch.empty((self.controller_count, 3), dtype=torch.float32, pin_memory=True)
            self._angles_numpy = self._angles_cpu.numpy()

            if bindings is not None:
                self._init_dynamics(asset, bindings, tensor)

            route = getattr(asset, "route", None)
            self._has_routes = route is not None
            route_colors = np.full((self.gaussian_count, 3), (.16, .20, .25), np.float32)
            if route is not None:
                palette = np.array([[.06, .70, .84], [1., .40, .08], [.40, .85, .28]], np.float32)
                route_colors[:] = palette[route]
            motion_colors = np.full_like(route_colors, (.16, .20, .25))
            if bindings is not None:
                motion_colors[~bindings.frozen_rows] = route_colors[~bindings.frozen_rows]
            self._colors = {"routes": tensor(route_colors), "motion_mask": tensor(motion_colors)}
            self._events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            if capture_graph and bindings is not None:
                self._capture_deformation()
            torch.cuda.synchronize(self.device)
        self.info = {"backend": "persistent_native_cuda", "device": torch.cuda.get_device_name(self.device),
                     "gaussians": self.gaussian_count, "active_gaussians": int((~bindings.frozen_rows).sum()) if bindings is not None else 0,
                     "controllers": self.controller_count, "sh_degree": self.max_sh_degree,
                     "cuda_graph": self._graph is not None, "graph_error": self.graph_error,
                     "all_gaussians": True, "dynamics_initialized": bindings is not None}

    def _init_dynamics(self, asset, bindings, tensor):
        edge_ids = bindings.active_edges
        shell = bindings.active_shell_rows
        guides = np.unique(np.concatenate((asset.edge_guide[edge_ids], asset.gs_guide[shell])))
        guide_local = np.full(asset.guide_count, -1, dtype=np.int64)
        guide_local[guides] = np.arange(len(guides))
        self._control_ids = tensor(bindings.control_ids[guides], torch.long)
        self._control_weights = tensor(bindings.control_weights[guides, :, None])
        self._guide_weight = tensor(bindings.guide_weight[guides, None])
        self._edge_guide = tensor(guide_local[asset.edge_guide[edge_ids]], torch.long)
        root_s = bindings.strand_root_s if bindings.strand_root_s is not None else np.zeros(asset.guide_count)
        edge_root = root_s[asset.edge_guide[edge_ids]]
        self._edge_s2 = tensor((edge_root+(1-edge_root)*asset.node_s[asset.edge_child[edge_ids]])[:, None] ** 2)
        rooted = guides[(guides >= asset.shell_count) & (root_s[guides] > 0)]
        self._skin_rooted = bool(len(rooted))
        if self._skin_rooted:
            base = asset.base_node_by_guide[rooted]
            self._root_nodes = tensor(base, torch.long)
            self._root_guide = tensor(guide_local[rooted], torch.long)
            self._root_anchor = tensor(asset.guide_anchors[rooted])
            self._root_offset = tensor(asset.nodes[base]-asset.guide_anchors[rooted])
            self._root_s2 = tensor(root_s[rooted, None] ** 2)
        self._edge_rows = tensor(asset.strand_rows[edge_ids], torch.long)
        self._edge_a = tensor(asset.edges[edge_ids, 0], torch.long)
        self._edge_b = tensor(asset.edges[edge_ids, 1], torch.long)
        self._edge_residual = tensor(asset.strand_center_residual[edge_ids])
        self._edge_rest_rotation = tensor(asset.rotation[asset.strand_rows[edge_ids]])
        self._layers = []
        for depth in np.unique(asset.edge_depth[edge_ids]):
            local = np.flatnonzero(asset.edge_depth[edge_ids] == depth)
            layer = edge_ids[local]
            self._layers.append((tensor(local, torch.long), tensor(asset.edge_parent[layer], torch.long),
                                 tensor(asset.edge_child[layer], torch.long), tensor(asset.edge_rest_vector[layer])))
        self._shell_rows = tensor(shell, torch.long)
        self._shell_guide = tensor(guide_local[asset.gs_guide[shell]], torch.long)
        shell_index = np.full(self.gaussian_count, -1, np.int64)
        shell_index[bindings.shell_rows] = np.arange(len(bindings.shell_rows))
        self._shell_s2 = tensor(bindings.shell_s[shell_index[shell], None] ** 2)
        self._shell_anchor = tensor(asset.guide_anchors[asset.gs_guide[shell]])
        self._shell_offset = tensor(asset.xyz[shell] - asset.guide_anchors[asset.gs_guide[shell]])
        self._shell_rest_rotation = tensor(asset.rotation[shell])


    def _deform_impl(self):
        guide_angle = (self._angles[self._control_ids] * self._control_weights).sum(1) * self._guide_weight
        edge_q = _quat_from_rotvec(guide_angle[self._edge_guide] * self._edge_s2)
        if self._skin_rooted:
            root_q = _quat_from_rotvec(guide_angle[self._root_guide] * self._root_s2)
            self.nodes[self._root_nodes] = self._root_anchor + _quat_apply(root_q, self._root_offset)
        for local, parent, child, vector in self._layers:
            self.nodes[child] = self.nodes[parent] + _quat_apply(edge_q[local], vector)
        self.xyz[self._edge_rows] = ((self.nodes[self._edge_a] + self.nodes[self._edge_b]) * .5
                                    + _quat_apply(edge_q, self._edge_residual))
        q = torch.nn.functional.normalize(_quat_multiply(edge_q, self._edge_rest_rotation), dim=-1)
        self.rotation[self._edge_rows] = q
        self._rotation_render[self._edge_rows] = q
        shell_q = _quat_from_rotvec(guide_angle[self._shell_guide] * self._shell_s2)
        self.xyz[self._shell_rows] = self._shell_anchor + _quat_apply(shell_q, self._shell_offset)
        q = torch.nn.functional.normalize(_quat_multiply(shell_q, self._shell_rest_rotation), dim=-1)
        self.rotation[self._shell_rows] = q
        self._rotation_render[self._shell_rows] = q

    def _capture_deformation(self):
        try:
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._deform_impl()
            torch.cuda.current_stream(self.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                self._deform_impl()
            torch.cuda.current_stream(self.device).wait_stream(stream)
            self._graph = graph
        except (RuntimeError, NotImplementedError) as error:
            # The persistent path remains usable if capture is unsupported.
            self.graph_error = str(error)
            warnings.warn(f"CUDA deformation graph unavailable; using resident tensor operations: {error}", RuntimeWarning)
            self._graph = None

    def deform(self, control_angles, *, apply_motion=True):
        """Update and return borrowed ``xyz``, ``rotation``, ``nodes`` CUDA tensors.

        Frozen rows are never written (base nodes only for skin-rooted strands). The exact zero-control
        case selects the untouched rest buffers and exactly recovers the asset.
        Device work is asynchronous; ``snapshot`` waits and copies explicitly.
        """
        if self._closed:
            raise RuntimeError("renderer has been closed")
        self._collision_pose = None
        if control_angles is None and self.bindings is None:
            control_angles = np.empty((0, 3), np.float32)
        angles = np.asarray(control_angles, dtype=np.float32)
        if angles.shape != (self.controller_count, 3) or not np.isfinite(angles).all():
            raise ValueError("control_angles must be a finite [controller_count,3] array")
        self._has_motion = bool(apply_motion and np.any(angles))
        with torch.cuda.device(self.device), torch.inference_mode():
            if self._has_motion:
                np.copyto(self._angles_numpy, angles)
                # Complete the tiny upload before reusing this pinned buffer.
                # Direct callers may issue deform() again without waiting for
                # a render/download; an asynchronous copy would race that write.
                self._angles.copy_(self._angles_cpu, non_blocking=False)
                if self._graph is not None:
                    self._graph.replay()
                else:
                    self._deform_impl()
        pose = {"xyz": self.xyz if self._has_motion else self.rest_xyz,
                "rotation": self.rotation if self._has_motion else self.rest_rotation,
                "nodes": self.nodes if self._has_motion else self.rest_nodes}
        # RiggedGSRenderer resolves contact after transporting to world space.
        return self._collide(pose, self.asset.body_vertices) if self.collision is not None and not hasattr(self, "rig") else pose

    def _collide(self, pose, body_vertices):
        if self.collision is None:
            return pose
        self._collision_pose = self.collision.resolve(pose, body_vertices)
        self._rotation_render.copy_(self._collision_pose["rotation"])
        self._has_motion = True
        return self._collision_pose

    def snapshot(self):
        """Copy the current pose to owned CPU arrays, for diagnostics/export."""
        with torch.cuda.device(self.device), torch.inference_mode():
            if self._collision_pose is not None:
                return {key: value.cpu().numpy().copy() for key, value in self._collision_pose.items()}
            return {"xyz": (self.xyz if self._has_motion else self.rest_xyz).cpu().numpy().copy(),
                    "rotation": (self.rotation if self._has_motion else self.rest_rotation).cpu().numpy().copy(),
                    "nodes": (self.nodes if self._has_motion else self.rest_nodes).cpu().numpy().copy()}

    def _camera_renderer(self, camera, background, degree):
        if not isinstance(camera, CameraSpec):
            raise TypeError("camera must be a CameraSpec")
        if camera.width < 1 or camera.height < 1:
            raise ValueError("camera resolution must be positive")
        bg = tuple(float(x) for x in background)
        if len(bg) != 3 or not np.isfinite(bg).all() or any(x < 0 or x > 1 for x in bg):
            raise ValueError("background must contain three finite values in [0,1]")
        key = (camera, bg, degree)
        if key == self._camera_key:
            return self._rasterizer
        view, projection, tx, ty = camera.matrices()
        if not np.isfinite(view).all() or not np.isfinite(projection).all():
            raise ValueError("invalid camera parameters")
        kwargs = dict(image_height=camera.height, image_width=camera.width,
                      tanfovx=tx, tanfovy=ty, bg=self._tensor(bg), scale_modifier=1.,
                      viewmatrix=self._tensor(view.T), projmatrix=self._tensor((projection @ view).T),
                      sh_degree=degree, campos=self._tensor(camera.eye), prefiltered=False, debug=False)
        if "antialiasing" in self._Settings._fields:
            kwargs["antialiasing"] = False
        self._rasterizer = self._Rasterizer(self._Settings(**kwargs))
        self._camera_key = key
        return self._rasterizer

    def _appearance_args(self, xyz, camera, degree):
        """Rasterizer color inputs for the trained appearance."""
        return {"shs": self._sh}

    def render(self, control_angles, camera, *, background=(.94, .95, .97),
               mode="appearance", sh_degree=None, apply_motion=True):
        """Render every Gaussian and return a PIL RGB image for the live UI.

        ``last_timings`` reports CUDA deformation/raster work and complete frame
        time including RGB conversion and GPU-to-CPU download. The Tk upload is
        measured by the UI separately.
        """
        if mode not in ("appearance", "motion_mask", "routes"):
            raise ValueError("mode must be appearance, motion_mask, or routes")
        if mode == "routes" and not self._has_routes:
            raise ValueError("This asset has no route metadata")
        if mode == "motion_mask" and self.bindings is None:
            raise ValueError("Motion masks require initialized UniFur dynamics")
        if sh_degree is None:
            sh_degree = self.max_sh_degree
        if int(sh_degree) != sh_degree or not 0 <= sh_degree <= self.max_sh_degree:
            raise ValueError(f"sh_degree must be between 0 and {self.max_sh_degree}")
        started = time.perf_counter()
        with torch.cuda.device(self.device), torch.inference_mode():
            begin, deform_end, raster_begin, raster_end = self._events
            begin.record()
            pose = self.deform(control_angles, apply_motion=apply_motion)
            deform_end.record()
            rasterizer = self._camera_renderer(camera, background, int(sh_degree))
            rotations = self._rotation_render if self._has_motion else self._rest_rotation_render
            args = dict(means3D=pose["xyz"], means2D=self._means2d, opacities=self._opacity,
                        scales=self._scales, rotations=rotations)
            if mode == "appearance":
                args.update(self._appearance_args(pose["xyz"], camera, int(sh_degree)))
            else:
                args["colors_precomp"] = self._colors[mode]
            raster_begin.record()
            result = rasterizer(**args)
            raster_end.record()
            downloading = time.perf_counter()
            pixels = result[0].clamp(0, 1).permute(1, 2, 0).mul(255).round().to(torch.uint8).cpu().numpy()
            downloaded = time.perf_counter()
            deform_ms = begin.elapsed_time(deform_end)
            raster_ms = raster_begin.elapsed_time(raster_end)
        image = Image.fromarray(pixels, "RGB")
        frame_ms = (time.perf_counter() - started) * 1000
        self.last_timings = {"deform_ms": deform_ms, "raster_ms": raster_ms,
                             "convert_download_wait_ms": (downloaded - downloading) * 1000,
                             "frame_ms": frame_ms, "fps": 1000 / max(frame_ms, 1e-6)}
        self.last_timings["collision"] = self.collision.last_stats.copy() if self.collision is not None else {"enabled": False}
        image.info["renderer"] = dict(self.info, sh_degree=int(sh_degree), mode=mode,
                                      timings=self.last_timings.copy())
        return image

    def close(self):
        """Release the captured graph; ordinary tensors follow Python lifetime."""
        if self._graph is not None:
            torch.cuda.synchronize(self.device)
            # Destruction releases the graph. PyTorch 2.1 resets it again in
            # its destructor, so an explicit reset here double-releases its pool.
            self._graph = None
        self._closed = True
