"""Source-mesh animation transport with existing root-relative fur dynamics.

The source clip is evaluated by Blender once; this module uploads one source
vertex frame and a small controller array at runtime. Trained Gaussian data,
attachment maps and deformation buffers remain resident on CUDA. No Blender
process, per-frame closest-point query or per-Gaussian CPU deformation is used.
"""
from __future__ import annotations

import numpy as np
import torch
import warnings

from .realtime_gpu import RealtimeGSRenderer, _quat_apply, _quat_multiply


def _matrix_to_quaternion(matrix):
    """Stable scalar-first quaternion conversion, including half turns."""
    m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]
    squares = torch.stack((1+m00+m11+m22, 1+m00-m11-m22,
                           1-m00+m11-m22, 1-m00-m11+m22), dim=-1)
    magnitudes = squares.clamp_min(0).sqrt()
    choices = torch.stack((
        torch.stack((squares[..., 0], m21-m12, m02-m20, m10-m01), dim=-1),
        torch.stack((m21-m12, squares[..., 1], m10+m01, m02+m20), dim=-1),
        torch.stack((m02-m20, m10+m01, squares[..., 2], m12+m21), dim=-1),
        torch.stack((m10-m01, m20+m02, m21+m12, squares[..., 3]), dim=-1)), dim=-2)
    choices = choices / (2*magnitudes.clamp_min(.1)[..., :, None])
    choice = magnitudes.argmax(-1)
    q = choices.gather(-2, choice[..., None, None].expand(choice.shape+(1, 4))).squeeze(-2)
    return torch.nn.functional.normalize(q, dim=-1)


def _triangle_frames(triangles):
    """Right-handed local material frame for each oriented triangle."""
    x = torch.nn.functional.normalize(triangles[:, 1]-triangles[:, 0], dim=-1, eps=1e-12)
    z = torch.nn.functional.normalize(torch.linalg.cross(x, triangles[:, 2]-triangles[:, 0], dim=-1), dim=-1, eps=1e-12)
    y = torch.linalg.cross(z, x, dim=-1)
    return torch.stack((x, y, z), dim=-1)


_SH_C0 = 0.28209479177387814
_SH_C1 = 0.4886025119029199
_SH_C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396)
_SH_C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
          -0.4570457994644658, 1.445305721320277, -0.5900435899266435)


def _sh_to_rgb(sh, directions, degree):
    """The native rasterizer's SH color (+0.5, clamped at 0) for given unit rays."""
    x, y, z = directions[:, 0:1], directions[:, 1:2], directions[:, 2:3]
    color = _SH_C0*sh[:, 0]
    if degree > 0:
        color = color - _SH_C1*y*sh[:, 1] + _SH_C1*z*sh[:, 2] - _SH_C1*x*sh[:, 3]
    if degree > 1:
        xx, yy, zz, xy, yz, xz = x*x, y*y, z*z, x*y, y*z, x*z
        color = (color + _SH_C2[0]*xy*sh[:, 4] + _SH_C2[1]*yz*sh[:, 5] + _SH_C2[2]*(2*zz-xx-yy)*sh[:, 6]
                 + _SH_C2[3]*xz*sh[:, 7] + _SH_C2[4]*(xx-yy)*sh[:, 8])
    if degree > 2:
        color = (color + _SH_C3[0]*y*(3*xx-yy)*sh[:, 9] + _SH_C3[1]*xy*z*sh[:, 10]
                 + _SH_C3[2]*y*(4*zz-xx-yy)*sh[:, 11] + _SH_C3[3]*z*(2*zz-3*xx-3*yy)*sh[:, 12]
                 + _SH_C3[4]*x*(4*zz-xx-yy)*sh[:, 13] + _SH_C3[5]*z*(xx-yy)*sh[:, 14]
                 + _SH_C3[6]*x*(xx-3*yy)*sh[:, 15])
    return (color+.5).clamp_min(0).contiguous()


def _reference_vertices(rig):
    if hasattr(rig, "driver_vertices_rest"):
        return np.asarray(rig.driver_vertices_rest, np.float32)
    numbers = np.asarray(rig.frame_numbers)
    found = np.flatnonzero(numbers == rig.reference_frame)
    if len(found) != 1:
        raise ValueError("Rig reference_frame must occur exactly once in frame_numbers")
    return np.asarray(rig.frames_vertices[int(found[0])], np.float32)


class RiggedGSRenderer(RealtimeGSRenderer):
    """Animated body and root-relative fur, sharing the native GS renderer.

    ``rig`` supplies driver_faces, driver_vertices_rest (or frames_vertices plus
    reference_frame/frame_numbers), gs_face/gs_bary, and body_driver_face /
    body_driver_bary / body_driver_offset. All coordinates are the GS asset's
    canonical coordinates. GS rest offsets may be supplied as gs_rest_offset.

    Every Gaussian, simulated or not, follows its own source triangle; the
    canonical fur bend is layered on as a displacement/rotation in that same
    frame. Carrying long mane strands rigidly with their root guide frames
    instead tears them from the surrounding fill when the neck twists.

    ``set_rig_pose`` sets source mesh vertices. ``deform`` always applies that
    body pose; apply_motion controls the extra fur bend only. Call
    ``controller_state`` after set_rig_pose for small CPU root-motion inputs.
    """

    def __init__(self, asset, bindings, rig, device="cuda", *, capture_graph=True):
        super().__init__(asset, bindings, device, capture_graph=capture_graph)
        self.rig = rig
        reference = np.ascontiguousarray(_reference_vertices(rig), dtype=np.float32)
        faces = np.asarray(rig.driver_faces, np.int64)
        gs_face = np.asarray(rig.gs_face, np.int64)
        gs_bary = np.asarray(rig.gs_bary, np.float32)
        body_face = np.asarray(rig.body_driver_face, np.int64)
        body_bary = np.asarray(rig.body_driver_bary, np.float32)
        body_offset = np.asarray(rig.body_driver_offset, np.float32)
        if (reference.ndim != 2 or reference.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3
                or gs_face.shape != (len(asset.xyz),) or gs_bary.shape != (len(asset.xyz), 3)
                or body_face.shape != (len(asset.body_vertices),)
                or body_bary.shape != (len(asset.body_vertices), 3)
                or body_offset.shape != (len(asset.body_vertices), 3)):
            raise ValueError("Rig attachment array shapes do not match the complete asset")
        if not all(np.isfinite(a).all() for a in (reference, gs_bary, body_bary, body_offset)):
            raise ValueError("Rig attachment arrays must be finite")
        if (len(faces) == 0 or faces.min() < 0 or faces.max() >= len(reference)
                or gs_face.min() < 0 or gs_face.max() >= len(faces)
                or body_face.min() < 0 or body_face.max() >= len(faces)):
            raise ValueError("Rig attachment index is out of range")
        if hasattr(rig, "body_vertices_rest") and not np.allclose(rig.body_vertices_rest, asset.body_vertices, atol=1e-10, rtol=0):
            raise ValueError("Rig belongs to a different canonical body mesh")
        if hasattr(rig, "body_faces") and not np.array_equal(rig.body_faces, asset.body_faces):
            raise ValueError("Rig body topology differs from the asset")
        self._driver_reference_numpy = reference.copy()
        self._rig_at_rest = True
        self._pose_dirty = True
        self._world_at_rest = True
        self._body_graph = self._rig_graph = None
        self.rig_graph_error = None
        with torch.cuda.device(self.device), torch.inference_mode():
            tensor = self._tensor
            self._driver_faces = tensor(faces, torch.long)
            self._driver_vertices = tensor(reference)
            self._driver_upload = torch.empty(reference.shape, dtype=torch.float32, pin_memory=True)
            self._driver_upload_numpy = self._driver_upload.numpy()
            ref_tri = self._driver_vertices[self._driver_faces]
            self._driver_rest_frames_t = _triangle_frames(ref_tri).transpose(-1, -2).contiguous()
            self._gs_face = tensor(gs_face, torch.long)
            self._gs_bary = tensor(gs_bary[:, :, None])
            gs_anchor = np.sum(reference[faces[gs_face]]*gs_bary[:, :, None], axis=1)
            offsets = np.asarray(getattr(rig, "gs_rest_offset", asset.xyz-gs_anchor), np.float32)
            if offsets.shape != asset.xyz.shape or not np.isfinite(offsets).all():
                raise ValueError("Invalid Gaussian rest offsets")
            self._gs_rig_offset = tensor(offsets)
            self._body_driver_face = tensor(body_face, torch.long)
            self._body_driver_bary = tensor(body_bary[:, :, None])
            self._body_driver_offset = tensor(body_offset)
            self._body_faces = tensor(asset.body_faces, torch.long)
            self._body_vertices_rest = tensor(asset.body_vertices)
            self._body_vertices = self._body_vertices_rest.clone()
            self._body_rest_frames_t = _triangle_frames(self._body_vertices_rest[self._body_faces]).transpose(-1, -2).contiguous()
            # Strand nodes borrow the source triangle of an incident strand row.
            node_row = np.full(len(asset.nodes), -1, np.int64)
            node_row[asset.edges[:, 0]] = asset.strand_rows
            node_row[asset.edges[:, 1]] = asset.strand_rows
            if np.any(node_row < 0):
                raise ValueError("Every strand node must belong to a strand edge")
            node_anchor = np.sum(reference.astype(np.float64)[faces[gs_face[node_row]]]
                                 * gs_bary[node_row, :, None], axis=1)
            self._node_row = tensor(node_row, torch.long)
            self._node_face = tensor(gs_face[node_row], torch.long)
            self._node_bary = tensor(gs_bary[node_row, :, None])
            self._node_offset = tensor(asset.nodes-node_anchor)
            self._guide_faces = tensor(asset.anchor_face, torch.long)
            self._guide_bary = tensor(asset.anchor_bary[:, :, None])
            self._guide_normal_rest = tensor(asset.guide_normals)
            self._active_fur_rows = tensor(np.flatnonzero(~bindings.frozen_rows), torch.long)
            self._controller_guides = tensor(bindings.controller_guide_ids, torch.long)
            self._world_xyz = self.rest_xyz.clone()
            self._world_rotation = self.rest_rotation.clone()
            self._world_nodes = self.rest_nodes.clone()
            self._rig_rest_mask = torch.ones((), dtype=torch.bool, device=self.device)
            self._fur_enabled_mask = torch.zeros((), dtype=torch.bool, device=self.device)
            self._conjugate = tensor(np.array([1., -1., -1., -1.]))
            self.sh_follows_body = True
            self._update_body_frames()
            if capture_graph:
                self._capture_rig_operations()
        self.info = dict(self.info, rigged=True, body_animation=True,
                         rig_driver_vertices=len(reference), rig_driver_triangles=len(faces),
                         rig_cuda_graph=self._rig_graph is not None, rig_graph_error=self.rig_graph_error,
                         body_transfer="source evaluated triangles; per-Gaussian surface frame for body and fur bend",
                         sh_view_frame="per-Gaussian trained-pose frame")

    def set_rig_pose(self, vertices):
        vertices = np.asarray(vertices, np.float32)
        if vertices.shape != self._driver_reference_numpy.shape or not np.isfinite(vertices).all():
            raise ValueError("Rig pose must be a finite [driver_vertex_count,3] array")
        self._rig_at_rest = bool(np.array_equal(vertices, self._driver_reference_numpy))
        with torch.cuda.device(self.device), torch.inference_mode():
            np.copyto(self._driver_upload_numpy, vertices)
            self._driver_vertices.copy_(self._driver_upload, non_blocking=False)
            self._rig_rest_mask.fill_(self._rig_at_rest)
        self._pose_dirty = True

    def _update_body_frames(self):
        if not self._pose_dirty:
            return
        if self._body_graph is not None:
            self._body_graph.replay()
        else:
            self._update_body_frames_impl()
        self._pose_dirty = False

    def _update_body_frames_impl(self):
        driver_tri = self._driver_vertices[self._driver_faces]
        driver_rotation = _triangle_frames(driver_tri) @ self._driver_rest_frames_t
        self._driver_triangles = driver_tri
        self._driver_rotation = driver_rotation
        self._body_vertices[:] = ((driver_tri[self._body_driver_face]*self._body_driver_bary).sum(1)
                                  + (driver_rotation[self._body_driver_face] @ self._body_driver_offset[..., None]).squeeze(-1))
        # A tensor condition keeps the identity correction dynamic inside the
        # captured graph as playback leaves and later returns to frame zero.
        self._body_vertices.copy_(torch.where(self._rig_rest_mask, self._body_vertices_rest, self._body_vertices))
        body_tri = self._body_vertices[self._body_faces]
        self._body_rotation = _triangle_frames(body_tri) @ self._body_rest_frames_t
        self._guide_rotation = self._body_rotation[self._guide_faces]
        self._guide_anchor = (body_tri[self._guide_faces]*self._guide_bary).sum(1)

    def _rig_deform_impl(self):
        gs_rotation = self._driver_rotation[self._gs_face]
        gs_q = _matrix_to_quaternion(gs_rotation)
        gs_anchor = (self._driver_triangles[self._gs_face]*self._gs_bary).sum(1)
        self._world_xyz[:] = gs_anchor + _quat_apply(gs_q, self._gs_rig_offset)
        self._world_rotation[:] = torch.nn.functional.normalize(_quat_multiply(gs_q, self.rest_rotation), dim=-1)
        # Parent buffers always contain canonical coordinates. Selecting rest
        # with a device scalar also works after disabling an already bent pose.
        local_xyz = torch.where(self._fur_enabled_mask, self.xyz, self.rest_xyz)
        local_rotation = torch.where(self._fur_enabled_mask, self.rotation, self.rest_rotation)
        local_nodes = torch.where(self._fur_enabled_mask, self.nodes, self.rest_nodes)
        rows = self._active_fur_rows
        fur_q = gs_q[rows]
        self._world_xyz[rows] += _quat_apply(fur_q, local_xyz[rows]-self.rest_xyz[rows])
        self._world_rotation[rows] = torch.nn.functional.normalize(_quat_multiply(fur_q, local_rotation[rows]), dim=-1)
        node_anchor = (self._driver_triangles[self._node_face]*self._node_bary).sum(1)
        self._world_nodes[:] = node_anchor + _quat_apply(gs_q[self._node_row],
                                                         self._node_offset+local_nodes-self.rest_nodes)
        self._rotation_render.copy_(self._world_rotation)

    def _capture_rig_operations(self):
        def capture(function):
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    function()
            torch.cuda.current_stream(self.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                function()
            torch.cuda.current_stream(self.device).wait_stream(stream)
            return graph
        try:
            # The body graph establishes stable frame/anchor tensor addresses.
            # The world graph consumes those addresses plus the parent's
            # captured local-fur buffers; no per-frame Python kernel storm.
            self._body_graph = capture(self._update_body_frames_impl)
            self._rig_graph = capture(self._rig_deform_impl)
        except (RuntimeError, NotImplementedError) as error:
            self.rig_graph_error = str(error)
            warnings.warn(f"Rig CUDA graph unavailable; using tensor operations: {error}", RuntimeWarning)
            self._rig_graph = None

    def _appearance_args(self, xyz, camera, degree):
        # SH were fitted in the trained pose, so evaluate them with each view ray
        # rotated back into that Gaussian's trained frame: view-dependent fur
        # shading turns with the head. The rest pose keeps the native SH path.
        if self._world_at_rest or not self.sh_follows_body:
            return super()._appearance_args(xyz, camera, degree)
        rays = torch.nn.functional.normalize(xyz-self._tensor(camera.eye), dim=-1)
        rotation = self._collision_pose["rotation"] if self._collision_pose is not None else self._world_rotation
        turn = _quat_multiply(rotation, self._rest_rotation_render*self._conjugate)
        local = _quat_apply(torch.nn.functional.normalize(turn, dim=-1)*self._conjugate, rays)
        return {"colors_precomp": _sh_to_rgb(self._sh, local, degree)}

    def controller_state(self):
        """Copy only controller anchors/frames/normals to CPU for live forces."""
        if self._closed:
            raise RuntimeError("renderer has been closed")
        with torch.cuda.device(self.device), torch.inference_mode():
            self._update_body_frames()
            ids = self._controller_guides
            rotation = self._guide_rotation[ids]
            normal = (rotation @ self._guide_normal_rest[ids, :, None]).squeeze(-1)
            return {"anchors": self._guide_anchor[ids].cpu().numpy().copy(),
                    "rotations": rotation.cpu().numpy().copy(),
                    "normals": normal.cpu().numpy().copy()}

    def deform(self, control_angles, *, apply_motion=True):
        # Parent buffers stay in canonical/rest space. Only its captured local
        # fur graph runs here; world-space output has separate persistent data.
        super().deform(control_angles, apply_motion=apply_motion)
        fur_active = self._has_motion
        with torch.cuda.device(self.device), torch.inference_mode():
            self._update_body_frames()
            self._world_at_rest = self._rig_at_rest and not fur_active
            if self._world_at_rest:
                self._has_motion = False
                return self._collide({"xyz": self.rest_xyz, "rotation": self.rest_rotation, "nodes": self.rest_nodes}, self._body_vertices)
            self._fur_enabled_mask.fill_(fur_active)
            if self._rig_graph is not None:
                self._rig_graph.replay()
            else:
                self._rig_deform_impl()
            self._has_motion = True
            return self._collide({"xyz": self._world_xyz, "rotation": self._world_rotation, "nodes": self._world_nodes}, self._body_vertices)

    def snapshot(self):
        with torch.cuda.device(self.device), torch.inference_mode():
            if self._collision_pose is not None:
                return dict({key: value.cpu().numpy().copy() for key, value in self._collision_pose.items()},
                            body_vertices=self._body_vertices.cpu().numpy().copy())
            xyz = self.rest_xyz if self._world_at_rest else self._world_xyz
            rotation = self.rest_rotation if self._world_at_rest else self._world_rotation
            nodes = self.rest_nodes if self._world_at_rest else self._world_nodes
            return {"xyz": xyz.cpu().numpy().copy(), "rotation": rotation.cpu().numpy().copy(),
                    "nodes": nodes.cpu().numpy().copy(), "body_vertices": self._body_vertices.cpu().numpy().copy()}

    def close(self):
        if self._body_graph is not None or self._rig_graph is not None:
            torch.cuda.synchronize(self.device)
            # Let graph destruction release each pool once (also on PyTorch 2.1).
            self._body_graph = self._rig_graph = None
        super().close()
