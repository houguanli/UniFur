"""One-way body contact: voxel SDF on CPU, batched fur projection on CUDA.

The field follows the current body mesh. Authored rest-pose embedding is kept
as a per-point allowance; new penetration below that allowance is corrected.
This is a position constraint, without self-contact, friction or CCD.
"""
from __future__ import annotations

import time
import numpy as np
import torch
from torch.nn import functional as F

from .realtime_gpu import _quat_apply, _quat_multiply


class BodyDistanceField:
    def __init__(self, vertices, faces, *, resolution=128, device="cuda"):
        import trimesh
        if type(resolution) is not int or not 32 <= resolution <= 256:
            raise ValueError("SDF resolution must be an integer between 32 and 256")
        self.faces = np.array(faces, np.int64, copy=True)
        vertices = np.asarray(vertices, np.float32)
        if (vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4
                or self.faces.ndim != 2 or self.faces.shape[1] != 3 or not len(self.faces)
                or self.faces.min() < 0 or self.faces.max() >= len(vertices)):
            raise ValueError("Invalid SDF body mesh")
        if not trimesh.Trimesh(vertices, self.faces, process=False).is_watertight:
            raise ValueError("Body SDF requires a closed, watertight body mesh")
        self.resolution, self.device = resolution, torch.device(device)
        self.vertices = None
        self.update(vertices)

    def update(self, vertices):
        import trimesh
        from scipy.ndimage import distance_transform_edt
        started = time.perf_counter()
        if isinstance(vertices, torch.Tensor):
            vertices = vertices.detach().cpu().numpy()
        vertices = np.asarray(vertices, np.float32)
        if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
                or (self.vertices is not None and vertices.shape != self.vertices.shape)):
            raise ValueError("Body pose must be a finite, topology-preserving vertex array")
        self.build_ms = 0.
        if self.vertices is not None and np.array_equal(vertices, self.vertices):
            return
        extent = float(np.ptp(vertices, axis=0).max())
        if extent <= 1e-8:
            raise ValueError("Degenerate body bounds")
        self.pitch = extent / self.resolution
        mesh = trimesh.Trimesh(vertices, self.faces, process=False)
        # ponytail: rebuild the small CPU voxel grid for each pose; use a GPU
        # SDF builder if measured construction time prevents interactive use.
        voxels = mesh.voxelized(self.pitch).fill()
        inside = np.pad(np.asarray(voxels.matrix, bool), 3)
        outside_distance = distance_transform_edt(~inside)
        inside_distance, nearest_outside = distance_transform_edt(inside, return_indices=True)
        distance = ((outside_distance - inside_distance) * self.pitch).astype(np.float32)
        gradient = np.stack(np.gradient(distance, self.pitch), axis=-1)
        flat = np.linalg.norm(gradient, axis=-1) < 1e-5
        if flat.any():
            index = np.stack(np.nonzero(flat), axis=1)
            direction = nearest_outside[:, flat].T - index
            gradient[flat] = direction / np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1.)
        channels = np.concatenate((distance[..., None], gradient), axis=-1)
        self.volume = torch.as_tensor(np.ascontiguousarray(channels.transpose(3, 2, 1, 0)[None]),
                                      dtype=torch.float32, device=self.device)
        self.origin = torch.as_tensor(voxels.transform[:3, 3] - 3*self.pitch,
                                      dtype=torch.float32, device=self.device)
        self.extent = torch.as_tensor((np.asarray(inside.shape)-1)*self.pitch,
                                      dtype=torch.float32, device=self.device)
        self.shape = list(inside.shape)
        self.vertices = vertices.copy()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.build_ms = (time.perf_counter()-started)*1000

    def sample(self, points):
        if points.numel() == 0:
            return points.new_empty((0,)), points.new_empty((0, 3))
        grid = (2*(points-self.origin)/self.extent-1).reshape(1, 1, 1, -1, 3)
        values = F.grid_sample(self.volume, grid, mode="bilinear", padding_mode="border",
                               align_corners=True)[0, :, 0, 0].T
        # The padded boundary is outside the body; extend distance beyond it.
        clipped = torch.maximum(self.origin, torch.minimum(points, self.origin+self.extent))
        distance = values[:, 0] + torch.linalg.vector_norm(points-clipped, dim=-1)
        normal = F.normalize(values[:, 1:], dim=-1, eps=1e-8)
        fallback = torch.zeros_like(normal)
        fallback[:, 2] = 1.
        normal = torch.where((normal.square().sum(-1) > .1)[:, None], normal, fallback)
        return distance, normal

    def project(self, points, clearance, steps=1):
        for _ in range(steps):
            distance, normal = self.sample(points)
            correction = (clearance-distance).clamp(min=0., max=4*self.pitch)
            points = points + correction[:, None]*normal
        return points


def _turn_between(before, after):
    a, b = F.normalize(before, dim=-1), F.normalize(after, dim=-1)
    dot = (a*b).sum(-1, keepdim=True).clamp(-1., 1.)
    quaternion = torch.cat((1+dot, torch.linalg.cross(a, b)), dim=-1)
    helper = F.one_hot(a.abs().argmin(-1), 3).to(a.dtype)
    half_turn = torch.cat((torch.zeros_like(dot), F.normalize(torch.linalg.cross(a, helper), dim=-1)), dim=-1)
    return F.normalize(torch.where(dot < -.99999, half_turn, quaternion), dim=-1)


class FurBodyCollision:
    def __init__(self, asset, bindings, *, resolution=128, margin=.003, iterations=12, device="cuda"):
        if not np.isfinite(margin) or not 0 <= margin <= .03:
            raise ValueError("SDF margin must be between 0 and .03 source units")
        if type(iterations) is not int or not 1 <= iterations <= 64:
            raise ValueError("Collision iterations must be between 1 and 64")
        self.field = BodyDistanceField(asset.body_vertices, asset.body_faces,
                                        resolution=resolution, device=device)
        self.device, self.margin, self.iterations = self.field.device, float(margin), iterations
        tensor = lambda a, dtype=torch.float32: torch.as_tensor(np.array(a, copy=True), device=self.device, dtype=dtype)
        self.rows = tensor(np.flatnonzero(~bindings.frozen_rows), torch.long)
        edge_ids = bindings.active_edges
        node_ids = np.unique(asset.edges[edge_ids])
        lookup = np.full(len(asset.nodes), -1, np.int64)
        lookup[node_ids] = np.arange(len(node_ids))
        self.node_ids = tensor(node_ids, torch.long)
        self.parent = tensor(lookup[asset.edge_parent[edge_ids]], torch.long)
        self.child = tensor(lookup[asset.edge_child[edge_ids]], torch.long)
        self.edge_rows = tensor(asset.strand_rows[edge_ids], torch.long)
        self.free = tensor(~np.isin(node_ids, asset.strand_base_node), torch.bool)
        self.layers = [tensor(np.flatnonzero(asset.edge_depth[edge_ids] == depth), torch.long)
                       for depth in np.unique(asset.edge_depth[edge_ids])]
        with torch.inference_mode():
            self.node_clearance = torch.minimum(self.field.sample(tensor(asset.nodes[node_ids]))[0],
                                                tensor(np.full(len(node_ids), margin)))
            self.gs_clearance = torch.minimum(self.field.sample(tensor(asset.xyz)[self.rows])[0],
                                              tensor(np.full(len(self.rows), margin)))
        self.last_stats = {}

    def resolve(self, pose, body_vertices):
        started = time.perf_counter()
        self.field.update(body_vertices)
        xyz, rotation, nodes = (pose[key].clone() for key in ("xyz", "rotation", "nodes"))
        original = nodes[self.node_ids].clone()
        points = original.clone()
        old_vector = original[self.child]-original[self.parent]
        lengths = torch.linalg.vector_norm(old_vector, dim=-1)
        before_nodes = self.field.sample(points)[0]
        before_gs = self.field.sample(xyz[self.rows])[0]
        for _ in range(self.iterations):
            for layer in self.layers:
                parent, child = points[self.parent[layer]], points[self.child[layer]]
                length = lengths[layer, None]
                vector = child-parent
                direction = F.normalize(vector, dim=-1, eps=1e-10)
                direction = torch.where((vector.square().sum(-1) > 1e-16)[:, None], direction,
                                        F.normalize(old_vector[layer], dim=-1))
                child = parent+direction*length
                distance, normal = self.field.sample(child)
                penetration = (self.node_clearance[self.child[layer]]-distance).clamp_min(0.)[:, None]
                # Intersect the edge-length sphere with the local SDF contact
                # plane. Unreachable contact stays reported, never stretches
                # a short strand whose pinned root lies inside the collider.
                height = ((child-parent)*normal).sum(-1, keepdim=True)+penetration
                height = torch.maximum(-length, torch.minimum(length, height))
                tangent = direction-normal*(direction*normal).sum(-1, keepdim=True)
                helper = F.one_hot(normal.abs().argmin(-1), 3).to(normal.dtype)
                tangent = torch.where((tangent.square().sum(-1) > 1e-12)[:, None], tangent,
                                      torch.linalg.cross(normal, helper))
                contact = parent+normal*height+F.normalize(tangent, dim=-1)*(length.square()-height.square()).clamp_min(0.).sqrt()
                points[self.child[layer]] = torch.where(penetration > 1e-8, contact, child)
        nodes[self.node_ids] = points
        turn = _turn_between(old_vector, points[self.child]-points[self.parent])
        old_mid = .5*(original[self.parent]+original[self.child])
        new_mid = .5*(points[self.parent]+points[self.child])
        xyz[self.edge_rows] = new_mid + _quat_apply(turn, xyz[self.edge_rows]-old_mid)
        rotation[self.edge_rows] = F.normalize(_quat_multiply(turn, rotation[self.edge_rows]), dim=-1)
        # Center proxies cover shell splats and residual offsets of strand splats.
        # Gaussian ellipsoid volumes are intentionally not full contact shapes.
        xyz[self.rows] = self.field.project(xyz[self.rows], self.gs_clearance, steps=12)
        after_nodes = self.field.sample(points)[0]
        after_gs = self.field.sample(xyz[self.rows])[0]
        tolerance = self.field.pitch*.02
        violation = lambda d, c: (c-d).clamp_min(0.)
        node_before = violation(before_nodes[self.free], self.node_clearance[self.free])
        node_after = violation(after_nodes[self.free], self.node_clearance[self.free])
        gs_before, gs_after = violation(before_gs, self.gs_clearance), violation(after_gs, self.gs_clearance)
        edge_error = (torch.linalg.vector_norm(points[self.child]-points[self.parent], dim=-1)-lengths).abs()
        maximum = lambda a: float(a.max().item()) if a.numel() else 0.
        self.last_stats = dict(enabled=True, sdf_build_ms=self.field.build_ms,
            sdf_pitch=self.field.pitch, sdf_shape=self.field.shape, resolution=self.field.resolution,
            margin=self.margin, iterations=self.iterations, active_nodes=len(points), active_gaussians=len(self.rows),
            tolerance=tolerance, node_contacts_before=int((node_before > tolerance).sum().item()),
            node_contacts_after=int((node_after > tolerance).sum().item()),
            gs_contacts_before=int((gs_before > tolerance).sum().item()),
            gs_contacts_after=int((gs_after > tolerance).sum().item()),
            max_node_penetration_before=maximum(node_before), max_node_penetration_after=maximum(node_after),
            max_gs_penetration_before=maximum(gs_before), max_gs_penetration_after=maximum(gs_after),
            absolute_inside_gs_before=int((before_gs < -tolerance).sum().item()),
            absolute_inside_gs_after=int((after_gs < -tolerance).sum().item()),
            pinned_roots_inside_body=int((self.field.sample(original[~self.free])[0] < -tolerance).sum().item()),
            max_edge_length_error=maximum(edge_error),
            mean_edge_length_error=float(edge_error.mean().item()) if edge_error.numel() else 0.,
            pinned_root_max_error=maximum((points[~self.free]-original[~self.free]).abs()),
            max_gs_correction=maximum(torch.linalg.vector_norm(xyz[self.rows]-pose["xyz"][self.rows], dim=-1)))
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.last_stats["collision_ms"] = (time.perf_counter()-started)*1000
        return dict(xyz=xyz, rotation=rotation, nodes=nodes)
