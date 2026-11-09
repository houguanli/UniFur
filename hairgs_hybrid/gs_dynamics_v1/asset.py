"""Validated immutable complete asset and rooted, array-order-independent graph."""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import MappingProxyType
import numpy as np


class CompleteAsset:
    REQUIRED = {"xyz", "scale", "rotation", "dc", "rest", "opacity", "body_vertices", "body_faces",
                "route", "anchor_face", "anchor_bary", "gs_guide", "guide_route", "nodes", "edges",
                "node_guide", "strand_rows", "strand_axis", "strand_base_node", "shell_guide_count"}

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1024*1024), b""):
                digest.update(block)
        with np.load(path, allow_pickle=False) as z:
            a = {k: z[k] for k in z.files}
        obj = cls.from_arrays(a)
        obj.source_path, obj.source_sha256 = path, digest.hexdigest()
        return obj

    @classmethod
    def from_arrays(cls, arrays):
        obj = cls()
        missing = cls.REQUIRED - set(arrays)
        if missing:
            raise ValueError(f"missing complete asset fields: {sorted(missing)}")
        a = {k: np.array(v, copy=True) for k, v in arrays.items()}
        if not all(x.dtype.kind in "fiu" and np.isfinite(x).all() for x in a.values()):
            raise ValueError("all asset arrays must be finite numeric arrays")
        for x in a.values():
            x.setflags(write=False)
        obj.arrays = MappingProxyType(a)
        obj.source_path = None
        obj.source_sha256 = None
        obj._validate_and_cache()
        return obj

    def __getattr__(self, name):
        a = self.__dict__.get("arrays", {})
        if name in a:
            return a[name]
        raise AttributeError(name)

    def _validate_and_cache(self):
        N, G = len(self.xyz), len(self.guide_route)
        S = int(self.shell_guide_count)
        self.guide_count, self.shell_count = G, S
        shapes = {"xyz": (N, 3), "scale": (N, 3), "rotation": (N, 4), "dc": (N, 1, 3),
                  "opacity": (N, 1), "route": (N,), "gs_guide": (N,),
                  "anchor_face": (G,), "anchor_bary": (G, 3), "node_guide": (len(self.nodes),),
                  "strand_rows": (len(self.edges),), "strand_axis": (len(self.edges),),
                  "strand_base_node": (G-S,)}
        if not 0 <= S <= G or N < 1 or G < 1:
            raise ValueError("invalid guide or Gaussian count")
        if any(getattr(self, k).shape != shape for k, shape in shapes.items()):
            raise ValueError("complete asset shape mismatch")
        if self.rest.ndim != 3 or self.rest.shape[0] != N or self.rest.shape[2] != 3:
            raise ValueError("invalid SH rest shape")
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 3 or self.edges.ndim != 2 or self.edges.shape[1] != 2:
            raise ValueError("nodes/edges shape mismatch")
        if self.body_vertices.ndim != 2 or self.body_vertices.shape[1] != 3 or self.body_faces.ndim != 2 or self.body_faces.shape[1] != 3:
            raise ValueError("body mesh shape mismatch")
        indices = [("body_faces", len(self.body_vertices)), ("anchor_face", len(self.body_faces)),
                   ("gs_guide", G), ("node_guide", G), ("edges", len(self.nodes)),
                   ("strand_rows", N), ("strand_axis", 3), ("strand_base_node", len(self.nodes))]
        for key, count in indices:
            x = getattr(self, key)
            if x.dtype.kind not in "iu" or (x.size and (x.min() < 0 or x.max() >= count)):
                raise ValueError(f"invalid indices in {key}")
        if np.any(self.anchor_bary < -1e-8) or np.max(np.abs(self.anchor_bary.sum(1)-1)) > 1e-6:
            raise ValueError("invalid anchor barycentric coordinates")
        if not np.all(self.guide_route[:S] == 0) or not np.all(self.guide_route[S:] == 1):
            raise ValueError("expected Shell prefix followed by Strand guides (0/1)")
        if not np.array_equal(self.route, self.guide_route[self.gs_guide]):
            raise ValueError("Gaussian and guide route mismatch")
        if not np.array_equal(np.sort(self.strand_rows), np.flatnonzero(self.route == 1)):
            raise ValueError("strand rows must match route=1 exactly")
        if np.any(np.linalg.norm(self.rotation, axis=1) < 1e-10):
            raise ValueError("zero Gaussian quaternion")
        tri = self.body_vertices[self.body_faces].astype(np.float64)
        fn = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
        if np.any(np.linalg.norm(fn, axis=1) < 1e-14):
            raise ValueError("degenerate body triangle")
        if np.einsum("ij,ij->", tri[:, 0], fn) < 0:
            fn = -fn
        self.face_normals = fn / np.linalg.norm(fn, axis=1, keepdims=True)
        vn = np.zeros_like(self.body_vertices, dtype=np.float64)
        for j in range(3):
            np.add.at(vn, self.body_faces[:, j], fn)
        vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-15)
        self.guide_anchors = np.sum(tri[self.anchor_face] * self.anchor_bary[:, :, None], axis=1)
        self.guide_normals = np.sum(vn[self.body_faces[self.anchor_face]] * self.anchor_bary[:, :, None], axis=1)
        self.guide_normals /= np.maximum(np.linalg.norm(self.guide_normals, axis=1, keepdims=True), 1e-15)
        self._root_chains()

    def _root_chains(self):
        n, e, G, S = len(self.nodes), self.edges, self.guide_count, self.shell_count
        degree = np.bincount(e.ravel(), minlength=n)
        if np.any(degree > 2) or np.any(degree < 1):
            raise ValueError("strand graph must contain open unbranched chains")
        if not np.array_equal(self.node_guide[e[:, 0]], self.node_guide[e[:, 1]]) or not np.array_equal(self.node_guide[e[:, 0]], self.gs_guide[self.strand_rows]):
            raise ValueError("edge/row guide mismatch")
        base_guides = self.node_guide[self.strand_base_node]
        if not np.array_equal(np.sort(base_guides), np.arange(S, G)) or np.any(degree[self.strand_base_node] != 1):
            raise ValueError("invalid strand base nodes")
        adjacent = np.full((n, 2), -1, np.int64)
        edge_ids = np.full((n, 2), -1, np.int64)
        used = np.zeros(n, np.int8)
        for k, (a, b) in enumerate(e):
            adjacent[a, used[a]], edge_ids[a, used[a]] = b, k
            used[a] += 1
            adjacent[b, used[b]], edge_ids[b, used[b]] = a, k
            used[b] += 1
        seen = np.zeros(n, bool)
        parent, child = np.empty(len(e), np.int64), np.empty(len(e), np.int64)
        depth, arc = np.empty(len(e), np.int32), np.zeros(n, np.float64)
        lengths = np.zeros(G, np.float64)
        for base in self.strand_base_node:
            prev, node, level = -1, int(base), 0
            while True:
                if seen[node]:
                    raise ValueError("strand cycle encountered")
                seen[node] = True
                slot = 0 if adjacent[node, 0] != prev else 1
                nxt = int(adjacent[node, slot])
                if nxt < 0:
                    break
                k = edge_ids[node, slot]
                parent[k], child[k], depth[k] = node, nxt, level
                arc[nxt] = arc[node] + np.linalg.norm(self.nodes[nxt]-self.nodes[node])
                prev, node, level = node, nxt, level+1
            lengths[self.node_guide[base]] = arc[node]
        if not seen.all() or np.any(lengths[S:] <= 1e-12):
            raise ValueError("disconnected graph, cycle or collapsed guide")
        self.edge_parent, self.edge_child, self.edge_depth = parent, child, depth
        self.edge_rest_vector = self.nodes[child].astype(np.float64)-self.nodes[parent]
        if np.any(np.linalg.norm(self.edge_rest_vector, axis=1) <= 1e-12):
            raise ValueError("collapsed strand edge")
        self.node_s = arc / lengths[self.node_guide]
        self.edge_s = (self.node_s[parent]+self.node_s[child])*.5
        self.guide_lengths = lengths
        self.edge_guide = self.node_guide[parent]
        self.base_node_by_guide = np.full(G, -1, np.int64)
        self.base_node_by_guide[base_guides] = self.strand_base_node
        self.strand_center_residual = self.xyz[self.strand_rows].astype(np.float64) - (self.nodes[e[:, 0]]+self.nodes[e[:, 1]])*.5
