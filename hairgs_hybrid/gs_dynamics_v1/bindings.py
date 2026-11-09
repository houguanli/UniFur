"""Calibrated lion coverage and sparse body-surface animation controls."""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import numpy as np

from .config import DynamicsConfig
from .math3d import smoothstep


@dataclass
class Bindings:
    guide_weight: np.ndarray
    controller_guide_ids: np.ndarray
    control_ids: np.ndarray
    control_weights: np.ndarray
    guide_frames: np.ndarray
    shell_rows: np.ndarray
    shell_s: np.ndarray
    frozen_rows: np.ndarray
    active_edges: np.ndarray
    active_shell_rows: np.ndarray
    diagnostics: dict
    # Per guide: share of the skin-anchor-to-tip distance taken by a floating
    # strand base (skin-rooted profiles only); None/0 keeps bases fixed.
    strand_root_s: np.ndarray | None = None


def protected_gaussians(xyz, profile):
    """Keep facial features/paws fixed while admitting lateral cheek fur.

    These are asset-coordinate artistic masks, not inferred hair semantics.
    A strand touching this region is protected as a whole to preserve its FK.
    """
    if profile == "conservative":
        return (xyz[:, 1] < -.76) | (xyz[:, 2] < .10)
    if profile in ("expanded", "full"):
        return (((xyz[:, 1] < -.76) & (np.abs(xyz[:, 0]) < .105))
                | (xyz[:, 1] < -.87) | (xyz[:, 2] < .10)
                | ((xyz[:, 1] < .30) & (xyz[:, 2] < .43))
                | ((xyz[:, 1] >= 0) & (xyz[:, 1] < .80)))
    raise ValueError("unknown motion profile")


def _surface_weights(asset, controls, active, neighbors):
    """Interpolate graph geodesic distances through each anchor triangle."""
    v, f = asset.body_vertices, asset.body_faces
    edges = np.unique(np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1), axis=0)
    graph = [[] for _ in v]
    for x, y in edges:
        length = float(np.linalg.norm(v[x]-v[y]))
        graph[x].append((int(y), length))
        graph[y].append((int(x), length))
    distances = []
    for g in controls:
        seeds = f[asset.anchor_face[g]]
        dist = np.full(len(v), np.inf)
        queue = []
        for seed in seeds:
            dist[seed] = np.linalg.norm(v[seed]-asset.guide_anchors[g])
            heapq.heappush(queue, (dist[seed], int(seed)))
        while queue:
            d, vertex = heapq.heappop(queue)
            if d > dist[vertex]:
                continue
            for nxt, length in graph[vertex]:
                nd = d + length
                if nd < dist[nxt]:
                    dist[nxt] = nd
                    heapq.heappush(queue, (nd, nxt))
        distances.append(dist)
    distances = np.asarray(distances, dtype=np.float64).T
    k = min(neighbors, len(controls))
    ids = np.zeros((asset.guide_count, k), np.int32)
    weights = np.zeros((asset.guide_count, k), np.float64)
    for start in range(0, len(active), 4096):
        group = active[start:start+4096]
        d = np.sum(distances[f[asset.anchor_face[group]]] * asset.anchor_bary[group, :, None], axis=1)
        choice = np.argsort(d, axis=1, kind="stable")[:, :k]
        local = np.take_along_axis(d, choice, axis=1)
        if np.any(~np.isfinite(local[:, 0])):
            raise ValueError("active guide lies on a mesh component without a controller")
        w = np.where(np.isfinite(local), 1 / np.maximum(local, 1e-6)**2, 0)
        w /= w.sum(1, keepdims=True)
        ids[group], weights[group] = choice, w
    return ids, weights


def _semantic_weights(asset, semantics):
    """Per-guide region weight and per-Gaussian facial guard from rig skinning.

    ``semantics`` holds per-Gaussian [N] weights in [0,1]: ``mane`` and ``tail``
    (movable fur) and ``face`` (eyes, lids, nose, lips, tongue, upper cheeks).
    """
    values = {}
    for key in ("mane", "tail", "face"):
        value = np.asarray(semantics[key], dtype=np.float64)
        if value.shape != (len(asset.xyz),) or not np.isfinite(value).all() or value.min() < 0 or value.max() > 1:
            raise ValueError(f"semantic {key} weights must be finite [N] values in [0,1]")
        values[key] = value
    movable = smoothstep(.2, .6, np.minimum(values["mane"]+values["tail"], 1.))
    total, count = np.zeros(asset.guide_count), np.zeros(asset.guide_count)
    np.add.at(total, asset.gs_guide, movable)
    np.add.at(count, asset.gs_guide, 1.)
    face = np.zeros(asset.guide_count)
    np.maximum.at(face, asset.gs_guide, values["face"])
    region = total/np.maximum(count, 1.)*(1-smoothstep(.15, .45, face))
    return region, values["face"] > .45


def build_bindings(asset, config=None, *, guide_weights=None, semantics=None):
    config = (config or DynamicsConfig()).validate()
    semantic = semantics is not None and guide_weights is None and config.motion_profile == "full"
    G, S, a = asset.guide_count, asset.shell_count, asset.guide_anchors
    shell_rows = np.flatnonzero(asset.route == 0)
    outward = np.einsum("ij,ij->i", asset.xyz-a[asset.gs_guide], asset.face_normals[asset.anchor_face[asset.gs_guide]])
    explicit = guide_weights is not None
    if explicit:
        weight = np.asarray(guide_weights, dtype=np.float64).copy()
        if weight.shape != (G,) or not np.isfinite(weight).all() or np.any((weight < 0) | (weight > 1)):
            raise ValueError("guide_weights must be finite [G] values in [0,1]")
        safe = np.ones(G, bool)
    else:
        y, z, feather = a[:, 1], a[:, 2], config.region_feather
        mane = (smoothstep(config.mane_y_min, config.mane_y_min+2*feather, y)
                * (1-smoothstep(config.mane_y_max-3.5*feather, config.mane_y_max, y))
                * smoothstep(config.mane_z_min, config.mane_z_min+3.75*feather, z)
                * np.maximum(smoothstep(config.face_y_cut-2*feather, config.face_y_cut, y),
                             smoothstep(config.crest_z_min, config.crest_z_min+1.75*feather, z)))
        tail = (smoothstep(config.tail_y_min, config.tail_y_min+2.25*feather, y)
                * smoothstep(config.tail_z_min, config.tail_z_min+feather, z)
                * (1-smoothstep(config.tail_z_max-1.75*feather, config.tail_z_max, z))) if config.enable_tail else np.zeros(G)
        extent, sigma, minimum = np.zeros(G), np.zeros(G), np.full(G, np.inf)
        np.maximum.at(extent, asset.node_guide, np.linalg.norm(asset.nodes-a[asset.node_guide], axis=1))
        np.maximum.at(extent, asset.gs_guide, np.linalg.norm(asset.xyz-a[asset.gs_guide], axis=1))
        np.maximum.at(sigma, asset.gs_guide, np.exp(asset.scale.astype(np.float64)).max(1))
        np.minimum.at(minimum, asset.gs_guide, outward)
        base_distance = np.zeros(G)
        base = asset.base_node_by_guide[S:]
        base_distance[S:] = np.linalg.norm(asset.nodes[base]-a[S:], axis=1)
        safe = ((extent < config.max_node_extent) & (sigma < config.max_gaussian_sigma)
                & (asset.guide_lengths < config.max_chain_length) & (base_distance < config.max_base_offset))
        if semantic:
            region, protected = _semantic_weights(asset, semantics)
        else:
            region = np.maximum(mane, tail)
            protected = protected_gaussians(asset.xyz, config.motion_profile)
        weight = region*safe
        weight[S:] *= smoothstep(config.min_strand_outward_distance-.005,
                                  config.min_strand_outward_distance+.015, minimum[S:])
        # Freeze entire strand chains touching the face/paws: never break shared-node FK.
        protected_guides = np.zeros(G, bool)
        np.logical_or.at(protected_guides, asset.gs_guide[asset.route == 1], protected[asset.route == 1])
        weight[protected_guides] = 0
    # Remove numerically negligible masks, leaving an explicit frozen set.
    weight[weight < 1e-4] = 0
    active = np.flatnonzero(weight > 0)
    if not len(active):
        raise ValueError("no active guides: adjust masks or supply guide_weights")
    k = min(config.controller_count, len(active))
    selected = []
    distance = np.full(len(active), np.inf)
    pick = int(np.argmax(weight[active]))
    for _ in range(k):
        g = active[pick]
        selected.append(g)
        distance = np.minimum(distance, np.sum((a[active]-a[g])**2, axis=1))
        score = distance * np.sqrt(weight[active])
        score[np.isin(active, selected)] = -1
        pick = int(np.argmax(score))
    controls = np.asarray(selected, np.int64)
    ids, control_weights = _surface_weights(asset, controls, active, config.neighbors)
    n = asset.guide_normals
    helper = np.eye(3)[np.abs(n).argmin(1)]
    t = np.cross(n, helper)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-15)
    frames = np.stack([t, np.cross(n, t), n], axis=2)
    s = smoothstep(config.shell_pin_distance, config.shell_full_distance, outward[shell_rows])
    if not explicit:
        s[protected[shell_rows]] = 0
    frozen = weight[asset.gs_guide] == 0
    frozen[shell_rows] |= s == 0
    active_edges = np.flatnonzero(weight[asset.edge_guide] > 0)
    root_s = np.zeros(G)
    if config.skin_rooted_strands:
        offset = np.linalg.norm(asset.nodes[asset.base_node_by_guide[S:]]-a[S:], axis=1)
        root_s[S:] = offset/np.maximum(offset+asset.guide_lengths[S:], 1e-12)
    active_shell = shell_rows[~frozen[shell_rows]]
    diagnostics = {"controllers": len(controls), "active_guides": len(active),
                   "active_gaussians": int((~frozen).sum()), "active_shell_gaussians": len(active_shell),
                   "active_strand_gaussians": len(active_edges), "frozen_gaussians": int(frozen.sum()),
                   "guard_rejected_guides": int((~safe).sum()),
                   "motion_profile": "explicit" if explicit else config.motion_profile,
                   "semantic_mask": "explicit" if explicit else ("full_lion_rig_skinning_v1" if semantic
                                                                 else f"{config.motion_profile}_lion_mane_tail_v2"),
                   "surface_weights": "body-edge geodesic, barycentrically interpolated inverse-square distances",
                   "collision_solver": "not implemented in stage 1",
                   "skin_rooted_strands": bool(config.skin_rooted_strands)}
    return Bindings(weight, controls, ids, control_weights, frames, shell_rows, s, frozen,
                    active_edges, active_shell, diagnostics, root_s)
