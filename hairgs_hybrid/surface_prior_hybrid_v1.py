"""A050 Hybrid ablation with an explicit attachment-surface prior.

This is intentionally a thin adapter over the frozen Hybrid model/trainer.  It
changes only root initialization and makes Shell layers grow outwards from the
attachment surface.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from plyfile import PlyData
from pytorch3d.transforms import quaternion_to_matrix

from hairgs_hybrid.rgb_only_init_v1.core import pose, require, sha
from hairgs_hybrid.teacher_free_shell_strand_v1.inputs import (
    TrainView, load_train_views, project_numpy,
)
from hairgs_hybrid.teacher_free_shell_strand_v1.model import ShellStrand
from hairgs_hybrid.teacher_free_shell_strand_v1.training import (
    Schedule, Trainer, camera_for, native_renderer,
)

SOURCE_LOCK = Path(__file__).parent / "rgb_only_init_v1/source_tail_v3.json"
SOURCE_LOCK_SHA = "886295cc47fa542b6f5a8869c8a59647ae686abd4b4369ee8ebe2b4cb6108877"
ROOT_COUNT = 19_215
LAYERS = 3
CAP = 239_171
SPACING = 2.929127857326941
SCHEDULE = Schedule(stage1024=1000, stage2048=3000, total_steps=8000,
                    max_shell_splits=0)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if path.suffix.lower() == ".ply":
        mesh = PlyData.read(path)
        vertex = mesh["vertex"].data
        vertices = np.column_stack([vertex[name] for name in "xyz"]).astype(np.float64)
        faces = np.asarray([np.asarray(row, dtype=np.int64)
                            for row in mesh["face"].data["vertex_indices"]])
    elif path.suffix.lower() == ".obj":
        vertices, faces = [], []
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            fields = line.split()
            if fields[:1] == ["v"]:
                vertices.append([float(x) for x in fields[1:4]])
            elif fields[:1] == ["f"]:
                polygon = [int(x.split("/")[0]) - 1 for x in fields[1:]]
                faces.extend([[polygon[0], polygon[i], polygon[i + 1]]
                              for i in range(1, len(polygon) - 1)])
        vertices, faces = np.asarray(vertices, np.float64), np.asarray(faces, np.int64)
    else:
        raise ValueError("surface prior must be triangular PLY or OBJ")
    require(vertices.ndim == 2 and vertices.shape[1] == 3 and faces.ndim == 2 and
            faces.shape[1] == 3 and len(vertices) and len(faces), "invalid triangular surface")
    require(np.isfinite(vertices).all() and faces.min() >= 0 and faces.max() < len(vertices),
            "invalid surface values/indices")
    return vertices, faces


def sample_attachment(vertices: np.ndarray, faces: np.ndarray, views: list[TrainView],
                      count: int, seed: int = 0, mask_threshold: float = .05,
                      min_support: int = 44, front_facing_only: bool = False,
                      min_visible_views: int = 1, support_ratio: float | None = None) -> dict:
    """Uniform area samples retained by >=44/48 train hair masks."""
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    outward_fraction = float((((tri.mean(1) - vertices.mean(0)) * cross).sum(1) >= 0).mean())
    globally_flipped = outward_fraction < .5
    if globally_flipped:
        cross = -cross
    double_area = np.linalg.norm(cross, axis=1)
    valid_faces = double_area > 1e-10
    require(valid_faces.any(), "zero-area surface")
    tri, cross, double_area = tri[valid_faces], cross[valid_faces], double_area[valid_faces]
    rng = np.random.default_rng(seed)
    candidate_count = count * 24
    chosen = rng.choice(len(tri), candidate_count, p=double_area / double_area.sum())
    u, v = rng.random(candidate_count), rng.random(candidate_count)
    root_u = np.sqrt(u)
    bary = np.column_stack((1 - root_u, root_u * (1 - v), root_u * v))
    points = (tri[chosen] * bary[:, :, None]).sum(1)
    normals = cross[chosen] / double_area[chosen, None]

    support = np.zeros(candidate_count, np.uint8)
    visible_count = np.zeros(candidate_count, np.uint8)
    masks = []
    for view in views:
        _, alpha, k = view.at(512)
        mask = alpha[0].numpy() > mask_threshold
        pixel, depth = project_numpy(points, view, k)
        inside = np.isfinite(pixel).all(1) & (depth > 0) & (pixel >= 0).all(1) & (pixel < 512).all(1)
        if front_facing_only:
            camera_center = -view.rotation.T @ view.translation
            to_camera = F.normalize(torch.from_numpy(camera_center - points), dim=1).numpy()
            inside &= (normals * to_camera).sum(1) > .05
        xy = np.floor(np.clip(np.nan_to_num(pixel), 0, 511)).astype(np.int64)
        support += inside & mask[xy[:, 1], xy[:, 0]]
        visible_count += inside
        masks.append((view, alpha[0].numpy(), k))
    eligible = np.flatnonzero(
        (visible_count >= min_visible_views) &
        ((support >= min_support) if support_ratio is None else
         (support >= np.ceil(visible_count * support_ratio))))
    require(len(eligible) >= count, f"surface/hair-mask intersection has only {len(eligible)} samples")
    selected = rng.choice(eligible, count, replace=False)
    points, normals, support = points[selected], normals[selected], support[selected]

    colors = np.zeros((count, 3), np.float64)
    weights = np.zeros(count, np.float64)
    fallback_colors = np.zeros_like(colors)
    fallback_weights = np.zeros(count, np.float64)
    for view, alpha, k in masks:
        rgb, _, _ = view.at(512)
        pixel, depth = project_numpy(points, view, k)
        inside = np.isfinite(pixel).all(1) & (depth > 0) & (pixel >= 0).all(1) & (pixel < 512).all(1)
        xy = np.floor(np.clip(np.nan_to_num(pixel), 0, 511)).astype(np.int64)
        a = alpha[xy[:, 1], xy[:, 0]] * inside
        camera_center = -view.rotation.T @ view.translation
        to_camera = F.normalize(torch.from_numpy(camera_center - points), dim=1).numpy()
        facing = np.maximum((normals * to_camera).sum(1), 0)
        sampled = rgb[:, xy[:, 1], xy[:, 0]].T.numpy()
        colors += (a * facing)[:, None] * sampled
        weights += a * facing
        fallback_colors += a[:, None] * sampled
        fallback_weights += a
    fallback = weights <= 1e-8
    require((fallback_weights[ fallback] > 0).all(), "surface roots without train RGB support")
    colors[~fallback] /= weights[~fallback, None]
    colors[fallback] = fallback_colors[fallback] / fallback_weights[fallback, None]
    return {"roots": points.astype(np.float32), "normals": normals.astype(np.float32),
            "colors": colors.astype(np.float32), "support": support,
            "surface_area": float(double_area.sum() * .5),
            "mesh_winding_global_flip": globally_flipped,
            "preflip_centroid_outward_fraction": outward_fraction,
            "eligible_candidates": int(len(eligible)), "candidate_count": candidate_count,
            "mask_threshold": float(mask_threshold), "min_support_views": int(min_support),
            "front_facing_only": bool(front_facing_only),
            "min_visible_views": int(min_visible_views),
            "support_ratio": None if support_ratio is None else float(support_ratio)}


class SurfacePriorHybrid(ShellStrand):
    """`shell_root` is the skin attachment; layers increase along +normal."""

    def shell_geometry(self):
        rotation = quaternion_to_matrix(F.normalize(self.shell_rotation, dim=1))
        gaps = F.softplus(self.shell_spacing) * self.spacing
        height = torch.cat((gaps.new_zeros((len(gaps), 1)), gaps.cumsum(1)), dim=1)
        xyz = self.shell_root[:, None, :] + height[:, :, None] * rotation[:, None, :, 2]
        tangent = self.shell_scale[..., :2].exp()
        thickness = tangent.min(-1, keepdim=True).values * .25 * torch.sigmoid(self.shell_scale[..., 2:3])
        return xyz, torch.cat((tangent, thickness), dim=-1), rotation

    def setup_optimizer(self, geometry_lr=1e-3, appearance_lr=2.5e-3):
        optimizer = super().setup_optimizer(geometry_lr, appearance_lr)
        for group in optimizer.param_groups:
            if group["name"] in ("shell_root", "shell_rotation"):
                group["base_lr"] = group["lr"] = 0.
        return optimizer

    def curve_geometry(self):
        """Project every Strand segment into its root's outward normal half-space."""
        if not len(self.strand_control):
            return self.strand_control, self.strand_control
        first = torch.cat((self.curve_counts.new_zeros(1), self.curve_counts.cumsum(0)[:-1]))
        previous = torch.roll(self.strand_control, 1, 0)
        previous = torch.index_fill(previous, 0, first, 0.)
        delta = self.strand_control - previous
        root_ids = torch.repeat_interleave(self.curve_roots, self.curve_counts)
        rotation = quaternion_to_matrix(F.normalize(self.shell_rotation, dim=1))
        normal = rotation[root_ids, :, 2]
        normal_step = torch.sum(delta * normal, dim=1, keepdim=True)
        minimum_step = self.spacing * 1e-4
        delta = delta + (minimum_step - normal_step).clamp_min(0.) * normal

        curve_id = torch.repeat_interleave(torch.arange(len(self.curve_counts), device=delta.device),
                                           self.curve_counts)
        local_id = torch.arange(len(delta), device=delta.device) - torch.repeat_interleave(first, self.curve_counts)
        padded = delta.new_zeros((len(self.curve_counts), int(self.curve_counts.max()), 3))
        padded[curve_id, local_id] = delta
        control = padded.cumsum(1)[curve_id, local_id]
        roots = self.shell_root[root_ids]
        end = roots + control
        start = roots + torch.roll(control, 1, 0)
        start = torch.index_copy(start, 0, first, roots[first])
        return start, end


def initialize(surface: Path, scale: float, translation: np.ndarray,
               surface_regions: Path | None = None) -> tuple[SurfacePriorHybrid, dict, list[TrainView]]:
    views, source = load_train_views(SOURCE_LOCK, SOURCE_LOCK_SHA)
    vertices, faces = load_mesh(surface)
    vertices = vertices * scale + translation[None]
    region_report = None
    if surface_regions is not None:
        with np.load(surface_regions) as data:
            labels = np.asarray(data["surface_triangle_region_id"])
        require(labels.shape == (len(faces),), "surface-region/mesh face mismatch")
        keep = labels >= 0
        require(keep.any(), "surface regions contain no certified fur-bearing faces")
        region_report = {"path": str(surface_regions.resolve()), "sha256": sha(surface_regions),
                         "rule": "surface_triangle_region_id >= 0 (native GT root binding)",
                         "kept_faces": int(keep.sum()), "total_faces": int(len(faces))}
        faces = faces[keep]
    samples = sample_attachment(vertices, faces, views, ROOT_COUNT)
    model = SurfacePriorHybrid(samples["roots"], samples["normals"], samples["colors"],
                               SPACING, layers=LAYERS, cap=CAP)
    report = {"surface": str(surface.resolve()), "surface_sha256": sha(surface),
              "vertices": len(vertices), "faces": len(faces), "scale": scale,
              "translation": translation.tolist(), "aabb": [vertices.min(0).tolist(), vertices.max(0).tolist()],
              "root_count": ROOT_COUNT, "support_views_min": int(samples["support"].min()),
              "support_views_median": float(np.median(samples["support"])),
              "candidate_count": samples["candidate_count"],
              "eligible_candidates": samples["eligible_candidates"],
              "mask_threshold": samples["mask_threshold"],
              "min_support_views": samples["min_support_views"],
              "front_facing_only": samples["front_facing_only"],
              "min_visible_views": samples["min_visible_views"],
              "support_ratio": samples["support_ratio"],
              "surface_area": samples["surface_area"], "source": source,
              "mesh_winding_global_flip": samples["mesh_winding_global_flip"],
              "preflip_centroid_outward_fraction": samples["preflip_centroid_outward_fraction"],
              "surface_region_filter": region_report,
              "shell_semantics": "fixed attachment root/normal plus outward learnable layers"}
    return model, report, views


def train(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    require(not output.exists(), "exclusive output required")
    output.mkdir(parents=True)
    torch.manual_seed(0); np.random.seed(0)
    started = time.perf_counter()
    model, init, views = initialize(args.surface.resolve(), args.scale, np.asarray(args.translation, float),
                                    args.surface_regions.resolve() if args.surface_regions else None)
    init["source"]["geometry_gt_read"] = bool(args.oracle)
    init["source"]["geometry_prior_read"] = True
    with (output / "initial_surface_roots.npz").open("xb") as stream:
        rotation = quaternion_to_matrix(F.normalize(model.shell_rotation, dim=1)).detach().numpy()
        np.savez_compressed(stream, roots=model.shell_root.detach().numpy(), normals=rotation[:, :, 2])
    require(torch.cuda.is_available() and torch.cuda.mem_get_info()[0] >= 8 * 2**30, "CUDA/headroom unavailable")
    model = model.cuda(); model.setup_optimizer()
    trainer = Trainer(model, views, SCHEDULE)
    torch.cuda.reset_peak_memory_stats()
    with (output / "steps.jsonl").open("x", encoding="utf-8") as log:
        while trainer.step_index < SCHEDULE.total_steps:
            row = trainer.step()
            log.write(json.dumps(row, allow_nan=False) + "\n")
            if trainer.step_index % 48 == 0:
                log.flush()
    with (output / "checkpoint.pt").open("xb") as stream:
        torch.save(trainer.checkpoint(), stream)
    report = {"schema": "surface-prior-hybrid-v1", "status": "complete", "prior_id": args.prior_id,
              "oracle": bool(args.oracle), "initialization": init, "schedule": asdict(SCHEDULE),
              "steps": trainer.step_index, "shell": len(model.shell_root) * model.layers,
              "strand": len(model.strand_control), "primitives": model.primitive_count,
              "wall_seconds": time.perf_counter() - started,
              "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
              "checkpoint_sha256": sha(output / "checkpoint.pt"), "loss_last": trainer.history[-1]}
    write_json(output / "report.json", report)
    print(json.dumps(report, indent=2, allow_nan=False))


def load_checkpoint(path: Path) -> SurfacePriorHybrid:
    checkpoint = torch.load(path, map_location="cpu")
    state, topology = checkpoint["model"], checkpoint["topology"]
    roots = state["shell_root"]
    model = SurfacePriorHybrid(roots.numpy(), np.tile([0, 0, 1], (len(roots), 1)),
                               np.full((len(roots), 3), .5), topology["spacing"],
                               topology["layers"], topology["cap"])
    for name in model.strand_names:
        setattr(model, name, nn.Parameter(torch.empty_like(state[name])))
    model.curve_roots = state["curve_roots"].clone()
    model.curve_counts = state["curve_counts"].clone()
    model.load_state_dict(state, strict=True)
    model.generation = topology["generation"]
    return model


def split_views(split: str) -> tuple[list[TrainView], Path]:
    lock = json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))
    root = Path(lock["release_root"]) / "shared/cases/A050_horned_cow" / split
    manifest = json.loads((root / "camera_manifest.json").read_text(encoding="utf-8"))
    views = []
    for item in manifest["observations"]:
        rotation, translation, _ = pose(item)
        with Image.open(root / "images" / item["image"]) as image:
            rgba = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255
        view = TrainView(item["image"], item["view_id"], rotation, translation,
                         np.asarray(item["intrinsics"][2:], float), rgba)
        view.view_index = int(item["view_index"])
        views.append(view)
    return views, root / "images"


@torch.no_grad()
def render(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    require(not output.exists(), "exclusive render output required")
    (output / "arrays").mkdir(parents=True)
    model = load_checkpoint(args.checkpoint.resolve()).cuda().eval()
    views, _ = split_views(args.split)
    observations = []
    for view in views:
        batch = model.primitives(2, 1.)
        camera = camera_for(view, 2048, model)
        rgb = native_renderer(camera, batch)["render"]
        alpha = native_renderer(camera, batch, torch.ones_like(batch.get_xyz))["render"][:1]
        path = output / "arrays" / f"{Path(view.name).stem}.npz"
        np.savez_compressed(path, rgb=rgb.permute(1, 2, 0).cpu().numpy().astype(np.float32),
                            mask=alpha[0].cpu().numpy().astype(np.float32))
        observations.append({"image": view.name, "view_id": view.view_id,
                             "view_index": view.view_index, "frame_index": 0,
                             "array": str(path), "array_sha256": sha(path)})
    manifest = {"schema": "external-render-manifest-v1", "status": "complete",
                "method": args.method, "case_id": "A050_horned_cow", "split": args.split,
                "render_size": [2048, 2048], "image_count": len(observations),
                "observations": observations, "checkpoint": str(args.checkpoint.resolve()),
                "checkpoint_sha256": sha(args.checkpoint),
                "alpha_semantics": "native continuous rasterized coverage",
                "rgb_semantics": "native black-background composite"}
    write_json(output / "render_manifest.json", manifest)
    print(json.dumps({"status": "complete", "manifest": str(output / "render_manifest.json")}, indent=2))


@torch.no_grad()
def audit(args: argparse.Namespace) -> None:
    """Eval-only signed-side check against GT surface and certified root normals."""
    import trimesh
    from fur_otc_pipeline.diagnose_a050_surface_intrusion import SurfaceProbe, RootProbe, summarize
    model = load_checkpoint(args.checkpoint.resolve()).eval()
    batch = model.primitives(2, 1.)
    centers = batch.get_xyz.numpy().astype(np.float64)
    shell_count = len(model.shell_root) * model.layers
    route = np.arange(len(centers)) >= shell_count
    starts, ends = model.curve_geometry()
    root_ids = torch.repeat_interleave(model.curve_roots, model.curve_counts)
    rotation = quaternion_to_matrix(F.normalize(model.shell_rotation, dim=1))
    normal_delta = torch.sum((ends - starts) * rotation[root_ids, :, 2], dim=1).numpy()
    mesh = trimesh.load(args.surface.resolve(), process=False)
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    archive = np.load(args.strands.resolve(), mmap_mode="r")
    gt_face = np.asarray(archive["source_face_id"], dtype=np.int64)
    bary = np.asarray(archive["root_barycentric"], dtype=np.float64)
    gt_normals = np.asarray(archive["root_normal"], dtype=np.float64)
    anchors = (vertices[faces[gt_face]] * bary[:, :, None]).sum(1)
    regions = np.asarray(archive["region_id"], dtype=np.int32)
    surface_probe = SurfaceProbe(vertices[faces], np.asarray(mesh.face_normals))
    root_probe = RootProbe(anchors, gt_normals, regions)
    tolerance = .03657022399902344

    def score(points: np.ndarray) -> dict:
        return summarize(surface_probe.query(points), root_probe.query(points),
                         np.ones(len(points), bool), tolerance)

    initial = np.load(args.initial_roots)["roots"].astype(np.float64)
    result = {"schema": "surface-prior-hybrid-intrusion-v1", "status": "complete_eval_only",
              "criterion": "nearest oriented GT triangle AND nearest certified GT root plane < -0.036570224",
              "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha(args.checkpoint),
              "initial_attachment_roots": score(initial), "trained_attachment_roots": score(model.shell_root.numpy()),
              "shell_centers": score(centers[~route]), "strand_centers": score(centers[route]),
              "strand_outward_constraint": {
                  "criterion": "dot(segment_delta, fixed_root_outward_normal) >= 0",
                  "minimum": float(normal_delta.min()) if len(normal_delta) else None,
                  "negative_fraction_below_minus_1e_7": float((normal_delta < -1e-7).mean()) if len(normal_delta) else 0.,
                  "zero_or_tangent_fraction_at_1e_7": float((normal_delta <= 1e-7).mean()) if len(normal_delta) else 0.},
              "counts": {"shell": int((~route).sum()), "strand": int(route.sum())}}
    write_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, allow_nan=False))


def self_check() -> None:
    roots = np.array([[0., 0., 0.]], np.float32)
    model = SurfacePriorHybrid(roots, np.array([[0., 0., 1.]], np.float32),
                               np.full((1, 3), .5, np.float32), 1., layers=3, cap=6)
    xyz, _, _ = model.shell_geometry()
    require(torch.allclose(xyz[0, 0], model.shell_root[0]) and
            bool((xyz[0, 1:, 2] > xyz[0, :-1, 2]).all()), "Shell does not grow outward")
    model.setup_optimizer()
    require(next(g for g in model.optimizer.param_groups if g["name"] == "shell_root")["lr"] == 0 and
            next(g for g in model.optimizer.param_groups if g["name"] == "shell_rotation")["lr"] == 0,
            "attachment frame is not frozen")
    model.birth(torch.tensor([0]), torch.tensor([[1., 0., 1.]]), stage=1, segments=3)
    with torch.no_grad():
        model.strand_control.copy_(torch.tensor([[.1, 0., .2], [.2, 0., .1], [.3, 0., .3]]))
    start, end = model.curve_geometry()
    normal = torch.tensor([0., 0., 1.])
    require(bool((((end - start) * normal).sum(1) >= -1e-7).all()),
            "Strand has an inward segment")
    require(torch.allclose(end[:, 0], torch.tensor([.1, .2, .3])),
            "outward projection changed tangential bending")
    print(json.dumps({"status": "self_check_pass", "shell_z": xyz[0, :, 2].tolist(),
                      "strand_segment_normal_delta": ((end - start) * normal).sum(1).tolist()}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-check")
    p = sub.add_parser("train")
    p.add_argument("--surface", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prior-id", required=True); p.add_argument("--oracle", action="store_true")
    p.add_argument("--surface-regions", type=Path)
    p.add_argument("--scale", type=float, default=1.); p.add_argument("--translation", nargs=3, type=float, default=(0, 0, 0))
    p = sub.add_parser("render")
    p.add_argument("--checkpoint", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", choices=("train_v48", "test_v12"), required=True); p.add_argument("--method", required=True)
    p = sub.add_parser("audit")
    p.add_argument("--checkpoint", type=Path, required=True); p.add_argument("--initial-roots", type=Path, required=True)
    p.add_argument("--surface", type=Path, required=True); p.add_argument("--strands", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "self-check": self_check()
    elif args.command == "train": train(args)
    elif args.command == "render": render(args)
    else: audit(args)


if __name__ == "__main__": main()
