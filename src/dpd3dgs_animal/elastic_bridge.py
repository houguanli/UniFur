from __future__ import annotations

import json
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .config import PipelineConfig
from .gaussian import bind_gaussians_to_surface, deform_gaussian_centers, load_gaussian_ply
from .render import camera_from_stage1_npz, default_camera_for_vertices, load_gt_frame, point_splat_render, render_losses
from .video import image_size


@dataclass
class ElasticForwardArtifacts:
    out_dir: Path
    state_npz: Path
    losses_json: Path | None
    vertices_bin: Path
    driver_log: Path


def run_elastic_forward(
    stage1_npz: str | Path,
    out_dir: str | Path,
    cfg: PipelineConfig,
    gaussian_ply: str | Path | None = None,
    frame_dir: str | Path | None = None,
    max_frames: int | None = None,
    driver_path: str | Path | None = None,
    render_size: tuple[int, int] | None = None,
    radius_scale: float | None = None,
    min_radius_scale: float | None = None,
    max_radius_scale: float | None = None,
    min_bone_length_scale: float | None = None,
) -> ElasticForwardArtifacts:
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stage1_npz = Path(stage1_npz).resolve()
    data = np.load(stage1_npz)

    input_dir = out_dir / "elastic_driver_inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    tet_prefix = input_dir / "stage1_tet"
    skeleton_bin = input_dir / "skeleton.bin"
    vertices_bin = out_dir / "elastic_vertices.bin"
    driver_log = out_dir / "elastic_driver.log"

    _write_tetgen_node_ele(
        tet_prefix,
        np.asarray(data["rest_tet_nodes"], dtype=np.float32),
        np.asarray(data["tets"], dtype=np.int64),
    )
    joints = np.asarray(data["skeleton_joints"], dtype=np.float32)
    if max_frames is not None:
        joints = joints[: int(max_frames)]
    parents = np.asarray(data["parents"], dtype=np.int32)
    _write_skeleton_binary(skeleton_bin, joints, parents)

    driver = Path(driver_path) if driver_path else _default_driver_path(cfg)
    if not driver.exists():
        raise FileNotFoundError(
            f"Elastic headless driver not found: {driver}. "
            "Build target HeadlessConstrainedFEM under third_party/elastic_simulator first."
        )

    command = [
        str(driver),
        "--tet-prefix",
        str(tet_prefix),
        "--skeleton",
        str(skeleton_bin),
        "--out",
        str(vertices_bin),
        "--dt",
        str(cfg.elastic_fem_dt),
        "--substeps",
        str(cfg.elastic_fem_substeps),
        "--youngs-modulus",
        str(cfg.elastic_fem_youngs_modulus),
        "--poisson-ratio",
        str(cfg.elastic_fem_poisson_ratio),
        "--density",
        str(cfg.elastic_fem_density),
        "--damping",
        str(cfg.elastic_fem_damping),
        "--gravity",
        str(cfg.gravity),
        "--radius-scale",
        str(radius_scale if radius_scale is not None else cfg.elastic_fem_radius_scale),
        "--min-radius-scale",
        str(min_radius_scale if min_radius_scale is not None else cfg.elastic_fem_min_radius_scale),
        "--max-radius-scale",
        str(max_radius_scale if max_radius_scale is not None else cfg.elastic_fem_max_radius_scale),
        "--min-bone-length-scale",
        str(
            min_bone_length_scale
            if min_bone_length_scale is not None
            else cfg.elastic_fem_min_bone_length_scale
        ),
        "--reset-each-frame",
        "1" if cfg.elastic_fem_reset_each_frame else "0",
    ]
    if max_frames is not None:
        command.extend(["--max-frames", str(int(max_frames))])

    result = subprocess.run(
        command,
        cwd=str(cfg.paths.elastic_root),
        text=True,
        capture_output=True,
        check=False,
    )
    driver_log.write_text(
        "$ " + " ".join(command) + "\n\n"
        + "STDOUT\n"
        + result.stdout
        + "\nSTDERR\n"
        + result.stderr,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(f"Elastic headless driver failed; see {driver_log}")

    tet_nodes = _read_vertices_binary(vertices_bin)
    surface_node_indices = np.asarray(data["surface_node_indices"], dtype=np.int64)
    surface_vertices = tet_nodes[:, surface_node_indices]
    state_npz = out_dir / "elastic_forward_state.npz"
    np.savez_compressed(
        state_npz,
        tet_nodes=tet_nodes,
        surface_vertices=surface_vertices,
        surface_faces=np.asarray(data["surface_faces"], dtype=np.int64),
        surface_node_indices=surface_node_indices,
        rest_tet_nodes=np.asarray(data["rest_tet_nodes"], dtype=np.float32),
        rest_surface_vertices=np.asarray(data["rest_surface_vertices"], dtype=np.float32),
        tets=np.asarray(data["tets"], dtype=np.int64),
        skeleton_joints=joints,
        parents=parents,
        stage1_npz=str(stage1_npz),
        driver=str(driver),
        command=np.asarray(command),
    )

    losses_json = None
    if gaussian_ply is not None and frame_dir is not None:
        losses_json = render_elastic_forward(
            state_npz=state_npz,
            stage1_npz=stage1_npz,
            gaussian_ply=gaussian_ply,
            frame_dir=frame_dir,
            out_dir=out_dir / "renders",
            cfg=cfg,
            render_size=render_size,
        )

    summary = {
        "stage1_npz": str(stage1_npz),
        "state_npz": str(state_npz),
        "vertices_bin": str(vertices_bin),
        "driver_log": str(driver_log),
        "frames": int(tet_nodes.shape[0]),
        "tet_nodes": int(tet_nodes.shape[1]),
        "surface_vertices": int(surface_vertices.shape[1]),
        "elastic_fem": {
            "dt": cfg.elastic_fem_dt,
            "substeps": cfg.elastic_fem_substeps,
            "youngs_modulus": cfg.elastic_fem_youngs_modulus,
            "poisson_ratio": cfg.elastic_fem_poisson_ratio,
            "density": cfg.elastic_fem_density,
            "damping": cfg.elastic_fem_damping,
            "gravity": cfg.gravity,
            "radius_scale": radius_scale if radius_scale is not None else cfg.elastic_fem_radius_scale,
            "min_radius_scale": (
                min_radius_scale if min_radius_scale is not None else cfg.elastic_fem_min_radius_scale
            ),
            "max_radius_scale": (
                max_radius_scale if max_radius_scale is not None else cfg.elastic_fem_max_radius_scale
            ),
            "min_bone_length_scale": (
                min_bone_length_scale
                if min_bone_length_scale is not None
                else cfg.elastic_fem_min_bone_length_scale
            ),
            "reset_each_frame": cfg.elastic_fem_reset_each_frame,
        },
        "losses_json": str(losses_json) if losses_json else None,
    }
    (out_dir / "elastic_forward_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return ElasticForwardArtifacts(out_dir, state_npz, losses_json, vertices_bin, driver_log)


def render_elastic_forward(
    state_npz: str | Path,
    stage1_npz: str | Path,
    gaussian_ply: str | Path,
    frame_dir: str | Path,
    out_dir: str | Path,
    cfg: PipelineConfig,
    render_size: tuple[int, int] | None = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state = np.load(state_npz)
    surface_vertices = np.asarray(state["surface_vertices"], dtype=np.float32)
    rest_surface_vertices = np.asarray(state["rest_surface_vertices"], dtype=np.float32)
    surface_faces = np.asarray(state["surface_faces"], dtype=np.int64)
    frame_paths = _frame_paths(frame_dir)
    width, height = _resolve_render_size(render_size, frame_paths, stage1_npz)
    camera = camera_from_stage1_npz(str(stage1_npz), width, height)
    if camera is None:
        camera = default_camera_for_vertices(rest_surface_vertices, width, height)

    cloud = load_gaussian_ply(str(gaussian_ply))
    if cloud.xyz.shape[0] > cfg.max_render_points:
        idx = np.linspace(0, cloud.xyz.shape[0] - 1, cfg.max_render_points).astype(np.int64)
        opacity = cloud.opacity[idx] if cloud.opacity is not None else None
        cloud = type(cloud)(cloud.xyz[idx], cloud.color[idx], opacity)
    binding = bind_gaussians_to_surface(
        cloud.xyz,
        rest_surface_vertices,
        surface_faces,
        device=cfg.device,
        vertex_k=cfg.gaussian_binding_k,
        pull_to_surface=cfg.pull_gaussians_to_surface,
    )

    n = min(len(frame_paths), surface_vertices.shape[0])
    per_frame: list[dict[str, float | int]] = []
    preview_indices = _preview_indices(n)
    preview_paths: list[str] = []
    video_writer = None
    video_path = out_dir / "elastic_render.mp4"
    try:
        video_writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            30.0,
            (width, height),
        )
        for frame_index in range(n):
            xyz = deform_gaussian_centers(
                binding,
                surface_vertices[frame_index],
                surface_faces,
            )
            pred = point_splat_render(
                cloud,
                camera,
                xyz=xyz,
                radius_px=cfg.render_radius_px,
                max_points=cfg.max_render_points,
                device=cfg.device,
            )
            gt = load_gt_frame(str(frame_paths[frame_index]), width, height)
            losses = render_losses(
                pred["rgb"],
                pred["mask"],
                gt["rgb"],
                gt["mask"],
                color_weight=cfg.color_loss_weight,
                mask_weight=cfg.mask_loss_weight,
            )
            per_frame.append({"frame": frame_index, **losses})
            image = (pred["rgb"].clip(0.0, 1.0) * 255.0).astype(np.uint8)
            if video_writer is not None:
                video_writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            if frame_index in preview_indices:
                path = out_dir / f"elastic_{frame_index:05d}.png"
                Image.fromarray(image).save(path)
                preview_paths.append(str(path))
    finally:
        if video_writer is not None:
            video_writer.release()

    mean = {
        key: float(np.mean([row[key] for row in per_frame]))
        for key in ("color", "mask", "total")
    }
    losses_json = out_dir / "elastic_forward_losses.json"
    losses_json.write_text(
        json.dumps(
            {
                "frames": n,
                "render_size": [width, height],
                "weights": {
                    "color": cfg.color_loss_weight,
                    "mask": cfg.mask_loss_weight,
                },
                "max_render_points": cfg.max_render_points,
                "gaussian_binding_k": cfg.gaussian_binding_k,
                "pull_gaussians_to_surface": cfg.pull_gaussians_to_surface,
                "mean": mean,
                "per_frame": per_frame,
                "previews": preview_paths,
                "video": str(video_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return losses_json


def _write_tetgen_node_ele(prefix: Path, nodes: np.ndarray, tets: np.ndarray) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    nodes = np.asarray(nodes, dtype=np.float32)
    tets = np.asarray(tets, dtype=np.int64)
    with open(prefix.with_suffix(".node"), "w", encoding="utf-8") as f:
        f.write(f"{nodes.shape[0]} 3 0 0\n")
        for index, xyz in enumerate(nodes):
            f.write(f"{index} {xyz[0]:.9g} {xyz[1]:.9g} {xyz[2]:.9g}\n")
    with open(prefix.with_suffix(".ele"), "w", encoding="utf-8") as f:
        f.write(f"{tets.shape[0]} 4 0\n")
        for index, tet in enumerate(tets):
            f.write(f"{index} {int(tet[0])} {int(tet[1])} {int(tet[2])} {int(tet[3])}\n")


def _write_skeleton_binary(path: Path, joints: np.ndarray, parents: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joints = np.asarray(joints, dtype=np.float32)
    parents = np.asarray(parents, dtype=np.int32)
    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise ValueError(f"Expected joints [frames, joints, 3], got {joints.shape}")
    if parents.shape[0] != joints.shape[1]:
        raise ValueError("parents length must match joint count")
    with open(path, "wb") as f:
        f.write(b"DPD3SKL1")
        f.write(struct.pack("<ii", int(joints.shape[0]), int(joints.shape[1])))
        f.write(parents.astype("<i4", copy=False).tobytes())
        f.write(joints.astype("<f4", copy=False).tobytes())


def _read_vertices_binary(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != b"DPD3VTX1":
            raise ValueError(f"Invalid Elastic vertices magic in {path}")
        frames, vertices = struct.unpack("<ii", f.read(8))
        raw = f.read()
    data = np.frombuffer(raw, dtype="<f4")
    expected = frames * vertices * 3
    if data.size != expected:
        raise ValueError(f"Expected {expected} floats in {path}, got {data.size}")
    return data.reshape(frames, vertices, 3).astype(np.float32, copy=False)


def _default_driver_path(cfg: PipelineConfig) -> Path:
    preferred = cfg.paths.elastic_root / "build-dpd3dgs/bin/HeadlessConstrainedFEM"
    if preferred.exists():
        return preferred
    return cfg.paths.elastic_root / "build-cuda118/bin/HeadlessConstrainedFEM"


def _frame_paths(frame_dir: str | Path) -> list[Path]:
    root = Path(frame_dir)
    if not root.exists():
        raise FileNotFoundError(root)
    return sorted(p for p in root.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})


def _resolve_render_size(
    render_size: tuple[int, int] | None,
    frame_paths: list[Path],
    stage1_npz: str | Path,
) -> tuple[int, int]:
    if render_size is not None:
        return int(render_size[0]), int(render_size[1])
    if frame_paths:
        return image_size(frame_paths[0])
    data = np.load(stage1_npz)
    if "native_frame_size" in data:
        size = np.asarray(data["native_frame_size"], dtype=np.int32)
        return int(size[0]), int(size[1])
    size = np.asarray(data["render_size"], dtype=np.int32)
    return int(size[0]), int(size[1])


def _preview_indices(n: int) -> set[int]:
    if n <= 0:
        return set()
    return {0, min(n - 1, 30), min(n - 1, 60), n - 1}
