from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .config import PipelineConfig
from .gaussian import (
    GaussianCloud,
    bind_gaussians_to_surface,
    deform_gaussian_centers,
    load_gaussian_ply,
)
from .mesh import (
    deform_points_by_joint_displacement,
    skin_points_by_bone_dqs,
    skin_points_by_bone_lbs,
)
from .render import (
    PinholeCamera,
    camera_from_stage1_npz,
    load_gt_frame,
    point_splat_render,
)


GROUP_COLORS = np.asarray(
    [
        [80, 200, 120],
        [155, 95, 220],
        [55, 155, 240],
        [245, 165, 45],
        [235, 80, 75],
    ],
    dtype=np.float32,
) / 255.0


def audit_pipeline(
    frame_dir: str | Path,
    gaussian_ply: str | Path,
    stage1_npz: str | Path,
    out_dir: str | Path,
    cfg: PipelineConfig,
    baseline_stage1_npz: str | Path | None = None,
    optimized_joints: str | Path | None = None,
    frame_indices: tuple[int, ...] = (0, 30, 60, 89),
) -> Path:
    frame_dir = Path(frame_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = sorted(
        p for p in frame_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not frame_paths:
        raise ValueError(f"No frames found in {frame_dir}")

    current = np.load(stage1_npz)
    width, height = Image.open(frame_paths[0]).size
    camera = camera_from_stage1_npz(str(stage1_npz), width, height)
    if camera is None:
        raise ValueError("Pipeline diagnostics require Stage 1 camera metadata")

    valid_indices = tuple(
        sorted({min(max(0, int(index)), len(frame_paths) - 1) for index in frame_indices})
    )
    _save_input_mask_sheet(frame_paths, valid_indices, out_dir / "01_input_and_mask.png")
    _save_skeleton_sheet(
        frame_paths,
        current["skeleton_joints"],
        current["parents"],
        camera,
        valid_indices,
        out_dir / "03_mocap_skeleton_overlay.png",
    )

    cloud = _sample_cloud(load_gaussian_ply(str(gaussian_ply)), cfg.max_render_points)
    static_render = point_splat_render(
        cloud,
        camera,
        radius_px=cfg.render_radius_px,
        max_points=cfg.max_render_points,
        device=cfg.device,
    )
    _save_static_sam3d(
        frame_paths[0],
        static_render,
        out_dir / "02_sam3d_static_frame0.png",
    )

    binding = bind_gaussians_to_surface(
        cloud.xyz,
        current["rest_surface_vertices"],
        current["surface_faces"],
        device=cfg.device,
        vertex_k=cfg.gaussian_binding_k,
        pull_to_surface=cfg.pull_gaussians_to_surface,
    )
    _save_weight_visualization(
        current,
        camera,
        out_dir / "04_surface_skinning_weights.png",
        cfg,
    )

    states: list[tuple[str, np.lib.npyio.NpzFile, np.ndarray, str]] = []
    if baseline_stage1_npz is not None:
        baseline = np.load(baseline_stage1_npz)
        states.append(
            ("baseline_translation", baseline, baseline["skeleton_joints"], "translation")
        )
    deformation_mode = (
        str(np.asarray(current["skinning_deformation_mode"]).reshape(-1)[0])
        if "skinning_deformation_mode" in current
        else "lbs"
    )
    states.append(
        (f"{deformation_mode}_initial", current, current["skeleton_joints"], deformation_mode)
    )
    if optimized_joints is not None:
        states.append(
            (f"{deformation_mode}_optimized", current, np.load(optimized_joints), deformation_mode)
        )

    rendered: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    surface_rendered: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    metrics: dict[str, dict[str, object]] = {}
    for name, data, joints, mode in states:
        state_renders: dict[int, dict[str, np.ndarray]] = {}
        state_surface_renders: dict[int, dict[str, np.ndarray]] = {}
        frame_metrics: list[dict[str, float]] = []
        edge_metrics: list[dict[str, float]] = []
        for frame_index in valid_indices:
            surface = _deformed_surface(data, joints, frame_index, mode)
            xyz = deform_gaussian_centers(binding, surface, current["surface_faces"])
            pred = point_splat_render(
                cloud,
                camera,
                xyz=xyz,
                radius_px=cfg.render_radius_px,
                max_points=cfg.max_render_points,
                device=cfg.device,
            )
            surface_cloud = GaussianCloud(
                surface.astype(np.float32),
                np.tile(
                    np.asarray([[0.25, 0.8, 1.0]], dtype=np.float32),
                    (surface.shape[0], 1),
                ),
                np.ones((surface.shape[0],), dtype=np.float32),
            )
            state_surface_renders[frame_index] = point_splat_render(
                surface_cloud,
                camera,
                radius_px=3,
                max_points=surface.shape[0],
                device=cfg.device,
            )
            gt = load_gt_frame(str(frame_paths[frame_index]), width, height)
            state_renders[frame_index] = pred
            frame_metrics.append(
                {"frame": frame_index, **_image_metrics(pred, gt)}
            )
            tet_nodes = _deformed_tet_nodes(data, joints, frame_index, mode)
            edge_metrics.append(
                {"frame": frame_index, **_edge_distortion(data, tet_nodes)}
            )
        rendered[name] = state_renders
        surface_rendered[name] = state_surface_renders
        metrics[name] = {
            "image": frame_metrics,
            "edge_distortion": edge_metrics,
            "mean": _mean_metrics(frame_metrics),
        }

    _save_motion_comparison(
        frame_paths,
        rendered,
        valid_indices,
        out_dir / "05_motion_stage_comparison.png",
    )
    _save_tet_gaussian_comparison(
        frame_paths,
        surface_rendered,
        rendered,
        valid_indices,
        out_dir / "06_tet_surface_vs_gs.png",
    )
    report = {
        "frames": list(valid_indices),
        "render_size": [width, height],
        "gaussians_used": int(cloud.xyz.shape[0]),
        "tet_nodes": int(current["rest_tet_nodes"].shape[0]),
        "tetrahedra": int(current["tets"].shape[0]),
        "surface_vertices": int(current["rest_surface_vertices"].shape[0]),
        "parents": current["parents"].astype(int).tolist(),
        "root_children": int(np.count_nonzero(current["parents"][1:] == 0)),
        "metrics": metrics,
    }
    report_path = out_dir / "pipeline_diagnostics.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path


def _deformed_tet_nodes(
    data: np.lib.npyio.NpzFile,
    joints: np.ndarray,
    frame_index: int,
    mode: str,
) -> np.ndarray:
    if mode == "translation":
        return deform_points_by_joint_displacement(
            data["rest_tet_nodes"],
            joints[0],
            joints[frame_index],
            data["tet_weights"],
        )
    skinning = skin_points_by_bone_dqs if mode == "dqs" else skin_points_by_bone_lbs
    return skinning(
        data["rest_tet_nodes"],
        joints[0],
        joints[frame_index],
        data["parents"],
        data["tet_weights"],
    )


def _deformed_surface(
    data: np.lib.npyio.NpzFile,
    joints: np.ndarray,
    frame_index: int,
    mode: str,
) -> np.ndarray:
    tet_nodes = _deformed_tet_nodes(data, joints, frame_index, mode)
    return tet_nodes[data["surface_node_indices"]]


def _save_input_mask_sheet(
    frame_paths: list[Path],
    frame_indices: tuple[int, ...],
    path: Path,
) -> None:
    rows: list[list[Image.Image]] = []
    for frame_index in frame_indices:
        rgba = np.asarray(Image.open(frame_paths[frame_index]).convert("RGBA"))
        rgb = rgba[..., :3]
        alpha = rgba[..., 3]
        mask_rgb = np.repeat(alpha[..., None], 3, axis=-1)
        cutout = rgb.copy()
        cutout[alpha <= 127] = 0
        rows.append(
            [
                _tile(Image.fromarray(rgb), f"RGB frame {frame_index}"),
                _tile(Image.fromarray(mask_rgb), "RMBG alpha"),
                _tile(Image.fromarray(cutout), "masked input"),
            ]
        )
    _save_grid(rows, path)


def _save_static_sam3d(
    frame_path: Path,
    pred: dict[str, np.ndarray],
    path: Path,
) -> None:
    rgba = np.asarray(Image.open(frame_path).convert("RGBA"))
    gt = rgba[..., :3].copy()
    gt[rgba[..., 3] <= 127] = 0
    pred_rgb = (np.clip(pred["rgb"], 0.0, 1.0) * 255).astype(np.uint8)
    pred_mask = pred["mask"] > 0.5
    overlay = gt.copy()
    boundary = cv2.morphologyEx(
        pred_mask.astype(np.uint8),
        cv2.MORPH_GRADIENT,
        np.ones((5, 5), dtype=np.uint8),
    )
    overlay[boundary > 0] = np.array([255, 70, 70], dtype=np.uint8)
    _save_grid(
        [[
            _tile(Image.fromarray(gt), "GT frame 0"),
            _tile(Image.fromarray(pred_rgb), "SAM3D static GS"),
            _tile(Image.fromarray(overlay), "SAM3D mask boundary"),
        ]],
        path,
    )


def _save_skeleton_sheet(
    frame_paths: list[Path],
    joints: np.ndarray,
    parents: np.ndarray,
    camera: PinholeCamera,
    frame_indices: tuple[int, ...],
    path: Path,
) -> None:
    rows = []
    for frame_index in frame_indices:
        image = np.asarray(Image.open(frame_paths[frame_index]).convert("RGB"))
        overlay = _draw_skeleton(image, joints[frame_index], parents, camera)
        rows.append([_tile(Image.fromarray(overlay), f"Mocap skeleton frame {frame_index}")])
    _save_grid(rows, path)


def _save_weight_visualization(
    data: np.lib.npyio.NpzFile,
    camera: PinholeCamera,
    path: Path,
    cfg: PipelineConfig,
) -> None:
    dominant = np.argmax(data["surface_weights"], axis=1)
    groups = np.asarray([_joint_group(int(index)) for index in dominant])
    colors = GROUP_COLORS[groups]
    cloud = GaussianCloud(
        data["rest_surface_vertices"].astype(np.float32),
        colors.astype(np.float32),
        np.ones((len(colors),), dtype=np.float32),
    )
    pred = point_splat_render(
        cloud,
        camera,
        radius_px=3,
        max_points=len(colors),
        device=cfg.device,
    )
    rgb = (np.clip(pred["rgb"], 0.0, 1.0) * 255).astype(np.uint8)
    _save_grid(
        [[_tile(Image.fromarray(rgb), "dominant bone weights: torso/tail/hind/front/head")]],
        path,
    )


def _save_motion_comparison(
    frame_paths: list[Path],
    rendered: dict[str, dict[int, dict[str, np.ndarray]]],
    frame_indices: tuple[int, ...],
    path: Path,
) -> None:
    rows: list[list[Image.Image]] = []
    state_names = list(rendered)
    for frame_index in frame_indices:
        rgba = np.asarray(Image.open(frame_paths[frame_index]).convert("RGBA"))
        gt = rgba[..., :3].copy()
        gt[rgba[..., 3] <= 127] = 0
        row = [_tile(Image.fromarray(gt), f"GT frame {frame_index}")]
        for state_name in state_names:
            rgb = (
                np.clip(rendered[state_name][frame_index]["rgb"], 0.0, 1.0) * 255
            ).astype(np.uint8)
            row.append(_tile(Image.fromarray(rgb), state_name))
        rows.append(row)
    _save_grid(rows, path)


def _save_tet_gaussian_comparison(
    frame_paths: list[Path],
    surface_rendered: dict[str, dict[int, dict[str, np.ndarray]]],
    gaussian_rendered: dict[str, dict[int, dict[str, np.ndarray]]],
    frame_indices: tuple[int, ...],
    path: Path,
) -> None:
    state_name = next(
        (name for name in surface_rendered if name.endswith("_optimized")),
        list(surface_rendered)[-1],
    )
    rows: list[list[Image.Image]] = []
    for frame_index in frame_indices:
        rgba = np.asarray(Image.open(frame_paths[frame_index]).convert("RGBA"))
        gt = rgba[..., :3].copy()
        gt[rgba[..., 3] <= 127] = 0
        surface_rgb = (
            np.clip(surface_rendered[state_name][frame_index]["rgb"], 0.0, 1.0)
            * 255
        ).astype(np.uint8)
        gaussian_rgb = (
            np.clip(gaussian_rendered[state_name][frame_index]["rgb"], 0.0, 1.0)
            * 255
        ).astype(np.uint8)
        rows.append(
            [
                _tile(Image.fromarray(gt), f"GT frame {frame_index}"),
                _tile(Image.fromarray(surface_rgb), f"tet surface: {state_name}"),
                _tile(Image.fromarray(gaussian_rgb), f"bound GS: {state_name}"),
            ]
        )
    _save_grid(rows, path)


def _draw_skeleton(
    image: np.ndarray,
    joints: np.ndarray,
    parents: np.ndarray,
    camera: PinholeCamera,
) -> np.ndarray:
    result = image.copy()
    uv = _project(joints, camera)
    for child, parent in enumerate(parents):
        if parent < 0:
            continue
        a = tuple(np.round(uv[int(parent)]).astype(int))
        b = tuple(np.round(uv[child]).astype(int))
        cv2.line(result, a, b, (255, 235, 20), 3, cv2.LINE_AA)
    for point in uv:
        cv2.circle(
            result,
            tuple(np.round(point).astype(int)),
            5,
            (255, 45, 45),
            -1,
            cv2.LINE_AA,
        )
    return result


def _project(points: np.ndarray, camera: PinholeCamera) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    camera_points = np.concatenate([points, ones], axis=1) @ camera.world_to_camera.T
    z = np.maximum(camera_points[:, 2], 1e-6)
    x = camera.fx * camera_points[:, 0] / z + camera.cx
    y_sign = 1.0 if camera.image_y_down else -1.0
    y = camera.cy + y_sign * camera.fy * camera_points[:, 1] / z
    return np.stack([x, y], axis=-1)


def _sample_cloud(cloud: GaussianCloud, max_points: int) -> GaussianCloud:
    if cloud.xyz.shape[0] <= max_points:
        return cloud
    indices = np.linspace(0, cloud.xyz.shape[0] - 1, max_points).astype(np.int64)
    opacity = cloud.opacity[indices] if cloud.opacity is not None else None
    return GaussianCloud(cloud.xyz[indices], cloud.color[indices], opacity)


def _image_metrics(
    pred: dict[str, np.ndarray],
    gt: dict[str, np.ndarray],
) -> dict[str, float]:
    pred_mask = pred["mask"] > 0.5
    gt_mask = gt["mask"] > 0.5
    intersection = np.count_nonzero(pred_mask & gt_mask)
    union = max(np.count_nonzero(pred_mask | gt_mask), 1)
    valid = gt_mask[..., None].astype(np.float32)
    color_mae = float(
        (np.abs(pred["rgb"] - gt["rgb"]) * valid).sum()
        / max(float(valid.sum() * 3.0), 1.0)
    )
    mask_error = float(np.mean(pred_mask != gt_mask))
    return {
        "color_mae": color_mae,
        "mask_01": mask_error,
        "mask_iou": float(intersection / union),
        "total": color_mae + 10.0 * mask_error,
    }


def _edge_distortion(
    data: np.lib.npyio.NpzFile,
    tet_nodes: np.ndarray,
) -> dict[str, float]:
    edges = _unique_tet_edges(data["tets"])
    rest = np.linalg.norm(
        data["rest_tet_nodes"][edges[:, 0]] - data["rest_tet_nodes"][edges[:, 1]],
        axis=-1,
    )
    posed = np.linalg.norm(
        tet_nodes[edges[:, 0]] - tet_nodes[edges[:, 1]],
        axis=-1,
    )
    relative = np.abs(posed / np.maximum(rest, 1e-8) - 1.0)
    return {
        "mean_abs_relative": float(relative.mean()),
        "p95_abs_relative": float(np.percentile(relative, 95)),
        "max_abs_relative": float(relative.max()),
    }


def _unique_tet_edges(tets: np.ndarray) -> np.ndarray:
    pairs = np.asarray(
        [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
        dtype=np.int64,
    )
    edges = np.asarray(tets, dtype=np.int64)[:, pairs].reshape(-1, 2)
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def _mean_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    keys = [key for key in metrics[0] if key != "frame"]
    return {key: float(np.mean([item[key] for item in metrics])) for key in keys}


def _joint_group(index: int) -> int:
    if 2 <= index <= 5:
        return 1
    if 7 <= index <= 18:
        return 2
    if 22 <= index <= 33:
        return 3
    if index >= 34:
        return 4
    return 0


def _tile(image: Image.Image, title: str, size: tuple[int, int] = (640, 360)) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (size[0], size[1] + 32), "black")
    tile.paste(image, ((size[0] - image.width) // 2, 32))
    ImageDraw.Draw(tile).text((10, 9), title, fill="white")
    return tile


def _save_grid(rows: list[list[Image.Image]], path: Path) -> None:
    columns = max(len(row) for row in rows)
    width = max(image.width for row in rows for image in row)
    height = max(image.height for row in rows for image in row)
    canvas = Image.new("RGB", (columns * width, len(rows) * height), "black")
    for row_index, row in enumerate(rows):
        for column_index, image in enumerate(row):
            canvas.paste(image, (column_index * width, row_index * height))
    canvas.save(path)
