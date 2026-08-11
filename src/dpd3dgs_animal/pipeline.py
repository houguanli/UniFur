from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .config import PipelineConfig
from .coordinates import fit_skeleton_to_mesh
from .gaussian import bind_gaussians_to_surface, deform_gaussian_centers, load_gaussian_ply
from .mesh import SkeletonDrivenTetModel, simplify_mesh_for_tet
from .mocap_adapter import MocapAnythingAdapter
from .render import PinholeCamera, camera_from_arrays, camera_to_arrays, default_camera_for_vertices, load_gt_frame, point_splat_render, render_losses
from .sam3d_adapter import (
    Sam3DObjectAdapter,
    load_sam3d_camera_metadata,
    mesh_to_arrays,
)
from .video import ensure_frame_sequence, make_reference_mask


@dataclass
class Stage1Artifacts:
    work_dir: Path
    frame_dir: Path
    sam3d_ply: Path | None
    sam3d_glb: Path | None
    mocap_prediction: Path | None
    tet_npz: Path
    summary_json: Path
    losses_json: Path | None
    sam3d_metadata: Path | None


def run_stage1(
    input_video: str | Path,
    work_dir: str | Path,
    cfg: PipelineConfig,
    mask_path: str | Path | None = None,
    mesh_path: str | Path | None = None,
    gaussian_ply: str | Path | None = None,
    mocap_prediction: str | Path | None = None,
    run_sam3d: bool = True,
    run_mocap: bool = True,
    max_frames: int | None = None,
    render_size: tuple[int, int] | None = None,
    sam3d_reference_image: str | Path | None = None,
    sam3d_reference_transform: str | Path | None = None,
    sam3d_metadata: str | Path | None = None,
) -> Stage1Artifacts:
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    frames = ensure_frame_sequence(
        input_video,
        work_dir / "mocap_images",
        max_frames=max_frames,
    )
    native_render_size = frames.size
    render_size = render_size or native_render_size

    ref_index = min(cfg.reference_frame, len(frames.frame_paths) - 1)
    ref_image = (
        Path(sam3d_reference_image)
        if sam3d_reference_image is not None
        else frames.frame_paths[ref_index]
    )
    if mask_path is None:
        mask_path = make_reference_mask(ref_image, work_dir / "sam3d/reference_mask.png")
    else:
        mask_path = Path(mask_path)
        if sam3d_reference_image is None:
            crop_candidate = mask_path.parent / "cat_ref_rgb.png"
            if crop_candidate.exists():
                ref_image = crop_candidate
        if sam3d_reference_transform is None:
            transform_candidate = mask_path.parent / "reference_transform.json"
            if transform_candidate.exists():
                sam3d_reference_transform = transform_candidate

    sam3d_result = None
    camera = None
    sam3d_metadata_path = Path(sam3d_metadata) if sam3d_metadata else None
    if mesh_path is None and gaussian_ply is None and run_sam3d:
        sam3d = Sam3DObjectAdapter(
            cfg.paths.sam3d_root,
            cfg.sam3d_tag,
            checkpoint_root=cfg.paths.sam3d_checkpoints,
        )
        sam3d_result = sam3d.reconstruct(
            ref_image,
            mask_path,
            work_dir / "sam3d",
            reference_transform_path=sam3d_reference_transform,
        )
        vertices, faces = mesh_to_arrays(sam3d_result.mesh)
        gaussian_ply = sam3d_result.camera_ply_path
        mesh_path = sam3d_result.camera_mesh_path
        camera = sam3d_result.camera
        sam3d_metadata_path = sam3d_result.metadata_path
    else:
        if mesh_path is None:
            raise ValueError("mesh_path is required when run_sam3d is false or only gaussian_ply is supplied")
        vertices, faces = mesh_to_arrays(mesh_path)
        if sam3d_metadata_path is not None:
            camera, _metadata = load_sam3d_camera_metadata(sam3d_metadata_path)

    if camera is None:
        raise ValueError(
            "Strict camera-space Stage 1 requires SAM3D camera metadata. "
            "Run SAM3D in this pipeline or pass --sam3d-metadata with camera-space mesh/PLY assets."
        )

    original_mesh_counts = {
        "vertices": int(vertices.shape[0]),
        "faces": int(faces.shape[0]),
    }
    tet_vertices, tet_faces, tet_mesh_stats = simplify_mesh_for_tet(
        vertices,
        faces,
        max_faces=cfg.tet_mesh_max_faces,
    )
    np.savez_compressed(
        work_dir / "stage1_tet_input_mesh.npz",
        vertices=tet_vertices,
        faces=tet_faces,
        original_vertices=vertices,
        original_faces=faces,
    )

    if mocap_prediction is None and run_mocap:
        mocap = MocapAnythingAdapter(
            cfg.paths.mocap_root,
            checkpoint_root=cfg.paths.mocap_checkpoints,
            zoo_root=cfg.paths.mocap_zoo,
        )
        mocap_cfg = mocap.write_video2pose_config(
            frames.image_root,
            work_dir / "mocap_outputs",
            work_dir / "mocap_video2pose.yaml",
            ref_seq=cfg.mocap_ref_seq,
            ref_idx=cfg.mocap_ref_idx,
        )
        mocap.run_video2pose(mocap_cfg)
        mocap_prediction = mocap.find_prediction(work_dir / "mocap_outputs", "exp2503", frames.seq_name)

    if mocap_prediction is None:
        raise ValueError("mocap_prediction is required when run_mocap is false")

    mocap = MocapAnythingAdapter(
        cfg.paths.mocap_root,
        checkpoint_root=cfg.paths.mocap_checkpoints,
        zoo_root=cfg.paths.mocap_zoo,
    )
    skeleton = mocap.load_skeleton_prior(mocap_prediction, cfg.mocap_ref_seq)
    aligned_joints, skeleton_transform = fit_skeleton_to_mesh(
        skeleton.joints,
        tet_vertices,
        axis_transform=cfg.mocap_axis_transform,
        scale_mode=cfg.skeleton_scale_mode,
        center_mode=cfg.skeleton_center_mode,
        fit_padding=cfg.skeleton_fit_padding,
    )
    skeleton.joints = aligned_joints
    skeleton.coordinate_frame = cfg.canonical_frame

    driven = SkeletonDrivenTetModel.build(
        tet_vertices,
        tet_faces,
        skeleton,
        cfg.paths.elastic_root,
        k_nearest_joints=cfg.skinning_weight_k,
        weight_mode=cfg.skinning_weight_mode,
        deformation_mode=cfg.skinning_deformation_mode,
    )
    tet_npz = work_dir / "stage1_tet_skeleton_surface.npz"

    npz_payload = {
        "rest_tet_nodes": driven.rest_tet_nodes,
        "tets": driven.tets,
        "rest_surface_vertices": driven.rest_surface_vertices,
        "surface_faces": driven.surface_faces,
        "surface_node_indices": driven.surface_node_indices,
        "skeleton_joints": driven.posed_joints,
        "parents": driven.parents,
        "tet_weights": driven.tet_weights,
        "surface_weights": driven.surface_weights,
        "gravity": np.array([cfg.gravity], dtype=np.float32),
        "skinning_weight_mode": np.array([cfg.skinning_weight_mode]),
        "skinning_weight_k": np.array([cfg.skinning_weight_k], dtype=np.int32),
        "skinning_deformation_mode": np.array([cfg.skinning_deformation_mode]),
        "native_frame_size": np.array(native_render_size, dtype=np.int32),
        "render_size": np.array(render_size, dtype=np.int32),
    }
    if camera is not None:
        intrinsics, world_to_camera = camera_to_arrays(camera)
        npz_payload["camera_intrinsics"] = intrinsics
        npz_payload["camera_world_to_camera"] = world_to_camera
        npz_payload["camera_image_y_down"] = np.array(
            [int(camera.image_y_down)],
            dtype=np.int8,
        )
    np.savez_compressed(tet_npz, **npz_payload)

    losses_json = None
    if gaussian_ply is not None and Path(gaussian_ply).exists():
        camera_intrinsics, camera_world_to_camera = camera_to_arrays(camera)
        render_camera = camera_from_arrays(
            camera_intrinsics,
            camera_world_to_camera,
            width=render_size[0],
            height=render_size[1],
            image_y_down=camera.image_y_down,
        )
        losses_json = _render_and_score(
            gaussian_ply=Path(gaussian_ply),
            driven=driven,
            frame_paths=frames.frame_paths,
            out_dir=work_dir / "renders",
            render_size=render_size,
            color_weight=cfg.color_loss_weight,
            mask_weight=cfg.mask_loss_weight,
            max_points=cfg.max_render_points,
            device=cfg.device,
            camera=render_camera,
            radius_px=cfg.render_radius_px,
            vertex_k=cfg.gaussian_binding_k,
            pull_to_surface=cfg.pull_gaussians_to_surface,
        )

    summary = {
        "input_video": str(input_video),
        "frame_dir": str(frames.sequence_dir),
        "num_video_frames": len(frames.frame_paths),
        "native_frame_size": list(native_render_size),
        "render_size": list(render_size),
        "sam3d_ply": str(gaussian_ply) if gaussian_ply else None,
        "sam3d_glb": str(mesh_path) if mesh_path else None,
        "sam3d_metadata": str(sam3d_metadata_path) if sam3d_metadata_path else None,
        "mocap_prediction": str(mocap_prediction),
        "skeleton_frames": int(driven.posed_joints.shape[0]),
        "skeleton_joints": int(driven.posed_joints.shape[1]),
        "sam3d_original_mesh": original_mesh_counts,
        "tet_input_mesh": tet_mesh_stats,
        "tet_nodes": int(driven.rest_tet_nodes.shape[0]),
        "tets": int(driven.tets.shape[0]),
        "surface_vertices": int(driven.rest_surface_vertices.shape[0]),
        "surface_faces": int(driven.surface_faces.shape[0]),
        "canonical_frame": "opencv_camera_x_right_y_down_z_forward",
        "mocap_axis_transform": cfg.mocap_axis_transform,
        "skeleton_scale_mode": cfg.skeleton_scale_mode,
        "skeleton_center_mode": cfg.skeleton_center_mode,
        "skeleton_fit_padding": cfg.skeleton_fit_padding,
        "skeleton_similarity_transform": asdict(skeleton_transform),
        "gravity": cfg.gravity,
        "skinning_weight_mode": cfg.skinning_weight_mode,
        "skinning_weight_k": cfg.skinning_weight_k,
        "skinning_deformation_mode": cfg.skinning_deformation_mode,
        "camera": {
            "origin": [0.0, 0.0, 0.0],
            "forward": [0.0, 0.0, 1.0],
            "world_to_camera": np.eye(4, dtype=np.float32),
            "image_y_down": True,
            "source": "sam3d_pose_and_crop_metadata",
        },
        "losses_json": str(losses_json) if losses_json else None,
    }
    summary_json = work_dir / "stage1_summary.json"
    _write_json(summary_json, summary)

    return Stage1Artifacts(
        work_dir=work_dir,
        frame_dir=frames.sequence_dir,
        sam3d_ply=Path(gaussian_ply) if gaussian_ply else None,
        sam3d_glb=sam3d_result.glb_path if sam3d_result else None,
        mocap_prediction=Path(mocap_prediction),
        tet_npz=tet_npz,
        summary_json=summary_json,
        losses_json=losses_json,
        sam3d_metadata=sam3d_metadata_path,
    )


def _render_and_score(
    gaussian_ply: Path,
    driven: SkeletonDrivenTetModel,
    frame_paths: list[Path],
    out_dir: Path,
    render_size: tuple[int, int],
    color_weight: float,
    mask_weight: float,
    max_points: int,
    device: str,
    camera: PinholeCamera | None = None,
    radius_px: int = 4,
    vertex_k: int = 8,
    pull_to_surface: bool = True,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    width, height = render_size
    cloud = load_gaussian_ply(str(gaussian_ply))
    if cloud.xyz.shape[0] > max_points:
        idx = np.linspace(0, cloud.xyz.shape[0] - 1, max_points).astype(np.int64)
        opacity = cloud.opacity[idx] if cloud.opacity is not None else None
        cloud = type(cloud)(cloud.xyz[idx], cloud.color[idx], opacity)
    binding = bind_gaussians_to_surface(
        cloud.xyz,
        driven.rest_surface_vertices,
        driven.surface_faces,
        device=device,
        vertex_k=vertex_k,
        pull_to_surface=pull_to_surface,
    )
    camera = camera or default_camera_for_vertices(driven.rest_surface_vertices, width, height)

    per_frame = []
    n = min(len(frame_paths), driven.posed_joints.shape[0])
    for frame_index in range(n):
        driven_frame = driven.frame(frame_index)
        xyz = deform_gaussian_centers(binding, driven_frame.surface_vertices, driven_frame.surface_faces)
        pred = point_splat_render(
            cloud,
            camera,
            xyz=xyz,
            radius_px=radius_px,
            max_points=max_points,
            device=device,
        )
        gt = load_gt_frame(str(frame_paths[frame_index]), width, height)
        losses = render_losses(
            pred["rgb"],
            pred["mask"],
            gt["rgb"],
            gt["mask"],
            color_weight=color_weight,
            mask_weight=mask_weight,
        )
        per_frame.append({"frame": frame_index, **losses})
        if frame_index < 8:
            Image.fromarray((np.clip(pred["rgb"], 0.0, 1.0) * 255).astype(np.uint8)).save(
                out_dir / f"pred_{frame_index:05d}.png"
            )

    mean = {
        "color": float(np.mean([x["color"] for x in per_frame])) if per_frame else 0.0,
        "mask": float(np.mean([x["mask"] for x in per_frame])) if per_frame else 0.0,
        "total": float(np.mean([x["total"] for x in per_frame])) if per_frame else 0.0,
    }
    path = out_dir / "losses.json"
    _write_json(path, {"mean": mean, "frames": per_frame})
    return path


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default)


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)
