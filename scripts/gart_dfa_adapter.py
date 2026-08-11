#!/usr/bin/env python3
"""Run GART on the frozen DFA Panda protocol without proprietary D-SMAL assets.

The adapter changes only GART's articulated template/data boundary.  GART's
GaussianTemplateModel, optimizer, densification, losses, and renderer are kept
unchanged.  The public DFA rest mesh, 93-way skinning weights, exact bone
matrices, calibrated training camera, and training RGBA frames replace the
licensed D-SMAL/BITE inputs expected by GART's animal demo.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from torch import nn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("validate", "smoke-template", "train", "render"), required=True
    )
    parser.add_argument("--gart-root", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--render-dir", type=Path)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-observations", type=int)
    return parser.parse_args()


ARGS = parse_args()
GART_ROOT = ARGS.gart_root.resolve()
sys.path.insert(0, str(GART_ROOT))

# GART imports its D-SMAL implementation at module-import time even when a
# different template backend is used.  The licensed data files are not part of
# the repository, so provide a deliberately non-instantiable import shim.  The
# DFA fitter replaces get_template before any model is constructed.
smal_package = types.ModuleType("smal")
smal_package.__path__ = []  # type: ignore[attr-defined]
smal_tpg = types.ModuleType("smal.smal_tpg")


class _UnavailableSMAL:
    def __init__(self, *_: Any, **__: Any) -> None:
        raise RuntimeError("D-SMAL is unavailable; the DFA adapter must replace this backend")


smal_tpg.SMAL = _UnavailableSMAL
sys.modules.setdefault("smal", smal_package)
sys.modules.setdefault("smal.smal_tpg", smal_tpg)

import solver as gart_solver  # noqa: E402
from lib_data.data_provider import RealDataOptimizablePoseProviderPose  # noqa: E402
from lib_gart.templates import VoxelDeformer  # noqa: E402
from lib_render.gauspl_renderer import render_cam_pcl  # noqa: E402


PROTOCOL_ID = "DFA-Panda-Walk-32f-v1"
TRAIN_SPLIT = "train_mono_t32"
TEST_SPLIT = "test_novel_v8_t32"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _camera_matrix(observation: dict[str, Any]) -> np.ndarray:
    matrix = np.asarray(observation["world_to_camera"], dtype=np.float32)
    if matrix.shape != (4, 4):
        raise ValueError(f"Invalid world_to_camera shape {matrix.shape}")
    return matrix


def _scaled_intrinsics(observation: dict[str, Any], width: int, height: int) -> np.ndarray:
    source_width, source_height, fx, fy, cx, cy = map(float, observation["intrinsics"])
    scale_x = width / source_width
    scale_y = height / source_height
    return np.asarray(
        [[fx * scale_x, 0.0, cx * scale_x], [0.0, fy * scale_y, cy * scale_y], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


class DFAProtocol:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.protocol = _load_json(self.root / "protocol.json")
        if self.protocol.get("protocol_id") != PROTOCOL_ID:
            raise ValueError(f"Expected {PROTOCOL_ID}, found {self.protocol.get('protocol_id')}")
        payload = np.load(self.root / "dfa_panda_walk_matrix_lbs_stage1.npz")
        self.vertices = payload["rest_surface_vertices"].astype(np.float32)
        self.faces = payload["surface_faces"].astype(np.int64)
        self.weights = payload["surface_weights"].astype(np.float32)
        self.bone_transforms = payload["bone_transforms"].astype(np.float32)
        self.rest_bones = self.bone_transforms[0]
        self.rest_bones_inverse = np.linalg.inv(self.rest_bones)
        if self.weights.shape != (len(self.vertices), self.bone_transforms.shape[1]):
            raise ValueError("DFA surface weights and bone transforms disagree")
        if not np.allclose(self.weights.sum(axis=1), 1.0, atol=2e-4):
            raise ValueError("DFA skinning weights do not sum to one")

    def manifest(self, split: str) -> dict[str, Any]:
        return _load_json(self.root / split / "camera_manifest.json")

    def relative_bones(self, motion_index: int) -> np.ndarray:
        return self.bone_transforms[motion_index] @ self.rest_bones_inverse

    def camera_bones(self, observation: dict[str, Any]) -> np.ndarray:
        relative = self.relative_bones(int(observation["motion_index"]))
        return _camera_matrix(observation)[None] @ relative

    def pose_tensor(self, observation: dict[str, Any]) -> np.ndarray:
        return self.camera_bones(observation)[:, :3, :4].reshape(-1, 12).astype(np.float32)

    def audit(self) -> dict[str, Any]:
        train = self.manifest(TRAIN_SPLIT)
        test = self.manifest(TEST_SPLIT)
        identity_error = float(
            np.abs(self.relative_bones(0) - np.eye(4, dtype=np.float32)[None]).max()
        )
        train_pairs = {(int(o["frame_index"]), int(o["view_index"])) for o in train["observations"]}
        test_pairs = {(int(o["frame_index"]), int(o["view_index"])) for o in test["observations"]}
        observation = train["observations"][min(7, len(train["observations"]) - 1)]
        count = min(512, len(self.vertices))
        homogeneous = np.concatenate(
            [self.vertices[:count], np.ones((count, 1), dtype=np.float32)], axis=-1
        )
        relative = self.relative_bones(int(observation["motion_index"]))
        per_bone_world = np.einsum("jab,nb->nja", relative, homogeneous)[..., :3]
        skinned_world = np.sum(self.weights[:count, :, None] * per_bone_world, axis=1)
        world_to_camera = _camera_matrix(observation)
        expected_camera = (
            world_to_camera[:3, :3] @ skinned_world.T + world_to_camera[:3, 3:4]
        ).T
        camera_transforms = self.camera_bones(observation)
        per_bone_camera = np.einsum("jab,nb->nja", camera_transforms, homogeneous)[..., :3]
        adapted_camera = np.sum(self.weights[:count, :, None] * per_bone_camera, axis=1)
        report = {
            "protocol": PROTOCOL_ID,
            "train_split": TRAIN_SPLIT,
            "test_split": TEST_SPLIT,
            "train_observations": len(train["observations"]),
            "test_observations": len(test["observations"]),
            "train_views": sorted({view for _, view in train_pairs}),
            "test_views": sorted({view for _, view in test_pairs}),
            "split_pair_overlap": len(train_pairs & test_pairs),
            "rest_relative_identity_max_error": identity_error,
            "vertex_count": int(len(self.vertices)),
            "face_count": int(len(self.faces)),
            "joint_count": int(self.weights.shape[1]),
            "weight_sum_max_error": float(np.abs(self.weights.sum(axis=1) - 1.0).max()),
            "camera_lbs_equivalence_max_error": float(
                np.abs(expected_camera - adapted_camera).max()
            ),
        }
        if report["train_observations"] != 32 or report["test_observations"] != 256:
            raise ValueError(f"Unexpected frozen split sizes: {report}")
        if report["split_pair_overlap"]:
            raise ValueError("Training and held-out observation pairs overlap")
        if report["camera_lbs_equivalence_max_error"] > 2e-5:
            raise ValueError(f"Camera/LBS composition audit failed: {report}")
        return report


class DFATemplate(nn.Module):
    """GART template interface backed by exact DFA matrix LBS."""

    def __init__(self, protocol: DFAProtocol, voxel_resolution: int, device: torch.device) -> None:
        super().__init__()
        self.dim = int(protocol.weights.shape[1])
        self.name = "dfa_panda"
        self.cano_pose_type = "dfa_rest"
        vertices = torch.from_numpy(protocol.vertices).float().to(device)
        faces = torch.from_numpy(protocol.faces).long().to(device)
        weights = torch.from_numpy(protocol.weights).float().to(device)
        self.register_buffer("vertices", vertices)
        self.register_buffer("faces", faces)
        identity = torch.eye(4, dtype=torch.float32, device=device)[None].repeat(self.dim, 1, 1)
        self.register_buffer("canonical_pose", identity[:, :3, :4].reshape(self.dim, 12))
        self.voxel_deformer = VoxelDeformer(
            vtx=vertices[None],
            vtx_features=weights[None],
            resolution_dhw=[voxel_resolution, voxel_resolution // 2, voxel_resolution],
            short_dim_dhw=1,
            long_dim_dhw=2,
        )

    @torch.no_grad()
    def get_init_vf(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.vertices, self.faces

    def get_rot_action(self, axis_angle: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("DFA adapter has no additional bones; get_rot_action must not be called")

    def forward(
        self, theta: torch.Tensor | None, xyz_canonical: torch.Tensor | None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if theta is None:
            transforms = None
        else:
            if theta.ndim != 3 or theta.shape[1:] != (self.dim, 12):
                raise ValueError(f"Expected pose [B,{self.dim},12], found {tuple(theta.shape)}")
            upper = theta.reshape(len(theta), self.dim, 3, 4)
            bottom = torch.zeros(
                len(theta), self.dim, 1, 4, dtype=upper.dtype, device=upper.device
            )
            bottom[..., 0, 3] = 1.0
            transforms = torch.cat([upper, bottom], dim=-2)
        weights = None if xyz_canonical is None else self.voxel_deformer(xyz_canonical)
        return weights, transforms


class DFADataset:
    def __init__(self, protocol: DFAProtocol, width: int, height: int) -> None:
        self.protocol = protocol
        self.width = width
        self.height = height
        self.split_root = protocol.root / TRAIN_SPLIT
        self.observations = protocol.manifest(TRAIN_SPLIT)["observations"]

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, index: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        observation = self.observations[index]
        image_path = self.split_root / "images" / observation["image"]
        rgba = np.asarray(
            Image.open(image_path).convert("RGBA").resize((self.width, self.height), Image.Resampling.LANCZOS),
            dtype=np.float32,
        ) / 255.0
        sample = {
            "smpl_beta": np.zeros(1, dtype=np.float32),
            "smpl_pose": self.protocol.pose_tensor(observation),
            "smpl_trans": np.zeros(3, dtype=np.float32),
            "K": _scaled_intrinsics(observation, self.width, self.height),
            "rgb": rgba[..., :3],
            "mask": rgba[..., 3],
        }
        return sample, observation


class DFAFitter(gart_solver.TGFitter):
    def __init__(self, *args: Any, protocol: DFAProtocol, **kwargs: Any) -> None:
        self.dfa_protocol = protocol
        super().__init__(*args, **kwargs)

    def _get_model_optimizer(self, betas: Any, add_bones_total_t: int = 0):
        original_get_template = gart_solver.get_template

        def create_template(**_: Any) -> DFATemplate:
            return DFATemplate(
                self.dfa_protocol,
                voxel_resolution=int(getattr(self, "VOXEL_DEFORMER_RES", 64)),
                # Match upstream GART: initialize the mesh/voxel template on
                # CPU, then GaussianTemplateModel.to(device) moves the full
                # module.  Trimesh initialization cannot consume CUDA faces.
                device=torch.device("cpu"),
            )

        gart_solver.get_template = create_template
        try:
            return super()._get_model_optimizer(betas, add_bones_total_t)
        finally:
            gart_solver.get_template = original_get_template


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _make_fitter(protocol: DFAProtocol, output: Path) -> DFAFitter:
    return DFAFitter(
        log_dir=str(output),
        profile_fn=str(ARGS.profile.resolve()),
        mode="dog",
        template_model_path=None,
        device=torch.device(ARGS.device),
        FAST_TRAINING=True,
        protocol=protocol,
    )


def train(protocol: DFAProtocol) -> None:
    output = ARGS.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    audit = protocol.audit()
    (output / "protocol_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    dataset = DFADataset(protocol, ARGS.width, ARGS.height)
    provider = RealDataOptimizablePoseProviderPose(dataset, balance=False)
    fitter = _make_fitter(protocol, output)
    provider.to(fitter.device)
    if bool(getattr(fitter, "DATA_STAY_GPU_FLAG", False)):
        provider.move_images_to_device(fitter.device)
    started = time.time()
    fitter.run(real_data_provider=provider)
    elapsed = time.time() - started
    report = {
        "schema": "gart-dfa-training-v1",
        "status": "complete",
        "method": "GART-DFA adapter",
        "upstream": "GART",
        "upstream_commit": _git_commit(GART_ROOT),
        "protocol": PROTOCOL_ID,
        "fit_split": TRAIN_SPLIT,
        "input_regime": "monocular_dynamic_video",
        "training_images": len(dataset),
        "render_size": [ARGS.width, ARGS.height],
        "steps": int(fitter.TOTAL_steps),
        "elapsed_seconds": elapsed,
        "checkpoint": str(output / "model.pth"),
        "adapter_boundary": "DFA mesh/93-bone matrix-LBS replaces licensed D-SMAL/BITE input only",
    }
    (output / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


@torch.no_grad()
def render(protocol: DFAProtocol) -> None:
    output = ARGS.output_dir.resolve()
    checkpoint = output / "model.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Train GART first; missing {checkpoint}")
    render_root = (ARGS.render_dir or (output / "heldout_v8_t32_render")).resolve()
    arrays = render_root / "arrays"
    previews = render_root / "previews"
    arrays.mkdir(parents=True, exist_ok=True)
    previews.mkdir(parents=True, exist_ok=True)
    fitter = _make_fitter(protocol, output)
    model = fitter.load_saved_model(str(checkpoint))
    model.cache_for_fast()
    manifest = protocol.manifest(TEST_SPLIT)
    observations = manifest["observations"]
    if ARGS.max_observations is not None:
        observations = observations[: ARGS.max_observations]
    rendered: list[dict[str, Any]] = []
    started = time.time()
    for index, observation in enumerate(observations):
        pose = torch.from_numpy(protocol.pose_tensor(observation))[None].to(fitter.device)
        translation = torch.zeros(1, 3, dtype=torch.float32, device=fitter.device)
        mu, frame, scale, opacity, sph, _ = model(
            pose,
            translation,
            additional_dict={},
            active_sph_order=int(fitter.MAX_SPH_ORDER),
            fast=True,
        )
        intrinsics = torch.from_numpy(
            _scaled_intrinsics(observation, ARGS.width, ARGS.height)
        ).to(fitter.device)
        package = render_cam_pcl(
            mu[0],
            frame[0],
            scale[0],
            opacity[0],
            sph[0],
            ARGS.height,
            ARGS.width,
            intrinsics,
            False,
            int(fitter.MAX_SPH_ORDER),
            np.zeros(3, dtype=np.float32),
        )
        rgb = package["rgb"].permute(1, 2, 0).clamp(0.0, 1.0).cpu().numpy()
        alpha = package["alpha"].squeeze(0).clamp(0.0, 1.0).cpu().numpy()
        stem = Path(observation["image"]).stem
        array_path = arrays / f"{stem}.npz"
        np.savez_compressed(array_path, rgb=rgb.astype(np.float16), mask=alpha.astype(np.float16))
        if index % max(1, len(observations) // 8) == 0:
            imageio.imwrite(previews / f"{stem}_rgb.png", np.round(rgb * 255).astype(np.uint8))
            imageio.imwrite(previews / f"{stem}_alpha.png", np.round(alpha * 255).astype(np.uint8))
        rendered.append(
            {
                "image": observation["image"],
                "frame_index": int(observation["frame_index"]),
                "view_index": int(observation["view_index"]),
                "array": str(array_path),
            }
        )
        if (index + 1) % 16 == 0 or index + 1 == len(observations):
            print(f"Rendered {index + 1}/{len(observations)}", flush=True)
    report = {
        "schema": "external-render-manifest-v1",
        "status": "complete" if len(rendered) == int(manifest["image_count"]) else "smoke",
        "method": "GART-DFA adapter",
        "protocol": PROTOCOL_ID,
        "fit_split": TRAIN_SPLIT,
        "test_split": TEST_SPLIT,
        "render_size": [ARGS.width, ARGS.height],
        "image_count": len(rendered),
        "elapsed_seconds": time.time() - started,
        "observations": rendered,
    }
    path = render_root / "render_manifest.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"render_manifest": str(path), "image_count": len(rendered)}, indent=2))


def main() -> None:
    os.chdir(GART_ROOT)
    protocol = DFAProtocol(ARGS.protocol_root)
    if ARGS.mode == "validate":
        print(json.dumps(protocol.audit(), indent=2))
    elif ARGS.mode == "smoke-template":
        template = DFATemplate(protocol, voxel_resolution=8, device=torch.device("cpu"))
        observation = protocol.manifest(TRAIN_SPLIT)["observations"][0]
        pose = torch.from_numpy(protocol.pose_tensor(observation))[None]
        query = torch.from_numpy(protocol.vertices[:128])[None]
        weights, transforms = template(pose, query)
        report = {
            "weights_shape": list(weights.shape),
            "transforms_shape": list(transforms.shape),
            "queried_weight_sum_max_error": float((weights.sum(dim=-1) - 1.0).abs().max()),
            "finite": bool(torch.isfinite(weights).all() and torch.isfinite(transforms).all()),
        }
        if not report["finite"] or report["queried_weight_sum_max_error"] > 1e-3:
            raise RuntimeError(f"DFA template smoke failed: {report}")
        print(json.dumps(report, indent=2))
    elif ARGS.mode == "train":
        train(protocol)
    else:
        render(protocol)


if __name__ == "__main__":
    main()
