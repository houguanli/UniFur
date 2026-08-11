#!/usr/bin/env python3
"""Run official 4D-Animal optimization and rendering on DFA-Panda-Walk.

This adapter supplies the official optimizer with the frozen protocol's RGB,
alpha, time index, and calibrated cameras.  CSE/PartGLEE/BITE/BootsTAP terms
are disabled because those predictions do not exist for DFA; the released
SMAL, free-form vertex offsets, ARAP, silhouette/chamfer, color, pose MLP, and
Duplex shell texture components remain upstream 4D-Animal code.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as pth_transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("validate", "precompute", "train", "render"), required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--render-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--max-observations", type=int)
    return parser.parse_args()


ARGS = parse_args()
REPO_ROOT = ARGS.repo_root.resolve()
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from data.input_cop import InputCop  # noqa: E402
from model.inferencer import Inferencer  # noqa: E402
from model.texture_models.model_utils import render_images_wrapper  # noqa: E402
from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection  # noqa: E402
from rendering.renderer import Renderer  # noqa: E402


PROTOCOL_ID = "DFA-Panda-Walk-32f-v1"
TRAIN_SPLIT = "train_mono_t32"
TEST_SPLIT = "test_novel_v8_t32"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


class DFAProtocol:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.protocol = _load_json(self.root / "protocol.json")
        if self.protocol.get("protocol_id") != PROTOCOL_ID:
            raise ValueError("Unexpected protocol")

    def manifest(self, split: str) -> dict[str, Any]:
        return _load_json(self.root / split / "camera_manifest.json")

    def observations(self, split: str) -> list[dict[str, Any]]:
        return self.manifest(split)["observations"]

    def image(self, split: str, observation: dict[str, Any]) -> np.ndarray:
        return np.asarray(
            Image.open(self.root / split / "images" / observation["image"]).convert("RGBA"),
            dtype=np.float32,
        ) / 255.0


def _scaled_k(observation: dict[str, Any], width: int, height: int) -> np.ndarray:
    source_width, source_height, fx, fy, cx, cy = map(float, observation["intrinsics"])
    sx, sy = width / source_width, height / source_height
    return np.asarray(
        [[fx * sx, 0.0, cx * sx], [0.0, fy * sy, cy * sy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _pytorch3d_cameras(
    observations: list[dict[str, Any]], width: int, height: int, device: str
):
    world_to_camera = np.stack(
        [np.asarray(observation["world_to_camera"], dtype=np.float32) for observation in observations]
    )
    rotation = torch.from_numpy(world_to_camera[:, :3, :3])
    translation = torch.from_numpy(world_to_camera[:, :3, 3])
    intrinsics = torch.from_numpy(np.stack([_scaled_k(o, width, height) for o in observations]))
    image_size = torch.tensor([[height, width]], dtype=torch.float32).repeat(len(observations), 1)
    return cameras_from_opencv_projection(
        R=rotation, tvec=translation, camera_matrix=intrinsics, image_size=image_size
    ).to(device)


class DFADataset:
    def __init__(self, protocol: DFAProtocol, cache_dir: Path, device: str) -> None:
        self.protocol = protocol
        self.observations = protocol.observations(TRAIN_SPLIT)
        self.cache_dir = cache_dir.resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

    def __len__(self) -> int:
        return len(self.observations)

    def get_imgs_rgb(self, indices: list[int]) -> np.ndarray:
        return np.stack([self.protocol.image(TRAIN_SPLIT, self.observations[i])[..., :3] for i in indices])

    def get_masks(self, indices: list[int]) -> np.ndarray:
        return np.stack(
            [self.protocol.image(TRAIN_SPLIT, self.observations[i])[..., 3:4] for i in indices]
        )

    def get_dino_feature(self, indices: list[int]) -> np.ndarray:
        cache = self.cache_dir / "dino_vits8_480_features.npy"
        if not cache.is_file():
            self._precompute_dino(cache)
        features = np.load(cache, mmap_mode="r")
        return np.asarray(features[indices], dtype=np.float32)

    @torch.no_grad()
    def _precompute_dino(self, cache: Path) -> None:
        model = torch.hub.load("facebookresearch/dino:main", "dino_vits8")
        model.eval().to(self.device)
        transform = pth_transforms.Compose(
            [
                pth_transforms.Resize((480, 480)),
                pth_transforms.ToTensor(),
                pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        features = []
        for index, observation in enumerate(self.observations):
            image = Image.fromarray(
                np.round(self.protocol.image(TRAIN_SPLIT, observation)[..., :3] * 255).astype(np.uint8)
            )
            tensor = transform(image)[None].to(self.device)
            output = model.get_intermediate_layers(tensor, n=1)[0][:, 1:, :][0]
            output = (output - output.min()) / (output.max() - output.min()).clamp_min(1e-8)
            features.append(output.cpu().numpy().astype(np.float32))
            print(f"DINO {index + 1}/{len(self.observations)}", flush=True)
        np.save(cache, np.stack(features))


class DFAInputCop(InputCop):
    def __init__(self, protocol: DFAProtocol, output: Path, device: str) -> None:
        super().__init__(
            sequence_index="dfa-panda-walk-mono",
            dataset_source="CUSTOM",
            cse_mesh_name="smal",
            frame_limit=32,
            image_size=256,
            train_test_split=(32, 0),
            N_cse_kps=1000,
            filter_cse_kps=False,
            moving_camera=True,
            cse_version="original",
            device=device,
            category="dog",
        )
        self.protocol = protocol
        self.output = output

    @property
    def dataset(self) -> DFADataset:
        if getattr(self, "_dataset", None) is None:
            self._dataset = DFADataset(self.protocol, self.output / "preprocess", self.device)
        return self._dataset

    @property
    def cameras(self):
        if getattr(self, "_cameras", None) is None:
            self._cameras = _pytorch3d_cameras(
                self.protocol.observations(TRAIN_SPLIT), self.image_size, self.image_size, "cpu"
            )
        return self._cameras

    @property
    def cameras_original(self):
        return self.cameras

    @property
    def cameras_canonical(self):
        return self.cameras


def _build_input(_: Any, device: str = "cuda") -> DFAInputCop:
    return DFAInputCop(PROTOCOL, ARGS.output_dir.resolve(), device)


def _audit(protocol: DFAProtocol) -> dict[str, Any]:
    train = protocol.observations(TRAIN_SPLIT)
    test = protocol.observations(TEST_SPLIT)
    train_pairs = {(int(o["frame_index"]), int(o["view_index"])) for o in train}
    test_pairs = {(int(o["frame_index"]), int(o["view_index"])) for o in test}
    cameras = _pytorch3d_cameras(train[:1], 512, 288, "cpu")
    stage1 = np.load(protocol.root / "dfa_panda_walk_matrix_lbs_stage1.npz")
    points = torch.from_numpy(stage1["rest_surface_vertices"][:256])[None]
    projected = cameras.transform_points_screen(
        points, image_size=torch.tensor([[288.0, 512.0]])
    )[0, :, :2]
    transform = np.asarray(train[0]["world_to_camera"], dtype=np.float32)
    camera_points = (transform[:3, :3] @ points[0].numpy().T + transform[:3, 3:4]).T
    intrinsics = _scaled_k(train[0], 512, 288)
    manual = np.stack(
        [
            intrinsics[0, 0] * camera_points[:, 0] / camera_points[:, 2] + intrinsics[0, 2],
            intrinsics[1, 1] * camera_points[:, 1] / camera_points[:, 2] + intrinsics[1, 2],
        ],
        axis=-1,
    )
    report = {
        "protocol": PROTOCOL_ID,
        "fit_split": TRAIN_SPLIT,
        "test_split": TEST_SPLIT,
        "train_observations": len(train),
        "test_observations": len(test),
        "train_views": sorted({view for _, view in train_pairs}),
        "test_views": sorted({view for _, view in test_pairs}),
        "split_pair_overlap": len(train_pairs & test_pairs),
        "opencv_to_pytorch3d_projection_max_error_pixels": float(
            np.abs(projected.numpy() - manual).max()
        ),
    }
    if len(train) != 32 or len(test) != 256 or train_pairs & test_pairs:
        raise RuntimeError(f"Frozen split audit failed: {report}")
    if report["opencv_to_pytorch3d_projection_max_error_pixels"] > 1e-2:
        raise RuntimeError(f"Camera conversion audit failed: {report}")
    return report


def train(protocol: DFAProtocol) -> None:
    import data.input_cop as input_module

    input_module.get_input_cop_from_cfg = _build_input
    import main_optimize_scene as upstream

    upstream.get_input_cop_from_cfg = _build_input
    output = ARGS.output_dir.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    sys.argv = [
        "main_optimize_scene.py",
        'exp.sequence_index="dfa-panda-walk-mono"',
        f'exp.experiment_folder="{output.parent}"',
        'exp.dataset_source="CUSTOM"',
        "exp.frame_limit=32",
        "exp.train_split=32",
        "exp.test_split=0",
        "exp.image_size=256",
        "exp.n_steps=10000",
        "exp.l_optim_tracking=0",
        "exp.l_optim_cse_kp=0",
        "exp.l_optim_part_kp=0",
        "exp.l_optim_part_chamfer=0",
        "exp.l_optim_sparse_kp=0",
        "exp.moving_camera=true",
    ]
    started = time.time()
    upstream.main_train()
    report = {
        "schema": "fourdanimal-dfa-training-v1",
        "status": "complete",
        "method": "4D-Animal-DFA adapter",
        "protocol": PROTOCOL_ID,
        "fit_split": TRAIN_SPLIT,
        "training_images": 32,
        "steps": 10000,
        "elapsed_seconds": time.time() - started,
        "disabled_unavailable_terms": ["CSE", "PartGLEE parts", "BITE keypoints", "BootsTAP tracks"],
        "kept_upstream_components": [
            "SMAL", "free-form vertex offsets", "ARAP", "silhouette/chamfer", "pose MLP", "Duplex shell texture"
        ],
    }
    (output / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


@torch.no_grad()
def render(protocol: DFAProtocol) -> None:
    output = ARGS.output_dir.resolve()
    if not (output / "checkpoints").is_dir():
        raise FileNotFoundError(f"Missing trained 4D-Animal archive: {output}")
    input_cop = DFAInputCop(protocol, output, ARGS.device)
    dino = input_cop.dino_feature.to(ARGS.device)
    smal = input_cop.smal
    inferencer = Inferencer(str(output), use_archived_code=False)
    pose_model = inferencer.load_pose_model().to(ARGS.device).eval()
    texture_model = inferencer.load_texture_model().to(ARGS.device).eval()
    renderer = Renderer((ARGS.height, ARGS.width))
    test = protocol.observations(TEST_SPLIT)
    if ARGS.max_observations is not None:
        test = test[: ARGS.max_observations]
    render_root = (ARGS.render_dir or (output / "heldout_v8_t32_render")).resolve()
    arrays = render_root / "arrays"
    previews = render_root / "previews"
    arrays.mkdir(parents=True, exist_ok=True)
    previews.mkdir(parents=True, exist_ok=True)
    observations = []
    started = time.time()
    for index, observation in enumerate(test):
        frame = int(observation["frame_index"])
        x_ind = torch.tensor([frame], dtype=torch.long, device=ARGS.device)
        x_ts = x_ind.float() / 32.0
        camera = _pytorch3d_cameras([observation], ARGS.width, ARGS.height, ARGS.device)
        rgb, alpha = render_images_wrapper(
            texture_model,
            smal,
            renderer,
            pose_model,
            x_ind,
            x_ts,
            dino[[frame]],
            camera,
        )
        rgb_np = rgb[0].clamp(0.0, 1.0).cpu().numpy()
        alpha_np = alpha[0, ..., 0].clamp(0.0, 1.0).cpu().numpy()
        stem = Path(observation["image"]).stem
        array_path = arrays / f"{stem}.npz"
        np.savez_compressed(array_path, rgb=rgb_np.astype(np.float16), mask=alpha_np.astype(np.float16))
        if index % max(1, len(test) // 8) == 0:
            imageio.imwrite(previews / f"{stem}_rgb.png", np.round(rgb_np * 255).astype(np.uint8))
            imageio.imwrite(previews / f"{stem}_alpha.png", np.round(alpha_np * 255).astype(np.uint8))
        observations.append(
            {
                "image": observation["image"],
                "frame_index": frame,
                "view_index": int(observation["view_index"]),
                "array": str(array_path),
            }
        )
        if (index + 1) % 16 == 0 or index + 1 == len(test):
            print(f"Rendered {index + 1}/{len(test)}", flush=True)
    manifest = {
        "schema": "external-render-manifest-v1",
        "status": "complete" if len(observations) == 256 else "smoke",
        "method": "4D-Animal-DFA adapter",
        "protocol": PROTOCOL_ID,
        "fit_split": TRAIN_SPLIT,
        "test_split": TEST_SPLIT,
        "render_size": [ARGS.width, ARGS.height],
        "image_count": len(observations),
        "elapsed_seconds": time.time() - started,
        "observations": observations,
    }
    path = render_root / "render_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"render_manifest": str(path), "image_count": len(observations)}, indent=2))


PROTOCOL = DFAProtocol(ARGS.protocol_root)


def main() -> None:
    if ARGS.mode == "validate":
        print(json.dumps(_audit(PROTOCOL), indent=2))
    elif ARGS.mode == "precompute":
        dataset = DFADataset(PROTOCOL, ARGS.output_dir / "preprocess", ARGS.device)
        features = dataset.get_dino_feature(list(range(len(dataset))))
        print(json.dumps({"dino_features": list(features.shape)}, indent=2))
    elif ARGS.mode == "train":
        train(PROTOCOL)
    else:
        render(PROTOCOL)


if __name__ == "__main__":
    main()
