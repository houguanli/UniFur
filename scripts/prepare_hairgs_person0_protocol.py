#!/usr/bin/env python3
"""Create Hair-GS COLMAP-text train/test datasets for person0 protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation


def _estimate_orientation_field(
    image: np.ndarray,
    kernel_size: int = 31,
    sigma: float = 2,
    lambda_: float = 3,
    gamma: float = 0.5,
    num_angles: int = 180,
) -> tuple[np.ndarray, np.ndarray]:
    """Hair-GS' released Gabor orientation/confidence estimator.

    This intentionally mirrors ``hair-gs/utils/vision.py`` instead of converting
    GaussianHaircut's orientation variance.  Hair-GS multiplies this confidence
    by a weight of 100, so substituting another confidence calibration makes the
    structural term dominate RGB and causes the strand graph to collapse.
    """

    def angdiff(angle1: np.ndarray, angle2: np.ndarray) -> np.ndarray:
        return np.pi / 2 - np.abs(np.abs(angle1 - angle2) - np.pi / 2)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    orientations = np.linspace(0, np.pi, num_angles)
    kernels = [
        cv2.getGaborKernel(
            (kernel_size, kernel_size), sigma, theta, lambda_, gamma, 0,
            ktype=cv2.CV_32F,
        )
        for theta in orientations
    ]
    responses = np.stack(
        [np.abs(cv2.filter2D(gray, -1, kernel)) for kernel in kernels], axis=2
    )
    max_response = np.argmax(responses, axis=2)
    orientation_field = orientations[max_response]
    orientation_field_repeated = np.repeat(
        orientation_field[:, :, np.newaxis], len(orientations), axis=2
    )
    orientations_mat = np.ones(
        (height, width, len(orientations)), dtype=orientations.dtype
    ) * orientations
    diff = angdiff(orientation_field_repeated, orientations_mat)
    variance = np.sum(diff * diff * responses, axis=2) / (
        np.sum(responses, axis=2) + 1e-7
    )
    has_variance = variance != 0
    confidence = np.ones(orientation_field.shape, dtype=np.float32)
    valid_confidence = 1 / (variance * variance)[has_variance]
    if valid_confidence.size:
        valid_confidence = valid_confidence / np.max(valid_confidence)
        confidence[has_variance] = valid_confidence
    return orientation_field, confidence

def _write_points(path: Path, stage1_npz: Path) -> None:
    """Match Hair-GS' native NeRSemble initialization: grey head vertices.

    Hair-GS initializes Stage I from the fitted FLAME head vertices, not from an
    already-densified 3DGS point cloud.  Using the latter starts person0 with
    hundreds of thousands of points and makes native densification explode.
    """
    with np.load(stage1_npz, allow_pickle=False) as stage1:
        xyz = stage1["rest_surface_vertices"].astype(np.float32)
    rgb = np.full((len(xyz), 3), 128, dtype=np.uint8)
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ]
    values = np.empty(len(xyz), dtype=dtype)
    values["x"], values["y"], values["z"] = xyz.T
    values["nx"] = values["ny"] = values["nz"] = 0.0
    values["red"], values["green"], values["blue"] = rgb.T
    PlyData([PlyElement.describe(values, "vertex")]).write(path)


def _write_geometry(
    output: Path, stage1_npz: Path, scalp_indices: np.ndarray
) -> None:
    (output / "sparse/0").mkdir(parents=True, exist_ok=True)
    _write_points(output / "sparse/0/points3D.ply", stage1_npz)
    with np.load(stage1_npz, allow_pickle=False) as stage1:
        vertices = stage1["rest_surface_vertices"].astype(np.float32)
    if scalp_indices.min(initial=0) < 0 or scalp_indices.max(initial=-1) >= len(vertices):
        raise ValueError(
            f"FLAME scalp mask is incompatible with {len(vertices)} head vertices"
        )
    np.savez_compressed(
        output / "head_reconstruction_data.npz",
        head_verts=vertices,
        scalp_verts=vertices[scalp_indices],
    )


def _prepare_split(
    source: Path,
    manifest_path: Path,
    output: Path,
    stage1_npz: Path,
    scalp_indices: np.ndarray,
    scalp_mask_sha256: str,
    body_mask_root: Path,
    estimate_orientation: bool,
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    observations = manifest["observations"]
    for relative in ("images", "masks", "orientations", "sparse/0"):
        (output / relative).mkdir(parents=True, exist_ok=True)

    camera_lines = ["# Camera list", "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]"]
    image_lines = ["# Image list", "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME"]
    confidence_means: list[float] = []
    foreground_leakage: list[float] = []
    for local_index, item in enumerate(observations, start=1):
        name = str(item["image"])
        rgba = Image.open(source / "images" / name).convert("RGBA")
        rgb = np.asarray(rgba.convert("RGB"), dtype=np.uint8)
        body_alpha_path = body_mask_root / name
        if not body_alpha_path.is_file():
            raise FileNotFoundError(f"missing body alpha for {name}: {body_alpha_path}")
        body_alpha = np.asarray(
            Image.open(body_alpha_path).convert("L"), dtype=np.float32
        ) / 255.0
        # Hair-GS' NeRSemble parser trains RGB only on an alpha-matted subject.
        # Keeping the room background forces head-initialized Gaussians to grow
        # across the entire image and destroys the later strand merge.
        masked_rgb = np.rint(rgb.astype(np.float32) * body_alpha[..., None]).astype(
            np.uint8
        )
        Image.fromarray(masked_rgb, mode="RGB").save(output / "images" / name)
        foreground_leakage.append(
            float(masked_rgb[body_alpha <= 0].max()) if np.any(body_alpha <= 0) else 0.0
        )

        # Hair-GS loads masks as bool; preserve an anti-aliased PNG would turn
        # every non-zero fringe pixel into foreground, so emit the native binary
        # hair-mask convention explicitly.
        hair_mask = np.asarray(rgba.getchannel("A"), dtype=np.uint8) >= 128
        Image.fromarray(hair_mask.astype(np.uint8) * 255, mode="L").save(
            output / "masks" / name
        )
        width, height, fx, fy, cx, cy = [float(v) for v in item["intrinsics"]]
        camera_lines.append(
            f"{local_index} PINHOLE {int(width)} {int(height)} "
            f"{fx:.12g} {fy:.12g} {cx:.12g} {cy:.12g}"
        )
        w2c = np.asarray(item["world_to_camera"], dtype=np.float64)
        quaternion_xyzw = Rotation.from_matrix(w2c[:3, :3]).as_quat()
        qx, qy, qz, qw = quaternion_xyzw
        tx, ty, tz = w2c[:3, 3]
        image_lines.extend(
            [
                f"{local_index} {qw:.12g} {qx:.12g} {qy:.12g} {qz:.12g} "
                f"{tx:.12g} {ty:.12g} {tz:.12g} {local_index} {name}",
                "",
            ]
        )
        if estimate_orientation:
            orientation_field, confidence = _estimate_orientation_field(masked_rgb)
            orientation_u8 = np.clip(
                orientation_field * 255.0 / np.pi, 0, 255
            ).astype(np.uint8)
            confidence_u8 = np.clip(confidence * 255.0, 0, 255).astype(np.uint8)
            # The released loss already selects the hair mask, but zeroing outside
            # it makes diagnostics unambiguous and prevents future consumers from
            # accidentally treating background confidence as supervision.
            orientation_u8[~hair_mask] = 0
            confidence_u8[~hair_mask] = 0
            Image.fromarray(orientation_u8, mode="L").save(
                output / "orientations" / f"{Path(name).stem}_orientation.png"
            )
            Image.fromarray(confidence_u8, mode="L").save(
                output / "orientations" / f"{Path(name).stem}_confidence.png"
            )
            if np.any(hair_mask):
                confidence_means.append(float(confidence[hair_mask].mean()))

    (output / "sparse/0/cameras.txt").write_text(
        "\n".join(camera_lines) + "\n", encoding="utf-8"
    )
    (output / "sparse/0/images.txt").write_text(
        "\n".join(image_lines) + "\n", encoding="utf-8"
    )
    (output / "sparse/0/points3D.txt").write_text(
        "# Empty: initialization is supplied by points3D.ply\n", encoding="utf-8"
    )
    _write_geometry(output, stage1_npz, scalp_indices)
    (output / "protocol_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    sanity = {
        "split": str(manifest.get("split", output.name)),
        "image_count": len(observations),
        "background_rgb_max": float(max(foreground_leakage, default=0.0)),
        "orientation_estimator": "Hair-GS released Gabor estimator"
        if estimate_orientation
        else "not generated for held-out split",
        "hair_confidence_mean": float(np.mean(confidence_means))
        if confidence_means
        else None,
        "scalp_vertex_source": "official FLAME_masks.pkl/scalp",
        "scalp_vertex_count": int(len(scalp_indices)),
        "scalp_mask_sha256": scalp_mask_sha256,
    }
    (output / "preprocess_sanity.json").write_text(
        json.dumps(sanity, indent=2), encoding="utf-8"
    )
    if sanity["background_rgb_max"] != 0:
        raise RuntimeError(f"background removal failed: {sanity}")
    print(json.dumps(sanity))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--stage1-npz", type=Path, required=True)
    parser.add_argument(
        "--flame-mask-path",
        type=Path,
        required=True,
        help="Official FLAME_masks.pkl; Hair-GS uses its `scalp` vertex indices",
    )
    parser.add_argument(
        "--body-mask-root",
        type=Path,
        required=True,
        help="GaussianHaircut body alpha folder used to remove RGB background",
    )
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument(
        "--geometry-only",
        action="store_true",
        help="Reuse already generated images/orientations and only refresh FLAME geometry",
    )
    args = parser.parse_args()
    with args.flame_mask_path.open("rb") as handle:
        flame_masks = pickle.load(handle, encoding="latin1")
    if "scalp" not in flame_masks:
        raise KeyError(f"missing `scalp` in {args.flame_mask_path}")
    scalp_indices = np.asarray(flame_masks["scalp"], dtype=np.int64)
    scalp_mask_sha256 = hashlib.sha256(args.flame_mask_path.read_bytes()).hexdigest()
    if args.geometry_only:
        for split in ("train", "test"):
            output = args.out_root / split
            if not (output / "protocol_manifest.json").is_file():
                raise FileNotFoundError(
                    f"cannot use --geometry-only before full preprocessing: {output}"
                )
            _write_geometry(output, args.stage1_npz, scalp_indices)
            sanity_path = output / "preprocess_sanity.json"
            sanity = json.loads(sanity_path.read_text(encoding="utf-8"))
            sanity.update(
                scalp_vertex_source="official FLAME_masks.pkl/scalp",
                scalp_vertex_count=int(len(scalp_indices)),
                scalp_mask_sha256=scalp_mask_sha256,
            )
            sanity_path.write_text(json.dumps(sanity, indent=2), encoding="utf-8")
        print(args.out_root)
        return
    for split in ("train", "test"):
        source = args.protocol_root / "protocol" / split
        _prepare_split(
            source,
            source / "camera_manifest.json",
            args.out_root / split,
            args.stage1_npz,
            scalp_indices,
            scalp_mask_sha256,
            args.body_mask_root,
            estimate_orientation=split == "train",
        )
    print(args.out_root)


if __name__ == "__main__":
    main()
