#!/usr/bin/env python3
"""Prepare person0 frame 0049 for Im2Haircut using its calibrated scaffold.

This is deliberately labelled scaffold-conditioned, not pure single-view:
appearance/orientation/depth come from only frame 0049, while the released
person0 FLAME scalp and calibrated camera resolve the subject-frame ambiguity.
"""

import argparse
from pathlib import Path

import cv2
import face_alignment
import numpy as np
from PIL import Image


def resize(array: np.ndarray, interpolation: int) -> np.ndarray:
    return cv2.resize(array, (512, 512), interpolation=interpolation)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--im2-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--transform", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=49)
    parser.add_argument("--folder", default="person0_scaffold")
    args = parser.parse_args()

    name = f"{args.frame:04d}.png"
    output = args.im2_root / "data" / args.folder
    rgba = np.asarray(Image.open(args.protocol / "protocol/train/images" / name).convert("RGBA"))
    rgb = rgba[..., :3]
    hair = rgba[..., 3]
    body = np.asarray(Image.open(args.protocol / "masks_2/body" / name).convert("L"))
    orient = np.asarray(Image.open(args.protocol / "orientations_2/angles" / name).convert("L"))
    depth_file = args.im2_root / "data/person0_singleview/depth_apple_pro" / name.replace(".png", ".npz")
    depth = np.load(depth_file)["depth"].astype(np.float32)

    rgb = resize(rgb, cv2.INTER_AREA)
    hair = resize(hair, cv2.INTER_NEAREST)
    body = resize(body, cv2.INTER_NEAREST)
    orient = resize(orient, cv2.INTER_LINEAR)
    depth = resize(depth, cv2.INTER_LINEAR)

    try:
        landmark_type = face_alignment.LandmarksType.TWO_D
    except AttributeError:
        landmark_type = face_alignment.LandmarksType._2D
    detector = face_alignment.FaceAlignment(landmark_type, flip_input=False, device="cuda")
    target = np.asarray(Image.open(args.im2_root / "data/aligned_image.png").convert("RGB").resize((512, 512)))
    source_landmarks = detector.get_landmarks_from_image(rgb)[0]
    target_landmarks = detector.get_landmarks_from_image(target)[0]
    affine, _ = cv2.estimateAffinePartial2D(source_landmarks, target_landmarks)
    if affine is None:
        raise RuntimeError("Could not estimate person0-to-Im2Haircut image alignment")

    def warp(array: np.ndarray, interpolation: int) -> np.ndarray:
        return cv2.warpAffine(array, affine, (512, 512), flags=interpolation)

    rgb_a = warp(rgb, cv2.INTER_LINEAR)
    hair_a = warp(hair, cv2.INTER_NEAREST)
    body_a = warp(body, cv2.INTER_NEAREST)
    orient_a = warp(orient, cv2.INTER_LINEAR)
    depth_a = warp(depth, cv2.INTER_LINEAR)

    # Gabor angles are pi-periodic.  This deterministic lift provides the two
    # direction channels expected by the released Im2Haircut dataset.  The
    # network may flip a strand by pi without changing its geometry.
    theta = orient_a.astype(np.float32) / 255.0 * np.pi
    mask = (hair_a.astype(np.float32) / 255.0)[..., None]
    strand = np.concatenate(
        [mask, (0.5 + 0.5 * np.cos(theta))[..., None] * mask, (0.5 + 0.5 * np.sin(theta))[..., None] * mask],
        axis=-1,
    )

    payload = {
        "resized_img": rgb,
        "seg": hair,
        "body_img": body,
        "orientation_maps": orient,
        "strand_map": resize((strand * 255).astype(np.uint8), cv2.INTER_LINEAR),
        "resized_img_aligned": rgb_a,
        "seg_aligned": hair_a,
        "body_img_aligned": body_a,
        "orientation_maps_aligned": orient_a,
        "strand_map_aligned": (strand * 255).astype(np.uint8),
    }
    for folder, array in payload.items():
        destination = output / folder / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(array).save(destination)
    for folder, array in (("depth_apple_pro", depth), ("depth_apple_pro_aligned", depth_a)):
        destination = output / folder / name.replace(".png", ".npz")
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, depth=array)

    cameras = np.load(args.protocol / "cameras.npz")["arr_0"]
    person_camera = cameras[args.frame, :3]
    head = np.load(args.protocol / "static_head_stage1.npz")
    head_frame = np.eye(4)
    head_frame[:3, :3] *= float(head["head_frame_scale"])
    head_frame[:3, 3] = head["head_frame_translation"]
    canonical_camera = person_camera @ head_frame @ np.loadtxt(args.transform)
    image_scale = np.diag([0.25, 0.25, 1.0])
    aligned_image = np.eye(3)
    aligned_image[:2] = affine
    camera_512 = image_scale @ canonical_camera
    camera_aligned = aligned_image @ camera_512
    for folder, camera in (("proj_matx_inv", camera_512), ("proj_matx_inv_aligned", camera_aligned)):
        destination = output / folder / name.replace(".png", ".txt")
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(destination, camera)

    np.save(output / "alignment_affine.npy", affine)
    (output / "PROTOCOL.txt").write_text(
        "Im2Haircut person0 frame 0049; single-image appearance conditioned on released calibrated person0 FLAME scaffold.\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
