"""CPU geometry checks. All observations must originate in train RGB matches."""
from pathlib import Path
import hashlib
import importlib.util

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

COLMAP = Path('/home/aoki/miniconda3/envs/im2haircut/bin/colmap')
MODEL_IO = Path('/home/aoki/gaussian_splatting/utils/read_write_model.py')
CONTRACT = dict(train_views=48, side=2048, min_track=3, max_reprojection_px=2.,
                min_angle_degrees=1.5, initial_unique_points=25000, deployment_cap=239171,
                body_probability=.5, refine_poses=False, refine_intrinsics=False,
                coverage_grid=32, coverage_radius_px=8, mask_threshold=.5,
                native_foreground_alpha_u8_gt=2,
                tail_roi_source='pre-registered train RGB polygons; no semantic GT',
                unknown_body_prior_not_segmentation=True)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def confined(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    require(not relative.is_absolute() and '..' not in relative.parts, 'nonrelative input path')
    path = (root / relative).resolve()
    require(root in path.parents, 'input escapes approved root')
    denied = ('test_v12', 'eval_only', 'dense_gt', 'position', 'normal', 'mask_body',
              'mask_hair', 'strand_gt', 'asset.blend', 'quality_v2')
    require(not any(token in str(path).lower() for token in denied), 'forbidden initialization input')
    return path


def model_io():
    spec = importlib.util.spec_from_file_location('_rgb_colmap_io', MODEL_IO)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pose(observation):
    matrix = np.asarray(observation['world_to_camera'], dtype=np.float64)
    require(matrix.shape == (4, 4) and np.isfinite(matrix).all(), 'invalid camera matrix')
    if observation.get('camera_convention') != 'opencv_colmap':
        require(observation.get('image_y_down') is True, 'undeclared source camera convention')
        matrix = np.diag([1., -1., -1., 1.]) @ matrix
    require(np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-9), 'invalid homogeneous pose')
    require(np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=2e-5), 'nonrigid camera')
    require(np.linalg.det(matrix[:3, :3]) > 0, 'reflected camera')
    # Same float32-matrix-to-SO(3) convention as the existing dataset converter.
    rotation = Rotation.from_matrix(matrix[:3, :3])
    xyzw = rotation.as_quat()
    q = np.r_[xyzw[3], xyzw[:3]]
    if q[0] < 0:
        q = -q
    return rotation.as_matrix(), matrix[:3, 3], q


def project(xyz, image, camera):
    xyz = np.asarray(xyz, dtype=np.float64)
    local = xyz @ image.qvec2rotmat().T + image.tvec
    depth = local[..., 2]
    fx, fy, cx, cy = camera.params
    with np.errstate(divide='ignore', invalid='ignore'):
        pixel = local[..., :2] / depth[..., None] * [fx, fy] + [cx, cy]
    return pixel, depth


def audit_poses(observations, cameras, images):
    expected = {Path(o['image']).name: o for o in observations}
    require(len(expected) == len(images) and {i.name for i in images.values()} == set(expected), 'camera basename mismatch')
    errors = []
    for image in images.values():
        obs = expected[image.name]
        rotation, translation, _ = pose(obs)
        camera = cameras[image.camera_id]
        values = np.asarray(obs['intrinsics'], dtype=float)
        require(camera.model == 'PINHOLE' and [camera.width, camera.height] == values[:2].tolist(), 'intrinsics model/size mismatch')
        er = float(np.max(np.abs(image.qvec2rotmat() - rotation)))
        et = float(np.max(np.abs(image.tvec - translation)))
        ek = float(np.max(np.abs(camera.params - values[2:])))
        require(er <= 1e-9 and et <= 1e-8 and ek <= 1e-8, 'fixed camera changed numerically')
        errors.append([er, et, ek])
    return dict(max_rotation_translation_intrinsics_error=np.max(errors, axis=0).tolist(),
                tolerances=[1e-9, 1e-8, 1e-8], source_raw_hash_separately_required=True)


def local_view_direction(vectors):
    """Use one actual track hemisphere; never average opposing ring cameras."""
    vectors = np.asarray(vectors, float)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    agreement = vectors @ vectors.T
    # Deterministic medoid of a locally agreeing subset; ties follow image-id order.
    chosen = int(np.argmax(np.sum(np.maximum(agreement, 0.), axis=1)))
    selected = agreement[chosen] >= .5
    direction = vectors[selected].mean(axis=0)
    return direction / np.linalg.norm(direction)


def filter_sparse(cameras, images, points, masks):
    rows, colors, directions, view_ids, point_ids, tracks = [], [], [], [], [], []
    rejected = {}
    def reject(reason):
        rejected[reason] = rejected.get(reason, 0) + 1
    seen = set()
    for point_id, point in sorted(points.items()):
        xyz = np.asarray(point.xyz, dtype=np.float64)
        if not np.isfinite(xyz).all():
            reject('nonfinite'); continue
        pair = sorted(zip(point.image_ids, point.point2D_idxs))
        if len(pair) != len(set(int(i) for i, _ in pair)):
            reject('duplicate_camera_track'); continue
        if len({int(i) for i, _ in pair}) < CONTRACT['min_track']:
            reject('short_track'); continue
        centers, observations = [], []
        valid = True
        for image_id, index in pair:
            require(image_id in images, 'track contains unapproved image id')
            image = images[image_id]
            require(0 <= index < len(image.xys) and image.point3D_ids[index] == point_id, 'broken track/point index')
            camera = cameras[image.camera_id]
            pixel, depth = project(xyz, image, camera)
            match = image.xys[index]
            if (not np.isfinite(pixel).all() or depth <= 0 or np.linalg.norm(pixel-match) > 2.
                    or (match < 0).any() or match[0] >= camera.width or match[1] >= camera.height):
                valid = False; break
            col, row = np.floor(match).astype(int)  # COLMAP corner-origin pixel convention.
            if not masks[image.name][row, col]:
                valid = False; break
            centers.append(-image.qvec2rotmat().T @ image.tvec)
            observations.append((int(image_id), int(index)))
        if not valid:
            reject('depth_reprojection_or_mask'); continue
        vectors = np.asarray(centers) - xyz
        norms = np.linalg.norm(vectors, axis=1)
        if not (norms > 0).all():
            reject('camera_coincident'); continue
        unit = vectors / norms[:, None]
        # Supplementary rays near 180 degrees are just as degenerate as parallel rays.
        angle = np.degrees(np.arccos(np.clip(np.abs(unit @ unit.T), 0., 1.))).max()
        if angle < CONTRACT['min_angle_degrees']:
            reject('low_triangulation_angle'); continue
        key = tuple(xyz.astype(np.float32))
        if key in seen:
            reject('duplicate_float32_position'); continue
        seen.add(key)
        direction = local_view_direction(vectors)
        closest = int(np.argmax(unit @ direction))
        rows.append(xyz); colors.append(point.rgb); directions.append(direction)
        view_ids.append(observations[closest][0]); point_ids.append(int(point_id)); tracks.append(observations)
    require(rows, 'no valid RGB triangulated points; no fallback')
    pool = dict(points=np.asarray(rows, np.float32), colors=np.asarray(colors, np.uint8),
                view_direction=np.asarray(directions, np.float32), source_view=np.asarray(view_ids, np.int32),
                point3D_id=np.asarray(point_ids, np.int64), body_probability=np.full(len(rows), .5, np.float32))
    return pool, tracks, dict(input_points=len(points), accepted_unique_points=len(rows), rejected=rejected,
                             unknown_body_prior=.5, no_geometry_gt=True)


def coverage(mask, xy, roi=None):
    """Pre-registered all-view grid plus train-RGB-only tail polygon, not a metric gate."""
    mask = np.asarray(mask, bool)
    pixels = np.zeros(mask.shape, np.uint8)
    xy = np.asarray(xy, float).reshape(-1, 2)
    valid = np.isfinite(xy).all(1) & (xy >= 0).all(1) & (xy < [mask.shape[1], mask.shape[0]]).all(1)
    indices = np.floor(xy[valid]).astype(int)
    pixels[indices[:, 1], indices[:, 0]] = 1
    radius = CONTRACT['coverage_radius_px']
    footprint = cv2.dilate(pixels, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))) > 0
    selected = mask if roi is None else mask & np.asarray(roi, bool)
    grid = CONTRACT['coverage_grid']
    cells, covered = 0, 0
    for y in range(0, mask.shape[0], grid):
        for x in range(0, mask.shape[1], grid):
            foreground = selected[y:y+grid, x:x+grid]
            if foreground.any():
                cells += 1
                covered += int((foreground & footprint[y:y+grid, x:x+grid]).any())
    return dict(foreground_pixels=int(selected.sum()), cells=cells, covered_cells=covered,
                cell_coverage=covered/cells if cells else None,
                radius8_pixel_coverage=float((selected & footprint).sum()/selected.sum()) if selected.any() else None,
                semantic_claim='observed-match coverage only; not surface recall')
