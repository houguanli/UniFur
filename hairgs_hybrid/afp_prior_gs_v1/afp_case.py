"""AnimalFurPrior input cases from strict-SMAL A025 TRAIN views only.

The AFP contract wants the body bbox-centred and divided by its own bbox
diagonal D, OpenCV w2c with pixel-centre K, and straight RGBA PNGs. Heldout
(test) cameras and pixels never enter a case. The pilot case also omits the
four pilot-validation images, so its prior cannot see validation pixels.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .common import BODY_SHA256, DATA, EXPERIMENT, VALIDATION_VIEWS, sha, train_rows, write


def outward_vertex_normals(vertices, faces):
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    volume = np.einsum('ij,ij->i', tri[:, 0], cross).sum() / 6
    assert abs(volume) > 1e-9, 'open or degenerate body'
    cross *= 1 if volume > 0 else -1
    normals = np.zeros_like(vertices)
    for j in range(3):
        np.add.at(normals, faces[:, j], cross)  # area weighted
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    assert length.min() > 0, 'isolated body vertex'
    return normals / length


def body_frame():
    assert sha(DATA / 'gt/body_smal.npz') == BODY_SHA256
    with np.load(DATA / 'gt/body_smal.npz', allow_pickle=False) as body:
        vertices, faces = body['vertices'].astype(np.float64), body['faces'].astype(np.int64)
    low, high = vertices.min(0), vertices.max(0)
    return vertices, faces, (low + high) / 2, float(np.linalg.norm(high - low))


def build(output):
    output = Path(output)
    assert not output.exists(), 'case output must be new'
    vertices, faces, center, diagonal = body_frame()
    normals = outward_vertex_normals(vertices, faces)
    rows = train_rows()
    K, w2c = [], []
    for row in rows:
        k = np.asarray(row['K'], np.float64).copy()
        k[:2, 2] -= .5  # dataset K is corner-origin; AFP projects to pixel centres
        w = np.diag([1., -1., -1., 1.]) @ np.asarray(row['world_to_camera'], np.float64)
        w[:3, 3] = (w[:3, :3] @ center + w[:3, 3]) / diagonal
        K.append(k); w2c.append(w)
    view_ids = [row['view_id'] for row in rows]
    validation = [view_ids.index(v) for v in VALIDATION_VIEWS]
    pngs = {}
    for name, train in [('final28', list(range(28))), ('pilot24', [i for i in range(28) if i not in validation])]:
        case = output / name
        (case / 'images').mkdir(parents=True); (case / 'masks').mkdir()
        np.savez(case / 'body.npz', vertices=((vertices - center) / diagonal).astype(np.float32),
                 triangles=faces, vertex_normals=normals.astype(np.float32), center=center, diagonal=np.float64(diagonal))
        heldout = [] if name == 'final28' else validation
        np.savez(case / 'cameras.npz', K=np.stack(K).astype(np.float32), w2c=np.stack(w2c).astype(np.float32),
                 height=np.int64(2048), width=np.int64(2048), train_views=np.asarray(train, np.int64),
                 heldout_views=np.asarray(heldout, np.int64))
        for index in train:
            row = rows[index]
            path = DATA / row['image']; assert sha(path) == row['sha256']
            raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            assert raw.shape == (2048, 2048, 4) and raw.dtype == np.uint16
            rgba = np.rint(raw[:, :, [2, 1, 0, 3]].astype(np.float64) / 257).astype(np.uint8)
            Image.fromarray(rgba, 'RGBA').save(case / 'images' / f'{index:03d}.png')
            Image.fromarray(rgba[:, :, 3], 'L').save(case / 'masks' / f'{index:03d}.png')
            pngs.setdefault(index, sha(case / 'images' / f'{index:03d}.png'))
            assert sha(case / 'images' / f'{index:03d}.png') == pngs[index], 'cases must share identical map inputs'
        # Camera/normalisation self-check: body centroid projects inside every train image.
        centroid = np.r_[((vertices - center) / diagonal).mean(0), 1]
        for k, w in zip(K, w2c):
            p = k @ (w @ centroid)[:3]; p = p[:2] / p[2]
            assert (p > 0).all() and (p < 2047).all()
        write(case / 'CASE.json', dict(case=name, source=str(DATA), body_sha256=BODY_SHA256,
            center=center.tolist(), diagonal=diagonal, train_views=train, heldout_views=heldout,
            afp_index_to_dataset_view=view_ids, validation_dataset_views=list(VALIDATION_VIEWS) if heldout else [],
            images='8-bit straight RGBA from RGBA16 (round(v/257)); masks = alpha',
            test_pixels_present=False, validation_pixels_present=name == 'final28',
            K='dataset corner-origin K with cx,cy -0.5 (AFP pixel-centre convention)',
            w2c='OpenCV diag(1,-1,-1) @ dataset GL w2c, translation re-expressed in normalized frame'))
    return output


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=EXPERIMENT / 'afp/cases')
    print(build(p.parse_args().output))
