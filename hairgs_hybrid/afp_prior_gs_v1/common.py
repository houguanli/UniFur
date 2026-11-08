"""Shared paths, split identities and small IO helpers; no CUDA imports."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get('UNIFUR_ROOT', '/root/Public/h84435299/UniFur'))
AFP = Path(os.environ.get('AFP_ROOT', '/root/Public/h84435299/AnimalFurPrior'))
DATA = ROOT / 'dataset/A025_strict_smal_lion_v2_2k'
ORACLE = ROOT / 'experiments/a025_oracle_surface_routes_20261017'
FREE_THEN_ROUTE = ROOT / 'outputs/a025_oracle_surface_routes_20261017/free_then_route'
EXPERIMENT = ROOT / 'outputs/a025_afp_prior_free_gs_20261105'
AFP_RUN = AFP / 'output/original_adaptive_dense_20261103_v2'
AFP_CHECKPOINT = AFP_RUN / 'training/best.pt'
AFP_CALIBRATION = AFP_RUN / 'calibration.json'
BODY_SHA256 = '369a66c68f6f7d97b0b11826c6bb3784c5e7740511d18795f729bde5e121e1fb'
# Pilot-only validation: interior TRAIN views with train neighbours at +-10 deg,
# mirroring the 20 deg gaps around test views. Final runs train on all 28 views.
VALIDATION_VIEWS = ('0003', '0012', '0022', '0028')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def train_rows():
    complete = read(DATA / 'COMPLETE.json')
    rows = sorted((v for v in complete['views'] if v['split'] == 'train'), key=lambda v: v['view_id'])
    assert len(rows) == 28 and all(int(v['view_id']) % 5 for v in rows)
    return rows
