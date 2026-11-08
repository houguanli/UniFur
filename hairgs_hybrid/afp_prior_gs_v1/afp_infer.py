"""Frozen DINOv3 maps + AFP dense inference for the strict-SMAL A025 cases.

Run with AnimalFurPrior's DINOv3 runtime. AFP code, checkpoint and calibration
are imported read-only; every output lives under this experiment directory.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

from .common import AFP, AFP_CALIBRATION, AFP_CHECKPOINT, EXPERIMENT, sha, write

sys.path.insert(0, str(AFP))


def extract_maps(case, maps, device):
    import torch
    from furprior.features.dinov3 import FeatureConfig, FrozenDINOv3, extract_view, require_checkpoint_contract
    from furprior.inference.checkpoint import Predictor
    checkpoint = Predictor.load(AFP_CHECKPOINT, 'cpu').checkpoint
    values = dict(checkpoint['feature_encoder_identity']['config'])
    values['input_hw'] = tuple(values['input_hw'])
    started = time.monotonic()
    encoder = FrozenDINOv3(FeatureConfig(**values), AFP, device)
    # Same encoder identity as the AFP training cache, or fail before extracting.
    require_checkpoint_contract(checkpoint, encoder.identity)
    with np.load(case / 'cameras.npz', allow_pickle=False) as cams:
        views = cams['train_views'].tolist()
    receipts = []
    for view in views:
        path, receipt = extract_view(encoder, case, int(view), maps)
        receipts.append(dict(view=int(view), path=str(path), sha256=receipt['sha256']))
        print('DINOV3_MAP', view, path, flush=True)
    torch.cuda.synchronize()
    return dict(views=receipts, seconds=time.monotonic() - started,
                peak_allocated_bytes=torch.cuda.max_memory_allocated())


def infer(case, maps, output, density_scale, device, seed):
    from furprior.inference.dense_stream_v2 import generate
    return generate(case, AFP_CHECKPOINT, AFP_CALIBRATION, output, 'A025_strict_smal', device, maps,
                    density_scale=density_scale, chunk_size=8192, seed=seed)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases', type=Path, default=EXPERIMENT / 'afp/cases')
    p.add_argument('--maps', type=Path, default=EXPERIMENT / 'afp/maps')
    p.add_argument('--predictions', type=Path, default=EXPERIMENT / 'afp/predictions')
    p.add_argument('--density-scale', type=float, default=.2)
    p.add_argument('--seed', type=int, default=20261105)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    receipt = dict(checkpoint=str(AFP_CHECKPOINT), checkpoint_sha256=sha(AFP_CHECKPOINT),
                   calibration_sha256=sha(AFP_CALIBRATION), density_scale=a.density_scale, seed=a.seed,
                   maps=extract_maps(a.cases / 'final28', a.maps, a.device))
    for name in ['final28', 'pilot24']:
        meta = infer(a.cases / name, a.maps, a.predictions / name, a.density_scale, a.device, a.seed)
        receipt[name] = {k: meta[k] for k in ('guide_count', 'child_count', 'total_count', 'curve_sha256',
                                              'condition_views', 'seconds', 'budget')}
        print(json.dumps(dict(case=name, **receipt[name]), default=str), flush=True)
    write(a.predictions / 'AFP_INFERENCE.json', dict(receipt, status='complete', target_groom_read=False,
        test_pixels_read=False, note='AFP training families include A025 (different body/pose/render); '
                                     'this is an in-family prior, not unseen-animal generalisation'))


if __name__ == '__main__':
    main()
