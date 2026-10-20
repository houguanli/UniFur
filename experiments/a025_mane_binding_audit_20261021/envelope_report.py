"""Independent sampled-native-groom envelope corroboration; never drives a fit."""
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import audit as base
r,SOURCE,OUT=base.r,base.SOURCE,base.OUT


def main():
    assert not (OUT/'ENVELOPE_REPORT.json').exists()
    start=time.perf_counter();a=r.load(SOURCE/'complete_asset.npz')
    assert r.study.sha(SOURCE/'complete_asset.npz')==base.EXPECTED
    D=float(np.linalg.norm(np.ptp(a['body_vertices'],axis=0))); w=r.load(OUT/'visibility_weights.npz')
    with np.load(OUT/'sampled_GT_envelope.npz',allow_pickle=False) as z:
        points=z['points']; half=z['half_curve_samples']
    receipt=json.loads((OUT/'GT_EVALUATION_RECEIPT.json').read_text())
    tree=cKDTree(half); dh,_=tree.query(a['xyz'],workers=4);del tree,half;gc.collect()
    tree=cKDTree(points);d,ids=tree.query(a['xyz'],workers=4)
    direction=(a['xyz']-points[ids])/np.maximum(d[:,None],1e-20)
    R=Rotation.from_quat(a['rotation'][:,[1,2,3,0]]).as_matrix()
    local=np.einsum('nji,nj->ni',R,direction)
    sigma=np.linalg.norm(local*np.exp(a['scale']),axis=1)
    gap=(d-2*sigma)/D
    # Conservative diagnostic margin, not a claim that a Shell must match a hair.
    far=gap>.005; half_far=(dh-2*sigma)/D>.005
    shell=(a['route']==0)&~a['body_support_context_mask'];strand=a['route']==1
    report={}
    for name,weight in dict(global_count=np.ones(len(a['xyz'])),**w).items():
        report[name]={}
        for route,mask in [('Shell',shell),('Strand',strand)]:
            ww=weight*mask
            report[name][route]=dict(nearest_sampled_fur_distance_bbox=base.stats(d/D,ww),
                two_sigma_gap_bbox=base.stats(gap,ww),
                far_beyond_2sigma_plus_0p5pctD=float(np.average(far,weights=ww)),
                same_flag_half_curve_sample=float(np.average(half_far,weights=ww)),
                nearest_distance_sampling_change_bbox=base.stats((dh-d)/D,ww))
    report.update(GT_points=len(points),GT_source_receipt=receipt,source_sha256=r.study.sha(Path(__file__)),
        scope='Sampled native Catmull-Rom envelope, not exact full dense-groom distance/OTC. Margin0.5%D after2sigma covariance support; half-curve sensitivity is an uncertainty check.',
        no_GT_fitting=True,no_static_or_wind_changes=True,seconds=time.perf_counter()-start)
    np.savez_compressed(OUT/'GT_envelope_fields.npz',distance=d/D,two_sigma_gap=gap,
        far=far,nearest=points[ids])
    r.study.write(OUT/'ENVELOPE_REPORT.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='GT_source_receipt'},indent=2),flush=True)


if __name__=='__main__':main()
