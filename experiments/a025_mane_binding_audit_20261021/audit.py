"""Frozen-model spatial + material-coordinate audit. No fitting or motion edits."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'a025_rooted_strands_20261020_review002'))
import experiment as r
sys.path.insert(0, str(HERE.parent/'a025_fixed_mesh_wind_20261020'))
import wind
sys.path.insert(0, str(HERE.parent/'a025_flow_consistency_20261020'))
import diagnose as flow

ROOT = r.ROOT
SOURCE = ROOT/'outputs/a025_rooted_strands_20261020_review002/born_routed'
OLD = ROOT/'outputs/a025_curve_frame_wind_20261020/geometry_v2/gust'
OUT = ROOT/'outputs/a025_mane_binding_audit_20261021/audit001'
EXPECTED = 'a01e93e463361e49de1e6f97236a9988ee9d489a4826b5b017eb51f2ed6f5317'
BOXES = dict(fringe=[786,1217,1112,1297], broad=[800,1010,1130,1290])


def stats(x, w=None):
    x = np.asarray(x); w = np.ones(len(x)) if w is None else np.asarray(w)
    valid = np.isfinite(x)&(w > 0); x, w = x[valid], w[valid]
    if not len(x): return None
    order = np.argsort(x); cumulative = np.cumsum(w[order]); cumulative /= cumulative[-1]
    q = np.interp([.1,.5,.9,.99,1], cumulative, x[order])
    return dict(mean=float(np.average(x, weights=w)), **dict(zip(['p10','median','p90','p99','max'], map(float,q))))


def visible_weights(a, view):
    """d(sum ROI red)/d(GS red) = exact full-batch Ti*alphai integration.

    This is a read-only renderer derivative, NOT model optimization. Constant
    .25 colors keep the official [0,1] output clamp inactive. Validate with an
    independent forward indicator render before using weighted quantiles.
    """
    native = r.study.native(); model = native.GaussianModel(3)
    r.study.assign(model,a); model.active_sh_degree = 3
    for field in r.study.FIELDS.values(): getattr(model,field).requires_grad_(False)
    cam = r.study.camera(view); bg = torch.zeros(3,device='cuda')
    with torch.no_grad(): _, alpha = r.study.rgb_alpha(native,cam,model)
    weights = {}; checks = {}
    for name,(x0,y0,x1,y1) in BOXES.items():
        color = torch.full((len(a['xyz']),3),.25,device='cuda',requires_grad=True)
        image = native.render(cam,model,native.pipe,bg,override_color=color)['render']
        gradient, = torch.autograd.grad(image[0,y0:y1,x0:x1].sum(),color)
        weight = gradient[:,0].detach().cpu().numpy().astype(float)
        assert weight.min() >= -1e-7
        weight = np.maximum(weight,0)
        reference = float(alpha[0,y0:y1,x0:x1].sum())
        error = abs(weight.sum()/reference-1)
        assert error < 2e-4, (name,error)
        weights[name] = weight
        checks[name] = dict(sum_weights=float(weight.sum()),alpha_sum=reference,relative_sum_error=error)
    colors = np.c_[(a['route']==0)&~a['body_support_context_mask'],a['route']==1,a['body_support_context_mask']].astype(np.float32)
    with torch.no_grad():
        check = native.render(cam,model,native.pipe,bg,override_color=torch.tensor(colors,device='cuda'))['render']
    for name,(x0,y0,x1,y1) in BOXES.items():
        forward = check[:,y0:y1,x0:x1].sum((1,2)).cpu().numpy()/checks[name]['alpha_sum']
        backward = weights[name]@colors/weights[name].sum()
        assert np.max(abs(forward-backward)) < 2e-4
        checks[name]['forward_backward_route_error_max'] = float(np.max(abs(forward-backward)))
    return weights,checks


def main():
    inventory = subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True)
    available = {line.split(',')[0].strip():int(line.split(',')[1]) for line in inventory.splitlines()}
    assert available[os.environ['CUDA_VISIBLE_DEVICES']] < 128
    OUT.mkdir(parents=True,exist_ok=True); assert not (OUT/'COMPLETE.json').exists()
    started = time.perf_counter(); assert r.study.sha(SOURCE/'complete_asset.npz') == EXPECTED
    a = r.load(SOURCE/'complete_asset.npz'); b = r.load(OLD/'binding.npz'); q = r.load(SOURCE/'optimized_queries.npz')
    target = r.load(SOURCE/'optimized_asset.npz')
    assert r.study.sha(SOURCE/'optimized_asset.npz') == json.loads((SOURCE/'QUERY_RECEIPT.json').read_text())['asset_sha256']
    assert all(np.array_equal(a[k],target[k]) for k in ('xyz','nodes','body_vertices','body_faces'))
    topology = r.topo.validate(a)
    view = r.study.load_views(limit=1)[0]; assert view.id == '0001'
    cfg = json.loads((HERE.parent/'a025_curve_frame_wind_20261020/config.json').read_text())
    s = wind.Wind(a,np.linalg.inv(view.w2c)[:3,0],cfg); D = s.D
    weights,checks = visible_weights(a,view)
    np.savez_compressed(OUT/'visibility_weights.npz',**weights)
    R = Rotation.from_quat(a['rotation'][:,[1,2,3,0]]).as_matrix(); sig = np.exp(a['scale'].astype(float))
    nn = q['center_normal']; sigma_n = np.linalg.norm(np.einsum('nji,nj->ni',R,nn)*sig,axis=1)
    signed = np.where(q['center_inside']==1,-1,1)*q['center_distance']/D
    nearest = cKDTree(a['xyz']).query(a['xyz'],k=2,workers=4)[0][:,1]
    isolated = (nearest > 4*sig.max(1))&(nearest > 4*np.median(nearest))
    anchors = r.topo.roots(a); root_dist = np.linalg.norm(a['xyz']-anchors[a['gs_guide']],axis=1)/D
    opacity = 1/(1+np.exp(-a['opacity'][:,0].astype(float)))
    fields = dict(signed_body_distance_bbox=signed,absolute_body_distance_bbox=abs(signed),
        sigma_max_bbox=sig.max(1)/D,sigma_normal_bbox=sigma_n/D,
        root_to_center_bbox=root_dist,nearest_center_spacing_bbox=nearest/D,opacity=opacity,
        inside_0p1pctD=signed < -.001,inside_0p01pctD=signed < -.0001,
        fully_buried_2sigma=(signed+2*sigma_n/D < -.0001),
        isolated_scale_relative=isolated,far_body_5pctD=abs(signed) > .05)
    context = a['body_support_context_mask']; shell = (a['route']==0)&~context; strand = a['route']==1
    masks = dict(Shell=shell,Strand=strand,body_context=context)
    report = dict(input_sha256=EXPECTED,body_diagonal_world=D,topology=topology,
        query_reuse='All xyz/nodes/body fields bitwise match the exact-BVH query target.',visibility_checks=checks)
    spatial = {}
    for scope,w in dict(global_count=np.ones(len(a['xyz'])),**weights).items():
        spatial[scope] = {}
        for name,mask in masks.items():
            ww = w*mask; den = ww.sum()
            spatial[scope][name] = dict(count=int(mask.sum()),mass_fraction=float(den/w.sum()),
                **{key:(float(np.average(v,weights=ww)) if v.dtype == bool else stats(v,ww)) for key,v in fields.items()})
    report['spatial'] = spatial
    rows,col,edge,f,w = b['long_rows'],b['long_column'],b['long_edge'],b['long_fraction'],b['long_weights']
    total = np.add.reduceat(s.length,s.start); curve = s.group[edge]
    coordinate = b['long_arc']*total[curve]
    length = (w*total[curve]).sum(1)
    ratio = np.linalg.norm(b['long_offset'],axis=1)/np.maximum(length,1e-30)
    percurve_bridge = np.add.reduceat(s.length*(a['segment_kind'][s.rows]==1),s.start)/total
    binding_fields = dict(arc_fraction=(w*b['long_arc']).sum(1),
        sample_in_first_10pct=(w*(b['long_arc'] <= .1)).sum(1),
        sample_in_first_25pct=(w*(b['long_arc'] <= .25)).sum(1),
        sample_at_root=(w*(b['long_arc'] <= 1e-8)).sum(1),
        on_root_connector=(w*(a['segment_kind'][s.rows][edge]==1)).sum(1),
        effective_carrier_length_bbox=length/D,used_arc_length_bbox=(w*coordinate).sum(1)/D,
        offset_to_carrier_bbox=np.linalg.norm(b['long_offset'],axis=1)/D,
        offset_over_carrier_length=ratio,
        carrier_shorter_than_root_center=length/D < root_dist[rows],
        carrier_bridge_length_fraction=(w*percurve_bridge[curve]).sum(1),
        driver_root_distance_bbox=(w*np.linalg.norm(s.roots[curve]-b['root'][col,None],axis=2)).sum(1)/D)
    report['binding'] = {}
    for scope,weight in dict(global_count=np.ones(len(a['xyz'])),**weights).items():
        ww = weight[rows]
        report['binding'][scope] = dict(long_Shell_mass_fraction=float(ww.sum()/weight.sum()),
            **{key:(float(np.average(v,weights=ww)) if key.startswith(('sample_','on_','carrier_shorter')) else stats(v,ww)) for key,v in binding_fields.items()})
    np.savez_compressed(OUT/'spatial_fields.npz',rows=rows,**fields,**{'binding_'+k:v for k,v in binding_fields.items()})
    # Reuse the previous unsigned, multiscale, connected-surface flow protocol.
    report['flow'] = flow.static(a,q,flow.SurfacePatches(a['body_vertices'],a['body_faces']),D,
        np.random.default_rng(20261020),OUT,'current')
    report.update(seconds=time.perf_counter()-started,
        semantics='Body distance is not strand-GT error: long mane may legitimately be far outside skin. Covariance support and visible weight matter. Parallel tangents are not labelled incorrect.',
        threshold_note='0.1%D/0.01%D,2sigma and scale-relative isolation are frozen previous diagnostic conventions, not a newly certified physics-quality gate.',
        GT_groom_read=False,model_optimized=False,source_sha256=r.study.sha(Path(__file__)))
    assert r.study.sha(SOURCE/'complete_asset.npz') == EXPECTED
    r.study.write(OUT/'COMPLETE.json',report)
    print(json.dumps({k:v for k,v in report.items() if k != 'flow'},indent=2),flush=True)


if __name__ == '__main__': main()
