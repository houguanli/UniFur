"""No-GT counterfactual: root-nearest3 vs full connected-patch curve geometry.

No GS position, curve node, physics parameter, or trained appearance is changed.
"""
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE)); import audit as audit_base
r, wind, flow = audit_base.r, audit_base.wind, audit_base.flow
ROOT, SOURCE, OLD, OUT = audit_base.ROOT, audit_base.SOURCE, audit_base.OLD, audit_base.OUT


def closest(x, p0, delta):
    f = np.clip(np.einsum('nej,ej->ne',x[:,None]-p0[None],delta)/np.sum(delta*delta,axis=1),0,1)
    points = p0[None]+f[...,None]*delta[None]
    d2 = np.sum((x[:,None]-points)**2,axis=2)
    pick = np.argmin(d2,axis=1)
    return pick,f[np.arange(len(x)),pick],points[np.arange(len(x)),pick],np.sqrt(d2[np.arange(len(x)),pick])


def main():
    assert (OUT/'COMPLETE.json').exists() and not (OUT/'CANDIDATE_AUDIT.json').exists()
    start = time.perf_counter(); a = r.load(SOURCE/'complete_asset.npz'); b = r.load(OLD/'binding.npz')
    assert r.study.sha(SOURCE/'complete_asset.npz') == audit_base.EXPECTED
    cfg = json.loads((HERE.parent/'a025_curve_frame_wind_20261020/config.json').read_text())
    s = wind.Wind(a,[1,0,0],cfg); D=s.D; total=np.add.reduceat(s.length,s.start)
    patches=flow.SurfacePatches(a['body_vertices'],a['body_faces']); S=int(a['shell_guide_count'])
    byface={}
    for g,f in enumerate(a['anchor_face'][S:]): byface.setdefault(int(f),[]).append(g)
    normal=b['normal']; rows=b['long_rows']; col=b['long_column']; N=len(rows)
    chosen=np.full(N,-1,int); edge=np.zeros(N,int); fraction=np.zeros(N); centers=np.zeros((N,3)); distance=np.zeros(N)
    candidates_count=[]; guide_record={}; missing=[]
    kinds=a['segment_kind'][s.rows]; opacity=1/(1+np.exp(-a['opacity'][rows,0].astype(float)))
    rootdistance=np.linalg.norm(a['xyz'][rows]-b['root'][col],axis=1)
    sigma=np.exp(a['scale'][rows]).max(1)
    for column in np.unique(col):
        ii=np.flatnonzero(col==column); f=int(a['anchor_face'][column]); found={f}; frontier={f}
        for _ in range(8):
            frontier={j for ff in frontier for j in patches.adj[ff]}-found; found.update(frontier)
        candidates=[g for ff in sorted(found) for g in byface.get(ff,[]) if
            s.normal[g]@normal[column] >= .5 and np.linalg.norm(s.roots[g]-b['root'][column]) <= .06*D
            and total[g] >= .035*D]
        required=max(.035*D,float(np.quantile(np.maximum(rootdistance[ii]-2*sigma[ii],0),.9)))
        candidates=[g for g in candidates if total[g] >= required]
        candidates_count.append(len(candidates)); best=None
        x=a['xyz'][rows[ii]]
        for g in candidates:
            indices=np.arange(s.start[g],s.start[g]+s.count[g])
            native=indices[kinds[indices] != 1]
            if not len(native): continue
            shift=b['root'][column]-s.roots[g]
            p0=a['nodes'][s.edges[native,0]]+shift
            pick,frac,point,d=closest(x,p0,s.rest_delta[native])
            score=float(np.average(d*d,weights=opacity[ii]))
            key=(score,float(np.linalg.norm(shift)),g)
            if best is None or key < best[0]: best=(key,g,native[pick],frac,point,d)
        if best is None:
            missing.append(int(column)); continue
        _,g,ee,ff,pp,dd=best
        chosen[ii]=g; edge[ii]=ee; fraction[ii]=ff; centers[ii]=pp; distance[ii]=dd
        guide_record[str(int(column))]=dict(curve=int(g),candidate_count=len(candidates),
            required_extent_bbox=required/D,carrier_length_bbox=total[g]/D,
            root_shift_bbox=float(np.linalg.norm(b['root'][column]-s.roots[g])/D),score_world2=best[0][0])
    valid=chosen>=0
    cumulative=np.cumsum(s.length); before=cumulative-s.length-np.r_[0.,cumulative[s.start[1:]-1]][s.group]
    arc=np.zeros(N); arc[valid]=(before[edge[valid]]+fraction[valid]*s.length[edge[valid]])/total[chosen[valid]]
    visibility=r.load(OUT/'visibility_weights.npz'); report={}
    for scope,w0 in dict(global_count=np.ones(len(a['xyz'])),**visibility).items():
        w=w0[rows]; v=w*valid
        report[scope]=dict(covered_long_Shell_mass_fraction=float(v.sum()/w.sum()),
            old_offset_bbox=audit_base.stats(np.linalg.norm(b['long_offset'],axis=1)/D,w),
            new_native_offset_bbox=audit_base.stats(distance/D,v),
            old_normalized_arc=audit_base.stats((b['long_arc']*b['long_weights']).sum(1),w),
            new_native_normalized_arc=audit_base.stats(arc,v),
            new_offset_beyond_2sigma_plus_0p5pctD=float(np.average(distance>2*sigma+.005*D,weights=v)),
            new_sample_in_first10pct=float(np.average(arc<.1,weights=v)))
    np.savez_compressed(OUT/'candidate_bindings.npz',rows=rows,column=col,curve=chosen,edge=edge,
        fraction=fraction,center=centers,distance=distance,arc=arc)
    report.update(candidate_pool='All8-hop connected body faces, same root-distance<=0.06D/normal-dot>=0.5; actual curves>=max(0.035D,column90%root-relative2sigma extent).',
        fit='Single primary carrier minimizes all-column opacity-weighted point-to-NATIVE-render-edge squared distance. Excludes artificial root connectors from material matching, retains them in physical graph. No ROI or GT optimization.',
        candidate_counts=audit_base.stats(np.asarray(candidates_count)),missing_columns=missing,
        guides=guide_record,no_model_writes=True,GT_read=False,seconds=time.perf_counter()-start,
        source_sha256=r.study.sha(Path(__file__)))
    r.study.write(OUT/'CANDIDATE_AUDIT.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='guides'},indent=2),flush=True)


if __name__=='__main__': main()
