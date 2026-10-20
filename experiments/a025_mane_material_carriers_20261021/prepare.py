"""Geometry-derived Shell carriers; no GT hair, view weights, fitting or GS edits."""
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'a025_rooted_strands_20261020_review002'))
import experiment as r
CFG=json.loads((HERE/'config.json').read_text()); ROOT=r.ROOT; OUT=ROOT/CFG['prepared']


def path_from_cluster(root, normal, xyz, opacity, D):
    radius=np.linalg.norm(xyz-root,axis=1)
    bins=max(2,min(12,int(np.ceil(np.quantile(radius,.95)/(CFG['guide_bin_bbox']*D)))))
    # Radius ordering estimates the material path from its known root. A single
    # neighbour average removes bin jitter without a learned/GT-dependent curve.
    labels=np.minimum(np.floor(radius/max(radius.max(),1e-12)*bins).astype(int),bins-1)
    means=[np.average(xyz[labels==k],axis=0,weights=opacity[labels==k]) for k in range(bins) if np.any(labels==k)]
    path=np.r_[root[None],np.asarray(means)]
    if len(path)>3: path[1:-1]=.25*path[:-2]+.5*path[1:-1]+.25*path[2:]
    path=np.r_[root[None],(root+.001*D*normal)[None],path[1:]]
    distance=np.linalg.norm(np.diff(path,axis=0),axis=1)
    keep=np.r_[True,distance>1e-8*D];path=path[keep]
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
    count=max(4,min(64,int(np.ceil(arc[-1]/(CFG['guide_edge_bbox']*D)))+1))
    s=np.linspace(0,arc[-1],count)
    result=np.stack([np.interp(s,arc,path[:,k]) for k in range(3)],axis=1)
    result[0]=root
    # Preserve an explicit outward first edge, without changing the clump tip.
    result[1]=root+.001*D*normal
    return result


def proxy_asset(a, paths, columns):
    nodes=[];edges=[];base=[];group=[];offset=0
    for g,p in enumerate(paths):
        base.append(offset);nodes.extend(p)
        edges.extend(np.c_[np.arange(offset,offset+len(p)-1),np.arange(offset+1,offset+len(p))])
        group.extend([g]*(len(p)-1));offset+=len(p)
    nodes=np.asarray(nodes,float);edges=np.asarray(edges,int);group=np.asarray(group,int)
    delta=nodes[edges[:,1]]-nodes[edges[:,0]];length=np.linalg.norm(delta,axis=1)
    assert (length>1e-10).all()
    z=delta/length[:,None];helper=np.tile([1.,0,0],(len(z),1));helper[abs(z[:,0])>.9]=[0,1,0]
    x=np.cross(helper,z);x/=np.linalg.norm(x,axis=1,keepdims=True)
    frame=np.stack([x,np.cross(z,x),z],axis=2);N=len(edges)
    return dict(body_vertices=a['body_vertices'],body_faces=a['body_faces'],
        anchor_face=a['anchor_face'][columns],anchor_bary=a['anchor_bary'][columns],
        shell_guide_count=np.array(0),nodes=nodes,edges=edges,strand_base_node=np.asarray(base),
        strand_rows=np.arange(N),gs_guide=group,xyz=nodes[edges].mean(1),
        rotation=Rotation.from_matrix(frame).as_quat()[:,[3,0,1,2]],
        scale=np.log(np.c_[np.full((N,2),.0001),length/4]),
        dc=np.zeros((N,1,3)),rest=np.zeros((N,15,3)),opacity=np.zeros((N,1)),
        source_shell_columns=np.asarray(columns))


def main():
    OUT.mkdir(parents=True,exist_ok=True); assert not (OUT/'unchecked_guides.npz').exists()
    assert r.study.sha(ROOT/CFG['input'])==CFG['input_sha256']
    a=r.load(ROOT/CFG['input']);b=r.load(ROOT/'outputs/a025_curve_frame_wind_20261020/geometry_v2/gust/binding.npz')
    fields=r.load(ROOT/CFG['audit']/'spatial_fields.npz')
    reasonable=~fields['fully_buried_2sigma']&~fields['isolated_scale_relative']
    rows=b['long_rows'];col=b['long_column'];D=float(np.linalg.norm(np.ptp(a['body_vertices'],axis=0)))
    paths=[];columns=[];support=[];rejected=[]
    opacity=1/(1+np.exp(-a['opacity'][:,0].astype(float)))
    for column in np.unique(col):
        ii=np.flatnonzero((col==column)&reasonable[rows])
        if len(ii)<2: rejected.append(int(column));continue
        path=path_from_cluster(b['root'][column],b['normal'][column],a['xyz'][rows[ii]],opacity[rows[ii]],D)
        paths.append(path);columns.append(column);support.extend(rows[ii])
    proxy=proxy_asset(a,paths,np.asarray(columns,int))
    np.savez_compressed(OUT/'unchecked_guides.npz',**proxy)
    np.savez_compressed(OUT/'supported_render_rows.npz',rows=np.sort(support))
    r.study.write(OUT/'PREPARATION.json',dict(input_sha256=CFG['input_sha256'],physical_curves=len(paths),
        physical_nodes=len(proxy['nodes']),physical_edges=len(proxy['edges']),render_GS=len(a['xyz']),
        supported_render_Shell=len(support),fallback_columns=rejected,
        curve_source='Opacity-weighted frozen Shell centers in root-radius bins; one neighbour smoothing; known root/normal.',
        GT_read=False,view_weight_used=False,no_render_GS_added=True,source_sha256=r.study.sha(Path(__file__))))
    print('PREPARED',len(paths),'physical curves;',len(support),'Shell GS; NO new rendering GS',flush=True)


if __name__=='__main__':main()
