"""Known-body initialization clearance ONLY; never reads groom GT."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE.parent/'a025_flow_refinement_20261020'))
import query_surface_reference as query
CFG=json.loads((HERE/'config.json').read_text());OUT=ROOT/CFG['prepared']


def main():
    assert not (OUT/'GUIDES_COMPLETE.json').exists()
    with np.load(OUT/'unchecked_guides.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
    body,faces=a['body_vertices'],a['body_faces'];D=float(np.linalg.norm(np.ptp(body,axis=0)))
    tri=body[faces];raw=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    sign=1 if np.einsum('ij,ij->i',tri[:,0],raw).sum()>0 else -1
    root=a['strand_base_node'];old=a['nodes'].copy();query.self_check()
    result=query.query(query.make_tree(body,faces),a['nodes'],D,sign)
    bad=(result['inside']==1);bad[root]=False
    a['nodes'][bad]=result['nearest'][bad]+CFG['guide_initial_clearance_bbox']*D*result['normal'][bad]
    assert np.array_equal(a['nodes'][root],old[root])
    projection_max=float(np.linalg.norm(a['nodes']-old,axis=1).max()/D)
    # Several interior samples may project to the same mesh vertex. Remove only
    # consecutive duplicate PHYSICAL nodes; never delete a rendering Gaussian.
    paths=[];kept=[];removed=0
    for g,begin in enumerate(root):
        end=root[g+1] if g+1<len(root) else len(a['nodes'])
        p=a['nodes'][begin:end]
        keep=np.r_[True,np.linalg.norm(np.diff(p,axis=0),axis=1)>1e-8*D]
        removed+=int((~keep).sum());p=p[keep]
        if len(p)>=2:paths.append(p);kept.append(g)
    nodes=[];edges=[];groups=[];bases=[];offset=0
    for g,p in enumerate(paths):
        bases.append(offset);nodes.extend(p)
        edges.extend(np.c_[np.arange(offset,offset+len(p)-1),np.arange(offset+1,offset+len(p))])
        groups.extend([g]*(len(p)-1));offset+=len(p)
    a['nodes']=np.asarray(nodes);a['edges']=np.asarray(edges,int);a['strand_base_node']=np.asarray(bases)
    for key in ('anchor_face','anchor_bary','source_shell_columns'):a[key]=a[key][kept]
    a['gs_guide']=np.asarray(groups);N=len(edges);a['strand_rows']=np.arange(N)
    a['dc']=np.zeros((N,1,3));a['rest']=np.zeros((N,15,3));a['opacity']=np.zeros((N,1))
    a['scale']=np.full((N,3),np.log(.0001))
    delta=a['nodes'][a['edges'][:,1]]-a['nodes'][a['edges'][:,0]]
    length=np.linalg.norm(delta,axis=1);assert (length>1e-9*D).all()
    # Solver rest axes come from nodes; stored proxy covariance is not rendered.
    a['xyz']=a['nodes'][a['edges']].mean(1);a['scale'][:,2]=np.log(length/4)
    z=delta/length[:,None];quat=np.c_[1+z[:,2],-z[:,1],z[:,0],np.zeros(N)]
    quat[z[:,2]<-.999999]=[0,1,0,0];quat/=np.linalg.norm(quat,axis=1,keepdims=True);a['rotation']=quat
    samples=a['nodes'][a['edges'][:,0]][:,None]+np.array([.25,.5,.75])[None,:,None]*delta[:,None]
    check=query.query(query.make_tree(body,faces),samples.reshape(-1,3),D,sign)
    deep=(check['inside']==1)&(check['distance']>.0001*D)
    np.savez_compressed(OUT/'guide_asset.npz',**a)
    receipt=dict(guide_sha256=hashlib.sha256((OUT/'guide_asset.npz').read_bytes()).hexdigest(),
        projected_initial_nodes=int(bad.sum()),initial_nodes=len(old),
        projection_max_bbox=projection_max,consecutive_physical_nodes_removed=removed,
        discarded_collapsed_guides=int(len(root)-len(kept)),
        initial_edges_with_deep_inside_samples=int(deep.reshape(-1,3).any(1).sum()),
        root_motion=0.,GT_groom_read=False,render_asset_changed=False,
        scope='Point initialization clearance only, NOT a dynamic contact solver or swept-edge collision guarantee.')
    (OUT/'GUIDES_COMPLETE.json').write_text(json.dumps(receipt,indent=2));print(receipt,flush=True)


if __name__=='__main__':main()
