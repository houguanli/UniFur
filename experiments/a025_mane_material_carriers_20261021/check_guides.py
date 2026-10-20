"""Reject spatially implausible material carriers, not rendering GS."""
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE.parent/'a025_flow_refinement_20261020'))
import query_surface_reference as query
OUT=ROOT/'outputs/a025_mane_material_carriers_20261021/prepared001'


def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def main():
    assert not (OUT/'SPATIAL_PROOF.json').exists()
    raw=load(OUT/'unchecked_guides.npz');a=load(OUT/'guide_asset.npz')
    body,faces=a['body_vertices'],a['body_faces'];D=np.linalg.norm(np.ptp(body,axis=0))
    tri=body[faces];cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);sign=1 if np.einsum('ij,ij->i',tri[:,0],cross).sum()>0 else -1
    tree=query.make_tree(body,faces)
    initial=query.query(tree,raw['nodes'],D,sign)
    depth=np.where(initial['inside']==1,initial['distance']/D,0.)
    G=len(raw['strand_base_node']);maxdepth=np.zeros(G)
    node_group=np.repeat(np.arange(G),np.diff(np.r_[raw['strand_base_node'],len(raw['nodes'])]))
    np.maximum.at(maxdepth,node_group,depth)
    edge=a['edges'];p0=a['nodes'][edge[:,0]];delta=a['nodes'][edge[:,1]]-p0
    points=p0[:,None]+np.array([.25,.5,.75])[None,:,None]*delta[:,None]
    result=query.query(tree,points.reshape(-1,3),D,sign)
    deep=(result['inside']==1)&(result['distance']>.0001*D)
    flags=np.zeros(G,bool);np.logical_or.at(flags,a['gs_guide'],deep.reshape(-1,3).any(1))
    assert np.array_equal(raw['source_shell_columns'],a['source_shell_columns'])
    accepted=(maxdepth<=.005)&~flags
    payload=dict(source_columns=a['source_shell_columns'],accepted=accepted,
        raw_max_inside_depth_bbox=maxdepth,corrected_edge_deep=flags)
    np.savez_compressed(OUT/'spatial_proof.npz',**payload)
    report=dict(total=G,accepted=int(accepted.sum()),rejected=int((~accepted).sum()),
        raw_depth_over_0p5pctD=int((maxdepth>.005).sum()),corrected_deep_edge_guides=int(flags.sum()),
        criteria='Initial curve maximum inside-body depth<=0.5%D and no corrected edge quarter/mid samples deeper than0.01%D. Not a dynamic or volume guarantee.',
        accepted_columns=a['source_shell_columns'][accepted].tolist(),GT_groom_read=False)
    (OUT/'SPATIAL_PROOF.json').write_text(json.dumps(report,indent=2));print({k:v for k,v in report.items() if k!='accepted_columns'},flush=True)


if __name__=='__main__':main()
