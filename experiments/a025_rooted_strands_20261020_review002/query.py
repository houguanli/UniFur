"""Read-only body queries after training. Known body only; no GT groom."""
from pathlib import Path
import sys
import json
import hashlib
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE.parent/'a025_flow_refinement_20261020'))
import query_surface_reference as q


def main():
    out=ROOT/'outputs/a025_rooted_strands_20261020_review002/born_routed';path=out/'optimized_asset.npz'
    with np.load(path,allow_pickle=False) as z:a={k:z[k] for k in z.files}
    q.self_check();v,f=a['body_vertices'],a['body_faces'];D=np.linalg.norm(np.ptp(v,axis=0));tree=q.make_tree(v,f);data={}
    for name,points in [('center',a['xyz']),('node',a['nodes'])]:
        print('QUERY',name,len(points),flush=True)
        data.update({name+'_'+k:x for k,x in q.query(tree,points,D,1).items()})
    np.savez_compressed(out/'optimized_queries.npz',**data)
    (out/'QUERY_RECEIPT.json').write_text(json.dumps(dict(asset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),D=float(D),exact_BVH=True,GT_groom_read=False),indent=2))


if __name__=='__main__':main()
