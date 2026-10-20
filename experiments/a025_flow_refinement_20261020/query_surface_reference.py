"""Read-only Blender BVH queries against each asset's supplied known body."""
from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_tree(vertices, faces):
    return BVHTree.FromPolygons(vertices.tolist(), faces.tolist(), all_triangles=True)


def query(tree, points, diagonal, normal_sign):
    count=len(points); nearest=np.empty((count,3)); normals=np.empty((count,3))
    face=np.empty(count,np.int32); distance=np.empty(count)
    inside=np.zeros(count,np.int8); conflicts=np.zeros(count,bool)
    epsilon=float(1e-6*diagonal)
    directions=[Vector(v).normalized() for v in [(0.733,0.421,0.534),(-0.387,0.751,0.535),(0.329,-0.517,0.790)]]

    def parity(point,direction):
        origin=Vector(point); crossings=0
        for _ in range(64):
            location,normal,index,length=tree.ray_cast(origin,direction,4*diagonal)
            if location is None:return crossings%2
            crossings+=1; origin=location+direction*epsilon
        raise ValueError('Unexpected ray traversal; do not silently accept a sign')

    for i,p in enumerate(points):
        q,n,f,d=tree.find_nearest(Vector(p))
        assert q is not None
        nearest[i]=q;normals[i]=normal_sign*np.asarray(n);face[i]=f;distance[i]=d
        if d>epsilon:
            vote=parity(p,directions[0])
            local_inside=float(np.dot(p-nearest[i],normals[i]))<0
            if vote!=local_inside:
                conflicts[i]=True
                other=parity(p,directions[1])
                vote=int(vote+other+parity(p,directions[2])>=2) if other!=vote else vote
            inside[i]=1 if vote else -1
        if (i+1)%100000==0:print('queried',i+1,'/',count,flush=True)
    return dict(nearest=nearest,normal=normals,face=face,distance=distance,
                inside=inside,ray_local_disagreement=conflicts)


def self_check():
    v=np.array([[0.,0,0],[1.,0,0],[0,1.,0],[0,0,1.]])
    f=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]])
    q=query(make_tree(v,f),np.array([[.1,.1,.1],[2.,0,0],[.1,.1,0]]),np.sqrt(3),1)
    assert q['inside'].tolist()==[1,-1,0]
    assert np.allclose(q['distance'],[.1,1.,0],atol=1e-6)
    q2=query(make_tree(v,f[:,::-1]),np.array([[.1,.1,.1],[2.,0,0]]),np.sqrt(3),-1)
    assert q2['inside'].tolist()==[1,-1]
    print('PASS: known inside/outside/surface distances and reversed mesh winding',flush=True)


def main(root):
    self_check();root.mkdir(parents=True,exist_ok=True)
    source=Path('F:/UniFur/exports/A025_routes_3DGS_20261018/complete_assets')
    expected={'free_then_route':'094fe1c4de1ae70745dc74bee99c6c744a0cbe39c91db3e212570e212d9fbd01',
              'born_routed':'f343bb6243005df7c6fd06829d041eb0d0cb7bad3a559361a9e3b0986a23d099'}
    receipt={}
    for arm,digest in expected.items():
        out=root/(arm+'_queries.npz');assert not out.exists(),'Fresh audit output required'
        path=source/(arm+'_complete_asset.npz');assert sha(path)==digest
        with np.load(path,allow_pickle=False) as z:a={k:z[k] for k in z.files}
        v,f=a['body_vertices'],a['body_faces'];tri=v[f]
        cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
        volume=np.einsum('ij,ij->i',tri[:,0],cross).sum()/6
        edge=np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]])
        signed=np.where(edge[:,0]<edge[:,1],1,-1)
        unique,ids,counts=np.unique(np.sort(edge,axis=1),axis=0,return_inverse=True,return_counts=True)
        assert (counts==2).all() and np.all(np.bincount(ids,weights=signed)==0)
        assert abs(volume)>1e-9 and (np.linalg.norm(cross,axis=1)>1e-14).all()
        D=float(np.linalg.norm(np.ptp(v,axis=0)));tree=make_tree(v,f);began=time.time();data={}
        for name,points in [('center',a['xyz']),('node',a['nodes'])]:
            print(arm,name,len(points),flush=True)
            data.update({name+'_'+k:value for k,value in query(tree,points,D,1 if volume>0 else -1).items()})
        np.savez_compressed(out,**data)
        receipt[arm]=dict(source_sha256=digest,diagonal=D,signed_volume=float(volume),
                          closed_consistently_oriented=True,euler_characteristic=len(v)-len(unique)+len(f),
                          sign_precision_band=1e-6*D,seconds=time.time()-began,
                          method='Blender BVH closest triangle; ray parity, disagreement fallback votes; no ICP, no fitting',
                          self_intersection_of_body_not_certified=True)
        assert sha(path)==digest
        (root/'QUERY_RECEIPT.json').write_text(json.dumps(receipt,indent=2))
        print(arm,receipt[arm],flush=True)


if __name__=='__main__':main(Path(sys.argv[sys.argv.index('--')+1]))
