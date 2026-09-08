"""Train-only images and mask-cone visual hull, never skin/mesh ground truth."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, binary_erosion
from scipy.optimize import linprog
import torch
import torch.nn.functional as F

from hairgs_hybrid.rgb_only_init_v1.core import pose, require, confined, sha


def source_lock(path,digest):
    """Pure input validation, intentionally independent of obsolete SQLite/MVS runners."""
    require(sha(path)==digest,'source lock differs from reviewed release')
    lock=json.loads(Path(path).read_text())
    require(lock.get('schema')=='rgb-only-train48-source-v1' and lock.get('status')=='frozen','source not frozen')
    require(lock.get('rgb_semantics')=='straight_rgba','straight source RGB semantics required')
    require(lock.get('release_root') and lock.get('release_files'),'publisher provenance required')
    for relative,expected in lock['release_files'].items():
        require(sha(confined(lock['release_root'],relative))==expected,'publisher release changed')
    root=Path(lock['train_root']).resolve()
    require(root.name=='train_v48','only isolated train48 input allowed')
    names=lock['image_names']
    require(len(names)==len(set(names))==48 and all(Path(n).name==n for n in names),'exactly48 unique training basenames required')
    require(set(lock['files'])=={'camera_manifest.json'}|{'images/'+n for n in names},'source must contain exactly manifest and48 RGBAs')
    for relative,expected in lock['files'].items():
        require(sha(confined(root,relative))==expected,'train input changed')
    manifest=json.loads(confined(root,'camera_manifest.json').read_text())
    require(manifest['split']=='train_v48' and manifest['image_count']==len(manifest['observations'])==48,'wrong train split')
    require([o['image'] for o in manifest['observations']]==names,'camera/image order changed')
    require(manifest['view_ids']==lock['view_ids'] and len(set(lock['view_ids']))==48,'publisher view identity changed')
    for observation in manifest['observations']:
        require(observation['intrinsics'][:2]==[2048,2048],'only original2K input allowed')
        pose(observation)
    return root,manifest


@dataclass
class TrainView:
    name: str
    view_id: str | int  # Publisher identity, NOT a COLMAP or contiguous index.
    rotation: np.ndarray
    translation: np.ndarray
    intrinsics: np.ndarray  # fx,fy,cx,cy at original size
    rgba: torch.Tensor  # straight native RGBA, CHW, CPU

    def at(self, side):
        require(side in (512,1024,2048) or side <= 64, 'unapproved image stage')
        height,width=self.rgba.shape[1:]
        require(height==width and side<=width,'square image/downsample contract')
        # Composite before area reduction: avoids alpha-edge colour bias/double alpha.
        alpha=self.rgba[3:4]
        rgb=F.interpolate((self.rgba[:3]*alpha)[None],size=(side,side),mode='area')[0]
        alpha=F.interpolate(alpha[None],size=(side,side),mode='area')[0]
        k=np.asarray(self.intrinsics,float)*(side/width)
        return rgb,alpha,k


def load_train_views(lock_path, lock_sha):
    """Only the publisher's 48 RGBA + camera manifest whitelist is admitted."""
    root,manifest=source_lock(Path(lock_path),lock_sha)
    views=[]
    for view_id,obs in zip(manifest['view_ids'],manifest['observations']):
        path=confined(root,'images/'+obs['image'])
        with Image.open(path) as image: raw=np.asarray(image).copy()
        require(raw.shape==(2048,2048,4) and raw.dtype==np.uint8,'native2K straight RGBA8 required')
        r,t,_=pose(obs)
        views.append(TrainView(obs['image'],view_id,r,t,np.asarray(obs['intrinsics'][2:],float),
                               torch.from_numpy(raw).permute(2,0,1).float()/255))
    require(len(views)==48,'all48 training views required')
    return views,dict(source_lock_sha256=lock_sha,inputs='48 train RGB/camera/alpha only',
                      teacher_read=False,mvs_read=False,geometry_gt_read=False)


def project_numpy(points, view, k=None):
    local=np.asarray(points)@view.rotation.T+view.translation
    k=view.intrinsics if k is None else k
    with np.errstate(divide='ignore',invalid='ignore'):
        pixel=local[:,:2]/local[:,2:3]*k[:2]+k[2:]
    return pixel,local[:,2]


def mask_cone_bounds(views, side=512):
    """Finite AABB from calibrated silhouette bounding cones via six linear programs."""
    require(len(views)>=3 and len({v.view_id for v in views})==len(views),'independent training views required')
    equations=[];rhs=[];masks=[];intrinsics=[]
    for view in views:
        _,alpha,k=view.at(side)
        mask=alpha[0].numpy()>2/255
        require(mask.any(),'empty foreground scene/view')
        require(not (mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any()),
                'truncated foreground mask: finite full-object hull not established')
        y,x=np.nonzero(mask);r,t=view.rotation,view.translation
        # Bounds contain pixel footprints, not just centres. No scene near/far is invented.
        for axis,lo,hi in ((0,x.min(),x.max()+1),(1,y.min(),y.max()+1)):
            a=k[axis]*r[axis]-(hi-k[axis+2])*r[2]
            b=k[axis]*t[axis]-(hi-k[axis+2])*t[2]
            equations.append(a);rhs.append(-b)
            a=(lo-k[axis+2])*r[2]-k[axis]*r[axis]
            b=(lo-k[axis+2])*t[2]-k[axis]*t[axis]
            equations.append(a);rhs.append(-b)
        equations.append(-r[2]);rhs.append(t[2])
        masks.append(mask);intrinsics.append(k)
    bounds=[]
    for axis in range(3):
        direction=np.eye(3)[axis]
        low=linprog(direction,A_ub=equations,b_ub=rhs,bounds=[(None,None)]*3,method='highs')
        high=linprog(-direction,A_ub=equations,b_ub=rhs,bounds=[(None,None)]*3,method='highs')
        require(low.success and high.success,'mask cones empty or unbounded; no arbitrary scene-scale fallback')
        bounds.append([low.fun,-high.fun])
    bounds=np.asarray(bounds)
    require(np.isfinite(bounds).all() and (np.diff(bounds,axis=1)>0).all(),'degenerate silhouette bounds')
    return bounds,masks,intrinsics


def visual_hull(views, side=512, grid=64, layers=3, primitive_cap=239171):
    require(grid>=12 and layers>=2,'insufficient hull grid/layers')
    bounds,masks,intrinsics=mask_cone_bounds(views,side)
    # Two padding voxels are derived from the finite mask-cone extent, not GT bounds.
    voxel=(bounds[:,1]-bounds[:,0])/(grid-4)
    lower=bounds[:,0]-2*voxel
    axes=[lower[a]+(np.arange(grid)+.5)*voxel[a] for a in range(3)]
    points=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3)
    occupied=np.ones(len(points),bool)
    for view,mask,k in zip(views,masks,intrinsics):
        pixel,z=project_numpy(points,view,k)
        valid=np.isfinite(pixel).all(1)&(z>0)&(pixel>=0).all(1)&(pixel<side).all(1)
        xy=np.floor(np.clip(np.nan_to_num(pixel),0,side-1)).astype(int)
        occupied &= valid&mask[xy[:,1],xy[:,0]]
    occupancy=occupied.reshape(grid,grid,grid)
    require(occupancy.any(),'empty visual hull; no geometry fallback')
    touch=any(np.take(occupancy,index,axis=axis).any() for axis in range(3) for index in (0,grid-1))
    require(not touch,'visual hull touches reconstruction grid boundary')
    boundary=occupancy&~binary_erosion(occupancy)
    field=gaussian_filter(occupancy.astype(float),.8)
    gradient=np.stack(np.gradient(field,*voxel),axis=-1)
    normals=-gradient[boundary]
    lengths=np.linalg.norm(normals,axis=1)
    require((lengths>1e-12).all(),'ambiguous hull normals; do not invent outward orientation')
    roots=points[boundary.ravel()];normals/=lengths[:,None]
    require(len(roots)*layers<=primitive_cap,'initial Shell exceeds cap; request reviewed grid change')
    # RGB estimate from alpha-supported outward-facing projections, not a teacher.
    colors=np.zeros_like(roots);weight=np.zeros(len(roots))
    for view in views:
        pixel,z=project_numpy(roots,view)
        size=view.rgba.shape[-1]
        valid=np.isfinite(pixel).all(1)&(z>0)&(pixel>=0).all(1)&(pixel<size).all(1)
        xy=np.floor(np.clip(np.nan_to_num(pixel),0,size-1)).astype(int)
        center=-view.rotation.T@view.translation
        towards=center-roots;towards/=np.linalg.norm(towards,axis=1,keepdims=True)
        w=np.maximum((normals*towards).sum(1),0)*valid
        rgba=view.rgba[:,xy[:,1],xy[:,0]].T.numpy()
        w*=rgba[:,3]
        colors+=w[:,None]*rgba[:,:3];weight+=w
    require((weight>0).all(),'hull has roots without observed RGB support')
    return dict(roots=roots.astype(np.float32),normals=normals.astype(np.float32),
        colors=(colors/weight[:,None]).astype(np.float32),spacing=float(np.cbrt(voxel.prod())),
        report=dict(schema='teacher-free-mask-hull-v1',grid=grid,mask_resolution=side,
            cone_bounds=bounds.tolist(),voxel_size=voxel.tolist(),occupied_voxels=int(occupied.sum()),
            shell_roots=len(roots),shell_layers=layers,initial_primitives=len(roots)*layers,
            boundary_touch=False,truncated_masks=False,
            semantics='RGB-alpha visual-hull outer envelope / learnable support; not skin/scalp ground truth',
            color_prior='normal-facing alpha-supported coarse RGB prior; not occlusion-certified visibility and never used as birth visibility',
            teacher_read=False,mvs_read=False,geometry_gt_read=False))
