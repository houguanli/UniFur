"""Continuous mesh-driven displacement instead of independent face-frame pivots.

No reconstruction training, opacity/GS pruning, warp-amplitude change or groom
input. Original material roots remain exact barycentric boundary conditions.
"""
from pathlib import Path
import sys
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'a025_surface_binding_stage_a_20261017'))
import binding as base


class MeshField:
    def __init__(self,rest,moving,device='cpu'):
        self.rest=np.asarray(rest,float);self.moving=np.asarray(moving,float);self.device=device
        if self.rest.shape!=self.moving.shape or not np.isfinite(self.moving).all():
            raise ValueError('invalid mesh deformation')
        self.identity=np.array_equal(self.rest,self.moving)
        self.rbf=None if self.identity else RBFInterpolator(self.rest,self.moving-self.rest,kernel='thin_plate_spline',degree=1)
        if self.rbf is not None:
            # SciPy is pinned by the experiment; verify its private coefficient
            # convention against the public evaluator before using chunked GPU math.
            assert np.array_equal(self.rbf.powers,[[0,0,0],[1,0,0],[0,1,0],[0,0,1]])
            probe=self.rest[::max(1,len(rest)//17)]
            u,_=self.evaluate(probe)
            assert np.max(abs(u-self.rbf(probe)))<1e-8, 'RBF coefficient layout changed'

    def evaluate(self,points,jacobian=False):
        import torch
        points=np.asarray(points,float)
        if self.identity:
            return np.zeros_like(points),np.broadcast_to(np.eye(3),(len(points),3,3)).copy() if jacobian else None
        def t(x):return torch.as_tensor(x,dtype=torch.float64,device=self.device)
        centers=t(self.rest);coef=t(self.rbf._coeffs[:-4]);poly=t(self.rbf._coeffs[-4:])
        shift=t(self.rbf._shift);scale=t(self.rbf._scale)
        derivative_coef=(centers[:,:,None]*coef[:,None,:]).reshape(len(centers),9)
        values=[];gradients=[]
        with torch.no_grad():
            for begin in range(0,len(points),256):
                x=t(points[begin:begin+256])
                squared=(x.square().sum(1)[:,None]+centers.square().sum(1)[None]-2*x@centers.T).clamp_min(1e-28)
                log=squared.log()
                u=.5*(squared*log)@coef+poly[0]+((x-shift)/scale)@poly[1:]
                values.append(u.cpu().numpy())
                if jacobian:
                    kernel=log+1
                    # derivative d u_output / d x_input; transpose to (N,out,in)
                    g=x[:,:,None]*(kernel@coef)[:,None,:]-(kernel@derivative_coef).reshape(-1,3,3)
                    g+=poly[1:][None]/scale[None,:,None]
                    gradients.append((torch.eye(3,device=self.device)[None]+g.transpose(1,2)).cpu().numpy())
        return np.concatenate(values),np.concatenate(gradients) if jacobian else None


def transport_quaternion(old_axis,new_axis):
    a=old_axis/np.linalg.norm(old_axis,axis=1,keepdims=True)
    b=new_axis/np.linalg.norm(new_axis,axis=1,keepdims=True)
    dot=np.clip((a*b).sum(1),-1,1)
    q=np.c_[1+dot,np.cross(a,b)]
    opposite=dot<-.999999
    if opposite.any():
        helper=np.zeros_like(a[opposite]);helper[:,0]=1
        helper[np.abs(a[opposite,0])>.9]=[0,1,0]
        axis=np.cross(a[opposite],helper);axis/=np.linalg.norm(axis,axis=1,keepdims=True)
        q[opposite]=np.c_[np.zeros(len(axis)),axis]
    return q/np.linalg.norm(q,axis=1,keepdims=True)


def deform(asset,vertices,field=None):
    vertices=np.asarray(vertices,float)
    root0,_=base.anchors(asset,asset['body_vertices']);root,_=base.anchors(asset,vertices)
    field=field or MeshField(asset['body_vertices'],vertices)
    result={k:asset[k].copy() for k in base.FIELDS}
    if field.identity:return result,asset['nodes'].copy()
    u_root,_=field.evaluate(root0)
    correction=root-root0-u_root  # root pin, not a face-frame rotation of the rest offset
    u,jac=field.evaluate(asset['xyz'],jacobian=True)
    result['xyz']=(asset['xyz']+u+correction[asset['gs_guide']]).astype(np.float32)
    un,_=field.evaluate(asset['nodes'])
    nodes=asset['nodes']+un+correction[asset['node_guide']]

    shell=np.flatnonzero(asset['route']==0)
    R0=Rotation.from_quat(asset['rotation'][shell][:,[1,2,3,0]]).as_matrix()
    L=jac[shell]@R0*np.exp(asset['scale'][shell,None,:])
    rotation,sigma,_=np.linalg.svd(L)
    rotation[np.linalg.det(rotation)<0,:,0]*=-1
    result['scale'][shell]=np.log(sigma)
    result['rotation'][shell]=Rotation.from_matrix(rotation).as_quat()[:,[3,0,1,2]]

    rows=asset['strand_rows'];edge=asset['edges']
    p0,p1=nodes[edge[:,0]],nodes[edge[:,1]];direction=p1-p0
    length=np.linalg.norm(direction,axis=1)
    if (length<1e-10).any():raise ValueError('collapsed Strand after deformation')
    # Transport material twist from the smooth Jacobian first, then align its
    # longitudinal axis with the actual shared-node edge. Tangent-only rotation
    # would lose body twist for elliptical Strand cross-sections.
    U,_,Vt=np.linalg.svd(jac[rows]);material=U@Vt
    if (np.linalg.det(material)<0).any():raise ValueError('orientation-reversing material field')
    body_q=Rotation.from_matrix(material).as_quat()[:,[3,0,1,2]]
    candidate=base.multiply_quat(body_q,asset['rotation'][rows])
    candidate_axis=Rotation.from_quat(candidate[:,[1,2,3,0]]).as_matrix()[:,:,2]
    dq=transport_quaternion(candidate_axis,direction)
    result['xyz'][rows]=(p0+p1)/2
    result['rotation'][rows]=base.multiply_quat(dq,candidate)
    result['scale'][rows,2]=np.log(length/4)
    # Preserve learned transverse widths, appearance, opacity, order and count.
    # The physical rod decoder is not a general sheared volume Gaussian.
    return result,nodes


def selftest():
    rng=np.random.default_rng(4)
    body=rng.normal(size=(40,3));moving=body.copy();moving[:,2]+=.03*np.sin(body[:,0])
    f=MeshField(body,moving)
    point=rng.normal(size=(20,3));u,J=f.evaluate(point,True)
    assert np.max(abs(u-f.rbf(point)))<1e-8
    eps=1e-5
    for axis in range(3):
        delta=np.eye(3)[axis]*eps
        diff=(f.evaluate(point+delta)[0]-f.evaluate(point-delta)[0])/(2*eps)
        assert np.max(abs(diff-(J[:,:,axis]-np.eye(3)[None,:,axis])))<1e-7
    R=Rotation.from_euler('xyz',[.2,.1,-.1]).as_matrix();translation=np.array([.1,.2,.3])
    rigid=MeshField(body,body@R.T+translation)
    d,G=rigid.evaluate(point,True)
    assert np.max(abs(point+d-(point@R.T+translation)))<1e-8
    assert np.max(abs(G-R))<1e-8
    q=transport_quaternion(np.array([[0.,0,1],[0,0,1]]),np.array([[0.,1,0],[0,0,-1]]))
    assert np.max(abs(Rotation.from_quat(q[:,[1,2,3,0]]).apply([[0,0,1],[0,0,1]])-[[0,1,0],[0,0,-1]]))<1e-8
    v=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]])
    faces=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]])
    qq=Rotation.from_euler('z',[.1,.3,.4]).as_quat()[:,[3,0,1,2]].astype(np.float32)
    a=dict(body_vertices=v,body_faces=faces,anchor_face=np.array([3,0]),anchor_bary=np.array([[1/3]*3,[.6,.2,.2]]),
        gs_guide=np.array([0,1,1]),node_guide=np.array([1,1,1]),nodes=np.array([[.2,.2,0],[.2,.2,.2],[.2,.2,.4]]),
        edges=np.array([[0,1],[1,2]]),strand_rows=np.array([1,2]),route=np.array([0,1,1]),
        xyz=np.array([[.4,.3,.4],[.2,.2,.1],[.2,.2,.3]],np.float32),
        scale=np.log(np.array([[.02,.04,.01],[.01,.02,.05],[.01,.02,.05]],np.float32)),rotation=qq,
        opacity=np.zeros((3,1),np.float32),dc=np.zeros((3,1,3),np.float32),rest=np.zeros((3,15,3),np.float32))
    zero,nodes=deform(a,v)
    assert all(np.array_equal(zero[k],a[k]) for k in base.FIELDS)
    value,nodes=deform(a,v@R.T+translation)
    assert np.max(abs(value['xyz']-(a['xyz']@R.T+translation)))<1e-7
    assert np.max(abs(base.covariance(value)-R[None]@base.covariance(a)@R.T[None]))<1e-8
    assert np.max(abs(nodes-(a['nodes']@R.T+translation)))<1e-8
    print('PASS: public SciPy interpolation/Jacobian, affine/rigid reproduction, actual rod/shell decoder, material twist/full covariance, shared nodes, exact rest')


if __name__=='__main__':selftest()
