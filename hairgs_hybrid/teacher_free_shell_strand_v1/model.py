"""Independent coherent Shell columns and dynamically born connected curves."""
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix

from hairgs_hybrid.rgb_only_init_v1.core import require

C0=0.28209479177387814


def frame(x, guide):
    x=F.normalize(x,dim=-1)
    helper=F.one_hot(x.abs().argmin(-1),3).to(x)
    z=guide-(guide*x).sum(-1,keepdim=True)*x
    fallback=helper-(helper*x).sum(-1,keepdim=True)*x
    z=F.normalize(torch.where(z.norm(dim=-1,keepdim=True)>1e-5,z,fallback),dim=-1)
    y=F.normalize(torch.cross(z,x,dim=-1),dim=-1)
    return torch.stack((x,y,torch.cross(x,y,dim=-1)),dim=-1)


@dataclass
class GaussianBatch:
    get_xyz: torch.Tensor
    get_scaling: torch.Tensor
    get_rotation: torch.Tensor
    get_opacity: torch.Tensor
    get_features: torch.Tensor
    active_sh_degree: int
    max_sh_degree: int=2

    def get_covariance(self, scaling_modifier=1.):
        rotation=quaternion_to_matrix(self.get_rotation)
        transform=rotation*self.get_scaling[:,None,:]*scaling_modifier
        covariance=transform@transform.transpose(1,2)
        return covariance[:,[0,0,0,1,1,2],[0,1,2,1,2,2]]


class ShellStrand(nn.Module):
    shell_names=('shell_root','shell_rotation','shell_spacing','shell_scale','shell_dc','shell_rest','shell_opacity')
    strand_names=('strand_control','strand_scale','strand_twist','strand_dc','strand_rest','strand_opacity')

    def __init__(self, roots, normals, colors, spacing, layers=3, cap=239171, device='cpu'):
        super().__init__()
        roots=torch.as_tensor(roots,dtype=torch.float32,device=device)
        normals=torch.as_tensor(normals,dtype=torch.float32,device=device)
        colors=torch.as_tensor(colors,dtype=torch.float32,device=device)
        require(roots.ndim==2 and roots.shape[1]==3 and len(roots)>0,'empty/invalid Shell support')
        require(normals.shape==colors.shape==roots.shape and torch.isfinite(roots).all() and
                torch.isfinite(normals).all() and torch.isfinite(colors).all() and (normals.norm(dim=1)>0).all(),
                'invalid support normals/colors')
        require(layers>=2 and spacing>0 and len(roots)*layers<=cap,'invalid Shell thickness/budget')
        self.layers=layers;self.cap=cap;self.spacing=float(spacing);self.generation=0
        normals=F.normalize(normals,dim=1)
        helper=F.one_hot(normals.abs().argmin(1),3).to(roots)
        tangent=F.normalize(torch.cross(helper,normals,dim=1),dim=1)
        rotation=matrix_to_quaternion(frame(tangent,normals))
        parameter=lambda value:nn.Parameter(value.contiguous())
        self.shell_root=parameter(roots.clone());self.shell_rotation=parameter(rotation)
        self.shell_spacing=parameter(roots.new_full((len(roots),layers-1),float(np.log(np.expm1(.2)))))
        self.shell_scale=parameter(roots.new_tensor([np.log(spacing*.65),np.log(spacing*.65),-1.5]).expand(len(roots),layers,3).clone())
        self.shell_dc=parameter(((colors-.5)/C0)[:,None,None,:].expand(-1,layers,1,-1).clone())
        self.shell_rest=parameter(roots.new_zeros((len(roots),layers,8,3)))
        self.shell_opacity=parameter(roots.new_full((len(roots),layers,1),float(np.log(.15/.85))))
        # No reserved curves, endpoints, coefficients or candidate geometry at step zero.
        self.strand_control=parameter(roots.new_empty((0,3)))
        self.strand_scale=parameter(roots.new_empty((0,3)))
        self.strand_twist=parameter(roots.new_empty((0,1)))
        self.strand_dc=parameter(roots.new_empty((0,1,3)))
        self.strand_rest=parameter(roots.new_empty((0,8,3)))
        self.strand_opacity=parameter(roots.new_empty((0,1)))
        self.register_buffer('curve_roots',torch.empty(0,dtype=torch.long,device=device))
        self.register_buffer('curve_counts',torch.empty(0,dtype=torch.long,device=device))
        self.optimizer=None

    @property
    def primitive_count(self):return len(self.shell_root)*self.layers+len(self.strand_control)

    def setup_optimizer(self, geometry_lr=1e-3, appearance_lr=2.5e-3):
        groups=[]
        for name,p in self.named_parameters():
            geometry=any(token in name for token in ('root','rotation','spacing','control','scale','twist'))
            base=geometry_lr if geometry else appearance_lr
            # World coordinates are in camera units, not an assumed normalized scene.
            if name in ('shell_root','strand_control'):base*=self.spacing
            groups.append(dict(params=[p],name=name,lr=base,base_lr=base))
        self.optimizer=torch.optim.Adam(groups,lr=0,eps=1e-15)
        return self.optimizer

    def set_shell_lr(self, factor):
        require(0<factor<=1,'Shell must remain learnable')
        for group in self.optimizer.param_groups:
            group['lr']=group['base_lr']*(factor if group['name'].startswith('shell_') else 1.)

    def shell_geometry(self):
        rotation=quaternion_to_matrix(F.normalize(self.shell_rotation,dim=1))
        gaps=F.softplus(self.shell_spacing)*self.spacing
        height=torch.cat((gaps.new_zeros((len(gaps),1)),gaps.cumsum(1)),dim=1)
        height=height-height[:,-1:]  # last layer is the learnable outer support/root
        xyz=self.shell_root[:,None,:]+height[:,:,None]*rotation[:,None,:,2]
        tangent=self.shell_scale[...,:2].exp()
        thickness=tangent.min(-1,keepdim=True).values*.25*torch.sigmoid(self.shell_scale[...,2:3])
        return xyz,torch.cat((tangent,thickness),dim=-1),rotation

    def curve_geometry(self):
        if not len(self.strand_control):return self.strand_control,self.strand_control
        root_ids=torch.repeat_interleave(self.curve_roots,self.curve_counts)
        roots=self.shell_root[root_ids]
        # Control offsets are free world-space vectors, so bending/outward growth is possible.
        end=roots+self.strand_control
        start=roots+torch.roll(self.strand_control,1,0)
        first=torch.cat((self.curve_counts.new_zeros(1),self.curve_counts.cumsum(0)[:-1]))
        start=torch.index_copy(start,0,first,roots[first])
        return start,end

    def primitives(self, stage=0, strand_gate=1., route='all', detach_shell=False):
        require(stage in (0,1,2) and route in ('all','shell','strand'),'invalid stage/route')
        require(0<strand_gate<=1,'new Strand opacity gate must not be zero')
        xyz,scales,rotation=self.shell_geometry()
        shell=[xyz.reshape(-1,3),scales.reshape(-1,3),
               F.normalize(self.shell_rotation,dim=1)[:,None,:].expand(-1,self.layers,-1).reshape(-1,4),
               self.shell_opacity.sigmoid().reshape(-1,1),
               torch.cat((self.shell_dc,self.shell_rest),dim=2).reshape(-1,9,3)]
        # Detail-path detach includes root geometry used by the curves below.
        if detach_shell:shell=[v.detach() for v in shell]
        degree=0 if stage==0 else stage
        if route=='shell' or not len(self.strand_control) or stage==0:
            require(route!='strand','Strand route unavailable at this stage or empty')
            return GaussianBatch(*shell,active_sh_degree=degree)
        starts,ends=self.curve_geometry()
        if detach_shell:
            ids=torch.repeat_interleave(self.curve_roots,self.curve_counts)
            roots=self.shell_root[ids]
            starts=starts-roots+roots.detach();ends=ends-roots+roots.detach()
        delta=ends-starts;length=delta.norm(dim=1,keepdim=True).clamp_min(self.spacing*1e-6)
        rootids=torch.repeat_interleave(self.curve_roots,self.curve_counts)
        guide=rotation[rootids,:,2]
        if detach_shell:guide=guide.detach()
        basis=frame(delta,guide)
        twist=self.strand_twist;cos=twist.cos();sin=twist.sin()
        x,y,z=basis[:,:,0],basis[:,:,1],basis[:,:,2]
        basis=torch.stack((x,cos*y+sin*z,-sin*y+cos*z),dim=2)
        strand=[(starts+ends)/2,torch.cat((length*.5*self.strand_scale[:,:1].exp(),self.strand_scale[:,1:].exp()),dim=1),
                matrix_to_quaternion(basis),self.strand_opacity.sigmoid()*strand_gate,
                torch.cat((self.strand_dc,self.strand_rest),dim=1)]
        packed=strand if route=='strand' else [torch.cat((a,b),dim=0) for a,b in zip(shell,strand)]
        return GaussianBatch(*packed,active_sh_degree=degree)

    @torch.no_grad()
    def _replace(self,name,values,old_rows):
        """HairGS-style tensor Adam: preserve shared step and old moments; zero new rows.

        New rows share the tensor's existing bias-correction age, NOT fresh-row Adam.
        The first ever Strand birth starts at step zero because its tensor was empty.
        """
        old=getattr(self,name);new=nn.Parameter(values.detach().clone())
        if self.optimizer is not None:
            group=next(g for g in self.optimizer.param_groups if g['name']==name)
            state=self.optimizer.state.pop(old,{})
            if state:
                keep=old_rows>=0
                for key in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
                    if key in state:
                        moment=torch.zeros_like(new)
                        moment[keep]=state[key][old_rows[keep]]
                        state[key]=moment
                self.optimizer.state[new]=state
            group['params']=[new]
        setattr(self,name,new)

    @torch.no_grad()
    def birth(self,root_ids,directions,stage,segments=3):
        require(stage>0,'Strand birth forbidden during512 Shell warmup')
        require(segments>=2,'birth must create a multisegment curve')
        root_ids=torch.as_tensor(root_ids,dtype=torch.long,device=self.shell_root.device)
        directions=torch.as_tensor(directions,dtype=self.shell_root.dtype,device=self.shell_root.device)
        require(directions.shape==(len(root_ids),3) and torch.isfinite(directions).all() and
                (directions.norm(dim=1)>0).all() and ((root_ids>=0)&(root_ids<len(self.shell_root))).all(), 'invalid birth evidence')
        count=min(len(root_ids),(self.cap-self.primitive_count)//segments)
        if count==0:return 0
        root_ids=root_ids[:count];directions=F.normalize(directions[:count],dim=1)
        distances=torch.arange(1,segments+1,device=directions.device).to(directions)*self.spacing*.6
        controls=(directions[:,None,:]*distances[None,:,None]).reshape(-1,3)
        n=count*segments;rgb=self.shell_dc[root_ids,-1].repeat_interleave(segments,0)
        additions=dict(strand_control=controls,
            strand_scale=directions.new_tensor([0,np.log(self.spacing*.08),np.log(self.spacing*.08)]).expand(n,3),
            strand_twist=directions.new_zeros((n,1)),strand_dc=rgb,strand_rest=directions.new_zeros((n,8,3)),
            strand_opacity=directions.new_full((n,1),float(np.log(.05/.95))))
        oldlen=len(self.strand_control)
        indices=torch.cat((torch.arange(oldlen,device=directions.device),torch.full((n,),-1,device=directions.device)))
        for name,value in additions.items():self._replace(name,torch.cat((getattr(self,name),value)),indices)
        self.curve_roots=torch.cat((self.curve_roots,root_ids))
        self.curve_counts=torch.cat((self.curve_counts,root_ids.new_full((count,),segments)))
        self.generation+=1
        return n

    @torch.no_grad()
    def split_segments(self,selected):
        selected=set(int(i) for i in selected)
        require(all(0<=i<len(self.strand_control) for i in selected),'invalid split segment')
        selected=set(sorted(selected)[:self.cap-self.primitive_count])
        if not selected:return 0
        controls=[];oldrows=[];attributes=[];counts=[];offset=0
        for count in self.curve_counts.tolist():
            extra=0
            for local in range(count):
                index=offset+local
                if index in selected:
                    previous=self.strand_control[index-1] if local else torch.zeros_like(self.strand_control[index])
                    controls.append((previous+self.strand_control[index])/2);oldrows.append(-1);attributes.append(index);extra+=1
                controls.append(self.strand_control[index]);oldrows.append(index);attributes.append(index)
            counts.append(count+extra);offset+=count
        rows=torch.tensor(oldrows,device=self.shell_root.device)
        attr=torch.tensor(attributes,device=self.shell_root.device)
        values={name:getattr(self,name)[attr].clone() for name in self.strand_names}
        values['strand_control']=torch.stack(controls)
        for name,value in values.items():self._replace(name,value,rows)
        self.curve_counts=self.curve_counts.new_tensor(counts);self.generation+=1
        return len(selected)

    @torch.no_grad()
    def prune_curves(self,remove):
        remove=torch.as_tensor(remove,dtype=torch.bool,device=self.shell_root.device)
        require(remove.shape==self.curve_counts.shape,'prune whole curves, not disconnected segments')
        keep=torch.repeat_interleave(~remove,self.curve_counts)
        rows=torch.nonzero(keep).flatten()
        removed=int((~keep).sum())
        for name in self.strand_names:self._replace(name,getattr(self,name)[rows],rows)
        self.curve_roots=self.curve_roots[~remove];self.curve_counts=self.curve_counts[~remove]
        self.generation+=int(removed>0)
        return removed

    @torch.no_grad()
    def refine_shell(self,selected,split=True):
        selected=torch.as_tensor(selected,dtype=torch.long,device=self.shell_root.device)
        require(len(selected.unique())==len(selected) and ((selected>=0)&(selected<len(self.shell_root))).all(),'invalid column refinement')
        require(not split or not torch.isin(selected,self.curve_roots).any(),'do not split columns with attached curves')
        selected=selected[:(self.cap-self.primitive_count)//self.layers]
        if not len(selected):return 0
        oldn=len(self.shell_root);allrows=torch.arange(oldn,device=selected.device)
        kept=allrows[~torch.isin(allrows,selected)] if split else allrows
        source=torch.cat((kept,selected,selected)) if split else torch.cat((kept,selected))
        oldrows=torch.cat((kept,selected.new_full((len(source)-len(kept),),-1)))
        values={name:getattr(self,name)[source].clone() for name in self.shell_names}
        rotation=quaternion_to_matrix(F.normalize(self.shell_rotation[selected],dim=1))
        shift=rotation[:,:,0]*self.shell_scale[selected,0,:1].exp()*.35
        if split:
            values['shell_root'][-2*len(selected):-len(selected)]-=shift
            values['shell_root'][-len(selected):]+=shift
            values['shell_scale'][-2*len(selected):,:,:2]-=np.log(1.6)
        else:values['shell_root'][-len(selected):]+=shift
        remap=allrows.new_full((oldn,),-1);remap[kept]=torch.arange(len(kept),device=kept.device)
        if len(self.curve_roots):self.curve_roots=remap[self.curve_roots]
        for name,value in values.items():self._replace(name,value,oldrows)
        self.generation+=1
        return len(selected)*self.layers

    @torch.no_grad()
    def prune_shell(self,remove):
        remove=torch.as_tensor(remove,dtype=torch.bool,device=self.shell_root.device)
        require(remove.shape==(len(self.shell_root),) and (~remove).any(),'cannot prune all Shell support')
        require(not remove[self.curve_roots].any(),'cannot prune a root with an attached curve')
        rows=torch.nonzero(~remove).flatten();remap=torch.full_like(remove,-1,dtype=torch.long)
        remap[rows]=torch.arange(len(rows),device=rows.device)
        self.curve_roots=remap[self.curve_roots]
        for name in self.shell_names:self._replace(name,getattr(self,name)[rows],rows)
        removed=int(remove.sum())*self.layers;self.generation+=int(removed>0)
        return removed
