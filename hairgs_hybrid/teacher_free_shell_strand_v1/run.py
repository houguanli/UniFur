"""Exclusive-output CPU preflight / owner-executed native smoke / fresh training."""
import argparse
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image
import torch

from hairgs_hybrid.rgb_only_init_v1.core import require,sha
from .inputs import load_train_views,visual_hull
from .model import ShellStrand
from .training import Trainer,Schedule,native_renderer,camera_for

REPO=Path(__file__).resolve().parents[2]


def write_json(path,value):
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False)


def runtime_manifest(path,digest):
    require(sha(path)==digest,'run configuration hash mismatch')
    config=json.loads(path.read_text())
    require(config['schema']=='teacher-free-shell-strand-run-v1','wrong run configuration')
    require(config['primitive_cap']==239171 and config['shell_layers']==3 and config['max_sh_degree']==2,'unreviewed representation settings')
    expected={str(p.relative_to(REPO)) for p in Path(__file__).parent.glob('*.py')}
    expected.add('hairgs_hybrid/rgb_only_init_v1/core.py')
    expected.add('third_party/hair-gs/gaussian_renderer/__init__.py')
    require(set(config['code_sha256'])==expected,'source file pin set differs')
    for relative,expected_sha in config['code_sha256'].items():
        require(sha(REPO/relative)==expected_sha,'source changed since run freeze: '+relative)
    Schedule(**config['schedule']).at(0);Schedule(**config['smoke_schedule']).at(0)
    return config


def native_provenance():
    renderer=importlib.import_module('gaussian_renderer')
    raster=importlib.import_module('diff_gaussian_rasterization')
    extension=importlib.import_module('diff_gaussian_rasterization._C')
    expected=REPO/'third_party/hair-gs/gaussian_renderer/__init__.py'
    require(Path(renderer.__file__).resolve()==expected.resolve(),'renderer is not the repository HairGS source')
    return {name:dict(path=str(Path(module.__file__).resolve()),sha256=sha(module.__file__))
            for name,module in [('gaussian_renderer',renderer),('diff_gaussian_rasterization',raster),('raster_extension',extension)]}


def memory_sample(step):
    free,total=torch.cuda.mem_get_info()
    result=subprocess.run(['nvidia-smi','--query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu',
                           '--format=csv,noheader,nounits','-i','0'],capture_output=True,text=True,timeout=10,check=True)
    return dict(step=step,wall_time=time.time(),allocated=torch.cuda.memory_allocated(),
        reserved=torch.cuda.memory_reserved(),max_allocated=torch.cuda.max_memory_allocated(),
        max_reserved=torch.cuda.max_memory_reserved(),device_free=free,device_total=total,nvidia_smi=result.stdout.strip())


def snapshot(output,trainer,stage,gate):
    model=trainer.model;view=trainer.views[(trainer.step_index-1)%len(trainer.views)]
    side=(512,1024,2048)[stage];camera=camera_for(view,side,model)
    with torch.no_grad():
        batch=model.primitives(stage,gate)
        rgb=native_renderer(camera,batch)['render'];alpha=native_renderer(camera,batch,torch.ones_like(batch.get_xyz))['render'][:1]
        require(torch.isfinite(rgb).all() and torch.isfinite(alpha).all(),'nonfinite native RGB/alpha snapshot')
        alpha_min,alpha_max=float(alpha.min()),float(alpha.max())
        require(alpha_min>=-1e-6 and alpha_max<=1+1e-5,'native alpha outside [0,1] tolerance')
        require(all(torch.isfinite(value).all() for value in (batch.get_xyz,batch.get_scaling,batch.get_rotation,batch.get_opacity,batch.get_features)),
                'nonfinite native primitive parameters')
        local=batch.get_xyz@camera.rotation.T+camera.translation;depth=local[:,2]
        kk=local.new_tensor(camera.intrinsics)
        pixel=local[:,:2]/torch.where(depth.abs()>1e-9,depth,torch.ones_like(depth))[:,None]*kk[:2]+kk[2:]
        finite=torch.isfinite(pixel).all(1)&torch.isfinite(depth)&(depth.abs()>1e-9)
        positive=finite&(depth>0)
        in_frame=positive&(pixel>=0).all(1)&(pixel<side).all(1)
        numerical_audit=dict(render_finite=True,alpha_finite=True,alpha_min=alpha_min,alpha_max=alpha_max,
            render_preclamp_min=float(rgb.min()),render_preclamp_max=float(rgb.max()),
            projected_finite_fraction=float(finite.float().mean()),positive_depth_fraction=float(positive.float().mean()),
            in_frame_positive_fraction=float(in_frame.float().mean()),projection_semantics='corner-origin K; native array center u-.5')
    folder=output/f'stage_{side}';folder.mkdir(exist_ok=False)
    Image.fromarray((rgb.clamp(0,1).permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)).save(folder/'render.png')
    Image.fromarray((alpha[0].clamp(0,1).cpu().numpy()*255).round().astype(np.uint8)).save(folder/'alpha.png')
    with (folder/'geometry.npz').open('xb') as stream:
        np.savez_compressed(stream,xyz=batch.get_xyz.detach().cpu().numpy(),scaling=batch.get_scaling.detach().cpu().numpy(),
            rotation=batch.get_rotation.detach().cpu().numpy(),opacity=batch.get_opacity.detach().cpu().numpy(),
            features=batch.get_features.detach().cpu().numpy(),curve_roots=model.curve_roots.cpu().numpy(),
            curve_counts=model.curve_counts.cpu().numpy())
    return dict(side=side,step=trainer.step_index,view=view.name,actual_rendered_primitives=len(batch.get_xyz),
                numerical_audit=numerical_audit,
                shell_primitives=len(model.shell_root)*model.layers,strand_primitives=len(model.strand_control),
                output_sha256={p.name:sha(p) for p in folder.iterdir()})


def execute(stage,config_path,config_sha,output):
    config=runtime_manifest(config_path,config_sha)
    require(output.resolve().parent.name=='animals60_a050_teacher_free_shell_strand_v1','wrong independent output family')
    require(not output.exists(),'exclusive new output required; no overwrites/resume implied')
    if stage=='train':require(config['status']=='approved_for_fresh_formal_training','formal training not approved in manifest')
    output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();torch.manual_seed(config['seed']);np.random.seed(config['seed'])
    views,source=load_train_views(config['source_lock'],config['source_lock_sha256'])
    hull=visual_hull(views,grid=config['hull_grid'],layers=config['shell_layers'],primitive_cap=config['primitive_cap'])
    model=ShellStrand(hull['roots'],hull['normals'],hull['colors'],hull['spacing'],layers=config['shell_layers'],cap=config['primitive_cap'])
    report=dict(schema='teacher-free-execution-v1',stage=stage,run_config_sha256=config_sha,source=source,
        initialization=hull['report'],initial_strand_primitives=len(model.strand_control),
        initial_shell_primitives=model.primitive_count,cpu_initialization_seconds=time.perf_counter()-started,
        teacher_data_reads=0,mvs_data_reads=0,geometry_gt_data_reads=0,test_data_reads=0,
        input_read_claim_basis='pure source allowlist loader + CPU isolation tests; not an OS-wide syscall audit',
        strict_resume_supported=False,diagnostic_only=stage!='train')
    write_json(output/'step0.json',report)
    with (output/'step0_state.pt').open('xb') as stream:torch.save(model.state_dict(),stream)
    if stage=='preflight':
        report.update(status='cpu_initialization_pass_gpu_unverified',step0_state_sha256=sha(output/'step0_state.pt'))
        write_json(output/'report.json',report);return report
    require(torch.cuda.is_available(),'CUDA unavailable; do not substitute CPU for native smoke')
    require(torch.cuda.mem_get_info()[0]>=config['minimum_free_gpu_mib']*2**20,'insufficient currently free GPU memory; leave UE untouched')
    report['native_runtime']=native_provenance()
    model=model.to('cuda');model.setup_optimizer()
    schedule=Schedule(**config['smoke_schedule' if stage=='smoke' else 'schedule'])
    trainer=Trainer(model,views,schedule)
    torch.cuda.reset_peak_memory_stats();memory=[memory_sample(-1)];snapshots=[];forced_split=0
    try:
        with (output/'steps.jsonl').open('x') as log:
            while trainer.step_index<schedule.total_steps:
                record=trainer.step();torch.cuda.synchronize()
                # Smoke checks topology numerics, not an inferred split-quality decision.
                if stage=='smoke' and record['stage']==2 and len(model.strand_control) and not forced_split:
                    forced_split=model.split_segments([0]);trainer.evidence=type(trainer.evidence)(model,[v.view_id for v in views])
                    record['smoke_only_split_exercise']=forced_split
                log.write(json.dumps(record,allow_nan=False)+'\n');log.flush()
                memory.append(memory_sample(trainer.step_index))
                require(memory[-1]['device_free']>=config['reserve_gpu_mib']*2**20,'GPU headroom below reserved margin')
                boundaries=(schedule.stage1024,schedule.stage2048,schedule.total_steps)
                if trainer.step_index in boundaries:
                    snapshots.append(snapshot(output,trainer,record['stage'],record['gate']))
                    torch.cuda.reset_peak_memory_stats()
        if stage=='smoke':
            require([s['side'] for s in snapshots]==[512,1024,2048],'missing native LOD stage')
            require(len(model.strand_control)>0 and forced_split>0,'native smoke did not produce evidence-backed birth and real split; do not fabricate a curve to pass')
        with (output/'diagnostic_checkpoint.pt').open('xb') as stream:torch.save(trainer.checkpoint(),stream)
        report.update(status='native_smoke_pass' if stage=='smoke' else 'fresh_training_finished',
            schedule=asdict(schedule),steps=trainer.step_index,snapshots=snapshots,
            final_shell_primitives=len(model.shell_root)*model.layers,final_strand_primitives=len(model.strand_control),
            primitive_count=model.primitive_count,unused_capacity=model.cap-model.primitive_count,
            smoke_split_is_topology_exercise_not_quality_evidence=bool(forced_split),
            loss_last=trainer.history[-1],wall_seconds=time.perf_counter()-started)
    except Exception as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}',steps=trainer.step_index,snapshots=snapshots)
        write_json(output/'memory.json',memory);write_json(output/'report.json',report)
        raise
    write_json(output/'memory.json',memory);write_json(output/'report.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['preflight','smoke','train'])
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--config-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(execute(args.stage,args.config,args.config_sha256,args.output.resolve()),allow_nan=False))


if __name__=='__main__':main()
