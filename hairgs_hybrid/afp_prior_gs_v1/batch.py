"""Run the signed Round18 list with one persistent worker per idle GPU."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import fcntl
import os
from pathlib import Path
from queue import Empty, Queue
import re
import subprocess
from threading import Lock
import time

from .common import AFP, ROOT, read, sha, write
from .round18 import DATASET, TEST_IDS, load_case

CODE = Path(__file__).resolve().parents[2]
NATIVE = ROOT / 'compare_methods/3DGS/a025_20261015'
PRIOR_PYTHON = ROOT / 'compare_methods/HairGS/env/py310/bin/python'
NATIVE_PYTHON = NATIVE / 'env/py310/bin/python'
REGISTRY = ROOT / 'experiments/structured_fur_v2_20261016/panda_baselines/registry'
SELECTION = Path(__file__).with_name('round18_cases.json')
MODULE = 'hairgs_hybrid.afp_prior_gs_v1.round18'


def selected_cases(dataset, names):
    plan = read(SELECTION)
    assert sha(dataset / 'DATASET_PLAN.json') == plan['dataset_plan_sha256']
    selected = plan['cases']
    assert len(selected) == len(set(selected)) == 33
    assert all(re.fullmatch(r'[A-Za-z0-9_]+', name) for name in selected)
    assert set(names) <= set(selected) and len(names) == len(set(names))
    return [dataset / 'cases' / name for name in names or selected]


def gpu_rows():
    lines = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                                     '--format=csv,noheader,nounits'], text=True).splitlines()
    return [(int(i), uuid, int(mem), int(util)) for i, uuid, mem, util in
            (line.split(', ') for line in lines)]


def lock_gpus(stack, count):
    ready = []
    for index, uuid, _, _ in gpu_rows():
        lock = open(REGISTRY / f'{uuid}.lock', 'a')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            continue
        row = next(x for x in gpu_rows() if x[1] == uuid)
        if row[2] >= 200 or row[3] >= 5:
            lock.close()
            continue
        stack.enter_context(lock)
        ready.append((index, uuid))
        if len(ready) == count:
            return ready
    raise RuntimeError(f'Requested {count} idle GPUs; acquired {len(ready)}')


def run_stage(stage, source, case_dir, gpu, smoke):
    case_dir.mkdir(parents=True, exist_ok=True)
    script = ['-u', '-B', '-m', MODULE, stage, str(source), str(case_dir)]
    if smoke:
        script.append('--smoke')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED='1', PYTHONNOUSERSITE='1',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               TORCH_HOME=str(ROOT / 'cache/torch'), UNIFUR_ROOT=str(ROOT), AFP_ROOT=str(AFP))
    env.pop('PYTHONPATH', None)
    if stage == 'prior':
        env['PYTHONPATH'] = os.pathsep.join((str(CODE), str(AFP)))
        command = [str(PRIOR_PYTHON), *script]
    else:
        command = ['bash', '-c', 'source "$1"; export PYTHONPATH="$2:$PYTHONPATH"; shift 2; exec "$@"',
                   'native', str(NATIVE / 'env.sh'), str(CODE), str(NATIVE_PYTHON), *script]
    with (case_dir / f'{stage}.log').open('a') as log:
        print(f'\n{time.strftime("%Y-%m-%d %H:%M:%S %z")} GPU={gpu} stage={stage}', file=log, flush=True)
        subprocess.run(command, cwd=CODE, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def verify_case(path, name, smoke):
    proof = read(path / 'CASE_COMPLETE.json')
    train = read(path / 'TRAIN_COMPLETE.json')
    assert proof['status'] == 'complete' and proof['case'] == name and proof['smoke'] == smoke
    assert (train['free_steps'], train['postroute_steps']) == ((3, 2) if smoke else (30000, 2000))
    assert proof['asset_sha256'] == sha(path / 'complete_asset.npz')
    assert proof['final_sha256'] == train['final_sha256'] == sha(path / 'final.npz')
    assert [r['view_id'] for r in proof['test_views']] == [f'{i:03d}' for i in TEST_IDS]
    for row in proof['test_views']:
        assert row['image'] == f"test/{row['view_id']}.png"
        assert row['sha256'] == sha(path / row['image'])
    return proof


def dispatch(sources, output, gpu_count, smoke):
    output.mkdir(parents=True, exist_ok=True)
    mode = 'smoke' if smoke else 'reconstruction'
    pending = Queue()
    status = {}
    for source in sources:
        case_dir = output / 'cases' / source.name / mode
        if (case_dir / 'CASE_COMPLETE.json').exists():
            verify_case(case_dir, source.name, smoke)
            status[source.name] = dict(state='complete')
        else:
            pending.put(source)
            status[source.name] = dict(state='pending')
    started = time.time()
    mutex = Lock()
    filename = output / ('SMOKE_STATUS.json' if smoke else 'BATCH_STATUS.json')
    locked = []

    def publish():
        write(filename, dict(mode=mode, started=started, updated=time.time(),
             gpus=dict(locked), cases=status))

    def set_status(source, **state):
        with mutex:
            status[source.name] = state
            publish()

    def worker(index, uuid):
        while True:
            try:
                source = pending.get_nowait()
            except Empty:
                return
            case_dir = output / 'cases' / source.name
            for attempt in (1, 2):
                try:
                    for stage in ('prior', 'fit', 'deliver'):
                        set_status(source, state='running', stage=stage, gpu=index, attempt=attempt)
                        run_stage(stage, source, case_dir, uuid, smoke)
                    verify_case(case_dir / mode, source.name, smoke)
                    set_status(source, state='complete', gpu=index, attempt=attempt)
                    break
                except (subprocess.CalledProcessError, OSError, ValueError, AssertionError) as error:
                    set_status(source, state='retrying' if attempt == 1 else 'failed',
                               stage=stage, gpu=index, attempt=attempt, error=str(error))

    with ExitStack() as stack:
        lock = stack.enter_context((output / f'{mode}.lock').open('a'))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not pending.empty():
            locked.extend(lock_gpus(stack, min(gpu_count, pending.qsize())))
            publish()
            with ThreadPoolExecutor(max_workers=len(locked)) as pool:
                workers = [pool.submit(worker, index, uuid) for index, uuid in locked]
                for future in workers:
                    future.result()
        publish()
    failed = [name for name, row in status.items() if row['state'] != 'complete']
    assert not failed, f'Failed cases: {failed}'
    complete = {p.name: verify_case(output / 'cases' / p.name / mode, p.name, smoke) for p in sources}
    write(output / ('SMOKE_COMPLETE.json' if smoke else 'BATCH_COMPLETE.json'),
          dict(status='complete', cases=complete, gpu_count=len(locked),
               images=sum(len(v['test_views']) for v in complete.values()), wall_seconds=time.time() - started))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DATASET)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpus', type=int, default=8)
    parser.add_argument('--case', action='append', default=[])
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    assert args.gpus > 0 and (not args.smoke or len(args.case) == 1)
    sources = selected_cases(args.dataset.resolve(), args.case)
    for source in sources:
        load_case(source)
    if args.check:
        print(f'{len(sources)} cases passed signed body and camera validation')
    else:
        dispatch(sources, args.output.resolve(), args.gpus, args.smoke)


if __name__ == '__main__':
    main()
