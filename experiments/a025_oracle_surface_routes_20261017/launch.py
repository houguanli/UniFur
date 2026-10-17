"""Two unique GPU jobs; train -> frozen export/metrics, no automatic retraining."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = Path('/root/Public/h84435299/UniFur')
NATIVE = ROOT / 'compare_methods/3DGS/a025_20261015'
PYTHON = NATIVE / 'env/py310/bin/python'


def write(path, value):
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(value, indent=2) + '\n'); tmp.replace(path)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--arm', required=True, choices=['free_then_route', 'born_routed']); p.add_argument('--gpu', required=True)
    a = p.parse_args(); registry = HERE / 'registry'; registry.mkdir(exist_ok=True)
    with (registry / (a.arm + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = registry / (a.arm + '.json'); assert not path.exists(), 'unique run already registered'
        lines = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).splitlines()
        rows = [s.split(',') for s in lines if s.split(',')[0].strip() == a.gpu]
        assert len(rows) == 1 and int(rows[0][1]) < 200 and int(rows[0][2]) < 5, 'GPU no longer idle'
        env = os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES=a.gpu, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
                                           TORCH_HOME=str(ROOT / 'cache/torch'))
        state = dict(status='running', arm=a.arm, GPU_UUID=a.gpu, PID=os.getpid(), phase='admitted', started_unix=time.time())
        write(path, state)
        try:
            for mode in ['smoke', 'train', 'evaluate']:
                script = HERE / ('evaluate.py' if mode == 'evaluate' else 'run.py')
                arguments = ([] if mode == 'evaluate' else [mode]) + ['--arm', a.arm]
                command = ['bash', '-c', 'source "$1"; shift; exec "$@"', 'surface-routes', str(NATIVE / 'env.sh'),
                           str(PYTHON), '-B', str(script), *arguments]
                state.update(phase=mode, phase_started_unix=time.time()); write(path, state)
                child = subprocess.Popen(command, cwd=ROOT, env=env); state['child_PID'] = child.pid; write(path, state)
                result = child.wait()
                if result: raise RuntimeError(mode + ' failed: ' + str(result))
            state.update(status='complete', phase='complete', completed_unix=time.time())
        except BaseException:
            state.update(status='failed', traceback=traceback.format_exc()); raise
        finally: write(path, state)


if __name__ == '__main__': main()
