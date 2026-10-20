"""Single new experiment identity and idle-GPU launcher; no checkpoint restart."""
import fcntl,json,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];gpu=sys.argv[1]
with (HERE/'dispatch.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (HERE/'DISPATCH.json').exists()
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True)
    used={u.strip():int(m.strip()) for u,m in (r.split(',') for r in rows.splitlines())}
    assert used[gpu]<1024,('not idle',gpu,used[gpu])
    log=(HERE/'formal.log').open('xb')
    p=subprocess.Popen(['bash',str(HERE/'run.sh'),gpu],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    j=dict(PID=p.pid,GPU=gpu,timeUTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),log=str(HERE/'formal.log'))
    (HERE/'DISPATCH.json').write_text(json.dumps(j,indent=2));print(json.dumps(j))
