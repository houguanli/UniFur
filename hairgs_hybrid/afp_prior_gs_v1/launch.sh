#!/usr/bin/env bash
# One exclusive GPU job: train -> frozen evaluation. Usage: launch.sh ARM GPU_UUID [--set k=v ...]
# Pilot arms evaluate on the 4 withheld TRAIN views; final arms on test8 + train28.
set -euo pipefail
cd /root/Public/h84435299/UniFur
ARM=$1; GPU=$2; shift 2
E=outputs/a025_afp_prior_free_gs_20261105
LOCK=experiments/structured_fur_v2_20261016/panda_baselines/registry/$GPU.lock
PY=compare_methods/3DGS/a025_20261015/env/py310/bin/python
row=$(nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits | grep "^$GPU,")
used=$(echo "$row" | cut -d, -f2 | tr -d ' '); util=$(echo "$row" | cut -d, -f3 | tr -d ' ')
MAXMEM=200; MAXUTIL=5; [[ ${ALLOW_SHARED:-0} == 1 ]] && { MAXMEM=2000; MAXUTIL=10; }  # idle foreign context only
[[ -n "$row" && $used -lt $MAXMEM && $util -lt $MAXUTIL ]] || { echo "GPU $GPU not idle: $row"; exit 3; }
mkdir -p $E/logs
source compare_methods/3DGS/a025_20261015/env.sh
export CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 TORCH_HOME=/root/Public/h84435299/UniFur/cache/torch OMP_NUM_THREADS=8
EXTRA=("$@")
SMOKE=0; for x in "${EXTRA[@]}"; do [[ $x == smoke=true ]] && SMOKE=1; done
RUN=$E/runs/$ARM; [[ $SMOKE == 1 ]] && RUN=$E/smoke/$ARM
evaluate() { $PY -B -m hairgs_hybrid.afp_prior_gs_v1.evaluate --model "$1" --splits "${@:3}" --output "$2"; }
exec 9>>"$LOCK"
flock -n 9 || { echo "GPU lock busy: $LOCK"; exit 4; }
{
  $PY -B -m hairgs_hybrid.afp_prior_gs_v1.train train --arm "$ARM" --set "${EXTRA[@]}"
  if [[ $SMOKE == 1 ]]; then  # smoke never reads test pixels
    evaluate $RUN/free.npz $RUN/eval_free validation
    [[ -f $RUN/final.npz ]] && evaluate $RUN/final.npz $RUN/eval_final validation
  elif [[ $ARM == pilot_* ]]; then
    evaluate $RUN/free.npz $RUN/eval_free validation
  else
    evaluate $RUN/free.npz $RUN/eval_free test train
    [[ -f $RUN/final.npz ]] && evaluate $RUN/final.npz $RUN/eval_final test train
  fi
  echo "LAUNCH_COMPLETE $ARM"
} > $E/logs/$ARM$([[ $SMOKE == 1 ]] && echo _smoke).log 2>&1
