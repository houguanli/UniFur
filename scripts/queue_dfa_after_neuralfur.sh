#!/usr/bin/env bash
set -euo pipefail

NEURALFUR_PID="${1:-110687}"
REPO_ROOT="/home/aoki/Differentiable_Physical_Driven_3DGS_for_monocular_Animal_Motion_Reconstruction"
NEURALFUR_OUTPUT="/mnt/f/fur_hair_unified_data/benchmarks/neuralfur_panda_shared/neuralfur_4k_full20k_lrbody_r512"

echo "queue_started=$(date --iso-8601=seconds) neuralfur_pid=$NEURALFUR_PID"
while kill -0 "$NEURALFUR_PID" 2>/dev/null; do
  latest_checkpoint=$(find "$NEURALFUR_OUTPUT/checkpoints" -maxdepth 1 -type f -name '*.pth' -printf '%f\n' 2>/dev/null | sort -V | tail -1 || true)
  echo "waiting_for_neuralfur=$(date --iso-8601=seconds) latest_checkpoint=${latest_checkpoint:-none}"
  sleep 30
done

if [[ -f "$NEURALFUR_OUTPUT/checkpoints/20000.pth" ]]; then
  echo "neuralfur_terminal=complete"
else
  echo "neuralfur_terminal=incomplete_dynamic_runs_continue=true"
fi

cd "$REPO_ROOT"
bash scripts/run_dfa_panda_dual_benchmark.sh all
echo "queue_finished=$(date --iso-8601=seconds)"
