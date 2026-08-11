#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/aoki/Differentiable_Physical_Driven_3DGS_for_monocular_Animal_Motion_Reconstruction}"
STAGE2_PID="${STAGE2_PID:-137203}"
VIDU_LOG_ROOT="${VIDU_LOG_ROOT:-/mnt/f/fur_hair_unified_data/baselines/vidu4d/logs}"
QUEUE_ROOT="${QUEUE_ROOT:-/mnt/f/fur_hair_unified_data/baselines/external_same_case_queue}"
STATUS_LOG="${QUEUE_ROOT}/queue_status.tsv"

mkdir -p "${QUEUE_ROOT}"

status() {
  printf '%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$1" "$2" | tee -a "${STATUS_LOG}"
}

run_step() {
  local name="$1"
  shift
  status "${name}" "running"
  "$@" 2>&1 | tee "${QUEUE_ROOT}/${name}.log"
  status "${name}" "complete"
}

status "vidu4d_stage2" "waiting_pid_${STAGE2_PID}"
while kill -0 "${STAGE2_PID}" 2>/dev/null; do
  sleep 60
done

VIDU_STAGE2="${VIDU_LOG_ROOT}/dfa-panda-walk-mono-base"
test -f "${VIDU_STAGE2}/ckpt_0020.pth" || {
  status "vidu4d_stage2" "failed_missing_ckpt_0020"
  exit 2
}
test -f "${VIDU_STAGE2}/021-fg-geo.obj" || {
  status "vidu4d_stage2" "failed_missing_021_fg_geo"
  exit 2
}
status "vidu4d_stage2" "complete"

cd "${PROJECT_ROOT}"
run_step neuralfur_render_eval bash scripts/run_neuralfur_static_benchmark.sh all
run_step vidu4d_stage3 bash scripts/run_vidu4d_dfa_benchmark.sh stage3
run_step vidu4d_render_eval bash scripts/run_vidu4d_dfa_benchmark.sh render_and_evaluate
run_step gart_dfa_all bash scripts/run_gart_dfa_benchmark.sh all
run_step fourdanimal_dfa_all bash scripts/run_fourdanimal_dfa_benchmark.sh all

status "external_same_case_queue" "complete"
