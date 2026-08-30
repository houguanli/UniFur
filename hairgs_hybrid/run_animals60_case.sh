#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
source "${EXPERIMENT_ROOT}/scripts/lib/runtime.sh"
CONDA_BIN="$(resolve_conda_bin)"
CONDA_ENV="${CONDA_ENV:-unifur}"
HAIRGS_ROOT="${HAIRGS_ROOT:-${EXPERIMENT_ROOT}/third_party/hair-gs}"
COMPAT_ROOT="${HAIRGS_COMPAT_ROOT:-${EXPERIMENT_ROOT}/compat/hairgs_sitecustomize}"
SOURCE_CASES="${ANIMALS60_CASES_ROOT:-/mnt/f/fur_hair_unified_data/benchmarks/unifur_animals_60view_shared/cases}"
BENCHMARK_ROOT="${ANIMALS60_OUTPUT_ROOT:-${EXPERIMENT_ROOT}/outputs/animals60_v48_25k}"
CASE_ID="${1:?usage: run_animals60_case.sh CASE_ID}"
CASE_SOURCE="${SOURCE_CASES}/${CASE_ID}"
CASE_ROOT="${BENCHMARK_ROOT}/cases/${CASE_ID}"
PROTOCOL_ROOT="${CASE_ROOT}/protocol"
TARGET_ACTIVE="${TARGET_ACTIVE:-25000}"
WIDTH="${WIDTH:-512}"
HEIGHT="${HEIGHT:-512}"
STAGE1_STEPS="${STAGE1_STEPS:-8000}"
STAGE3_STEPS="${STAGE3_STEPS:-5000}"
SOFT_STEPS="${SOFT_STEPS:-5000}"
REFINE_STEPS="${REFINE_STEPS:-3000}"
DENSE_DENSIFY_UNTIL="${DENSE_DENSIFY_UNTIL:-6000}"
DENSE_FINAL=$((STAGE1_STEPS + STAGE3_STEPS))
HYBRID_FINAL=$((SOFT_STEPS + REFINE_STEPS))
PROTOCOL_ID="unifur-animals60-${CASE_ID}-v48-test12-r${WIDTH}-v1"

export PYTHONPATH="${COMPAT_ROOT}:${HAIRGS_ROOT}:${EXPERIMENT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
export OPENCV_IO_ENABLE_OPENEXR=1

run_python() {
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" python "$@"
}

start_monitor() {
  local output="$1"
  mkdir -p "$(dirname -- "${output}")"
  stdbuf -oL nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu \
    --format=csv,noheader,nounits -l 1 > "${output}" &
  MONITOR_PID=$!
}

stop_monitor() {
  if [[ -n "${MONITOR_PID:-}" ]]; then
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
    MONITOR_PID=""
  fi
}

timed_python() {
  local time_file="$1" log_file="$2"
  shift 2
  /usr/bin/time -v -o "${time_file}" \
    "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" python "$@" \
    2>&1 | tee "${log_file}"
}

preflight() {
  test -f "${CASE_SOURCE}/train_v48/camera_manifest.json"
  test -f "${CASE_SOURCE}/test_v12/camera_manifest.json"
  test -f "${HAIRGS_ROOT}/train.py"
  mkdir -p "${CASE_ROOT}/audit"
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
    echo "GPU already has a compute process; refusing duplicate training" >&2
    exit 4
  fi
  run_python "${EXPERIMENT_ROOT}/scripts/verify_reproducibility.py" \
    | tee "${CASE_ROOT}/audit/preflight.log"
}

prepare_protocol() {
  if [[ ! -f "${PROTOCOL_ROOT}/input_report.json" ]]; then
    run_python "${EXPERIMENT_ROOT}/hairgs_hybrid/prepare_animals60_hairgs.py" \
      --case-root "${CASE_SOURCE}" --output-root "${PROTOCOL_ROOT}" \
      --width "${WIDTH}" --height "${HEIGHT}" \
      --initial-points "${TARGET_ACTIVE}" \
      > "${CASE_ROOT}/audit/prepare.log"
  fi
  run_python - "${PROTOCOL_ROOT}/input_report.json" "${TARGET_ACTIVE}" "${CASE_SOURCE}" <<'PY'
import hashlib, json, sys
from pathlib import Path
report=json.loads(Path(sys.argv[1]).read_text())
assert report["status"] == "complete"
assert report["train"]["count"] == 48 and report["test"]["count"] == 12
assert report["heldout_geometry_used"] is False
assert report["initialization"]["initial_points"] == int(sys.argv[2])
case=Path(sys.argv[3])
def sha256(path):
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
assert report["train_manifest_sha256"] == sha256(case / "train_v48/camera_manifest.json")
assert report["test_manifest_sha256"] == sha256(case / "test_v12/camera_manifest.json")
PY
}

train_native() {
  local label="$1" densify_until="$2"
  local root="${CASE_ROOT}/${label}"
  local stage1="${root}/stage1" model="${root}/model"
  local stage1_ply="${stage1}/point_cloud/iteration_${STAGE1_STEPS}/point_cloud.ply"
  local final_ply="${model}/point_cloud/iteration_${DENSE_FINAL}/point_cloud.ply"
  mkdir -p "${root}/audit"
  start_monitor "${root}/audit/train_gpu.csv"
  trap stop_monitor EXIT
  if [[ ! -f "${stage1_ply}" ]]; then
    cd "${HAIRGS_ROOT}"
    timed_python "${root}/audit/stage1_time.txt" "${root}/audit/stage1.log" \
      "${EXPERIMENT_ROOT}/hairgs_hybrid/train_native.py" -s "${PROTOCOL_ROOT}/train_v48" -m "${stage1}" -r 1 \
      --sh_degree 3 --iterations "${STAGE1_STEPS}" \
      --position_lr_max_steps "${STAGE1_STEPS}" \
      --densify_until_iter "${densify_until}" --densify_from_iter 500 \
      --densification_interval 100 --densify_grad_threshold 0.00005 \
      --lambda_orientation 0 --lambda_mask 0.01 --lambda_smooth 0 \
      --save_frequency "${STAGE1_STEPS}" --eval_frequency "${STAGE1_STEPS}" \
      --logger tensorboard --quiet
  fi
  if [[ ! -f "${model}/point_cloud/iteration_${STAGE1_STEPS}/point_cloud.ply" ]]; then
    run_python "${EXPERIMENT_ROOT}/hairgs_hybrid/convert_gaussian_to_hairgs.py" \
      --input-ply "${stage1_ply}" --output-model "${model}" \
      --iteration "${STAGE1_STEPS}" --source-cfg "${stage1}/cfg_args" \
      --dataset "${PROTOCOL_ROOT}/train_v48"
  fi
  if [[ ! -f "${final_ply}" ]]; then
    cd "${HAIRGS_ROOT}"
    timed_python "${root}/audit/stage3_time.txt" "${root}/audit/stage3.log" \
      "${EXPERIMENT_ROOT}/hairgs_hybrid/train_native.py" -s "${PROTOCOL_ROOT}/train_v48" -m "${model}" -r 1 \
      --sh_degree 3 --iterations "${STAGE3_STEPS}" \
      --position_lr_max_steps "${STAGE3_STEPS}" --densify_until_iter 0 \
      --merge_interval 100000 --growth_interval 100000 \
      --lambda_orientation 0 --lambda_mask 0.01 --lambda_smooth 0 \
      --save_frequency "${STAGE3_STEPS}" --eval_frequency "${STAGE3_STEPS}" \
      --logger tensorboard --quiet
  fi
  stop_monitor
  trap - EXIT
  run_python - "${final_ply}" "${label}" "${TARGET_ACTIVE}" <<'PY'
import sys
from plyfile import PlyData
count=PlyData.read(sys.argv[1])["edge"].count
if sys.argv[2] == "hairgs_25k": assert count == int(sys.argv[3]), count
print({"method":sys.argv[2],"segments":count})
PY
}

prepare_hybrid_model() {
  local source_ply="${CASE_ROOT}/hairgs_unrestricted/model/point_cloud/iteration_${DENSE_FINAL}/point_cloud.ply"
  local root="${CASE_ROOT}/unifur_25k"
  local bank="${root}/bank"
  mkdir -p "${root}/audit"
  if [[ ! -f "${bank}/candidate_bank_report.json" ]]; then
    local source_count group_count
    source_count="$(run_python - "${source_ply}" <<'PY'
import sys
from plyfile import PlyData
print(PlyData.read(sys.argv[1])["edge"].count)
PY
)"
    group_count=$((source_count / 4))
    (( group_count > 10000 )) && group_count=10000
    if (( group_count < (TARGET_ACTIVE + 3) / 4 )); then
      echo "Dense teacher has only ${source_count} segments; cannot expose a 25k 1/2/4 bank" >&2
      exit 7
    fi
    run_python "${EXPERIMENT_ROOT}/hairgs_hybrid/build_topology_candidate_bank.py" \
      --input-ply "${source_ply}" --output-root "${bank}" \
      --groups "${group_count}" --target-active "${TARGET_ACTIVE}"
  fi
}

train_hybrid() {
  local root="${CASE_ROOT}/unifur_25k"
  local bank="${root}/bank"
  local soft="${root}/soft" hard="${root}/hard" final="${root}/model"
  local soft_final="${soft}/point_cloud/iteration_${SOFT_STEPS}/hybrid_state.npz"
  local final_state="${final}/point_cloud/iteration_${HYBRID_FINAL}/hybrid_state.npz"
  mkdir -p "${root}/audit"
  start_monitor "${root}/audit/train_gpu.csv"
  trap stop_monitor EXIT
  if [[ ! -f "${soft}/.prepared" ]]; then
    mkdir -p "${soft}/point_cloud/iteration_0"
    cp "${bank}/point_cloud.ply" "${soft}/point_cloud/iteration_0/point_cloud.ply"
    cp "${bank}/routes.npz" "${soft}/point_cloud/iteration_0/routes.npz"
    touch "${soft}/.prepared"
  fi
  if [[ ! -f "${soft_final}" ]]; then
    cd "${HAIRGS_ROOT}"
    timed_python "${root}/audit/soft_time.txt" "${root}/audit/soft.log" \
      "${EXPERIMENT_ROOT}/hairgs_hybrid/train_hybrid.py" \
      -s "${PROTOCOL_ROOT}/train_v48" -m "${soft}" -r 1 --sh_degree 3 \
      --route-sidecar "${soft}/point_cloud/iteration_0/routes.npz" \
      --hybrid-init-mode neutral --hybrid-route-mode soft \
      --hybrid-route-schedule staged --hybrid-route-warmup-steps 500 \
      --hybrid-temperature 1.5 --hybrid-temperature-final 0.35 \
      --hybrid-temperature-anneal-steps "${SOFT_STEPS}" \
      --hybrid-neutral-activation-steps 500 --hybrid-topology-hard-forward-start 1000 \
      --iterations "${SOFT_STEPS}" --position_lr_max_steps "${SOFT_STEPS}" \
      --densify_until_iter 0 --merge_interval 100000 --growth_interval 100000 \
      --lambda_orientation 0 --lambda_mask 0 --lambda_smooth 0 \
      --hybrid-shell-strength 0.16 --hybrid-fin-strength 0.20 \
      --hybrid-strength-max 0.80 --lambda-hybrid-mask-bce 0.30 \
      --lambda-hybrid-mask-fp 0.60 --lambda-hybrid-mask-fn 0.60 \
      --lambda-hybrid-hole 0.20 --lambda-hybrid-overflow-peak 0.60 \
      --lambda-hybrid-background-rgb 0.50 --lambda-hybrid-route-cost 0.0002 \
      --lambda-hybrid-route-entropy 0.00005 --lambda-hybrid-route-explore 0.0002 \
      --lambda-hybrid-budget 1.0 --hybrid-target-active-splats "${TARGET_ACTIVE}" \
      --lambda-hybrid-shape-reg 0.00002 --lambda-hybrid-center-anchor 0.50 \
      --lambda-hybrid-length-anchor 0.25 --lambda-hybrid-width-anchor 0.10 \
      --save_frequency "${SOFT_STEPS}" --eval_frequency "${SOFT_STEPS}" \
      --logger tensorboard --quiet
  fi
  if [[ ! -f "${hard}/point_cloud/iteration_${SOFT_STEPS}/hybrid_state.npz" ]]; then
    run_python "${EXPERIMENT_ROOT}/hairgs_hybrid/harden_topology_checkpoint.py" \
      --source-model "${soft}" --source-iteration "${SOFT_STEPS}" \
      --output-model "${hard}" --output-iteration "${SOFT_STEPS}" \
      --target-active "${TARGET_ACTIVE}"
  fi
  if [[ ! -f "${final}/.prepared" ]]; then
    mkdir -p "${final}/point_cloud/iteration_${SOFT_STEPS}"
    cp "${hard}/point_cloud/iteration_${SOFT_STEPS}/point_cloud.ply" \
      "${final}/point_cloud/iteration_${SOFT_STEPS}/point_cloud.ply"
    cp "${hard}/point_cloud/iteration_${SOFT_STEPS}/hybrid_state.npz" \
      "${final}/point_cloud/iteration_${SOFT_STEPS}/hybrid_state.npz"
    cp "${hard}/point_cloud/iteration_${SOFT_STEPS}/routes.npz" \
      "${final}/point_cloud/iteration_${SOFT_STEPS}/routes.npz"
    run_python - "${hard}/cfg_args" "${final}" "${PROTOCOL_ROOT}/train_v48" "${SOFT_STEPS}" <<'PY'
import sys
from argparse import Namespace
from pathlib import Path
source, output, dataset, iteration = sys.argv[1:]
args=eval(Path(source).read_text(), {"Namespace":Namespace})
args.model_path=output
args.source_path=dataset
args.route_sidecar=str(Path(output)/f"point_cloud/iteration_{iteration}/routes.npz")
args.hybrid_route_mode="hard"
Path(output,"cfg_args").write_text(repr(args))
PY
    touch "${final}/.prepared"
  fi
  if [[ ! -f "${final_state}" ]]; then
    cd "${HAIRGS_ROOT}"
    timed_python "${root}/audit/refine_time.txt" "${root}/audit/refine.log" \
      "${EXPERIMENT_ROOT}/hairgs_hybrid/train_hybrid.py" \
      -s "${PROTOCOL_ROOT}/train_v48" -m "${final}" -r 1 --sh_degree 3 \
      --route-sidecar "${final}/point_cloud/iteration_${SOFT_STEPS}/routes.npz" \
      --hybrid-init-mode neutral --hybrid-route-mode hard \
      --hybrid-route-schedule constant --hybrid-route-warmup-steps 0 \
      --hybrid-temperature 0.35 --hybrid-temperature-final 0.35 \
      --hybrid-temperature-anneal-steps 0 --hybrid-neutral-activation-steps 0 \
      --iterations "${REFINE_STEPS}" --position_lr_max_steps "${REFINE_STEPS}" \
      --densify_until_iter 0 --merge_interval 100000 --growth_interval 100000 \
      --lambda_orientation 0 --lambda_mask 0 --lambda_smooth 0 \
      --hybrid-shell-strength 0.16 --hybrid-fin-strength 0.20 \
      --hybrid-strength-max 0.80 --lambda-hybrid-mask-bce 0.30 \
      --lambda-hybrid-mask-fp 0.60 --lambda-hybrid-mask-fn 0.60 \
      --lambda-hybrid-hole 0.20 --lambda-hybrid-overflow-peak 0.60 \
      --lambda-hybrid-background-rgb 0.50 --lambda-hybrid-route-cost 0 \
      --lambda-hybrid-route-entropy 0 --lambda-hybrid-route-explore 0 \
      --lambda-hybrid-budget 0 --hybrid-target-active-splats "${TARGET_ACTIVE}" \
      --lambda-hybrid-shape-reg 0.00002 --lambda-hybrid-center-anchor 0.50 \
      --lambda-hybrid-length-anchor 0.25 --lambda-hybrid-width-anchor 0.10 \
      --save_frequency "${REFINE_STEPS}" --eval_frequency "${REFINE_STEPS}" \
      --logger tensorboard --quiet
  fi
  stop_monitor
  trap - EXIT
}

render_and_evaluate() {
  local method="$1" kind="$2" model="$3" iteration="$4" split="$5"
  local dataset="${PROTOCOL_ROOT}/${split}" root="${CASE_ROOT}/${method}"
  local output="${root}/evaluation_${split}"
  if [[ -f "${output}/external_evaluation/evaluation.json" ]]; then return; fi
  rm -rf "${model}/render/train/iteration_${iteration}"
  cd "${HAIRGS_ROOT}"
  if [[ "${kind}" == "hybrid" ]]; then
    run_python "${EXPERIMENT_ROOT}/hairgs_hybrid/render_hybrid.py" \
      -s "${dataset}" -m "${model}" -r 1 --skip_test --type 0 --quiet
    run_python "${EXPERIMENT_ROOT}/hairgs_hybrid/render_hybrid.py" \
      -s "${dataset}" -m "${model}" -r 1 --skip_test --type 2 --quiet
  else
    run_python "${HAIRGS_ROOT}/render.py" \
      -s "${dataset}" -m "${model}" -r 1 --skip_test --type 0 --quiet
    run_python "${HAIRGS_ROOT}/render.py" \
      -s "${dataset}" -m "${model}" -r 1 --skip_test --type 2 --quiet
  fi
  cd "${EXPERIMENT_ROOT}"
  run_python "${EXPERIMENT_ROOT}/scripts/export_hairgs_external_renders.py" \
    --model-dir "${model}" --camera-dataset "${dataset}" \
    --camera-manifest "${dataset}/camera_manifest.json" \
    --output-dir "${output}/external_renders" --iteration "${iteration}" \
    --output-size "${WIDTH}" "${HEIGHT}" --hairgs-root "${HAIRGS_ROOT}"
  run_python "${EXPERIMENT_ROOT}/scripts/evaluate_external_renders.py" \
    --render-manifest "${output}/external_renders/render_manifest.json" \
    --ground-truth-dir "${CASE_SOURCE}/${split}/images" \
    --output-dir "${output}/external_evaluation" --method "${method}" \
    --protocol-id "${PROTOCOL_ID}-${split}" --device cuda
}

benchmark_method() {
  local method="$1" kind="$2" model="$3"
  local output="${CASE_ROOT}/${method}/audit/render_benchmark.json"
  [[ -f "${output}" ]] && return
  run_python "${EXPERIMENT_ROOT}/hairgs_hybrid/benchmark_render.py" \
    --model-dir "${model}" --dataset "${PROTOCOL_ROOT}/test_v12" \
    --kind "${kind}" --warmup 3 --repeats 20 --output "${output}"
}

preflight
prepare_protocol
train_native hairgs_unrestricted "${DENSE_DENSIFY_UNTIL}"
train_native hairgs_25k 0
prepare_hybrid_model
train_hybrid

for split in train_v48 test_v12; do
  render_and_evaluate hairgs_unrestricted native \
    "${CASE_ROOT}/hairgs_unrestricted/model" "${DENSE_FINAL}" "${split}"
  render_and_evaluate hairgs_25k native \
    "${CASE_ROOT}/hairgs_25k/model" "${DENSE_FINAL}" "${split}"
  render_and_evaluate unifur_25k hybrid \
    "${CASE_ROOT}/unifur_25k/model" "${HYBRID_FINAL}" "${split}"
done

benchmark_method hairgs_unrestricted native "${CASE_ROOT}/hairgs_unrestricted/model"
benchmark_method hairgs_25k native "${CASE_ROOT}/hairgs_25k/model"
benchmark_method unifur_25k hybrid "${CASE_ROOT}/unifur_25k/model"

run_python "${EXPERIMENT_ROOT}/hairgs_hybrid/audit_animals60_benchmark.py" \
  --benchmark-root "${BENCHMARK_ROOT}" --case "${CASE_ID}"
touch "${CASE_ROOT}/COMPLETE"
echo "Animals60 case complete: ${CASE_ID}"
