#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
source "${EXPERIMENT_ROOT}/scripts/lib/runtime.sh"
CONDA_BIN="$(resolve_conda_bin)"
CONDA_ENV="${CONDA_ENV:-unifur}"
HAIRGS_ROOT="${HAIRGS_ROOT:-${EXPERIMENT_ROOT}/third_party/hair-gs}"
COMPAT_ROOT="${HAIRGS_COMPAT_ROOT:-${EXPERIMENT_ROOT}/compat/hairgs_sitecustomize}"
FIXED_ROOT="${ANIMALS60_FIXED_ROOT:-${EXPERIMENT_ROOT}/outputs/animals60_v48_25k}"
OUTPUT_ROOT="${ANIMALS60_RATIO_ROOT:-${EXPERIMENT_ROOT}/outputs/animals60_v48_ratio_ladder}"
SOURCE_CASES="${ANIMALS60_CASES_ROOT:-/mnt/f/fur_hair_unified_data/benchmarks/unifur_animals_60view_shared/cases}"
CASE_ID="${1:?usage: run_animals60_ratio_case.sh CASE_ID COMPRESSION}"
COMPRESSION="${2:?usage: run_animals60_ratio_case.sh CASE_ID COMPRESSION}"
FIXED_CASE="${FIXED_ROOT}/cases/${CASE_ID}"
PROTOCOL_ROOT="${FIXED_CASE}/protocol"
CASE_SOURCE="${SOURCE_CASES}/${CASE_ID}"
DENSE_MODEL="${FIXED_CASE}/hairgs_unrestricted/model"
DENSE_ITERATION="${DENSE_ITERATION:-13000}"
DENSE_PLY="${DENSE_MODEL}/point_cloud/iteration_${DENSE_ITERATION}/point_cloud.ply"
SOFT_STEPS="${SOFT_STEPS:-5000}"
REFINE_STEPS="${REFINE_STEPS:-3000}"
NATIVE_STEPS="${NATIVE_STEPS:-8000}"
WIDTH="${WIDTH:-512}"
HEIGHT="${HEIGHT:-512}"

export PYTHONPATH="${COMPAT_ROOT}:${HAIRGS_ROOT}:${EXPERIMENT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
export OPENCV_IO_ENABLE_OPENEXR=1

run_python() {
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" python "$@"
}

read -r SOURCE_SEGMENTS TARGET_ACTIVE GROUP_COUNT RUN_TAG < <(
  run_python - "${DENSE_PLY}" "${COMPRESSION}" <<'PY'
import sys
from pathlib import Path
from plyfile import PlyData
from hairgs_hybrid.plan_animals60_ratio_ladder import ratio_budget, neutral_group_count
source=int(PlyData.read(Path(sys.argv[1]))["edge"].count)
compression=float(sys.argv[2])
target=ratio_budget(source, compression)
groups=neutral_group_count(source, target)
tag=f"c{compression*100:g}_r{(1-compression)*100:g}_b{target}"
print(source,target,groups,tag)
PY
)

RUN_ROOT="${OUTPUT_ROOT}/cases/${CASE_ID}/${RUN_TAG}"
NATIVE_ROOT="${RUN_ROOT}/hairgs_native"
UNIFUR_ROOT="${RUN_ROOT}/unifur"
PROTOCOL_ID="unifur-animals60-${CASE_ID}-v48-test12-r${WIDTH}-${RUN_TAG}-v1"

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
  test -f "${FIXED_CASE}/COMPLETE"
  test -f "${DENSE_PLY}"
  test -f "${PROTOCOL_ROOT}/train_v48/camera_manifest.json"
  test -f "${PROTOCOL_ROOT}/test_v12/camera_manifest.json"
  mkdir -p "${RUN_ROOT}/audit"
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
    echo "GPU already has a compute process; refusing duplicate ratio training" >&2
    exit 4
  fi
  run_python - "${DENSE_PLY}" "${SOURCE_SEGMENTS}" "${TARGET_ACTIVE}" "${GROUP_COUNT}" <<'PY'
import sys
from plyfile import PlyData
count=int(PlyData.read(sys.argv[1])["edge"].count)
assert count == int(sys.argv[2])
target,groups=map(int,sys.argv[3:])
assert groups <= target <= 4*groups
assert groups <= count//4
PY
  cat > "${RUN_ROOT}/audit/budget.json" <<EOF
{"case":"${CASE_ID}","compression":${COMPRESSION},"source_segments":${SOURCE_SEGMENTS},"target_active_segments":${TARGET_ACTIVE},"group_count":${GROUP_COUNT},"candidate_segments":$((7*GROUP_COUNT)),"run_tag":"${RUN_TAG}"}
EOF
}

train_native_control() {
  local model="${NATIVE_ROOT}/model"
  local final="${model}/point_cloud/iteration_${NATIVE_STEPS}/point_cloud.ply"
  mkdir -p "${NATIVE_ROOT}/audit"
  if [[ ! -f "${model}/compression_report.json" && ! -f "${NATIVE_ROOT}/compression_report.json" ]]; then
    run_python "${EXPERIMENT_ROOT}/hairgs_hybrid/compress_hairgs_teacher.py" \
      --input-ply "${DENSE_PLY}" --source-cfg "${DENSE_MODEL}/cfg_args" \
      --dataset "${PROTOCOL_ROOT}/train_v48" --output-model "${model}" \
      --output-iteration 0 --target-active "${TARGET_ACTIVE}" \
      | tee "${NATIVE_ROOT}/audit/compress.log"
  fi
  if [[ ! -f "${final}" ]]; then
    start_monitor "${NATIVE_ROOT}/audit/train_gpu.csv"
    trap stop_monitor EXIT
    cd "${HAIRGS_ROOT}"
    timed_python "${NATIVE_ROOT}/audit/train_time.txt" "${NATIVE_ROOT}/audit/train.log" \
      "${EXPERIMENT_ROOT}/hairgs_hybrid/train_native.py" \
      -s "${PROTOCOL_ROOT}/train_v48" -m "${model}" -r 1 \
      --sh_degree 3 --iterations "${NATIVE_STEPS}" \
      --position_lr_max_steps "${NATIVE_STEPS}" --densify_until_iter 0 \
      --merge_interval 100000 --growth_interval 100000 \
      --lambda_orientation 0 --lambda_mask 0.01 --lambda_smooth 0 \
      --save_frequency "${NATIVE_STEPS}" --eval_frequency "${NATIVE_STEPS}" \
      --logger tensorboard --quiet
    stop_monitor
    trap - EXIT
  fi
  run_python - "${final}" "${TARGET_ACTIVE}" <<'PY'
import sys
from plyfile import PlyData
assert int(PlyData.read(sys.argv[1])["edge"].count) == int(sys.argv[2])
PY
}

build_and_train_unifur() {
  local bank="${UNIFUR_ROOT}/bank" soft="${UNIFUR_ROOT}/soft"
  local hard="${UNIFUR_ROOT}/hard" model="${UNIFUR_ROOT}/model"
  mkdir -p "${UNIFUR_ROOT}/audit"
  if [[ ! -f "${bank}/candidate_bank_report.json" ]]; then
    run_python "${EXPERIMENT_ROOT}/hairgs_hybrid/build_topology_candidate_bank.py" \
      --input-ply "${DENSE_PLY}" --output-root "${bank}" \
      --groups "${GROUP_COUNT}" --target-active "${TARGET_ACTIVE}" \
      | tee "${UNIFUR_ROOT}/audit/bank.log"
  fi
  if [[ ! -f "${soft}/.prepared" ]]; then
    mkdir -p "${soft}/point_cloud/iteration_0"
    cp "${bank}/point_cloud.ply" "${soft}/point_cloud/iteration_0/point_cloud.ply"
    cp "${bank}/routes.npz" "${soft}/point_cloud/iteration_0/routes.npz"
    touch "${soft}/.prepared"
  fi
  if [[ ! -f "${soft}/point_cloud/iteration_${SOFT_STEPS}/hybrid_state.npz" ]]; then
    start_monitor "${UNIFUR_ROOT}/audit/train_gpu.csv"
    trap stop_monitor EXIT
    cd "${HAIRGS_ROOT}"
    timed_python "${UNIFUR_ROOT}/audit/soft_time.txt" "${UNIFUR_ROOT}/audit/soft.log" \
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
    stop_monitor
    trap - EXIT
  fi
  if [[ ! -f "${hard}/point_cloud/iteration_${SOFT_STEPS}/hybrid_state.npz" ]]; then
    run_python "${EXPERIMENT_ROOT}/hairgs_hybrid/harden_topology_checkpoint.py" \
      --source-model "${soft}" --source-iteration "${SOFT_STEPS}" \
      --output-model "${hard}" --output-iteration "${SOFT_STEPS}" \
      --target-active "${TARGET_ACTIVE}"
  fi
  if [[ ! -f "${model}/.prepared" ]]; then
    mkdir -p "${model}/point_cloud/iteration_${SOFT_STEPS}"
    cp "${hard}/point_cloud/iteration_${SOFT_STEPS}/point_cloud.ply" "${model}/point_cloud/iteration_${SOFT_STEPS}/point_cloud.ply"
    cp "${hard}/point_cloud/iteration_${SOFT_STEPS}/hybrid_state.npz" "${model}/point_cloud/iteration_${SOFT_STEPS}/hybrid_state.npz"
    cp "${hard}/point_cloud/iteration_${SOFT_STEPS}/routes.npz" "${model}/point_cloud/iteration_${SOFT_STEPS}/routes.npz"
    run_python - "${hard}/cfg_args" "${model}" "${PROTOCOL_ROOT}/train_v48" "${SOFT_STEPS}" <<'PY'
import sys
from argparse import Namespace
from pathlib import Path
source,output,dataset,iteration=sys.argv[1:]
args=eval(Path(source).read_text(),{"Namespace":Namespace})
args.model_path=output; args.source_path=dataset
args.route_sidecar=str(Path(output)/f"point_cloud/iteration_{iteration}/routes.npz")
args.hybrid_route_mode="hard"
Path(output,"cfg_args").write_text(repr(args))
PY
    touch "${model}/.prepared"
  fi
  if [[ ! -f "${model}/point_cloud/iteration_$((SOFT_STEPS+REFINE_STEPS))/hybrid_state.npz" ]]; then
    start_monitor "${UNIFUR_ROOT}/audit/refine_gpu.csv"
    trap stop_monitor EXIT
    cd "${HAIRGS_ROOT}"
    timed_python "${UNIFUR_ROOT}/audit/refine_time.txt" "${UNIFUR_ROOT}/audit/refine.log" \
      "${EXPERIMENT_ROOT}/hairgs_hybrid/train_hybrid.py" \
      -s "${PROTOCOL_ROOT}/train_v48" -m "${model}" -r 1 --sh_degree 3 \
      --route-sidecar "${model}/point_cloud/iteration_${SOFT_STEPS}/routes.npz" \
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
    stop_monitor
    trap - EXIT
  fi
  run_python - "${model}/point_cloud/iteration_$((SOFT_STEPS+REFINE_STEPS))/point_cloud.ply" "${TARGET_ACTIVE}" <<'PY'
import sys
from plyfile import PlyData
assert int(PlyData.read(sys.argv[1])["edge"].count) == int(sys.argv[2])
PY
}

render_evaluate() {
  local method="$1" kind="$2" model="$3" iteration="$4" split="$5"
  local output="${RUN_ROOT}/${method}/evaluation_${split}"
  [[ -f "${output}/external_evaluation/evaluation.json" ]] && return
  rm -rf "${model}/render/train/iteration_${iteration}"
  cd "${HAIRGS_ROOT}"
  if [[ "${kind}" == hybrid ]]; then
    run_python "${EXPERIMENT_ROOT}/hairgs_hybrid/render_hybrid.py" -s "${PROTOCOL_ROOT}/${split}" -m "${model}" -r 1 --skip_test --type 0 --quiet
    run_python "${EXPERIMENT_ROOT}/hairgs_hybrid/render_hybrid.py" -s "${PROTOCOL_ROOT}/${split}" -m "${model}" -r 1 --skip_test --type 2 --quiet
  else
    run_python "${HAIRGS_ROOT}/render.py" -s "${PROTOCOL_ROOT}/${split}" -m "${model}" -r 1 --skip_test --type 0 --quiet
    run_python "${HAIRGS_ROOT}/render.py" -s "${PROTOCOL_ROOT}/${split}" -m "${model}" -r 1 --skip_test --type 2 --quiet
  fi
  cd "${EXPERIMENT_ROOT}"
  run_python scripts/export_hairgs_external_renders.py \
    --model-dir "${model}" --camera-dataset "${PROTOCOL_ROOT}/${split}" \
    --camera-manifest "${PROTOCOL_ROOT}/${split}/camera_manifest.json" \
    --output-dir "${output}/external_renders" --iteration "${iteration}" \
    --output-size "${WIDTH}" "${HEIGHT}" --hairgs-root "${HAIRGS_ROOT}"
  run_python scripts/evaluate_external_renders.py \
    --render-manifest "${output}/external_renders/render_manifest.json" \
    --ground-truth-dir "${CASE_SOURCE}/${split}/images" \
    --output-dir "${output}/external_evaluation" --method "${method}-${RUN_TAG}" \
    --protocol-id "${PROTOCOL_ID}-${split}" --device cuda
}

benchmark() {
  local method="$1" kind="$2" model="$3"
  local output="${RUN_ROOT}/${method}/audit/render_benchmark.json"
  [[ -f "${output}" ]] && return
  run_python hairgs_hybrid/benchmark_render.py --model-dir "${model}" \
    --dataset "${PROTOCOL_ROOT}/test_v12" --kind "${kind}" \
    --warmup 3 --repeats 20 --output "${output}"
}

write_comparison() {
  run_python - "${RUN_ROOT}" "${DENSE_PLY}" "${TARGET_ACTIVE}" "${COMPRESSION}" <<'PY'
import json,sys
from pathlib import Path
from plyfile import PlyData
root,dense,target,compression=Path(sys.argv[1]),Path(sys.argv[2]),int(sys.argv[3]),float(sys.argv[4])
payload={"schema":"unifur-animals60-ratio-case-v1","source_segments":int(PlyData.read(dense)["edge"].count),"target_segments":target,"compression":compression,"methods":{}}
for method in ("hairgs_native","unifur"):
    item={}
    for split in ("train_v48","test_v12"):
        p=root/method/f"evaluation_{split}/external_evaluation/evaluation.json"
        item[split]=json.loads(p.read_text())["aggregate"]
    b=root/method/"audit/render_benchmark.json"
    item["render_benchmark"]=json.loads(b.read_text())
    payload["methods"][method]=item
(root/"comparison.json").write_text(json.dumps(payload,indent=2)+"\n")
PY
}

preflight
train_native_control
build_and_train_unifur
for split in train_v48 test_v12; do
  render_evaluate hairgs_native native "${NATIVE_ROOT}/model" "${NATIVE_STEPS}" "${split}"
  render_evaluate unifur hybrid "${UNIFUR_ROOT}/model" "$((SOFT_STEPS+REFINE_STEPS))" "${split}"
done
benchmark hairgs_native native "${NATIVE_ROOT}/model"
benchmark unifur hybrid "${UNIFUR_ROOT}/model"
write_comparison
touch "${RUN_ROOT}/COMPLETE"
echo "Ratio case complete: ${CASE_ID} ${RUN_TAG}"
