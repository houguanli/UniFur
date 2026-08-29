#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
source "${EXPERIMENT_ROOT}/scripts/lib/runtime.sh"
EVAL_PROJECT="${EVAL_PROJECT:-${EXPERIMENT_ROOT}}"
HAIRGS_ROOT="${HAIRGS_ROOT:-${EXPERIMENT_ROOT}/third_party/hair-gs}"
CONDA_BIN="$(resolve_conda_bin)"
CONDA_ENV="${CONDA_ENV:-unifur}"
COMPAT_ROOT="${HAIRGS_COMPAT_ROOT:-${EXPERIMENT_ROOT}/compat/hairgs_sitecustomize}"
DATA_ROOT="${UNIFUR_DATA_ROOT:-${DATA_ROOT:-${EXPERIMENT_ROOT}/data}}"
OUTPUT_ROOT="${UNIFUR_OUTPUT_ROOT:-${OUTPUT_ROOT:-${EXPERIMENT_ROOT}/outputs}}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-${OUTPUT_ROOT}/protocols/panda_r480}"
PANDA_ROOT="${PANDA_ROOT:-${DATA_ROOT}/benchmarks/neuralfur_panda_shared}"
SOURCE_ROOT="${SOURCE_ROOT:-${OUTPUT_ROOT}/experiments/panda_topology_g10k_st_hard22k_v2}"
RESULT_ROOT="${RESULT_ROOT:-${OUTPUT_ROOT}/experiments/panda_topology_g10k_st_hard22k_refine3k_v3}"
SOURCE_ITERATION="${SOURCE_ITERATION:-35000}"
STEPS="${STEPS:-3000}"
FINAL_ITERATION=$((SOURCE_ITERATION + STEPS))
PROTOCOL_ID="panda-hairgs-28fit-8heldout-r480-v1"

export PYTHONPATH="${COMPAT_ROOT}:${HAIRGS_ROOT}:${EXPERIMENT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
run_hairgs() { "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" python "$@"; }

prepare() {
  [[ -f "${RESULT_ROOT}/.prepared" ]] && return
  test -f "${SOURCE_ROOT}/point_cloud/iteration_${SOURCE_ITERATION}/hybrid_state.npz"
  mkdir -p "${RESULT_ROOT}/point_cloud/iteration_${SOURCE_ITERATION}" "${RESULT_ROOT}/audit"
  cp "${SOURCE_ROOT}/point_cloud/iteration_${SOURCE_ITERATION}/point_cloud.ply" \
    "${RESULT_ROOT}/point_cloud/iteration_${SOURCE_ITERATION}/point_cloud.ply"
  cp "${SOURCE_ROOT}/point_cloud/iteration_${SOURCE_ITERATION}/hybrid_state.npz" \
    "${RESULT_ROOT}/point_cloud/iteration_${SOURCE_ITERATION}/hybrid_state.npz"
  cp "${SOURCE_ROOT}/point_cloud/iteration_${SOURCE_ITERATION}/routes.npz" \
    "${RESULT_ROOT}/point_cloud/iteration_${SOURCE_ITERATION}/routes.npz"
  run_hairgs - <<PY
from argparse import Namespace
from pathlib import Path
source=Path("${SOURCE_ROOT}/cfg_args")
args=eval(source.read_text(), {"Namespace": Namespace})
args.model_path="${RESULT_ROOT}"
args.source_path="${PROTOCOL_ROOT}/train_v28"
args.route_sidecar="${RESULT_ROOT}/point_cloud/iteration_${SOURCE_ITERATION}/routes.npz"
args.hybrid_route_mode="hard"
Path("${RESULT_ROOT}/cfg_args").write_text(repr(args))
PY
  touch "${RESULT_ROOT}/.prepared"
}

train() {
  [[ -f "${RESULT_ROOT}/point_cloud/iteration_${FINAL_ITERATION}/hybrid_state.npz" ]] && return
  cd "${HAIRGS_ROOT}"
  /usr/bin/time -v -o "${RESULT_ROOT}/audit/refine_time.txt" \
    "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" python \
      "${EXPERIMENT_ROOT}/hairgs_hybrid/train_hybrid.py" \
      -s "${PROTOCOL_ROOT}/train_v28" -m "${RESULT_ROOT}" -r 1 --sh_degree 3 \
      --route-sidecar "${RESULT_ROOT}/point_cloud/iteration_${SOURCE_ITERATION}/routes.npz" \
      --hybrid-init-mode neutral --hybrid-route-mode hard \
      --hybrid-route-schedule constant --hybrid-route-warmup-steps 0 \
      --hybrid-temperature 0.35 --hybrid-temperature-final 0.35 \
      --hybrid-temperature-anneal-steps 0 --hybrid-neutral-activation-steps 0 \
      --iterations "${STEPS}" --densify_until_iter 0 \
      --merge_interval 100000 --growth_interval 100000 \
      --position_lr_max_steps "${STEPS}" --lambda_mask 0 \
      --lambda_orientation 10 --lambda_smooth 0 \
      --hybrid-shell-strength 0.16 --hybrid-fin-strength 0.20 \
      --hybrid-strength-max 0.80 \
      --lambda-hybrid-mask-bce 0.30 --lambda-hybrid-mask-fp 0.60 \
      --lambda-hybrid-mask-fn 0.60 --lambda-hybrid-hole 0.20 \
      --lambda-hybrid-overflow-peak 0.60 --lambda-hybrid-background-rgb 0.50 \
      --lambda-hybrid-route-cost 0 --lambda-hybrid-route-entropy 0 \
      --lambda-hybrid-route-explore 0 --lambda-hybrid-route-prior 0 \
      --lambda-hybrid-budget 0 --hybrid-target-active-splats 22000 \
      --lambda-hybrid-shape-reg 0.00002 \
      --lambda-hybrid-center-anchor 0.50 --lambda-hybrid-length-anchor 0.25 \
      --lambda-hybrid-width-anchor 0.10 \
      --save_frequency "${STEPS}" --eval_frequency "${STEPS}" \
      --logger tensorboard --quiet 2>&1 | tee "${RESULT_ROOT}/audit/refine.log"
}

evaluate() {
  local split="$1" dataset_name="$2"
  local dataset="${PROTOCOL_ROOT}/${dataset_name}"
  local output="${RESULT_ROOT}/evaluation_${split}"
  [[ -f "${output}/external_evaluation/evaluation.json" ]] && return
  mkdir -p "${output}"
  cd "${HAIRGS_ROOT}"
  run_hairgs "${EXPERIMENT_ROOT}/hairgs_hybrid/render_hybrid.py" \
    -s "${dataset}" -m "${RESULT_ROOT}" -r 1 --iterations "${FINAL_ITERATION}" \
    --skip_test --type 0 --quiet
  run_hairgs "${EXPERIMENT_ROOT}/hairgs_hybrid/render_hybrid.py" \
    -s "${dataset}" -m "${RESULT_ROOT}" -r 1 --iterations "${FINAL_ITERATION}" \
    --skip_test --type 2 --quiet
  cd "${EVAL_PROJECT}"
  run_hairgs scripts/export_hairgs_external_renders.py \
    --model-dir "${RESULT_ROOT}" --camera-dataset "${dataset}" \
    --camera-manifest "${dataset}/camera_manifest.json" \
    --output-dir "${output}/external_renders" --iteration "${FINAL_ITERATION}" \
    --output-size 480 270 --hairgs-root "${HAIRGS_ROOT}"
  run_hairgs scripts/evaluate_external_renders.py \
      --render-manifest "${output}/external_renders/render_manifest.json" \
      --ground-truth-dir "${PANDA_ROOT}/${dataset_name}/images" \
      --output-dir "${output}/external_evaluation" \
      --method "Panda fixed topology hard 22k + ${STEPS} refinement" \
      --protocol-id "${PROTOCOL_ID}-${split}" --device cuda
}

prepare
train
evaluate train train_v28
evaluate test test_v8
echo "Panda hard topology refinement complete: ${RESULT_ROOT}"
