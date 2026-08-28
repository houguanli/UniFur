#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/mnt/f/unifur_hairgs_hybrid_experiment}"
EVAL_PROJECT="${EVAL_PROJECT:-${EXPERIMENT_ROOT}}"
HAIRGS_ROOT="${HAIRGS_ROOT:-/home/aoki/fur_hair_baselines/hair-gs}"
CONDA_BIN="${CONDA_BIN:-/home/aoki/miniconda3/bin/conda}"
COMPAT_ROOT="${HAIRGS_COMPAT_ROOT:-${EXPERIMENT_ROOT}/compat/hairgs_sitecustomize}"
PANDA_ROOT="${PANDA_ROOT:-/mnt/f/fur_hair_unified_data/benchmarks/neuralfur_panda_shared}"
OFFICIAL_ROOT="${OFFICIAL_ROOT:-/mnt/f/fur_hair_unified_data/neuralfur_official/extracted/data/Artemis/panda_processed_GH2/walk}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-/mnt/f/fur_hair_unified_data/benchmarks/hairgs_hybrid/panda_protocol_r480}"
RESULT_ROOT="${RESULT_ROOT:-/mnt/f/fur_hair_unified_data/benchmarks/hairgs_hybrid/panda_hybrid_full97k_5k_v1}"
INITIAL_PLY="${INITIAL_PLY:-${PROTOCOL_ROOT}/initial/point_cloud.ply}"
ROUTE_SIDECAR="${ROUTE_SIDECAR:-${PROTOCOL_ROOT}/initial/routes.npz}"
EXPECTED_SEGMENTS="${EXPECTED_SEGMENTS:-96991}"
VARIANT="${VARIANT:-adaptive}"
ROUTE_MODE="${ROUTE_MODE:-soft}"
ROUTE_SCHEDULE="${ROUTE_SCHEDULE:-constant}"
ROUTE_WARMUP_STEPS="${ROUTE_WARMUP_STEPS:-0}"
ROUTE_TEMPERATURE="${ROUTE_TEMPERATURE:-0.75}"
ROUTE_TEMPERATURE_FINAL="${ROUTE_TEMPERATURE_FINAL:-${ROUTE_TEMPERATURE}}"
ROUTE_TEMPERATURE_ANNEAL_STEPS="${ROUTE_TEMPERATURE_ANNEAL_STEPS:-0}"
NEUTRAL_ACTIVATION_STEPS="${NEUTRAL_ACTIVATION_STEPS:-1000}"
TOPOLOGY_HARD_FORWARD_START="${TOPOLOGY_HARD_FORWARD_START:--1}"
LAMBDA_ROUTE_EXPLORE="${LAMBDA_ROUTE_EXPLORE:-0}"
LAMBDA_ROUTE_PRIOR="${LAMBDA_ROUTE_PRIOR:-0}"
LAMBDA_BUDGET="${LAMBDA_BUDGET:-0}"
TARGET_ACTIVE_SPLATS="${TARGET_ACTIVE_SPLATS:-0}"
STEPS="${STEPS:-5000}"
START_ITERATION=30000
FINAL_ITERATION=$((START_ITERATION + STEPS))
PROTOCOL_ID="panda-hairgs-28fit-8heldout-r480-v1"
SKIP_EVAL="${SKIP_EVAL:-0}"

export PYTHONPATH="${COMPAT_ROOT}:${HAIRGS_ROOT}:${EXPERIMENT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

run_hairgs() { "${CONDA_BIN}" run --no-capture-output -n hair-gs python "$@"; }

preflight() {
  mkdir -p "${RESULT_ROOT}/audit"
  for path in \
    "${PANDA_ROOT}/initial_body_gaussians.ply" \
    "${PANDA_ROOT}/static_stage1.npz" \
    "${PANDA_ROOT}/train_v28/camera_manifest.json" \
    "${PANDA_ROOT}/test_v8/camera_manifest.json"; do
    test -f "${path}"
  done
  if [[ ! -f "${PROTOCOL_ROOT}/input_report.json" ]]; then
    run_hairgs "${EXPERIMENT_ROOT}/hairgs_hybrid/prepare_panda_hairgs.py" \
      --panda-root "${PANDA_ROOT}" --official-root "${OFFICIAL_ROOT}" \
      --output-root "${PROTOCOL_ROOT}" --width 480 --height 270 \
      > "${RESULT_ROOT}/audit/prepare_inputs.log"
  fi
  run_hairgs "${EXPERIMENT_ROOT}/hairgs_hybrid/prepare_hair_mask_manifest.py" \
    --dataset "${PROTOCOL_ROOT}/train_v28" \
    --dataset "${PROTOCOL_ROOT}/test_v8" \
    --output "${RESULT_ROOT}/audit/fur_mask_manifest.json"
  run_hairgs - <<PY
import json
import numpy as np
from pathlib import Path
from plyfile import PlyData
report=json.loads(Path("${PROTOCOL_ROOT}/input_report.json").read_text())
assert report["status"] == "complete"
assert report["train"]["count"] == 28 and report["test"]["count"] == 8
assert report["semantic_policy"] == "all segments are fur; no head/body/residual route"
ply=PlyData.read("${INITIAL_PLY}")
routes=np.load("${ROUTE_SIDECAR}")["route_id"]
assert ply["edge"].count == int("${EXPECTED_SEGMENTS}")
assert routes.shape == (int("${EXPECTED_SEGMENTS}"),)
print({"segments": ply["edge"].count, "routes": np.bincount(routes, minlength=3).tolist()})
PY
  sha256sum \
    "${PANDA_ROOT}/initial_body_gaussians.ply" \
    "${PANDA_ROOT}/static_stage1.npz" \
    "${INITIAL_PLY}" \
    "${ROUTE_SIDECAR}" \
    "${PROTOCOL_ROOT}/train_v28/camera_manifest.json" \
    "${PROTOCOL_ROOT}/test_v8/camera_manifest.json" \
    > "${RESULT_ROOT}/audit/provenance.sha256"
}

prepare_model() {
  if [[ -e "${RESULT_ROOT}/cfg_args" && ! -f "${RESULT_ROOT}/.prepared" ]]; then
    echo "Refusing ambiguous restart: ${RESULT_ROOT}" >&2
    exit 3
  fi
  [[ -f "${RESULT_ROOT}/.prepared" ]] && return
  mkdir -p "${RESULT_ROOT}/point_cloud/iteration_${START_ITERATION}" "${RESULT_ROOT}/audit"
  cp "${INITIAL_PLY}" \
    "${RESULT_ROOT}/point_cloud/iteration_${START_ITERATION}/point_cloud.ply"
  cp "${ROUTE_SIDECAR}" \
    "${RESULT_ROOT}/point_cloud/iteration_${START_ITERATION}/routes.npz"
  touch "${RESULT_ROOT}/.prepared"
}

train_model() {
  [[ -f "${RESULT_ROOT}/point_cloud/iteration_${FINAL_ITERATION}/hybrid_state.npz" ]] && return
  cd "${HAIRGS_ROOT}"
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu \
    --format=csv,noheader,nounits -l 1 > "${RESULT_ROOT}/audit/train_gpu.csv" &
  local monitor_pid=$!
  trap 'kill "${monitor_pid}" 2>/dev/null || true' EXIT
  /usr/bin/time -v -o "${RESULT_ROOT}/audit/train_time.txt" \
    "${CONDA_BIN}" run --no-capture-output -n hair-gs python \
      "${EXPERIMENT_ROOT}/hairgs_hybrid/train_hybrid.py" \
      -s "${PROTOCOL_ROOT}/train_v28" -m "${RESULT_ROOT}" -r 1 \
      --sh_degree 3 \
      --route-sidecar "${RESULT_ROOT}/point_cloud/iteration_${START_ITERATION}/routes.npz" \
      --hybrid-init-mode "${VARIANT}" \
      --hybrid-route-mode "${ROUTE_MODE}" \
      --hybrid-route-schedule "${ROUTE_SCHEDULE}" \
      --hybrid-route-warmup-steps "${ROUTE_WARMUP_STEPS}" \
      --hybrid-temperature "${ROUTE_TEMPERATURE}" \
      --hybrid-temperature-final "${ROUTE_TEMPERATURE_FINAL}" \
      --hybrid-temperature-anneal-steps "${ROUTE_TEMPERATURE_ANNEAL_STEPS}" \
      --hybrid-neutral-activation-steps "${NEUTRAL_ACTIVATION_STEPS}" \
      --hybrid-topology-hard-forward-start "${TOPOLOGY_HARD_FORWARD_START}" \
      --iterations "${STEPS}" --densify_until_iter 0 \
      --merge_interval 100000 --growth_interval 100000 \
      --position_lr_max_steps "${STEPS}" --lambda_mask 0 \
      --lambda_orientation "${LAMBDA_ORIENTATION:-10}" --lambda_smooth 0 \
      --hybrid-shell-strength 0.16 --hybrid-fin-strength 0.20 \
      --hybrid-strength-max 0.80 \
      --lambda-hybrid-mask-bce "${LAMBDA_MASK_BCE:-0.30}" \
      --lambda-hybrid-mask-fp "${LAMBDA_MASK_FP:-0.60}" \
      --lambda-hybrid-mask-fn "${LAMBDA_MASK_FN:-0.60}" \
      --lambda-hybrid-hole "${LAMBDA_HOLE:-0.20}" \
      --lambda-hybrid-overflow-peak "${LAMBDA_OVERFLOW_PEAK:-0.60}" \
      --lambda-hybrid-background-rgb "${LAMBDA_BACKGROUND_RGB:-0.50}" \
      --lambda-hybrid-route-cost 0.0002 \
      --lambda-hybrid-route-entropy 0.00005 \
      --lambda-hybrid-route-explore "${LAMBDA_ROUTE_EXPLORE}" \
      --lambda-hybrid-route-prior "${LAMBDA_ROUTE_PRIOR}" \
      --lambda-hybrid-budget "${LAMBDA_BUDGET}" \
      --hybrid-target-active-splats "${TARGET_ACTIVE_SPLATS}" \
      --lambda-hybrid-shape-reg 0.00002 \
      --lambda-hybrid-center-anchor "${LAMBDA_CENTER_ANCHOR:-0.50}" \
      --lambda-hybrid-length-anchor "${LAMBDA_LENGTH_ANCHOR:-0.25}" \
      --lambda-hybrid-width-anchor "${LAMBDA_WIDTH_ANCHOR:-0.10}" \
      --save_frequency "${STEPS}" --eval_frequency "${STEPS}" \
      --logger tensorboard --quiet 2>&1 | tee "${RESULT_ROOT}/audit/train.log"
  kill "${monitor_pid}" 2>/dev/null || true
  wait "${monitor_pid}" 2>/dev/null || true
  trap - EXIT
  test -f "${RESULT_ROOT}/point_cloud/iteration_${FINAL_ITERATION}/point_cloud.ply"
  test -f "${RESULT_ROOT}/point_cloud/iteration_${FINAL_ITERATION}/hybrid_state.npz"
}

render_split() {
  local split="$1" dataset="${PROTOCOL_ROOT}/$2" output="${RESULT_ROOT}/evaluation_$1"
  [[ -f "${output}/external_evaluation/evaluation.json" ]] && return
  mkdir -p "${output}"
  rm -rf "${RESULT_ROOT}/render/train/iteration_${FINAL_ITERATION}"
  cd "${HAIRGS_ROOT}"
  "${CONDA_BIN}" run --no-capture-output -n hair-gs python \
    "${EXPERIMENT_ROOT}/hairgs_hybrid/render_hybrid.py" \
    -s "${dataset}" -m "${RESULT_ROOT}" -r 1 --iterations "${FINAL_ITERATION}" \
    --skip_test --type 0 --quiet
  "${CONDA_BIN}" run --no-capture-output -n hair-gs python \
    "${EXPERIMENT_ROOT}/hairgs_hybrid/render_hybrid.py" \
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
      --ground-truth-dir "${PANDA_ROOT}/$2/images" \
      --output-dir "${output}/external_evaluation" \
      --method "Panda HairGS hybrid ${VARIANT}/${ROUTE_MODE} (${STEPS} steps, ${EXPECTED_SEGMENTS} segments)" \
      --protocol-id "${PROTOCOL_ID}-${split}" --device cuda
}

preflight
prepare_model
train_model
if [[ "${SKIP_EVAL}" != "1" ]]; then
  render_split train train_v28
  render_split test test_v8
fi

echo "Panda HairGS hybrid complete: ${RESULT_ROOT}"
