#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/mnt/f/unifur_hairgs_hybrid_experiment}"
EVAL_PROJECT="${EVAL_PROJECT:-/home/aoki/Differentiable_Physical_Driven_3DGS_for_monocular_Animal_Motion_Reconstruction}"
HAIRGS_ROOT="${HAIRGS_ROOT:-/home/aoki/fur_hair_baselines/hair-gs}"
CONDA_BIN="${CONDA_BIN:-/home/aoki/miniconda3/bin/conda}"
COMPAT_ROOT="${HAIRGS_COMPAT_ROOT:-${EVAL_PROJECT}/compat/hairgs_sitecustomize}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-/mnt/f/fur_hair_unified_data/benchmarks/hairgs_wcurly_static_protocol}"
TRAIN_DATASET="${PROTOCOL_ROOT}/hairgs_train12_dataset_500"
TEST_DATASET="${PROTOCOL_ROOT}/hairgs_test4_dataset_500"
BASELINE="${BASELINE:-/mnt/f/fur_hair_unified_data/benchmarks/hair_baseline_local24gb/wcurly/hairgs_native500_s1_15k_s3_15k}"
ROUTE_SOURCE="${ROUTE_SOURCE:-/mnt/f/fur_hair_unified_data/benchmarks/hairgs_adaptive_routes/wcurly_stage2_routes_v1}"
INITIAL_PLY="${INITIAL_PLY:-${ROUTE_SOURCE}/control.ply}"
ROUTE_SIDECAR="${ROUTE_SIDECAR:-${ROUTE_SOURCE}/control.routes.npz}"
EXPECTED_TOTAL="${EXPECTED_TOTAL:-70214}"
EXPECTED_ACTIVE="${EXPECTED_ACTIVE:-40936}"
RESULT_ROOT="${RESULT_ROOT:-/mnt/f/fur_hair_unified_data/benchmarks/hairgs_hybrid/wcurly_haironly_maskbounded5k_v1}"
VARIANT="${VARIANT:-adaptive}"
START_ITERATION=15007
STEPS="${STEPS:-5000}"
FINAL_ITERATION=$((START_ITERATION + STEPS))
PROTOCOL_ID="wcurly-static-train12-test4-local500-v1"
SKIP_EVAL="${SKIP_EVAL:-0}"
LAMBDA_MASK_BCE="${LAMBDA_MASK_BCE:-0.04}"
LAMBDA_MASK_FP="${LAMBDA_MASK_FP:-0.20}"
LAMBDA_MASK_FN="${LAMBDA_MASK_FN:-0.08}"
LAMBDA_HOLE="${LAMBDA_HOLE:-0.03}"
LAMBDA_OVERFLOW_PEAK="${LAMBDA_OVERFLOW_PEAK:-0.12}"

export PYTHONPATH="${COMPAT_ROOT}:${HAIRGS_ROOT}:${EXPERIMENT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

run_hairgs() { "${CONDA_BIN}" run --no-capture-output -n hair-gs python "$@"; }

preflight() {
  mkdir -p "${RESULT_ROOT}/audit"
  run_hairgs "${EXPERIMENT_ROOT}/hairgs_hybrid/prepare_hair_mask_manifest.py" \
    --dataset "${TRAIN_DATASET}" --dataset "${TEST_DATASET}" \
    --output "${RESULT_ROOT}/audit/hair_only_mask_manifest.json"
  test -f "${INITIAL_PLY}"
  test -f "${ROUTE_SIDECAR}"
  test -f "${BASELINE}/cfg_args"
  run_hairgs - <<PY
import numpy as np
from plyfile import PlyData
ply = PlyData.read("${INITIAL_PLY}")
routes = np.load("${ROUTE_SIDECAR}")["route_id"]
assert ply["edge"].count == int("${EXPECTED_TOTAL}")
assert routes.shape == (int("${EXPECTED_TOTAL}"),)
assert int((routes >= 0).sum()) == int("${EXPECTED_ACTIVE}")
assert int((routes < 0).sum()) == int("${EXPECTED_TOTAL}") - int("${EXPECTED_ACTIVE}")
print({"total": len(routes), "base": int((routes < 0).sum()),
       "active": int((routes >= 0).sum()),
       "routes": np.bincount(routes[routes >= 0], minlength=3).tolist()})
PY
  sha256sum "${INITIAL_PLY}" "${ROUTE_SIDECAR}" \
    "${PROTOCOL_ROOT}/train/camera_manifest.json" \
    "${PROTOCOL_ROOT}/test/camera_manifest.json" \
    > "${RESULT_ROOT}/audit/provenance.sha256"
}

prepare() {
  if [[ -e "${RESULT_ROOT}/cfg_args" && ! -f "${RESULT_ROOT}/.prepared" ]]; then
    echo "Refusing ambiguous restart: ${RESULT_ROOT}" >&2
    exit 3
  fi
  if [[ -f "${RESULT_ROOT}/.prepared" ]]; then return; fi
  mkdir -p "${RESULT_ROOT}/point_cloud/iteration_${START_ITERATION}" "${RESULT_ROOT}/audit"
  cp -f "${INITIAL_PLY}" \
    "${RESULT_ROOT}/point_cloud/iteration_${START_ITERATION}/point_cloud.ply"
  cp -f "${ROUTE_SIDECAR}" \
    "${RESULT_ROOT}/point_cloud/iteration_${START_ITERATION}/routes.npz"
  cp -f "${BASELINE}/cfg_args" "${RESULT_ROOT}/cfg_args"
  for name in hair_eval_data.npz head_reconstruction_data.npz head_mesh.ply input.ply; do
    ln -sfn "${BASELINE}/${name}" "${RESULT_ROOT}/${name}"
  done
  touch "${RESULT_ROOT}/.prepared"
}

train_model() {
  if [[ -f "${RESULT_ROOT}/point_cloud/iteration_${FINAL_ITERATION}/hybrid_state.npz" ]]; then
    return
  fi
  cd "${HAIRGS_ROOT}"
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu \
    --format=csv,noheader,nounits -l 1 > "${RESULT_ROOT}/audit/train_gpu.csv" &
  local monitor_pid=$!
  trap 'kill "${monitor_pid}" 2>/dev/null || true' EXIT
  /usr/bin/time -v -o "${RESULT_ROOT}/audit/train_time.txt" \
    "${CONDA_BIN}" run --no-capture-output -n hair-gs python \
      "${EXPERIMENT_ROOT}/hairgs_hybrid/train_hybrid.py" \
      -s "${TRAIN_DATASET}" -m "${RESULT_ROOT}" -r 1 \
      --route-sidecar "${RESULT_ROOT}/point_cloud/iteration_${START_ITERATION}/routes.npz" \
      --hybrid-init-mode "${VARIANT}" \
      --iterations "${STEPS}" --densify_until_iter 0 \
      --merge_interval 100000 --growth_interval 100000 \
      --position_lr_max_steps "${STEPS}" --lambda_mask 0 \
      --hybrid-shell-strength 0.16 --hybrid-fin-strength 0.20 \
      --hybrid-strength-max 0.80 \
      --lambda-hybrid-mask-bce "${LAMBDA_MASK_BCE}" \
      --lambda-hybrid-mask-fp "${LAMBDA_MASK_FP}" \
      --lambda-hybrid-mask-fn "${LAMBDA_MASK_FN}" \
      --lambda-hybrid-hole "${LAMBDA_HOLE}" \
      --lambda-hybrid-overflow-peak "${LAMBDA_OVERFLOW_PEAK}" \
      --lambda-hybrid-route-cost 0.0002 \
      --lambda-hybrid-route-entropy 0.00005 \
      --lambda-hybrid-shape-reg 0.00002 \
      --save_frequency "${STEPS}" --eval_frequency "${STEPS}" \
      --logger tensorboard --quiet 2>&1 | tee "${RESULT_ROOT}/audit/train.log"
  kill "${monitor_pid}" 2>/dev/null || true
  wait "${monitor_pid}" 2>/dev/null || true
  trap - EXIT
  test -f "${RESULT_ROOT}/point_cloud/iteration_${FINAL_ITERATION}/point_cloud.ply"
  test -f "${RESULT_ROOT}/point_cloud/iteration_${FINAL_ITERATION}/hybrid_state.npz"
}

audit_base() {
  run_hairgs "${EXPERIMENT_ROOT}/hairgs_hybrid/audit_frozen_base.py" \
    --initial-ply "${RESULT_ROOT}/point_cloud/iteration_${START_ITERATION}/point_cloud.ply" \
    --final-ply "${RESULT_ROOT}/point_cloud/iteration_${FINAL_ITERATION}/point_cloud.ply" \
    --routes "${RESULT_ROOT}/point_cloud/iteration_${START_ITERATION}/routes.npz" \
    --output "${RESULT_ROOT}/audit/frozen_base_audit.json"
}

render_split() {
  local split="$1" dataset="$2" manifest="$3" output="$4"
  if [[ -f "${output}/external_evaluation/evaluation.json" ]]; then return; fi
  mkdir -p "${output}"
  rm -rf "${RESULT_ROOT}/render/train/iteration_${FINAL_ITERATION}"
  cd "${HAIRGS_ROOT}"
  /usr/bin/time -v -o "${output}_render_rgb_time.txt" \
    "${CONDA_BIN}" run --no-capture-output -n hair-gs python \
      "${EXPERIMENT_ROOT}/hairgs_hybrid/render_hybrid.py" \
      -s "${dataset}" -m "${RESULT_ROOT}" -r 1 --iterations "${FINAL_ITERATION}" \
      --skip_test --type 0 --quiet
  /usr/bin/time -v -o "${output}_render_mask_time.txt" \
    "${CONDA_BIN}" run --no-capture-output -n hair-gs python \
      "${EXPERIMENT_ROOT}/hairgs_hybrid/render_hybrid.py" \
      -s "${dataset}" -m "${RESULT_ROOT}" -r 1 --iterations "${FINAL_ITERATION}" \
      --skip_test --type 2 --quiet
  cd "${EVAL_PROJECT}"
  run_hairgs scripts/export_hairgs_external_renders.py \
    --model-dir "${RESULT_ROOT}" --camera-dataset "${dataset}" \
    --camera-manifest "${manifest}" --output-dir "${output}/external_renders" \
    --iteration "${FINAL_ITERATION}" --output-size 500 500 --hairgs-root "${HAIRGS_ROOT}"
  "${CONDA_BIN}" run --no-capture-output -n dpd3dgs-animal python \
    scripts/evaluate_external_renders.py \
      --render-manifest "${output}/external_renders/render_manifest.json" \
      --ground-truth-dir "${PROTOCOL_ROOT}/${split}/images" \
      --output-dir "${output}/external_evaluation" \
      --method "HairGS hybrid hair-only mask-bounded (${STEPS} steps)" \
      --protocol-id "${PROTOCOL_ID}-${split}" --device cuda
}

preflight
prepare
train_model
audit_base
if [[ "${SKIP_EVAL}" != "1" ]]; then
  render_split train "${TRAIN_DATASET}" "${PROTOCOL_ROOT}/train/camera_manifest.json" \
    "${RESULT_ROOT}/evaluation_train"
  render_split test "${TEST_DATASET}" "${PROTOCOL_ROOT}/test/camera_manifest.json" \
    "${RESULT_ROOT}/evaluation_test"
fi
