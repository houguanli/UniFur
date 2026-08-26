#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/mnt/f/unifur_hairgs_hybrid_experiment}"
EVAL_PROJECT="${EVAL_PROJECT:-/home/aoki/Differentiable_Physical_Driven_3DGS_for_monocular_Animal_Motion_Reconstruction}"
HAIRGS_ROOT="${HAIRGS_ROOT:-/home/aoki/fur_hair_baselines/hair-gs}"
CONDA_BIN="${CONDA_BIN:-/home/aoki/miniconda3/bin/conda}"
COMPAT_ROOT="${HAIRGS_COMPAT_ROOT:-${EVAL_PROJECT}/compat/hairgs_sitecustomize}"
PANDA_ROOT="${PANDA_ROOT:-/mnt/f/fur_hair_unified_data/benchmarks/neuralfur_panda_shared}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-/mnt/f/fur_hair_unified_data/benchmarks/hairgs_hybrid/panda_protocol_r480}"
RESULT_ROOT="${RESULT_ROOT:-/mnt/f/fur_hair_unified_data/benchmarks/hairgs_hybrid/panda_native_hairgs_5k_v1}"
INITIAL_PLY="${INITIAL_PLY:-${PROTOCOL_ROOT}/initial/point_cloud.ply}"
EXPECTED_SEGMENTS="${EXPECTED_SEGMENTS:-96991}"
STEPS="${STEPS:-5000}"
START_ITERATION=30000
FINAL_ITERATION=$((START_ITERATION + STEPS))
PROTOCOL_ID="panda-hairgs-28fit-8heldout-r480-v1"

export PYTHONPATH="${COMPAT_ROOT}:${HAIRGS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

run_hairgs() { "${CONDA_BIN}" run --no-capture-output -n hair-gs python "$@"; }

preflight() {
  mkdir -p "${RESULT_ROOT}/audit"
  "${CONDA_BIN}" run -n hair-gs python - <<PY
import json
from pathlib import Path
from plyfile import PlyData
report = json.loads(Path("${PROTOCOL_ROOT}/input_report.json").read_text())
assert report["status"] == "complete"
assert report["train"]["count"] == 28 and report["test"]["count"] == 8
assert report["semantic_policy"] == "all segments are fur; no head/body/residual route"
assert PlyData.read("${INITIAL_PLY}")["edge"].count == int("${EXPECTED_SEGMENTS}")
PY
  sha256sum \
    "${INITIAL_PLY}" \
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
  mkdir -p "${RESULT_ROOT}/point_cloud/iteration_${START_ITERATION}"
  cp "${INITIAL_PLY}" \
    "${RESULT_ROOT}/point_cloud/iteration_${START_ITERATION}/point_cloud.ply"
  touch "${RESULT_ROOT}/.prepared"
}

train_model() {
  [[ -f "${RESULT_ROOT}/point_cloud/iteration_${FINAL_ITERATION}/point_cloud.ply" ]] && return
  cd "${HAIRGS_ROOT}"
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu \
    --format=csv,noheader,nounits -l 1 > "${RESULT_ROOT}/audit/train_gpu.csv" &
  local monitor_pid=$!
  trap 'kill "${monitor_pid}" 2>/dev/null || true' EXIT
  /usr/bin/time -v -o "${RESULT_ROOT}/audit/train_time.txt" \
    "${CONDA_BIN}" run --no-capture-output -n hair-gs python train.py \
      -s "${PROTOCOL_ROOT}/train_v28" -m "${RESULT_ROOT}" -r 1 \
      --sh_degree 3 --iterations "${STEPS}" --densify_until_iter 0 \
      --merge_interval 100000 --growth_interval 100000 \
      --position_lr_max_steps "${STEPS}" \
      --lambda_mask "${LAMBDA_MASK:-0.01}" \
      --lambda_orientation "${LAMBDA_ORIENTATION:-10}" --lambda_smooth 0 \
      --save_frequency "${STEPS}" --eval_frequency "${STEPS}" \
      --logger tensorboard --quiet 2>&1 | tee "${RESULT_ROOT}/audit/train.log"
  kill "${monitor_pid}" 2>/dev/null || true
  wait "${monitor_pid}" 2>/dev/null || true
  trap - EXIT
  test -f "${RESULT_ROOT}/point_cloud/iteration_${FINAL_ITERATION}/point_cloud.ply"
}

render_split() {
  local split="$1" dataset="${PROTOCOL_ROOT}/$2" output="${RESULT_ROOT}/evaluation_$1"
  [[ -f "${output}/external_evaluation/evaluation.json" ]] && return
  mkdir -p "${output}" "${RESULT_ROOT}/audit"
  rm -rf "${RESULT_ROOT}/render/train/iteration_${FINAL_ITERATION}"
  cd "${HAIRGS_ROOT}"
  /usr/bin/time -f '%e' -o "${RESULT_ROOT}/audit/render_${split}_rgb_seconds.txt" \
    "${CONDA_BIN}" run --no-capture-output -n hair-gs python render.py \
      -s "${dataset}" -m "${RESULT_ROOT}" -r 1 --iterations "${FINAL_ITERATION}" \
      --skip_test --type 0 --quiet
  /usr/bin/time -f '%e' -o "${RESULT_ROOT}/audit/render_${split}_mask_seconds.txt" \
    "${CONDA_BIN}" run --no-capture-output -n hair-gs python render.py \
      -s "${dataset}" -m "${RESULT_ROOT}" -r 1 --iterations "${FINAL_ITERATION}" \
      --skip_test --type 2 --quiet
  cd "${EVAL_PROJECT}"
  run_hairgs scripts/export_hairgs_external_renders.py \
    --model-dir "${RESULT_ROOT}" --camera-dataset "${dataset}" \
    --camera-manifest "${dataset}/camera_manifest.json" \
    --output-dir "${output}/external_renders" --iteration "${FINAL_ITERATION}" \
    --output-size 480 270 --hairgs-root "${HAIRGS_ROOT}"
  "${CONDA_BIN}" run --no-capture-output -n dpd3dgs-animal python \
    scripts/evaluate_external_renders.py \
      --render-manifest "${output}/external_renders/render_manifest.json" \
      --ground-truth-dir "${PANDA_ROOT}/$2/images" \
      --output-dir "${output}/external_evaluation" \
      --method "Panda native HairGS (${STEPS} steps, ${EXPECTED_SEGMENTS} segments)" \
      --protocol-id "${PROTOCOL_ID}-${split}" --device cuda
}

preflight
prepare_model
train_model
render_split train train_v28
render_split test test_v8
echo "Panda native HairGS complete: ${RESULT_ROOT}"
