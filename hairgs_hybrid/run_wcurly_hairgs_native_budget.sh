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
INITIAL_PLY="${INITIAL_PLY:?INITIAL_PLY is required}"
EXPECTED_TOTAL="${EXPECTED_TOTAL:?EXPECTED_TOTAL is required}"
RESULT_ROOT="${RESULT_ROOT:?RESULT_ROOT is required}"
START_ITERATION=15007
STEPS="${STEPS:-5000}"
FINAL_ITERATION=$((START_ITERATION + STEPS))
PROTOCOL_ID="wcurly-static-train12-test4-local500-v1"
SKIP_EVAL="${SKIP_EVAL:-0}"

export PYTHONPATH="${COMPAT_ROOT}:${HAIRGS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

run_hairgs() { "${CONDA_BIN}" run --no-capture-output -n hair-gs python "$@"; }

preflight() {
  mkdir -p "${RESULT_ROOT}/audit"
  test -f "${INITIAL_PLY}"
  test -f "${BASELINE}/cfg_args"
  run_hairgs - <<PY
from plyfile import PlyData
ply = PlyData.read("${INITIAL_PLY}")
assert ply["edge"].count == int("${EXPECTED_TOTAL}")
assert ply["segment"].count == int("${EXPECTED_TOTAL}")
print({"segments": ply["edge"].count})
PY
  sha256sum "${INITIAL_PLY}" \
    "${PROTOCOL_ROOT}/train/camera_manifest.json" \
    "${PROTOCOL_ROOT}/test/camera_manifest.json" \
    > "${RESULT_ROOT}/audit/provenance.sha256"
}

prepare() {
  if [[ -e "${RESULT_ROOT}/cfg_args" && ! -f "${RESULT_ROOT}/.prepared" ]]; then
    echo "Refusing ambiguous restart: ${RESULT_ROOT}" >&2
    exit 3
  fi
  [[ -f "${RESULT_ROOT}/.prepared" ]] && return
  mkdir -p "${RESULT_ROOT}/point_cloud/iteration_${START_ITERATION}" "${RESULT_ROOT}/audit"
  cp -f "${INITIAL_PLY}" \
    "${RESULT_ROOT}/point_cloud/iteration_${START_ITERATION}/point_cloud.ply"
  cp -f "${BASELINE}/cfg_args" "${RESULT_ROOT}/cfg_args"
  for name in hair_eval_data.npz head_reconstruction_data.npz head_mesh.ply input.ply; do
    ln -sfn "${BASELINE}/${name}" "${RESULT_ROOT}/${name}"
  done
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
      -s "${TRAIN_DATASET}" -m "${RESULT_ROOT}" -r 1 \
      --iterations "${STEPS}" --densify_until_iter 0 \
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
  local split="$1" dataset="$2" manifest="$3" output="$4"
  [[ -f "${output}/external_evaluation/evaluation.json" ]] && return
  mkdir -p "${output}"
  rm -rf "${RESULT_ROOT}/render/train/iteration_${FINAL_ITERATION}"
  cd "${HAIRGS_ROOT}"
  /usr/bin/time -v -o "${output}_render_rgb_time.txt" \
    "${CONDA_BIN}" run --no-capture-output -n hair-gs python render.py \
      -s "${dataset}" -m "${RESULT_ROOT}" -r 1 --iterations "${FINAL_ITERATION}" \
      --skip_test --type 0 --quiet
  /usr/bin/time -v -o "${output}_render_mask_time.txt" \
    "${CONDA_BIN}" run --no-capture-output -n hair-gs python render.py \
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
      --method "WCurly native HairGS fixed-budget (${STEPS} steps, ${EXPECTED_TOTAL} total segments)" \
      --protocol-id "${PROTOCOL_ID}-${split}" --device cuda
}

preflight
prepare
train_model
if [[ "${SKIP_EVAL}" != "1" ]]; then
  render_split train "${TRAIN_DATASET}" "${PROTOCOL_ROOT}/train/camera_manifest.json" \
    "${RESULT_ROOT}/evaluation_train"
  render_split test "${TEST_DATASET}" "${PROTOCOL_ROOT}/test/camera_manifest.json" \
    "${RESULT_ROOT}/evaluation_test"
fi
echo "WCurly native HairGS fixed-budget complete: ${RESULT_ROOT}"
