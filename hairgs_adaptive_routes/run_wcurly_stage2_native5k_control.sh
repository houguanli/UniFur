#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/aoki/Differentiable_Physical_Driven_3DGS_for_monocular_Animal_Motion_Reconstruction}"
HAIRGS_ROOT="${HAIRGS_ROOT:-/home/aoki/fur_hair_baselines/hair-gs}"
CONDA_BIN="${CONDA_BIN:-/home/aoki/miniconda3/bin/conda}"
COMPAT_ROOT="${HAIRGS_COMPAT_ROOT:-${PROJECT_ROOT}/compat/hairgs_sitecustomize}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-/mnt/f/fur_hair_unified_data/benchmarks/hairgs_wcurly_static_protocol}"
TRAIN_DATASET="${PROTOCOL_ROOT}/hairgs_train12_dataset_500"
TEST_DATASET="${PROTOCOL_ROOT}/hairgs_test4_dataset_500"
BASELINE="${BASELINE:-/mnt/f/fur_hair_unified_data/benchmarks/hair_baseline_local24gb/wcurly/hairgs_native500_s1_15k_s3_15k}"
SOURCE_ROOT="${SOURCE_ROOT:-/mnt/f/fur_hair_unified_data/benchmarks/hairgs_adaptive_routes/wcurly_stage2_routes_v1}"
RESULT_ROOT="${RESULT_ROOT:-/mnt/f/fur_hair_unified_data/benchmarks/hairgs_adaptive_routes/wcurly_stage2_native4k1k_control_v1}"
START_ITERATION=15007
PHASE1_STEPS=4000
PHASE2_STEPS=1000
PHASE1_ITERATION=$((START_ITERATION + PHASE1_STEPS))
FINAL_ITERATION=$((PHASE1_ITERATION + PHASE2_STEPS))
PROTOCOL_ID="wcurly-static-train12-test4-local500-v1"

export PYTHONPATH="${COMPAT_ROOT}:${HAIRGS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

run_hairgs() { "${CONDA_BIN}" run --no-capture-output -n hair-gs python "$@"; }

prepare_routes() {
  if [[ -f "${SOURCE_ROOT}/control.ply" \
        && -f "${SOURCE_ROOT}/control.routes.npz" \
        && -f "${SOURCE_ROOT}/control_report.json" ]]; then return; fi
  mkdir -p "${SOURCE_ROOT}"
  run_hairgs "${PROJECT_ROOT}/hairgs_adaptive_routes/assign_routes.py" \
    --input "${BASELINE}/point_cloud/iteration_${START_ITERATION}/point_cloud.ply" \
    --output "${SOURCE_ROOT}/control.ply" \
    --report "${SOURCE_ROOT}/control_report.json"
}

prepare() {
  if [[ -e "${RESULT_ROOT}/cfg_args" && ! -f "${RESULT_ROOT}/.prepared" ]]; then
    echo "Refusing ambiguous restart: ${RESULT_ROOT}" >&2
    exit 3
  fi
  if [[ -f "${RESULT_ROOT}/.prepared" ]]; then return; fi
  mkdir -p "${RESULT_ROOT}/point_cloud/iteration_${START_ITERATION}" "${RESULT_ROOT}/audit"
  cp -f "${SOURCE_ROOT}/control.ply" \
    "${RESULT_ROOT}/point_cloud/iteration_${START_ITERATION}/point_cloud.ply"
  cp -f "${BASELINE}/cfg_args" "${RESULT_ROOT}/cfg_args"
  for name in hair_eval_data.npz head_reconstruction_data.npz head_mesh.ply input.ply; do
    ln -sfn "${BASELINE}/${name}" "${RESULT_ROOT}/${name}"
  done
  touch "${RESULT_ROOT}/.prepared"
}

train_phase() {
  local name="$1" steps="$2" final_iteration="$3"
  local marker="${RESULT_ROOT}/.${name}_complete"
  if [[ -f "${marker}" ]]; then return; fi
  cd "${HAIRGS_ROOT}"
  /usr/bin/time -v -o "${RESULT_ROOT}/audit/${name}_time.txt" \
    "${CONDA_BIN}" run --no-capture-output -n hair-gs python train.py \
      -s "${TRAIN_DATASET}" -m "${RESULT_ROOT}" -r 1 \
      --iterations "${steps}" --densify_until_iter 0 \
      --merge_interval 100000 --growth_interval 100000 \
      --position_lr_max_steps "${steps}" \
      --save_frequency "${steps}" --eval_frequency "${steps}" \
      --logger none --quiet 2>&1 | tee "${RESULT_ROOT}/audit/${name}.log"
  test -f "${RESULT_ROOT}/point_cloud/iteration_${final_iteration}/point_cloud.ply"
  touch "${marker}"
}

render_split() {
  local split="$1" dataset="$2" manifest="$3" output="$4"
  if [[ -f "${output}/external_evaluation/evaluation.json" ]]; then return; fi
  mkdir -p "${output}"
  rm -rf "${RESULT_ROOT}/render/train/iteration_${FINAL_ITERATION}"
  cd "${HAIRGS_ROOT}"
  run_hairgs render.py -s "${dataset}" -m "${RESULT_ROOT}" -r 1 \
    --iterations "${FINAL_ITERATION}" --skip_test --type 0 --quiet
  run_hairgs render.py -s "${dataset}" -m "${RESULT_ROOT}" -r 1 \
    --iterations "${FINAL_ITERATION}" --skip_test --type 2 --quiet
  cd "${PROJECT_ROOT}"
  run_hairgs scripts/export_hairgs_external_renders.py \
    --model-dir "${RESULT_ROOT}" --camera-dataset "${dataset}" \
    --camera-manifest "${manifest}" --output-dir "${output}/external_renders" \
    --iteration "${FINAL_ITERATION}" --output-size 500 500 --hairgs-root "${HAIRGS_ROOT}"
  "${CONDA_BIN}" run --no-capture-output -n dpd3dgs-animal python \
    scripts/evaluate_external_renders.py \
      --render-manifest "${output}/external_renders/render_manifest.json" \
      --ground-truth-dir "${PROTOCOL_ROOT}/${split}/images" \
      --output-dir "${output}/external_evaluation" \
      --method "HairGS Stage2 + native 4k + native 1k control" \
      --protocol-id "${PROTOCOL_ID}-${split}" --device cuda
}

prepare_routes
prepare
train_phase native4k "${PHASE1_STEPS}" "${PHASE1_ITERATION}"
train_phase native1k "${PHASE2_STEPS}" "${FINAL_ITERATION}"
render_split train "${TRAIN_DATASET}" "${PROTOCOL_ROOT}/train/camera_manifest.json" \
  "${RESULT_ROOT}/evaluation_train"
render_split test "${TEST_DATASET}" "${PROTOCOL_ROOT}/test/camera_manifest.json" \
  "${RESULT_ROOT}/evaluation_test"
