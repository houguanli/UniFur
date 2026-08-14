#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/aoki/Differentiable_Physical_Driven_3DGS_for_monocular_Animal_Motion_Reconstruction}"
CONDA_BIN="${CONDA_BIN:-/home/aoki/miniconda3/bin/conda}"
DATA_ROOT="${DATA_ROOT:-/mnt/f/fur_hair_unified_data/benchmarks/gaussian_haircut_person0_protocol}"
GHC_RESULT_ROOT="${GHC_RESULT_ROOT:-/mnt/f/fur_hair_unified_data/benchmarks/gaussian_haircut_person0_results/fixedcam_25k_densify10k}"
RESULT_ROOT="${RESULT_ROOT:-/mnt/f/fur_hair_unified_data/benchmarks/gaussian_haircut_person0_results/unifur_aligned_scalp_teacher_v2}"
GAUSSIAN_STEPS="${GAUSSIAN_STEPS:-25000}"
GAUSSIAN_PLY="${GAUSSIAN_PLY:-${GHC_RESULT_ROOT}/stage1/point_cloud/iteration_${GAUSSIAN_STEPS}/raw_point_cloud.ply}"
STAGE1="${DATA_ROOT}/static_head_stage1.npz"
TRAIN_IMAGES="${DATA_ROOT}/protocol/train/images"
TRAIN_MANIFEST="${DATA_ROOT}/protocol/train/camera_manifest.json"
TEST_IMAGES="${DATA_ROOT}/protocol/test/images"
TEST_MANIFEST="${DATA_ROOT}/protocol/test/camera_manifest.json"
PROTOCOL="gaussian-haircut-person0-odd33fit-even33test-v1"

run_cli() {
  PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    "${CONDA_BIN}" run --no-capture-output -n hair-gs python -m \
    dpd3dgs_animal.cli --config "$1" "${@:2}"
}

evaluate_checkpoint() {
  local config="$1" checkpoint="$2" output="$3" route="$4" method="$5"
  run_cli "${config}" fiber-eval \
    --stage1-npz "${STAGE1}" --gaussian-ply "${GAUSSIAN_PLY}" \
    --checkpoint "${checkpoint}" --frame-dir "${TEST_IMAGES}" \
    --camera-manifest "${TEST_MANIFEST}" --out-dir "${output}" \
    --renderer hairgs --route-mode "${route}" \
    --render-width 1024 --render-height 1024 --export-external-renders
  "${CONDA_BIN}" run --no-capture-output -n dpd3dgs-animal python \
    "${PROJECT_ROOT}/scripts/evaluate_external_renders.py" \
    --render-manifest "${output}/external_render_manifest.json" \
    --ground-truth-dir "${TEST_IMAGES}" \
    --output-dir "${output}/external_evaluation" \
    --method "${method}" --protocol-id "${PROTOCOL}" --device cuda
}

train_residual() {
  local config="${PROJECT_ROOT}/configs/fiber_gaussian_haircut_person0_residual.yaml"
  local out="${RESULT_ROOT}/residual_12k"
  [[ -f "${GAUSSIAN_PLY}" ]] || { echo "Missing GaussianHaircut scaffold: ${GAUSSIAN_PLY}" >&2; exit 2; }
  if [[ ! -f "${out}/unified_fiber_field.pt" ]]; then
    run_cli "${config}" fiber-stage2 \
      --stage1-npz "${STAGE1}" --gaussian-ply "${GAUSSIAN_PLY}" \
      --frame-dir "${TRAIN_IMAGES}" --camera-manifest "${TRAIN_MANIFEST}" \
      --out-dir "${out}" --renderer hairgs \
      --render-width 1024 --render-height 1024
  fi
  evaluate_checkpoint "${config}" "${out}/unified_fiber_field.pt" \
    "${RESULT_ROOT}/residual_12k_eval" residual "Residual-only 3DGS"
}

train_unified() {
  local config="${PROJECT_ROOT}/configs/fiber_gaussian_haircut_person0_unified.yaml"
  local residual="${RESULT_ROOT}/residual_12k/unified_fiber_field.pt"
  local out="${RESULT_ROOT}/unified_teacher_hull_16k"
  [[ -f "${residual}" ]] || { echo "Missing person0 residual bootstrap: ${residual}" >&2; exit 2; }
  if [[ ! -f "${out}/unified_fiber_field.pt" ]]; then
    run_cli "${config}" fiber-stage2 \
      --stage1-npz "${STAGE1}" --gaussian-ply "${GAUSSIAN_PLY}" \
      --frame-dir "${TRAIN_IMAGES}" --camera-manifest "${TRAIN_MANIFEST}" \
      --out-dir "${out}" --renderer hairgs \
      --residual-bootstrap-checkpoint "${residual}" \
      --render-width 1024 --render-height 1024
  fi
  evaluate_checkpoint "${config}" "${out}/unified_fiber_field.pt" \
    "${RESULT_ROOT}/unified_teacher_hull_16k_eval_soft" soft "UniFur teacher+hull (soft)"
  evaluate_checkpoint "${config}" "${out}/unified_fiber_field.pt" \
    "${RESULT_ROOT}/unified_teacher_hull_16k_eval_hard" hard "UniFur teacher+hull (hard)"
  run_cli "${config}" fiber-route-audit \
    --stage1-npz "${STAGE1}" --gaussian-ply "${GAUSSIAN_PLY}" \
    --checkpoint "${out}/unified_fiber_field.pt" \
    --frame-dir "${TEST_IMAGES}" --camera-manifest "${TEST_MANIFEST}" \
    --out-dir "${RESULT_ROOT}/unified_teacher_hull_16k_route_audit" \
    --renderer hairgs --render-width 1024 --render-height 1024
}

case "${MODE}" in
  residual) train_residual ;;
  unified) train_unified ;;
  all) train_residual; train_unified ;;
  *) echo "Usage: $0 {residual|unified|all}" >&2; exit 2 ;;
esac
