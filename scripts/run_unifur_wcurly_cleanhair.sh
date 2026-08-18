#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/aoki/Differentiable_Physical_Driven_3DGS_for_monocular_Animal_Motion_Reconstruction}"
DATA_ROOT="${DATA_ROOT:-/mnt/f/fur_hair_unified_data}"
CONDA_BIN="${CONDA_BIN:-/home/aoki/miniconda3/bin/conda}"
PROTOCOL_ROOT="${DATA_ROOT}/benchmarks/hairgs_wcurly_static_protocol"
RESULT_ROOT="${DATA_ROOT}/benchmarks/hairgs_wcurly_static_results"
CLEAN_STAGE1_ROOT="${CLEAN_STAGE1_ROOT:-${RESULT_ROOT}/clean_hair_stage1_visualhull80k_mvcal_v11b_adaptivegamut_projection}"
FIXED_BASE_PLY="${FIXED_BASE_PLY:-${RESULT_ROOT}/hairgs_official_train12_30k30k/point_cloud/iteration_30000/point_cloud.ply}"
STAGE1="${PROTOCOL_ROOT}/static_head_stage1.npz"
RESIDUAL_CONFIG="${PROJECT_ROOT}/configs/fiber_hairgs_wcurly_cleanhair_residual_teacher_4k.yaml"
UNIFIED_CONFIG="${PROJECT_ROOT}/configs/fiber_hairgs_wcurly_cleanhair_unified_6k.yaml"
RESIDUAL_OUT="${CLEAN_RESIDUAL_OUT:-${RESULT_ROOT}/cleanhair_v11_residual_teacher_4k_v1}"
UNIFIED_OUT="${CLEAN_UNIFIED_OUT:-${RESULT_ROOT}/cleanhair_v11_unified_6k_v1}"
PROTOCOL_ID="hairgs-wcurly-static-train12-test4-v2-camera-fixed"

run_cli() {
  local config="$1"
  shift
  PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    "${CONDA_BIN}" run --no-capture-output -n hair-gs python -m \
    dpd3dgs_animal.cli --config "${config}" "$@"
}

resolve_clean_ply() {
  [[ -f "${CLEAN_STAGE1_ROOT}/clean_stage1_metadata.json" ]] || {
    echo "Clean Stage-1 is not complete: ${CLEAN_STAGE1_ROOT}" >&2
    return 2
  }
  local point_cloud
  point_cloud="$({
    find "${CLEAN_STAGE1_ROOT}/point_cloud" -mindepth 2 -maxdepth 2 \
      -type f -name point_cloud.ply -print
  } | sort -V | tail -1)"
  [[ -f "${point_cloud}" ]] || {
    echo "No calibrated clean Stage-1 PLY found" >&2
    return 2
  }
  printf '%s\n' "${point_cloud}"
}

train_residual() {
  [[ -f "${RESIDUAL_OUT}/unified_fiber_field.pt" ]] && return
  local clean_ply
  clean_ply="$(resolve_clean_ply)"
  run_cli "${RESIDUAL_CONFIG}" fiber-stage2 \
    --stage1-npz "${STAGE1}" \
    --gaussian-ply "${clean_ply}" \
    --fixed-base-gaussian-ply "${FIXED_BASE_PLY}" \
    --frame-dir "${PROTOCOL_ROOT}/train/images" \
    --camera-manifest "${PROTOCOL_ROOT}/train/camera_manifest.json" \
    --out-dir "${RESIDUAL_OUT}" --renderer hairgs \
    --render-width 1000 --render-height 1000
}

train_unified() {
  train_residual
  [[ -f "${UNIFIED_OUT}/unified_fiber_field.pt" ]] && return
  local clean_ply
  clean_ply="$(resolve_clean_ply)"
  run_cli "${UNIFIED_CONFIG}" fiber-stage2 \
    --stage1-npz "${STAGE1}" \
    --gaussian-ply "${clean_ply}" \
    --fixed-base-gaussian-ply "${FIXED_BASE_PLY}" \
    --frame-dir "${PROTOCOL_ROOT}/train/images" \
    --camera-manifest "${PROTOCOL_ROOT}/train/camera_manifest.json" \
    --out-dir "${UNIFIED_OUT}" --renderer hairgs \
    --residual-bootstrap-checkpoint "${RESIDUAL_OUT}/unified_fiber_field.pt" \
    --render-width 1000 --render-height 1000
}

evaluate_one() {
  local config="$1" checkpoint="$2" run_name="$3" route="$4"
  local split="$5" method="$6"
  local clean_ply output ground_truth
  clean_ply="$(resolve_clean_ply)"
  output="${RESULT_ROOT}/${run_name}_eval_${route}_${split}"
  ground_truth="${PROTOCOL_ROOT}/${split}/images"
  if [[ ! -f "${output}/external_evaluation/evaluation.json" ]]; then
    run_cli "${config}" fiber-eval \
      --stage1-npz "${STAGE1}" \
      --gaussian-ply "${clean_ply}" \
      --fixed-base-gaussian-ply "${FIXED_BASE_PLY}" \
      --checkpoint "${checkpoint}" \
      --frame-dir "${ground_truth}" \
      --camera-manifest "${PROTOCOL_ROOT}/${split}/camera_manifest.json" \
      --out-dir "${output}" --renderer hairgs --route-mode "${route}" \
      --render-width 1000 --render-height 1000 --export-external-renders
    "${CONDA_BIN}" run --no-capture-output -n dpd3dgs-animal python \
      "${PROJECT_ROOT}/scripts/evaluate_external_renders.py" \
      --render-manifest "${output}/external_render_manifest.json" \
      --ground-truth-dir "${ground_truth}" \
      --output-dir "${output}/external_evaluation" \
      --method "${method}" --protocol-id "${PROTOCOL_ID}" --device cuda
  fi
}

evaluate_all() {
  train_unified
  local residual_run_name unified_run_name
  residual_run_name="$(basename "${RESIDUAL_OUT}")"
  unified_run_name="$(basename "${UNIFIED_OUT}")"
  for split in train test; do
    evaluate_one "${RESIDUAL_CONFIG}" \
      "${RESIDUAL_OUT}/unified_fiber_field.pt" \
      "${residual_run_name}" residual "${split}" \
      "Residual-only, head-aware clean hair Stage-1 v11 adaptive gamut"
    for route in soft hard; do
      evaluate_one "${UNIFIED_CONFIG}" \
        "${UNIFIED_OUT}/unified_fiber_field.pt" \
        "${unified_run_name}" "${route}" "${split}" \
        "UniFur head-aware clean hair Stage-1 v11 adaptive gamut (${route})"
    done
  done
}

evaluate_residual() {
  train_residual
  local residual_run_name split
  residual_run_name="$(basename "${RESIDUAL_OUT}")"
  for split in train test; do
    evaluate_one "${RESIDUAL_CONFIG}" \
      "${RESIDUAL_OUT}/unified_fiber_field.pt" \
      "${residual_run_name}" residual "${split}" \
      "Residual-only, head-aware clean hair Stage-1 v11 adaptive gamut"
  done
}

case "${MODE}" in
  residual) train_residual ;;
  residual-evaluate) evaluate_residual ;;
  unified) train_unified ;;
  evaluate|all) evaluate_all ;;
  *) echo "Usage: $0 [residual|residual-evaluate|unified|evaluate|all]" >&2; exit 2 ;;
esac
