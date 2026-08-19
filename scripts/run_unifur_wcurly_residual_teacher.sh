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
CONFIG="${PROJECT_ROOT}/configs/fiber_hairgs_wcurly_cleanhair_residual_teacher_4k.yaml"
OUT_DIR="${CLEAN_RESIDUAL_OUT:-${RESULT_ROOT}/cleanhair_v11_residual_teacher_4k_v1}"
PROTOCOL_ID="hairgs-wcurly-static-train12-test4-v2-camera-fixed"

run_cli() {
  PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    "${CONDA_BIN}" run --no-capture-output -n hair-gs python -m \
    dpd3dgs_animal.cli --config "${CONFIG}" "$@"
}

resolve_clean_ply() {
  [[ -f "${CLEAN_STAGE1_ROOT}/clean_stage1_metadata.json" ]] || {
    echo "Clean Stage-1 is not complete: ${CLEAN_STAGE1_ROOT}" >&2
    return 2
  }
  local point_cloud
  point_cloud="$(find "${CLEAN_STAGE1_ROOT}/point_cloud" -mindepth 2 -maxdepth 2 \
    -type f -name point_cloud.ply -print | sort -V | tail -1)"
  [[ -f "${point_cloud}" ]] || {
    echo "No calibrated clean Stage-1 PLY found" >&2
    return 2
  }
  printf '%s\n' "${point_cloud}"
}

train_teacher() {
  [[ -f "${OUT_DIR}/unified_fiber_field.pt" ]] && return
  local clean_ply
  clean_ply="$(resolve_clean_ply)"
  run_cli fiber-stage2 \
    --stage1-npz "${STAGE1}" \
    --gaussian-ply "${clean_ply}" \
    --fixed-base-gaussian-ply "${FIXED_BASE_PLY}" \
    --frame-dir "${PROTOCOL_ROOT}/train/images" \
    --camera-manifest "${PROTOCOL_ROOT}/train/camera_manifest.json" \
    --out-dir "${OUT_DIR}" --renderer hairgs \
    --render-width 1000 --render-height 1000
}

evaluate_split() {
  local split="$1" clean_ply output ground_truth
  clean_ply="$(resolve_clean_ply)"
  output="${RESULT_ROOT}/$(basename "${OUT_DIR}")_eval_residual_${split}"
  ground_truth="${PROTOCOL_ROOT}/${split}/images"
  [[ -f "${output}/external_evaluation/evaluation.json" ]] && return
  run_cli fiber-eval \
    --stage1-npz "${STAGE1}" \
    --gaussian-ply "${clean_ply}" \
    --fixed-base-gaussian-ply "${FIXED_BASE_PLY}" \
    --checkpoint "${OUT_DIR}/unified_fiber_field.pt" \
    --frame-dir "${ground_truth}" \
    --camera-manifest "${PROTOCOL_ROOT}/${split}/camera_manifest.json" \
    --out-dir "${output}" --renderer hairgs --route-mode residual \
    --render-width 1000 --render-height 1000 --export-external-renders
  "${CONDA_BIN}" run --no-capture-output -n dpd3dgs-animal python \
    "${PROJECT_ROOT}/scripts/evaluate_external_renders.py" \
    --render-manifest "${output}/external_render_manifest.json" \
    --ground-truth-dir "${ground_truth}" \
    --output-dir "${output}/external_evaluation" \
    --method "Residual-only clean-hair teacher" \
    --protocol-id "${PROTOCOL_ID}" --device cuda
}

case "${MODE}" in
  train) train_teacher ;;
  evaluate|all) train_teacher; evaluate_split train; evaluate_split test ;;
  *) echo "Usage: $0 [train|evaluate|all]" >&2; exit 2 ;;
esac
