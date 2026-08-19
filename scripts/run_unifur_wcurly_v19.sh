#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-train}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/aoki/Differentiable_Physical_Driven_3DGS_for_monocular_Animal_Motion_Reconstruction}"
DATA_ROOT="${DATA_ROOT:-/mnt/f/fur_hair_unified_data}"
CONDA_BIN="${CONDA_BIN:-/home/aoki/miniconda3/bin/conda}"
PROTOCOL_ROOT="${DATA_ROOT}/benchmarks/hairgs_wcurly_static_protocol"
RESULT_ROOT="${DATA_ROOT}/benchmarks/hairgs_wcurly_static_results"
CLEAN_STAGE1_ROOT="${CLEAN_STAGE1_ROOT:-${RESULT_ROOT}/clean_hair_stage1_visualhull80k_mvcal_v11b_adaptivegamut_projection}"
CLEAN_PLY="${CLEAN_PLY:-${CLEAN_STAGE1_ROOT}/point_cloud/iteration_1/point_cloud.ply}"
FIXED_BASE_PLY="${FIXED_BASE_PLY:-${RESULT_ROOT}/hairgs_official_train12_30k30k/point_cloud/iteration_30000/point_cloud.ply}"
STAGE1="${PROTOCOL_ROOT}/static_head_stage1.npz"
CONFIG="${UNIFIED_CONFIG:-${PROJECT_ROOT}/configs/fiber_hairgs_wcurly_scalpatlas_signed3d_v19_8k.yaml}"
RESIDUAL_CHECKPOINT="${RESIDUAL_CHECKPOINT:-${RESULT_ROOT}/cleanhair_v11_residual_teacher_4k_v1/unified_fiber_field.pt}"
OUT="${UNIFIED_OUT:-${RESULT_ROOT}/cleanhair_v19_scalpatlas_signed3d_birth_8k_v2}"
PROTOCOL_ID="hairgs-wcurly-static-train12-test4-v2-camera-fixed"

run_cli() {
  PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    "${CONDA_BIN}" run --no-capture-output -n hair-gs python -m \
    dpd3dgs_animal.cli --config "${CONFIG}" "$@"
}

validate_inputs() {
  local path
  for path in "${CLEAN_PLY}" "${FIXED_BASE_PLY}" "${STAGE1}" \
    "${RESIDUAL_CHECKPOINT}" "${CONFIG}"; do
    [[ -f "${path}" ]] || {
      echo "Missing required input: ${path}" >&2
      return 2
    }
  done
}

train() {
  validate_inputs
  [[ -f "${OUT}/unified_fiber_field.pt" ]] && return
  run_cli fiber-stage2 \
    --stage1-npz "${STAGE1}" \
    --gaussian-ply "${CLEAN_PLY}" \
    --fixed-base-gaussian-ply "${FIXED_BASE_PLY}" \
    --frame-dir "${PROTOCOL_ROOT}/train/images" \
    --camera-manifest "${PROTOCOL_ROOT}/train/camera_manifest.json" \
    --out-dir "${OUT}" --renderer hairgs \
    --residual-bootstrap-checkpoint "${RESIDUAL_CHECKPOINT}" \
    --render-width 1000 --render-height 1000
}

evaluate_split() {
  local route="$1" split="$2"
  local output="${RESULT_ROOT}/$(basename "${OUT}")_eval_${route}_${split}"
  local ground_truth="${PROTOCOL_ROOT}/${split}/images"
  [[ -f "${output}/external_evaluation/evaluation.json" ]] && return
  run_cli fiber-eval \
    --stage1-npz "${STAGE1}" \
    --gaussian-ply "${CLEAN_PLY}" \
    --fixed-base-gaussian-ply "${FIXED_BASE_PLY}" \
    --checkpoint "${OUT}/unified_fiber_field.pt" \
    --frame-dir "${ground_truth}" \
    --camera-manifest "${PROTOCOL_ROOT}/${split}/camera_manifest.json" \
    --out-dir "${output}" --renderer hairgs --route-mode "${route}" \
    --render-width 1000 --render-height 1000 --export-external-renders
  "${CONDA_BIN}" run --no-capture-output -n dpd3dgs-animal python \
    "${PROJECT_ROOT}/scripts/evaluate_external_renders.py" \
    --render-manifest "${output}/external_render_manifest.json" \
    --ground-truth-dir "${ground_truth}" \
    --output-dir "${output}/external_evaluation" \
    --method "UniFur v19 scalp-atlas signed-field 3D-deficit (${route})" \
    --protocol-id "${PROTOCOL_ID}" --device cuda
}

evaluate() {
  train
  local split route
  for split in train test; do
    for route in hard soft; do
      evaluate_split "${route}" "${split}"
    done
  done
}

case "${MODE}" in
  train) train ;;
  evaluate|all) evaluate ;;
  *) echo "Usage: $0 [train|evaluate|all]" >&2; exit 2 ;;
esac
