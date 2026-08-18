#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/aoki/Differentiable_Physical_Driven_3DGS_for_monocular_Animal_Motion_Reconstruction}"
DATA_ROOT="${DATA_ROOT:-/mnt/f/fur_hair_unified_data}"
HAIRGS_ROOT="${HAIRGS_ROOT:-/home/aoki/fur_hair_baselines/hair-gs}"
CONDA_BIN="${CONDA_BIN:-/home/aoki/miniconda3/bin/conda}"
PROTOCOL_ROOT="${DATA_ROOT}/benchmarks/hairgs_wcurly_static_protocol"
RESULT_ROOT="${DATA_ROOT}/benchmarks/hairgs_wcurly_static_results"
SOURCE="${DATA_ROOT}/hair-gs_parsed_official16_1000/cem_yuksel/wCurly"
STAGE1="${PROTOCOL_ROOT}/static_head_stage1.npz"
GAUSSIAN="${SOURCE}/sparse/0/points3D.ply"
TRAIN_IMAGES="${PROTOCOL_ROOT}/train/images"
TRAIN_MANIFEST="${PROTOCOL_ROOT}/train/camera_manifest.json"
TEST_IMAGES="${PROTOCOL_ROOT}/test/images"
TEST_MANIFEST="${PROTOCOL_ROOT}/test/camera_manifest.json"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/fiber_hairgs_wcurly_static_fieldgrow.yaml}"
RESIDUAL="${RESULT_ROOT}/residual_adaptive43k_8k/unified_fiber_field.pt"
RUN_NAME="${RUN_NAME:-unified_fin_fieldgrow_adaptive43k_16k}"
PREFLIGHT_OUT="${RESULT_ROOT}/${RUN_NAME}_preflight600"
UNIFIED_OUT="${RESULT_ROOT}/${RUN_NAME}"
GEOMETRY_OUT="${RESULT_ROOT}/wcurly_geometry_${RUN_NAME}"
PROTOCOL="hairgs-wcurly-static-train12-test4-v2-camera-fixed"

run_cli() {
  PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    "${CONDA_BIN}" run --no-capture-output -n hair-gs python -m \
    dpd3dgs_animal.cli --config "${CONFIG}" "$@"
}

train_to() {
  local output="$1" steps="${2:-}"
  [[ -f "${output}/unified_fiber_field.pt" ]] && return
  local step_args=()
  [[ -n "${steps}" ]] && step_args=(--steps "${steps}")
  run_cli fiber-stage2 \
    --stage1-npz "${STAGE1}" --gaussian-ply "${GAUSSIAN}" \
    --frame-dir "${TRAIN_IMAGES}" --camera-manifest "${TRAIN_MANIFEST}" \
    --out-dir "${output}" --renderer hairgs \
    --residual-bootstrap-checkpoint "${RESIDUAL}" \
    --render-width 512 --render-height 512 "${step_args[@]}"
}

select_structural_checkpoint() {
  "${CONDA_BIN}" run --no-capture-output -n dpd3dgs-animal python \
    "${PROJECT_ROOT}/scripts/select_structural_checkpoint.py" \
      --metrics "${UNIFIED_OUT}/training_metrics.jsonl" \
      --checkpoint-dir "${UNIFIED_OUT}/checkpoints" \
      --output "${UNIFIED_OUT}/structural_checkpoint_selection.json" \
      --require-risk-calibration
}

evaluate_route() {
  local route="$1"
  local output="${RESULT_ROOT}/${RUN_NAME}_eval_${route}_test4"
  run_cli fiber-eval \
    --stage1-npz "${STAGE1}" --gaussian-ply "${GAUSSIAN}" \
    --checkpoint "${UNIFIED_OUT}/unified_fiber_field.pt" \
    --frame-dir "${TEST_IMAGES}" --camera-manifest "${TEST_MANIFEST}" \
    --out-dir "${output}" --renderer hairgs --route-mode "${route}" \
    --render-width 512 --render-height 512 --export-external-renders
  "${CONDA_BIN}" run --no-capture-output -n dpd3dgs-animal python \
    "${PROJECT_ROOT}/scripts/evaluate_external_renders.py" \
    --render-manifest "${output}/external_render_manifest.json" \
    --ground-truth-dir "${TEST_IMAGES}" \
    --output-dir "${output}/external_evaluation" \
    --method "${RUN_NAME} (${route})" \
    --protocol-id "${PROTOCOL}" --device cuda
}

export_and_evaluate_geometry() {
  mkdir -p "${GEOMETRY_OUT}"
  for mode in structured_deployed strand_deployed strand_target; do
    local stem="fieldgrow_${mode}"
    PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
      "${CONDA_BIN}" run --no-capture-output -n dpd3dgs-animal python \
      "${PROJECT_ROOT}/scripts/export_unifur_hair_eval.py" \
      --stage1-npz "${STAGE1}" --gaussian-ply "${GAUSSIAN}" \
      --checkpoint "${UNIFIED_OUT}/unified_fiber_field.pt" \
      --mode "${mode}" --spacing-mm 3 \
      --out-npz "${GEOMETRY_OUT}/${stem}.npz" \
      --out-report "${GEOMETRY_OUT}/${stem}_export.json"
    (
      cd "${HAIRGS_ROOT}"
      "${CONDA_BIN}" run --no-capture-output -n hair-gs python \
        "${PROJECT_ROOT}/scripts/hairgs_eval_fixed.py" \
        -s "${SOURCE}" -p "${GEOMETRY_OUT}/${stem}.npz" \
        -pt hair_eval_npz --out_json "${GEOMETRY_OUT}/${stem}_metrics.json"
    )
  done
}

case "${MODE}" in
  preflight) train_to "${PREFLIGHT_OUT}" 600 ;;
  train) train_to "${UNIFIED_OUT}" ;;
  select) train_to "${UNIFIED_OUT}"; select_structural_checkpoint ;;
  evaluate) train_to "${UNIFIED_OUT}"; evaluate_route soft; evaluate_route hard ;;
  geometry) train_to "${UNIFIED_OUT}"; export_and_evaluate_geometry ;;
  all)
    train_to "${UNIFIED_OUT}"
    select_structural_checkpoint
    evaluate_route soft
    evaluate_route hard
    export_and_evaluate_geometry
    ;;
  *) echo "Usage: $0 [preflight|train|select|evaluate|geometry|all]" >&2; exit 2 ;;
esac
