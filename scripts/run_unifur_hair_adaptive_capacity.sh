#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/aoki/Differentiable_Physical_Driven_3DGS_for_monocular_Animal_Motion_Reconstruction}"
DATA_ROOT="${DATA_ROOT:-/mnt/f/fur_hair_unified_data}"
CONDA_BIN="${CONDA_BIN:-/home/aoki/miniconda3/bin/conda}"
PROTOCOL_ROOT="${DATA_ROOT}/benchmarks/hairgs_wcurly_static_protocol"
RESULT_ROOT="${DATA_ROOT}/benchmarks/hairgs_wcurly_static_results"
GAUSSIAN="${DATA_ROOT}/hair-gs_parsed_official16_1000/cem_yuksel/wCurly/sparse/0/points3D.ply"
STAGE1="${PROTOCOL_ROOT}/static_head_stage1.npz"
TRAIN_IMAGES="${PROTOCOL_ROOT}/train/images"
TRAIN_MANIFEST="${PROTOCOL_ROOT}/train/camera_manifest.json"
TEST_IMAGES="${PROTOCOL_ROOT}/test/images"
TEST_MANIFEST="${PROTOCOL_ROOT}/test/camera_manifest.json"
RESIDUAL_CONFIG="${PROJECT_ROOT}/configs/fiber_hairgs_wcurly_static_residual_adaptive.yaml"
UNIFIED_CONFIG="${PROJECT_ROOT}/configs/fiber_hairgs_wcurly_static_fin_adaptive.yaml"
RESIDUAL_OUT="${RESULT_ROOT}/residual_adaptive43k_8k"
UNIFIED_OUT="${RESULT_ROOT}/unified_fin_carrier_adaptive43k_14k"
PROTOCOL="hairgs-wcurly-static-train12-test4-v2-camera-fixed"

run_cli() {
  PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    "${CONDA_BIN}" run --no-capture-output -n hair-gs python -m \
    dpd3dgs_animal.cli --config "$1" "${@:2}"
}

external_eval() {
  "${CONDA_BIN}" run --no-capture-output -n dpd3dgs-animal python \
    "${PROJECT_ROOT}/scripts/evaluate_external_renders.py" \
    --render-manifest "$1" --ground-truth-dir "${TEST_IMAGES}" \
    --output-dir "$2" --method "$3" --protocol-id "${PROTOCOL}" --device cuda
}

train_residual() {
  if [[ ! -f "${RESIDUAL_OUT}/unified_fiber_field.pt" ]]; then
    run_cli "${RESIDUAL_CONFIG}" fiber-stage2 \
      --stage1-npz "${STAGE1}" --gaussian-ply "${GAUSSIAN}" \
      --frame-dir "${TRAIN_IMAGES}" --camera-manifest "${TRAIN_MANIFEST}" \
      --out-dir "${RESIDUAL_OUT}" --renderer hairgs \
      --render-width 512 --render-height 512
  fi
}

train_unified() {
  [[ -f "${RESIDUAL_OUT}/unified_fiber_field.pt" ]] || train_residual
  if [[ ! -f "${UNIFIED_OUT}/unified_fiber_field.pt" ]]; then
    run_cli "${UNIFIED_CONFIG}" fiber-stage2 \
      --stage1-npz "${STAGE1}" --gaussian-ply "${GAUSSIAN}" \
      --frame-dir "${TRAIN_IMAGES}" --camera-manifest "${TRAIN_MANIFEST}" \
      --out-dir "${UNIFIED_OUT}" --renderer hairgs \
      --residual-bootstrap-checkpoint "${RESIDUAL_OUT}/unified_fiber_field.pt" \
      --render-width 512 --render-height 512
  fi
}

evaluate_checkpoint() {
  local config="$1" checkpoint="$2" output="$3" route="$4" method="$5"
  run_cli "${config}" fiber-eval \
    --stage1-npz "${STAGE1}" --gaussian-ply "${GAUSSIAN}" \
    --checkpoint "${checkpoint}" --frame-dir "${TEST_IMAGES}" \
    --camera-manifest "${TEST_MANIFEST}" --out-dir "${output}" \
    --renderer hairgs --route-mode "${route}" \
    --render-width 512 --render-height 512 --export-external-renders
  external_eval "${output}/external_render_manifest.json" \
    "${output}/external_evaluation" "${method}"
}

evaluate_all() {
  train_unified
  evaluate_checkpoint "${RESIDUAL_CONFIG}" \
    "${RESIDUAL_OUT}/unified_fiber_field.pt" \
    "${RESULT_ROOT}/residual_adaptive43k_8k_eval_test4" residual \
    "Residual-only 3DGS (adaptive 43k)"
  for route in soft hard; do
    evaluate_checkpoint "${UNIFIED_CONFIG}" \
      "${UNIFIED_OUT}/unified_fiber_field.pt" \
      "${RESULT_ROOT}/unified_fin_carrier_adaptive43k_14k_eval_${route}_test4" \
      "${route}" "UniFur adaptive 43k (${route})"
  done
}

render_video() {
  train_unified
  local calibrated="${UNIFIED_OUT}/unified_fiber_field_carrier_calibrated.pt"
  if [[ ! -f "${calibrated}" ]]; then
    PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
      "${CONDA_BIN}" run --no-capture-output -n dpd3dgs-animal python \
      "${PROJECT_ROOT}/scripts/calibrate_simulation_carriers.py" \
      --config "${UNIFIED_CONFIG}" \
      --checkpoint "${UNIFIED_OUT}/unified_fiber_field.pt" \
      --output-checkpoint "${calibrated}"
  fi
  local video_out="${RESULT_ROOT}/unified_fin_carrier_adaptive43k_14k_simulation_video"
  if [[ ! -f "${video_out}/simulation_edit.mp4" ]]; then
    PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
      "${CONDA_BIN}" run --no-capture-output -n hair-gs python \
      "${PROJECT_ROOT}/scripts/render_simulation_asset_video.py" \
      --config "${UNIFIED_CONFIG}" --stage1-npz "${STAGE1}" \
      --gaussian-ply "${GAUSSIAN}" --checkpoint "${calibrated}" \
      --frame-dir "${TEST_IMAGES}" --camera-manifest "${TEST_MANIFEST}" \
      --out-dir "${video_out}" --render-width 512 --render-height 512 \
      --frames 96 --fps 24 --wind-scale 0.05 --length-amplitude 0.20 \
      --hard-carriers
  fi
}

case "${MODE}" in
  residual) train_residual ;;
  unified) train_unified ;;
  evaluate) evaluate_all ;;
  video) render_video ;;
  all) evaluate_all; render_video ;;
  *) echo "Usage: $0 [residual|unified|evaluate|video|all]" >&2; exit 2 ;;
esac
