#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/aoki/Differentiable_Physical_Driven_3DGS_for_monocular_Animal_Motion_Reconstruction}"
DATA_ROOT="${DATA_ROOT:-/mnt/f/fur_hair_unified_data}"
CONDA_BIN="${CONDA_BIN:-/home/aoki/miniconda3/bin/conda}"
ROOT="${DATA_ROOT}/benchmarks/neuralfur_panda_shared"
CONFIG="${UNIFIED_CONFIG:-${PROJECT_ROOT}/configs/fiber_panda_multiview_intrinsic_v17.yaml}"
STAGE1="${ROOT}/static_stage1.npz"
GAUSSIAN="${ROOT}/initial_body_gaussians.ply"
RESIDUAL="${ROOT}/full_residual_balanced_v28_20k_r480/unified_fiber_field.pt"
OUT="${UNIFIED_OUT:-${ROOT}/full_unified_intrinsic_v17_12k}"
PROTOCOL_ID="F-mv-official-prior-28fit-8test-r480-v2"

run_cli() {
  PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    "${CONDA_BIN}" run --no-capture-output -n hair-gs python -m \
    dpd3dgs_animal.cli --config "${CONFIG}" "$@"
}

validate_inputs() {
  local path
  for path in "${STAGE1}" "${GAUSSIAN}" "${RESIDUAL}" "${CONFIG}"; do
    [[ -f "${path}" ]] || { echo "Missing required input: ${path}" >&2; return 2; }
  done
}

train() {
  validate_inputs
  [[ -f "${OUT}/unified_fiber_field.pt" ]] && return
  run_cli fiber-stage2 \
    --stage1-npz "${STAGE1}" --gaussian-ply "${GAUSSIAN}" \
    --frame-dir "${ROOT}/train_v28/images" \
    --camera-manifest "${ROOT}/train_v28/camera_manifest.json" \
    --out-dir "${OUT}" --renderer hairgs \
    --residual-bootstrap-checkpoint "${RESIDUAL}" \
    --render-width 480 --render-height 270
}

evaluate_split() {
  local route="$1" split="$2" count="$3"
  local images="${ROOT}/${split}_v${count}/images"
  local manifest="${ROOT}/${split}_v${count}/camera_manifest.json"
  local experiment="$(basename "${OUT}")"
  local output="${ROOT}/eval_${experiment}_${route}_${split}_v${count}"
  [[ -f "${output}/external_evaluation/evaluation.json" ]] && return
  run_cli fiber-eval \
    --stage1-npz "${STAGE1}" --gaussian-ply "${GAUSSIAN}" \
    --checkpoint "${OUT}/unified_fiber_field.pt" \
    --frame-dir "${images}" --camera-manifest "${manifest}" \
    --out-dir "${output}" --renderer hairgs --route-mode "${route}" \
    --render-width 480 --render-height 270 --export-external-renders
  "${CONDA_BIN}" run --no-capture-output -n dpd3dgs-animal python \
    "${PROJECT_ROOT}/scripts/evaluate_external_renders.py" \
    --render-manifest "${output}/external_render_manifest.json" \
    --ground-truth-dir "${images}" \
    --output-dir "${output}/external_evaluation" \
    --method "UniFur v17 intrinsic surface propagation (${route})" \
    --protocol-id "${PROTOCOL_ID}" --device cuda
}

evaluate() {
  train
  local route
  for route in hard soft; do evaluate_split "${route}" train 28; done
  for route in hard soft; do evaluate_split "${route}" test 8; done
}

case "${MODE}" in
  train) train ;;
  evaluate|all) evaluate ;;
  *) echo "Usage: $0 [train|evaluate|all]" >&2; exit 2 ;;
esac
