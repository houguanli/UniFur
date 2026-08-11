#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/aoki/Differentiable_Physical_Driven_3DGS_for_monocular_Animal_Motion_Reconstruction}"
GART_ROOT="${GART_ROOT:-/home/aoki/fur_hair_baselines/GART}"
PYTHON="${GART_PYTHON:-/home/aoki/miniconda3/envs/gart-repro/bin/python}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-/mnt/f/fur_hair_unified_data/benchmarks/dfa_panda_walk_dual}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/f/fur_hair_unified_data/baselines/gart/dfa-panda-walk-mono}"
RENDER_DIR="${RENDER_DIR:-${OUTPUT_DIR}/heldout_v8_t32_render}"
EVAL_DIR="${EVAL_DIR:-${OUTPUT_DIR}/heldout_v8_t32_evaluation}"
PROFILE="${PROFILE:-${PROJECT_ROOT}/configs/gart_dfa_panda_mono.yaml}"
ADAPTER="${PROJECT_ROOT}/scripts/gart_dfa_adapter.py"
EVALUATOR="${PROJECT_ROOT}/scripts/evaluate_external_renders.py"
GT_DIR="${PROTOCOL_ROOT}/test_novel_v8_t32/images"

mkdir -p "${OUTPUT_DIR}"

validate() {
  "${PYTHON}" "${ADAPTER}" \
    --mode validate \
    --gart-root "${GART_ROOT}" \
    --profile "${PROFILE}" \
    --protocol-root "${PROTOCOL_ROOT}" \
    --output-dir "${OUTPUT_DIR}"
}

train() {
  "${PYTHON}" "${ADAPTER}" \
    --mode train \
    --gart-root "${GART_ROOT}" \
    --profile "${PROFILE}" \
    --protocol-root "${PROTOCOL_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    2>&1 | tee "${OUTPUT_DIR}/train.log"
}

render() {
  "${PYTHON}" "${ADAPTER}" \
    --mode render \
    --gart-root "${GART_ROOT}" \
    --profile "${PROFILE}" \
    --protocol-root "${PROTOCOL_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --render-dir "${RENDER_DIR}" \
    2>&1 | tee "${OUTPUT_DIR}/render.log"
}

evaluate() {
  PYTHONPATH="${PROJECT_ROOT}/src" \
  /home/aoki/miniconda3/envs/hair-gs/bin/python "${EVALUATOR}" \
    --render-manifest "${RENDER_DIR}/render_manifest.json" \
    --ground-truth-dir "${GT_DIR}" \
    --output-dir "${EVAL_DIR}" \
    --method "GART-DFA adapter" \
    --protocol-id "DFA-Panda-Walk-32f-v1" \
    --device cuda
}

case "${MODE}" in
  validate) validate ;;
  smoke-template)
    "${PYTHON}" "${ADAPTER}" \
      --mode smoke-template \
      --gart-root "${GART_ROOT}" \
      --profile "${PROFILE}" \
      --protocol-root "${PROTOCOL_ROOT}" \
      --output-dir "${OUTPUT_DIR}"
    ;;
  train) validate; train ;;
  render) render ;;
  evaluate) evaluate ;;
  render_and_evaluate) render; evaluate ;;
  all) validate; train; render; evaluate ;;
  *) echo "Usage: $0 {validate|smoke-template|train|render|evaluate|render_and_evaluate|all}" >&2; exit 2 ;;
esac
