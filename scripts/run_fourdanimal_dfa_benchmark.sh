#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/aoki/Differentiable_Physical_Driven_3DGS_for_monocular_Animal_Motion_Reconstruction}"
REPO_ROOT="${FOURDANIMAL_ROOT:-/home/aoki/fur_hair_baselines/4D-Animal}"
PYTHON="${FOURDANIMAL_PYTHON:-/home/aoki/miniconda3/envs/animal4d-repro/bin/python}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-/mnt/f/fur_hair_unified_data/benchmarks/dfa_panda_walk_dual}"
RESOURCE_ROOT="${RESOURCE_ROOT:-/mnt/f/fur_hair_unified_data/4d_animal_official}"
ARCHIVE="${RESOURCE_ROOT}/external_data.tar.gz"
EXTRACT_ROOT="${RESOURCE_ROOT}/extracted"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/f/fur_hair_unified_data/baselines/4d-animal/dfa-panda-walk-mono}"
RENDER_DIR="${RENDER_DIR:-${OUTPUT_DIR}/heldout_v8_t32_render}"
EVAL_DIR="${EVAL_DIR:-${OUTPUT_DIR}/heldout_v8_t32_evaluation}"
ADAPTER="${PROJECT_ROOT}/scripts/fourdanimal_dfa_adapter.py"
EVALUATOR="${PROJECT_ROOT}/scripts/evaluate_external_renders.py"
GT_DIR="${PROTOCOL_ROOT}/test_novel_v8_t32/images"

prepare() {
  test -s "${ARCHIVE}" || { echo "Missing official resource archive: ${ARCHIVE}" >&2; exit 1; }
  if [ ! -f "${EXTRACT_ROOT}/.complete" ]; then
    gzip -t "${ARCHIVE}" || {
      echo "Official resource archive is incomplete or corrupt: ${ARCHIVE}" >&2
      exit 1
    }
    mkdir -p "${EXTRACT_ROOT}"
    tar -xzf "${ARCHIVE}" -C "${EXTRACT_ROOT}"
    touch "${EXTRACT_ROOT}/.complete"
  fi
  local asset_root="${EXTRACT_ROOT}"
  if [ -d "${EXTRACT_ROOT}/external_data" ]; then
    asset_root="${EXTRACT_ROOT}/external_data"
  fi
  if [ -e "${REPO_ROOT}/external_data" ] && [ ! -L "${REPO_ROOT}/external_data" ]; then
    echo "Refusing to replace non-symlink ${REPO_ROOT}/external_data" >&2
    exit 1
  fi
  ln -sfn "${asset_root}" "${REPO_ROOT}/external_data"
}

adapter() {
  local adapter_mode="$1"
  "${PYTHON}" "${ADAPTER}" \
    --mode "${adapter_mode}" \
    --repo-root "${REPO_ROOT}" \
    --protocol-root "${PROTOCOL_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --render-dir "${RENDER_DIR}" \
    --device cuda
}

evaluate() {
  PYTHONPATH="${PROJECT_ROOT}/src" \
  /home/aoki/miniconda3/envs/hair-gs/bin/python "${EVALUATOR}" \
    --render-manifest "${RENDER_DIR}/render_manifest.json" \
    --ground-truth-dir "${GT_DIR}" \
    --output-dir "${EVAL_DIR}" \
    --method "4D-Animal-DFA adapter" \
    --protocol-id "DFA-Panda-Walk-32f-v1" \
    --device cuda
}

mkdir -p "${OUTPUT_DIR}"
case "${MODE}" in
  prepare) prepare ;;
  validate) adapter validate ;;
  precompute) prepare; adapter precompute 2>&1 | tee "${OUTPUT_DIR}/precompute.log" ;;
  train) prepare; adapter train 2>&1 | tee "${OUTPUT_DIR}/train.log" ;;
  render) prepare; adapter render 2>&1 | tee "${OUTPUT_DIR}/render.log" ;;
  evaluate) evaluate ;;
  render_and_evaluate) prepare; adapter render 2>&1 | tee "${OUTPUT_DIR}/render.log"; evaluate ;;
  all) prepare; adapter validate; adapter precompute 2>&1 | tee "${OUTPUT_DIR}/precompute.log"; adapter train 2>&1 | tee "${OUTPUT_DIR}/train.log"; adapter render 2>&1 | tee "${OUTPUT_DIR}/render.log"; evaluate ;;
  *) echo "Usage: $0 {prepare|validate|precompute|train|render|evaluate|render_and_evaluate|all}" >&2; exit 2 ;;
esac
