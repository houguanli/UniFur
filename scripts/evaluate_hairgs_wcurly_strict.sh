#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/aoki/Differentiable_Physical_Driven_3DGS_for_monocular_Animal_Motion_Reconstruction}"
HAIRGS_ROOT="${HAIRGS_ROOT:-/home/aoki/fur_hair_baselines/hair-gs}"
CONDA_BIN="${CONDA_BIN:-/home/aoki/miniconda3/bin/conda}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-/mnt/f/fur_hair_unified_data/benchmarks/hairgs_wcurly_static_protocol}"
MODEL_DIR="${MODEL_DIR:-/mnt/f/fur_hair_unified_data/benchmarks/hairgs_wcurly_static_results/hairgs_official_train12_30k30k}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/f/fur_hair_unified_data/benchmarks/hairgs_wcurly_static_results/hairgs_official_train12_30k30k_strict_test4}"
CAMERA_DATASET="${PROTOCOL_ROOT}/hairgs_test4_dataset"
GROUND_TRUTH="${PROTOCOL_ROOT}/test/images"
PROTOCOL_ID="hairgs-wcurly-static-train12-test4-v2-camera-fixed"
COMPAT_ROOT="${HAIRGS_COMPAT_ROOT:-${PROJECT_ROOT}/compat/hairgs_sitecustomize}"

[[ -d "${MODEL_DIR}/point_cloud" ]] || {
  echo "Missing trained Hair-GS model: ${MODEL_DIR}" >&2
  exit 2
}
[[ -f "${CAMERA_DATASET}/sparse/0/images.bin" ]] || {
  echo "Missing held-out Hair-GS camera dataset: ${CAMERA_DATASET}" >&2
  exit 2
}

iteration="$({
  find "${MODEL_DIR}/point_cloud" -mindepth 2 -maxdepth 2 -type f \
    -name point_cloud.ply -printf '%h\n'
} | awk -F'iteration_' '{print $2}' | sort -n | tail -1)"
[[ -n "${iteration}" ]] || { echo "Could not resolve Hair-GS iteration" >&2; exit 2; }

cd "${HAIRGS_ROOT}"
export PYTHONPATH="${COMPAT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
"${CONDA_BIN}" run --no-capture-output -n hair-gs python render.py \
  -s "${CAMERA_DATASET}" -m "${MODEL_DIR}" \
  --iterations "${iteration}" --skip_test --quiet

cd "${PROJECT_ROOT}"
"${CONDA_BIN}" run --no-capture-output -n hair-gs python \
  scripts/export_hairgs_external_renders.py \
  --model-dir "${MODEL_DIR}" \
  --camera-dataset "${CAMERA_DATASET}" \
  --camera-manifest "${PROTOCOL_ROOT}/test/camera_manifest.json" \
  --output-dir "${OUTPUT_ROOT}/external_renders" \
  --iteration "${iteration}" \
  --output-size 512 512 \
  --hairgs-root "${HAIRGS_ROOT}"

"${CONDA_BIN}" run --no-capture-output -n dpd3dgs-animal python \
  scripts/evaluate_external_renders.py \
  --render-manifest "${OUTPUT_ROOT}/external_renders/render_manifest.json" \
  --ground-truth-dir "${GROUND_TRUTH}" \
  --output-dir "${OUTPUT_ROOT}/evaluation" \
  --method Hair-GS \
  --protocol-id "${PROTOCOL_ID}" \
  --device cuda

printf 'iteration=%s\nevaluation=%s\n' \
  "${iteration}" "${OUTPUT_ROOT}/evaluation/evaluation.json"
