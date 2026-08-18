#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/aoki/Differentiable_Physical_Driven_3DGS_for_monocular_Animal_Motion_Reconstruction}"
HAIRGS_ROOT="${HAIRGS_ROOT:-/home/aoki/fur_hair_baselines/hair-gs}"
CONDA_BIN="${CONDA_BIN:-/home/aoki/miniconda3/bin/conda}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-/mnt/f/fur_hair_unified_data/benchmarks/hairgs_wcurly_static_protocol}"
RESULT_ROOT="${RESULT_ROOT:-/mnt/f/fur_hair_unified_data/benchmarks/hairgs_wcurly_static_results}"
MODEL_DIR="${MODEL_DIR:-${RESULT_ROOT}/clean_hair_stage1_visualhull80k_mvcal_v10_warmgamut_cleanup1k}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RESULT_ROOT}/clean_hair_stage1_visualhull80k_mvcal_v10_warmgamut_cleanup1k_strict_test4}"
CAMERA_DATASET="${PROTOCOL_ROOT}/hairgs_test4_dataset"
COMPAT_ROOT="${HAIRGS_COMPAT_ROOT:-${PROJECT_ROOT}/compat/hairgs_sitecustomize}"
PROTOCOL_ID="hairgs-wcurly-static-train12-test4-v2-camera-fixed"

[[ -f "${MODEL_DIR}/clean_stage1_metadata.json" ]] || {
  echo "Clean Stage-1 has not completed: ${MODEL_DIR}" >&2
  exit 2
}
iteration="$({
  find "${MODEL_DIR}/point_cloud" -mindepth 2 -maxdepth 2 -type f \
    -name point_cloud.ply -printf '%h\n'
} | awk -F'iteration_' '{print $2}' | sort -n | tail -1)"
[[ -n "${iteration}" ]] || { echo "No clean Stage-1 iteration found" >&2; exit 2; }

cd "${HAIRGS_ROOT}"
export PYTHONPATH="${COMPAT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
"${CONDA_BIN}" run --no-capture-output -n hair-gs python render.py \
  -s "${CAMERA_DATASET}" -m "${MODEL_DIR}" \
  --iterations "${iteration}" --skip_test --quiet

cd "${PROJECT_ROOT}"
"${CONDA_BIN}" run --no-capture-output -n dpd3dgs-animal python \
  scripts/build_masked_image_set.py \
  --image-dir "${CAMERA_DATASET}/images" \
  --mask-dir "${CAMERA_DATASET}/masks" \
  --output-dir "${OUTPUT_ROOT}/hair_only_ground_truth"

"${CONDA_BIN}" run --no-capture-output -n hair-gs python \
  scripts/export_hairgs_external_renders.py \
  --model-dir "${MODEL_DIR}" --camera-dataset "${CAMERA_DATASET}" \
  --camera-manifest "${PROTOCOL_ROOT}/test/camera_manifest.json" \
  --output-dir "${OUTPUT_ROOT}/external_renders" \
  --iteration "${iteration}" --output-size 1000 1000 \
  --hairgs-root "${HAIRGS_ROOT}"

"${CONDA_BIN}" run --no-capture-output -n dpd3dgs-animal python \
  scripts/evaluate_external_renders.py \
  --render-manifest "${OUTPUT_ROOT}/external_renders/render_manifest.json" \
  --ground-truth-dir "${OUTPUT_ROOT}/hair_only_ground_truth" \
  --output-dir "${OUTPUT_ROOT}/evaluation" \
  --method "Clean hair-only HairGS Stage-1" \
  --protocol-id "${PROTOCOL_ID}" --device cuda

printf 'iteration=%s\nevaluation=%s\n' \
  "${iteration}" "${OUTPUT_ROOT}/evaluation/evaluation.json"
