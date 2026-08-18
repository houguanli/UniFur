#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/aoki/Differentiable_Physical_Driven_3DGS_for_monocular_Animal_Motion_Reconstruction}"
DATA_ROOT="${DATA_ROOT:-/mnt/f/fur_hair_unified_data}"
PROTOCOL_ROOT="${DATA_ROOT}/benchmarks/hairgs_wcurly_static_protocol"
RESULT_ROOT="${DATA_ROOT}/benchmarks/hairgs_wcurly_static_results"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/fiber_hairgs_wcurly_cleanbase_mvtopology_6k.yaml}"
STAGE1="${PROTOCOL_ROOT}/static_head_stage1.npz"
GAUSSIAN="${RESULT_ROOT}/hairgs_official_train12_30k30k/point_cloud/iteration_30000/point_cloud.ply"
CHECKPOINT="${RESULT_ROOT}/clean_stage1_full124k_exact_teacher_v2/unified_fiber_field.pt"
CONDA_BIN="${CONDA_BIN:-/home/aoki/miniconda3/bin/conda}"
PROTOCOL_ID="hairgs-wcurly-static-train12-test4-v2-camera-fixed"

cd "${PROJECT_ROOT}"
for spec in 040:0.040 030:0.030 025:0.025 020:0.020; do
  tag="${spec%%:*}"
  fraction="${spec#*:}"
  output="${RESULT_ROOT}/clean_stage1_scaleclip_${tag}_teacher_test4_ssaa"
  PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    "${CONDA_BIN}" run --no-capture-output -n hair-gs python -m \
    dpd3dgs_animal.cli --config "${CONFIG}" fiber-eval \
    --stage1-npz "${STAGE1}" --gaussian-ply "${GAUSSIAN}" \
    --checkpoint "${CHECKPOINT}" \
    --frame-dir "${PROTOCOL_ROOT}/test/images" \
    --camera-manifest "${PROTOCOL_ROOT}/test/camera_manifest.json" \
    --out-dir "${output}" --renderer hairgs --route-mode residual \
    --render-width 1000 --render-height 1000 --export-external-renders \
    --residual-max-scale-fraction "${fraction}"
  "${CONDA_BIN}" run --no-capture-output -n dpd3dgs-animal python \
    scripts/downsample_external_renders.py \
    --render-manifest "${output}/external_render_manifest.json" \
    --output-dir "${output}/external_renders_512" --width 512 --height 512
  "${CONDA_BIN}" run --no-capture-output -n dpd3dgs-animal python \
    scripts/evaluate_external_renders.py \
    --render-manifest "${output}/external_renders_512/render_manifest.json" \
    --ground-truth-dir "${PROTOCOL_ROOT}/test/images" \
    --output-dir "${output}/external_evaluation_512" \
    --method "clean_stage1_scaleclip_${tag}" \
    --protocol-id "${PROTOCOL_ID}" --device cuda
done
