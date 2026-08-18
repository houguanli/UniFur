#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/aoki/Differentiable_Physical_Driven_3DGS_for_monocular_Animal_Motion_Reconstruction}"
DATA_ROOT="${DATA_ROOT:-/mnt/f/fur_hair_unified_data}"
RESULT_ROOT="${DATA_ROOT}/benchmarks/hairgs_wcurly_static_results"
PROTOCOL_ROOT="${DATA_ROOT}/benchmarks/hairgs_wcurly_static_protocol"
CHECKPOINT="${RESULT_ROOT}/cleanhair_v13_adaptive_residualfree_6k_v1/unified_fiber_field.pt"
STAGE1="${PROTOCOL_ROOT}/static_head_stage1.npz"
CLEAN_PLY="${RESULT_ROOT}/clean_hair_stage1_visualhull80k_mvcal_v11b_adaptivegamut_projection/point_cloud/iteration_1/point_cloud.ply"
BASE_PLY="${RESULT_ROOT}/hairgs_official_train12_30k30k/point_cloud/iteration_30000/point_cloud.ply"
CONDA_BIN="${CONDA_BIN:-/home/aoki/miniconda3/bin/conda}"
CONFIG_FULL="${PROJECT_ROOT}/configs/fiber_hairgs_wcurly_cleanhair_adaptive_v13_6k.yaml"
CONFIG_FIN_OFF="${PROJECT_ROOT}/configs/fiber_hairgs_wcurly_cleanhair_adaptive_v13_fin_off_eval.yaml"
PROTOCOL_ID="hairgs-wcurly-static-train12-test4-v2-camera-fixed"

run_case() {
  local split="$1" spec="$2" route config gt cameras out
  route="hard"
  config="${CONFIG_FULL}"
  case "${spec}" in
    fin_off_hard) config="${CONFIG_FIN_OFF}" ;;
    shell_only) route="shell" ;;
    strand_only) route="strand" ;;
    *) echo "Unknown diagnostic: ${spec}" >&2; return 2 ;;
  esac

  gt="${PROTOCOL_ROOT}/${split}/images"
  cameras="${PROTOCOL_ROOT}/${split}/camera_manifest.json"
  out="${RESULT_ROOT}/cleanhair_v13_adaptive_residualfree_6k_v1_diag_${spec}_${split}"
  [[ -f "${out}/external_evaluation/evaluation.json" ]] && return

  PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    "${CONDA_BIN}" run --no-capture-output -n hair-gs python -m \
    dpd3dgs_animal.cli --config "${config}" fiber-eval \
    --stage1-npz "${STAGE1}" \
    --gaussian-ply "${CLEAN_PLY}" \
    --fixed-base-gaussian-ply "${BASE_PLY}" \
    --checkpoint "${CHECKPOINT}" \
    --frame-dir "${gt}" \
    --camera-manifest "${cameras}" \
    --out-dir "${out}" --renderer hairgs --route-mode "${route}" \
    --render-width 1000 --render-height 1000 --export-external-renders

  "${CONDA_BIN}" run --no-capture-output -n dpd3dgs-animal python \
    "${PROJECT_ROOT}/scripts/evaluate_external_renders.py" \
    --render-manifest "${out}/external_render_manifest.json" \
    --ground-truth-dir "${gt}" \
    --output-dir "${out}/external_evaluation" \
    --method "UniFur v13 ${spec}" --protocol-id "${PROTOCOL_ID}" --device cuda
}

for split in train test; do
  for spec in fin_off_hard shell_only strand_only; do
    run_case "${split}" "${spec}"
  done
done
