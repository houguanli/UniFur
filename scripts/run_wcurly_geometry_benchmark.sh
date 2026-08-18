#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/aoki/Differentiable_Physical_Driven_3DGS_for_monocular_Animal_Motion_Reconstruction}"
DATA_ROOT="${DATA_ROOT:-/mnt/f/fur_hair_unified_data}"
HAIRGS_ROOT="${HAIRGS_ROOT:-/home/aoki/fur_hair_baselines/hair-gs}"
CONDA_BIN="${CONDA_BIN:-/home/aoki/miniconda3/bin/conda}"
RESULT_ROOT="${DATA_ROOT}/benchmarks/hairgs_wcurly_static_results"
OUT="${RESULT_ROOT}/wcurly_geometry_eval_20260814"
SOURCE="${DATA_ROOT}/hair-gs_parsed_official16_1000/cem_yuksel/wCurly"
STAGE1="${DATA_ROOT}/benchmarks/hairgs_wcurly_static_protocol/static_head_stage1.npz"
GAUSSIAN="${SOURCE}/sparse/0/points3D.ply"
UNIFIED="${RESULT_ROOT}/unified_fin_carrier_adaptive43k_14k/unified_fiber_field.pt"
RESIDUAL="${RESULT_ROOT}/residual_adaptive43k_8k/unified_fiber_field.pt"
HAIRGS_TRAIN12="${RESULT_ROOT}/hairgs_official_train12_30k30k/point_cloud/iteration_60009/point_cloud.ply"

mkdir -p "${OUT}"

export_geometry() {
  local mode="$1" checkpoint="$2" stem="$3"
  PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    "${CONDA_BIN}" run --no-capture-output -n dpd3dgs-animal python \
    "${PROJECT_ROOT}/scripts/export_unifur_hair_eval.py" \
    --stage1-npz "${STAGE1}" --gaussian-ply "${GAUSSIAN}" \
    --checkpoint "${checkpoint}" --mode "${mode}" --spacing-mm 3 \
    --out-npz "${OUT}/${stem}.npz" \
    --out-report "${OUT}/${stem}_export.json"
}

evaluate_npz() {
  local stem="$1"
  (
    cd "${HAIRGS_ROOT}"
    "${CONDA_BIN}" run --no-capture-output -n hair-gs python \
      "${PROJECT_ROOT}/scripts/hairgs_eval_fixed.py" \
      -s "${SOURCE}" -p "${OUT}/${stem}.npz" -pt hair_eval_npz \
      --out_json "${OUT}/${stem}_metrics.json"
  )
}

export_geometry strand_deployed "${UNIFIED}" unifur_strand_deployed
export_geometry structured_deployed "${UNIFIED}" unifur_structured_deployed
export_geometry strand_target "${UNIFIED}" unifur_strand_target
export_geometry residual_points "${RESIDUAL}" residual_points

evaluate_npz unifur_strand_deployed
evaluate_npz unifur_structured_deployed
evaluate_npz unifur_strand_target
evaluate_npz residual_points

(
  cd "${HAIRGS_ROOT}"
  "${CONDA_BIN}" run --no-capture-output -n hair-gs python \
    "${PROJECT_ROOT}/scripts/hairgs_eval_fixed.py" \
    -s "${SOURCE}" -p "${HAIRGS_TRAIN12}" -pt gs \
    --out_json "${OUT}/hairgs_train12_metrics.json"
)

printf 'geometry_benchmark=%s\n' "${OUT}"
