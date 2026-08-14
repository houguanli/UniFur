#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-evaluate}"
PROJECT="${PROJECT:-/home/aoki/Differentiable_Physical_Driven_3DGS_for_monocular_Animal_Motion_Reconstruction}"
ROOT="${ROOT:-/home/aoki/fur_hair_baselines/Im2Haircut}"
CONDA_BIN="${CONDA_BIN:-/home/aoki/miniconda3/bin/conda}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/f/fur_hair_unified_data/benchmarks/im2haircut_singleview_results/person0_scaffold}"
SCENE_OUTPUT="${OUTPUT_ROOT}/0049.png"
STRANDS="${STRANDS:-${SCENE_OUTPUT}/pointclouds_train/pred_000300.ply}"
PROTOCOL="${PROTOCOL:-/mnt/f/fur_hair_unified_data/benchmarks/gaussian_haircut_person0_protocol}"
EVALUATION="${EVALUATION:-/mnt/f/fur_hair_unified_data/benchmarks/im2haircut_singleview_results/person0_scaffold_0049_structure_evaluation}"

run_reconstruction() {
  SCENE=0049.png FOLDER=data/person0_singleview OUTPUT_ROOT="${OUTPUT_ROOT}" \
    "${PROJECT}/scripts/run_im2haircut_singleview.sh"
}

run_evaluation() {
  [[ -f "${STRANDS}" ]] || {
    echo "Missing Im2Haircut person0 strands: ${STRANDS}" >&2
    exit 2
  }
  "${CONDA_BIN}" run --no-capture-output -n dpd3dgs-animal python \
    "${PROJECT}/scripts/evaluate_im2haircut_person0_structure.py" \
    --strand-ply "${STRANDS}" \
    --protocol-root "${PROTOCOL}" \
    --canonical-to-head "${ROOT}/data/person0_singleview/im2canonical_to_person0.txt" \
    --output-dir "${EVALUATION}"
}

case "${MODE}" in
  reconstruct) run_reconstruction ;;
  evaluate) run_evaluation ;;
  full) run_reconstruction; run_evaluation ;;
  *) echo "Usage: $0 {reconstruct|evaluate|full}" >&2; exit 2 ;;
esac
