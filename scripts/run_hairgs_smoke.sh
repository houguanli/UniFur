#!/usr/bin/env bash
set -euo pipefail

CONDA_BIN="${CONDA_BIN:-/home/aoki/miniconda3/bin/conda}"
BASELINE_ROOT="${BASELINE_ROOT:-/home/aoki/fur_hair_baselines/hair-gs}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/aoki/Differentiable_Physical_Driven_3DGS_for_monocular_Animal_Motion_Reconstruction}"
DATA_ROOT="${DATA_ROOT:-/mnt/f/fur_hair_unified_data}"
OUTPUT_DIR="${1:-${DATA_ROOT}/hair-gs_outputs/wStraight_repro_smoke}"

export PYOPENGL_PLATFORM=glx
export PYTHONPATH="${PROJECT_ROOT}/compat/hairgs_sitecustomize${PYTHONPATH:+:${PYTHONPATH}}"
cd "${BASELINE_ROOT}"

if [[ ! -f dataset/parsed/cem_yuksel/wStraight/hair_eval_data.npz ]]; then
  "${CONDA_BIN}" run -n hair-gs python scripts/download_parse_cy.py \
    --cameras 4 --height 256 --width 256
fi

if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to overwrite existing output: ${OUTPUT_DIR}" >&2
  echo "Pass a new output directory as the first argument." >&2
  exit 2
fi
mkdir -p "$(dirname "${OUTPUT_DIR}")"

"${CONDA_BIN}" run -n hair-gs python train.py \
  -s dataset/parsed/cem_yuksel/wStraight -m "${OUTPUT_DIR}" \
  --iterations 10 --save_frequency 10 --eval_frequency 10 --quiet
"${CONDA_BIN}" run -n hair-gs python merge.py \
  -s dataset/parsed/cem_yuksel/wStraight -m "${OUTPUT_DIR}" --quiet
"${CONDA_BIN}" run -n hair-gs python train.py \
  -s dataset/parsed/cem_yuksel/wStraight -m "${OUTPUT_DIR}" \
  --iterations 30 --save_frequency 30 --eval_frequency 30 --quiet

echo "HairGS three-stage smoke output: ${OUTPUT_DIR}"
