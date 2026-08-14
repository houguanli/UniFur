#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/aoki/fur_hair_baselines/hair-gs}"
CONDA_BIN="${CONDA_BIN:-/home/aoki/miniconda3/bin/conda}"
PROJECT="${PROJECT:-/home/aoki/Differentiable_Physical_Driven_3DGS_for_monocular_Animal_Motion_Reconstruction}"
DATA="${DATA:-/mnt/f/fur_hair_unified_data/benchmarks/hairgs_person0_protocol_official_v2}"
MODEL="${MODEL:-/mnt/f/fur_hair_unified_data/benchmarks/hairgs_person0_results/odd33_officialpreproc_v2_preflight3k}"
EVAL="${EVAL:-/mnt/f/fur_hair_unified_data/benchmarks/hairgs_person0_results/odd33_officialpreproc_v2_preflight3k_evaluation}"
ITERATIONS="${ITERATIONS:-3000}"
DENSIFY_UNTIL="${DENSIFY_UNTIL:-2500}"

run_hairgs() {
  (cd "${ROOT}" && "${CONDA_BIN}" run --no-capture-output -n hair-gs "$@")
}

if [[ -e "${MODEL}/.complete" ]]; then
  echo "Preflight already complete: ${MODEL}"
  exit 0
fi
if [[ -d "${MODEL}/point_cloud" ]]; then
  echo "Refusing to reuse a partial preflight directory: ${MODEL}" >&2
  exit 2
fi
mkdir -p "${MODEL}" "${EVAL}"

run_hairgs python train.py -s "${DATA}/train" -m "${MODEL}" \
  --iterations "${ITERATIONS}" --densify_until_iter "${DENSIFY_UNTIL}" \
  --save_frequency "${ITERATIONS}" --quiet

run_hairgs python render.py -s "${DATA}/test" -m "${MODEL}" --type 0 --quiet
run_hairgs python render.py -s "${DATA}/test" -m "${MODEL}" --type 2 --quiet

PYTHONPATH="${PROJECT}/src" "${CONDA_BIN}" run --no-capture-output -n hair-gs \
  python "${PROJECT}/scripts/export_hairgs_external_renders.py" \
  --model-dir "${MODEL}" --camera-dataset "${DATA}/test" \
  --camera-manifest "${DATA}/test/protocol_manifest.json" \
  --output-dir "${EVAL}" --iteration "${ITERATIONS}" --output-size 1024 1024

"${CONDA_BIN}" run --no-capture-output -n dpd3dgs-animal \
  python "${PROJECT}/scripts/evaluate_external_renders.py" \
  --render-manifest "${EVAL}/render_manifest.json" \
  --ground-truth-dir "/mnt/f/fur_hair_unified_data/benchmarks/gaussian_haircut_person0_protocol/protocol/test/images" \
  --output-dir "${EVAL}/external_evaluation" --method "Hair-GS preflight 3k" \
  --protocol-id "gaussian-haircut-person0-odd33fit-even33test-v1-preflight" \
  --device cuda

# Merge is part of the gate: a healthy Stage-I field must yield a non-trivial
# number of scalp-connected endpoints before the expensive Stage-III run.
MERGE_LOG="${MODEL}/preflight_merge.log"
run_hairgs python merge.py -s "${DATA}/train" -m "${MODEL}" --quiet 2>&1 | tee "${MERGE_LOG}"

"${CONDA_BIN}" run --no-capture-output -n hair-gs python - \
  "${EVAL}/external_evaluation/evaluation.json" "${MERGE_LOG}" <<'PY'
import json
import math
import re
import sys

evaluation_path, merge_log_path = sys.argv[1:]
aggregate = json.load(open(evaluation_path, encoding="utf-8"))["aggregate"]
merge_log = open(merge_log_path, encoding="utf-8", errors="replace").read()
matches = re.findall(r"Identified\s+(\d+)\s+endpoints", merge_log)
if not matches:
    raise SystemExit("preflight failed: merge.py did not report endpoint count")
endpoints = int(matches[-1])
foreground_psnr = float(aggregate["foreground_psnr"])
mask_iou = float(aggregate["mask_iou"])
print(json.dumps({
    "foreground_psnr": foreground_psnr,
    "mask_iou": mask_iou,
    "merge_endpoints": endpoints,
}, indent=2))
if not math.isfinite(foreground_psnr) or foreground_psnr < 6.0:
    raise SystemExit("preflight failed: invalid held-out foreground reconstruction")
if not math.isfinite(mask_iou) or mask_iou < 0.05:
    raise SystemExit("preflight failed: learned hair foreground does not overlap GT")
if endpoints < 100:
    raise SystemExit("preflight failed: too few scalp-connected endpoints")
PY
touch "${MODEL}/.complete"
