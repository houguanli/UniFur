#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/aoki/Differentiable_Physical_Driven_3DGS_for_monocular_Animal_Motion_Reconstruction}"
CONDA_BIN="${CONDA_BIN:-/home/aoki/miniconda3/bin/conda}"
MODEL_DIR="${MODEL_DIR:-/mnt/f/fur_hair_unified_data/benchmarks/neuralfur_panda_shared/neuralfur_4k_scale4_r480_preflight500_v3_hairrasterizer}"
ITERATIONS="${ITERATIONS:-500}"
CHECKPOINT="${MODEL_DIR}/checkpoints/${ITERATIONS}.pth"
RENDER_ROOT="${MODEL_DIR}/heldout_preflight_render"
EVAL_ROOT="${MODEL_DIR}/heldout_preflight_evaluation"

if [[ -f "${MODEL_DIR}/.complete" ]]; then
  echo "NeuralFur Panda preflight already complete: ${MODEL_DIR}"
  exit 0
fi
if [[ -d "${MODEL_DIR}/checkpoints" && ! -f "${CHECKPOINT}" ]]; then
  echo "Refusing to silently reuse a partial NeuralFur preflight: ${MODEL_DIR}" >&2
  exit 2
fi

MODEL_DIR="${MODEL_DIR}" ITERATIONS="${ITERATIONS}" \
  "${PROJECT}/scripts/run_neuralfur_panda_4k_24gb.sh"

MODEL_DIR="${MODEL_DIR}" CHECKPOINT="${CHECKPOINT}" \
RENDER_ROOT="${RENDER_ROOT}" EVAL_ROOT="${EVAL_ROOT}" \
ITERATIONS="${ITERATIONS}" MAX_OBSERVATIONS=2 \
METHOD_NAME="NeuralFur preflight (4k active, scale4/r480)" \
  "${PROJECT}/scripts/run_neuralfur_static_benchmark.sh" render

MODEL_DIR="${MODEL_DIR}" CHECKPOINT="${CHECKPOINT}" \
RENDER_ROOT="${RENDER_ROOT}" EVAL_ROOT="${EVAL_ROOT}" \
ITERATIONS="${ITERATIONS}" MAX_OBSERVATIONS=2 \
METHOD_NAME="NeuralFur preflight (4k active, scale4/r480)" \
  "${PROJECT}/scripts/run_neuralfur_static_benchmark.sh" evaluate

"${CONDA_BIN}" run --no-capture-output -n hair-gs python - \
  "${RENDER_ROOT}/render_manifest.json" "${EVAL_ROOT}/evaluation.json" <<'PY'
import json
import math
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
evaluation = json.load(open(sys.argv[2], encoding="utf-8"))
diagnostics = manifest["diagnostics"]
aggregate = evaluation["aggregate"]
summary = {
    "camera_raster_size": diagnostics.get("camera_raster_size"),
    "body_visible_count": diagnostics.get("body_visible_count"),
    "hair_visible_count": diagnostics.get("hair_visible_count"),
    "foreground_psnr": aggregate.get("foreground_psnr"),
    "mask_iou": aggregate.get("mask_iou"),
}
print(json.dumps(summary, indent=2))
if manifest.get("status") != "complete":
    raise SystemExit("NeuralFur preflight render did not complete")
if diagnostics.get("camera_raster_size") != [480, 270]:
    raise SystemExit("NeuralFur camera/raster scaling is inconsistent")
if int(diagnostics.get("body_visible_count", 0)) < 100:
    raise SystemExit("NeuralFur body prior is outside the held-out camera")
if int(diagnostics.get("hair_visible_count", 0)) < 100:
    raise SystemExit("NeuralFur generated fur is outside the held-out camera")
psnr = float(aggregate["foreground_psnr"])
iou = float(aggregate["mask_iou"])
if not math.isfinite(psnr) or psnr < 5.0:
    raise SystemExit("NeuralFur preflight RGB is invalid")
if not math.isfinite(iou) or iou < 0.05:
    raise SystemExit("NeuralFur preflight foreground is invalid")
PY

touch "${MODEL_DIR}/.complete"
