#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/aoki/fur_hair_baselines/hair-gs}"
CONDA_BIN="${CONDA_BIN:-/home/aoki/miniconda3/bin/conda}"
PROJECT="${PROJECT:-/home/aoki/Differentiable_Physical_Driven_3DGS_for_monocular_Animal_Motion_Reconstruction}"
DATA="${DATA:-/mnt/f/fur_hair_unified_data/benchmarks/hairgs_person0_protocol_official_v2}"
MODEL="${MODEL:-/mnt/f/fur_hair_unified_data/benchmarks/hairgs_person0_results/odd33_officialpreproc_v2_30k30k}"
EVAL="${EVAL:-/mnt/f/fur_hair_unified_data/benchmarks/hairgs_person0_results/odd33_officialpreproc_v2_30k30k_evaluation}"
PROTOCOL="gaussian-haircut-person0-odd33fit-even33test-v1"
GEOMETRY_REPORT="${DATA}/diagnostics_official_scalp/geometry_alignment.json"

run_hairgs() {
  (cd "${ROOT}" && "${CONDA_BIN}" run --no-capture-output -n hair-gs "$@")
}

mkdir -p "${MODEL}" "${EVAL}"
if [[ ! -f "${DATA}/train/preprocess_sanity.json" ]]; then
  echo "Missing repaired Hair-GS preprocessing: ${DATA}/train/preprocess_sanity.json" >&2
  exit 2
fi
"${CONDA_BIN}" run -n hair-gs python - \
  "${DATA}/train/preprocess_sanity.json" "${GEOMETRY_REPORT}" <<'PY'
import json
import sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
if report.get("background_rgb_max") != 0:
    raise SystemExit(f"invalid RGB background removal: {report}")
if report.get("orientation_estimator") != "Hair-GS released Gabor estimator":
    raise SystemExit(f"invalid orientation estimator: {report}")
if report.get("scalp_vertex_source") != "official FLAME_masks.pkl/scalp":
    raise SystemExit(f"invalid scalp vertex source: {report}")
geometry = json.load(open(sys.argv[2], encoding="utf-8"))
if not geometry.get("passed"):
    raise SystemExit(f"invalid FLAME/camera/scalp alignment: {geometry['aggregate']}")
PY
if [[ ! -f "${MODEL}/.stage1_complete" ]]; then
  if [[ -f "${MODEL}/point_cloud/iteration_10000/point_cloud.ply" ]]; then
    # The native 0.9*30k densification schedule exceeds a 24GB card on
    # person0 and begins degrading RGB after 10k.  Resume the original LR
    # schedule from the last reliable checkpoint, with density fixed.
    run_hairgs python "${PROJECT}/scripts/hairgs_train_resume.py" \
      -s "${DATA}/train" -m "${MODEL}" --iterations 30000 \
      --resume_step 10000 --densify_until_iter 10000 \
      --save_frequency 5000 --quiet
  else
    run_hairgs python train.py -s "${DATA}/train" -m "${MODEL}" \
      --iterations 30000 --densify_until_iter 10000 \
      --save_frequency 5000 --quiet
  fi
  touch "${MODEL}/.stage1_complete"
fi
if [[ ! -f "${MODEL}/.merge_complete" ]]; then
  MERGE_LOG="${MODEL}/stage1_merge.log"
  run_hairgs python merge.py -s "${DATA}/train" -m "${MODEL}" --quiet 2>&1 | tee "${MERGE_LOG}"
  ENDPOINTS="$(grep -Eo 'Identified [0-9]+ endpoints' "${MERGE_LOG}" | tail -n 1 | grep -Eo '[0-9]+' || true)"
  if [[ -z "${ENDPOINTS}" || "${ENDPOINTS}" -lt 300 ]]; then
    echo "Stage-I merge gate failed: endpoints=${ENDPOINTS:-missing}" >&2
    exit 3
  fi
  touch "${MODEL}/.merge_complete"
fi
if [[ ! -f "${MODEL}/.stage3_complete" ]]; then
  run_hairgs python train.py -s "${DATA}/train" -m "${MODEL}" \
    --iterations 30000 --densify_until_iter 0 \
    --save_frequency 5000 --quiet
  touch "${MODEL}/.stage3_complete"
fi

FINAL_ITER="$(${CONDA_BIN} run -n hair-gs python -c \
  "from pathlib import Path; print(max(int(p.name.split('_')[-1]) for p in Path('${MODEL}/point_cloud').glob('iteration_*') if (p/'point_cloud.ply').is_file()))" \
  | tail -n 1)"
# Render only the two channels required by the common evaluator.  The upstream
# default additionally requests held-out orientation GT and crashes when that
# intentionally withheld supervision is absent.
EXPECTED_VIEWS="$(${CONDA_BIN} run -n hair-gs python -c \
  "import json; print(len(json.load(open('${DATA}/test/protocol_manifest.json'))['observations']))" \
  | tail -n 1)"
render_channel() {
  local type="$1"
  local name="$2"
  local output="${MODEL}/render/train/iteration_${FINAL_ITER}/renders/${name}"
  local count=0
  count="$(find "${output}" -maxdepth 1 -type f -name '*.png' 2>/dev/null | wc -l)"
  if [[ "${count}" -eq "${EXPECTED_VIEWS}" ]]; then
    echo "Reusing complete Hair-GS ${name} render (${count} views)"
    return
  fi
  if ! run_hairgs python render.py -s "${DATA}/test" -m "${MODEL}" \
      --type "${type}" --quiet; then
    # Upstream has occasionally returned non-zero during interpreter teardown
    # after all PNGs were flushed.  Accept that only when the frozen manifest's
    # exact view count is present; any missing render remains a hard failure.
    count="$(find "${output}" -maxdepth 1 -type f -name '*.png' 2>/dev/null | wc -l)"
    if [[ "${count}" -ne "${EXPECTED_VIEWS}" ]]; then
      echo "Hair-GS ${name} render failed: ${count}/${EXPECTED_VIEWS} views" >&2
      exit 4
    fi
    echo "Warning: render.py exited non-zero after writing all ${count} ${name} views" >&2
  fi
}
render_channel 0 rgb
render_channel 2 mask_foreground

PYTHONPATH="${PROJECT}/src" "${CONDA_BIN}" run --no-capture-output -n hair-gs \
  python "${PROJECT}/scripts/export_hairgs_external_renders.py" \
  --model-dir "${MODEL}" --camera-dataset "${DATA}/test" \
  --camera-manifest "${DATA}/test/protocol_manifest.json" \
  --output-dir "${EVAL}" --iteration "${FINAL_ITER}" --output-size 1024 1024

"${CONDA_BIN}" run --no-capture-output -n dpd3dgs-animal \
  python "${PROJECT}/scripts/evaluate_external_renders.py" \
  --render-manifest "${EVAL}/render_manifest.json" \
  --ground-truth-dir "/mnt/f/fur_hair_unified_data/benchmarks/gaussian_haircut_person0_protocol/protocol/test/images" \
  --output-dir "${EVAL}/external_evaluation" \
  --method "Hair-GS (official preprocessing, 24GB densify<=10k)" \
  --protocol-id "${PROTOCOL}" --device cuda
