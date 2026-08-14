#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/aoki/Differentiable_Physical_Driven_3DGS_for_monocular_Animal_Motion_Reconstruction}"
PYTHON="${PYTHON:-/home/aoki/miniconda3/envs/hair-gs/bin/python}"
PROTOCOL="${PROTOCOL:-/mnt/f/fur_hair_unified_data/benchmarks/gaussian_haircut_person0_protocol}"
OUTPUT="${OUTPUT:-/mnt/f/fur_hair_unified_data/benchmarks/hairgs_person0_protocol_official_v2}"
FLAME_MASK="${FLAME_MASK:-/home/aoki/fur_hair_baselines/hair-gs/dataset/FLAME/FLAME_masks.pkl}"
FORCE="${FORCE:-0}"

if [[ "${FORCE}" != "1" && -f "${OUTPUT}/diagnostics_official_scalp/geometry_alignment.json" ]]; then
  if "${PYTHON}" - "${OUTPUT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
sanity = json.loads((root / "train/preprocess_sanity.json").read_text())
geometry = json.loads(
    (root / "diagnostics_official_scalp/geometry_alignment.json").read_text()
)
valid = (
    sanity.get("background_rgb_max") == 0
    and sanity.get("orientation_estimator") == "Hair-GS released Gabor estimator"
    and sanity.get("scalp_vertex_source") == "official FLAME_masks.pkl/scalp"
    and geometry.get("passed") is True
)
raise SystemExit(0 if valid else 1)
PY
  then
    echo "Reusing validated Hair-GS person0 protocol: ${OUTPUT}"
    exit 0
  fi
fi

[[ -f "${FLAME_MASK}" ]] || {
  echo "Missing licensed/downloaded FLAME vertex masks: ${FLAME_MASK}" >&2
  exit 2
}

"${PYTHON}" "${PROJECT}/scripts/prepare_hairgs_person0_protocol.py" \
  --protocol-root "${PROTOCOL}" \
  --stage1-npz "${PROTOCOL}/static_head_stage1.npz" \
  --flame-mask-path "${FLAME_MASK}" \
  --body-mask-root "${PROTOCOL}/masks_2/body" \
  --out-root "${OUTPUT}"

"${PYTHON}" "${PROJECT}/scripts/validate_hairgs_person0_protocol.py" \
  --dataset "${OUTPUT}/train" \
  --output-dir "${OUTPUT}/diagnostics_official_scalp" \
  --strict
