#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
OUTPUT_ROOT="${ANIMALS60_OUTPUT_ROOT:-${EXPERIMENT_ROOT}/outputs/animals60_v48_25k}"
mkdir -p "${OUTPUT_ROOT}/audit"

CASES=(
  A003_silver_fox
  A008_gray_wolf
  A027_polar_bear
  A029_goat
  A037_horse
  A045_rabbit
  A050_horned_cow
  A063_wild_ass
)

for case_id in "${CASES[@]}"; do
  if [[ -f "${OUTPUT_ROOT}/cases/${case_id}/COMPLETE" ]]; then
    continue
  fi
  bash "${SCRIPT_DIR}/run_animals60_case.sh" "${case_id}" \
    2>&1 | tee "${OUTPUT_ROOT}/audit/${case_id}.log"
done

source "${EXPERIMENT_ROOT}/scripts/lib/runtime.sh"
CONDA_BIN="$(resolve_conda_bin)"
"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV:-unifur}" python \
  "${EXPERIMENT_ROOT}/hairgs_hybrid/audit_animals60_benchmark.py" \
  --benchmark-root "${OUTPUT_ROOT}"
echo "Animals60 benchmark complete: ${OUTPUT_ROOT}"
