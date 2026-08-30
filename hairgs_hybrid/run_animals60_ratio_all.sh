#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
FIXED_ROOT="${ANIMALS60_FIXED_ROOT:-${EXPERIMENT_ROOT}/outputs/animals60_v48_25k}"
OUTPUT_ROOT="${ANIMALS60_RATIO_ROOT:-${EXPERIMENT_ROOT}/outputs/animals60_v48_ratio_ladder}"
mkdir -p "${OUTPUT_ROOT}/audit"

CASES=(A003_silver_fox A008_gray_wolf A027_polar_bear A029_goat A037_horse A045_rabbit A050_horned_cow A063_wild_ass)
COMPRESSIONS=(0.75 0.875 0.9375)

for case_id in "${CASES[@]}"; do
  test -f "${FIXED_ROOT}/cases/${case_id}/COMPLETE"
done

for compression in "${COMPRESSIONS[@]}"; do
  for case_id in "${CASES[@]}"; do
    bash "${SCRIPT_DIR}/run_animals60_ratio_case.sh" "${case_id}" "${compression}" \
      2>&1 | tee "${OUTPUT_ROOT}/audit/${case_id}_c${compression}.log"
  done
done

echo "Animals60 percentage-ratio ladder complete: ${OUTPUT_ROOT}"
