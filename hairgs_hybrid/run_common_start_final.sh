#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/mnt/f/unifur_hairgs_hybrid_experiment}"
CASE="${1:-both}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/f/fur_hair_unified_data/benchmarks/hairgs_hybrid/common_start_final}"

case "${CASE}" in
  panda)
    RUN_PANDA=1
    RUN_WCURLY=0
    ;;
  wcurly)
    RUN_PANDA=0
    RUN_WCURLY=1
    ;;
  both)
    RUN_PANDA=1
    RUN_WCURLY=1
    ;;
  *)
    echo "Usage: $0 [panda|wcurly|both]" >&2
    exit 2
    ;;
esac

export OUTPUT_ROOT RUN_PANDA RUN_WCURLY
export VARIANTS=neutral_immediate_staged
exec bash "${EXPERIMENT_ROOT}/hairgs_hybrid/run_neutral_initialization_ablation.sh"
