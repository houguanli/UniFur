#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
result_root="${WCURLY_RESULT_ROOT:-/mnt/f/fur_hair_unified_data/benchmarks/hairgs_wcurly_static_results}"

# This pass changes appearance only.  It projects the v10 DC colours onto a
# robust warm-colour cone estimated from all train12 hair-mask pixels; geometry,
# opacity, topology, and semantic ownership remain bitwise unchanged.
export WCURLY_CLEAN_RESUME_PLY="${WCURLY_CLEAN_RESUME_PLY:-${result_root}/clean_hair_stage1_visualhull80k_mvcal_v10_warmgamut_cleanup1k/point_cloud/iteration_1000/point_cloud.ply}"
export WCURLY_CLEAN_OUTPUT="${WCURLY_CLEAN_OUTPUT:-${result_root}/clean_hair_stage1_visualhull80k_mvcal_v11b_adaptivegamut_projection}"
export WCURLY_CLEAN_ITERATIONS=1
export WCURLY_CLEAN_DENSIFY_FROM=0
export WCURLY_CLEAN_DENSIFY_UNTIL=0
export WCURLY_CLEAN_POSITION_LR_INIT=0
export WCURLY_CLEAN_POSITION_LR_FINAL=0
export WCURLY_CLEAN_FEATURE_LR=0
export WCURLY_CLEAN_OPACITY_LR=0
export WCURLY_CLEAN_SCALING_LR=0
export WCURLY_CLEAN_OPACITY_RESET_INTERVAL=100000
export WCURLY_CLEAN_ENFORCE_WARM_GAMUT=1
export WCURLY_CLEAN_WARM_GAMUT_QUANTILE="${WCURLY_CLEAN_WARM_GAMUT_QUANTILE:-0.999}"
export WCURLY_CLEAN_SAVE_FREQUENCY=1

exec bash "${repo_root}/scripts/run_hairgs_wcurly_clean_stage1.sh"
