#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
result_root="${WCURLY_RESULT_ROOT:-/mnt/f/fur_hair_unified_data/benchmarks/hairgs_wcurly_static_results}"

export WCURLY_CLEAN_RESUME_PLY="${WCURLY_CLEAN_RESUME_PLY:-${result_root}/clean_hair_stage1_visualhull80k_mvcal_v8_coverage_refine4k/point_cloud/iteration_4000/point_cloud.ply}"
export WCURLY_CLEAN_OUTPUT="${WCURLY_CLEAN_OUTPUT:-${result_root}/clean_hair_stage1_visualhull80k_mvcal_v9_headocclusion_growth6k}"
export WCURLY_CLEAN_ITERATIONS="${WCURLY_CLEAN_ITERATIONS:-6000}"
export WCURLY_CLEAN_DENSIFY_FROM="${WCURLY_CLEAN_DENSIFY_FROM:-500}"
export WCURLY_CLEAN_DENSIFY_UNTIL="${WCURLY_CLEAN_DENSIFY_UNTIL:-4500}"
export WCURLY_CLEAN_DENSIFY_INTERVAL="${WCURLY_CLEAN_DENSIFY_INTERVAL:-500}"
export WCURLY_CLEAN_POSITION_LR_INIT="${WCURLY_CLEAN_POSITION_LR_INIT:-0.000016}"
export WCURLY_CLEAN_POSITION_LR_FINAL="${WCURLY_CLEAN_POSITION_LR_FINAL:-0.0000016}"
export WCURLY_CLEAN_FEATURE_LR="${WCURLY_CLEAN_FEATURE_LR:-0.0025}"
export WCURLY_CLEAN_OPACITY_LR="${WCURLY_CLEAN_OPACITY_LR:-0.005}"
export WCURLY_CLEAN_SCALING_LR="${WCURLY_CLEAN_SCALING_LR:-0.0005}"
export WCURLY_CLEAN_OPACITY_RESET_INTERVAL=100000
export WCURLY_CLEAN_COVERAGE_WEIGHT="${WCURLY_CLEAN_COVERAGE_WEIGHT:-2.0}"
export WCURLY_CLEAN_DICE_WEIGHT="${WCURLY_CLEAN_DICE_WEIGHT:-1.0}"
export WCURLY_CLEAN_SAVE_FREQUENCY="${WCURLY_CLEAN_SAVE_FREQUENCY:-1000}"

exec bash "${repo_root}/scripts/run_hairgs_wcurly_clean_stage1.sh"
