#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
hairgs_root="${HAIRGS_ROOT:-/home/aoki/fur_hair_baselines/hair-gs}"
conda_bin="${CONDA_BIN:-/home/aoki/miniconda3/bin/conda}"
compat_root="${HAIRGS_COMPAT_ROOT:-${repo_root}/compat/hairgs_sitecustomize}"
protocol_root="${WCURLY_PROTOCOL_ROOT:-/mnt/f/fur_hair_unified_data/benchmarks/hairgs_wcurly_static_protocol}"
source_dataset="${WCURLY_SOURCE_DATASET:-${protocol_root}/hairgs_train12_dataset}"
clean_dataset="${WCURLY_CLEAN_DATASET:-${protocol_root}/hairgs_train12_clean_visualhull80k_v1}"
output_root="${WCURLY_CLEAN_OUTPUT:-/mnt/f/fur_hair_unified_data/benchmarks/hairgs_wcurly_static_results/clean_hair_stage1_visualhull80k_mvcal_v7_cal4_childcap}"
iterations="${WCURLY_CLEAN_ITERATIONS:-15000}"

export PYTHONPATH="${compat_root}:${hairgs_root}${PYTHONPATH:+:${PYTHONPATH}}"
run_hairgs() {
  "${conda_bin}" run --no-capture-output -n hair-gs python "$@"
}

resume_args=()
if [[ -n "${WCURLY_CLEAN_RESUME_PLY:-}" ]]; then
  resume_args=(--clean_resume_ply "${WCURLY_CLEAN_RESUME_PLY}")
fi
gamut_args=()
if [[ "${WCURLY_CLEAN_ENFORCE_WARM_GAMUT:-0}" == "1" ]]; then
  gamut_args=(--clean_enforce_warm_hair_gamut)
fi

if [[ ! -e "${clean_dataset}" ]]; then
  cd "${hairgs_root}"
  run_hairgs "${repo_root}/scripts/build_hairgs_clean_scaffold_dataset.py" \
    --source-dataset "${source_dataset}" \
    --train-manifest "${protocol_root}/train/camera_manifest.json" \
    --output-dataset "${clean_dataset}" \
    --candidate-count "${WCURLY_CLEAN_CANDIDATES:-1200000}" \
    --initial-count "${WCURLY_CLEAN_INITIAL_POINTS:-80000}" \
    --minimum-fit-support "${WCURLY_CLEAN_INIT_SUPPORT:-6}" \
    --calibration-ordinals "${WCURLY_CLEAN_CAL_ORDINALS:-0,3,6,9}" \
    --seed "${WCURLY_CLEAN_SEED:-17}"
fi

if [[ -e "${output_root}" ]]; then
  echo "refusing to overwrite existing clean Stage-1 output: ${output_root}" >&2
  exit 1
fi

cd "${hairgs_root}"
run_hairgs "${repo_root}/scripts/hairgs_train_clean_scaffold.py" \
  -s "${clean_dataset}" -m "${output_root}" -r 1 \
  --iterations "${iterations}" \
  --position_lr_init "${WCURLY_CLEAN_POSITION_LR_INIT:-0.00016}" \
  --position_lr_final "${WCURLY_CLEAN_POSITION_LR_FINAL:-0.0000016}" \
  --position_lr_max_steps "${iterations}" \
  --feature_lr "${WCURLY_CLEAN_FEATURE_LR:-0.025}" \
  --opacity_lr "${WCURLY_CLEAN_OPACITY_LR:-0.05}" \
  --scaling_lr "${WCURLY_CLEAN_SCALING_LR:-0.005}" \
  --opacity_reset_interval "${WCURLY_CLEAN_OPACITY_RESET_INTERVAL:-3000}" \
  --densify_from_iter "${WCURLY_CLEAN_DENSIFY_FROM:-500}" \
  --densify_until_iter "${WCURLY_CLEAN_DENSIFY_UNTIL:-12000}" \
  --densification_interval "${WCURLY_CLEAN_DENSIFY_INTERVAL:-500}" \
  --densify_grad_threshold "${WCURLY_CLEAN_DENSIFY_GRAD:-0.00025}" \
  --save_frequency "${WCURLY_CLEAN_SAVE_FREQUENCY:-3000}" \
  --logger none \
  --lambda_orientation 0 \
  --lambda_mask 0 \
  --mask_lr 0 \
  --clean_outside_rgb_weight "${WCURLY_CLEAN_OUTSIDE_RGB_WEIGHT:-1.0}" \
  --clean_spill_weight "${WCURLY_CLEAN_SPILL_WEIGHT:-1.2}" \
  --clean_coverage_weight "${WCURLY_CLEAN_COVERAGE_WEIGHT:-0.6}" \
  --clean_dice_weight "${WCURLY_CLEAN_DICE_WEIGHT:-0.5}" \
  --clean_local_outside_blend "${WCURLY_CLEAN_LOCAL_OUTSIDE_BLEND:-0.75}" \
  --clean_local_outside_padding "${WCURLY_CLEAN_LOCAL_OUTSIDE_PADDING:-12}" \
  --clean_rgb_min "${WCURLY_CLEAN_RGB_MIN:-0.0}" \
  --clean_rgb_max "${WCURLY_CLEAN_RGB_MAX:-1.0}" \
  --clean_warm_gamut_quantile "${WCURLY_CLEAN_WARM_GAMUT_QUANTILE:-0.999}" \
  --clean_warm_rg_gap_cap "${WCURLY_CLEAN_WARM_RG_GAP_CAP:--1.0}" \
  --clean_warm_gb_gap_cap "${WCURLY_CLEAN_WARM_GB_GAP_CAP:--1.0}" \
  --clean_head_occlusion_margin "${WCURLY_CLEAN_HEAD_OCCLUSION_MARGIN:-0.002}" \
  --clean_orientation_weight "${WCURLY_CLEAN_ORIENTATION_WEIGHT:-2.0}" \
  --clean_scale_weight "${WCURLY_CLEAN_SCALE_WEIGHT:-0.15}" \
  --clean_anisotropy_weight "${WCURLY_CLEAN_ANISOTROPY_WEIGHT:-0.05}" \
  --clean_world_scale_cap "${WCURLY_CLEAN_SCALE_CAP:-0.03}" \
  --clean_min_anisotropy "${WCURLY_CLEAN_MIN_ANISOTROPY:-2.5}" \
  --clean_max_anisotropy "${WCURLY_CLEAN_MAX_ANISOTROPY:-12}" \
  --clean_hard_max_anisotropy "${WCURLY_CLEAN_HARD_MAX_ANISOTROPY:-20}" \
  --clean_max_projected_radius "${WCURLY_CLEAN_MAX_PROJECTED_RADIUS:-20}" \
  --clean_scale_clamp_every "${WCURLY_CLEAN_SCALE_CLAMP_EVERY:-20}" \
  --clean_min_hull_support "${WCURLY_CLEAN_DENSIFY_SUPPORT:-4}" \
  --clean_max_gaussians "${WCURLY_CLEAN_MAX_GAUSSIANS:-350000}" \
  --clean_max_new_per_event "${WCURLY_CLEAN_MAX_NEW_PER_EVENT:-1000}" \
  --clean_new_child_alpha "${WCURLY_CLEAN_NEW_CHILD_ALPHA:-0.005}" \
  --clean_child_total_alpha_budget "${WCURLY_CLEAN_CHILD_TOTAL_ALPHA_BUDGET:-5.0}" \
  --clean_child_jitter_scale "${WCURLY_CLEAN_CHILD_JITTER_SCALE:-0.10}" \
  --clean_child_scale_factor "${WCURLY_CLEAN_CHILD_SCALE_FACTOR:-0.5}" \
  --clean_child_probation_max_alpha "${WCURLY_CLEAN_CHILD_PROBATION_MAX_ALPHA:-0.02}" \
  --clean_child_min_calibration_support "${WCURLY_CLEAN_CHILD_MIN_CAL_SUPPORT:-4}" \
  --clean_topology_probation "${WCURLY_CLEAN_TOPOLOGY_PROBATION:-100}" \
  --clean_topology_min_fit_gain "${WCURLY_CLEAN_TOPOLOGY_MIN_FIT_GAIN:-0.000001}" \
  --clean_calibration_view_margin "${WCURLY_CLEAN_CAL_VIEW_MARGIN:-0.002}" \
  --clean_calibration_mean_margin "${WCURLY_CLEAN_CAL_MEAN_MARGIN:-0.0005}" \
  "${resume_args[@]}" \
  "${gamut_args[@]}"

printf 'clean_dataset=%s\nclean_stage1=%s\n' "${clean_dataset}" "${output_root}"
