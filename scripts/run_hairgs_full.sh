#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <hairgs-data-dir> <output-dir> [iterations-per-stage]" >&2
  exit 2
fi

data_dir="$(realpath "$1")"
output_dir="$(realpath -m "$2")"
iterations="${3:-30000}"
hairgs_root="${HAIRGS_ROOT:-/home/aoki/fur_hair_baselines/hair-gs}"
conda_bin="${CONDA_BIN:-/home/aoki/miniconda3/bin/conda}"
compat_root="${HAIRGS_COMPAT_ROOT:-/home/aoki/Differentiable_Physical_Driven_3DGS_for_monocular_Animal_Motion_Reconstruction/compat/hairgs_sitecustomize}"

if [[ -e "$output_dir" ]]; then
  echo "refusing to overwrite existing output: $output_dir" >&2
  exit 1
fi

mkdir -p "$(dirname "$output_dir")"
cd "$hairgs_root"
export PYTHONPATH="$compat_root${PYTHONPATH:+:$PYTHONPATH}"

run_hairgs() {
  "$conda_bin" run --no-capture-output -n hair-gs python "$@"
}

run_hairgs train.py \
  -s "$data_dir" -m "$output_dir" \
  --iterations "$iterations" --save_frequency 5000 --eval_frequency 5000 --quiet

run_hairgs merge.py -s "$data_dir" -m "$output_dir" --quiet

run_hairgs train.py \
  -s "$data_dir" -m "$output_dir" \
  --iterations "$iterations" --save_frequency 5000 --eval_frequency 5000 \
  --logger "${HAIRGS_STAGE3_LOGGER:-none}" --quiet

final_ply="$({
  find "$output_dir/point_cloud" -mindepth 2 -maxdepth 2 -type f -name point_cloud.ply \
    -printf '%h\n'
} | awk -F'iteration_' '{print $2 "\t" $0}' | sort -n | tail -1 | cut -f2-)/point_cloud.ply"

run_hairgs eval.py -s "$data_dir" -p "$final_ply" -pt gs \
  | tee "$output_dir/official_geometry_metrics.txt"
run_hairgs render.py -s "$data_dir" -m "$output_dir" --quiet

printf 'final_ply=%s\n' "$final_ply"
