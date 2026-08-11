#!/usr/bin/env bash
set -euo pipefail

GART_ROOT="${GART_ROOT:-/home/aoki/fur_hair_baselines/GART}"
GART_ENV="${GART_ENV:-/home/aoki/miniconda3/envs/gart-repro}"
DATA_ROOT="${DATA_ROOT:-/mnt/f/fur_hair_unified_data/baselines/gart}"
SEQ="${SEQ:-shiba}"
LOG_DIR="${LOG_DIR:-${DATA_ROOT}/logs/${SEQ}_official}"
SMAL_DIR="${GART_ROOT}/lib_gart/smal/smal_data"

required_assets=(
  "${SMAL_DIR}/mean_dog_bone_lengths.txt"
  "${SMAL_DIR}/new_dog_models/my_smpl_39dogsnorm_newv3_dog.pkl"
)
missing=()
for asset in "${required_assets[@]}"; do
  [[ -f "${asset}" ]] || missing+=("${asset}")
done
if ((${#missing[@]})); then
  printf 'GART requires licensed D-SMAL/BITE assets. Missing:\n' >&2
  printf '  %s\n' "${missing[@]}" >&2
  printf 'Place the legally downloaded package under %s; no substitute model is safe.\n' "${SMAL_DIR}" >&2
  exit 3
fi

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.8}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export PATH="${GART_ENV}/bin:${CUDA_HOME}/bin:${PATH}"
export PYTHONPATH="${GART_ROOT}:${PYTHONPATH:-}"

mkdir -p "${LOG_DIR}"
cd "${GART_ROOT}"
exec python solver.py \
  --profile ./profiles/dog/dog.yaml \
  --dataset dog_demo \
  --seq "${SEQ}" \
  --log_dir "${LOG_DIR}" \
  --fast
