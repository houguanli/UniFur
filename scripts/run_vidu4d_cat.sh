#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
VIDU4D_ROOT="${VIDU4D_ROOT:-/home/aoki/fur_hair_baselines/Vidu4D}"
VIDU4D_ENV="${VIDU4D_ENV:-/home/aoki/miniconda3/envs/vidu4d-repro}"
DATA_ROOT="${DATA_ROOT:-/mnt/f/fur_hair_unified_data/baselines/vidu4d}"
SEQNAME="${SEQNAME:-cat-local-controlled}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.8}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export TORCH_HOME="${TORCH_HOME:-${DATA_ROOT}/torch_cache}"
export PATH="${VIDU4D_ENV}/bin:${CUDA_HOME}/bin:${PATH}"
export PYTHONPATH="${VIDU4D_ROOT}:${PYTHONPATH:-}"

cd "${VIDU4D_ROOT}"

case "${MODE}" in
  smoke)
    # Minimum legal schedule: OneCycleLR uses two warm-up rounds, so 1-2 fail.
    exec python lab4d/train.py \
      --seqname "${SEQNAME}" \
      --logname base-smoke-r3 \
      --fg_motion bob \
      --num_rounds 3 \
      --save_freq 1 \
      --rgb_timefree \
      --rgb_dirfree
    ;;
  stage2)
    exec python lab4d/train.py \
      --seqname "${SEQNAME}" \
      --logname base \
      --fg_motion bob \
      --num_rounds 21 \
      --rgb_timefree \
      --rgb_dirfree
    ;;
  stage3)
    BASE_DIR="logdir/${SEQNAME}-base"
    [[ -f "${BASE_DIR}/ckpt_0020.pth" ]] || {
      echo "Missing ${BASE_DIR}/ckpt_0020.pth; finish Stage 2 first." >&2
      exit 2
    }
    [[ -f "${BASE_DIR}/021-fg-geo.obj" ]] || {
      echo "Missing ${BASE_DIR}/021-fg-geo.obj; finish Stage 2 export first." >&2
      exit 2
    }
    exec python lab4d/train.py \
      --seqname "${SEQNAME}" \
      --logname gs-frzwarp \
      --fg_motion gs-bob \
      --num_rounds 61 \
      --load_path "${BASE_DIR}/ckpt_0020.pth" \
      --gs_init_mesh "${BASE_DIR}/021-fg-geo.obj" \
      --imgs_per_gpu 1 \
      --pixels_per_image -1 \
      --eval_res 256 \
      --rgb_timefree \
      --rgb_dirfree \
      --rgb_loss_only \
      --gs_optim_warp=False \
      --data_prefix full \
      --force_center_cam
    ;;
  *)
    echo "Usage: $0 {smoke|stage2|stage3}" >&2
    exit 2
    ;;
esac
