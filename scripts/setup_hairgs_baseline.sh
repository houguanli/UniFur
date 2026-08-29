#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/lib/runtime.sh"
CONDA_BIN="$(resolve_conda_bin)"
CONDA_ENV="${CONDA_ENV:-unifur}"
HAIRGS_ROOT="${HAIRGS_ROOT:-${BASELINE_ROOT:-${REPOSITORY_ROOT}/third_party/hair-gs}}"
HAIRGS_COMMIT="16588656b1f6f048bc3bc83f3cb98c2da8596754"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.8}"

if [[ "${HAIRGS_ROOT}" == "${REPOSITORY_ROOT}/third_party/hair-gs" ]]; then
  git -C "${REPOSITORY_ROOT}" submodule update --init --recursive third_party/hair-gs
fi
actual_commit="$(git -C "${HAIRGS_ROOT}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${HAIRGS_COMMIT}" ]]; then
  echo "HairGS submodule mismatch: expected ${HAIRGS_COMMIT}, got ${actual_commit}" >&2
  exit 3
fi
# DrvFS exposes every file as executable unless metadata is enabled.  Ignore
# that host-specific bit so a clean submodule remains clean on WSL-mounted
# Windows drives.
if [[ "${HAIRGS_ROOT}" == /mnt/* ]]; then
  git -C "${HAIRGS_ROOT}" config core.filemode false
fi

export PYTHONPATH="${REPOSITORY_ROOT}/compat/hairgs_sitecustomize:${HAIRGS_ROOT}:${REPOSITORY_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if ! "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
  "${CONDA_BIN}" create -y -n "${CONDA_ENV}" \
    python=3.10 pip setuptools=69.5.1 wheel ninja cmake
fi

PIP_INDEX_URL=https://pypi.org/simple PIP_EXTRA_INDEX_URL= \
  "${CONDA_BIN}" run -n "${CONDA_ENV}" python -m pip --isolated install setuptools==69.5.1
PIP_INDEX_URL=https://pypi.org/simple \
PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu118 \
  "${CONDA_BIN}" run -n "${CONDA_ENV}" python -m pip --isolated install \
  -r "${REPOSITORY_ROOT}/requirements/cuda118.txt"
PIP_INDEX_URL=https://pypi.org/simple PIP_EXTRA_INDEX_URL= \
  "${CONDA_BIN}" run -n "${CONDA_ENV}" python -m pip --isolated install \
  -r "${REPOSITORY_ROOT}/requirements/dev.txt"
"${CONDA_BIN}" run -n "${CONDA_ENV}" python -m pip --isolated install \
  --no-deps --no-cache-dir \
  -r "${REPOSITORY_ROOT}/requirements/pytorch3d-cu118.txt"

if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  echo "CUDA 11.8 toolkit not found at ${CUDA_HOME}; set CUDA_HOME explicitly" >&2
  exit 4
fi
if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
  TORCH_CUDA_ARCH_LIST="$(
    "${CONDA_BIN}" run -n "${CONDA_ENV}" python -c \
      'import torch; p=torch.cuda.get_device_capability(); print(f"{p[0]}.{p[1]}")'
  )"
fi

cd "${HAIRGS_ROOT}"
CUDA_HOME="${CUDA_HOME}" \
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
MAX_JOBS="${MAX_JOBS:-8}" \
"${CONDA_BIN}" run -n "${CONDA_ENV}" python -m pip --isolated install --no-build-isolation \
  ./c_utils ./submodules/diff-gaussian-rasterization ./submodules/simple-knn -e .

cd "${REPOSITORY_ROOT}"
"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
  python scripts/verify_reproducibility.py
"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
  python -m pip check
"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
  python -m pytest -q
