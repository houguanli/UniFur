#!/usr/bin/env bash
set -euo pipefail

CONDA_BIN="${CONDA_BIN:-/home/aoki/miniconda3/bin/conda}"
ENV_NAME="${ENV_NAME:-dpd3dgs-animal}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  "${CONDA_BIN}" create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi

"${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip install --upgrade pip
"${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip install -e "${REPO_ROOT}[dev]"

cat <<EOF
UniFur environment ready.

  conda activate ${ENV_NAME}
  unifur --help
  pytest -q

The native HairGS rasterizer remains in the separate hair-gs environment.
EOF
