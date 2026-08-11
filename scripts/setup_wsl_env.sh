#!/usr/bin/env bash
set -euo pipefail

CONDA_BIN="${CONDA_BIN:-/home/aoki/miniconda3/bin/conda}"
ENV_NAME="${ENV_NAME:-dpd3dgs-animal}"
SAM_ENV="${SAM_ENV:-sam3d-objects}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ELASTIC_ROOT="${ELASTIC_ROOT:-$REPO_ROOT/third_party/elastic_simulator}"

if ! "$CONDA_BIN" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  "$CONDA_BIN" create --name "$ENV_NAME" --clone "$SAM_ENV" -y
fi

"$CONDA_BIN" run -n "$ENV_NAME" python -m pip install \
  diffusers==0.32.2 \
  taichi==1.7.3 \
  tetgen==0.8.4 \
  diso==0.1.4

"$CONDA_BIN" run -n "$ENV_NAME" python -m pip install -e "$ELASTIC_ROOT"
"$CONDA_BIN" run -n "$ENV_NAME" python -m pip install -e "$REPO_ROOT"

if [[ -d "$ELASTIC_ROOT/.git" ]]; then
  git -C "$ELASTIC_ROOT" submodule update --init --recursive
fi

cat <<EOF
Environment ready.
Activate with:
  conda activate $ENV_NAME

Runtime PYTHONPATH for MocapAnything:
  export PYTHONPATH="$REPO_ROOT/third_party/mocap_anything:$REPO_ROOT/third_party/mocap_anything/TripoSG:\${PYTHONPATH:-}"
EOF
