#!/usr/bin/env bash

resolve_conda_bin() {
  if [[ -n "${CONDA_BIN:-}" ]]; then
    test -x "${CONDA_BIN}" || {
      echo "CONDA_BIN is not executable: ${CONDA_BIN}" >&2
      return 2
    }
    printf '%s\n' "${CONDA_BIN}"
    return
  fi
  if command -v conda >/dev/null 2>&1; then
    command -v conda
    return
  fi
  local candidate
  for candidate in "${HOME}/miniconda3/bin/conda" "${HOME}/anaconda3/bin/conda"; do
    if [[ -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return
    fi
  done
  echo "Conda was not found; set CONDA_BIN explicitly" >&2
  return 2
}
