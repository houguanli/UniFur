#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-status}"
REPO_ROOT="${REPO_ROOT:-/home/aoki/Differentiable_Physical_Driven_3DGS_for_monocular_Animal_Motion_Reconstruction}"
PYTHON="${PYTHON:-/home/aoki/miniconda3/envs/hair-gs/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/f/fur_hair_unified_data/benchmarks/dfa_panda_walk_dual}"
RESULT_ROOT="${RESULT_ROOT:-/mnt/f/fur_hair_unified_data/benchmarks/dfa_panda_walk_dual_results}"
NEURALFUR_PANDA_ROOT="${NEURALFUR_PANDA_ROOT:-/mnt/f/fur_hair_unified_data/neuralfur_official/extracted/data/Artemis/panda_processed_GH2/walk}"
STEPS="${STEPS:-20000}"
RENDER_WIDTH="${RENDER_WIDTH:-512}"
RENDER_HEIGHT="${RENDER_HEIGHT:-288}"

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

prepare() {
  "$PYTHON" "$REPO_ROOT/scripts/prepare_dfa_panda_dual_protocol.py" \
    --output-root "$DATA_ROOT" \
    --neuralfur-panda-root "$NEURALFUR_PANDA_ROOT" \
    --frame-count 32 \
    --temporal-train-count 24 \
    --width 960
}

restore_hairgs_rasterizer() {
  "$PYTHON" -m pip install --no-build-isolation --force-reinstall --no-deps \
    "/home/aoki/fur_hair_baselines/hair-gs/submodules/diff-gaussian-rasterization"
}

run_setting() {
  local setting="$1"
  local fit_split
  case "$setting" in
    mono) fit_split="train_mono_t32" ;;
    mv4) fit_split="train_mv4_t32" ;;
    mv8) fit_split="train_mv8_t32" ;;
    *) echo "Unknown setting: $setting" >&2; return 2 ;;
  esac

  local stage1="$DATA_ROOT/dfa_panda_walk_matrix_lbs_stage1.npz"
  # Neutral template points avoid leaking the released all-view 30k Gaussian
  # appearance into either the monocular or multi-view dynamic leaderboard.
  local gaussians="$DATA_ROOT/initial_neutral_template_gaussians_20k.ply"
  local fit_dir="$DATA_ROOT/$fit_split/images"
  local fit_manifest="$DATA_ROOT/$fit_split/camera_manifest.json"
  local test_dir="$DATA_ROOT/test_novel_v8_t32/images"
  local test_manifest="$DATA_ROOT/test_novel_v8_t32/camera_manifest.json"

  for method in residual unified; do
    local config="$REPO_ROOT/configs/fiber_dfa_panda_dynamic_${method}.yaml"
    local train_out="$RESULT_ROOT/${setting}_${method}_20k"
    local eval_out="$RESULT_ROOT/${setting}_${method}_20k_eval_novel_v8"
    mkdir -p "$train_out" "$eval_out"
    if [[ ! -f "$train_out/unified_fiber_field.pt" ]]; then
      "$PYTHON" -m dpd3dgs_animal.cli --config "$config" fiber-stage2 \
        --stage1-npz "$stage1" \
        --gaussian-ply "$gaussians" \
        --frame-dir "$fit_dir" \
        --camera-manifest "$fit_manifest" \
        --out-dir "$train_out" \
        --steps "$STEPS" \
        --renderer hairgs \
        --render-width "$RENDER_WIDTH" \
        --render-height "$RENDER_HEIGHT" \
        2>&1 | tee "$train_out/stdout.log"
    fi
    "$PYTHON" -m dpd3dgs_animal.cli --config "$config" fiber-eval \
      --stage1-npz "$stage1" \
      --gaussian-ply "$gaussians" \
      --checkpoint "$train_out/unified_fiber_field.pt" \
      --frame-dir "$test_dir" \
      --camera-manifest "$test_manifest" \
      --out-dir "$eval_out" \
      --renderer hairgs \
      --route-mode soft \
      --render-width "$RENDER_WIDTH" \
      --render-height "$RENDER_HEIGHT" \
      --export-external-renders \
      2>&1 | tee "$eval_out/stdout.log"
    "$PYTHON" "$REPO_ROOT/scripts/evaluate_external_renders.py" \
      --render-manifest "$eval_out/external_render_manifest.json" \
      --ground-truth-dir "$test_dir" \
      --output-dir "$eval_out/external_evaluation" \
      --method "internal_${method}" \
      --protocol-id "DFA-Panda-Walk-32f-v1-${setting}-internal"
    "$PYTHON" "$REPO_ROOT/scripts/summarize_dual_input_benchmark.py" \
      --dynamic-root "$RESULT_ROOT"
  done
}

case "$MODE" in
  prepare)
    prepare
    ;;
  mono|mv4|mv8)
    [[ -f "$DATA_ROOT/protocol.json" ]] || prepare
    restore_hairgs_rasterizer
    run_setting "$MODE"
    ;;
  all)
    [[ -f "$DATA_ROOT/protocol.json" ]] || prepare
    restore_hairgs_rasterizer
    run_setting mono
    run_setting mv4
    run_setting mv8
    ;;
  status)
    if [[ -f "$DATA_ROOT/protocol.json" ]]; then
      echo "data=ready"
    else
      echo "data=missing"
    fi
    find "$RESULT_ROOT" -maxdepth 2 -type f \
      \( -name evaluation.json -o -name unified_fiber_report.json \) \
      -print 2>/dev/null | sort || true
    ;;
  *)
    echo "Usage: $0 {prepare|mono|mv4|mv8|all|status}" >&2
    exit 2
    ;;
esac
