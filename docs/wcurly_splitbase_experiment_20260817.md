# wCurly split-base experiment (2026-08-17)

## Protocol

- Dataset: wCurly static multiview.
- Fit views: 12; held-out novel views: 4.
- Resolution: 1000 x 1000.
- Hair scaffold: 95,150 learnable hair sources.
- Fixed base: 29,315 immutable head/body Gaussians used for depth compositing only.
- Fixed-base maximum scale: 0.04 of the rest-surface scene diagonal.
- Structured coverage seeds: 2,500 shell and 2,500 strand sources.
- Optimisation: 6,000 steps with four-view leave-one-view-out topology validation.

The fixed base has no trainable parameters, does not enter route probabilities,
and is not serialized in the field state dict. Evaluation contact sheets show a
hair-layer diagnostic separately from the joint composite and paint the fixed
base gray in route maps.

## Comparable results

All rows below use the same fixed-base scale cap and rendering protocol.

| Split | Method | Foreground PSNR | Masked SSIM | Masked LPIPS | Mask IoU |
|---|---|---:|---:|---:|---:|
| train12 | fixed residual teacher | 24.2042 | 0.953385 | 0.045602 | 0.921796 |
| train12 | UniFur v2 soft | 23.8986 | 0.951993 | 0.046932 | 0.922265 |
| train12 | UniFur v2 hard | 23.6487 | 0.951315 | 0.048339 | 0.922683 |
| novel4 | fixed residual teacher | 15.5076 | 0.855458 | 0.137293 | 0.848978 |
| novel4 | UniFur v2 soft | 15.5175 | 0.855356 | 0.136483 | 0.848175 |
| novel4 | UniFur v2 hard | 15.5169 | 0.855456 | 0.136440 | 0.848554 |

Relative to the capped teacher, the soft model gains 0.0099 dB foreground PSNR
and improves masked LPIPS by 0.00081 on novel views, while losing 0.3056 dB on
fit views. This is a stable structured representation, not a large rendering
improvement.

The preceding v1 checkpoint evaluated with the same cap obtains 15.5190 dB on
novel views. The 0.0015 dB difference from v2 is negligible; v2 is selected as
the reproducible checkpoint because the cap is stored in training metadata and
automatically restored by evaluation.

## Routing and topology result

- Final soft route mass: shell 0.4925%, strand 0.9745%, residual 98.5330%.
- Final hard route mass: shell 0.4929%, strand 0.9743%, residual 98.5328%.
- Nine prune/grow/densify proposals were tested.
- Every proposal was rejected and reverted because it worsened at least one of
  the four calibration views.
- Final topology remains 95,150 residual, 2,500 shell, and 2,500 strand sources.

## Conclusion

The physical split fixes the semantic bug: head/body geometry is no longer a
learnable hair route and cannot consume the shell/strand allocation. Fit-view
hair reconstruction is coherent after the split. The low novel-view result is
still inherited from the Stage-1 hair scaffold: oversized/anistropic residual
hair Gaussians and incomplete multiview geometry already appear in the fixed
teacher. Post-hoc clipping of all hair Gaussians was tested and reduced PSNR,
so the next useful change is to rebuild the Stage-1 hair scaffold with
hair-only masks, multiview scale/anisotropy regularisation, and held-out-view
densification checks rather than adding more UniFur optimisation steps.

## Artifacts

- Model: `F:/fur_hair_unified_data/benchmarks/hairgs_wcurly_static_results/splitbase_unifur_hair_full6k_basecap004_v2/unified_fiber_field.pt`
- Train soft evaluation: `F:/fur_hair_unified_data/benchmarks/hairgs_wcurly_static_results/splitbase_unifur_hair_full6k_basecap004_v2_eval_train12_soft/evaluation.json`
- Novel soft evaluation: `F:/fur_hair_unified_data/benchmarks/hairgs_wcurly_static_results/splitbase_unifur_hair_full6k_basecap004_v2_eval_test4_soft/evaluation.json`
- Reproduction script: `scripts/evaluate_splitbase_basecap_v2.sh`
