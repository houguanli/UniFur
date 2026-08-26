# HairGS Hybrid Covariance Experiment

This directory implements the first auditable increment of the HairGS-based
Shell–Fin–Strand design.  It keeps the trained HairGS endpoints, topology, SH,
opacity, and hair mask, and expands each segment from the native
`(length, width, width)` covariance to a differentiable three-route covariance:

- **strand**: native line Gaussian;
- **fin**: tangent-aligned ribbon with broad binormal and thin normal;
- **shell**: tangent-plane patch with broad tangent/binormal and thin normal.

The default soft router, shape strength, scalp-frame roll residual, and native
HairGS hair attributes are optimized together.  The audited deployment path can
instead use `--hybrid-route-mode hard`: fixed ownership is converted to one-hot
Shell/Fin/Strand routing, only route-effective shape/roll variables remain
trainable, and scale plus rotation are constructed once per optimizer state and
shared by the HairGS rasterizations.  This is a fused **state-construction and
cache path**, not a replacement CUDA rasterizer.

Head/base primitives are excluded using
the exact route sidecar and are frozen at both endpoint and per-segment level.
They remain in the renderer only as immutable occluders/background for a valid
full composite. Topology is intentionally frozen in this increment so quality
changes are attributable to the representation rather than birth/prune events.

`run_wcurly_haironly_maskbounded5k.sh` is the strict hair-only training path. It
verifies the binary hair masks and their hashes before training, masks RGB/SSIM
gradients to hair pixels, renders exact hair-route visibility behind the frozen
head, and applies separate FP, FN, largest-hole, and largest-overflow terms.
`run_wcurly_hybrid_calibration.sh` provides a short frozen-route calibration for
the coverage/overflow tradeoff.

## Correctness and evaluation flow

1. `check_equivalence.py` must show an all-strand maximum pixel error at or below
   `1e-5` before training.
2. `prepare_hair_mask_manifest.py` verifies that all train/test masks are
   hair-only, binary, correctly sized, and basename-aligned.
3. `run_wcurly_haironly_maskbounded5k.sh` starts from the same HairGS Stage-2
   PLY and runs the frozen train12/test4, 500x500 protocol. Head/body fields have
   explicit zero-gradient hooks and are checked bitwise after training by
   `audit_frozen_base.py`.
4. Hair alpha uses exact route-sidecar membership rather than a learnable mask
   threshold. RGB/SSIM gradients exist only inside the GT hair mask; FP, FN,
   largest-hole, and largest-overflow terms supervise its two boundaries.
5. `run_hybrid_ablation_audit.sh` disables primitives by learned hard ownership
   and evaluates every leave-one-route-out model on the same held-out views.
6. `audit_hybrid.py` validates protocol identity, same-budget quality, largest
   FP/FN components, route contribution, structure, time, and storage.
7. `make_haironly_comparison.py` creates a focused audit sheet containing GT,
   previous/new composites, exact hair alpha, and red-FP/cyan-FN overlays.

The native PLY remains compatible with upstream HairGS.  Hybrid parameters are
stored in an adjacent `hybrid_state.npz`; both files are required to reproduce a
hybrid render. Soft checkpoints retain the dense v1 state. Hard checkpoints use
the compact v2 codec: 2-bit packed ownership, FP16 route-effective strengths,
and FP16 non-strand roll only. Dense logits, inactive route parameters, and
redundant index arrays are omitted. Both v1 and v2 can be loaded by the hard
runtime, which also makes soft-to-hard deployment explicit and reproducible.

## Audited hard-route deployment

The fixed-budget knees were rerun for 5,000 steps from the same compressed PLYs
and training-only route allocations:

- Panda 22k: 198.7 s training, 262.8 FPS, 75,677-byte sidecar; held-out FG/Full
  PSNR 20.232/26.984 dB. Relative to soft routing this is 26.2% less training
  time, 22.9% more FPS, and 84.0% less sidecar storage for -0.056/-0.066 dB.
- WCurly 6k hair + 29,278 frozen base: 226.8 s, 244.7 FPS, 19,301-byte sidecar;
  held-out FG/Full PSNR 17.571/17.538 dB. Relative to soft routing this is 22.9%
  less training time, 23.2% more FPS, and 85.7% less sidecar storage, while both
  PSNR values are slightly higher.

The hard renderer is within 3--5% of same-budget HairGS FPS and its total model
storage is within 0.5--1.3%. Observed training GPU peaks did not improve
consistently, so this change supports latency/time/storage efficiency, not a
VRAM-reduction claim. Machine-readable reports are stored under each result
root as `audit/hard_route_comparison.json`.

## Background-radiance calibration

Hair/fur RGB supervision inside a binary mask does not constrain bright,
low-opacity Gaussian tails outside that mask. Hybrid training therefore
supports `--lambda-hybrid-background-rgb`. It compares the full render with a
cached base-only render outside the hair mask. Panda has no base and uses a
black reference; human-hair cases retain the frozen head/body reference. This
isolates route-induced spill without training hair Gaussians to repair the
head.

The compression sweep keeps the compressed PLY byte-identical between hybrid
and native controls at each budget. Consequently a quality delta measures the
Shell/Fin/Strand parameterization rather than a different spatial sample.
Run `run_adaptive_compression_sweep.sh` to reproduce the audited Panda
25k/16k/10k and WCurly 12k/8k/5k study.  The measured compression knee and all
quality/efficiency trade-offs are recorded in `COMPRESSION_SWEEP_REPORT.md`.

## Adaptive fixed-budget Shell--Fin--Strand compression

`adaptive_compress_scaffold.py` converts a learned dense HairGS hybrid teacher
to an exact immutable segment budget. It uses no species label and never reads
held-out images. Allocation is inferred from three sources:

- the teacher's learned soft route probabilities;
- route evidence (interior density for shell, multi-view training silhouette
  boundary for fin, and length/local direction frequency for strand);
- positive leave-one-route-out contribution measured on training views only.
  A route must help across views to obtain capacity; a non-positive route keeps
  only the minimum exploration budget.

The default `shared-coverage` policy first builds a route-neutral,
optical-density-preserving exact-N scaffold. The hybrid and native control PLYs
are byte-identical at initialization. Shell/fin/strand ownership is then
assigned on this common support by interpolated utility. This is essential:
global route-specific top-k selection was measured to create avoidable spatial
coverage differences on WCurly. The old behavior remains available only through
`--geometry-policy route-specialized` as an ablation.

`run_train_route_contribution_audit.sh` creates the train-only hard-route
ablations. `summarize_route_contributions.py` enforces matching training frames
and emits the signed/positive cross-view allocation report.

For WCurly, two controls must be reported separately. The upstream/standard
HairGS control updates both head and hair segments with its full-frame loss. The
strict capacity control freezes the same 29,278 head/base segments and trains
only the same 12,000 hair segments with the same hair-only loss, using
`VARIANT=all_strand`. Only the latter isolates the cost of the adaptive
covariance router.

The default Panda mask calibration is bidirectional (`FN=0.60`, `FP=0.60`,
`hole=0.20`, `overflow_peak=0.60`).  A quality-first setting can be reproduced
with `LAMBDA_MASK_FN=1.0 LAMBDA_MASK_FP=0.35 LAMBDA_HOLE=0.30
LAMBDA_OVERFLOW_PEAK=0.35`; it improves masked reconstruction more strongly but
allows a softer silhouette.

## Scope limit

This version validates route-conditioned anisotropic Gaussian covariance and
hair-only/frozen-base semantics on a fixed HairGS scaffold. A 2D mask loss cannot
hard-clip unseen views without making the representation camera-dependent, so
held-out FP/FN remain evaluation criteria rather than guaranteed zeros. This is
not yet an implementation of independent scalp shell topology, explicit fin
strips, or transactional topology birth/pruning.
