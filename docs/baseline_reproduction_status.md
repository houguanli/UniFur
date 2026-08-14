# Baseline reproduction status

This file records which outputs are admissible for UniFur comparisons.  RGB
metrics are only compared when method, ground truth and render manifests share
the same held-out cameras and resolution.  Explicit-geometry methods without a
held-out appearance renderer are reported in a separate structural protocol.

## Hair person0: odd33 fit / even33 held out / 1024 square

### Hair-GS

- Previous `odd33_nativehead_30k30k` is invalid and retained for diagnosis only.
- Root causes: room RGB was not alpha-matted, GaussianHaircut orientation
  variance was used as Hair-GS confidence, and UniFur's inferred scalp vertices
  replaced Hair-GS' FLAME scalp mask.
- Repaired protocol: `hairgs_person0_protocol_official_v2`.
- Required gates: black RGB background; released Gabor confidence estimator;
  official `FLAME_masks.pkl/scalp`; strict camera/head/scalp projection audit;
  successful held-out preflight and at least 300 Stage-I merge endpoints.
- The 24 GB run keeps the official 30k + 30k optimization budget, while ending
  Stage-I topology growth at 10k to avoid density explosion.
- Repaired final result (`odd33_officialpreproc_v2_30k30k`): foreground PSNR
  15.838, masked PSNR 20.911, masked SSIM 0.771, LPIPS 0.237 and mask IoU
  0.599.  The invalid run was 11.039 / 16.112 / 0.745 / 0.259 / 0.196.
- Stage-I merge found 479 roots from the 489-vertex official scalp region.

### GaussianHaircut

- Completed as a released-scaffold-conditioned, fixed-camera 24 GB adapter in
  `gaussian_haircut_person0_results/fixedcam_25k_densify10k`.
- Stage I uses 25k steps with topology fixed after 10k, latent fitting uses 20k,
  and curve fitting uses 5k strands for 10k steps.
- The public 30k-curve configuration is retained as
  `stage3_curves_native30k_intractable_step14`; at 1024 square it required about
  12 minutes per step on the 24 GB card and is computationally intractable.
- It is a capacity adapter, not a native-capacity claim.

### Im2Haircut

- Official example and the person0 frame-0049 scaffold-conditioned run complete.
- The public method predicts explicit strands but no novel-view RGB appearance,
  so no PSNR is invented.
- `evaluate_im2haircut_person0_structure.py` measures even33 strand occupancy,
  visual-hull consistency and projected orientation.  This is a separate
  single-image-plus-released-scaffold structural protocol.
- Person0 frame-0049 result: projected mask IoU 0.644, visual-hull sample
  inlier ratio 0.861, mean orientation error 46.57 degrees and 15.35 percent of
  segments within 15 degrees.  The coarse hair volume transfers, strand flow
  does not.
- A fully automatic raw-image-only run remains blocked by the licensed BFM09
  asset required by the public Deep3DFaceRecon preprocessing.  The local
  scaffold adapter is reported transparently and is not called pure single view.

## Fur Panda: official 28 fit / 8 held out / 480 x 270

### NeuralFur

- Previous `neuralfur_4k_full20k_lrbody_r512` is invalid for the common table:
  supervision used `-r 1` while camera intrinsics/rasterization used scale 4.
- Repaired runner uses `-r 4`, `scale_factor=4` and a 480 x 270 raster together.
- `run_neuralfur_panda_preflight.sh` checks camera size, body/fur visibility,
  held-out RGB and mask overlap before the full 20k run.
- The repaired 500-step preflight reached held-out foreground IoU 0.881 with
  all 11,960 body points and 396,000 fur segment Gaussians visible.  Its low RGB
  PSNR is expected at this stage because the released `furless_lr.obj` carries
  no texture; the full appearance run must be reported separately.
- The local 4k-active/100k-candidate setting is a declared 24 GB capacity
  adapter; the released 15k/500k setting OOMs on this card.
