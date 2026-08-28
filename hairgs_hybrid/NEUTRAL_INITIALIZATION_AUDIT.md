# Common-start adaptive Shell–Fin–Strand routing

## Selected initialization

The selected preset is `configs/route_init_neutral_immediate_staged.yaml`.
Every active primitive starts with route logits `[0, 0, 0]`, no hard route
quota, no case-specific fraction, and no evidence prior.  The corresponding
soft probabilities are uniformly `1/3`; three all-zero probabilities would be
invalid because a differentiable categorical distribution must sum to one.

Route-conditioned geometry is enabled from the first training step.  For the
first 1,000 steps, reconstruction gradients may break route symmetry while
route-cost and low-entropy commitment are held back.  Temperature is annealed
from 1.5 to 0.5 over 4,000 steps.  The trained soft state is projected by
argmax to a compact hard sidecar for deployment.

## Correctness gates

- Hybrid and same-budget HairGS receive byte-identical initial PLY geometry.
- The only semantic information used at initialization is frozen base versus
  active hair membership; prior Shell/Fin/Strand proportions are ignored.
- Held-out views are not used for allocation or training.
- Offline soft rendering uses the final temperature and fully activated route
  geometry.  This prevents evaluation from silently replaying iteration zero.
- Panda uses 28 fit / 8 held-out views at 480x270.  WCurly uses 12 fit / 4
  held-out views at 500x500.  Results from the two protocols are not mixed.

## Corrected held-out ablation

| Case | Variant | FG PSNR | Full PSNR | masked LPIPS | IoU | largest FN |
|---|---|---:|---:|---:|---:|---:|
| Panda | delayed constant | 20.2695 | 27.0559 | 0.08334 | 0.94988 | 0.01131 |
| Panda | delayed staged | 20.2550 | 27.0256 | 0.08438 | 0.95031 | 0.01139 |
| Panda | **immediate staged** | **20.3109** | **27.1096** | 0.08414 | 0.95027 | 0.01292 |
| WCurly | delayed constant | 17.4492 | 17.4543 | 0.14589 | 0.83102 | 0.04776 |
| WCurly | delayed staged | 17.4907 | 17.4591 | 0.14645 | 0.83061 | 0.04728 |
| WCurly | immediate constant | 17.5013 | 17.4695 | **0.14469** | 0.83094 | 0.04682 |
| WCurly | **immediate staged** | **17.5520** | **17.4874** | 0.14580 | 0.83084 | **0.04585** |

The delayed variants suppress route-conditioned geometry for 1,000 steps, so
early reconstruction gradients cannot identify which expert should own a
primitive.  Immediate activation removes that unidentifiable interval.  The
staged regularizer is selected over immediate-constant because it gives the
best cross-case foreground/full PSNR and WCurly hole score, despite a small
LPIPS trade-off on Panda.

## Final compact-hard result versus equal-count HairGS

| Case | Method | active GS | FG PSNR | Full PSNR | masked LPIPS | IoU | largest FN | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Panda | HairGS | 22,000 | 19.7876 | 26.0876 | 0.08600 | 0.93038 | 0.00656 | 276.98 |
| Panda | **UniFur common-start hard** | 22,000 | **20.2831** | **27.0918** | **0.08426** | **0.95008** | 0.01292 | 269.63 |
| WCurly | HairGS strict | 6,000 | 17.4607 | 17.4833 | 0.14723 | 0.82906 | 0.06770 | 252.28 |
| WCurly | **UniFur common-start hard** | 6,000 | **17.5458** | **17.4916** | **0.14569** | **0.83074** | **0.04587** | **262.40** |

At equal active count, Panda gains 0.495 dB foreground PSNR and 1.004 dB full
PSNR.  WCurly gains 0.085 dB foreground PSNR; the larger result is structural:
its largest false-negative component falls by 32.25%.

## Previous type-based initialization

The preceding version did not share a common start.  It compressed a learned
dense teacher and used train-view route contribution to assign a fixed initial
quota:

| Case | initialization | Shell | Fin | Strand |
|---|---|---:|---:|---:|
| Panda | previous fixed type quota | 17,619 (80.09%) | 3,168 (14.40%) | 1,213 (5.51%) |
| Panda | common-start final hard | 7,126 (32.39%) | 7,795 (35.43%) | 7,079 (32.18%) |
| WCurly | previous fixed type quota | 3,388 (56.47%) | 1,351 (22.52%) | 1,261 (21.02%) |
| WCurly | common-start final hard | 2,917 (48.62%) | 1,838 (30.63%) | 1,245 (20.75%) |

Both common-start runs began from zero logits; the second and fourth rows are
learned final ownership rather than initialization hyperparameters.

| Case | hard model | FG PSNR | Full PSNR | masked LPIPS | IoU | largest FN | FPS |
|---|---|---:|---:|---:|---:|---:|---:|
| Panda | previous type-based | 20.2321 | 26.9843 | 0.08440 | **0.95019** | 0.01409 | 262.78 |
| Panda | **common-start** | **20.2831** | **27.0918** | **0.08426** | 0.95008 | **0.01292** | **269.63** |
| WCurly | **previous type-based** | **17.5707** | **17.5378** | **0.14484** | **0.83190** | **0.04018** | 244.65 |
| WCurly | common-start | 17.5458 | 17.4916 | 0.14569 | 0.83074 | 0.04587 | **262.40** |

Panda improves slightly after removing the type-specific quota: +0.051 dB
foreground and +0.108 dB full PSNR, with an 8.25% smaller largest hole. WCurly
gives back 0.025 dB foreground PSNR and its largest hole grows by 14.2%, while
render FPS improves by 7.25%. Therefore the common start matches or slightly
improves the previous prior on Panda and is practically close—but not better—
on WCurly. Its justification is cross-case adaptivity, not universal quality
dominance over a hand-informed start.

Common-start soft training is also more expensive than the previous direct
hard route training: 285.3 versus 198.7 seconds on Panda and 298.1 versus 226.8
seconds on WCurly. Compact-hard deployment size remains effectively unchanged.

## Dense-resource reference

Panda has a strict native dense HairGS reference with 96,991 primitives.  The
22k hard model reduces active points by 77.3%, deployment bytes by 77.1%, and
render peak allocation by 42.6%, while increasing FPS from 225.33 to 269.63.
Its foreground PSNR is only 0.041 dB below dense HairGS, although masked LPIPS
is 0.0132 worse.  Soft routing training takes 285.3 s versus 310.5 s for dense
HairGS and therefore does **not** establish a large training-time advantage;
the strong claim is deployment/storage/render cost.

WCurly's full native HairGS baseline is the final 15k Stage-3 checkpoint, not
the intermediate Stage-2 teacher. It contains 107,610 active hair segments and
135,141 total segments. The compact model uses 6,000 active and 35,278 total:
active count falls 94.4%, total count falls 73.9%, deployment bytes fall 62.6%,
FPS rises from 232.40 to 262.40, and render peak allocation falls from 107.73
MiB to 67.29 MiB. Its measured 5k common-start optimization takes 298.1 seconds
versus 1,376.2 seconds for the full 15k Stage-3 span, a 78.3% reduction.

## Full-baseline quality proximity

| Case | model | FG PSNR | Full PSNR | masked SSIM | masked LPIPS | IoU | largest FN |
|---|---|---:|---:|---:|---:|---:|---:|
| Panda | common-start 22k | 20.2831 | **27.0918** | 0.92389 | 0.08426 | **0.95008** | 0.01292 |
| Panda | full HairGS 96,991 | **20.3245** | 26.3744 | **0.93220** | **0.07105** | 0.92926 | **0.00749** |
| WCurly | common-start 6k | **17.5458** | 17.4916 | **0.87350** | 0.14569 | 0.83074 | 0.04587 |
| WCurly | full HairGS 107,610 | 16.8499 | **17.5558** | 0.85711 | **0.13087** | **0.85413** | **0.03272** |

The compact models are close to full HairGS in PSNR: Panda foreground differs
by -0.041 dB, while WCurly has +0.696 dB foreground and -0.064 dB full PSNR.
They are **not** fully equivalent in all dimensions. Panda masked LPIPS is
0.0132 worse; WCurly masked LPIPS is 0.0148 worse, IoU is 0.0234 lower, and the
largest hole is 0.0131 larger. Visually, full HairGS retains finer/denser fiber
detail, whereas common-start hard routing suppresses more background overflow.

Training views show that this is not full reconstruction equivalence:

| Case | model | train FG PSNR | train Full PSNR | train masked LPIPS | train IoU |
|---|---|---:|---:|---:|---:|
| Panda | common-start 22k | 21.2040 | 27.5778 | 0.08049 | **0.95030** |
| Panda | full HairGS 96,991 | **22.5596** | **28.0167** | **0.05796** | 0.93691 |
| WCurly | common-start 6k | 22.0350 | 16.7204 | 0.07863 | 0.86148 |
| WCurly | full HairGS 107,610 | **28.9437** | **16.9917** | **0.01206** | **0.91276** |

Panda loses 1.36 dB and WCurly loses 6.91 dB training foreground PSNR. The
WCurly result is best interpreted as strong capacity regularization: it gives
better held-out foreground PSNR and lower overflow with far fewer primitives,
but cannot fit the full baseline's high-frequency hair detail. A claim of
"near-full novel-view PSNR at much lower deployment cost" is supported; a
claim of "full-baseline reconstruction fidelity" is not.

## Interpretation limit

The learned hard allocations differ despite identical logits: Panda ends at
7,126 Shell / 7,795 Fin / 7,079 Strand, whereas WCurly ends at 2,917 / 1,838 /
1,245.  This proves case-adaptive allocation, but it does not by itself prove
perfect semantic purity of each route.  The route labels remain functional
covariance experts whose meaning is constrained by their allowed geometry and
cost, not by species-level target ratios.
