# Panda topology-router milestone

## Scope

This milestone tests whether Shell, Fin, and Strand can be learned as different
Gaussian topologies under a fixed deployment budget.  It uses one Panda seed,
28 fit views, 8 held-out views, 480x270 images, fixed calibrated cameras, and
the same evaluator for every reported model.

The goal is not to beat HairGS simultaneously in optimization time, every
perceptual metric, and reconstruction quality.  The validated goal is:

1. use a common initialization with no case-specific route quota;
2. learn spatially meaningful topology ownership;
3. improve over an equal-count HairGS control;
4. remain close to full HairGS at much lower deployment cost.

## Representation

The dense HairGS scaffold contains 96,991 source line-Gaussian segments.  They
are partitioned into 10,000 balanced spatial groups; each group contains 9 or
10 source segments.  Each group generates three mutually exclusive candidate
representations:

| route | candidate count per group | role |
|---|---:|---|
| Shell | 1 | aggregated low-cost regional coverage |
| Fin | 2 | intermediate silhouette and transition support |
| Strand | 4 | higher-capacity directional/high-frequency detail |

The differentiable bank therefore contains 70,000 candidate splats.  Its
opacity, color, orientation, position, and scale are initialized by
optical-mass-aware aggregation of the source segments.  The three route logits
of every group are initialized to `[0, 0, 0]`; no Panda-specific proportion is
injected.

During soft routing, candidate opacity is gated by the group route probability.
The budget objective uses the real route costs `[1, 2, 4]`, so it estimates the
number of active splats rather than a heuristic label cost.  Straight-through
hard forwarding exposes the chosen topology during the later optimization
phase while retaining gradients for route logits.

## Hard projection and refinement

Hard projection solves the learned route choice under an exact 22,000-splat
budget.  A Lagrangian selection is followed by exact budget repair.  Inactive
candidates are physically removed from the PLY and state sidecar; reported
deployment size and rendering speed never count the 70k training bank.

The selected topology is subsequently frozen and refined for 3,000 compact
steps.  Route entropy, route budget, and route exploration losses are disabled
in this phase.  Rendering variables of the retained splats remain trainable,
including position, scale/orientation, opacity, and spherical-harmonic color.

## Learned allocation

The final group ownership is:

| route | groups | group share | deployed splats | splat share |
|---|---:|---:|---:|---:|
| Shell | 4,470 | 44.70% | 4,470 | 20.32% |
| Fin | 2,295 | 22.95% | 4,590 | 20.86% |
| Strand | 3,235 | 32.35% | 12,940 | 58.82% |
| total | 10,000 | 100% | 22,000 | 100% |

The difference between group share and splat share is intentional: Strand has
four times the local capacity of Shell.

## Strict held-out results

| Method | splats | FG PSNR | Full PSNR | masked PSNR | masked SSIM | masked LPIPS | IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| HairGS full | 96,991 | **20.3245** | 26.3744 | **28.0173** | **0.9322** | **0.0711** | 0.9293 |
| HairGS equal-count | 22,000 | 19.7876 | 26.0876 | 27.4804 | 0.9235 | 0.0860 | 0.9304 |
| previous common-start UniFur | 22,000 | 20.2831 | **27.0918** | 27.9759 | 0.9239 | 0.0843 | **0.9501** |
| topology-routed, direct hard | 22,000 | 18.5328 | 25.6287 | 26.2256 | 0.9031 | 0.1138 | 0.9472 |
| **topology-routed + refinement** | **22,000** | **20.0893** | 26.9131 | 27.7821 | 0.9203 | 0.0907 | 0.9442 |
| route-shuffle control | 22,000 | 15.5609 | -- | -- | -- | -- | -- |

The final model gains +0.302 dB foreground and +0.825 dB full PSNR over
equal-count HairGS.  Relative to full HairGS, foreground PSNR differs by
-0.235 dB and deployment bytes fall from 26,771,158 to 6,109,329 (-77.18%).
Measured rendering throughput changes from 209.0 FPS for full HairGS and 277.0
FPS for equal-count HairGS to 250.7 FPS for topology-routed UniFur.

The full soft mixture reaches 21.420 dB held-out foreground PSNR, but its mean
maximum route probability is only 0.392.  It is therefore reported only as a
candidate-bank capacity diagnostic and never as a deployable result.

## Attribution controls

The route-shuffle control keeps:

- exactly 22,000 splats;
- the same Shell/Fin/Strand histogram;
- the same rendering and evaluation protocol.

It changes only which spatial group owns each topology.  Foreground PSNR falls
from 20.089 to 15.561 dB, a 4.528 dB drop.  Therefore the result cannot be
explained by point count or a favorable global route ratio alone.

Straight-through training reduces the discrepancy between the soft model's
hard-forward render and the physically projected checkpoint to about 0.25 dB.
Its remaining low quality is an optimization problem: candidates selected late
receive intermittent appearance and geometry updates.  Fixed-topology
refinement recovers most of the lost quality.

## Resource interpretation

| Method | deploy bytes | FPS | measured optimization wall |
|---|---:|---:|---:|
| HairGS full | 26,771,158 | 209.0 | 310 s |
| HairGS equal-count | 6,073,641 | **277.0** | **185 s** |
| topology-routed + refinement | 6,109,329 | 250.7 | 574 s |

The topology model passes the storage and near-full-quality gates, but not a
training-efficiency gate.  This is acceptable for the current representation
milestone and must be stated explicitly in any paper draft.

## Reproduction

```bash
# Build bank, train router, project exact-budget models, render, and evaluate.
bash hairgs_hybrid/run_panda_topology_router.sh

# Refine the physically compact topology and evaluate train/test views.
bash hairgs_hybrid/refine_panda_hard_topology.sh

# Unit and integration-level topology checks.
python -m pytest -q tests/test_topology_router.py \
  tests/test_hairgs_inference_config.py
```

Generated datasets, checkpoints, renders, and evaluation outputs stay outside
the repository.  The runners expose environment variables for all external
paths and refuse to start duplicate GPU training.

## Claim boundary

- The result establishes functional and spatial topology allocation, not
  perfect physical semantics for every route.
- Shell, Fin, and Strand are still Gaussian representations, not literal shell
  volumes, ribbon meshes, or continuous strand graphs.
- The full HairGS model remains better in masked SSIM/LPIPS and fine fiber
  detail.  “Near-full held-out PSNR at 77% lower storage” is supported;
  “lossless compression” is not.
- Training-time superiority is not supported.
- Only one requested Panda seed is reported, so no variance claim is made.
