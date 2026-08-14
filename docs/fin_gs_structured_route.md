# Fin-GS structured route

## Why this is not a fourth free residual expert

UnityFurURP's Fin renderer emits thin fur cross-sections only for triangles
near a grazing view. Its geometry shader tests the absolute dot product of the
view direction and face normal, extrudes a segmented fin along the fur
direction, and uses root-to-tip coordinates for opacity/occlusion and motion.
The relevant primary implementation is:

- <https://github.com/hecomi/UnityFurURP/blob/main/Assets/Fur/Shaders/Fin/Lit.hlsl>
- <https://github.com/hecomi/UnityFurURP/tree/main/Assets/Fur/Shaders/Fin>
- Author's implementation note: <https://tips.hecomi.com/entry/2021/07/24/121420>
- DeepWiki overview supplied for this experiment:
  <https://deepwiki.com/hecomi/UnityFurURP/2.3-fin-based-fur-rendering>

UniFur ports the *inductive bias*, not Unity's geometry shader. A shell sample
is represented as a thin anisotropic Gaussian ribbon in the local
fur-direction/surface-tangent frame. Its renderer-side gate is

`sigmoid((tau - abs(dot(view, surface_normal))) / softness)`.

This concentrates shell capacity around silhouettes and prevents it from
duplicating a free volumetric RGB residual throughout the object.

## Safe optimization path

1. Fit and freeze a complete Residual-only 3DGS teacher.
2. Initialize Fin/strand geometry and opacity increments at exactly zero.
3. Preserve the full teacher opacity. Structured routes are additive; closing
   a view-conditioned fin cannot delete residual opacity.
4. Activate fins from projected multi-view silhouette-band support. This is a
   surrogate only for the exactly-zero state, which HairGS culls before CUDA
   backward. Once active, the ordinary differentiable renderer governs them.
5. Use held-out non-regression against the frozen teacher.
6. Reward route mass only for positive leave-one-route-out damage. Charge
   negative ablation contribution directly against the route.
7. Cull strands with the calibrated multi-view visual hull.

The key distinction is `teacher + structured delta`, not another three-way
opacity partition. The old mutually exclusive partition lost opacity whenever
a fin was closed in a frontal view.

## Current controlled evidence

Protocol: Panda official-prior static multi-view, 28 fit views / 8 held-out
views, 480x270, HairGS rasterizer, 20k source Gaussians.

| Model | Steps | FG PSNR | masked PSNR | masked LPIPS | mask IoU |
|---|---:|---:|---:|---:|---:|
| Residual-only teacher | full | 24.2619 | 31.9547 | 0.06366 | 0.98960 |
| Old UniFur moderate soft | full | 23.6851 | 31.3779 | 0.05649 | 0.98819 |
| Post-hoc Fin on old UniFur | eval only | 22.7148 | 30.4076 | 0.05920 | 0.98826 |
| Additive Fin-GS smoke | 300 | 24.2743 | 31.9671 | 0.06313 | 0.98923 |
| Additive Fin-GS soft | 12k | 24.3898 | 32.0826 | 0.06239 | 0.98867 |
| Additive Fin-GS hard | 12k | **24.4032** | **32.0960** | **0.06212** | 0.98841 |

The post-hoc result is a negative ablation: Fin must be trained under the
additive teacher objective. Relative to Residual-only, the 12k hard result is
+0.1413 dB in foreground/masked PSNR and improves masked LPIPS by 0.00154, but
mask IoU is lower by 0.00119 and full-frame PSNR by 0.1137 dB. This is a real
Pareto trade-off, not a claim that Fin already dominates every metric.

Protocol: HairGS `wCurly` static multi-view, 12 fit views / 4 held-out views,
512x512, fixed cameras, HairGS rasterizer, 20k source Gaussians.

| Model | Steps | FG PSNR | masked PSNR | masked SSIM | masked LPIPS | mask IoU | full PSNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Residual-only teacher | 6k | 19.0187 | 26.7714 | 0.88326 | 0.15573 | 0.77230 | 20.4188 |
| Additive Fin-GS smoke | 300 | **19.4504** | **27.2031** | 0.88660 | **0.15306** | 0.77892 | 20.4111 |
| Additive Fin-GS soft | 12k | 19.3609 | 27.1137 | **0.88671** | 0.15430 | **0.77983** | **20.6085** |
| Additive Fin-GS hard | 12k | 19.2850 | 27.0378 | 0.88572 | 0.15445 | 0.77981 | 20.5939 |

The full soft model improves over Residual-only by +0.3423 dB foreground and
+0.1896 dB full-frame PSNR, +0.00753 mask IoU, and -0.00143 masked LPIPS. Its
final route mass is 3.95% shell, 18.64% strand and 77.41% residual. The final
leave-one-route-out damages are positive for shell (0.00180) and strand
(0.03836), so these routes are not merely carrying nonzero mass: deleting
them hurts held-out reconstruction. Background opacity is nevertheless worse
by 0.00207, and the source HairGS scaffold still has conspicuous holes and
view-dependent streaks. The 300-step row is an early-budget ablation, not a
checkpoint selected on the test set.

## Downstream claim and evaluation

Pure RGB metrics are a safety constraint, not the strongest reason to keep
the structured distribution. The representation should additionally be
evaluated on operations that a residual cloud does not parameterize:

- fur/hair length scaling with fixed roots;
- wind/gravity deformation whose displacement grows monotonically root-to-tip;
- scalp/body motion retargeting with root attachment and penetration metrics;
- level-of-detail pruning, measuring silhouette quality at equal Gaussian
  budgets;
- route-only edits and relighting without retraining the residual teacher.

For a paper, report teacher-relative RGB delta together with boundary F-score,
root slip, scalp penetration, temporal warping error, edit locality, and
Gaussian count. A valid win is a Pareto result: non-inferior reconstruction
plus a significant downstream structural/editing advantage.

The current implementation exposes root-relative length and wind edits via
`edit_structured_fibers`. Both preserve the frozen residual route exactly;
the wind displacement follows a root-to-tip power law. Unit tests verify zero
teacher drift, zero residual edit drift, fixed roots and increasing tip
response. This establishes the required semantics, but a paper-level claim
of *significantly* better downstream output still requires the render-space
edit/retarget/LOD benchmark above across multiple subjects.

## Simulation-ready carrier stage

Rendering ownership is now separated from deformation ownership.  Every
source Gaussian has a learned `surface/shell/strand` carrier distribution and
a root-to-tip attachment coordinate.  Residual-rendering Gaussians therefore
remain available for photometric safety but are no longer free particles:
surface carriers follow the body/scalp frame, shell carriers follow short-fur
motion, and strand carriers follow guide-curve motion.  Carrier attachment,
entropy, neighborhood, root-tip and route-family constraints are optimized
without requiring the residual render route to disappear.

After reconstruction, a deterministic carrier calibration preserves the
learned surface-vs-fiber decision, enforces at least the learned structured
render mass, and allocates fiber mass between shell/strand from their final
relative route evidence.  On Panda this produces 7.3% surface, 90.0% shell and
2.7% strand source carriers; on HairGS wCurly it produces 72.8% surface, 8.5%
shell and 18.8% strand.  Opacity-weighted residual fiber attachment is 92.9%
and 33.2%, respectively.

The simulation-ready rerun preserves Panda appearance (soft FG PSNR 24.3888)
and improves the Hair soft result to FG PSNR 19.4073, masked LPIPS 0.15394,
mask IoU 0.78080 and full PSNR 20.6214.  The validation video is an editability
audit, not a physical-simulation benchmark: it applies root-relative length
and wind fields to all Gaussians through hard carrier assignments.  Collision,
inertia and temporal ground truth remain future evaluation requirements.

Research/code review was AI-assisted; all implementation claims above should
be verified against the pinned local UnityFurURP source commit before release.
