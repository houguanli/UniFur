# UniFur

HairGS-based adaptive shell/fin/strand reconstruction for multi-view hair.

This branch replaces the maintained hair reconstruction path with native
HairGS Stage-1/Stage-2 geometry and rendering. UniFur adds deterministic route
ownership, route-aware optimization, unified held-out evaluation, and
leave-one-route-out auditing without changing HairGS camera or alpha semantics.

## Current milestone

The WCurly milestone uses 12 training views and 4 held-out views at 500 x 500
with fixed calibrated cameras. It retains the native HairGS Stage-2 model,
freezes topology, and replaces the native 15k Stage-3 with route-aware 4k plus
1k calibration.

| Method | train FG PSNR | held-out FG PSNR | held-out LPIPS | held-out IoU | Stage-3 wall |
|---|---:|---:|---:|---:|---:|
| HairGS native 15k | 28.9437 | 16.8499 | 0.1309 | 0.8541 | 1376.22 s |
| HairGS native 5k control | 28.0057 | 17.3031 | 0.1269 | 0.8467 | 317.89 s |
| UniFur adaptive-route 5k | 27.9968 | **17.3480** | **0.1264** | 0.8466 | 323.08 s |

Against HairGS native 15k, the adaptive milestone reduces:

- Stage-3 wall time by 76.52%;
- Stage-3 optimization steps by 66.67%;
- final deployment bytes by 33.18%;
- segment count by 48.04%.

The median end-to-end render command improves by only 8.08%, so this milestone
does not claim significant render-speed acceleration.

## Claim boundary

The same-budget native 5k control is practically equivalent to adaptive 5k:
adaptive routing changes held-out foreground PSNR by +0.0449 dB and mask IoU by
-0.0002. The validated milestone claim is therefore **quality retention plus
training/storage efficiency and explicit route ownership**. It is not a claim
that route-aware gradients significantly improve reconstruction quality.

All three routes are currently native HairGS line Gaussians:

- `shell`: dense, low-curvature, scalp-near coverage;
- `fin`: silhouette-supporting intermediate fibers;
- `strand`: long, curved, tip-bearing detail.

Non-hair head/base segments use route `-1`, remain available for composite RGB,
and are excluded from hair route mass. The routes are not yet an explicit shell
volume, fin ribbon mesh, or simulation-ready continuous strand graph.

## Positive pipeline

```text
native HairGS Stage-1 / Stage-2
              |
deterministic base / shell / fin / strand assignment
              |
route-aware 4k optimization, frozen topology
              |
1k calibration
              |
train12 + held-out test4 rendering at 500 x 500
              |
unified evaluator + route removal + topology audit
```

Route assignment uses chain length, curvature, scalp distance, tip distance,
opacity, hair mask, and width. The current gradient scales are base 0.20, shell
0.55, fin 0.80, and strand 1.00.

## Reproduce

Required external paths are configurable through environment variables; the
tested defaults are WSL paths under `/home/aoki` and `/mnt/f`.

Run the adaptive milestone:

```bash
bash hairgs_adaptive_routes/run_wcurly_stage2_routeaware5k.sh
```

Run the same-budget native HairGS control:

```bash
bash hairgs_adaptive_routes/run_wcurly_stage2_native5k_control.sh
```

Run leave-one-route-out evaluation:

```bash
bash hairgs_adaptive_routes/run_route_ablation_audit.sh
```

Generate the frozen audit report:

```bash
conda run -n hair-gs python hairgs_adaptive_routes/audit_wcurly.py \
  --baseline /path/to/hairgs_native_15k \
  --candidate /path/to/adaptive_route_5k \
  --native-control /path/to/native_5k_control \
  --baseline-render-control /path/to/exact_baseline_render_control \
  --ablations /path/to/route_ablations \
  --output-dir /path/to/adaptive_route_5k/audit
```

The numeric gates and attribution rule are frozen in
[`hairgs_adaptive_routes/EVALUATION_STANDARD.md`](hairgs_adaptive_routes/EVALUATION_STANDARD.md).

## Maintained code

```text
hairgs_adaptive_routes/
  assign_routes.py                         deterministic Stage-2 assignment
  train_adaptive.py                        route-aware HairGS optimization
  route_ablate.py                          leave-one-route-out model export
  run_wcurly_stage2_routeaware5k.sh        promoted training/evaluation flow
  run_wcurly_stage2_native5k_control.sh    same-budget native control
  run_route_ablation_audit.sh              contribution experiment
  audit_wcurly.py                          protocol/quality/efficiency audit
  make_wcurly_comparison.py                compact visual comparison
  EVALUATION_STANDARD.md                   frozen promotion rules
```

Datasets, external HairGS source, checkpoints, rendered images, and generated
audit outputs are not committed.

## Known structural limit

The audited model contains 67,114 open-chain components, including 64,511
single-segment components. Only 7,801 components contain an explicit root; the
remainder use HairGS reference-scalp inference. The next structural milestone
must address chain continuity and explicit roots before making downstream
simulation claims.
