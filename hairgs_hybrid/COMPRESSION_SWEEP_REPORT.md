# Adaptive compression sweep

## Protocol

This sweep compares UniFur Shell--Fin--Strand routing with HairGS at the same
active hair/fur Gaussian budget.  For every row, both methods start from the
same byte-identical compressed PLY. Allocation uses only the dense teacher,
training-view hair masks, and positive training-view leave-one-route-out
contribution; species labels and held-out views are not inputs.

- Panda: 28 fit / 8 held-out views at 480x270, no separate base Gaussians.
- WCurly: train12 / test4 at 500x500, with the same frozen 29,278-point head
  base in both methods. Only the active hair budget changes.
- Each model receives 5,000 optimization steps.
- Full, foreground, mask, LPIPS, training time, storage, and renderer timings
  come from completed `compression_sweep_audit.json` artifacts.

## Panda fur results

| Active points | Method | FG PSNR | Full PSNR | Masked LPIPS | IoU | FN | FP | Train | FPS | Size |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32k | UniFur | 20.658 | 27.332 | 0.0780 | 0.9534 | 0.0464 | 0.0002 | 284 s | 202.0 | 9.08 MiB |
| 32k | HairGS | 20.409 | 26.471 | 0.0735 | 0.9337 | 0.0142 | 0.0561 | 203 s | 250.9 | 8.42 MiB |
| 28k | UniFur | 20.535 | 27.253 | 0.0810 | 0.9517 | 0.0481 | 0.0002 | 269 s | 200.2 | 7.94 MiB |
| 28k | HairGS | 20.021 | 26.264 | 0.0807 | 0.9298 | 0.0161 | 0.0586 | 199 s | 274.0 | 7.37 MiB |
| 25k | UniFur | 20.379 | 27.154 | 0.0817 | 0.9509 | 0.0489 | 0.0002 | 267 s | 203.7 | 7.09 MiB |
| 25k | HairGS | 19.998 | 26.231 | 0.0797 | 0.9312 | 0.0156 | 0.0574 | 193 s | 265.2 | 6.58 MiB |
| **22k** | **UniFur** | **20.288** | **27.050** | **0.0833** | **0.9504** | **0.0494** | **0.0003** | **269 s** | **213.8** | **6.24 MiB** |
| **22k** | **HairGS** | **19.788** | **26.088** | **0.0860** | **0.9304** | **0.0164** | **0.0576** | **185 s** | **277.0** | **5.79 MiB** |
| 20k | UniFur | 20.054 | 26.849 | 0.0894 | 0.9486 | 0.0513 | 0.0002 | 264 s | 213.7 | 5.68 MiB |
| 20k | HairGS | 19.919 | 26.131 | 0.0854 | 0.9329 | 0.0137 | 0.0574 | 184 s | 272.7 | 5.27 MiB |
| 18k | UniFur | 19.982 | 26.763 | 0.0912 | 0.9479 | 0.0520 | 0.0002 | 255 s | 215.6 | 5.11 MiB |
| 18k | HairGS | 19.695 | 25.943 | 0.0893 | 0.9300 | 0.0179 | 0.0562 | 198 s | 306.0 | 4.74 MiB |
| 16k | UniFur | 19.836 | 26.656 | 0.0950 | 0.9465 | 0.0534 | 0.0001 | 255 s | 198.9 | 4.54 MiB |
| 16k | HairGS | 19.754 | 25.899 | 0.0888 | 0.9283 | 0.0149 | 0.0615 | 176 s | 296.0 | 4.21 MiB |
| 14k | UniFur | 19.634 | 26.571 | 0.1039 | 0.9435 | 0.0563 | 0.0002 | 273 s | 204.3 | 3.98 MiB |
| 14k | HairGS | 19.319 | 25.672 | 0.0956 | 0.9303 | 0.0147 | 0.0594 | 189 s | 270.5 | 3.69 MiB |
| 12k | UniFur | 19.248 | 26.243 | 0.1061 | 0.9418 | 0.0581 | 0.0001 | 259 s | 231.4 | 3.41 MiB |
| 12k | HairGS | 19.315 | 25.553 | 0.0993 | 0.9253 | 0.0163 | 0.0635 | 178 s | 282.3 | 3.16 MiB |
| 10k | UniFur | 18.910 | 26.026 | 0.1161 | 0.9401 | 0.0598 | 0.0001 | 243 s | 226.2 | 2.84 MiB |
| 10k | HairGS | 19.167 | 25.521 | 0.1047 | 0.9253 | 0.0116 | 0.0685 | 170 s | 280.1 | 2.63 MiB |
| 8k | UniFur | 18.633 | 25.617 | 0.1246 | 0.9366 | 0.0633 | 0.0001 | 253 s | 203.9 | 2.27 MiB |
| 8k | HairGS | 18.828 | 25.150 | 0.1154 | 0.9231 | 0.0153 | 0.0670 | 167 s | 290.2 | 2.11 MiB |

The learned route allocation is stable across budgets: 80.1% Shell, 14.4%
Fin, and 5.5% Strand.  The fine sweep moves the recommended compression/quality
knee to **22k**, or 22.7% of the 96,991 source points.  Against byte-identical,
same-budget HairGS initialization, 22k improves FG PSNR by +0.500 dB, Full PSNR
by +0.962 dB, masked LPIPS by -0.00273, full LPIPS by -0.01946, and IoU by
+0.02001.  Its masked SSIM is effectively tied but slightly lower (-0.00031).
The exact allocation is 17,619 Shell, 3,168 Fin, and 1,213 Strand.

For absolute quality, **28k** is the quality-first setting: it has 20.535 dB
FG PSNR and 27.253 dB Full PSNR, while masked LPIPS differs from 28k HairGS by
only +0.00024.  The 32k setting has the highest absolute FG PSNR but loses more
masked LPIPS, so extra capacity does not improve every dimension.  Below 20k,
masked LPIPS and the largest missing component deteriorate; at 12k and below,
same-budget foreground PSNR reverses.  No tested budget dominates HairGS in
every metric because the two models choose different alpha tradeoffs.

The Panda Full-PSNR improvement is not a background-accounting artifact:
outside-mask RGB is explicitly calibrated against black, and at 22k FP falls
from 0.05760 to 0.00026.  The corresponding increase in FN (0.01635 to 0.04936)
and largest-FN component (0.00656 to 0.01220) must be reported: UniFur removes
the visible HairGS halo but is more conservative at paws, belly, and thin fur.

The complete train/test table contains all 16 evaluation metrics plus training
time, observed GPU peak, PLY/sidecar bytes, render latency, FPS, and render GPU
peak.  It is stored as `panda_all_budget_comparison.csv` and JSON beside the
fine-sweep audit rather than truncating those fields in this readable table.

### Uncompressed HairGS reference

The 22k hard deployment was also audited against HairGS trained with all 96,991
source segments. This is intentionally not a same-budget comparison: it asks
whether the compressed model preserves useful dense-reference quality while
materially reducing time and space.

| Method | Segments | FG PSNR | Full PSNR | Masked LPIPS | Train | FPS | Train GPU | Render GPU | Size |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HairGS dense | 96,991 | 20.324 | 26.374 | 0.0711 | 310.5 s | 225.3 | 4,639 MiB | 119.0 MiB | 25.53 MiB |
| UniFur hard | 22,000 | 20.232 | 26.984 | 0.0844 | 198.7 s | 275.3 | 4,017 MiB | 68.6 MiB | 5.86 MiB |

UniFur uses 22.7% of the segments, trains 36.0% faster, stores 77.0% fewer
bytes, renders with 18.1% lower latency, and uses 42.4% less allocated render
memory. Held-out FG PSNR is within 0.093 dB and Full PSNR is +0.610 dB because
the hard model removes most of the dense HairGS silhouette spill. This gain has
the opposite boundary tradeoff: FN rises from 0.0185 to 0.0496 while FP falls
from 0.0566 to 0.0003. Masked LPIPS is 0.0134 worse, and the visual comparison
shows weaker fine fur detail. The supported conclusion is therefore
near-quality dense-reference rendering with substantial time/space savings,
not perceptual equivalence.

## WCurly hair results

| Active hair | Method | FG PSNR | Full PSNR | Masked LPIPS | IoU | FN | FP | Train | FPS | Total size |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12k | UniFur | 17.246 | 17.447 | 0.1388 | 0.8387 | 0.1201 | 0.0484 | 311 s | 183.8 | 4.00 MiB |
| 12k | HairGS | 17.321 | 17.434 | 0.1391 | 0.8363 | 0.1222 | 0.0487 | 264 s | 250.7 | 3.75 MiB |
| 8k | UniFur | 17.415 | 17.474 | 0.1431 | 0.8332 | 0.1294 | 0.0442 | 296 s | 191.0 | 3.61 MiB |
| 8k | HairGS | 17.368 | 17.394 | 0.1437 | 0.8310 | 0.1319 | 0.0439 | 259 s | 249.4 | 3.44 MiB |
| 7k | UniFur | 17.449 | 17.503 | 0.1435 | 0.8329 | 0.1291 | 0.0449 | 302 s | 179.3 | 3.51 MiB |
| 7k | HairGS | 17.390 | 17.511 | 0.1444 | 0.8313 | 0.1308 | 0.0448 | 262 s | 264.5 | 3.36 MiB |
| **6k** | **UniFur** | **17.551** | **17.534** | **0.1446** | **0.8330** | **0.1305** | **0.0431** | **294 s** | **198.6** | **3.41 MiB** |
| **6k** | **HairGS** | **17.461** | **17.483** | **0.1472** | **0.8291** | **0.1335** | **0.0443** | **251 s** | **252.3** | **3.29 MiB** |
| 5k | UniFur | 17.748 | 17.586 | 0.1478 | 0.8297 | 0.1358 | 0.0408 | 296 s | 183.6 | 3.31 MiB |
| 5k | HairGS | 17.714 | 17.623 | 0.1474 | 0.8292 | 0.1353 | 0.0421 | 249 s | 252.1 | 3.21 MiB |
| 4k | UniFur | 17.384 | 17.458 | 0.1518 | 0.8250 | 0.1432 | 0.0377 | 286 s | 187.4 | 3.21 MiB |
| 4k | HairGS | 17.550 | 17.477 | 0.1514 | 0.8249 | 0.1435 | 0.0374 | 248 s | 254.5 | 3.13 MiB |
| 3k | UniFur | 17.628 | 17.485 | 0.1552 | 0.8143 | 0.1530 | 0.0392 | 285 s | 191.5 | 3.11 MiB |
| 3k | HairGS | 17.689 | 17.519 | 0.1558 | 0.8176 | 0.1490 | 0.0400 | 245 s | 232.1 | 3.05 MiB |
| 2k | UniFur | 17.676 | 17.607 | 0.1571 | 0.8108 | 0.1576 | 0.0381 | 280 s | 196.3 | 3.01 MiB |
| 2k | HairGS | 17.766 | 17.618 | 0.1576 | 0.8096 | 0.1623 | 0.0339 | 238 s | 247.2 | 2.97 MiB |

The learned route allocation remains stable at 56.5% Shell, 22.5% Fin, and
21.0% Strand.  The finer sweep moves the recommended knee from 8k to **6k**,
or 14.7% of the 40,936 source hair points.  The exact 6k allocation is 3,388
Shell, 1,351 Fin, and 1,261 Strand.  Against the byte-identical, same-budget
HairGS initialization, it improves FG PSNR by +0.091 dB, Full PSNR by +0.051
dB, masked LPIPS by -0.00268, IoU by +0.00391, and the largest false-negative
component from 0.0677 to 0.0483.  This is the strongest consistent relative
gap in the sweep, though still a modest visual rather than a generational
improvement.

The capacity knee is sharp rather than monotonic.  At 5k the Full-PSNR and
LPIPS wins disappear; at 4k the routed model loses FG/Full PSNR and LPIPS; at
3k it also loses IoU.  The isolated metric reversals at 2k do not constitute a
recovery: IoU falls to 0.8108 and the largest missing connected component rises
to 0.0896.  These low-budget failures are dominated by loss of spatial hair
coverage, so further compression requires a coverage-aware topology change,
not another budget reduction with the current shared support.

The WCurly budget counts only active hair Gaussians.  Both methods also carry
the same frozen 29,278-point head/base.  Therefore the 6k model has 35,278
total points, versus 70,214 points in the dense source: an 85.3% reduction of
active hair points and a 49.8% reduction of total points.

## Efficiency conclusion

The original sweep supports **quality per active Gaussian**. Its soft router
adds a sidecar and route-specific covariance work: training is 14--46% longer
and rendering is 22--33% slower than the same-budget HairGS control. The Shell
claim remains a compression result: large low-frequency fur regions tolerate
fewer active Gaussians.

The subsequent hard-route deployment closes most of that execution gap without
changing the selected topology. At the Panda 22k knee it reduces training time
from 269.4 to 198.7 s, render latency from 4.677 to 3.805 ms, and the sidecar
from 473,397 to 75,677 bytes. FG/Full PSNR change by only -0.056/-0.066 dB. At
the WCurly 6k knee it reduces training time from 294.2 to 226.8 s, latency from
5.035 to 4.087 ms, and the sidecar from 135,228 to 19,301 bytes; FG/Full PSNR
increase by +0.019/+0.003 dB. In both cases FPS is 23% higher than soft routing
and within 3--5% of the same-budget HairGS control.

This is a fused route-conditioned scale/rotation construction and cache feeding
the existing HairGS rasterizer, not a custom fused CUDA rasterizer. Total model
storage is now within 0.5--1.3% of HairGS. Training GPU peaks were inconsistent
(Panda +3.9%, WCurly -0.3% versus soft), so no VRAM improvement is claimed.

Reproduce with:

```bash
bash hairgs_hybrid/run_adaptive_compression_sweep.sh

# Panda-only fine sweep used for the 32k/28k/22k/20k/18k/14k/12k/8k rows
RUN_PANDA=1 RUN_WCURLY=0 \
PANDA_BUDGETS="32000 28000 22000 20000 18000 14000 12000 8000" \
bash hairgs_hybrid/run_adaptive_compression_sweep.sh

# WCurly-only fine sweep used for the 7k/6k/4k/3k/2k rows
RUN_PANDA=0 RUN_WCURLY=1 \
HAIR_BUDGETS="7000 6000 4000 3000 2000" \
bash hairgs_hybrid/run_adaptive_compression_sweep.sh
```

The canonical local result roots are
`F:\fur_hair_unified_data\benchmarks\hairgs_hybrid\compression_sweep_spill_v1`,
`F:\fur_hair_unified_data\benchmarks\hairgs_hybrid\compression_sweep_panda_fine_v1`,
and `F:\fur_hair_unified_data\benchmarks\hairgs_hybrid\compression_sweep_wcurly_fine_v1`.
