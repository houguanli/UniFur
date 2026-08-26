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
  come from one completed `compression_sweep_audit.json`.

## Panda fur results

| Active points | Method | FG PSNR | Full PSNR | Masked LPIPS | IoU | FN | FP | Train | FPS | Size |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 25k | UniFur | 20.379 | 27.154 | 0.0817 | 0.9509 | 0.0489 | 0.0002 | 267 s | 203.7 | 7.09 MiB |
| 25k | HairGS | 19.998 | 26.231 | 0.0797 | 0.9312 | 0.0156 | 0.0574 | 193 s | 265.2 | 6.58 MiB |
| 16k | UniFur | 19.836 | 26.656 | 0.0950 | 0.9465 | 0.0534 | 0.0001 | 255 s | 198.9 | 4.54 MiB |
| 16k | HairGS | 19.754 | 25.899 | 0.0888 | 0.9283 | 0.0149 | 0.0615 | 176 s | 296.0 | 4.21 MiB |
| 10k | UniFur | 18.910 | 26.026 | 0.1161 | 0.9401 | 0.0598 | 0.0001 | 243 s | 226.2 | 2.84 MiB |
| 10k | HairGS | 19.167 | 25.521 | 0.1047 | 0.9253 | 0.0116 | 0.0685 | 170 s | 280.1 | 2.63 MiB |

The learned route allocation is stable across budgets: 80.1% Shell, 14.4%
Fin, and 5.5% Strand.  The recommended knee is **16k**, or 16.5% of the 96,991
source points.  It retains a +0.757 dB same-budget Full-PSNR advantage and even
exceeds the 25k HairGS Full PSNR by +0.425 dB with 36% fewer points.  It does
not exceed 25k HairGS in foreground PSNR or masked LPIPS.  At 10k, foreground
quality reverses, so 10k is an aggressive storage setting rather than the
quality operating point.

The Panda Full-PSNR improvement is not a background-accounting artifact:
outside-mask RGB is explicitly calibrated against black, and FP drops by over
0.06 at 16k.  The corresponding increase in FN must be reported: the method
prefers a conservative silhouette and leaves more missing fur than HairGS.

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

The experiment supports **quality per active Gaussian**, not a faster
same-point renderer. The current soft router adds a sidecar and route-specific
covariance work: training is 14--45% longer and rendering is 22--33% slower
than the same-budget HairGS control. The Shell claim is therefore presently a
compression result: large low-frequency fur regions tolerate fewer active
Gaussians. A fused hard-route renderer and compact route-state serialization
are required before claiming wall-clock or total-storage superiority at equal
point count.

Reproduce with:

```bash
bash hairgs_hybrid/run_adaptive_compression_sweep.sh

# WCurly-only fine sweep used for the 7k/6k/4k/3k/2k rows
RUN_PANDA=0 RUN_WCURLY=1 \
HAIR_BUDGETS="7000 6000 4000 3000 2000" \
bash hairgs_hybrid/run_adaptive_compression_sweep.sh
```

The canonical local result roots are
`F:\fur_hair_unified_data\benchmarks\hairgs_hybrid\compression_sweep_spill_v1`
and
`F:\fur_hair_unified_data\benchmarks\hairgs_hybrid\compression_sweep_wcurly_fine_v1`.
