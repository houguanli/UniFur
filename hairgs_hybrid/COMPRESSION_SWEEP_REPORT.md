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
| 5k | UniFur | 17.748 | 17.586 | 0.1478 | 0.8297 | 0.1358 | 0.0408 | 296 s | 183.6 | 3.31 MiB |
| 5k | HairGS | 17.714 | 17.623 | 0.1474 | 0.8292 | 0.1353 | 0.0421 | 249 s | 252.1 | 3.21 MiB |

The learned route allocation is 56.5% Shell, 22.5% Fin, and 21.0% Strand. The
recommended knee is **8k**, or 19.5% of the 40,936 source hair points. It gives
small but consistent same-budget improvements in FG PSNR (+0.048 dB), Full
PSNR (+0.080 dB), masked LPIPS (-0.00065), and IoU (+0.00215). At 5k, Full
PSNR and LPIPS no longer beat HairGS, so the apparent higher foreground PSNR is
not a general quality improvement.

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
```

The canonical local result root is
`F:\fur_hair_unified_data\benchmarks\hairgs_hybrid\compression_sweep_spill_v1`.
