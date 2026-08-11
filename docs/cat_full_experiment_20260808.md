# Cat unified fur/hair full optimization experiment

## Reproducible subset

- Data root: `F:/fur_hair_unified_data/cat_sequence_subset`
- Frames: 40 consecutive 1920x1080 RGBA frames with preserved sequence indices.
- Train: frames 0-31.
- Hold-out validation: frames 32-39.
- Training render resolution: 512x288.
- Representation: 20,000 surface-anchored source Gaussians; 2 shell samples, 5 strand samples, and one residual Gaussian per source.
- Renderer: HairGS CUDA differentiable Gaussian rasterizer.
- Optimization: Adam, 1,200 steps, 200-step residual scaffold warm-up.

## Results

| run / deployment mode | foreground PSNR | foreground L1 | mask IoU | mask F1 |
|---|---:|---:|---:|---:|
| v1 hard routing | 15.5766 dB | 0.1263 | 0.5440 | 0.7044 |
| v1 soft mixture | 16.2403 dB | 0.1157 | 0.5678 | 0.7242 |
| v2 hard routing | **15.8383 dB** | **0.1218** | **0.5596** | **0.7174** |
| v2 soft mixture | **16.3983 dB** | **0.1128** | **0.5796** | **0.7337** |

The v2 hard-routing model improves over v1 hard routing by 0.2617 dB PSNR,
0.0156 mask IoU, and 0.0130 mask F1. Its final hard route allocation is
20.31% shell, 21.00% strand, and 58.69% residual. The result therefore uses
all three representations rather than collapsing to one route.

The first eight training frames score 20.2967 dB / 0.7694 IoU while the eight
future hold-out frames score 15.8383 dB / 0.5596 IoU. This is a material
generalization gap and must not be hidden by reporting training renders alone.

## Prototype optimization motivated by v1

v1 optimized a soft route mixture but exported a hard route. The loss curve
also showed an abrupt discontinuity when residual-only warm-up ended. v2 adds:

1. Residual-to-structured route continuation from steps 200-400.
2. Soft-to-straight-through hard route annealing from steps 400-1200.
3. Effective-route logging, live JSONL metrics, EMA diagnostics, finite-loss
   guards, optimizer checkpoints, and held-out hard/soft/forced-route evaluation.

This closes part of the train/deploy mismatch and improves both hard and soft
hold-out metrics. It does not fix the remaining fuzzy silhouette, missing thin
limbs, or the temporal extrapolation gap. The next modeling priority is a
spatially coherent route field plus boundary-aware/densification losses, not
simply more optimization steps.
