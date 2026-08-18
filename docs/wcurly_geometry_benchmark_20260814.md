# wCurly GT geometry benchmark — 2026-08-14

## Protocol

- Dataset: Cem Yuksel `wCurly`, canonical coordinates in metres.
- Training/render split: the existing fixed-camera 12-train / 4-novel protocol.
- Geometry GT: `hair_eval_data.npz`, 3,341,580 points, 50,000 strands and
  3,291,580 edges.
- Metric: the official HairGS bidirectional point/orientation correspondence
  evaluation at 2 mm/20 degrees, 3 mm/30 degrees, 4 mm/40 degrees and
  4 mm/90 degrees.
- UniFur curves are arc-length sampled at 3 mm. The 2 mm and 4 mm sensitivity
  runs preserve the conclusions.
- No ICP, scale fitting or post-hoc rigid alignment is applied.
- Strand consistency (SC) is not reported for Residual-only because its points
  have no strand topology. Inventing an ordering would make SC invalid.

## Main geometry result

| Representation | P@4/40 | R@4/40 | F@4/40 | SC@4/40 |
|---|---:|---:|---:|---:|
| HairGS, same 12-view input | 0.5641 | 0.4693 | 0.5123 | 0.1424 |
| UniFur, shell + strand deployed | 0.5596 | 0.0581 | 0.1053 | 0.0288 |
| UniFur, strand deployed | 0.5711 | 0.0534 | 0.0977 | 0.0269 |
| UniFur, strand target (diagnostic only) | 0.2050 | 0.2153 | 0.2100 | 0.0824 |
| Residual-only, unstructured points | 0.1614 | 0.0545 | 0.0815 | N/A |

The primary UniFur output is `shell + strand deployed`. Its precision is almost
the same as HairGS, but its recall is only 12.4% of HairGS recall. The present
failure is therefore structural coverage, not primarily local orientation
accuracy.

Compared with Residual-only at 4 mm/40 degrees, the deployed structured output
improves precision from 0.1614 to 0.5596 and F-score from 0.0815 to 0.1053
(+29.2%), while recall changes only from 0.0545 to 0.0581. This is evidence that
route allocation selects useful hair geometry, but not that it reconstructs a
complete hairstyle.

## Structural audit

- Hard route mass: shell 1,817 (4.16%), strand 7,903 (18.10%), residual 33,942
  (77.74%).
- Deployed strand route: 4,590/7,903 curves are non-degenerate; 41.9% collapse.
- Deployed shell+strand: 5,310/9,720 curves are non-degenerate; 45.4% collapse.
- Deployed strand median length is 4.55 mm; the analytic target median is
  56.94 mm.
- The deployed geometry remains inside the GT canonical bounding box. The
  ungated analytic target extends far outside it.

The ungated target raises recall and SC, but loses precision and exits the hair
volume. Forcing all continuation gains to one is therefore not a valid fix.

## Rendering context

On the same 12/4 fixed-camera split, the previously measured novel-view
foreground PSNR is 19.8495 dB for UniFur and 19.6855 dB for Residual-only
(+0.164 dB). Rendering remains dominated by the residual teacher. This geometry
benchmark prevents that small photometric gain from being mistaken for complete
strand reconstruction.

## Decision and next milestone

The unified carrier is a viable proof of concept: its structured subset is
geometrically much cleaner than unstructured residual points and can support
editing/simulation semantics. The current model is not yet competitive with
HairGS for explicit hairstyle recovery.

The next implementation priority is to replace independent per-Gaussian short
curves with scalp-seeded, shared orientation-field growth and graph
connection/merging, then couple route assignment to non-degenerate deployed
length and multi-view visual-hull support. Continuation should unfold only
supported geometry. The next gate is F@4/40 >= 0.35 and SC@4/40 >= 0.10 while
keeping novel-view foreground PSNR within 0.5 dB of Residual-only.
