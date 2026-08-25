# Frozen audit standard

The baseline and every candidate must use the exact WCurly protocol:

- train: `train12`, held-out: `test4`;
- fixed cameras, 500 x 500 pixels;
- composite head-plus-hair RGB with hair-foreground alpha/mask semantics;
- export through `scripts/export_hairgs_external_renders.py`;
- metrics only from `scripts/evaluate_external_renders.py`.

Promotion gates are fixed before looking at candidate metrics.

## Quality retention

- held-out foreground PSNR drop no larger than 0.50 dB;
- held-out masked LPIPS increase no larger than 0.010;
- held-out mask IoU drop no larger than 0.010;
- train foreground PSNR drop no larger than 1.00 dB;
- largest connected FN and FP may each increase by at most 0.020.

## Efficiency

At least two of the following three gates must pass, and model-size reduction is
mandatory:

- final PLY size reduced by at least 30%;
- Stage-3 optimization wall time reduced by at least 50%;
- median render wall time or peak allocated VRAM reduced by at least 25%.

## Structural validity

- every hair-foreground segment belongs to exactly one of shell, fin, strand;
- non-hair head/base segments are fixed and labelled `route_id=-1`; they are
  excluded from route-mass claims and must be preserved for composite RGB;
- every route is non-empty;
- all connected components remain open chains and have a valid root;
- route-removal renders are evaluated separately; a route that improves the
  frozen held-out objective when removed is reported as a negative contributor;
- no candidate can be promoted from train-view metrics alone.

## Claim attribution

The gates above remain frozen. Separately, any claim that adaptive
shell/fin/strand allocation improves reconstruction quality requires a
same-initialization, same-step, same-restart native HairGS control. Without that
control, a candidate may be promoted for quality retention and efficiency, but
its gain cannot be attributed to routing. This attribution check does not
retroactively change the numeric gates.
