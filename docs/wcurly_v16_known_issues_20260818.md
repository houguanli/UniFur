# WCurly v16 milestone and known limitations

Date: 2026-08-18

This milestone records the residual-free shell/strand routing prototype with:

- multi-view orientation moment supervision;
- occlusion-aware visual-hull constraints;
- adaptive soft shell/strand migration;
- grouped worst-view calibration;
- fixed head/base separation for hair-only optimization;
- unified train/test rendering and route diagnostics.

The implementation and experiment configurations through `cleanhair_v16_groupcal_occlusion_6k_v1` are reproducible from this revision. The current milestone is an engineering checkpoint, not the final reconstruction model.

## Verified behavior

- Final soft route mass is approximately 46.27% shell and 53.73% strand, with no residual route mass.
- The grouped calibration improves the previously weakest rear held-out view relative to v15.
- Training-view foreground coverage is mostly complete, but held-out generalization remains below the desired quality.

## Known structural limitations

1. `nearest_vertex` binding collapses 108,981 nominal sources to only 5,802 distinct root positions (94.68% duplicate roots).
2. Each analytic strand is represented by only five Gaussian samples, compared with a median of 67 GT samples per strand in WCurly.
3. The same transverse radius is used from root to tip, so tips do not taper and can appear unnaturally thick.
4. Median learned analytic strand length is about 0.0466, versus a GT median arc length of about 0.414.
5. Median structured geometry gain is only about 0.284, so residual-teacher geometry still dominates the physical placement even though the residual route mass is zero.
6. This run records zero accepted topology updates; it therefore cannot create missing 3D support in large held-out holes.
7. At alpha threshold 0.1, the 12 training views have approximately 1.09% missing foreground and 8.22% spill, while the four held-out views have approximately 7.48% missing foreground and 15.29% spill.

## Required next work

- replace nearest-vertex anchoring with continuous surface/scalp binding and learnable root offsets;
- deduplicate and redistribute roots before adding capacity;
- use adaptive 8--16 segment strands with monotonic tip taper and optical-mass conservation;
- add multi-view-consistent birth/prune events in a continuous scalp/fiber density field;
- establish equal-budget GT-to-GS oracle renders to separate representation limits from optimization failures.
