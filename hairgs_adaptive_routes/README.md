# HairGS adaptive shell/fin/strand extension

This directory contains the isolated HairGS-based replacement pipeline. It
does not mutate the frozen HairGS baseline. The promoted WCurly experiment
starts from native HairGS Stage-2, assigns every hair-foreground chain to one
of three routes, freezes topology, and replaces the native 15k Stage-3 with a
route-aware 4k phase plus a 1k calibration phase:

- `shell`: dense, low-curvature, scalp-near interior coverage;
- `fin`: silhouette-supporting, locally coherent ribbon-like coverage;
- `strand`: long, curved, tip-bearing explicit detail.

All routes remain native line Gaussians, so the upstream HairGS differentiable
rasterizer and losses are retained. Route-specific gradient bandwidth allocates
less update capacity to the fixed base and shell and more to fin and strand.
Route labels are stored in a compressed `.routes.npz` sidecar aligned with PLY
segment order; the PLY remains readable by an unmodified HairGS loader.

This is an efficiency and route-ownership result. The three routes are not yet
an explicit shell volume, fin ribbon mesh, or simulation-ready continuous
strand graph. `audit_wcurly.py` enforces this distinction and applies the frozen
quality, efficiency, topology, and leave-one-route-out gates.

The positive-flow entry points are:

- `assign_routes.py`: deterministic HairGS Stage-2 route assignment;
- `run_wcurly_stage2_routeaware5k.sh`: Stage-2 to route-aware 4k plus 1k,
  train/test rendering, and unified evaluation;
- `run_route_ablation_audit.sh`: shell/fin/strand leave-one-route-out renders;
- `audit_wcurly.py`: protocol, quality, efficiency, topology, contribution,
  and provenance audit.

The audit also requires the native 4k+1k same-budget control before attributing
any image-quality change to routing. On WCurly the adaptive and native 5k
variants are practically equivalent, so the promoted claim is efficiency plus
route ownership—not superior reconstruction caused by routing.
