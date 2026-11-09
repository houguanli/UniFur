# AFP-guided animal fur reconstruction

A025 final_prior combines frozen AnimalFurPrior guidance with native 3DGS. It fits free Gaussians for 30,000 steps, routes them into Shell and connected Strand, then refines for 2,000 steps. The Round18 entry point reuses those stages, optimizer settings, AFP checkpoint and calibration.

The A025 entry points retain their original signed dataset and run paths. Round18 uses the 33-case case_108 selection in round18_cases.json. The separate A025 strict-SMAL source run is outside that selection. Each Round18 case fits views 000–029 and renders the six held-out views 030–035 after training. Case body and camera arrays are validated against the signed receipt. AFP sees only training images and the known naked body. The adapter scales body vertices, cameras and predicted curves together to the A025 reference body diagonal of 2.323113865546574 model units so the reference grid spacing and loss tolerances apply across cases.

Use the existing A800 installation. UNIFUR_ROOT and AFP_ROOT can point to its UniFur and AnimalFurPrior directories if the checkout is elsewhere. The AFP phase uses the DINOv3 Python environment (Torch 2.7.1) for AFP and Open3D. The fit and render phases use the official 3DGS environment. The selected prior is AnimalFurPrior/output/original_adaptive_dense_20261103_v2/training/best.pt and its sibling calibration.json. Both are pinned by SHA-256.

```bash
cd <source-checkout>
PY=/root/Public/h84435299/UniFur/compare_methods/3DGS/a025_20261015/env/py310/bin/python
OUT=/root/Public/h84435299/UniFur/outputs/afp_prior_gs_round18_20261108
"$PY" -B -m hairgs_hybrid.afp_prior_gs_v1.batch --output "$OUT" --check
"$PY" -B -m hairgs_hybrid.afp_prior_gs_v1.batch --output "$OUT" --case 7431_006 --gpus 1 --smoke
"$PY" -B -m hairgs_hybrid.afp_prior_gs_v1.batch --output "$OUT" --gpus 8
```

Each worker claims one GPU UUID through the shared registry. The smoke phase reuses its prior for the formal run. Re-running the same command verifies completed cases and resumes pending cases from the last completed phase. Incomplete artifacts are retained in interrupted directories. Each formal case emits its source receipt, the AFP field, free and routed Gaussian tensors, complete_asset.npz, final_3dgs.ply, and six 16-bit RGBA test images under cases/<id>/reconstruction/test. BATCH_COMPLETE.json records each render's SHA-256 after validating all selected cases.

For the A025 Lion_Run animation, see the [v2/v3 physics replay](../../experiments/a025_running_fur_physics_20261109/README.md). It takes the paired A025 training export and run rig.
