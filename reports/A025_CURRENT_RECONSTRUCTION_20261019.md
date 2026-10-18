# A025 current reconstruction release — 2026-10-19

This release preserves both completed known-body reconstruction routes from
`eb5f741e6db45658818467b4ceccadea04390616`. No training loss, route assignment,
checkpoint, native renderer or reported static metric has changed. It adds the
subsequently validated material-binding builder, continuous mesh-driven decoder
and full standard 3DGS PLY export. Successful reconstruction does not establish
physically correct roots, anatomical route ownership or collision-free dynamics.

## Included positive path

1. Run the archived oracle-surface reconstruction as documented in
   `experiments/a025_oracle_surface_routes_20261017/docs/README.md`.
2. Build a self-contained asset with
   `a025_surface_binding_stage_a_20261017/binding.py::build`: retain all native
   GS fields; add body mesh, `gs_guide`, per-guide triangle/barycentric anchors,
   Shell layer labels and Strand nodes/edges. Existing multi-segment chains are
   retained for route 1; route 2 rods are not relabeled as connected chains.
3. For nonrigid body motion use
   `a025_surface_follow_repair_20261017/smooth_binding.py::deform`, **not** the
   earlier independent per-face rigid-offset decoder. The continuous body-driven
   field pins material roots, moves each shared node once and transports Shell
   covariance with the local deformation Jacobian. World-frame SH is unchanged.
4. Export the trained native fields with
   `a025_oracle_surface_routes_20261017/tools/export_gaussians.py`. This calls
   upstream `GaussianModel.save_ply` on CPU and verifies all 62 native PLY fields
   after reading back. No group/route/binding properties are placed in the PLY.

Paths in step 2/3 are under `experiments/`. Source NPZ and bound assets are
external artifacts, deliberately excluded from Git; both contain complete native
GS fields, but only the bound asset contains the mesh and material anchors.

## Checks

With the documented NumPy/SciPy/Torch environment, from the repository root:

```sh
python -B experiments/a025_surface_binding_stage_a_20261017/binding.py
python -B experiments/a025_surface_follow_repair_20261017/smooth_binding.py
python -B experiments/a025_oracle_surface_routes_20261017/tools/export_gaussians.py \
  outputs/a025_oracle_surface_routes_20261017 \
  outputs/a025_oracle_surface_routes_20261017/3dgs_ply_export \
  compare_methods/3DGS/a025_20261015/source
```

The export destination must be fresh. The pinned native environment/build must
already exist; this is not a new turnkey environment installer. No CUDA training
is started by these checks or export.

The full-model validation retained 215,495 / 122,767 GS, exact rest RGB and true
alpha on train28, no barycentric root slip, and rendered shared-endpoint errors
below 1e-7 world units in the repaired test. These are kinematic correctness
checks, not dynamic ground truth or proof of anatomical root identity.

Frozen model SHA256:

- route 1: `bf04de7d72c24e179d2fe5dd45ee573c3dae4ad631fb66bfd3ecf591062af242`
- route 2: `876f5da3770e0ba6f9f99c1fb144ba4578d89edcdd619a11ff3749336e25af5b`

## Known limits

- Shape-based route 1 labeling and image-gradient-based route 2 labeling do not
  guarantee a short-body-fur versus long-mane-fur semantic partition.
- Material roots are pinned, while visible GS retain fitted rest offsets;
  invisible root-to-visible-base connectors can be long.
- Route 2's tangent-plane inequality is not global signed-distance collision.
- Shell column/layer labels are not continuous collision-tested surface layers.
- The simple angular-guide swing demo is not a validated flexible rod solver;
  neither it nor the rejected full-particle attempt is included as reconstruction
  training code. No new physical material, self-collision or relighting claim.
- Licensed body/dataset assets, outputs, checkpoints, temporary exports and
  historical failed diagnostics are excluded from this source release.
