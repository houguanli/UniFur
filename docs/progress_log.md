# Progress Log

## 2026-06-18 Stage 1

Completed:

- Verified `F:\from_4090sever\MocapAnything_inference_only.tar.gz` exists and is readable.
- Extracted MocapAnything to `/home/aoki/MocapAnything_inference_only`.
- Verified WSL projects:
  - `/home/aoki/sam3d-obj`
  - `/home/aoki/ElasticSimulator`
- Confirmed ElasticSimulator submodules are initialized:
  - `extern/tetgen`
  - `extern/glm`
  - `extern/glfw`
- Created WSL conda env `dpd3dgs-animal` by cloning `sam3d-objects`.
- Installed additional dependencies:
  - `diffusers==0.32.2`
  - `taichi==1.7.3`
  - `tetgen==0.8.4`
  - `diso==0.1.4`
  - editable `/home/aoki/ElasticSimulator`
  - editable current integration repo
- Implemented Stage 1 integration package `dpd3dgs_animal`.
- Added config, setup script, exact conda export, and integration docs.

Implemented pipeline coverage:

- video or frame-folder input
- frame extraction for MocapAnything
- SAM3D reference-frame mesh and 3DGS reconstruction adapter
- MocapAnything `video2pose` config writer and runner
- Mocap skeleton prior loader
- explicit Mocap-to-canonical coordinate conversion
- skeleton-to-mesh similarity fit
- ElasticSimulator TetGen tetrahedralization
- skeleton-driven tet-node and surface-vertex deformation
- surface triangle binding for 3D Gaussian centers
- preview point-splat rendering
- color L1 loss plus higher-weight binary mask loss

Verification:

- Python compile check passed for `src/dpd3dgs_animal`.
- `dpd3dgs-animal --help` works inside `dpd3dgs-animal`.
- MocapAnything inference modules and TripoSG pipeline import inside the unified env when `PYTHONPATH` includes the Mocap root and `TripoSG`.
- Synthetic smoke test passed:
  - cube mesh tetrahedralized with ElasticSimulator TetGen path
  - generated 8 tet nodes and 6 tetrahedra
  - skeleton drove surface vertices
  - Gaussian points bound to surface triangles and deformed
  - preview renderer produced RGB/mask and loss values

Observed issue:

- The WSL test shell reported `libcuda.so lib not found`, so the smoke test used CPU/Vulkan fallback. Full SAM3D and MocapAnything GPU inference requires WSL CUDA driver visibility.
- SAM3D's `sam3d_objects.__init__` imports a missing `sam3d_objects.init` in the original env too. The integration avoids that path by setting `LIDRA_SKIP_INIT=1` and using the notebook inference API.

## 2026-06-18 Stage 2

Implemented:

- Added PyTorch backward path for the Stage 1 state in `dpd3dgs_animal.optimize`.
- Added per-frame skeleton nodes as trainable parameters.
- Added differentiable chain:
  - skeleton nodes
  - tet-node displacement
  - surface-vertex displacement
  - barycentric Gaussian centers
  - soft point-splat render
  - color + mask loss
- Added tet edge and tet volume regularizers to preserve ElasticSimulator/TetGen rest shape.
- Added bone-length, temporal smoothness, and Mocap-prior regularizers.
- Added `dpd3dgs-animal stage2` CLI.
- Added `dpd3dgs-animal calibrate-coordinates` to rank axis transforms against a SAM3D mesh and a raw Mocap prediction.

Notes:

- The Stage 2 renderer uses a straight-through binary mask loss: the forward value is the requested hard `0/1` mask error, while gradients flow through the soft rendered mask. It also records `mask_soft` for diagnostics.
- The physical backward path is implemented as a differentiable PyTorch surrogate around ElasticSimulator's tetrahedralized rest state. ElasticSimulator still supplies tetrahedral topology; PyTorch supplies gradients to skeleton nodes.

Verification:

- Python compile check passed after Stage 2 changes.
- `dpd3dgs-animal stage2 --help` works.
- Synthetic Stage 2 smoke test passed:
  - loaded a synthetic Stage 1 NPZ, Gaussian PLY, and RGBA GT frames
  - optimized for several steps
  - confirmed nonzero skeleton-node gradient norm
  - wrote `stage2_optimized_state.npz`, `optimized_joints.npy`, `stage2_losses.json`, and preview renders
- `dpd3dgs-animal calibrate-coordinates` ran on a synthetic mesh and Mocap-like prediction.

## 2026-06-18 MocapAnything Built-In Dog Sample

Ran the built-in Dog sample from:

- frames: `/home/aoki/MocapAnything_inference_only/zoo/image/Dog#Dog-Galloping/y30`
- video: `/home/aoki/MocapAnything_inference_only/zoo/video/Dog#Dog-Galloping/y30.mp4`
- GT pose package: `/home/aoki/MocapAnything_inference_only/zoo/bvh_pose/Dog#Dog-Galloping/y30.npz`

Because the inference-only package has no `.obj`, `.ply`, or `.glb` mesh files,
the check used a generated skeleton-tube proxy surface and proxy Gaussian PLY.

Results:

- Built proxy Stage 1 files in `output/mocap_dog_sample_proxy`.
- Ran Stage 2 optimization against the real Dog sample frames using the bundled
  GT `bvh_pose` skeleton. Loss changed from `10.0283` to `9.9361` over 20 steps;
  skeleton gradient was nonzero.
- Ran actual MocapAnything `video2pose` inference on the Dog sample with CUDA.
  It produced `Dog_y30_pred.npy` and separate front/side/image comparison mp4s.
  Some frames failed RMBG preprocessing, so the output prediction had 17 frames.
  The final hstack mp4 failed because ffmpeg lacks `libopenh264`, but the
  separate videos were written.
- Ran Stage 2 optimization using the actual `Dog_y30_pred.npy`. Loss changed
  from `10.0040` to `9.9080` over 20 steps; skeleton gradient was nonzero.

## 2026-06-18 SAM3D Cat Full Pipeline Run

Inputs and preprocessing:

- Source video: `/home/aoki/MocapAnything_inference_only/zoo/video/Cat#Cat-Walk/y30.mp4`
- Extracted 90 raw frames to `output/sam3d_cat_full/raw_frames`.
- Built transparent-background RGBA frames from the manual foreground mask extraction in `output/sam3d_cat_full/rgba_frames`.
- Used the RGBA reference crop/mask in `output/sam3d_cat_full/sam3d_ref` for SAM3DObject.

SAM3DObject outputs:

- Full mesh: `output/sam3d_cat_full/sam3d/sam3d_mesh.glb` with 175,538 vertices and 351,000 faces.
- Gaussian cloud: `output/sam3d_cat_full/sam3d/sam3d_gaussian.ply` with 269,440 points.
- For practical tet/binding/runtime, generated a watertight Open3D quadric-decimated SAM3D mesh: `output/sam3d_cat_full/sam3d/sam3d_mesh_simplified_5k.obj` with 2,538 vertices and 5,000 faces.

MocapAnything:

- Ran video2pose with `Cat#Cat-Walk/y30` as reference.
- Selected transparent-frame prediction: `output/sam3d_cat_full/mocap_inference/exp2503/rgba_frames/Cat_rgba_frames_pred.npy`, shape `(90, 44, 3)`.
- The combined hstack mp4 failed only because system ffmpeg lacks `libopenh264`; the separate front/side/image videos were written.

Stage 1:

- Wrote `output/sam3d_cat_full/stage1_real_sam3d_simplified/stage1_tet_skeleton_surface.npz`.
- TetGen output: 17,203 tet nodes and 90,370 tetrahedra.
- Stage 1 render mean loss at 256x256: color `0.2260`, mask `0.0599`, total `0.8249`.

Stage 2, 100 epochs:

- Output directory: `output/sam3d_cat_full/stage2_100epochs_real_sam3d`.
- Used all 90 frames, 256x256 render, 12,000 sampled SAM3D Gaussian points, lr `0.003`.
- `stage2_losses.json` decreased from total `0.8196` to `0.5335`.
- Independent evaluation in `eval_metrics.json`:
  - total `0.8196 -> 0.5303` (`35.29%` lower)
  - color `0.2056 -> 0.1691` (`17.76%` lower)
  - mask 01 error `0.0614 -> 0.0361` (`41.17%` lower)
  - mask IoU `0.2192 -> 0.4566` (`+0.2374`, about `108.30%` relative)
  - mean skeleton joint L2 displacement `0.2029`, max `0.5781`
- Rendered videos:
  - `output/sam3d_cat_full/stage2_100epochs_real_sam3d/rendered_videos/initial_render.mp4`
  - `output/sam3d_cat_full/stage2_100epochs_real_sam3d/rendered_videos/optimized_render.mp4`
  - `output/sam3d_cat_full/stage2_100epochs_real_sam3d/rendered_videos/gt_initial_optimized_rgb.mp4`

Caveat:

- The optimization clearly improves the current canonical renderer metrics, but visual previews still show a compact Gaussian render rather than a full cat silhouette. This points to remaining camera/coordinate calibration work: SAM3D's actual reconstruction camera should be persisted and used instead of the current default canonical camera when comparing against the original monocular video frames.

## 2026-06-18 Frame-0 Camera Alignment and Zero-Gravity Run

Implemented:

- Added `dpd3dgs_animal.camera_calibration`.
- Stage 1 now fits a static frame-0 camera by rendering sampled SAM3D Gaussian centers against the transparent RGBA frame-0 alpha mask.
- The fitted camera is stored in the Stage 1 NPZ as `camera_intrinsics` and `camera_world_to_camera`; Stage 2 loads this camera before falling back to the default canonical camera.
- Stage 1 also stores `gravity`; the Cat config and Stage 2 loss log use `gravity: 0.0`.
- This keeps the alignment chain explicit: SAM3D mesh/3DGS defines canonical object space, MocapAnything joints are similarity-fit to that mesh, ElasticSimulator/TetGen builds tets from the same mesh, and the frame-0 camera maps that shared 3D space into the video frame.

Run:

- Stage 1: `output/sam3d_cat_full/stage1_camera_aligned_g0`
- Stage 2: `output/sam3d_cat_full/stage2_100epochs_camera_aligned_g0`
- Camera fit selected view direction `[-1, 0, 0]`, up `[0, 0, 1]`.
- Frame-0 static camera fit: mask IoU `0.4823`, mask 01 error `0.0382`.
- Stage 1 render mean at 256x256: color `0.2010`, mask `0.0389`, total `0.5898`.
- Stage 2 confirms `camera_source: stage1_npz` and `gravity: 0.0`.

Independent 90-frame evaluation:

- Initial total `0.5454`, color `0.1720`, mask 01 error `0.0373`, mask IoU `0.4550`.
- Optimized total `0.3580`, color `0.1178`, mask 01 error `0.0240`, mask IoU `0.6434`.
- Relative improvement: total `34.35%`, color `31.52%`, mask 01 error `35.65%`.
- Mask IoU improved by `+0.1884` absolute, about `41.41%` relative.
- Mean skeleton joint L2 displacement `0.1203`, max `0.4899`.

Artifacts:

- Camera alignment: `output/sam3d_cat_full/stage1_camera_aligned_g0/camera_alignment.json`
- Optimized joints: `output/sam3d_cat_full/stage2_100epochs_camera_aligned_g0/optimized_joints.npy`
- Metrics: `output/sam3d_cat_full/stage2_100epochs_camera_aligned_g0/eval_metrics.json`
- Rendered videos:
  - `output/sam3d_cat_full/stage2_100epochs_camera_aligned_g0/rendered_videos/initial_render.mp4`
  - `output/sam3d_cat_full/stage2_100epochs_camera_aligned_g0/rendered_videos/optimized_render.mp4`
  - `output/sam3d_cat_full/stage2_100epochs_camera_aligned_g0/rendered_videos/gt_initial_optimized_rgb.mp4`

Remaining limitation:

- The camera-aligned render is now a side-view cat silhouette rather than the previous vertical blob, and metrics improve substantially. Remaining mismatch is local shape/detail rather than the dominant camera-coordinate failure; likely next improvements are richer camera refinement, denser Gaussian binding, and a more articulated skinning model.

## 2026-06-18 Fixed Mask, 120k GS, Multi-Vertex Surface Binding

Problem diagnosis:

- The previous alpha mask used per-pixel distance from the gray background. Cat fur contains gray/dark texture close to the background, so body pixels were misclassified as transparent holes.
- The high-resolution renderer was using only 12k sampled SAM3D Gaussian points. At 1920x1080 this produced visibly sparse/blocky splats.
- The old Gaussian binding was one nearest triangle plus a fixed local offset. During skeleton/tet deformation this can preserve off-surface offsets and create cracks between adjacent regions.

Implemented:

- Added `dpd3dgs_animal.preprocess.video_to_transparent_frames`.
- New mask logic marks only border-connected background-color regions as transparent, then fills foreground-internal holes. This keeps cat body opaque and leaves true background gaps transparent.
- Added multi-vertex surface binding in `dpd3dgs_animal.gaussian`: each GS point is projected/pulled to the surface and driven by inverse-distance weights over 8 nearest surface vertices.
- Stage 2 uses `gaussian_binding_k: 8` and `pull_gaussians_to_surface: true`.
- Increased Cat run from 12k to 120k Gaussian render points.
- Stage 1 preview renderer now uses Gaussian splats instead of square point blocks.
- Camera fitting now uses the same pulled-to-surface rest GS centers as the renderer.

Run:

- Output root: `output/sam3d_cat_fixed`
- Fixed masks: `output/sam3d_cat_fixed/masks`
- SAM3D mesh: `output/sam3d_cat_fixed/sam3d/sam3d_mesh.glb`
- SAM3D Gaussian: `output/sam3d_cat_fixed/sam3d/sam3d_gaussian.ply`, 259,424 points
- Tet mesh for physics: `output/sam3d_cat_fixed/sam3d/sam3d_mesh_simplified_5k_meshfix.obj`
- Stage 1: `output/sam3d_cat_fixed/stage1_native_fixed_mask_multibind`
- Stage 2: `output/sam3d_cat_fixed/stage2_native_fixed_mask_multibind_100ep`

Stage 1:

- Native render size: 1920x1080
- TetGen output: 11,244 tet nodes and 43,162 tets
- Camera fit: mask IoU `0.4145`, mask 01 error `0.0388`
- Initial mean losses: color `0.2262`, mask `0.0424`, total `0.6504`

Stage 2 100 epochs:

- Confirmed `render_size: [1920, 1080]`, `render_resolution_source: input_frame`
- Confirmed `max_render_points: 120000`, `gaussian_binding_k: 8`, `pull_gaussians_to_surface: true`, `gravity: 0.0`
- total `0.6605 -> 0.4912` (`25.64%` lower)
- render `0.6605 -> 0.4898` (`25.84%` lower)
- color `0.2270 -> 0.1847` (`18.63%` lower)
- mask 01 error `0.04335 -> 0.03051` (`29.62%` lower)

Remaining note:

- Full 259k-point frame0 preview was similar to 120k-point preview, so remaining visible gaps are mostly due to SAM3D reconstruction/mesh repair local coverage rather than point subsampling. The new binding prevents one-triangle tearing, but cannot synthesize missing SAM3D surface content.

## 2026-06-18 Strict SAM3D Camera-Space Cat Rerun

Run configuration:

- Output root: `output/cat_camera_aligned_20260618`
- Native render/loss resolution: 1920x1080
- SAM3D camera convention: OpenCV camera space, x right, y down, z forward
- Fixed intrinsics: fx/fy `1570.368`, principal point `(949, 543)`
- MocapAnything: 90 frames, 44 Cat joints
- ElasticSimulator/TetGen: 11,705 nodes, 43,101 tetrahedra
- Surface: 9,833 vertices, 19,666 triangles
- Gaussian render points: 120,000
- Binding: topology-local 8-vertex weighted surface displacement
- Gravity: 0
- Optimization: 100 epochs, Adam, learning rate 0.003

Camera and Stage 1 checks:

- SAM3D point-map reprojection median error: `1.292 px`
- Static projected point-mask IoU: `0.6965`
- Stage 1 frame-0 mask 01 error: `0.02117`
- Stage 1 90-frame mean: color `0.14826`, mask `0.03602`, total `0.50850`

Independent 90-frame evaluation:

- Total: `0.50489 -> 0.22406` (`55.62%` lower)
- Color MAE: `0.14974 -> 0.10804` (`27.85%` lower)
- Color PSNR: `14.34 -> 16.69 dB`
- Mask 01 error: `0.03551 -> 0.01160` (`67.33%` lower)
- Mask IoU: `0.56474 -> 0.81693`
- Mask precision: `0.66187 -> 0.90104`
- Mask recall: `0.79131 -> 0.89789`
- Mean skeleton-node displacement: `0.11145`; maximum: `0.46864`

Artifacts:

- Loss history: `output/cat_camera_aligned_20260618/stage2_100ep/stage2_losses.json`
- Independent metrics: `output/cat_camera_aligned_20260618/stage2_100ep/evaluation_metrics.json`
- Loss curve: `output/cat_camera_aligned_20260618/stage2_100ep/loss_curve.png`
- Comparison sheet: `output/cat_camera_aligned_20260618/stage2_100ep/comparison_gt_initial_optimized.png`
- GT/optimized video: `output/cat_camera_aligned_20260618/stage2_100ep/comparison_gt_optimized.mp4`
- Optimized render: `output/cat_camera_aligned_20260618/stage2_100ep/optimized_render.mp4`

Remaining limitation:

- The fixed camera and mask now give a stable initialization and strong silhouette improvement. Middle motion frames still show compressed or misplaced legs. The dominant remaining issue is the linear skeleton-to-tet displacement model and the single-view SAM3D geometry, not a global camera-coordinate failure.

## 2026-06-19 Cat Stage-by-Stage Rigging Audit

Output root:

- `output/cat_pipeline_audit_20260619`

Confirmed working:

- MP4/RMBG input and masks.
- SAM3D static camera-space reconstruction.
- OpenCV camera convention and fixed intrinsics.
- MocapAnything position inference.
- MocapAnything `video2pose2rot` pose/rot6d NPY inference.
- TetGen tetrahedralization and boundary extraction.
- Multi-vertex GS-to-surface binding.
- Native-resolution differentiable rendering and gradients to skeleton nodes.

Fixed defects:

- `joint_relation` is an all-pairs graph-distance matrix, not binary
  adjacency. The old parser made 43 joints children of Hips. Parents now come
  directly from the reference BVH; Cat has only one direct root child.
- Added tet-graph geodesic bone weights and differentiable DQS.
- Added `dpd3dgs-animal diagnose` with input, SAM3D, skeleton, skinning,
  motion, tet-surface and GS contact sheets.
- Modified vendored `video2pose2rot` preprocessing to load DINO components
  directly and skip optional BVH/Blender export when character assets are
  absent.

Root cause of leg collapse:

- The Mocap skeleton is visually reasonable in frames 0/30/60/89.
- The tet boundary is already collapsed before GS attachment, so GS binding
  is not the primary cause.
- Several reference bones lie outside the reconstructed tet volume; nearest
  distances reach roughly `0.48` for head/tail helper bones.
- SAM3D mesh has no transferred character rig. Euclidean/geodesic LBS and DQS
  remain automatic approximations and cannot establish anatomical
  correspondence.
- The integrated Python path currently uses ElasticSimulator for
  tetrahedralization only. A follow-up audit found that the vendored
  ElasticSimulator C++/CUDA solver does have kinematic-cylinder Dirichlet
  constraints and an `ArmBendGPU` example; that constrained solver is not yet
  bridged into `dpd3dgs_animal`.

Short DQS optimization:

- Stage 1: `output/cat_pipeline_audit_20260619/12_stage1_dqs`
- Stage 2, 10 epochs:
  `output/cat_pipeline_audit_20260619/13_stage2_dqs_10ep`
- total `0.4891 -> 0.3823`
- color `0.1453 -> 0.1277`
- mask 01 error `0.03438 -> 0.02546`

The image loss falls, but frames 30 and 60 remain anatomically invalid and
tet edge distortion is not consistently reduced. More optimization epochs
are not justified until skeleton embedding and constrained FEM are added.

Final audit:

- `output/cat_pipeline_audit_20260619/14_diagnostics_final/03_mocap_skeleton_overlay.png`
- `output/cat_pipeline_audit_20260619/14_diagnostics_final/05_motion_stage_comparison.png`
- `output/cat_pipeline_audit_20260619/14_diagnostics_final/06_tet_surface_vs_gs.png`
- `output/cat_pipeline_audit_20260619/14_diagnostics_final/pipeline_diagnostics.json`

## 2026-06-19 ElasticSimulator Constraint Audit

Finding:

- Upstream ElasticSimulator contains solver-level kinematic constraints:
  `AddKinematicCylinder`, `AttachKinematicConstraints`,
  `RotateKinematicCylinderXKeepingLocalPoint`, and
  `UpdateKinematicConstraints`.
- The GPU explicit, dense implicit, and sparse implicit paths apply those
  constraints through per-DoF target flags and Dirichlet projection.
- `Example/ArmBendGPU/main.cpp` is the relevant sample. It attaches tet
  vertices to two cylinders and bends the forearm with sparse implicit FEM.
- The current project did not use this path because the Python integration
  imported only the Taichi/TetGen helper and then used PyTorch DQS for the
  differentiable chain.

Verification:

- `third_party/elastic_simulator/build-cuda118/bin/sparseCG` ran successfully.
- `third_party/elastic_simulator/build-cuda118/bin/ExplicitFEM_CPU` started the
  sample FEM simulation and exported OBJ frames under a short timeout.

Detailed notes:

- `docs/elastic_simulator_constraint_audit.md`

## 2026-06-19 Headless ElasticSimulator Bridge

Implemented:

- Added `SetKinematicCylinderSegment(start, end, radius)` to
  `ElasticitySolverT` so arbitrary Mocap bone segments can update a kinematic
  cylinder pose each frame.
- Added `Example/HeadlessConstrainedFEM`, a non-viewer C++/CUDA executable
  that reads TetGen `.node/.ele` plus a binary skeleton sequence, attaches
  kinematic cylinders, runs `IMPLICIT_SPARSE` FEM, and writes per-frame tet
  vertices to a binary stream.
- Added `src/dpd3dgs_animal/elastic_bridge.py` and CLI command
  `dpd3dgs-animal elastic-forward`.
- The Python bridge exports Stage 1 NPZ to TetGen files, runs the C++ driver,
  reads tet vertices back, extracts surface vertices, attaches existing
  Gaussian bindings, renders at native resolution, and writes losses/previews.
- Cat config now uses larger FEM handle radii:
  `radius_scale=0.25`, `min_radius_scale=0.025`, `max_radius_scale=0.12`,
  with gravity still `0`.

Verification before clearing output:

- Built `third_party/elastic_simulator/build-dpd3dgs/bin/HeadlessConstrainedFEM`.
- `elastic-forward` 2-frame FEM-only smoke ran successfully.
- Cat config smoke attached 21 kinematic handles and 1096 constrained tet
  vertices.
- 2-frame full chain smoke
  `ElasticSimulator -> surface -> attached 3DGS -> native 1920x1080 render/loss`
  ran successfully.
- Existing `tests/test_skinning.py` still passes: `3 passed`.

Remaining:

- This bridge is a real constrained FEM forward path, but not yet a
  differentiable ElasticSimulator adjoint. Stage 2 optimization still uses the
  PyTorch DQS surrogate unless/until a differentiable constrained solve is
  implemented.

## 2026-06-19 Cleared Output And Reran Cat With Elastic Forward

Output cleanup:

- Cleared the root `output/` directory before rerunning.
- New run root: `output/cat_elastic_constrained_20260619`.

Additional fixes during rerun:

- `MocapAnythingAdapter` now uses `sys.executable` instead of hard-coded
  `python` for subprocesses, so non-login conda invocations work.
- Added Stage 1 mesh simplification/repair before TetGen:
  `open3d` quadric decimation followed by `pymeshfix`.
- Added adaptive chunk sizing in GS-to-surface binding to avoid CUDA OOM when
  the surface has many faces.
- Added `SetVertexPositions(..., zeroVelocity)` to ElasticSimulator and made
  the headless driver default to per-frame quasi-static reset. Cumulative time
  integration diverged around frame 30; reset-each-frame removed NaNs and kept
  PCG iterations bounded.

Rerun configuration:

- Input video: `samples/mocap_anything/zoo/video/Cat#Cat-Walk/y30.mp4`
- Frames/masks: RMBG, 90 frames, 1920x1080
- SAM3D original camera mesh: 180,344 vertices, 360,632 faces
- Physical mesh after simplification/repair: 2,783 vertices, 5,566 faces
- ElasticSimulator/TetGen: 13,671 tet nodes, 50,700 tets
- Surface: 11,298 vertices, 22,596 faces
- Kinematic handles: 21
- Constrained tet vertices: 744
- FEM params: `dt=0.001`, `E=100000`, `nu=0.4`, `density=1000`,
  `damping=0.02`, `gravity=0`, `reset_each_frame=true`

Results:

- Stage 1 DQS mean loss:
  - color `0.14363`
  - mask `0.03524`
  - total `0.49605`
- Elastic constrained FEM forward mean loss:
  - color `0.14003`
  - mask `0.03167`
  - total `0.45673`
- Relative total loss improvement over Stage 1: about `7.9%`.
- Elastic forward tet vertices contain no NaNs.
- Tet displacement from rest:
  - global mean `0.00621`
  - max `0.61724`
  - frame 30 mean/max `0.00674 / 0.52114`
  - frame 60 mean/max `0.00823 / 0.27910`
  - frame 89 mean/max `0.00310 / 0.21494`

Artifacts:

- `output/cat_elastic_constrained_20260619/stage1/stage1_summary.json`
- `output/cat_elastic_constrained_20260619/stage1/renders/losses.json`
- `output/cat_elastic_constrained_20260619/elastic_forward/elastic_forward_summary.json`
- `output/cat_elastic_constrained_20260619/elastic_forward/renders/elastic_forward_losses.json`
- `output/cat_elastic_constrained_20260619/elastic_forward/renders/elastic_render.mp4`
- Preview frames:
  - `elastic_00000.png`
  - `elastic_00030.png`
  - `elastic_00060.png`
  - `elastic_00089.png`

## 2026-06-19 Differentiable Constrained FEM Stage 2

Implemented:

- Added `src/dpd3dgs_animal/fem_optimize.py`.
- Added CLI command `dpd3dgs-animal elastic-stage2`.
- The new optimizer uses per-frame skeleton nodes as parameters, builds
  skeleton-segment soft Dirichlet targets, assembles a linear tetrahedral FEM
  stiffness matrix from the Stage 1 tet mesh, solves the constrained equilibrium
  with differentiable unrolled CG, extracts the surface, drives attached SAM3D
  Gaussians, renders at native resolution, and back-propagates color + mask loss
  to skeleton node coordinates.
- Added config fields:
  `fem_cg_iters`, `fem_elastic_stiffness`, `fem_handle_stiffness`,
  `fem_diagonal_reg`, `fem_handle_support`, `fem_handle_weight_power`.
- Cat config now uses `fem_cg_iters=16`.
- Added `tests/test_fem_optimize.py`, verifying that the constrained FEM solve
  is finite and produces non-zero gradients on skeleton joints.
- Preview saving now uses distributed frames `0/30/60/last` instead of only the
  first four frames.

Verification:

- `PYTHONPATH=src python -m pytest -q tests/test_skinning.py tests/test_fem_optimize.py`
  - `4 passed`
- Cat smoke:
  - output: `output/cat_elastic_stage2_smoke_20260619`
  - 2 frames, 1 step, native `1920x1080`
  - render loss `0.33462 -> 0.33198`
  - grad norm `0.20719`
- Cat partial run:
  - output: `output/cat_elastic_stage2_20260619`
  - 30 frames, 20 steps, native `1920x1080`
  - render loss `0.47786 -> 0.43803`
- Cat full run:
  - output: `output/cat_elastic_stage2_full100_20260619`
  - 90 frames, 100 steps, native `1920x1080`
  - 120k Gaussian points
  - 13,671 tet nodes / 50,700 tets
  - 41 skeleton handles / 3,744 constrained tet nodes
  - final render loss `0.38502`
  - visual QA failed: preview shows leg blurring/merging despite lower 2D loss.
    Do not treat this directory as a successful reconstruction checkpoint.

Full-run loss comparison:

| method | color | mask | total/render |
|---|---:|---:|---:|
| Stage1 DQS baseline | 0.14363 | 0.03524 | 0.49605 |
| ElasticSimulator constrained FEM forward | 0.14003 | 0.03167 | 0.45673 |
| PyTorch constrained FEM stage2 step 0 | 0.14020 | 0.03163 | 0.45653 |
| PyTorch constrained FEM stage2 final eval | 0.12589 | 0.02591 | 0.38502 |

Interpretation:

- `output/cat_elastic_stage2_20260619` is the visually better run so far
  (30 frames, 20 steps).
- `output/cat_elastic_stage2_full100_20260619` is an over-optimized/failed run:
  scalar render loss improves, but the geometry is worse.
- The failure is caused by the current objective being dominated by 2D
  color/mask alignment. Bone-length, temporal, mocap-prior, and geometric QA
  terms are not strong enough to reject bad 3D deformations.

Remaining:

- This is a PyTorch differentiable linear FEM solve on the same tet topology,
  not an adjoint of ElasticSimulator's CUDA sparse implicit solver.
- Visual previews remain imperfect: local Gaussian block/outlier artifacts and
  leg detail issues are still visible.
- Next engineering targets are skeleton embedding inside the tet volume,
  anatomy-aware volumetric weights, branch-limited capsule constraints, and
  Gaussian outlier filtering.
