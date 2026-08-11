# ElasticSimulator Constraint Audit

Date: 2026-06-19

## Conclusion

ElasticSimulator is not just a tetrahedralizer. Its C++/CUDA solver already
implements solver-level kinematic constraints through cylinder handles and
applies them as Dirichlet constraints in the GPU explicit, dense implicit, and
sparse implicit paths.

The current `dpd3dgs_animal` integration has not used that path yet. It imports
ElasticSimulator's Python/Taichi helper for tetrahedralization and surface
extraction, then performs the skeleton-driven deformation in PyTorch with DQS
and regularizers. That statement is the precise limitation: the integrated
pipeline does not yet bridge into ElasticSimulator's C++ constrained solver.

## Evidence In ElasticSimulator

Public API in `third_party/elastic_simulator/include/Solver.h`:

- `AddKinematicCylinder(center, radius, height)`
- `AttachKinematicConstraints(shapeID)`
- `RotateKinematicCylinderXKeepingLocalPoint(shapeID, radians, localPoint, worldPin)`
- `UpdateKinematicConstraints()`
- `GetKinematicConstraintCount()`

Constraint state stored by the solver:

- `h_kinematicCylinders`
- `h_kinematicConstraints`
- `h_constraint_dof_flags`
- `h_constraint_dof_targets`
- CUDA copies `d_constraint_dof_flags` and `d_constraint_dof_targets`

Implementation in `third_party/elastic_simulator/src/Solver/Solver.cpp`:

- `AttachKinematicConstraints` scans tetrahedral vertices and attaches vertices
  inside a cylinder.
- Each attached vertex stores its local cylinder-space position.
- `UpdateKinematicConstraints` converts those local points back to current
  world-space targets and uploads per-DoF flags/targets to CUDA.
- `SimulateFrame` calls `UpdateKinematicConstraints` before stepping the solver.

GPU solver implementation in `third_party/elastic_simulator/src/Solver/Solvergpu.cu`:

- `k_applyKinematicTargets` hard-sets constrained vertex coordinates and zeroes
  constrained velocity components.
- `k_applyDenseDirichletConstraints` and `k_applyDenseDirichletRhs` apply dense
  implicit Dirichlet constraints.
- `k_applyCsrDirichletConstraints` applies sparse implicit Dirichlet
  constraints.
- `Step_Explicit`, `Step_Implicit`, and `Step_Implicit_Sparse` all call the
  constraint kernels.

## Existing Example

`third_party/elastic_simulator/Example/ArmBendGPU/main.cpp` is the relevant
sample.

It builds a simple arm-like cylinder mesh, initializes the sparse implicit GPU
solver, creates two kinematic cylinders, attaches tet vertices to them, and
rotates the forearm cylinder around a pinned local point:

```cpp
const int shoulder = solver.AddKinematicCylinder(...);
const int forearm = solver.AddKinematicCylinder(...);
solver.AttachKinematicConstraints(shoulder);
solver.AttachKinematicConstraints(forearm);
solver.RotateKinematicCylinderXKeepingLocalPoint(...);
solver.AdvanceFrame(false);
```

The example uses the OpenGL realtime viewer, so it is interactive and does not
terminate by itself in a headless command. The built binaries are present under:

```text
third_party/elastic_simulator/build-cuda118/bin/
```

I verified non-GUI examples in this vendored tree:

- `sparseCG` runs and prints the expected sparse matrix/solution check.
- `ExplicitFEM_CPU` starts simulation, tetrahedralizes the sample mesh, and
  exports OBJ frames before the short timeout.

## What Is Missing For This Project

To use the existing constraints for animal reconstruction, the missing work is
not FEM itself, but the integration layer:

1. A C++ executable or pybind/ctypes bridge that exposes constrained solver
   setup and per-frame stepping to Python.
2. A mapping from Mocap skeleton bones to kinematic cylinders/capsules in the
   SAM3D/OpenCV camera-space tet mesh.
3. Per-frame bone transform updates from Mocap joint coordinates, not only the
   demo's fixed X-axis rotation helper.
4. Host-side access to the simulated tet vertices or surface vertices after
   each step for GS binding and rendering. The existing API exposes CUDA device
   vertices for the viewer, but not a Python-friendly array interface.
5. Differentiation. The existing C++ solver is forward simulation only. To
   optimize skeleton node coordinates from render loss, we still need implicit
   differentiation through the constrained solve, an unrolled differentiable
   solver, or a surrogate used explicitly as such.

## Practical Next Step

The right next implementation step is to add a headless C++/CUDA driver based
on `ArmBendGPU`, but replacing the hard-coded arm handles with a per-frame
Mocap bone-cylinder schedule:

```text
SAM3D mesh -> ElasticSimulator tet mesh
Mocap joints -> bone cylinders/capsules in OpenCV camera space
attach tet vertices inside each cylinder
for each video frame:
  update each cylinder transform from joint endpoints
  AdvanceFrame()
  copy simulated vertices to host
  update surface + attached GS
  render/loss
```

This will give a real constrained-FEM forward path. The current PyTorch DQS
path can remain as a differentiable baseline until the adjoint/unrolled solver
is implemented.
