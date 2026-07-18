# Features

What Carina supports today. Items marked *planned* are design goals with no
current implementation — they are called out explicitly so the input reference
is not read as promising more than it delivers.

## Execution

- **Single source, three backends.** CPU, NVIDIA (CUDA), and AMD (ROCm)
  from one code base via
  [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl).
- **Vendor-free library.** The GPU packages live in the launcher environment,
  not in Carina itself, so the library installs and runs without CUDA or ROCm
  present.
- **CPU multithreading** of element assembly and constitutive evaluation.
- **Matrix-free explicit path.** Central difference opts the assembler into
  matrix-free mode, skipping sparse preallocation entirely — on a 530k-DOF mesh
  that is several GB of allocation avoided.

## Physics

- **Finite-deformation solid mechanics**, built on
  [ReferenceFiniteElements.jl](https://github.com/Cthonios/ReferenceFiniteElements.jl)
  and
  [FiniteElementContainers.jl](https://github.com/Cthonios/FiniteElementContainers.jl).
- **Material models** from
  [ConstitutiveModels.jl](https://github.com/Cthonios/ConstitutiveModels.jl):
  neo-Hookean, linear elastic, Hencky, Saint Venant–Kirchhoff, Seth–Hill,
  small-strain J2 plasticity, and finite-deformation J2 plasticity.
- *Planned:* heat conduction, multiphysics coupling, contact, and multidomain
  Schwarz coupling. Note that `model.type` is currently parsed but not
  dispatched on — solid mechanics is the only physics.
- *Not supported:* more than one material per mesh. Carina applies a single
  constitutive model and density to the whole domain; see
  [Materials](reference/materials.md).

## Time integration

| Scheme | Kind | Nonlinear solve |
|---|---|---|
| Quasi-static | Equilibrium, no inertia | Newton / NLCG / steepest descent |
| Newmark-β, HHT-α | Implicit dynamics | Newton / NLCG / steepest descent |
| Central difference | Explicit dynamics | none |

- **Adaptive time stepping** for the implicit schemes: shrink on solver
  failure, grow on success, within user bounds.
- **CFL-based stable step estimation** for explicit runs, computed from element
  characteristic length and the material wave speed, with optional periodic
  recomputation as the mesh deforms.
- **Output-interval subcycling.** The integrator takes as many internal steps as
  needed between output frames, landing exactly on output times.

## Solvers

**Nonlinear:** Newton–Raphson with Armijo line search (on by default),
nonlinear conjugate gradient, and preconditioned steepest descent. The latter
two are matrix-free.

**Linear:** sparse direct (LU), conjugate gradient, and L-BFGS.

**Preconditioners:**

| | CPU | GPU |
|---|:---:|:---:|
| Jacobi | ✓ | ✓ |
| Chebyshev polynomial | ✓ | ✓ |
| Incomplete Cholesky (LDLᵀ) | ✓ | — |
| Smoothed-aggregation AMG | ✓ | — |

The AMG preconditioner uses the six rigid-body modes as its near-nullspace,
rebuilt from the **current** configuration at each hierarchy build, which keeps
CG iteration counts nearly independent of the time step. Its hierarchy setup is
lagged and reused. GPU AMG is *planned*; the GPU path currently relies on
Jacobi or Chebyshev.

**Composable termination criteria.** Convergence and failure are expressed as a
tree of status tests — residual, update, iteration count, divergence,
stagnation, and finiteness — combined with `any`/`all` logic. See
[Solvers](reference/solvers.md).

**Dirichlet conditions by DOF elimination**, never by penalty. The reduced
system carries no artificial conditioning, which matters directly for Krylov
convergence.

## Boundary and initial conditions

- **Dirichlet** on side sets or node sets, per component, as a function of
  space and time.
- **Neumann** as surface tractions on side sets, or point loads on node sets.
- **Body forces** per element block.
- **Initial conditions** for displacement and velocity.
- **Traveling-wave initial conditions**, where you supply the displacement
  profile, a propagation axis, and a signed wave speed, and Carina derives the
  consistent velocity field `v₀ = −c·∂u₀/∂s` by **symbolic differentiation** —
  no hand-written derivatives.
- **Expression language** over `x`, `y`, `z`, `t` with named bindings, compiled
  once to an allocation-free scalar function.

## Input and output

- **Exodus** mesh input and result output, readable by ParaView.
- **YAML-driven** setup — no recompilation to change a problem.
- **Selectable output fields:** displacement (always), velocity, acceleration,
  per-quadrature-point stress, deformation gradient, and constitutive state.
- **Nodal recovery** of element quantities by lumped or consistent (full L2)
  projection.
- **Typo detection.** Unknown keys in most sections produce a warning with a
  Levenshtein-based suggestion. Note the sections this does *not* cover, listed
  in the [reference overview](reference/index.md#how-input-errors-surface).

## Scale

The explicit GPU path has been run to **7.8 M DOF** (2.5 M elements) on a
single 8 GB consumer GPU. Throughput is memory-bandwidth bound, and the
GPU-to-CPU advantage widens with problem size before plateauing at roughly 3×
on that hardware — approximately 80% of the memory-bandwidth ratio between the
two devices.
