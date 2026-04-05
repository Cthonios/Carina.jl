# Carina.jl Reference Guide

Complete reference for YAML input configuration, solver combinations, and
material models.

---

## Table of Contents

- [YAML Input Structure](#yaml-input-structure)
- [Material Models](#material-models)
- [Time Integrators](#time-integrators)
- [Solvers](#solvers)
- [Preconditioners](#preconditioners)
- [Solver Compatibility Matrix](#solver-compatibility-matrix)
- [Boundary Conditions](#boundary-conditions)
- [Initial Conditions](#initial-conditions)
- [Output Control](#output-control)
- [Function Expressions](#function-expressions)

---

## YAML Input Structure

A minimal input file:

```yaml
type: single

input mesh file:  mesh.g
output mesh file: output.e

model:
  type: solid mechanics
  material:
    blocks:
      my_block: neohookean
    neohookean:
      elastic modulus: 10.0e9
      Poisson's ratio: 0.25
      density: 1000.0

time integrator:
  type: quasi static
  initial time: 0.0
  final time:   1.0
  time step:    0.1

boundary conditions:
  Dirichlet:
    - sideset: face_z_minus
      component: z
      function: "0.0"

solver:
  type: newton
  linear solver:
    type: direct
```

### Top-Level Keys

| Key | Required | Description |
|-----|----------|-------------|
| `type` | yes | Simulation type. Currently only `single`. |
| `device` | no | `cpu` (default), `cuda`, or `rocm`. |
| `input mesh file` | yes | Path to Exodus (.g) mesh. |
| `output mesh file` | yes | Path for Exodus (.e) output. |
| `output interval` | no | Write output every N steps (default: 1). |
| `model` | yes | Physics model and material definitions. |
| `time integrator` | yes | Time stepping scheme and parameters. |
| `boundary conditions` | yes | Dirichlet and/or Neumann BCs. |
| `solver` | yes | Nonlinear and linear solver configuration. |
| `body forces` | no | Volumetric source terms. |
| `initial conditions` | no | Displacement and/or velocity ICs. |
| `output` | no | Control which fields are written. |
| `quadrature` | no | Override quadrature rule. |

---

## Material Models

All materials require `density` (kg/m^3).  Elastic constants can be
specified using any of the listed aliases.

### Elastic Constants

Any two of the following define an isotropic elastic material:

| YAML key | Aliases |
|----------|---------|
| `elastic modulus` | `Young's modulus`, `youngs modulus` |
| `Poisson's ratio` | `poissons ratio` |
| `bulk modulus` | |
| `shear modulus` | |
| `Lame's first constant` | `lames first constant` |

### Hyperelastic Models

| YAML name | Aliases | Constants | Notes |
|-----------|---------|-----------|-------|
| `neohookean` | `neo-hookean`, `neo hookean` | E, nu | Finite deformation. Coincides with linear elastic at small strain. General-purpose default. |
| `linear elastic` | `linearelastic` | E, nu | Small-strain only. Faster than neo-Hookean but invalid for large deformation. |
| `hencky` | | E, nu | Logarithmic (Hencky) strain measure. Good for moderate strains. |
| `saint venant kirchhoff` | `svk`, `saintvenant-kirchhoff` | E, nu | Green-Lagrange strain. Simple but unstable in compression; avoid for large strains. |
| `seth-hill` | `seth hill`, `sethhill` | E, nu, m, n | Generalized strain family parameterized by exponents m, n. |

### Elastoplastic Models

| YAML name | Aliases | Extra constants | Notes |
|-----------|---------|-----------------|-------|
| `linear elasto plasticity` | | E, nu, `yield stress`, `hardening modulus` | Small-strain J2 plasticity with linear isotropic hardening. |
| `j2 plasticity` | `finitedefj2plasticity` | E, nu, `yield stress`, `hardening modulus` | Finite-deformation J2 plasticity. Path-dependent; requires small time steps. |

### Example

```yaml
model:
  type: solid mechanics
  material:
    blocks:
      block_1: neohookean
      block_2: j2 plasticity
    neohookean:
      elastic modulus: 200.0e9
      Poisson's ratio: 0.3
      density: 7800.0
    j2 plasticity:
      elastic modulus: 70.0e9
      Poisson's ratio: 0.36
      density: 2700.0
      yield stress: 250.0e6
      hardening modulus: 0.7e9
```

---

## Time Integrators

### Quasi-Static

For problems with no inertia (statics, slow loading).

```yaml
time integrator:
  type: quasi static
  initial time: 0.0
  final time:   1.0
  time step:    0.1
```

Adaptive time stepping (all four keys required together, or omit all):

```yaml
  minimum time step: 0.001
  maximum time step: 0.1
  decrease factor:   0.5    # multiply dt on Newton failure (must be < 1)
  increase factor:   1.5    # multiply dt on Newton success (must be > 1)
```

| Key | Default | Description |
|-----|---------|-------------|
| `initial equilibrium` | `false` | Solve R=0 at t=0 before advancing. |

### Newmark (Implicit Dynamics)

Second-order implicit time integration for structural dynamics.

```yaml
time integrator:
  type: newmark
  initial time: 0.0
  final time:   0.01
  time step:    1.0e-4
  beta:  0.3025
  gamma: 0.6
  alpha: -0.1     # HHT-alpha (0 = standard Newmark)
```

| Key | Default | Range | Description |
|-----|---------|-------|-------------|
| `beta` | 0.25 | > 0 | Controls displacement accuracy. 0.25 = trapezoidal rule (no numerical damping). |
| `gamma` | 0.5 | >= 0.5 | Controls velocity accuracy. 0.5 = no numerical damping. |
| `alpha` | 0.0 | [-1/3, 0] | HHT-alpha parameter. Negative values add algorithmic damping to suppress high-frequency noise. |

Standard choices:
- **Trapezoidal rule** (no damping): beta=0.25, gamma=0.5, alpha=0
- **HHT-alpha** (mild damping): beta=0.3025, gamma=0.6, alpha=-0.1

Adaptive time stepping is supported (same keys as quasi-static).

### Central Difference (Explicit Dynamics)

Conditionally stable explicit integration.  No nonlinear solve required.

```yaml
time integrator:
  type: central difference
  initial time: 0.0
  final time:   0.001
  time step:    1.0e-7
  gamma: 0.5
```

| Key | Default | Description |
|-----|---------|-------------|
| `gamma` | 0.5 | Velocity update parameter. 0.5 = standard central difference. |

The stable time step is limited by the CFL condition.  Carina estimates
and prints the stable dt at startup; set your `time step` at or below this
value.

---

## Solvers

### Nonlinear Solvers

| YAML `type` | Aliases | Needs linear solver? | GPU? | Description |
|-------------|---------|---------------------|------|-------------|
| `newton` | | Yes | Via linear solver | Standard Newton-Raphson. Assembles Jacobian each iteration. Quadratic convergence near the solution. **Recommended for most problems.** |
| `nlcg` | `nonlinear cg` | No | Yes | Nonlinear conjugate gradient (Polak-Ribiere). Matrix-free. Linear convergence. Best for GPU where Jacobian assembly is expensive. |
| `steepest descent` | | No | Yes | Preconditioned steepest descent. Matrix-free. Slowest convergence but most robust. Useful as a fallback. |

All nonlinear solvers share these keys:

| Key | Default | Description |
|-----|---------|-------------|
| `maximum iterations` | 20 | Max Newton/CG/SD iterations per time step. |
| `minimum iterations` | 0 | Force at least this many iterations. |
| `absolute tolerance` | 1e-10 | Converged when \|R\| < tol. |
| `relative tolerance` | 1e-14 | Converged when \|R\|/\|R_0\| < tol. |
| `use line search` | false | Enable Armijo backtracking line search. |
| `line search backtrack factor` | 0.5 | Step reduction per backtrack. |
| `line search decrease factor` | 1e-4 | Armijo sufficient decrease parameter. |
| `line search maximum iterations` | 10 | Max backtracking steps. |

NLCG additional keys:

| Key | Default | Description |
|-----|---------|-------------|
| `orthogonality tolerance` | 0.5 | Restart CG when consecutive gradients lose orthogonality. |
| `restart interval` | 0 | Force restart every N iterations (0 = disabled). |

### Linear Solvers

Specified inside the `solver:` block under `linear solver:`.

| YAML `type` | Aliases | CPU | GPU | Description |
|-------------|---------|-----|-----|-------------|
| `direct` | | Yes | **No** | Sparse LU factorization. Most robust, no tolerance tuning. **Recommended baseline for CPU.** |
| `iterative` | `cg`, `krylov`, `conjugate gradient`, `minres` | Yes | Yes | Conjugate Gradient (always CG regardless of alias). Requires SPD system. |
| `lbfgs` | | Yes | Yes | L-BFGS quasi-Newton. Matrix-free, no Jacobian assembly. Approximates the inverse Hessian from gradient history. |
| `none` | | Yes | Yes | No linear solve (used with NLCG / steepest descent). |

Iterative solver keys:

| Key | Default | Description |
|-----|---------|-------------|
| `maximum iterations` | 1000 | Max CG iterations. |
| `tolerance` | 1e-8 | Relative residual tolerance for CG. |
| `preconditioner` | none | See [Preconditioners](#preconditioners). |

L-BFGS solver keys:

| Key | Default | Description |
|-----|---------|-------------|
| `history size` | 10 | Number of gradient pairs stored. More = better Hessian approximation but more memory. |

---

## Preconditioners

Specified inside `linear solver:` under `preconditioner:`.

| YAML `type` | CPU | GPU | Cost per CG iter | Description |
|-------------|-----|-----|-------------------|-------------|
| `jacobi` | Yes | Yes | 1 vector scale | Diagonal scaling. Cheap but weak. Always available. |
| `ic` | Yes | **No** | 1 triangular solve | Incomplete LDL^T factorization. Strong for ill-conditioned systems. **Recommended for CPU iterative solves.** |
| `incomplete cholesky` | Yes | **No** | 1 triangular solve | Alias for `ic`. |
| `chebyshev` | Yes | Yes | 2k matvecs | Chebyshev polynomial (degree k). Matrix-free, GPU-friendly. Stronger than Jacobi but weaker than IC. Experimental. |
| *(none)* | Yes | Yes | 0 | No preconditioning. |

Chebyshev keys:

| Key | Default | Description |
|-----|---------|-------------|
| `degree` | 5 | Polynomial degree. Higher = stronger but more expensive (2k matvecs per CG iteration). |

### Preconditioner Examples

```yaml
# Jacobi (CPU or GPU)
preconditioner:
  type: jacobi

# Incomplete Cholesky (CPU only)
preconditioner:
  type: ic

# Chebyshev polynomial (CPU or GPU)
preconditioner:
  type: chebyshev
  degree: 5
```

---

## Solver Compatibility Matrix

Not all combinations of integrator, nonlinear solver, linear solver, and
preconditioner are valid or sensible.

### Integrator + Nonlinear Solver

| | Newton | NLCG | Steepest Descent |
|---|:---:|:---:|:---:|
| **Quasi-static** | **Recommended** | OK (GPU) | OK (fallback) |
| **Newmark** | **Recommended** | OK (GPU) | OK (fallback) |
| **Central difference** | N/A (explicit) | N/A | N/A |

Central difference uses no nonlinear solver; it is always explicit.

### Newton: Linear Solver Combinations

| | Direct | CG | CG+Jacobi | CG+IC | CG+Chebyshev | L-BFGS |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **CPU** | **Best** | Weak | OK | **Good** | Experimental | Good |
| **GPU** | **Invalid** | Weak | OK | **Invalid** | Experimental | Good |

**Recommendations by device:**

- **CPU, small-medium problems:** `direct`. No tuning, always converges.
- **CPU, large problems:** `cg` + `ic`. Good convergence (typically 50-100 CG iterations on well-shaped meshes).
- **GPU:** `lbfgs` (no linear solver needed) or `cg` + `jacobi`. `chebyshev` is available but experimental.

### NLCG / Steepest Descent: Preconditioner Combinations

NLCG and steepest descent use their own internal preconditioner (Jacobi
diagonal scaling), not the `linear solver:` block.  They do not use a
linear solver.

| | Jacobi | IC | Chebyshev |
|---|:---:|:---:|:---:|
| **NLCG** | Built-in | N/A | N/A |
| **Steepest Descent** | Built-in | N/A | N/A |

### Invalid Combinations

These will error at startup or produce incorrect results:

| Combination | Problem |
|-------------|---------|
| GPU + `direct` | Direct solver requires sparse matrix assembly (CPU only). |
| GPU + `ic` preconditioner | Incomplete factorization requires assembled sparse matrix. |
| Central difference + any nonlinear solver | Explicit integration has no nonlinear solve step. |
| NLCG/SD + `linear solver:` block | Ignored; these solvers are matrix-free by design. |

### Dubious Combinations

These are valid but likely suboptimal:

| Combination | Why |
|-------------|-----|
| Newton + CG (no preconditioner) | CG without preconditioning is very slow on FEM systems. Always use at least Jacobi. |
| Newton + Chebyshev on CPU | IC is available and much stronger on CPU. Use Chebyshev only on GPU. |
| Quasi-static + steepest descent | Very slow convergence. Use Newton or NLCG instead. |
| Large dt + high-order plasticity | J2 plasticity is path-dependent; large steps can miss the yield surface and cause divergence or DomainErrors. |

---

## Boundary Conditions

### Dirichlet (Essential)

Fix displacement components on surfaces or node sets.

```yaml
boundary conditions:
  Dirichlet:
    - sideset: face_name      # or: nodeset: set_name
      component: x            # x, y, or z
      function: "0.005 * t"   # expression in t, x, y, z
```

Either `sideset` or `nodeset` must be specified (not both).  The names must
match side sets or node sets defined in the Exodus mesh file.

### Neumann (Natural)

Apply traction (force per unit area) on surfaces.

```yaml
boundary conditions:
  Neumann:
    - sideset: face_name
      component: y
      function: "1.0e6 * t"    # Pa (force/area)
```

Neumann BCs require a `sideset` (not nodeset).

### Sources (Body Forces)

Volumetric force density applied to element blocks.

```yaml
body forces:
  - block: block_name     # or "all"
    component: y
    function: "-9.81 * 2700.0"   # N/m^3 (acceleration * density)
```

---

## Initial Conditions

Displacement or velocity fields at t = 0.

```yaml
initial conditions:
  displacement:
    - node set: all_nodes
      component: z
      function: "0.001 * x"
  velocity:
    - node set: all_nodes
      component: x
      function: "100.0 * sin(pi * y / L)"
```

---

## Output Control

By default, only displacement is written.  Enable additional fields:

```yaml
output:
  velocity: true
  acceleration: true
  stress: true
  deformation gradient: true
  internal variables: true   # plasticity state (equivalent plastic strain, etc.)
  recovery: lumped           # lumped (default), consistent, or none
```

The `recovery` method controls how element quadrature-point fields (stress,
etc.) are projected to nodes for visualization:
- **lumped**: Fast diagonal mass matrix. Slight smoothing. Default.
- **consistent**: Full L2 projection. More accurate but requires a linear solve.
- **none**: No nodal projection; skip stress output.

---

## Function Expressions

Boundary conditions, body forces, and initial conditions accept string
expressions that are compiled to Julia functions.  Available variables:

| Variable | Description |
|----------|-------------|
| `t` | Current time |
| `x`, `y`, `z` | Spatial coordinates of the node/integration point |

Expressions support standard math: `+`, `-`, `*`, `/`, `^`, `sin`, `cos`,
`exp`, `log`, `sqrt`, `abs`, `pi`, `atan`, etc.

Multiple statements can be separated by `;`:

```yaml
function: "L = 0.1; A = 0.005; A * sin(pi * x / L) * t"
```

Constants are evaluated once; the entire expression is compiled to an
efficient Julia function.
