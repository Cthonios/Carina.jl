# Testing

```bash
julia --project=. test/runtests.jl
```

The full suite runs 208 tests and takes roughly two and a half minutes on a
modern desktop CPU.

## Selecting tests

The runner takes the same flags as Norma's:

```bash
julia --project=. test/runtests.jl              # everything
julia --project=. test/runtests.jl --quick      # fast subset
julia --project=. test/runtests.jl --filter torsion
julia --project=. test/runtests.jl --list       # list tests with indices
julia --project=. test/runtests.jl 1 3 5        # run by index
```

| Flag | Effect |
|---|---|
| `--list` | Print every test with its index and exit. |
| `--filter <substring>` | Run only tests whose name contains the substring. |
| `--quick` | Run the fast subset, skipping the long-running cases. |
| *(bare integers)* | Run tests by index, as reported by `--list`. |

Start with `--list` to see what is available, then `--filter` to iterate on one
area.

## Coverage

The suite exercises each integrator, solver, preconditioner, and boundary- and
initial-condition path end to end, by running complete simulations and checking
results:

- **Integrators** — quasi-static, implicit dynamic (Newmark), explicit dynamic
  (central difference), and a rigid-body case
- **Solvers** — Newton with direct and iterative linear solves, L-BFGS,
  nonlinear CG, steepest descent
- **Preconditioners** — Jacobi, Chebyshev, and AMG (dynamic and quasi-static)
- **Materials** — neo-Hookean, linear elastic, and J2 plasticity
- **Boundary conditions** — Dirichlet, Neumann tractions, point loads, gravity
  body forces
- **Initial conditions** — including 37 tests covering the traveling-wave
  parser and its symbolic derivatives
- **Verification against closed-form solutions** — the clamped-wave cases
  compare explicit and implicit results against the analytical solution of
  Mota, Tezaur & Phlipot, IJNME 123:5036–5071, 2022
- **GPU device verification** — checks backend resolution; the heavier GPU
  paths require actual hardware

## Continuous integration

CI runs the full suite on every push, and nightly on a schedule. The nightly
job clones the three sibling repositories at their `main` branches, so it
tests against the current state of the dependency stack rather than a pinned
snapshot. A nightly failure with a green push therefore usually means a
sibling moved — see [Troubleshooting](troubleshooting.md).

## Adding a test

Tests live in `test/`, one file per case, and are registered in
`test/runtests.jl`. Most follow the same shape: write a YAML input into a
temporary directory, run the simulation, and assert on the results. Copying a
neighbouring case is the fastest way to start — for instance
`test/mechanics-implicit-dynamic-cube-amg.jl`, which reuses an existing example
mesh and varies only the solver configuration.
