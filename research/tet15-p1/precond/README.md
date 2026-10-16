# Preconditioning the TET15-P1 tangent at κ/μ = 10⁴

Study of 2026-10-15 on the punch indentation benchmark (research/tet15-p1,
quarter model), where Carina's implicit solver, conjugate gradients with the
device algebraic-multigrid V-cycle, needs 600 to 1300 iterations per solve on
the TET15-P1 tangent of the nearly incompressible J2 material (bulk modulus
κ = 10¹³ Pa, shear modulus μ = 10⁹ Pa), against 65 to 75 for the FETI
preconditioner of Sierra/SM TET15-P1 on the same
meshes.

## Scripts

- `state.jl deck.yaml out.jls`: runs a deck on the CPU, assembles the tangent
  K(κ) and K(κ' = μ) at the converged state, the element projection
  operators D and M of the volumetric term (D by forward-mode
  differentiation of the element projection θ̄ = M⁻¹ b(u)), and the
  right-hand side of the next Newton step, and checks that
  K(κ) − K(κ') = (κ − κ') D M⁻¹ Dᵀ.
- `experiments.jl state.jls results.tsv [minres-only]`: iteration counts of
  the methods below at κ/μ = 10², 10³, 10⁴, 10⁵, with K(κ) = K_u + κ G,
  G = D M⁻¹ Dᵀ, K_u = K(κ') − κ' G.
- `tangent-check.jl deck.yaml ...`: consistency of the assembled tangent and
  of the matrix-free action with central differences of the residual.

## Operators and methods

The TET15-P1 tangent is K(κ) = K_u + κ D M⁻¹ Dᵀ, with D (n_u × n_p) and M
(n_p × n_p, block-diagonal, 4 × 4 per element) the projection operators of
the linear volumetric projection, and K_u the κ-free part (isochoric and
initial-stress terms).  The same system is the saddle point

    [K_u  D ] [u]   [b]
    [Dᵀ −M/κ] [p] = [0]

in displacement and projected pressure p = κ M⁻¹ Dᵀ u.

- `cg-amg(K)`: conjugate gradients on K(κ), one V-cycle of smoothed
  aggregation built on K(κ) with the six rigid-body modes as near-null
  space and two damped Jacobi sweeps per side.  This is Carina's device
  preconditioner, reproduced on the host.
- `cg-amg(K')`: the same V-cycle built on K(κ' = μ), the tangent of the
  compressible material with ν' = 1/8, applied to K(κ).
- `minres-block`, `minres-scaled`, `gmres-block`: the saddle point with the
  block-diagonal preconditioner diag(V-cycle on K(κ'), ((1/κ + 1/κ') M)⁻¹),
  the augmented-Lagrangian preconditioner of Benzi and Olshanskii (2006)
  with the augmentation parameter κ'; MINRES on the form above, MINRES on
  the force-scaled form (q = p/κ, second row multiplied by κ), and
  right-preconditioned GMRES on the force-scaled form.

The stopping test is a relative residual of 1e-8 of each method's own system,
and the table reports the true relative residual of the displacement system,
‖K(κ) u − b‖/‖b‖.  For the saddle point the displacement residual equals
D M⁻¹ r_p − r_u, so a residual r_p of the pressure block is amplified by
|D M⁻¹| ~ 1/h; MINRES measures the residual in the norm of its preconditioner,
which weighs the pressure block by about κ'/κ², and stops at 204 iterations
with the displacement residual at 5e-5 (κ/μ = 10⁴); GMRES to 1e-12 on the
force-scaled system gives the count for a displacement residual of 1e-8.

## Results

State: the full-cylinder a/4 TETRA15 mesh (24,635 elements, 325,414
displacement unknowns, 98,540 projection unknowns, 3.05e7 nonzeros in K),
TET15-P1, J2 case A, after the first step of the doubling start (elastic,
|U| of order 10⁻⁹ m); the right-hand side is the residual of the following
Newton step.  CPU of Sirius, 16 threads.

| κ/μ | cg-amg(K), present | cg-amg(K') | gmres-block to 1e-12 | displacement residual of gmres-block |
|---|---|---|---|---|
| 10² | 274 | 330 | 326 | 3.6e-11 |
| 10³ | 747 | 483 | 328 | 3.8e-10 |
| 10⁴ | 1283 | 523 | 330 | 2.7e-9 |
| 10⁵ | 1982 | 678 | 340 | 6.1e-9 |

MINRES on the saddle point, either form, converges in its own norm in 77
(unscaled) or 204 (force-scaled) iterations independent of κ and leaves a
displacement residual of 0.28 and 5e-5 at κ/μ = 10⁴; the counts to compare
are those of GMRES.

Reading:

1. The present preconditioner degrades as about κ^0.4 (274 → 1982 over three
   decades).  Smoothed aggregation coarsens on the strong couplings, which
   are the κ-couplings, and the damped Jacobi smoother scaled to the
   diagonal of K(κ) leaves the deviatoric error, four orders of magnitude
   weaker in the matrix, to the Krylov iteration.
2. The V-cycle built on K(κ' = μ) and applied to K(κ) needs 2.5 times fewer
   iterations at κ/μ = 10⁴ and grows slowly with κ (330 → 678).  In Carina
   it needs no new kernel: the hierarchy is assembled on the host from the
   material properties, so it is one property change (λ' = μ/3) during the
   build, plus a κ' variant of the fine-level smoothing action on the
   device through the hook that the Float32 smoother uses.  It is the first
   step.
3. The augmented-Lagrangian block preconditioner gives 330 iterations
   independent of κ and 3.9 times fewer than the present method, 1.6 times
   fewer than `cg-amg(K')`, at about the same cost per iteration (one
   action of K_u, the products with D, Dᵀ and M⁻¹, one V-cycle).  On the
   device it needs the matrix-free actions of D, Dᵀ and M⁻¹ (element-local,
   from the same projection pass the residual kernel makes), the κ-free
   action K_u (the kernel with κ = 0), and a nonsymmetric or indefinite
   Krylov method with the force-scaled stopping test.  It is the second
   step, after the first has been measured on the punch runs.

## Validation of the compressible operator on the quarter mesh

Quarter a/4 TETRA15 mesh (84,387 unknowns), TET15-P1, case A, CPU of
Sirius, 10 threads, assembled CG with AMG to a relative residual of 1e-8,
predictor on, the doubling start to t = 0.025 (12 steps, the last four at
Δt = 6.25e-3 with plastic flow under the punch):

| operator of the hierarchy and smoother | solves | CG iterations | CG per solve | CG per solve at the plastic steps | wall time |
|---|---|---|---|---|---|
| `tangent` (present) | 57 | 115,416 | 2025 | 2500 to 3700 | 1 h 16 min |
| `compressible` (κ' = μ) | 55 | 34,637 | 630 | 700 to 1000 | 23 min |

The ratio is 3.3 in iterations and in time, larger than the 2.5 of the
elastic state above, because the present operator degrades further at the
plastic states while the compressible one does not.  On the GPU runs with
the cap at 1000 the present operator ends most plastic-step solves at the
cap; the compressible one stays under it.

## Tangent consistency and the Newton count

The Newton count of the punch runs, 8 to 29 iterations per step, is not a
linear-solver effect: with the direct solver on the quarter a/4 mesh (84,387
unknowns) case B needs 10 to 19 iterations per plastic step, the same as the
GPU runs at the CG cap.  `tangent-check.jl` on an 8-element TETRA15 cube
stretched to fifteen times the yield strain gives a tangent inconsistency of
4e-5 to 6e-5 of the norm, independent of the difference step, for TET15-P1 and
for TETRA15, with the assembled matrix and the matrix-free action equal to
1e-15 and symmetric to 1e-15; in an elastic state the error falls as ε².
The inconsistency is the symmetrization of the algorithmic tangent (BOX 9.2
of Simo and Hughes) alone, 1e-3 of the material tangent and 11 to 32 % along
the stress at zero hardening (Rigel's material-level check,
`cm-j2-tangent-check.jl` on the gist); the homogeneous plastic cube
converges in 2 to 3 Newton iterations per step with it.  The punch count
comes from quadrature points switching between elastic and plastic states
near the punch edge: the merit function rises by two orders of magnitude at
the full Newton step, the line search accepts α = 1/8 or less with a
decrease of 2 to 10 % per iteration, and the step ends when the active set
stops changing and two or three full steps are accepted.  On case B with the
direct solver, applying the Armijo condition to the first iteration after a
kept predictor gives 5, 12, 8, 11, 12 iterations in place of 10, 18, 13, 19,
18 on the first plastic steps and 17, 15, 15 in place of 7, 6, 6 after a cut;
without a line search the steps that converge take 4 to 6 iterations and
about half the attempts fail.
