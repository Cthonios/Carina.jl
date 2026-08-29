# Solver type definitions for Carina.
#
# This file contains ONLY type definitions — no methods.
#
# Preconditioner hierarchy:
#   Preconditioner → NoPreconditioner | JacobiPreconditioner | ICPreconditioner
#
# Linear solver hierarchy:
#   AbstractLinearSolver → DirectLinearSolver | KrylovLinearSolver | LBFGSLinearSolver | NoLinearSolver
#
# Forcing term hierarchy (inexact Newton, KrylovLinearSolver only):
#   AbstractForcingTerm → FixedForcing | EisenstatWalker
#
# Nonlinear solver hierarchy:
#   AbstractNonlinearSolver → ExplicitSolver | NewtonSolver{LS} | NLCGSolver | SteepestDescentSolver

# --------------------------------------------------------------------------- #
# Assembly flags — controls caching of mass/stiffness matrices and
# factorizations.
#
# Mass matrix M is constant (constant density): compute once.
# Stiffness K is constant only for linear elastic materials: compute once.
# Factorizations (LU, IC) can be cached when the system matrix is constant.
#
# Follows the flag pattern from Norma.jl: flags start true and are set to
# false after first computation, gating subsequent calls.
# --------------------------------------------------------------------------- #

mutable struct AssemblyFlags
    compute_stiffness    ::Bool   # false → K is cached, skip assemble_stiffness!
    compute_mass         ::Bool   # false → M is cached, skip assemble_mass!
    compute_factorization::Bool   # false → LU/IC factorization is cached
    is_linear            ::Bool   # all blocks use linear elastic material
    c_M_cached           ::Float64  # cached c_M value; refactorize if changed
end

AssemblyFlags(; is_linear::Bool=false) =
    AssemblyFlags(true, true, true, is_linear, 0.0)

# --------------------------------------------------------------------------- #
# Abstract type
# --------------------------------------------------------------------------- #

abstract type Preconditioner end

# --------------------------------------------------------------------------- #
# No preconditioner — identity
# --------------------------------------------------------------------------- #

struct NoPreconditioner <: Preconditioner end

# --------------------------------------------------------------------------- #
# Diagonal (Jacobi) preconditioner
#
# inv_diag[i] = 1 / d_ii where d_ii is the diagonal of the system matrix.
# For the Newmark effective stiffness K + c_M·M, using the mass-only
# approximation d_ii ≈ c_M·M_ii (valid when c_M·M dominates, i.e. small Δt).
# --------------------------------------------------------------------------- #

struct JacobiPreconditioner{V} <: Preconditioner
    inv_diag::V
end

# --------------------------------------------------------------------------- #
# Incomplete LDLᵀ preconditioner (CPU assembled path only)
#
# Computes an incomplete factorization of the symmetric part of K at the
# start of each Newton step.  Much stronger than Jacobi for ill-conditioned
# systems (e.g. J2 plasticity on non-uniform meshes).
# --------------------------------------------------------------------------- #

struct ICPreconditioner <: Preconditioner end

# --------------------------------------------------------------------------- #
# Chebyshev polynomial preconditioner (matrix-free, GPU-friendly)
#
# Approximates A⁻¹ via a degree-k Chebyshev polynomial p_k(A).
# Application requires only matrix-vector products (no triangular solves),
# making it ideal for the GPU matrix-free path where IC is unavailable
# and Jacobi is too weak.
#
# Spectral bounds [λ_min, λ_max] are estimated via short Lanczos iteration
# and updated each Newton step.
# --------------------------------------------------------------------------- #

struct ChebyshevPreconditioner{V} <: Preconditioner
    degree     ::Int                    # polynomial degree (inner matvecs per apply)
    lambda_min ::Base.RefValue{Float64} # estimated smallest eigenvalue
    lambda_max ::Base.RefValue{Float64} # estimated largest eigenvalue
    work1      ::V                      # scratch vector for recurrence
    work2      ::V                      # scratch vector for recurrence
    work3      ::V                      # scratch for squared-polynomial intermediate
end

# --------------------------------------------------------------------------- #
# Smoothed-aggregation AMG preconditioner (CPU assembled path only)
#
# Multilevel preconditioner with rigid-body-mode near-nullspace — the only
# preconditioner here whose CG iteration count is (nearly) independent of
# the Newmark Δt.  Hierarchy setup is expensive (~seconds at 500k DOF), so
# it is lagged: built once, then rebuilt only when c_M changes (time-step
# change) or when CG iteration growth shows the hierarchy has gone stale.
#
# The near-nullspace (6 rigid-body modes) is rebuilt from the CURRENT nodal
# coordinates X + u at every hierarchy build, not frozen at the reference
# configuration.  At finite deformation the true near-null space of K(U) is
# the rigid-body modes about the deformed configuration; a frozen reference
# near-nullspace degrades AMG badly once the body rotates substantially
# (observed: 17 → >1000 CG iters at peak twist of the torsion bar).
# --------------------------------------------------------------------------- #

mutable struct AMGPreconditioner <: Preconditioner
    udofs      ::Vector{Int}      # unknown (free) DOF indices, into the 3*nn vector
    P          ::Any              # AMG.Preconditioner (V-cycle apply); nothing until built
    built_c_M  ::Float64          # c_M at last hierarchy build
    base_iters ::Int              # CG iters on first solve after build (0 = unset)
    rebuild    ::Bool             # request hierarchy rebuild at next setup
    nbuilds    ::Int
end

AMGPreconditioner(udofs::Vector{Int}) = AMGPreconditioner(udofs, nothing, -1.0, 0, false, 0)

# GPU-resident AMG: hierarchy built on the host by the same SA machinery,
# applied on the device as a V-cycle with a matrix-free fine level
# (src/gpu_amg.jl).  `inv_diag` is the fine-level 1/diag(K_eff) on the device,
# refreshed every Newton iteration; the hierarchy itself is rebuilt lazily
# with the same c_M / iteration-growth staleness triggers as the CPU AMG.
mutable struct GPUAMGPreconditioner{V} <: Preconditioner
    udofs      ::Vector{Int}
    inv_diag   ::V
    hierarchy  ::Any              # DeviceAMGHierarchy; nothing until built
    lmax_fine  ::Float64
    built_c_M  ::Float64
    base_iters ::Int
    rebuild    ::Bool
    nbuilds    ::Int
end

# --------------------------------------------------------------------------- #
# Abstract solver types
# --------------------------------------------------------------------------- #

abstract type AbstractLinearSolver end
abstract type AbstractNonlinearSolver end

struct ExplicitSolver <: AbstractNonlinearSolver end

# --------------------------------------------------------------------------- #
# Inexact-Newton forcing terms
#
# A Newton step solves K ΔU = R only to make progress on the *nonlinear*
# residual.  Driving that linear solve to the deck's final tolerance while
# ‖R‖ is still far from converged is wasted work: on the 530k-DOF torsion
# bar the CG count per Newton iteration was flat (192 / 207 / 195) while
# ‖R‖ fell 7.10e3 → 4.94e1 → 1.09e-2 → 1.89e-9.  Every one of those solves
# was worked to 1e-8 and only the last one needed it.
#
# A forcing term replaces the fixed tolerance with a per-iteration ηₖ.
# --------------------------------------------------------------------------- #

abstract type AbstractForcingTerm end

# Every Newton iteration solves to the deck's `tolerance`.  The default, and
# the behavior of every deck written before forcing terms existed.
struct FixedForcing <: AbstractForcingTerm end

# Choice 2 of Eisenstat & Walker, SIAM J. Sci. Comput. 17(1):16-32, 1996:
#
#     ηₖ = γ (‖Rₖ‖ / ‖Rₖ₋₁‖)^α
#
# `eta_0` is used on the first Newton iteration of a step, where no ratio
# exists yet.  `eta_max` matters more than it looks: EW's safeguard against
# eta falling too fast engages when gamma*eta^alpha > 0.1, so an eta_max above
# ~0.24 keeps that safeguard permanently on and holds eta high.  The default
# of 0.2 sits deliberately below the knee.  `safety` scales Kelley's guard against over-solving the final
# step (Iterative Methods for Linear and Nonlinear Equations, §6.3); 0
# disables it.  See `_forcing_tolerance!` in linear_solvers.jl for how the
# three safeguards compose.
struct EisenstatWalker <: AbstractForcingTerm
    gamma  ::Float64   # scale factor, 0 < γ ≤ 1
    alpha  ::Float64   # exponent, 1 < α ≤ 2
    eta_max::Float64   # loosest tolerance any solve may use, 0 < η_max < 1
    eta_0  ::Float64   # η for the first Newton iteration of a step
    safety ::Float64   # over-solve guard factor; 0 disables
end

# --------------------------------------------------------------------------- #
# Concrete linear solver types
# --------------------------------------------------------------------------- #

struct DirectLinearSolver <: AbstractLinearSolver end

mutable struct KrylovLinearSolver{KW, Vec} <: AbstractLinearSolver
    itmax    ::Int
    rtol     ::Float64        # deck tolerance; also the tightest η ever used
    assembled::Bool           # true = CPU sparse K_eff; false = GPU matrix-free
    precond  ::Preconditioner
    workspace::KW
    scratch  ::Vec            # free-DOF sized: diagonals, preconditioner updates
    # --- inexact-Newton state, driven by `_update_forcing!` --------------- #
    forcing    ::AbstractForcingTerm
    rtol_eff   ::Float64      # tolerance the next solve will actually use
    eta_prev   ::Float64      # η used by the previous solve (NaN = none yet)
    norm_R_last::Float64      # ‖R‖ at the previous update (NaN = first iteration)
end

# LBFGSLinearSolver: ring-buffer and scratch vectors only.
# R_eff and F_int_n removed — they are integrator state, not solver state.
mutable struct LBFGSLinearSolver{Vec, PC <: Preconditioner} <: AbstractLinearSolver
    m         ::Int
    precond   ::PC
    S         ::Vector{Vec}
    Y         ::Vector{Vec}
    ρ         ::Vector{Float64}
    alpha_buf ::Vector{Float64}
    head      ::Int
    hist_fill ::Int
    R_old     ::Vec   # snapshot for y = R_old − R_new
    d         ::Vec   # descent direction
    q         ::Vec   # two-loop work / trial scratch
    M_d       ::Vec   # Newmark: M·d precomputed for line search
    M_dU      ::Vec   # Newmark: M·(U−U_pred) maintained incrementally
end

struct NoLinearSolver <: AbstractLinearSolver end

# --------------------------------------------------------------------------- #
# Newton nonlinear solver
# --------------------------------------------------------------------------- #

mutable struct NewtonSolver{LS <: AbstractLinearSolver} <: AbstractNonlinearSolver
    min_iters         ::Int
    max_iters         ::Int
    abs_increment_tol ::Float64
    abs_residual_tol  ::Float64
    rel_residual_tol  ::Float64
    linear_solver     ::LS
    # Line search parameters
    use_line_search   ::Bool
    ls_backtrack      ::Float64   # step reduction factor (default 0.5)
    ls_decrease       ::Float64   # Armijo sufficient decrease (default 1e-4)
    ls_max_iters      ::Int       # max backtracking steps (default 10)
end

# --------------------------------------------------------------------------- #
# Nonlinear Conjugate Gradient solver (matrix-free, GPU-friendly)
# --------------------------------------------------------------------------- #

mutable struct NLCGSolver{Vec, PC <: Preconditioner} <: AbstractNonlinearSolver
    min_iters         ::Int
    max_iters         ::Int
    abs_increment_tol ::Float64
    abs_residual_tol  ::Float64
    rel_residual_tol  ::Float64
    # Line search
    ls_backtrack      ::Float64   # step reduction factor (default 0.5)
    ls_decrease       ::Float64   # Armijo parameter (default 1e-4)
    ls_max_iters      ::Int       # max backtracking steps (default 10)
    # CG parameters
    orthogonality_tol ::Float64   # restart threshold (default 0.5)
    restart_interval  ::Int       # periodic restart every N iters (0 = disabled)
    # Preconditioner and work vectors (device-resident)
    precond   ::PC
    g         ::Vec    # preconditioned gradient M⁻¹R
    g_old     ::Vec    # previous preconditioned gradient
    d         ::Vec    # search direction
    U_save    ::Vec    # saved displacement for line search
end

# --------------------------------------------------------------------------- #
# Steepest Descent solver (matrix-free, GPU-friendly, energy-based line search)
# --------------------------------------------------------------------------- #

mutable struct SteepestDescentSolver{Vec, PC <: Preconditioner} <: AbstractNonlinearSolver
    min_iters         ::Int
    max_iters         ::Int
    abs_increment_tol ::Float64
    abs_residual_tol  ::Float64
    rel_residual_tol  ::Float64
    # Line search (Armijo backtracking on energy)
    ls_backtrack      ::Float64   # step reduction factor (default 0.5)
    ls_decrease       ::Float64   # Armijo parameter c (default 1e-4)
    ls_max_iters      ::Int       # max backtracking steps (default 30)
    # Preconditioner and work vectors
    precond   ::PC
    d         ::Vec    # search direction (preconditioned gradient)
    U_save    ::Vec    # saved displacement for line search
end
