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
# Nonlinear solver hierarchy:
#   AbstractNonlinearSolver → ExplicitSolver | NewtonSolver{LS} | NLCGSolver | SteepestDescentSolver

# --------------------------------------------------------------------------- #
# Assembly flags — controls caching of mass and stiffness matrices
#
# Mass matrix M is constant for constant-density materials: compute once.
# Stiffness K is constant only for linear elastic materials: compute once.
#
# Follows the flag pattern from Norma.jl: flags start true and are set to
# false after first assembly, gating subsequent calls in setup_jacobian!.
# --------------------------------------------------------------------------- #

mutable struct AssemblyFlags
    compute_stiffness::Bool   # false → K is cached, skip assemble_stiffness!
    compute_mass     ::Bool   # false → M is cached, skip assemble_mass!
    is_linear        ::Bool   # all blocks use linear elastic material
end

AssemblyFlags(; is_linear::Bool=false) =
    AssemblyFlags(true, true, is_linear)

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
# Abstract solver types
# --------------------------------------------------------------------------- #

abstract type AbstractLinearSolver end
abstract type AbstractNonlinearSolver end

struct ExplicitSolver <: AbstractNonlinearSolver end

# --------------------------------------------------------------------------- #
# Concrete linear solver types
# --------------------------------------------------------------------------- #

struct DirectLinearSolver <: AbstractLinearSolver end

mutable struct KrylovLinearSolver{KW, Vec} <: AbstractLinearSolver
    itmax    ::Int
    rtol     ::Float64
    assembled::Bool           # true = CPU sparse K_eff; false = GPU matrix-free
    precond  ::Preconditioner
    workspace::KW
    ones_v   ::Vec
    scratch  ::Vec
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
