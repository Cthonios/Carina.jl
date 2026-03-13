# Preconditioner types for Carina's matrix-free Krylov solvers.
#
# These are Carina-side abstractions; FEC is not modified.
# The preconditioner is applied as M⁻¹·v (i.e. it stores the INVERSE diagonal).

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
