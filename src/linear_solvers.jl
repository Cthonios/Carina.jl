# Linear solver implementations for Carina.
#
# Contains all linear solve dispatch, preconditioner helpers, Krylov setup,
# and L-BFGS two-loop recursion + helpers.
#
# Depends on types from solvers.jl and integrator types from integrators.jl.

using LinearAlgebra
using SparseArrays
import Adapt
import Krylov
import LinearOperators: LinearOperator
import IterativeSolvers
import LimitedLDLFactorizations: lldl

# --------------------------------------------------------------------------- #
# GPU Cholesky: factorize K on CPU, upload L to GPU for triangular solves.
# For linear elastic problems where K is constant, this gives direct-solver
# performance on GPU (two sparse triangular solves per step).
# --------------------------------------------------------------------------- #


function _build_gpu_cholesky!(gpu_asm, p_gpu)
    asm_cpu = _cpu_asm_ref[]
    p_cpu   = _cpu_params_ref[]
    (asm_cpu === nothing || p_cpu === nothing) && return

    n = length(asm_cpu.dof.unknown_dofs)
    U_cpu = zeros(n)
    FEC.assemble_stiffness!(asm_cpu, FEC.stiffness, U_cpu, p_cpu)
    K_raw = FEC.stiffness(asm_cpu)
    K_sym = Symmetric((K_raw + K_raw') / 2, :L)

    _carina_log(0, :setup, "Building GPU Cholesky factors...")
    t = @elapsed begin
        F = cholesky(K_sym)
        L_cpu = sparse(F.L)
        perm  = F.p
        iperm = invperm(perm)
        L_gpu = FEC.to_backend(_backend_ref[], L_cpu)
    end
    _gpu_cholesky_L[] = (L_gpu, perm, iperm)
    _carina_logf(0, :setup, "GPU Cholesky: L uploaded (%d nnz, %s)",
                 nnz(L_cpu), format_time(t))
    return nothing
end

# --------------------------------------------------------------------------- #
# Threaded CSR operator for the assembled (CPU) Krylov path.
#
# `SparseArrays.mul!` on a `SparseMatrixCSC` is single-threaded and cannot
# easily be otherwise: CSC walks columns and scatters into `y`, so parallel
# columns collide on the same outputs.  CSR walks rows and reduces into `y[i]`,
# which is conflict-free -- that is why Tpetra gives Albany/LCM a parallel SpMV
# and Julia's stdlib does not (benchmark/crosscode/README.md §3).
#
# Two facts make this nearly free here.  The assembled tangent is symmetric to
# roundoff -- measured 1.0e-16 relative in the infinity norm, for quasi-static
# (c_M = 0) as well as Newmark -- and for a symmetric matrix the CSC arrays ARE
# the CSR arrays, so no transpose is needed.  And `_csr_mul!` already exists in
# gpu_amg.jl as a KernelAbstractions kernel, so the CPU backend threads it with
# no new kernel code.
#
# Measured at 530k DOF / 40.2M nonzeros, 24 threads:
#   Symmetric(S, :L) mul!   15.31 ms   <- what CG applied before
#   SparseArrays CSC mul!   14.23 ms
#   threaded CSR mul!        9.43 ms
#
# The remaining ceiling is memory bandwidth, not cores: 9.43 ms moves ~490 MB,
# i.e. ~52 GB/s against 50-90 GB/s achievable on this box, so CSR scales only
# 1.35x from 1 to 24 threads and no amount of further threading will help.

# Tolerance on ||K - K'||_inf / ||K||_inf below which the operator is treated as
# symmetric.  Measured values are ~1e-16; 1e-10 leaves ample margin while still
# catching a genuinely non-symmetric tangent (a non-associative flow rule, say),
# for which CG is not a valid solver in the first place.
const _SYM_TOL = 1e-10

# Reinterpret a symmetric CSC matrix's arrays as CSR, without transposing.
function _symmetric_csc_as_csr(backend, K::SparseArrays.SparseMatrixCSC)
    rowptr = KA.allocate(backend, Int32, length(K.colptr))
    colval = KA.allocate(backend, Int32, length(K.rowval))
    nzval  = KA.allocate(backend, Float64, length(K.nzval))
    copyto!(rowptr, Int32.(K.colptr))
    copyto!(colval, Int32.(K.rowval))
    copyto!(nzval, K.nzval)
    return DeviceCSR(size(K, 1), size(K, 2), rowptr, colval, nzval)
end

# Checked once per run: if the tangent is not symmetric, CG is invalid and the
# caller must be told rather than quietly handed a wrong operator.
const _tangent_is_symmetric = Ref{Union{Nothing, Bool}}(nothing)

function _check_tangent_symmetry!(K::SparseArrays.SparseMatrixCSC)
    _tangent_is_symmetric[] !== nothing && return _tangent_is_symmetric[]
    nK = LinearAlgebra.norm(K, Inf)
    asym = nK == 0.0 ? 0.0 : LinearAlgebra.norm(K - K', Inf) / nK
    ok = asym <= _SYM_TOL
    ok || @warn "Assembled tangent is not symmetric; falling back to the " *
                "explicitly symmetrized operator. CG assumes symmetry, so a " *
                "tangent this asymmetric may not converge." asymmetry = asym
    _tangent_is_symmetric[] = ok
    return ok
end

"""
    _assembled_operator(K, n)

The operator to hand Krylov for an assembled solve: a threaded CSR apply when
the tangent is symmetric, and the previous `Symmetric((K + K')/2, :L)` when it
is not.  The fallback keeps non-symmetric tangents working exactly as before.
"""
function _assembled_operator(K::SparseArrays.SparseMatrixCSC, n::Int)
    if _check_tangent_symmetry!(K)
        backend = KA.CPU()
        csr = _symmetric_csc_as_csr(backend, K)
        return LinearOperator(Float64, n, n, true, true,
                              (y, v) -> (_csr_mul!(y, csr, v, backend); y))
    end
    return Symmetric((K + K') / 2, :L)
end

# --------------------------------------------------------------------------- #
# Helper: preconditioner updates
# --------------------------------------------------------------------------- #

# Assembled path: update Jacobi diagonal directly from sparse K_eff matrix.
function _update_jacobi_precond_assembled!(precond::JacobiPreconditioner, K_eff)
    d = diag(K_eff)
    @. precond.inv_diag = 1.0 / max(abs(d), eps(Float64))
    return nothing
end
_update_jacobi_precond_assembled!(::Preconditioner, _) = nothing  # IC/Chebyshev/AMG built elsewhere

# Current nodal coordinates X + u (full, interleaved 3*nn) from the params:
# p.coords is the reference field, p.field the current displacement field
# (BCs + unknowns synced before assembly).  Same layout the DOF map indexes.
_current_coords(p) = vec(p.coords.data) .+ vec(p.field.data)

# Rigid-body-mode near-nullspace (6 columns: 3 translations + 3 rotations
# about the centroid) from nodal coords `X` (length 3*nn, interleaved),
# restricted to the free DOFs `udofs`.  Passing CURRENT coordinates makes the
# rotational modes track the deformed configuration's near-null space.
function _rigid_body_modes(X::AbstractVector{<:Real}, udofs::AbstractVector{Int})
    nn = length(X) ÷ 3
    cx = cy = cz = 0.0
    for a in 1:nn
        i = 3 * (a - 1)
        cx += X[i+1]; cy += X[i+2]; cz += X[i+3]
    end
    cx /= nn; cy /= nn; cz /= nn
    B = zeros(3 * nn, 6)
    for a in 1:nn
        i = 3 * (a - 1)
        x = X[i+1] - cx; y = X[i+2] - cy; z = X[i+3] - cz
        B[i+1, 1] = 1.0
        B[i+2, 2] = 1.0
        B[i+3, 3] = 1.0
        B[i+2, 4] = -z; B[i+3, 4] =  y   # rotation about x
        B[i+1, 5] =  z; B[i+3, 5] = -x   # rotation about y
        B[i+1, 6] = -y; B[i+2, 6] =  x   # rotation about z
    end
    return B[udofs, :]
end

# AMG hierarchy build/rebuild on the assembled path.  Lagged: built once,
# then rebuilt only when c_M changes (Δt change via adaptive stepping) or
# when _amg_track_iters! has flagged staleness from CG iteration growth.
# `x_cur` is the current full nodal coordinate vector (X + u); the
# near-nullspace is recomputed from it each build.
function _update_amg_precond_assembled!(precond::AMGPreconditioner, K_eff_raw, c_M, x_cur)
    # c_M comparison is tolerant: the controller's substep Δt varies in the
    # last bits when landing on output stops, and sub-percent c_M drift does
    # not degrade the preconditioner.  Adaptive-stepping changes (≥ ~2×) and
    # the Newmark→QS distinction are far outside this tolerance.
    c_M_changed = abs(c_M - precond.built_c_M) > 1e-3 * abs(c_M)
    stale = precond.P === nothing || precond.rebuild || c_M_changed
    stale || return nothing
    A = (K_eff_raw + K_eff_raw') / 2   # exact symmetry for the SA setup
    t = @elapsed begin
        B = _rigid_body_modes(x_cur, precond.udofs)
        ml = AMG.smoothed_aggregation(A; B = B)
        precond.P = AMG.aspreconditioner(ml)
    end
    precond.built_c_M  = c_M
    precond.base_iters = 0
    precond.rebuild    = false
    precond.nbuilds   += 1
    _carina_logf(4, :solve, "    AMG hierarchy build #%d (%s)", precond.nbuilds, format_time(t))
    return nothing
end
_update_amg_precond_assembled!(::Preconditioner, _, _, _) = nothing

# Staleness detector: remember the iteration count right after a build;
# flag a rebuild when the count grows past 3× that baseline (and ≥ 30).
#
# Must be called after EVERY AMG-preconditioned solve, on both the quasi-static
# and the Newmark path.  It is the only thing that triggers a rebuild on the
# quasi-static path: `_build_precond_op` passes c_M = 0.0 there, so the
# `c_M_changed` test in `_update_amg_precond_assembled!` compares 0.0 > 0.0 and
# is never true.  Without this call the hierarchy is built once at the reference
# configuration and reused for the whole run, which makes the current-config
# near-nullspace pointless and lets iteration counts drift upward unchecked.
function _amg_track_iters!(precond::Union{AMGPreconditioner, GPUAMGPreconditioner}, iters::Int)
    if precond.base_iters == 0
        precond.base_iters = iters
    elseif iters > max(3 * precond.base_iters, 30)
        precond.rebuild = true
    end
    return nothing
end
_amg_track_iters!(::Preconditioner, _) = nothing

# --------------------------------------------------------------------------- #
# GPU AMG: host-built SA hierarchy, device V-cycle (src/gpu_amg.jl)
# --------------------------------------------------------------------------- #

function _compute_gpu_amg_precond(asm_cpu, template)
    inv_d = similar(template); fill!(inv_d, 0.0)
    return GPUAMGPreconditioner(collect(asm_cpu.dof.unknown_dofs),
                                inv_d, nothing, 1.0, -1.0, 0, false, 0)
end

# Host-side hierarchy (re)build with the same staleness triggers as the CPU
# AMG.  Assembles K_eff on the always-CPU assembler — the device assembler is
# stripped of its pattern precisely because assembly belongs on the host.
function _build_gpu_amg_hierarchy!(precond::GPUAMGPreconditioner, c_M, U_dev)
    c_M_changed = abs(c_M - precond.built_c_M) > 1e-3 * abs(c_M)
    stale = precond.hierarchy === nothing || precond.rebuild || c_M_changed
    stale || return nothing

    asm_cpu = _cpu_asm_ref[]; p_cpu = _cpu_params_ref[]; backend = _backend_ref[]
    (asm_cpu === nothing || p_cpu === nothing) &&
        error("GPU AMG requires the CPU assembler references to be set.")

    t = @elapsed begin
        U_cpu = Vector{Float64}(Adapt.adapt(Array, U_dev))
        FEC._update_for_assembly!(p_cpu, asm_cpu.dof, U_cpu)
        FEC.assemble_stiffness!(asm_cpu, FEC.stiffness, U_cpu, p_cpu)
        if c_M != 0.0
            FEC.assemble_mass!(asm_cpu, FEC.mass, U_cpu, p_cpu)
            @. asm_cpu.stiffness_storage += c_M * asm_cpu.mass_storage
        end
        K_raw = FEC.stiffness(asm_cpu)
        A = SparseArrays.sparse((K_raw + K_raw') / 2)
        B = _rigid_body_modes(_current_coords(p_cpu), precond.udofs)
        ml = _sa_hierarchy_lowmem(A, B)
        dinv_h = 1.0 ./ Vector(diag(A))
        precond.lmax_fine = _host_lambda_max(A, dinv_h)
        precond.hierarchy = DeviceAMGHierarchy(backend, ml, precond.inv_diag,
                                               precond.lmax_fine)
    end
    precond.built_c_M  = c_M
    precond.base_iters = 0
    precond.rebuild    = false
    precond.nbuilds   += 1
    _carina_logf(4, :solve, "    GPU AMG hierarchy build #%d (%s)",
                 precond.nbuilds, format_time(t))
    return nothing
end

# Per-Newton-iteration updates: refresh the fine diagonal (used by the
# V-cycle's fine smoother), then lazily rebuild the hierarchy.
function _update_gpu_amg_precond_qs!(precond::GPUAMGPreconditioner, asm, U, p)
    FEC.assemble_diagonal!(asm, StiffnessDiagonal(), U, p)
    d = FEC.diagonal(asm)
    @. precond.inv_diag = 1.0 / max(abs(d), eps(Float64))
    _build_gpu_amg_hierarchy!(precond, 0.0, U)
    return nothing
end
_update_gpu_amg_precond_qs!(::Preconditioner, args...) = nothing

function _update_gpu_amg_precond_eff!(precond::GPUAMGPreconditioner, asm, U,
                                      c_M, p)
    FEC.assemble_diagonal!(asm, NewmarkDiagonal(c_M), U, p)
    d = FEC.diagonal(asm)
    @. precond.inv_diag = 1.0 / max(abs(d), eps(Float64))
    _build_gpu_amg_hierarchy!(precond, c_M, U)
    return nothing
end
_update_gpu_amg_precond_eff!(::Preconditioner, args...) = nothing


# Compute (K + c_M·M)·v in one fused matrix-free assembly, storing the result
# in asm storage.  Both terms walk the same connectivity, so `NewmarkAction`
# evaluates them in a single element pass — one gather, one scatter.
function _apply_eff_stiffness!(asm, U, v, c_M, p)
    _assemble_action!(asm, NewmarkAction(c_M), U, v, p)
end

# Matrix-free Jacobi preconditioner: diag(K + c_M·M) in one diagonal-only
# assembly — no element matrix, no second pass for the mass diagonal.
function _update_jacobi_precond_eff!(precond::JacobiPreconditioner, asm, U, c_M, p)
    FEC.assemble_diagonal!(asm, NewmarkDiagonal(c_M), U, p)
    d = FEC.diagonal(asm)
    @. precond.inv_diag = 1.0 / max(abs(d), eps(Float64))
    return nothing
end
_update_jacobi_precond_eff!(::Preconditioner, args...) = nothing

# GPU matrix-free displacement Jacobian: y = (K + c_M·M)·v
function _eff_stiffness_matvec!(y, v, asm, U, c_M, p)
    _apply_eff_stiffness!(asm, U, v, c_M, p)
    copyto!(y, FEC.hvp(asm, v))
    return y
end

# QS matrix-free Jacobi: true diag(K) via the diagonal-only kernel.
function _update_jacobi_precond_qs!(precond::JacobiPreconditioner, asm, U, p)
    FEC.assemble_diagonal!(asm, StiffnessDiagonal(), U, p)
    d = FEC.diagonal(asm)
    @. precond.inv_diag = 1.0 / max(abs(d), eps(Float64))
    return nothing
end
_update_jacobi_precond_qs!(::Preconditioner, args...) = nothing

# Assembled Chebyshev: bounds are estimated inside _build_precond_op (which
# constructs the symmetrically-scaled operator S = D⁻¹/²AD⁻¹/²), so the
# setup_jacobian! update is a no-op on the assembled path.
_update_chebyshev_precond_assembled!(::ChebyshevPreconditioner, _) = nothing
_update_chebyshev_precond_assembled!(::Preconditioner, _) = nothing

# QS matrix-free path: estimate λ_max of D⁻¹K via power method.
# D⁻¹/² is stored in work3 for use by _chebyshev_precond_op.
function _update_chebyshev_precond_qs!(precond::ChebyshevPreconditioner, asm, U, p)
    n = length(U)
    FEC.assemble_diagonal!(asm, StiffnessDiagonal(), U, p)
    d = FEC.diagonal(asm)
    inv_sqrt_d = precond.work3
    @. inv_sqrt_d = 1.0 / sqrt(max(abs(d), eps(Float64)))
    inv_diag = similar(d)
    @. inv_diag = 1.0 / max(abs(d), eps(Float64))
    matvec! = (y, v) -> _stiffness_matvec_qs!(y, v, asm, U, p)
    _estimate_lambda_max!(precond, matvec!, inv_diag, n)
    return nothing
end
_update_chebyshev_precond_qs!(::Preconditioner, args...) = nothing

# Newmark matrix-free path: estimate λ_max of D⁻¹K_eff.
function _update_chebyshev_precond_eff!(precond::ChebyshevPreconditioner, asm, U, c_M, p)
    n = length(U)
    FEC.assemble_diagonal!(asm, NewmarkDiagonal(c_M), U, p)
    d = FEC.diagonal(asm)
    inv_sqrt_d = precond.work3
    @. inv_sqrt_d = 1.0 / sqrt(max(abs(d), eps(Float64)))
    inv_diag = similar(d)
    @. inv_diag = 1.0 / max(abs(d), eps(Float64))
    matvec! = (y, v) -> _eff_stiffness_matvec!(y, v, asm, U, c_M, p)
    _estimate_lambda_max!(precond, matvec!, inv_diag, n)
    return nothing
end
_update_chebyshev_precond_eff!(::Preconditioner, args...) = nothing

# Jacobi preconditioner as LinearOperator (shared by Krylov paths).
function _jacobi_precond_op(precond::JacobiPreconditioner, n)
    LinearOperator(Float64, n, n, true, true,
        (y, v) -> (@. y = precond.inv_diag * v; y))
end
_jacobi_precond_op(::NoPreconditioner, n) = nothing

# QS K·v via stiffness_action.
function _stiffness_matvec_qs!(y, v, asm, U, p)
    _assemble_action!(asm, FEC.stiffness_action, U, v, p)
    copyto!(y, FEC.hvp(asm, v))
    return y
end

# Reduced-precision twin of the above, for AMG smoothing only.  Identical
# gather/scatter and identical geometry; only the constitutive directional
# derivative drops to Float32 (see `stiffness_action_fp32` in physics.jl).
function _stiffness_matvec_qs_fp32!(y, v, asm, U, p)
    FEC.assemble_matrix_free_action!(asm, stiffness_action_fp32, U, v, p)
    copyto!(y, FEC.hvp(asm, v))
    return y
end

# Decide once per run whether the AMG smoother should use the Float32 action.
#
# The gate is not "did the user ask for it" but "does the model actually honour
# it".  A constitutive model written with Float64 literals promotes silently and
# returns a Float64 stress, which would leave the run correct, exactly as slow,
# and two conversions per quadrature point worse off — the failure mode has no
# symptom, so it is checked rather than assumed.  Probed against the CPU params
# because scalar-indexing `p.properties` on the device is not allowed, and the
# GPU AMG path already requires those references.
_use_fp32_smoother(::Preconditioner, asm) = false

function _use_fp32_smoother(::GPUAMGPreconditioner, asm)
    _fp32_smoother_ok[] !== nothing && return _fp32_smoother_ok[]
    asm_cpu = _cpu_asm_ref[]
    p_cpu   = _cpu_params_ref[]
    if asm_cpu === nothing || p_cpu === nothing
        _fp32_smoother_ok[] = false
        return false
    end
    ok = true
    offenders = String[]
    fspace = FEC.function_space(asm_cpu.dof)
    FEC.foreach_block(fspace, p_cpu) do physics, ref_fe, b
        props_el = FEC.properties(p_cpu.properties, 1, b)
        if !_fp32_action_is_effective(physics, props_el)
            ok = false
            push!(offenders, string(typeof(physics.constitutive_model)))
        end
        return nothing
    end
    if ok
        _carina_logf(4, :solve, "    AMG smoother: Float32 action")
    else
        @warn "AMG smoother falling back to Float64: these models promote " *
              "Float32 inputs back to Float64, so the reduced-precision " *
              "action would cost more than it saves. Make their `pk1_stress` " *
              "constants type-generic (e.g. `one(J)/2` rather than `0.5`) " *
              "to enable it." models = unique(offenders)
    end
    _fp32_smoother_ok[] = ok
    return ok
end

# --------------------------------------------------------------------------- #
# Chebyshev preconditioner: 4th-kind with optimal weights
#
# Based on Ifpack2's implementation of Chebyshev polynomials of the 4th kind
# with optimal coefficients (arxiv 2202.08830).
#
# Key advantages over standard Chebyshev:
#   - Only needs λ_max (no λ_min or eigenvalue ratio)
#   - Baked-in Jacobi scaling (D⁻¹ applied at each iteration)
#   - SPD by construction (optimal weights guarantee positive definiteness)
#   - k matvecs per application (not 2k like the squared polynomial)
#
# λ_max estimated via power method on D⁻¹A (10 iterations, cheap).
# --------------------------------------------------------------------------- #

# Optimal weights for 4th-kind Chebyshev, degrees 1–16 (arxiv 2202.08830).
const _CHEBYSHEV_OPT_WEIGHTS = (
    [1.12500000000000],
    [1.02387287570313, 1.26408905371085],
    [1.00842544782028, 1.08867839208730, 1.33753125909618],
    [1.00391310427285, 1.04035811188593, 1.14863498546254, 1.38268869241000],
    [1.00212930146164, 1.02173711549260, 1.07872433192603, 1.19810065292663,
     1.41322542791682],
    [1.00128517255940, 1.01304293035233, 1.04678215124113, 1.11616489419675,
     1.23829020218444, 1.43524297106744],
    [1.00083464397912, 1.00843949430122, 1.03008707768713, 1.07408384092003,
     1.15036186707366, 1.27116474046139, 1.45186658649364],
    [1.00057246631197, 1.00577427662415, 1.02050187922941, 1.05019803444565,
     1.10115572984941, 1.18086042806856, 1.29838585382576, 1.46486073151099],
)

# Estimate λ_max of D⁻¹A via power method (spectral radius).
function _estimate_lambda_max!(precond::ChebyshevPreconditioner, matvec!, inv_diag, n;
                                power_iters::Int=10, boost::Float64=1.1)
    x = precond.work1
    y = precond.work2
    fill!(x, 1.0 / sqrt(Float64(n)))

    for _ in 1:power_iters
        matvec!(y, x)           # y = A·x
        @. x = inv_diag * y     # x = D⁻¹·A·x
        nrm = sqrt(dot(x, x))
        nrm < 1e-14 && break
        @. x = x / nrm
    end

    # Final Rayleigh quotient: λ = xᵀ(D⁻¹A)x
    matvec!(y, x)
    @. y = inv_diag * y
    lmax = dot(x, y)
    precond.lambda_max[] = lmax * boost

    return nothing
end

_estimate_lambda_max!(::Preconditioner, args...; kwargs...) = nothing

# Apply 4th-kind Chebyshev preconditioner with optimal weights.
#
# Computes y = M·b where M ≈ A⁻¹, using the recurrence:
#   Z₁  = (4/3)·σ·D⁻¹·b,  X₄₁ = Z₁,  y = β₁·Z₁
#   For i = 1..k-1:
#     γ = (2i-1)/(2i+3), ρ = (8i+4)/(2i+3)·σ
#     Zᵢ₊₁ = ρ·D⁻¹·(b − A·X₄ᵢ) + γ·Zᵢ
#     X₄ᵢ₊₁ = X₄ᵢ + Zᵢ₊₁
#     y += βᵢ₊₁·Zᵢ₊₁
# where σ = 1/(λ_max·boost) and βᵢ are the optimal weights.
#
# Cost: k matvecs per application.
function _apply_chebyshev_precond!(y, b, precond::ChebyshevPreconditioner,
                                    matvec!, inv_diag)
    k    = precond.degree
    lmax = precond.lambda_max[]
    σ    = 1.0 / lmax

    betas = k <= length(_CHEBYSHEV_OPT_WEIGHTS) ? _CHEBYSHEV_OPT_WEIGHTS[k] : ones(k)

    z  = precond.work1   # current update step
    x4 = precond.work2   # raw 4th-kind iterate
    w  = precond.work3   # scratch for matvec output

    # Iteration 0 (zero initial guess)
    @. z  = (4.0/3.0 * σ) * inv_diag * b
    copyto!(x4, z)
    @. y  = betas[1] * z

    for i in 1:k-1
        γ = (2.0*i - 1.0) / (2.0*i + 3.0)
        ρ = (8.0*i + 4.0) / (2.0*i + 3.0) * σ
        matvec!(w, x4)                          # w = A·x4
        @. z  = ρ * inv_diag * (b - w) + γ * z  # new Z
        @. x4 = x4 + z                          # advance x4
        @. y  = y + betas[i+1] * z              # weighted accumulation
    end
    return y
end

# Wrap Chebyshev preconditioner as a LinearOperator for Krylov.jl.
# The 4th-kind Chebyshev has baked-in Jacobi (D⁻¹) so no external scaling
# is needed — just pass inv_diag from the Jacobi preconditioner.
function _chebyshev_precond_op(precond::ChebyshevPreconditioner, n, raw_matvec!;
                                inv_sqrt_d=nothing)
    # inv_sqrt_d is D⁻¹/² from setup; we need D⁻¹ = (D⁻¹/²)²
    if inv_sqrt_d !== nothing
        inv_d = copy(inv_sqrt_d)
        @. inv_d = inv_sqrt_d * inv_sqrt_d   # D⁻¹ = (D⁻¹/²)²
    else
        inv_d = ones(n)  # fallback: identity scaling
    end
    return LinearOperator(Float64, n, n, true, true,
        (y, v) -> _apply_chebyshev_precond!(y, v, precond, raw_matvec!, inv_d))
end

# --------------------------------------------------------------------------- #
# Two-loop L-BFGS recursion (unchanged)
# --------------------------------------------------------------------------- #
#
# Computes d = H_k · R_eff (L-BFGS descent direction).
# Convention: ∇Φ = −R_eff, so d = H·R_eff = −H·∇Φ is the descent direction.
# Ring buffer: newest history at slot `head` (1-indexed), `hfill` valid entries.
#
# Initial Hessian H₀ priority:
#   1. hfill > 0: Barzilai-Borwein γ₀ = (s·y)/(y·y) from last history pair.
#   2. precond is JacobiPreconditioner: H₀ = diag(inv_diag), giving correct
#      dimensional scaling on the first step (critical when c_M ≫ 1).
#   3. Fallback: H₀ = I.
function _lbfgs_two_loop!(d, q, R_eff, S, Y, ρ, alpha, head, hfill, m, precond)
    copyto!(q, R_eff)

    # First loop: newest → oldest (i = 1 → newest at S[head], i = hfill → oldest)
    for i in 1:hfill
        idx = mod1(head - i + 1, m)
        alpha[i] = ρ[idx] * dot(S[idx], q)
        @. q = q - alpha[i] * Y[idx]
    end

    # Apply initial Hessian H₀.
    if hfill > 0
        # Barzilai-Borwein scaling from most recent history pair.
        sy = dot(S[head], Y[head])
        yy = dot(Y[head], Y[head])
        γ₀ = (sy > 0.0 && yy > 0.0) ? sy / yy : 1.0
        @. d = γ₀ * q
    elseif !(precond isa NoPreconditioner)
        # Jacobi (diagonal) preconditioner: H₀ = diag(inv_diag).
        # Essential on the first step when c_M = 1/(β·Δt²) ≫ 1.
        @. d = precond.inv_diag * q
    else
        copyto!(d, q)   # H₀ = I (fallback)
    end

    # Second loop: oldest → newest (i = hfill → oldest, i = 1 → newest)
    for i in hfill:-1:1
        idx = mod1(head - i + 1, m)
        β = ρ[idx] * dot(Y[idx], d)
        @. d = d + (alpha[i] - β) * S[idx]
    end

    return d
end

# --------------------------------------------------------------------------- #
# Krylov setup: _setup_linear_ops
# Dispatch on BOTH integrator and linear solver types.
# --------------------------------------------------------------------------- #

_setup_linear_ops(ig, ::DirectLinearSolver, p)  = nothing
_setup_linear_ops(ig, ::LBFGSLinearSolver,  p)  = nothing
_setup_linear_ops(ig, ::NoLinearSolver,     p)  = nothing

# Generic preconditioner → LinearOperator for matrix-free path.
# `matvec!` is the system operator A (needed by Chebyshev; ignored by Jacobi).
# `smooth!` is the operator the preconditioner is allowed to approximate: the
# AMG V-cycle applies it five times per call and never feeds it to Krylov, so it
# may be the reduced-precision action.  Everything else takes the exact one.
_mf_precond_op(::NoPreconditioner, n, matvec!, smooth!)          = nothing
_mf_precond_op(p::JacobiPreconditioner, n, matvec!, smooth!)     = _jacobi_precond_op(p, n)
_mf_precond_op(p::GPUAMGPreconditioner, n, matvec!, smooth!) =
    LinearOperator(Float64, n, n, true, true,
        (y, v) -> (_amg_vcycle!(y, v, p.hierarchy, smooth!,
                                KA.get_backend(p.inv_diag)); y))
_mf_precond_op(p::ChebyshevPreconditioner, n, matvec!, smooth!)  =
    _chebyshev_precond_op(p, n, matvec!; inv_sqrt_d=p.work3)

function _setup_linear_ops(ig::QuasiStaticIntegrator, ls::KrylovLinearSolver, p)
    U = ig.U; n = length(U)
    ls.assembled && return (nothing, nothing)
    matvec! = (y, v) -> _stiffness_matvec_qs!(y, v, ig.asm, U, p)
    smooth! = _use_fp32_smoother(ls.precond, ig.asm) ?
        (y, v) -> _stiffness_matvec_qs_fp32!(y, v, ig.asm, U, p) : matvec!
    K_op = LinearOperator(Float64, n, n, true, true, matvec!)
    return K_op, _mf_precond_op(ls.precond, n, matvec!, smooth!)
end

function _setup_linear_ops(ig::NewmarkIntegrator, ls::KrylovLinearSolver, p)
    # ig.U is full-DOF in the Norma-shape integrator state; the linear
    # solver acts on the free-DOF (Newton) subspace, so size the operator
    # from unknown_dofs and pass the free-DOF view as the linearization
    # point for the matvec.
    Uu = _displacement(ig); n = length(Uu); c_M = ig.c_M
    ls.assembled && return (nothing, nothing)
    matvec! = (y, v) -> _eff_stiffness_matvec!(y, v, ig.asm, Uu, c_M, p)
    K_eff_op = LinearOperator(Float64, n, n, true, true, matvec!)
    # The smoother keeps the exact action here: K_eff = K + c_M·M needs a
    # reduced-precision `mass_action` to match, which does not exist yet, and
    # the measured 74.4% action share (benchmark/vcycle_bench.jl) is a
    # quasi-static figure — implicit dynamics defaults to Jacobi anyway
    # (benchmark_report.md §3), so there is no evidence this path would pay.
    return K_eff_op, _mf_precond_op(ls.precond, n, matvec!, matvec!)
end

# --------------------------------------------------------------------------- #
# Device-resident preconditioned CG (the matrix-free path)
#
# Krylov.jl's cg reads several scalars back from the device every iteration:
# each dot product and convergence check is a blocking device-to-host copy
# that drains the whole command pipeline before the host can issue the next
# kernel.  A CUPTI profile of a production Newmark step on an A100 measured
# the result — 2,430 cuStreamSynchronize calls per step and a device busy
# only 36% of the trace (benchmark/crosscode/README.md §4).
#
# This loop keeps every recurrence scalar (ρ = r·z, p·Ap, α, β) in 1-element
# device arrays: dot products reduce on the device (two-stage tree reduction,
# no atomics), scalar arithmetic is 1-element broadcasts, and the vector
# updates broadcast the device scalar directly, so nothing returns to the
# host inside an iteration.  The host reads back one number — ρ — every
# _CG_CHECK_EVERY iterations to decide termination, so the pipeline drains
# once per block instead of several times per iteration.  Runs may therefore
# execute up to _CG_CHECK_EVERY − 1 iterations past convergence, which costs
# far less than the synchronizations it removes, and the reported iteration
# count is a multiple of the block size.
#
# Convergence semantics match Krylov.cg with preconditioner M: terminate when
# sqrt(r·z) ≤ rtol·sqrt(r₀·z₀), the M-norm of the residual.  A non-finite ρ
# (the eversion guard's NaN poison arriving through the operator) exits the
# loop immediately and reports not-converged; the NaN then reaches the Newton
# logic through the solution vector exactly as on the Krylov path.
# --------------------------------------------------------------------------- #

const _CG_CHECK_EVERY     = 8
const _CG_REDUCE_NGROUPS  = 256
const _CG_REDUCE_GS       = 256      # workgroup size; must be a power of two

KA.@kernel function _dot_partials_kernel!(
    partials, x, y, ::Val{GS},
) where {GS}
    gid  = KA.@index(Group, Linear)
    lid  = KA.@index(Local, Linear)
    tile = KA.@localmem Float64 (GS,)
    n      = length(x)
    stride = GS * _CG_REDUCE_NGROUPS
    i   = (gid - 1) * GS + lid
    acc = 0.0
    while i <= n
        @inbounds acc += x[i] * y[i]
        i += stride
    end
    @inbounds tile[lid] = acc
    KA.@synchronize
    s = GS >> 1
    while s > 0
        if lid <= s
            @inbounds tile[lid] += tile[lid + s]
        end
        KA.@synchronize
        s >>= 1
    end
    if lid == 1
        @inbounds partials[gid] = tile[1]
    end
end

KA.@kernel function _reduce_partials_kernel!(
    out, partials, ::Val{GS},
) where {GS}
    lid  = KA.@index(Local, Linear)
    tile = KA.@localmem Float64 (GS,)
    acc = 0.0
    i = lid
    while i <= length(partials)
        @inbounds acc += partials[i]
        i += GS
    end
    @inbounds tile[lid] = acc
    KA.@synchronize
    s = GS >> 1
    while s > 0
        if lid <= s
            @inbounds tile[lid] += tile[lid + s]
        end
        KA.@synchronize
        s >>= 1
    end
    if lid == 1
        @inbounds out[1] = tile[1]
    end
end

# out[1] = x·y, entirely on the device; no host synchronization.
function _device_dot!(out, partials, x, y, backend)
    _dot_partials_kernel!(backend, _CG_REDUCE_GS)(
        partials, x, y, Val(_CG_REDUCE_GS);
        ndrange = _CG_REDUCE_GS * _CG_REDUCE_NGROUPS)
    _reduce_partials_kernel!(backend, _CG_REDUCE_GS)(
        out, partials, Val(_CG_REDUCE_GS); ndrange = _CG_REDUCE_GS)
    return out
end

# The CPU backend cannot split `@synchronize` inside the reduction loops (KA
# limitation), and has no command pipeline to drain in the first place — a
# plain dot costs nothing extra there.  The rest of the CG loop is identical,
# so CPU runs exercise the same block structure and device-scalar updates.
function _device_dot!(out, partials, x, y, ::KA.CPU)
    out[1] = dot(x, y)
    return out
end

struct _DeviceCGWorkspace{V}
    x        ::V
    r        ::V
    z        ::V
    p        ::V
    Ap       ::V
    ρ        ::V   # 1-element device scalars
    ρ_prev   ::V
    pAp      ::V
    α        ::V
    β        ::V
    partials ::V
end

const _device_cg_ws = Ref{Any}(nothing)

function _device_cg_workspace(R)
    ws = _device_cg_ws[]
    if !(ws isa _DeviceCGWorkspace) || length(ws.x) != length(R) ||
       typeof(ws.x) !== typeof(similar(R))
        s(len) = (v = similar(R, len); fill!(v, zero(eltype(R))); v)
        ws = _DeviceCGWorkspace(
            s(length(R)), s(length(R)), s(length(R)), s(length(R)), s(length(R)),
            s(1), s(1), s(1), s(1), s(1), s(_CG_REDUCE_NGROUPS))
        _device_cg_ws[] = ws
    end
    return ws
end

_apply_precond!(z, ::Nothing, r) = copyto!(z, r)
_apply_precond!(z, M_op, r)      = mul!(z, M_op, r)

# One device-to-host read of a 1-element array (the only synchronization the
# CG loop performs, once per _CG_CHECK_EVERY iterations).
_cg_scalar(a) = Array(a)[1]

function _device_pcg!(ΔU, A_op, R, M_op, rtol, itmax)
    ws = _device_cg_workspace(R)
    backend = KA.get_backend(R)
    (; x, r, z, p, Ap, ρ, ρ_prev, pAp, α, β, partials) = ws

    fill!(x, zero(eltype(x)))
    copyto!(r, R)
    _apply_precond!(z, M_op, r)
    copyto!(p, z)
    _device_dot!(ρ, partials, r, z, backend)
    ρ0 = _cg_scalar(ρ)
    if !isfinite(ρ0) || ρ0 <= 0.0
        # Zero RHS (already solved) or a poisoned residual; nothing to iterate.
        copyto!(ΔU, x)
        return 0, sqrt(abs(ρ0)), ρ0 == 0.0
    end
    tol2 = (rtol * sqrt(ρ0))^2

    iters     = 0
    converged = false
    ρ_h       = ρ0
    # Block size adapts to the measured convergence rate.  A fixed block
    # overruns by up to blk − 1 iterations past convergence, which is cheap
    # for Jacobi (many inexpensive iterations) but measurably wasteful for
    # Chebyshev, whose iterations each apply a five-matvec smoother and
    # converge in a few dozen.  From the contraction over the previous block,
    # θ = (ρ_now/ρ_before)^(1/blk), estimate the iterations left to reach
    # tolerance and size the next block to land just short of it; prediction
    # error replaces the fixed overrun, and CG's superlinear convergence makes
    # undershooting self-correcting at the next check.
    blk = _CG_CHECK_EVERY
    while iters < itmax
        blk_run = min(blk, itmax - iters)
        for _ in 1:blk_run
            mul!(Ap, A_op, p)
            _device_dot!(pAp, partials, p, Ap, backend)
            # pAp ≤ 0 can only mean r ≈ 0 to roundoff (A is SPD); a zero α
            # freezes the iterate instead of dividing 0/0 into a NaN.
            @. α = ifelse(pAp > 0.0, ρ / pAp, 0.0)
            @. x += α * p
            @. r -= α * Ap
            _apply_precond!(z, M_op, r)
            copyto!(ρ_prev, ρ)
            _device_dot!(ρ, partials, r, z, backend)
            @. β = ρ / ρ_prev
            @. p = z + β * p
            iters += 1
        end
        ρ_before = ρ_h
        ρ_h = _cg_scalar(ρ)
        if ρ_h <= tol2
            converged = true
            break
        end
        isfinite(ρ_h) || break
        θ = (ρ_h / ρ_before)^(1 / blk_run)
        if isfinite(θ) && 0.0 < θ < 1.0
            m = log(tol2 / ρ_h) / log(θ)
            blk = clamp(ceil(Int, 0.6 * m), 1, _CG_CHECK_EVERY)
        else
            blk = _CG_CHECK_EVERY
        end
    end
    copyto!(ΔU, x)
    return iters, sqrt(max(ρ_h, 0.0)), converged
end

# --------------------------------------------------------------------------- #
# Linear solvers: _linear_solve!(ls, ig, p, ops) → (ΔU, t_solve)
# Sign convention: K_eff · ΔU = ig.R_eff  (ig.R_eff is already negated residual)
# --------------------------------------------------------------------------- #

# Reusable CHOLMOD factor for the direct path, and a sticky flag set once a
# Cholesky attempt has failed.  Both reset per run by `_init_assembly_cache!`.
const _direct_chol   = Ref{Any}(nothing)
const _direct_use_lu = Ref(false)

# Cholesky when the tangent is symmetric positive definite, LU otherwise.
#
# This was an unconditional `lu(K)`, justified by a note claiming assembly was
# "~1e-7 asymmetric (AD material tangent)" and that `cholesky(Symmetric(K))`
# therefore gave a ~50% solve error.  Neither holds: measured asymmetry is
# 1.0e-16, and Cholesky agrees with LU to 1.9e-15 from either triangle while
# being more accurate against the original matrix.  `issymmetric(K)` does return
# false, since it tests exact equality -- the likely source of the original
# diagnosis.
#
# The factor is built ONCE and refactorized in place thereafter.  This is not an
# optimization but a correctness requirement: `_factorize_direct` runs per
# Newton iteration, and calling `cholesky` each time allocates a fresh ~6.3 GB
# supernodal factor.  CHOLMOD allocates outside Julia's heap, so Julia sees a
# small wrapper, feels no memory pressure and never collects; an 8-step run
# reached 59.5 GB and was OOM-killed at step 6.  `cholesky!` reuses the one
# factor, which is valid here because values change every Newton iteration but
# the sparsity pattern does not.
#
# Measured at 530k DOF / 40.2M nonzeros, 24 threads, over repeated
# factorizations with RSS sampled per repetition:
#
#                    per factorization   steady RSS   ||Kx-b||/||b||
#   lu                     37.8 s           9.6 GB       1.67e-14
#   cholesky! in place      6.7 s          16.4 GB       4.91e-15
#
# So 5.6x faster and more accurate, at ~6.8 GB of retained factor.  RSS is flat
# across repetitions, which is the property that matters.
function _factorize_direct(K)
    if !_direct_use_lu[] && _check_tangent_symmetry!(K)
        try
            if _direct_chol[] === nothing
                _direct_chol[] = cholesky(Symmetric(K, :L))
            else
                cholesky!(_direct_chol[], Symmetric(K, :L))
            end
            return _direct_chol[]
        catch e
            e isa LinearAlgebra.PosDefException || rethrow()
            # Buckling, softening, an everted element: the tangent is no longer
            # positive definite.  Drop the factor and stay on LU for the rest of
            # the run rather than paying a failed factorization every iteration.
            _direct_use_lu[] = true
            _direct_chol[]   = nothing
            _carina_log(4, :solve,
                        "Tangent lost positive definiteness; switching to LU.")
        end
    end
    return lu(K)
end

function _linear_solve!(::DirectLinearSolver, ig, p, _ops)
    K  = FEC.stiffness(ig.asm)
    af = _asm_flags
    t  = @elapsed begin
        if af.compute_factorization
            F = _factorize_direct(K)
            if af.is_linear
                _factorization_cache[] = F
                af.compute_factorization = false
            end
        else
            F = _factorization_cache[]
        end
        ΔU = F \ residual(ig)
    end
    return ΔU, t
end

# x_cur (current nodal coordinates) is consumed only by the AMG method, which
# rebuilds its near-nullspace from it; the other preconditioners ignore it.
# Which preconditioners actually read the matrix in `_build_precond_op`.
# Jacobi works from `precond.inv_diag`, already filled by `setup_jacobian!`.
_precond_reads_matrix(::Preconditioner)             = true
_precond_reads_matrix(::NoPreconditioner)           = false
_precond_reads_matrix(::JacobiPreconditioner)       = false

function _build_precond_op(::NoPreconditioner, K_sparse, n, x_cur)
    return nothing
end
function _build_precond_op(precond::JacobiPreconditioner, K_sparse, n, x_cur)
    return _jacobi_precond_op(precond, n)
end
function _build_precond_op(::ICPreconditioner, K_sparse, n, x_cur)
    # Incomplete LDLᵀ factorization.
    # K_sparse is already Symmetric from the symmetrization in _linear_solve!.
    # α > 0 adds a diagonal shift to guarantee positive definiteness
    # of the factor (at the cost of a weaker preconditioner).
    F_ic = lldl(K_sparse; memory=20, α=0.01)
    return LinearOperator(Float64, n, n, true, true,
        (y, v) -> ldiv!(y, F_ic, v))
end
# AMG on the quasi-static assembled path.  K_sparse arrives as a Symmetric
# wrapper; SA setup needs a plain CSC matrix.  The internal lazy guard in
# the build (P === nothing || rebuild) makes the per-Newton calls cheap;
# c_M = 0 since there is no mass term in the quasi-static tangent.
function _build_precond_op(precond::AMGPreconditioner, K_sparse, n, x_cur)
    _update_amg_precond_assembled!(precond, sparse(K_sparse), 0.0, x_cur)
    P = precond.P
    return LinearOperator(Float64, n, n, true, true,
        (y, v) -> ldiv!(y, P, v))
end
# Chebyshev-Jacobi on assembled path: polynomial on the symmetrically
# scaled system S = D⁻¹/²AD⁻¹/².  Penalty BCs create eigenvalues ~1e15
# in A but only ~1 in S, making the polynomial effective.
# Preconditioner: M = D⁻¹/² p_k(S)² D⁻¹/²  ≈ A⁻¹, and M is SPD.
function _build_precond_op(precond::ChebyshevPreconditioner, K_sparse, n, x_cur)
    d = diag(K_sparse)
    inv_diag = similar(d)
    @. inv_diag = 1.0 / max(abs(d), eps(Float64))

    # Estimate λ_max of D⁻¹A via power method
    matvec! = (y, v) -> mul!(y, K_sparse, v)
    _estimate_lambda_max!(precond, matvec!, inv_diag, n)

    # 4th-kind Chebyshev with baked-in Jacobi
    return LinearOperator(Float64, n, n, true, true,
        (y, v) -> _apply_chebyshev_precond!(y, v, precond, matvec!, inv_diag))
end

function _linear_solve!(ls::KrylovLinearSolver, ig::QuasiStaticIntegrator, p, ops)
    U = ig.U; asm = ig.asm; n = length(U)
    K_op, M_op = ops
    R = residual(ig)   # K·ΔU = R_eff (positive, already negated)
    af = _asm_flags

    # GPU Cholesky direct path: for linear elastic on GPU, factorize K on CPU
    # once, upload L to GPU, and solve via sparse triangular solves.
    if af.is_linear && _gpu_cholesky_L[] !== nothing
        t = @elapsed begin
            (L_gpu, perm, iperm) = _gpu_cholesky_L[]
            # P*K*P' = L*L' → x = P' * (L' \ (L \ (P*b)))
            Pb = R[perm]                           # apply permutation
            y  = LowerTriangular(L_gpu) \ Pb       # forward solve
            ΔU = LowerTriangular(L_gpu)' \ y       # backward solve
            ΔU = ΔU[iperm]                         # inverse permutation
        end
        _carina_logf(8, :solve, "    GPU Cholesky solve: %.2fs", t)
        return ΔU, t
    end

    ΔU = similar(R)
    local niter::Int, r_cg::Float64, solved::Bool
    t_kry = @elapsed begin
        if ls.assembled
            # Zero workspace solution to prevent stale warm-start.
            fill!(ls.workspace.x, zero(eltype(ls.workspace.x)))
            K_raw = FEC.stiffness(asm)
            A_op = _assembled_operator(K_raw, n)
            if af.compute_factorization
                x_cur = ls.precond isa AMGPreconditioner ? _current_coords(p) : nothing
                # Only IC, AMG and Chebyshev read the matrix in
                # `_build_precond_op`; Jacobi and none ignore it, so they never
                # pay for the symmetrized copy.
                K_sparse = _precond_reads_matrix(ls.precond) ?
                    Symmetric((K_raw + K_raw') / 2, :L) : K_raw
                M_op_asm = _build_precond_op(ls.precond, K_sparse, n, x_cur)
                if af.is_linear
                    _precond_op_cache[] = M_op_asm
                    af.compute_factorization = false
                end
            else
                M_op_asm = _precond_op_cache[]
            end
            if M_op_asm === nothing
                Krylov.krylov_solve!(ls.workspace, A_op, R;
                                     atol=0.0, rtol=ls.rtol, itmax=ls.itmax, history=true)
            else
                Krylov.krylov_solve!(ls.workspace, A_op, R;
                                     M=M_op_asm, atol=0.0, rtol=ls.rtol, itmax=ls.itmax, history=true)
            end
            copyto!(ΔU, Krylov.solution(ls.workspace))
            res    = ls.workspace.stats.residuals
            r_cg   = isempty(res) ? NaN : res[end]
            niter  = ls.workspace.stats.niter
            solved = ls.workspace.stats.solved
        else
            niter, r_cg, solved = _device_pcg!(ΔU, K_op, R, M_op, ls.rtol, ls.itmax)
        end
    end
    # Feed the iteration count back to the AMG staleness detector so a hierarchy
    # built at an earlier configuration gets rebuilt once it stops paying for
    # itself.  No-op for every other preconditioner.
    _amg_track_iters!(ls.precond, niter)
    _carina_logf(8, :solve, "    CG: %d iters : |r|_CG = %.2e : %s",
                 niter, r_cg, _cg_status_str(solved))

    # After first successful CG solve for linear elastic on GPU, build the
    # GPU Cholesky cache: factorize K on CPU, upload L to GPU.
    # Subsequent solves use fast GPU triangular solves instead of CG.
    if af.is_linear && !ls.assembled && _gpu_cholesky_L[] === nothing
        _build_gpu_cholesky!(ig.asm, p)
    end

    return ΔU, t_kry
end

function _linear_solve!(ls::KrylovLinearSolver, ig::NewmarkIntegrator, p, ops)
    asm = ig.asm
    R = residual(ig)
    n = length(R)             # free-DOF Newton system size (ig.U is full-DOF)
    K_eff_op, M_op_mf = ops
    af = _asm_flags
    ΔU = similar(R)
    t_kry = @elapsed begin
        try
            if ls.assembled
                # Zero workspace solution to prevent stale warm-start.
                fill!(ls.workspace.x, zero(eltype(ls.workspace.x)))
                K_eff_raw = FEC.stiffness(asm)
                # Threaded CSR apply when the tangent is symmetric, which it is
                # to roundoff (measured 1.0e-16, not the ~1e-7 an older comment
                # here claimed).  Only the IC branch below needs the matrix
                # itself; everything else applies it, so only that branch pays
                # for the symmetrized copy.
                A_op = _assembled_operator(K_eff_raw, length(R))
                if ls.precond isa ICPreconditioner
                    K_eff_sparse = Symmetric((K_eff_raw + K_eff_raw') / 2, :L)
                    if af.compute_factorization
                        F_ic = lldl(K_eff_sparse)
                        if af.is_linear
                            _factorization_cache[] = F_ic
                            af.compute_factorization = false
                        end
                    else
                        F_ic = _factorization_cache[]
                    end
                    ΔU_vec, cg_hist = IterativeSolvers.cg(K_eff_sparse, R;
                        Pl=F_ic, abstol=0.0, reltol=ls.rtol, log=true)
                elseif ls.precond isa AMGPreconditioner
                    ΔU_vec, cg_hist = IterativeSolvers.cg(A_op, R;
                        Pl=ls.precond.P, abstol=0.0, reltol=ls.rtol,
                        maxiter=ls.itmax, log=true)
                    _amg_track_iters!(ls.precond, length(cg_hist.data[:resnorm]))
                else
                    ΔU_vec, cg_hist = IterativeSolvers.cg(A_op, R;
                        abstol=0.0, reltol=ls.rtol, log=true)
                end
                _carina_logf(8, :solve, "    CG: %d iters : |r|_CG = %.2e : %s",
                    length(cg_hist.data[:resnorm]),
                    cg_hist.data[:resnorm][end],
                    _cg_status_str(cg_hist.isconverged))
                copyto!(ΔU, ΔU_vec)
            else
                niter, r_cg, solved = _device_pcg!(ΔU, K_eff_op, R, M_op_mf,
                                                   ls.rtol, ls.itmax)
                _amg_track_iters!(ls.precond, niter)
                _carina_logf(8, :solve, "    CG: %d iters : |r|_CG = %.2e : %s",
                             niter, r_cg, _cg_status_str(solved))
            end
        catch e
            e isa _MATH_ERRORS || rethrow()
            _carina_logf(4, :solve, "CG solve: caught %s", typeof(e))
            ig.failed[] = true
        end
    end
    return ΔU, t_kry
end

# --------------------------------------------------------------------------- #
# LBFGS helpers (dispatch on integrator type for Newmark vs QS differences)
# --------------------------------------------------------------------------- #

# ---- _lbfgs_init_M_dU! ----

_lbfgs_init_M_dU!(::QuasiStaticIntegrator, ls) = nothing

function _lbfgs_init_M_dU!(::NewmarkIntegrator, ls)
    fill!(ls.M_dU, zero(eltype(ls.M_dU)))
end

# ---- _lbfgs_precompute_M_d! ----

_lbfgs_precompute_M_d!(::QuasiStaticIntegrator, ls, p) = nothing

function _lbfgs_precompute_M_d!(ig::NewmarkIntegrator, ls, p)
    # Linearization point passes through as the free-DOF Uu; ls.d is the
    # free-DOF LBFGS direction.  Uses the free-DOF action because the
    # LBFGS step direction perturbs unknowns only (no BC contribution).
    FEC.assemble_matrix_free_action!(ig.asm, FEC.mass_action,
                                      _displacement(ig), ls.d, p)
    copyto!(ls.M_d, FEC.hvp(ig.asm, ls.d))
end

# ---- _lbfgs_update_M_dU! ----

_lbfgs_update_M_dU!(::QuasiStaticIntegrator, ls, step) = nothing

function _lbfgs_update_M_dU!(::NewmarkIntegrator, ls, step)
    @. ls.M_dU += step * ls.M_d
end

# ---- _lbfgs_trial_rhs! ----
# Sets ig.R_eff at trial point U + step*d.

function _lbfgs_trial_rhs!(ig::QuasiStaticIntegrator, ls, step, p)
    U = ig.U; asm = ig.asm
    @. ls.q = U + step * ls.d
    FEC.assemble_vector!(asm, FEC.residual, ls.q, p)
    FEC.assemble_vector_neumann_bc!(asm, ls.q, p)
    FEC.assemble_vector_source!(asm, ls.q, p)
    R_int_trial = FEC.residual(asm)
    @. ig.R_eff = -R_int_trial
    _apply_point_loads!(ig.R_eff, FEC.current_time(p.times))
end

function _lbfgs_trial_rhs!(ig::NewmarkIntegrator, ls, step, p)
    α_hht = ig.α_hht; c_M = ig.c_M
    # ls.q (= trial Uu) is free-DOF; built from the free slice of ig.U
    # plus a scaled LBFGS direction.
    Uu = _displacement(ig)
    @. ls.q = Uu + step * ls.d
    FEC.assemble_vector!(ig.asm, FEC.residual, ls.q, p)
    FEC.assemble_vector_neumann_bc!(ig.asm, ls.q, p)
    FEC.assemble_vector_source!(ig.asm, ls.q, p)
    R_int_trial = FEC.residual(ig.asm)
    @. ig.R_eff = -((1 + α_hht) * R_int_trial + c_M * (ls.M_dU + step * ls.M_d) - α_hht * ig.F_int_n)
    _apply_point_loads!(ig.R_eff, FEC.current_time(p.times))
end
