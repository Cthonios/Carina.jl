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
# The assembled tangent keeps its sparsity pattern for the whole run, so the
# Int32 index arrays of the CSR view are built once per pattern and only the
# values are copied at each Newton iteration (the conversion took 2 s per
# solve at 3.6 M unknowns).
const _csr_view_cache = Ref{Any}(nothing)

function _symmetric_csc_as_csr(backend::KA.CPU, K::SparseArrays.SparseMatrixCSC)
    c = _csr_view_cache[]
    if c !== nothing && c.n == size(K, 1) && length(c.colval) == length(K.rowval) &&
       c.colptr_ref === K.colptr && c.rowval_ref === K.rowval
        _par_copyto!(c.csr.nzval, K.nzval)
        return c.csr
    end
    csr = _symmetric_csc_as_csr_new(backend, K)
    _csr_view_cache[] = (n = size(K, 1), colval = csr.colval, colptr_ref = K.colptr,
                         rowval_ref = K.rowval, csr = csr)
    return csr
end
_symmetric_csc_as_csr(backend, K::SparseArrays.SparseMatrixCSC) =
    _symmetric_csc_as_csr_new(backend, K)

function _symmetric_csc_as_csr_new(backend, K::SparseArrays.SparseMatrixCSC)
    rowptr = KA.allocate(backend, Int32, length(K.colptr))
    colval = KA.allocate(backend, Int32, length(K.rowval))
    nzval  = KA.allocate(backend, Float64, length(K.nzval))
    _par_copyto!(rowptr, Int32.(K.colptr))
    _par_copyto!(colval, Int32.(K.rowval))
    _par_copyto!(nzval, K.nzval)
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
    A = SparseArrays.sparse((K_eff_raw + K_eff_raw') / 2)   # exact symmetry for the SA setup
    t = @elapsed begin
        B = _rigid_body_modes(x_cur, precond.udofs)
        precond.P = _cpu_amg_vcycle(A, B)
    end
    precond.built_c_M  = c_M
    precond.base_iters = 0
    precond.rebuild    = false
    precond.nbuilds   += 1
    _carina_logf(4, :solve, "    AMG hierarchy build #%d (%s)", precond.nbuilds, format_time(t))
    return nothing
end

# The V-cycle of the assembled CPU path.  It is the V-cycle of the device
# path (`DeviceAMGHierarchy`, `_amg_vcycle!`: smoothed aggregation with the
# rigid-body modes, two damped Jacobi sweeps per side, a dense coarse solve)
# run on the KernelAbstractions CPU backend, with the fine operator the
# matrix the hierarchy was built on, applied as a threaded CSR product.  The
# V-cycle of AlgebraicMultigrid.jl, used before 2026-10-16, runs on one
# thread: its Gauss-Seidel smoother is sequential and its sparse products are
# those of SparseArrays (CSC, single-threaded), so the assembled CG with AMG
# used one core of the host whatever the thread count (3.65 M unknowns on
# 36 cores: host load 1.3 during the solve).
struct _CPUAMGVCycle
    hierarchy::DeviceAMGHierarchy
    fine::DeviceCSR
end

function _cpu_amg_vcycle(A::SparseArrays.SparseMatrixCSC, B::Matrix{Float64})
    backend = KA.CPU()
    ml = _sa_hierarchy_lowmem(A, B)
    dinv = _par_copyto!(Vector{Float64}(undef, size(A, 1)), 1.0 ./ Vector(diag(A)))
    any(!isfinite, dinv) && error("Zero diagonal in the fine AMG operator.")
    h = DeviceAMGHierarchy(backend, ml, dinv, _host_lambda_max(A, dinv))
    return _CPUAMGVCycle(h, _symmetric_csc_as_csr_new(backend, A))
end

function LinearAlgebra.ldiv!(y::AbstractVector, P::_CPUAMGVCycle, v::AbstractVector)
    backend = KA.CPU()
    fine! = (out, x) -> _csr_mul!(out, P.fine, x, backend)
    _amg_vcycle!(y, v, P.hierarchy, fine!, backend)
    return y
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

# The time integrator updates the Dirichlet values of the device parameters
# only; the host copy keeps those of the initial time.  Host-side assembly on
# the state of a device run (the AMG hierarchy, the output) takes the current
# values from the device first, or it sees the constrained DOFs at their
# initial values.
function _sync_host_dirichlet!(p_cpu, p = _dev_params_ref[])
    (p === nothing || p === p_cpu) && return nothing
    src, dst = p.dirichlet_bcs.bc_cache, p_cpu.dirichlet_bcs.bc_cache
    copyto!(dst.vals, Adapt.adapt(Array, src.vals))
    copyto!(dst.vals_dot, Adapt.adapt(Array, src.vals_dot))
    copyto!(dst.vals_dot_dot, Adapt.adapt(Array, src.vals_dot_dot))
    return nothing
end

function _compute_gpu_amg_precond(asm_cpu, template, smoother::Symbol = :jacobi,
                                  compressible::Bool = false)
    smoother in (:jacobi, :block_jacobi) ||
        error("Unknown AMG smoother :$smoother. Expected :jacobi or :block_jacobi.")
    compressible && _compressible_lame_indices(_cpu_params_ref[])   # fails loudly if a model has no Lamé constants
    inv_d = similar(template); fill!(inv_d, 0.0)
    # Block storage is allocated at the first hierarchy build, where the free
    # DOF structure is read from the assembled operator; the three column
    # buffers are free-DOF sized and can be made now.
    blk_cols = smoother === :block_jacobi ?
        ntuple(_ -> (v = similar(template); fill!(v, 0.0); v), 3) : nothing
    return GPUAMGPreconditioner(collect(asm_cpu.dof.unknown_dofs),
                                inv_d, nothing, 1.0, -1.0, 0, false, 0,
                                smoother, nothing, nothing, blk_cols,
                                compressible, nothing, nothing, nothing)
end

# --------------------------------------------------------------------------- #
# The compressible operator of the AMG preconditioner (`operator: compressible`)
#
# At κ/μ ≫ 1 the smoothed-aggregation hierarchy built on the tangent K(κ)
# coarsens on the κ-couplings and its damped Jacobi smoother, scaled to the
# diagonal of K(κ), leaves the deviatoric error to the Krylov iteration: on the
# punch benchmark (research/tet15-p1/precond) the CG count grows as κ^0.4,
# 274 → 1982 for κ/μ from 10² to 10⁵.  The same V-cycle built on K(κ'), the
# tangent of the material with its bulk modulus set to the shear modulus
# (κ' = μ, ν' = 1/8), and applied to K(κ) needs 2.5 times fewer iterations at
# κ/μ = 10⁴ (523 against 1283) and grows slowly with κ.  The hierarchy, the
# fine diagonal and the fine smoothing action are then evaluated with the
# material's Lamé constant λ' = κ' − 2μ/3 = μ/3 in place of λ; the Krylov
# operator is the tangent K(κ) itself, so the solution is unchanged.
# --------------------------------------------------------------------------- #

# Index of λ and μ in the property vector of each block, from the model's
# property names.  Errors if a model has no Lamé constants.
function _compressible_lame_indices(p_cpu)
    p_cpu === nothing && error("The compressible AMG operator needs the host parameters.")
    map(enumerate(values(p_cpu.physics))) do (b, physics)
        names = CM.property_names(physics.constitutive_model)
        iλ = findfirst(==("Lamé's first constant"), names)
        iμ = findfirst(==("shear modulus"), names)
        (iλ === nothing || iμ === nothing) && error(
            "The compressible AMG operator requires a material with the properties " *
            "\"Lamé's first constant\" and \"shear modulus\"; block $b has " *
            "$(typeof(physics.constitutive_model)) with $(names).")
        (iλ, iμ)
    end
end

# Linear indices into the property data of every λ entry, with λ' = μ/3 for
# each.
function _compressible_lame_entries(p_cpu)
    pf = p_cpu.properties
    entries = Tuple{Int, Float64}[]
    for (b, (iλ, iμ)) in enumerate(_compressible_lame_indices(p_cpu))
        ne = pf.isblockconstant[b] == FEC.PROPS_CONST ? 1 : pf.nelems[b]
        for e in 1:ne
            pv = FEC.properties(pf, e, b)
            push!(entries, (pv.start + iλ - 1, pv[iμ] / 3))
        end
    end
    return entries
end

# Runs `f()` with the host properties at λ' and restores them.
function _with_compressible_properties(f, p_cpu)
    data = p_cpu.properties.data
    entries = _compressible_lame_entries(p_cpu)
    saved = [data[i] for (i, _) in entries]
    for (i, λp) in entries
        data[i] = λp
    end
    try
        return f()
    finally
        for (k, (i, _)) in enumerate(entries)
            data[i] = saved[k]
        end
    end
end

# The parameters whose tangent the smoother approximates: the run's own, or a
# copy with the compressible properties that shares every other array.
_smoothing_params(::Preconditioner, p) = p
function _smoothing_params(precond::GPUAMGPreconditioner, p)
    precond.compressible || return p
    if precond.params_kp === nothing
        p_cpu = _cpu_params_ref[]
        props_h = deepcopy(Adapt.adapt(Array, p.properties))
        for (i, λp) in _compressible_lame_entries(p_cpu)
            props_h.data[i] = λp
        end
        props_d = Adapt.adapt(Base.typename(typeof(p.properties.data)).wrapper, props_h)
        precond.params_kp = FEC.Parameters(
            p.ics, p.dirichlet_bcs, p.neumann_bcs, p.periodic_bcs, p.robin_bcs,
            p.sources, p.times, p.physics, props_d, p.state_old, p.state_new,
            p.coords, p.field, p.field_old, p.hvp_scratch_field)
    end
    return precond.params_kp
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
        _sync_host_dirichlet!(p_cpu)
        FEC._update_for_assembly!(p_cpu, asm_cpu.dof, U_cpu)
        if precond.compressible
            _with_compressible_properties(p_cpu) do
                FEC.assemble_stiffness!(asm_cpu, FEC.stiffness, U_cpu, p_cpu)
            end
        else
            FEC.assemble_stiffness!(asm_cpu, FEC.stiffness, U_cpu, p_cpu)
        end
        if c_M != 0.0
            FEC.assemble_mass!(asm_cpu, FEC.mass, U_cpu, p_cpu)
            @. asm_cpu.stiffness_storage += c_M * asm_cpu.mass_storage
        end
        K_raw = FEC.stiffness(asm_cpu)
        A = SparseArrays.sparse((K_raw + K_raw') / 2)
        B = _rigid_body_modes(_current_coords(p_cpu), precond.udofs)
        ml = _sa_hierarchy_lowmem(A, B)
        if precond.smoother === :block_jacobi
            # The block map is fixed by the free DOF structure; the block
            # values here are the build-point ones and are refreshed every
            # Newton iteration by `_refresh_gpu_amg_blocks!`.
            blk_dof_h, blk_inv_h, lmax = _host_block_structure(A, precond.udofs)
            if precond.blk_dof === nothing
                precond.blk_dof = KA.allocate(backend, Int32, 3, size(blk_dof_h, 2))
                precond.blk_inv = KA.allocate(backend, Float64, 3, 3, size(blk_dof_h, 2))
            end
            copyto!(precond.blk_dof, blk_dof_h)
            copyto!(precond.blk_inv, blk_inv_h)
            precond.lmax_fine = lmax
        else
            dinv_h = 1.0 ./ Vector(diag(A))
            precond.lmax_fine = _host_lambda_max(A, dinv_h)
        end
        precond.hierarchy = DeviceAMGHierarchy(backend, ml, precond.inv_diag,
                                               precond.lmax_fine;
                                               blk_inv1 = precond.blk_inv,
                                               blk_dof1 = precond.blk_dof)
    end
    precond.built_c_M  = c_M
    precond.base_iters = 0
    precond.rebuild    = false
    precond.nbuilds   += 1
    _carina_logf(4, :solve, "    GPU AMG hierarchy build #%d (%s)",
                 precond.nbuilds, format_time(t))
    return nothing
end

# Block-smoother refresh: three diagonal-shaped assemblies, one per column of
# the nodal block, then a device inversion.  Runs after the hierarchy build,
# which is what allocates `blk_dof`/`blk_inv` on the first call.  Three
# assemblies where the scalar path needs one: ~60 ms against ~66 s of CG per
# Newton iteration on the 530k-DOF torsion bar.
function _refresh_gpu_amg_blocks!(precond::GPUAMGPreconditioner, asm, U, p)
    precond.smoother === :block_jacobi || return nothing
    precond.blk_dof === nothing && return nothing     # hierarchy not built yet
    B1, B2, B3 = precond.blk_cols
    FEC.assemble_diagonal!(asm, StiffnessBlockColumn{1}(), U, p); copyto!(B1, FEC.diagonal(asm))
    FEC.assemble_diagonal!(asm, StiffnessBlockColumn{2}(), U, p); copyto!(B2, FEC.diagonal(asm))
    FEC.assemble_diagonal!(asm, StiffnessBlockColumn{3}(), U, p); copyto!(B3, FEC.diagonal(asm))
    backend = _backend_ref[]
    _block_invert_kernel!(backend)(precond.blk_inv, precond.blk_dof, B1, B2, B3;
                                   ndrange = size(precond.blk_dof, 2))
    return nothing
end

# Per-Newton-iteration updates: refresh the fine diagonal (used by the
# V-cycle's fine smoother), then lazily rebuild the hierarchy.
function _update_gpu_amg_precond_qs!(precond::GPUAMGPreconditioner, asm, U, p)
    ps = _smoothing_params(precond, p)
    FEC.assemble_diagonal!(asm, StiffnessDiagonal(), U, ps)
    d = FEC.diagonal(asm)
    @. precond.inv_diag = 1.0 / max(abs(d), eps(Float64))
    _build_gpu_amg_hierarchy!(precond, 0.0, U)
    _refresh_gpu_amg_blocks!(precond, asm, U, ps)
    return nothing
end
_update_gpu_amg_precond_qs!(::Preconditioner, args...) = nothing

# The fine-level bound λ_max(D⁻¹K') of the V-cycle's damped Jacobi smoother,
# ω = 4/(3 λ_max), re-estimated before every matrix-free solve.  The bound of
# the hierarchy build is that of the tangent at the build, and the hierarchy
# is kept while the tangent changes; the diagonal and the smoothing action
# are those of the current tangent.  On the punch (case B, a/4, step 10) the
# bound of the build was 2.64 against 4.23-4.57 for the current tangent:
# ω λ_max = 2.1-2.3 > 2, so the smoother amplified the highest modes and the
# V-cycle was not positive (rᵀM⁻¹r < 0 in 46-60% of the solves).  The
# estimate is ten power iterations from the previous eigenvector estimate,
# with the 1.1 factor of the build's estimate.
const _FINE_LMAX_ITERS = 10

_refresh_fine_lmax!(::Preconditioner) = nothing
function _refresh_fine_lmax!(precond::GPUAMGPreconditioner)
    h = precond.hierarchy
    S = precond.smooth_op
    (h === nothing || S === nothing) && return nothing
    if precond.lmax_vecs === nothing
        n = length(precond.inv_diag)
        v = similar(precond.inv_diag)
        copyto!(v, sin.(1.0:n))     # not smooth, so not close to a low mode
        precond.lmax_vecs = (v, similar(v), similar(v))
    end
    v, w, y = precond.lmax_vecs
    λ = 0.0
    v ./= norm(v)
    for _ in 1:_FINE_LMAX_ITERS
        S(w, v)
        _apply_fine_dinv!(y, w, precond, h)
        λ = norm(y)
        λ == 0.0 && return nothing
        @. v = y / λ
    end
    lmax = 1.1 * λ
    abs(lmax - h.lmax1) > 0.05 * h.lmax1 && _carina_logf(8, :solve,
        "    AMG fine-level bound λ_max(D⁻¹K'): %.3f → %.3f", h.lmax1, lmax)
    precond.lmax_fine = lmax
    precond.hierarchy = _with_lmax1(h, lmax)
    return nothing
end

# y = D⁻¹w for the fine smoother: the pointwise or the nodal-block inverse.
function _apply_fine_dinv!(y, w, precond::GPUAMGPreconditioner, h)
    if h.blk_inv1 === nothing
        @. y = precond.inv_diag * w
    else
        fill!(y, 0.0)
        _block_jacobi_kernel!(KA.get_backend(y))(y, w, h.blk_inv1, h.blk_dof1, 1.0;
                                                 ndrange = size(h.blk_dof1, 2))
    end
    return y
end

function _update_gpu_amg_precond_eff!(precond::GPUAMGPreconditioner, asm, U,
                                      c_M, p)
    precond.compressible && error(
        "The compressible AMG operator is implemented for the quasi-static integrator only.")
    # The block columns of K + c_M·M need a Newmark analog of
    # StiffnessBlockColumn; until it exists, the block smoother is quasi-static
    # only, and asking for it here is an error rather than a silent fallback
    # to blocks of K alone.
    precond.smoother === :block_jacobi && error(
        "AMG smoother \"block jacobi\" is not implemented for the Newmark " *
        "integrator; use \"jacobi\" or a quasi-static integrator.")
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
# The gate is not "did the user ask for it" but "does the model actually honor
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
        @warn "AMG smoother falling back to Float64: these physics have no " *
              "Float32 action (a model that promotes Float32 inputs back to " *
              "Float64, or the projected volumetric formulation). For a model, " *
              "make the `pk1_stress` constants type-generic (e.g. `one(J)/2` " *
              "rather than `0.5`) to enable it." models = unique(offenders)
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
# λ_max estimated via power method on D⁻¹A (10 iterations, low cost).
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
function _mf_precond_op(p::GPUAMGPreconditioner, n, matvec!, smooth!)
    p.smooth_op = smooth!
    return LinearOperator(Float64, n, n, true, true,
        (y, v) -> (_amg_vcycle!(y, v, p.hierarchy, smooth!,
                                KA.get_backend(p.inv_diag)); y))
end
_mf_precond_op(p::ChebyshevPreconditioner, n, matvec!, smooth!)  =
    _chebyshev_precond_op(p, n, matvec!; inv_sqrt_d=p.work3)

function _setup_linear_ops(ig::QuasiStaticIntegrator, ls::KrylovLinearSolver, p)
    U = ig.U; n = length(U)
    ls.assembled && return (nothing, nothing)
    matvec! = (y, v) -> _stiffness_matvec_qs!(y, v, ig.asm, U, p)
    ps = _smoothing_params(ls.precond, p)     # the run's parameters, or the compressible copy
    smooth! = _use_fp32_smoother(ls.precond, ig.asm) ?
        (y, v) -> _stiffness_matvec_qs_fp32!(y, v, ig.asm, U, ps) :
        ps === p ? matvec! : (y, v) -> _stiffness_matvec_qs!(y, v, ig.asm, U, ps)
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
    out[1] = _par_dot(x, y)
    return out
end

# Dot product in static blocks, one per thread, summed in block order (the
# same result for the same thread count).
function _par_dot(x::Vector{Float64}, y::Vector{Float64})
    n = length(x)
    nt = Threads.nthreads()
    (nt == 1 || n < 1 << 16) && return dot(x, y)
    part = zeros(nt)
    Threads.@threads :static for k in 1:nt
        lo = div((k - 1) * n, nt) + 1
        hi = div(k * n, nt)
        acc = 0.0
        @inbounds @simd for i in lo:hi
            acc += x[i] * y[i]
        end
        part[k] = acc
    end
    return sum(part)
end
_par_dot(x, y) = dot(x, y)

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

# The diagonal preconditioner a CG solve falls back to when its own is not
# positive; `inv_d === nothing` is none.
struct _DiagPrecond{V}
    inv_d::V
end
_apply_precond!(z, P::_DiagPrecond{Nothing}, r) = copyto!(z, r)
_apply_precond!(z, P::_DiagPrecond, r)          = (@. z = P.inv_d * r; z)

# One device-to-host read of a 1-element array (the only synchronization the
# CG loop performs, once per _CG_CHECK_EVERY iterations).
_cg_scalar(a) = Array(a)[1]

# The fallback and the report of a CG solve whose preconditioner is not
# positive for its residual.  The report gives rᵀz, rᵀr, rᵀKr (K the Krylov
# operator) and, for the AMG V-cycle, rᵀK'r (K' its smoothing operator).  A
# negative rᵀKr is an indefinite tangent; a negative rᵀK'r, an indefinite
# operator of the preconditioner; both positive, a V-cycle that is not
# positive although its operator is.
_cg_fallback(::Preconditioner)          = nothing
_cg_fallback(p::GPUAMGPreconditioner)   = p.inv_diag
_cg_smooth_op(::Preconditioner)         = nothing
_cg_smooth_op(p::GPUAMGPreconditioner)  = p.smooth_op

function _cg_breakdown_report(precond, A_op)
    return function (r, z, rz, rr)
        w = similar(r)
        mul!(w, A_op, r)
        rKr = dot(r, w)
        S = _cg_smooth_op(precond)
        rKpr = S === nothing ? NaN : (S(w, r); dot(r, w))
        _carina_logf(4, :solve,
            "    CG: preconditioner not positive: r'z = %.3e, r'r = %.3e, r'Kr = %.3e, " *
            "r'K'r = %.3e; diagonal preconditioner for this solve", rz, rr, rKr, rKpr)
    end
end

# Stops when the preconditioned residual has fallen by `rtol` relative to its
# initial value, or when the Euclidean norm of the residual is at most `atol`
# (checked at the same synchronization points, one more dot product each).
#
# If the preconditioner is not positive for a residual (rᵀM⁻¹r ≤ 0, at the
# start or at a convergence check), `report(r, z, rᵀz, rᵀr)` is called and
# the solve restarts from its current iterate with the diagonal preconditioner
# `fallback` (a vector of positive inverse diagonal entries, or `nothing` for
# none).
#
# rᵀM⁻¹r ≤ 0 for r ≠ 0 makes CG with M undefined.  Before 2026-10-18 the solve
# returned ΔU = 0 when it held at the start, and reported convergence when it
# held at a check (ρ ≤ tol² holds for ρ < 0); either left Newton at the same
# state until its iteration limit (46-60% of the solves on the punch, case B).
function _device_pcg!(ΔU, A_op, R, M_op, rtol, itmax; atol::Real = 0.0,
                      fallback = nothing, report = nothing)
    ws = _device_cg_workspace(R)
    backend = KA.get_backend(R)
    (; x, r, α, partials) = ws

    fill!(x, zero(eltype(x)))
    copyto!(r, R)
    if atol > 0
        _device_dot!(α, partials, r, r, backend)
        if sqrt(max(_cg_scalar(α), 0.0)) <= atol
            copyto!(ΔU, x)
            return 0, 0.0, true
        end
    end
    iters, ρ_h, status = _pcg_run!(ws, A_op, R, M_op, rtol, itmax, atol, backend, false)
    if status === :breakdown
        _device_dot!(α, partials, r, r, backend)
        report === nothing || report(r, ws.z, ρ_h, _cg_scalar(α))
        it2, ρ_h, status = _pcg_run!(ws, A_op, R, _DiagPrecond(fallback), rtol,
                                     itmax - iters, atol, backend, true)
        iters += it2
    end
    copyto!(ΔU, x)
    return iters, sqrt(max(ρ_h, 0.0)), status === :converged
end

# The CG iteration of `_device_pcg!` from the iterate in `ws.x` (zero, or the
# iterate at a breakdown when `restart`), with the tolerance relative to
# RᵀM⁻¹R.  Returns (iterations, ρ, status), status :converged, :stalled (the
# iteration limit, or a residual that is not finite) or :breakdown (ρ ≤ 0 for
# a residual that is not zero).
function _pcg_run!(ws, A_op, R, M_op, rtol, itmax, atol, backend, restart::Bool)
    (; x, r, z, p, Ap, ρ, ρ_prev, pAp, α, β, partials) = ws
    ρR = 0.0
    if restart
        _apply_precond!(z, M_op, R)
        _device_dot!(ρ, partials, R, z, backend)
        ρR = _cg_scalar(ρ)
        mul!(Ap, A_op, x)
        @. r = R - Ap
    end
    _apply_precond!(z, M_op, r)
    _device_dot!(ρ, partials, r, z, backend)
    ρ0 = _cg_scalar(ρ)
    restart || (ρR = ρ0)
    if !isfinite(ρ0) || ρ0 <= 0.0
        _device_dot!(α, partials, r, r, backend)
        rr = _cg_scalar(α)
        # Zero residual (already solved) or a poisoned one: nothing to iterate.
        rr == 0.0 && return 0, 0.0, :converged
        isfinite(rr) || return 0, ρ0, :stalled
        return 0, ρ0, :breakdown
    end
    copyto!(p, z)
    tol2 = (rtol * sqrt(ρR))^2
    ρ0 <= tol2 && return 0, ρ0, :converged

    iters = 0
    ρ_h   = ρ0
    # Block size adapts to the measured convergence rate.  A fixed block
    # overruns by up to blk − 1 iterations past convergence, which costs little
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
        isfinite(ρ_h) || return iters, ρ_h, :stalled
        ρ_h < 0.0 && return iters, ρ_h, :breakdown
        ρ_h <= tol2 && return iters, ρ_h, :converged
        if atol > 0
            _device_dot!(α, partials, r, r, backend)
            sqrt(max(_cg_scalar(α), 0.0)) <= atol && return iters, ρ_h, :converged
        end
        θ = (ρ_h / ρ_before)^(1 / blk_run)
        if isfinite(θ) && 0.0 < θ < 1.0
            m = log(tol2 / ρ_h) / log(θ)
            blk = clamp(ceil(Int, 0.6 * m), 1, _CG_CHECK_EVERY)
        else
            blk = _CG_CHECK_EVERY
        end
    end
    return iters, ρ_h, :stalled
end

# --------------------------------------------------------------------------- #
# Inexact Newton: per-iteration forcing terms
#
# `_reset_forcing!` is called once per nonlinear solve, `_update_forcing!`
# once per Newton iteration just before the linear solve.  Both are no-ops
# for every linear solver but Krylov, and for `FixedForcing`.
#
# The Eisenstat-Walker tolerance composes three safeguards on top of
# ηₖ = γ (‖Rₖ‖/‖Rₖ₋₁‖)^α:
#
#   * EW's own guard against η falling faster than the observed convergence
#     rate justifies — η ← max(η, γ η_prev^α) whenever that term exceeds 0.1;
#   * Kelley's guard against over-solving the *final* step — η ← max(η,
#     safety·τ/‖Rₖ‖), where τ is the residual norm Newton is aiming at;
#   * a hard clamp to [rtol, eta_max].
#
# The lower clamp at the deck's own `tolerance` is what makes this safe to
# turn on: a forcing term can only ever *loosen* a solve, never tighten one.
# The last Newton iterations therefore run at exactly the tolerance the deck
# asked for, the converged solution is as accurate as it was before, and an
# A/B against `fixed` compares like with like.
#
# One approximation worth naming: `_device_pcg!` stops on the preconditioned
# residual M-norm relative to its own initial value, not the unpreconditioned
# ratio the EW theory is stated for.  The forcing term is a heuristic either
# way; the clamp is what bounds the damage when the two disagree.
# --------------------------------------------------------------------------- #

_reset_forcing!(::AbstractLinearSolver) = nothing
_update_forcing!(::AbstractLinearSolver, ::Float64, ::Float64) = nothing

function _reset_forcing!(ls::KrylovLinearSolver)
    ls.rtol_eff    = ls.rtol
    ls.eta_prev    = NaN
    ls.norm_R_last = NaN
    return nothing
end

# norm_R is the nonlinear residual norm at the current iterate — the one the
# solve about to run will reduce.  tau is the norm Newton is trying to reach.
function _update_forcing!(ls::KrylovLinearSolver, norm_R::Float64, tau::Float64)
    _forcing_tolerance!(ls, ls.forcing, norm_R, tau)
    return nothing
end

_forcing_tolerance!(ls::KrylovLinearSolver, ::FixedForcing, ::Float64, ::Float64) =
    (ls.rtol_eff = ls.rtol; nothing)

function _forcing_tolerance!(ls::KrylovLinearSolver, f::EisenstatWalker,
                             norm_R::Float64, tau::Float64)
    # First iteration of a step, or a residual history we cannot form a ratio
    # from: fall back to the seed value.
    eta = if isfinite(ls.norm_R_last) && ls.norm_R_last > 0.0 && isfinite(norm_R)
        eta_a = f.gamma * (norm_R / ls.norm_R_last)^f.alpha
        guard = f.gamma * ls.eta_prev^f.alpha
        (isfinite(guard) && guard > 0.1) ? max(eta_a, guard) : eta_a
    else
        f.eta_0
    end
    if f.safety > 0.0 && norm_R > 0.0 && isfinite(tau)
        eta = max(eta, f.safety * tau / norm_R)
    end
    isfinite(eta) || (eta = f.eta_0)

    ls.rtol_eff    = clamp(eta, ls.rtol, f.eta_max)
    ls.eta_prev    = ls.rtol_eff
    ls.norm_R_last = norm_R
    ls.rtol_eff > ls.rtol &&
        _carina_logf(8, :solve, "    inexact Newton: η = %.2e", ls.rtol_eff)
    return nothing
end

# Iteration count rescaled to what the same solve would have cost at the deck's
# tolerance.
#
# The AMG staleness detector latches a baseline from the first CG count after a
# build and rebuilds once a later count passes 3x it.  With a forcing term
# running, those counts are taken at different tolerances and are not
# comparable: a loosened solve returns an artificially low number that would
# either poison the baseline or, measured against one, hide a hierarchy that
# has genuinely gone stale.
#
# Simply skipping loosened solves is not an option.  The over-solve guard means
# a run can converge without ever solving at the deck's tolerance, and the
# detector is the ONLY thing that triggers a rebuild on the quasi-static path
# (`_build_precond_op` passes c_M = 0 there, so the c_M test never fires).  A
# quasi-static AMG run would then build once at the reference configuration and
# keep it for the whole load history — the 63 → 400 iteration drift that
# test/amg-staleness.jl exists to prevent.
#
# CG's residual falls geometrically, so iterations go roughly as the digits of
# reduction requested, and dividing by that factor makes counts taken at
# different tolerances comparable.  It is an estimate, but the detector's 3x
# margin and its floor of 30 are far coarser than the estimate's error.  At
# rtol_eff == rtol it is exactly the identity, so a run with no forcing term
# feeds the detector precisely the numbers it always did.
function _tracked_iters(ls::KrylovLinearSolver, niter::Int)
    ls.rtol_eff <= ls.rtol && return niter
    # log10 of either tolerance is only a scale factor inside (0, 1); outside
    # it the rescaling is meaningless, so hand back the raw count.
    (0.0 < ls.rtol < 1.0 && 0.0 < ls.rtol_eff < 1.0) || return niter
    return round(Int, niter * log10(ls.rtol) / log10(ls.rtol_eff))
end
_tracked_iters(::AbstractLinearSolver, niter::Int) = niter

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
# The AMG hierarchy symmetrizes the matrix itself, and only when it is
# rebuilt; symmetrizing and converting it at every Newton iteration took 30 s
# of single-threaded work per solve at 3.6 M unknowns.
_precond_reads_matrix(::AMGPreconditioner)          = false

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
# the build (P === nothing || rebuild) makes the per-Newton calls inexpensive;
# c_M = 0 since there is no mass term in the quasi-static tangent.
function _build_precond_op(precond::AMGPreconditioner, K_sparse, n, x_cur)
    _update_amg_precond_assembled!(precond, K_sparse, 0.0, x_cur)
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
                # The compressible AMG operator: at a hierarchy build, the
                # stiffness is assembled once more at λ' and the run's
                # stiffness restored afterwards; `A_op` holds its own copy.
                if ls.precond isa AMGPreconditioner && ls.precond.compressible &&
                   (ls.precond.P === nothing || ls.precond.rebuild)
                    K_sparse = _with_compressible_properties(p) do
                        FEC.assemble_stiffness!(asm, FEC.stiffness, U, p)
                        copy(FEC.stiffness(asm))   # symmetrized at the hierarchy build
                    end
                    FEC.assemble_stiffness!(asm, FEC.stiffness, U, p)
                end
                M_op_asm = _build_precond_op(ls.precond, K_sparse, n, x_cur)
                if af.is_linear
                    _precond_op_cache[] = M_op_asm
                    af.compute_factorization = false
                end
            else
                M_op_asm = _precond_op_cache[]
            end
            # The absolute test is on the Euclidean norm of the residual,
            # which Krylov.jl's own `atol` (on the preconditioned norm) is not;
            # it runs as a callback on the unpreconditioned residual `r`.
            abs_met = Ref(false)
            cb = ws -> (ls.atol > 0 && (abs_met[] = norm(ws.r) <= ls.atol))
            if ls.atol > 0 && norm(R) <= ls.atol
                fill!(ΔU, zero(eltype(ΔU)))
                r_cg, niter, solved = 0.0, 0, true
            else
                if M_op_asm === nothing
                    Krylov.krylov_solve!(ls.workspace, A_op, R; atol=0.0, rtol=ls.rtol_eff,
                                         itmax=ls.itmax, history=true, callback=cb)
                else
                    Krylov.krylov_solve!(ls.workspace, A_op, R; M=M_op_asm, atol=0.0,
                                         rtol=ls.rtol_eff, itmax=ls.itmax, history=true,
                                         callback=cb)
                end
                copyto!(ΔU, Krylov.solution(ls.workspace))
                res    = ls.workspace.stats.residuals
                r_cg   = isempty(res) ? NaN : res[end]
                niter  = ls.workspace.stats.niter
                solved = ls.workspace.stats.solved || abs_met[]
            end
        else
            _refresh_fine_lmax!(ls.precond)
            niter, r_cg, solved = _device_pcg!(ΔU, K_op, R, M_op, ls.rtol_eff, ls.itmax;
                                               atol = ls.atol,
                                               fallback = _cg_fallback(ls.precond),
                                               report = _cg_breakdown_report(ls.precond, K_op))
        end
    end
    # Feed the iteration count back to the AMG staleness detector so a hierarchy
    # built at an earlier configuration gets rebuilt once it stops paying for
    # itself.  No-op for every other preconditioner.
    _amg_track_iters!(ls.precond, _tracked_iters(ls, niter))
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
                        Pl=F_ic, abstol=ls.atol, reltol=ls.rtol_eff, log=true)
                elseif ls.precond isa AMGPreconditioner
                    ΔU_vec, cg_hist = IterativeSolvers.cg(A_op, R;
                        Pl=ls.precond.P, abstol=ls.atol, reltol=ls.rtol_eff,
                        maxiter=ls.itmax, log=true)
                    _amg_track_iters!(ls.precond,
                        _tracked_iters(ls, length(cg_hist.data[:resnorm])))
                else
                    ΔU_vec, cg_hist = IterativeSolvers.cg(A_op, R;
                        abstol=ls.atol, reltol=ls.rtol_eff, log=true)
                end
                _carina_logf(8, :solve, "    CG: %d iters : |r|_CG = %.2e : %s",
                    length(cg_hist.data[:resnorm]),
                    cg_hist.data[:resnorm][end],
                    _cg_status_str(cg_hist.isconverged))
                copyto!(ΔU, ΔU_vec)
            else
                _refresh_fine_lmax!(ls.precond)
                niter, r_cg, solved = _device_pcg!(ΔU, K_eff_op, R, M_op_mf,
                                                   ls.rtol_eff, ls.itmax; atol = ls.atol,
                                                   fallback = _cg_fallback(ls.precond),
                                                   report = _cg_breakdown_report(ls.precond,
                                                                                 K_eff_op))
                _amg_track_iters!(ls.precond, _tracked_iters(ls, niter))
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
