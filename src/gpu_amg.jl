# GPU-resident AMG V-cycle over a host-built smoothed-aggregation hierarchy.
#
# Design (benchmark/design.md): the hierarchy is built on the CPU by the
# existing `_update_amg_precond_assembled!` machinery (AlgebraicMultigrid.jl
# smoothed aggregation + rigid-body near-nullspace + staleness-lagged
# rebuilds).  This file converts the resulting hierarchy to device-resident
# CSR and applies a V(ν,ν)-cycle entirely on the GPU:
#
#   level 1 (fine):   smoothing via the MATRIX-FREE K_eff action + diag(K_eff)
#                     — the fine matrix is never formed on the device.
#   levels 2..L:      assembled device CSR with Chebyshev-Jacobi smoothing.
#   coarsest:         dense pinv, applied as a device matvec.
#
# Everything is KernelAbstractions — vendor-agnostic by construction — and
# the apply path performs no allocations (per-level workspaces are
# preallocated at conversion time).

import KernelAbstractions: @kernel, @index

# --------------------------------------------------------------------------- #
# Device CSR
# --------------------------------------------------------------------------- #

struct DeviceCSR{VI <: AbstractVector{Int32}, VF <: AbstractVector{Float64}}
    nrows ::Int
    ncols ::Int
    rowptr::VI
    colval::VI
    nzval ::VF
end

@kernel function _csr_mul_kernel!(y, rowptr, colval,
                                     nzval, x)
    row = @index(Global, Linear)
    acc = 0.0
    @inbounds for k in rowptr[row]:(rowptr[row + 1] - Int32(1))
        acc += nzval[k] * x[colval[k]]
    end
    @inbounds y[row] = acc
end

# y = A * x  (overwrite)
function _csr_mul!(y, A::DeviceCSR, x, backend)
    _csr_mul_kernel!(backend)(y, A.rowptr, A.colval, A.nzval, x;
                              ndrange = A.nrows)
    return y
end

# Convert a host SparseMatrixCSC to device CSR.  CSR(A) = CSC(Aᵀ) with the
# roles of colptr/rowval swapped, so materialize the transpose once on the
# host and upload its arrays.
function _to_device_csr(backend, A::SparseArrays.SparseMatrixCSC)
    At = SparseArrays.sparse(transpose(A))       # CSC of Aᵀ ≡ CSR of A
    rowptr = KA.allocate(backend, Int32, length(At.colptr))
    colval = KA.allocate(backend, Int32, length(At.rowval))
    nzval  = KA.allocate(backend, Float64, length(At.nzval))
    copyto!(rowptr, Int32.(At.colptr))
    copyto!(colval, Int32.(At.rowval))
    copyto!(nzval, At.nzval)
    return DeviceCSR(size(A, 1), size(A, 2), rowptr, colval, nzval)
end

# --------------------------------------------------------------------------- #
# Fused vector kernels
# --------------------------------------------------------------------------- #

@kernel function _xpby_kernel!(y, x, β)
    i = @index(Global, Linear)
    @inbounds y[i] = x[i] + β * y[i]
end

@kernel function _jacobi_omega_kernel!(x, r, inv_d, ω)
    i = @index(Global, Linear)
    @inbounds x[i] += ω * inv_d[i] * r[i]
end

# --------------------------------------------------------------------------- #
# Hierarchy
# --------------------------------------------------------------------------- #

# One assembled level (levels 2..L of the V-cycle).
struct DeviceAMGLevel{CSR, VF}
    A      ::CSR        # level operator
    P      ::CSR        # prolongation:  this level ← next coarser
    R      ::CSR        # restriction:   next coarser ← this level
    inv_d  ::VF         # 1 ./ diag(A)
    lmax   ::Float64    # λ_max estimate of D⁻¹A (Chebyshev-Jacobi bound)
    x      ::VF         # correction workspace (n)
    b      ::VF         # rhs workspace (n)
    r      ::VF         # residual workspace (n)
end

struct DeviceAMGHierarchy{L <: DeviceAMGLevel, VF, MF}
    # Fine level (matrix-free): P₁/R₁ couple the fine grid to levels[1].
    P1      ::DeviceCSR
    R1      ::DeviceCSR
    inv_d1  ::VF        # 1 ./ diag(K_eff) on device (owned by caller)
    lmax1   ::Float64
    r1      ::VF        # fine residual workspace
    z1      ::VF        # fine smoothing workspace
    levels  ::Vector{L}
    coarse_pinv::MF     # dense pinv(A_L) on device
    coarse_x::VF
    coarse_b::VF
    nu      ::Int       # smoothing steps per side
end

# --------------------------------------------------------------------------- #
# Memory-bounded fine-level smoothed-aggregation setup.
#
# AlgebraicMultigrid's hierarchy build computes the fine-level Galerkin
# product with stdlib `spmatmul`, whose up-front nnz-estimate preallocation
# demands tens of GB at ≥1.5M DOF with a 6-column near-nullspace (measured:
# OOM with 17.5 GB free while the true product is ~4 GB).  Build the FIRST
# coarsening ourselves with the same AMG.jl stage functions, but evaluate
# R·(A·P) in column slabs of P so peak transient memory is one slab's
# product; hand the (small) coarse problem back to AMG.smoothed_aggregation
# for the remaining levels.
# --------------------------------------------------------------------------- #

# Threaded over slabs.  The slab width shrinks with the thread count so the
# peak transient memory — the reason this is slabbed at all (the stock
# hierarchy build OOMs above ~3M DOF) — stays at roughly one serial slab's
# product regardless of how many slabs are in flight.
function _slab_galerkin(R, A, P; slab::Int = 8_192)
    nc = size(P, 2)
    w  = max(512, slab ÷ Threads.nthreads())
    ranges = [j0:min(j0 + w - 1, nc) for j0 in 1:w:nc]
    parts = Vector{SparseArrays.SparseMatrixCSC{Float64, Int64}}(undef, length(ranges))
    Threads.@threads for i in eachindex(ranges)
        r = ranges[i]
        parts[i] = R * (A * P[:, r])
    end
    return hcat(parts...)
end

# Drop-in for AMG.fit_candidates(AggOp, B): per-aggregate thin QR of the
# near-nullspace, assembled DIRECTLY into CSC arrays.  The stock version
# inserts entries one at a time into a live SparseMatrixCSC (each insertion
# shifts the nnz arrays) and calls dropzeros! on the whole matrix once per
# aggregate — accidentally quadratic, and 41% of the hierarchy build at 528k
# DOF.  This one is O(nnz), threaded over aggregates (disjoint output
# ranges), and mathematically equivalent: same LAPACK QR per aggregate, so
# the same T and Bc up to entries the stock version drops below its 1e-10
# sparsification tolerance.
function _fit_candidates(AggOp, B::AbstractMatrix{Float64})
    At = SparseArrays.sparse(copy(transpose(SparseArrays.sparse(AggOp))))
    size(At, 2) == size(B, 1) && (At = SparseArrays.sparse(copy(transpose(At))))
    n_fine, n_agg = size(At)
    n_fine == size(B, 1) || error(
        "aggregation operator ($(size(At))) does not match nullspace rows ($(size(B, 1)))")
    m = size(B, 2)

    # Column pointer of T: column (agg-1)m + j holds the aggregate's rows for
    # j ≤ r = min(|agg|, m), and is empty past r (rank-deficient aggregates).
    colptr = Vector{Int}(undef, m * n_agg + 1)
    colptr[1] = 1
    for agg in 1:n_agg
        la = At.colptr[agg + 1] - At.colptr[agg]
        r  = min(la, m)
        off = (agg - 1) * m
        for j in 1:m
            colptr[off + j + 1] = colptr[off + j] + (j <= r ? la : 0)
        end
    end
    nnzT   = colptr[end] - 1
    rowval = Vector{Int}(undef, nnzT)
    nzval  = Vector{Float64}(undef, nnzT)
    Bc     = zeros(m * n_agg, m)

    Threads.@threads for agg in 1:n_agg
        rng = At.colptr[agg]:(At.colptr[agg + 1] - 1)
        la  = length(rng)
        la == 0 && continue
        rows = view(At.rowval, rng)          # sorted within a CSC column
        r = min(la, m)
        F = LinearAlgebra.qr(B[rows, :])
        Q = Matrix(F.Q)                      # thin: la × min(la, m)
        off = (agg - 1) * m
        for j in 1:r
            p0 = colptr[off + j] - 1
            for i in 1:la
                rowval[p0 + i] = rows[i]
                nzval[p0 + i]  = Q[i, j]
            end
        end
        Rf = F.R
        for j in 1:m, i in 1:min(r, size(Rf, 1))
            Bc[off + i, j] = Rf[i, j]
        end
    end

    T = SparseArrays.SparseMatrixCSC(n_fine, m * n_agg, colptr, rowval, nzval)
    # Match the stock version's sparsification: near-zero Q entries (rotation
    # residue from the QR) double nnz(T) and inflate every downstream product.
    SparseArrays.droptol!(T, 1e-10)
    return T, Bc
end

function _sa_hierarchy_lowmem(A::SparseArrays.SparseMatrixCSC, B::Matrix{Float64})
    S, _  = AMG.SymmetricStrength()(A, false)
    AggOp = AMG.StandardAggregation()(S)
    T, Bc = _fit_candidates(AggOp, B)
    P     = AMG.JacobiProlongation(4.0 / 3.0)(A, T, S, Bc)
    R     = SparseArrays.sparse(transpose(P))
    Ac    = _slab_galerkin(R, A, P)
    # Remaining levels on the (small) coarse problem via stock AMG.jl.
    ml_c = AMG.smoothed_aggregation(Ac; B = Bc)
    fine = AMG.Level(A, P, R)
    return (; levels = [fine; ml_c.levels], final_A = ml_c.final_A)
end

# λ_max(D⁻¹A) via a few power iterations on the host (setup-time only).
function _host_lambda_max(A::SparseArrays.SparseMatrixCSC, dinv::Vector{Float64})
    n = size(A, 1)
    v = ones(n) ./ sqrt(n)
    λ = 1.0
    for _ in 1:10
        v = dinv .* (A * v)
        λ = LinearAlgebra.norm(v)
        λ == 0.0 && return 1.0
        v ./= λ
    end
    return 1.1 * λ    # safety boost, same convention as the fine estimator
end

"""
Convert the CPU hierarchy inside an `AlgebraicMultigrid.MultiLevel` into a
fully device-resident V-cycle structure.  `inv_d1` is the device vector
holding 1/diag(K_eff) for the matrix-free fine level (already maintained by
the Jacobi-preconditioner machinery); `lmax1` its eigenvalue bound.
"""
function DeviceAMGHierarchy(backend, ml, inv_d1, lmax1::Float64; nu::Int = 2)
    isempty(ml.levels) && error(
        "AMG hierarchy has no levels — mesh too small for GPU AMG; use jacobi.")

    # Level 1 in ml couples the fine grid to the first coarse grid.
    P1 = _to_device_csr(backend, SparseArrays.sparse(ml.levels[1].P))
    R1 = _to_device_csr(backend, SparseArrays.sparse(ml.levels[1].R))
    nfine = P1.nrows
    r1 = KA.allocate(backend, Float64, nfine); fill!(r1, 0.0)
    z1 = KA.allocate(backend, Float64, nfine); fill!(z1, 0.0)

    levels = DeviceAMGLevel[]
    for l in 2:length(ml.levels)
        A  = SparseArrays.sparse(ml.levels[l].A)
        dv = Vector(LinearAlgebra.diag(A))
        any(iszero, dv) && error("Zero diagonal in AMG level $l operator.")
        dinv = 1.0 ./ dv
        n  = size(A, 1)
        x  = KA.allocate(backend, Float64, n); fill!(x, 0.0)
        b  = KA.allocate(backend, Float64, n); fill!(b, 0.0)
        r  = KA.allocate(backend, Float64, n); fill!(r, 0.0)
        id = KA.allocate(backend, Float64, n); copyto!(id, dinv)
        push!(levels, DeviceAMGLevel(
            _to_device_csr(backend, A),
            _to_device_csr(backend, SparseArrays.sparse(ml.levels[l].P)),
            _to_device_csr(backend, SparseArrays.sparse(ml.levels[l].R)),
            id, _host_lambda_max(A, dinv), x, b, r))
    end

    A_L = SparseArrays.sparse(ml.final_A)
    pinv_h = LinearAlgebra.pinv(Matrix(A_L))
    nc = size(A_L, 1)
    coarse_pinv = KA.allocate(backend, Float64, nc, nc)
    copyto!(coarse_pinv, pinv_h)
    coarse_x = KA.allocate(backend, Float64, nc); fill!(coarse_x, 0.0)
    coarse_b = KA.allocate(backend, Float64, nc); fill!(coarse_b, 0.0)

    # NB: `[levels...]` on an empty typed vector yields Vector{Any} (zero-arg
    # vect), which broke single-coarsening hierarchies (small meshes).  Pass
    # the typed vector itself.
    return DeviceAMGHierarchy(P1, R1, inv_d1, lmax1, r1, z1,
                              levels, coarse_pinv, coarse_x, coarse_b, nu)
end

# --------------------------------------------------------------------------- #
# Smoothing: damped Jacobi, ν sweeps.
#   x ← x + ω D⁻¹ (b − A x),  ω = 4/(3·λ_max)  (optimal-ish for SPD Jacobi)
# `matvec!(y, x)` computes y = A·x for the level operator (matrix-free on the
# fine level, CSR elsewhere).
# --------------------------------------------------------------------------- #

function _smooth!(x, b, r, matvec!, inv_d, lmax, nu, backend, n)
    ω = 4.0 / (3.0 * lmax)
    for _ in 1:nu
        matvec!(r, x)                      # r = A x
        @. r = b - r                       # r = b − A x   (device broadcast)
        _jacobi_omega_kernel!(backend)(x, r, inv_d, ω; ndrange = n)
    end
    return x
end

@kernel function _dense_mul_kernel!(y, A, x, n)
    row = @index(Global, Linear)
    acc = 0.0
    @inbounds for j in 1:n
        acc += A[row, j] * x[j]
    end
    @inbounds y[row] = acc
end

# --------------------------------------------------------------------------- #
# V-cycle
# --------------------------------------------------------------------------- #

"""
    _amg_vcycle!(z, r, h, fine_matvec!, backend)

Apply one V(ν,ν)-cycle of the device hierarchy to the fine-grid residual `r`,
writing the correction into `z`.  `fine_matvec!(y, x)` is the matrix-free
K_eff action.  Allocation-free.
"""
function _amg_vcycle!(z, r, h::DeviceAMGHierarchy, fine_matvec!, backend)
    nfine = length(z)

    # Pre-smooth on the fine level from zero initial guess.
    fill!(z, 0.0)
    _smooth!(z, r, h.r1, fine_matvec!, h.inv_d1, h.lmax1, h.nu, backend, nfine)

    # Fine residual → restrict to level 1 of the assembled hierarchy.
    fine_matvec!(h.z1, z)
    @. h.z1 = r - h.z1
    nlev = length(h.levels)
    b2 = nlev >= 1 ? h.levels[1].b : h.coarse_b
    _csr_mul!(b2, h.R1, h.z1, backend)

    # Descend through assembled levels.
    for l in 1:nlev
        lev = h.levels[l]
        fill!(lev.x, 0.0)
        mv! = (y, x) -> _csr_mul!(y, lev.A, x, backend)
        _smooth!(lev.x, lev.b, lev.r, mv!, lev.inv_d, lev.lmax, h.nu,
                 backend, lev.A.nrows)
        mv!(lev.r, lev.x)
        @. lev.r = lev.b - lev.r
        bnext = l < nlev ? h.levels[l + 1].b : h.coarse_b
        _csr_mul!(bnext, lev.R, lev.r, backend)
    end

    # Coarsest: dense pinv matvec.
    nc = length(h.coarse_x)
    _dense_mul_kernel!(backend)(h.coarse_x, h.coarse_pinv, h.coarse_b, nc;
                                ndrange = nc)

    # Ascend: prolong + post-smooth.
    for l in nlev:-1:1
        lev = h.levels[l]
        xc = l < nlev ? h.levels[l + 1].x : h.coarse_x
        _csr_mul!(lev.r, lev.P, xc, backend)      # r reused as P·x_coarse
        @. lev.x += lev.r
        mv! = (y, x) -> _csr_mul!(y, lev.A, x, backend)
        _smooth!(lev.x, lev.b, lev.r, mv!, lev.inv_d, lev.lmax, h.nu,
                 backend, lev.A.nrows)
    end

    # Prolong level-1 correction to the fine grid and post-smooth there.
    x1 = nlev >= 1 ? h.levels[1].x : h.coarse_x
    _csr_mul!(h.z1, h.P1, x1, backend)
    @. z += h.z1
    _smooth!(z, r, h.r1, fine_matvec!, h.inv_d1, h.lmax1, h.nu, backend, nfine)

    return z
end
