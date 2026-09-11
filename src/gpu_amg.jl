# GPU-resident AMG V-cycle over a host-built smoothed-aggregation hierarchy.
#
# Design (benchmark/design.md): the hierarchy is built on the CPU by the
# existing `_update_amg_precond_assembled!` machinery (AlgebraicMultigrid.jl
# smoothed aggregation + rigid-body near-nullspace + staleness-lagged
# rebuilds).  This file converts the resulting hierarchy to device-resident
# CSR and applies a V(ν,ν)-cycle entirely on the GPU:
#
#   level 1 (fine):   smoothing via the MATRIX-FREE K_eff action + diag(K_eff),
#                     or, with `smoother: block jacobi`, + the inverted 3x3
#                     nodal blocks of K_eff — the fine matrix is never formed
#                     on the device either way.
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
# Block Jacobi: one 3x3 tangent block per node, pre-inverted.
#
# This is the smoother-scale form of the idea that makes a sparse direct solve
# viable on a GPU (Tacho's "inverted diagonals"): pay to invert the diagonal
# blocks once, so that applying the preconditioner is a GEMV with no triangular
# solve and no data dependence between blocks.  Here the blocks are 3x3, the
# inversion is closed-form, and the apply is nine multiply-adds per node.
#
# Constrained components make the blocks ragged.  The fine-level vectors are
# indexed by FREE degree of freedom; a node with a Dirichlet component has a
# 2x2 or 1x1 block over its free components.  `blk_dof[i, n]` is the free
# index of component i of node n, or a non-positive sentinel when constrained
# (DofManager uses -1 for Dirichlet and -2 for the periodic slave side).  Both
# kernels skip those slots.  The inversion pads a constrained slot with the
# identity so the same 3x3 storage serves every node; the padded rows and
# columns are never read by the apply, because it skips them too.
# --------------------------------------------------------------------------- #

# Gather node n's free sub-block from the three assembled column vectors,
# B_k[dof(n,i)] = block_n[i,k], invert it, store the 3x3 with identity padding.
@kernel function _block_invert_kernel!(blk_inv, @Const(blk_dof), @Const(B1), @Const(B2), @Const(B3))
    n = @index(Global, Linear)
    @inbounds begin
        d1 = blk_dof[1, n]; d2 = blk_dof[2, n]; d3 = blk_dof[3, n]
        f1 = d1 > 0; f2 = d2 > 0; f3 = d3 > 0
        # Padded matrix M: free rows/cols hold the block, constrained diagonal
        # slots hold 1, everything else 0.  Column k is B_k over free rows.
        a11 = f1 ? B1[d1] : 1.0;  a12 = (f1 && f2) ? B2[d1] : 0.0;  a13 = (f1 && f3) ? B3[d1] : 0.0
        a21 = (f2 && f1) ? B1[d2] : 0.0;  a22 = f2 ? B2[d2] : 1.0;  a23 = (f2 && f3) ? B3[d2] : 0.0
        a31 = (f3 && f1) ? B1[d3] : 0.0;  a32 = (f3 && f2) ? B2[d3] : 0.0;  a33 = f3 ? B3[d3] : 1.0
        # Cofactor inverse.
        c11 = a22 * a33 - a23 * a32
        c12 = a13 * a32 - a12 * a33
        c13 = a12 * a23 - a13 * a22
        c21 = a23 * a31 - a21 * a33
        c22 = a11 * a33 - a13 * a31
        c23 = a13 * a21 - a11 * a23
        c31 = a21 * a32 - a22 * a31
        c32 = a12 * a31 - a11 * a32
        c33 = a11 * a22 - a12 * a21
        det = a11 * c11 + a12 * c21 + a13 * c31
        # A singular block means a node with no stiffness in some free
        # direction; the scalar smoother guards its diagonal with eps and this
        # does the same for the determinant rather than emitting Inf/NaN.
        idet = 1.0 / (abs(det) < eps(Float64) ? copysign(eps(Float64), det) : det)
        blk_inv[1, 1, n] = c11 * idet; blk_inv[1, 2, n] = c12 * idet; blk_inv[1, 3, n] = c13 * idet
        blk_inv[2, 1, n] = c21 * idet; blk_inv[2, 2, n] = c22 * idet; blk_inv[2, 3, n] = c23 * idet
        blk_inv[3, 1, n] = c31 * idet; blk_inv[3, 2, n] = c32 * idet; blk_inv[3, 3, n] = c33 * idet
    end
end

# x[free(n,:)] += ω · blk_inv[n] · r[free(n,:)], skipping constrained slots.
# Each free DOF belongs to exactly one node, so no atomics are needed.
@kernel function _block_jacobi_kernel!(x, @Const(r), @Const(blk_inv), @Const(blk_dof), ω)
    n = @index(Global, Linear)
    @inbounds begin
        d1 = blk_dof[1, n]; d2 = blk_dof[2, n]; d3 = blk_dof[3, n]
        r1 = d1 > 0 ? r[d1] : 0.0
        r2 = d2 > 0 ? r[d2] : 0.0
        r3 = d3 > 0 ? r[d3] : 0.0
        if d1 > 0
            x[d1] += ω * (blk_inv[1, 1, n] * r1 + blk_inv[1, 2, n] * r2 + blk_inv[1, 3, n] * r3)
        end
        if d2 > 0
            x[d2] += ω * (blk_inv[2, 1, n] * r1 + blk_inv[2, 2, n] * r2 + blk_inv[2, 3, n] * r3)
        end
        if d3 > 0
            x[d3] += ω * (blk_inv[3, 1, n] * r1 + blk_inv[3, 2, n] * r2 + blk_inv[3, 3, n] * r3)
        end
    end
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
    blk_inv1::Any       # 3x3xnnodes inverted nodal blocks (owned by caller), or nothing
    blk_dof1::Any       # 3xnnodes free-index map (owned by caller), or nothing
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

# Host-side block structure and block-preconditioned λ_max, both from the
# assembled fine operator at hierarchy-build time.  `udofs` lists the global
# DOF of each free index; global DOF 3(n-1)+i is component i of node n.
#
# Returns (blk_dof, blk_inv_host, lmax): the 3 x nnodes free-index map, the
# 3 x 3 x nnodes inverted blocks at the build point (a first value for the
# device array, refreshed every Newton iteration), and the power-method bound
# on ρ(D_blk⁻¹ A), which the scalar bound does not transfer to.
function _host_block_structure(A::SparseArrays.SparseMatrixCSC, udofs::Vector{Int})
    nfree  = length(udofs)
    nnodes = maximum(udofs) == 0 ? 0 : (maximum(udofs) - 1) ÷ 3 + 1
    blk_dof = zeros(Int32, 3, nnodes)
    for (k, g) in enumerate(udofs)
        n = (g - 1) ÷ 3 + 1
        i = g - 3 * (n - 1)
        blk_dof[i, n] = k
    end
    blk_inv = zeros(Float64, 3, 3, nnodes)
    for n in 1:nnodes
        M = Matrix{Float64}(LinearAlgebra.I, 3, 3)
        for i in 1:3, k in 1:3
            di = blk_dof[i, n]; dk = blk_dof[k, n]
            (di > 0 && dk > 0) && (M[i, k] = A[di, dk])
        end
        blk_inv[:, :, n] = inv(M)
    end
    # Power method on D_blk⁻¹ A.
    v = ones(nfree) ./ sqrt(nfree)
    w = similar(v)
    λ = 1.0
    for _ in 1:10
        Av = A * v
        fill!(w, 0.0)
        for n in 1:nnodes, i in 1:3
            di = blk_dof[i, n]; di > 0 || continue
            acc = 0.0
            for k in 1:3
                dk = blk_dof[k, n]; dk > 0 || continue
                acc += blk_inv[i, k, n] * Av[dk]
            end
            w[di] = acc
        end
        λ = LinearAlgebra.norm(w)
        λ == 0.0 && return blk_dof, blk_inv, 1.0
        v = w ./ λ
    end
    return blk_dof, blk_inv, 1.1 * λ
end

"""
Convert the CPU hierarchy inside an `AlgebraicMultigrid.MultiLevel` into a
fully device-resident V-cycle structure.  `inv_d1` is the device vector
holding 1/diag(K_eff) for the matrix-free fine level (already maintained by
the Jacobi-preconditioner machinery); `lmax1` its eigenvalue bound.
"""
function DeviceAMGHierarchy(backend, ml, inv_d1, lmax1::Float64; nu::Int = 2,
                            blk_inv1 = nothing, blk_dof1 = nothing)
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
    return DeviceAMGHierarchy(P1, R1, inv_d1, blk_inv1, blk_dof1, lmax1, r1, z1,
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

# Block-Jacobi smoothing on the fine level: same sweep, block apply.
function _smooth_block!(x, b, r, matvec!, blk_inv, blk_dof, lmax, nu, backend)
    ω = 4.0 / (3.0 * lmax)
    nblk = size(blk_dof, 2)
    for _ in 1:nu
        matvec!(r, x)
        @. r = b - r
        _block_jacobi_kernel!(backend)(x, r, blk_inv, blk_dof, ω; ndrange = nblk)
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
    if h.blk_inv1 === nothing
        _smooth!(z, r, h.r1, fine_matvec!, h.inv_d1, h.lmax1, h.nu, backend, nfine)
    else
        _smooth_block!(z, r, h.r1, fine_matvec!, h.blk_inv1, h.blk_dof1, h.lmax1, h.nu, backend)
    end

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
    if h.blk_inv1 === nothing
        _smooth!(z, r, h.r1, fine_matvec!, h.inv_d1, h.lmax1, h.nu, backend, nfine)
    else
        _smooth_block!(z, r, h.r1, fine_matvec!, h.blk_inv1, h.blk_dof1, h.lmax1, h.nu, backend)
    end

    return z
end
