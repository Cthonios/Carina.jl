# Stage 2 of the preconditioner study: iteration counts of Krylov methods on
# the TET15-P1 tangent of state.jl, for the present preconditioner and for
# the alternatives, and their dependence on the bulk modulus.
#
#   julia --project research/tet15-p1/precond/experiments.jl state.jls results.tsv
#
# Operators (all on the free displacement unknowns, n_u, and the projection
# unknowns, n_p = 4 per element):
#   K(κ)  = K_u + κ G,  G = D M⁻¹ Dᵀ,  K_u = K(κ') − κ' G  (κ' = μ)
#   saddle point  A(κ) = [K_u  D; Dᵀ  −M/κ],  equivalent to K(κ) after the
#   elimination of the projection unknowns.
# Methods:
#   cg-amg(K)     CG on K(κ), one V-cycle of smoothed aggregation built on
#                 K(κ) with the rigid-body near-null space and two damped
#                 Jacobi sweeps per side: the device preconditioner of Carina.
#   cg-amg(K')    the same V-cycle built on K(κ') applied to K(κ).
#   minres-block  MINRES on A(κ) with the block-diagonal preconditioner
#                 diag(V-cycle on K(κ'), ((1/κ + 1/κ') M)⁻¹): the augmented
#                 Lagrangian preconditioner (Benzi and Olshanskii 2006) with
#                 the augmentation parameter κ'.
# Each method stops at a relative residual of 1e-8 of its own system; the
# table reports the iterations, the time, and the true relative residual of
# the displacement system, ‖K x − b‖/‖b‖.

using Carina, SparseArrays, LinearAlgebra, Serialization, Printf
import AlgebraicMultigrid as AMG
import Krylov
using LinearOperators

state, out = ARGS[1], ARGS[2]
S = deserialize(state)
(; K, Kp, D, M, Minv, b, udofs, X, U, κ, κp, μ, nu, np) = S

G    = D * Minv * D'
G    = sparse((G + G') / 2)
Ku   = sparse(Kp - κp * G)
xcur = X .+ U
B    = Carina._rigid_body_modes(xcur, udofs)

# ---- the V-cycle of Carina's device preconditioner, on the host ----------- #
struct VCycle
    levels::Vector{Any}      # (A, P, R, dinv, ω)
    coarse                   # factorization of the coarsest matrix
    nu::Int
end
function lambda_max(A, dinv)
    v = ones(size(A, 1)) ./ sqrt(size(A, 1)); λ = 1.0
    for _ in 1:10
        v = dinv .* (A * v); λ = norm(v); v ./= λ
    end
    return 1.1 * λ
end
function VCycle(A::SparseMatrixCSC, B::Matrix{Float64}; nu::Int = 2)
    ml = AMG.smoothed_aggregation(A; B = B)
    levels = Any[]
    for lev in ml.levels
        dinv = 1.0 ./ max.(abs.(diag(lev.A)), eps())
        ω = 4.0 / (3.0 * lambda_max(lev.A, dinv))
        push!(levels, (lev.A, lev.P, lev.R, dinv, ω))
    end
    return VCycle(levels, lu(Matrix(ml.final_A) + 0 * I), nu)
end
function smooth!(x, b, A, dinv, ω, nu)
    for _ in 1:nu
        x .+= ω .* dinv .* (b .- A * x)
    end
    return x
end
function vcycle!(z, r, V::VCycle, l::Int = 1)
    if l > length(V.levels)
        z .= V.coarse \ r
        return z
    end
    A, P, R, dinv, ω = V.levels[l]
    fill!(z, 0.0)
    smooth!(z, r, A, dinv, ω, V.nu)
    rc = R * (r .- A * z)
    zc = zeros(length(rc))
    vcycle!(zc, rc, V, l + 1)
    z .+= P * zc
    smooth!(z, r, A, dinv, ω, V.nu)
    return z
end
precond_op(V::VCycle, n) = LinearOperator(Float64, n, n, true, true, (y, v) -> vcycle!(y, v, V))

# ---- the methods ---------------------------------------------------------- #
results = NamedTuple[]
function record!(method, κs, niter, t, resid)
    push!(results, (; method, κ_over_μ = κs / μ, niter, t, resid))
    @printf("%-14s κ/μ = %8.1e  iterations %6d  time %8.2f s  true residual %.2e\n",
            method, κs / μ, niter, t, resid)
end

function run_cg(Kκ, b, V, label, κs)
    n = length(b)
    t = @elapsed (x, st) = Krylov.cg(Kκ, b; M = precond_op(V, n), atol = 0.0, rtol = 1e-8,
                                     itmax = 20000, history = true)
    record!(label, κs, st.niter, t, norm(Kκ * x - b) / norm(b))
end

function run_minres_block(Ku, D, M, Minv, b, Vp, κs, κp)
    n = length(b); m = size(M, 1)
    A  = [Ku D; D' (-(1 / κs)) * M]
    bb = [b; zeros(m)]
    sp = κs * κp / (κs + κp)                 # ((1/κ + 1/κ') M)⁻¹ = sp · M⁻¹
    P  = LinearOperator(Float64, n + m, n + m, true, true, (y, v) -> begin
        vcycle!(view(y, 1:n), v[1:n], Vp)
        y[n+1:end] .= sp .* (Minv * v[n+1:end])
        y
    end)
    t = @elapsed (x, st) = Krylov.minres(A, bb; M = P, atol = 0.0, rtol = 1e-8,
                                         itmax = 20000, history = true)
    xu = x[1:n]
    Kκ = Ku + κs * G
    record!("minres-block", κs, st.niter, t, norm(Kκ * xu - b) / norm(b))
end

# The same saddle point with the projection unknown scaled to force units,
# q = p/κ and the second block row multiplied by κ:
#     [K_u  κD; κDᵀ  −κM] [u; q] = [b; 0],
# so that the residual MINRES measures is the force residual of both blocks;
# in the unscaled form the second block carries M/κ and its residual, small in
# the stopping test, multiplies κ in the displacement residual.  The Schur
# approximation scales to κ (1 + κ/κ') M.
function run_minres_scaled(Ku, D, M, Minv, b, Vp, κs, κp)
    n = length(b); m = size(M, 1)
    A  = [Ku (κs * D); (κs * D') (-κs) * M]
    bb = [b; zeros(m)]
    sp = 1.0 / (κs * (1.0 + κs / κp))
    P  = LinearOperator(Float64, n + m, n + m, true, true, (y, v) -> begin
        vcycle!(view(y, 1:n), v[1:n], Vp)
        y[n+1:end] .= sp .* (Minv * v[n+1:end])
        y
    end)
    # MINRES tests the residual in the norm of the preconditioner, which weighs
    # the second block by about κ'/κ²; the test that matters is the force
    # residual of the displacement system, checked every 10 iterations.
    Kκ = Ku + κs * G
    nb = norm(b)
    count = Ref(0)
    stop = ws -> begin
        count[] += 1
        count[] % 10 == 0 || return false
        xu = ws.x[1:n]
        return norm(Kκ * xu - b) / nb < 1e-8
    end
    t = @elapsed (x, st) = Krylov.minres(A, bb; M = P, atol = 0.0, rtol = 1e-14,
                                         itmax = 20000, history = true, callback = stop)
    xu = x[1:n]
    record!("minres-scaled", κs, st.niter, t, norm(Kκ * xu - b) / nb)
end

# GMRES with the block preconditioner applied on the right: its residual is
# the Euclidean residual of the force-scaled system, so the stopping test is
# on the force residual of both blocks.
function run_gmres_block(Ku, D, M, Minv, b, Vp, κs, κp)
    n = length(b); m = size(M, 1)
    A  = [Ku (κs * D); (κs * D') (-κs) * M]
    bb = [b; zeros(m)]
    sp = 1.0 / (κs * (1.0 + κs / κp))
    P  = LinearOperator(Float64, n + m, n + m, true, true, (y, v) -> begin
        vcycle!(view(y, 1:n), v[1:n], Vp)
        y[n+1:end] .= sp .* (Minv * v[n+1:end])
        y
    end)
    # The displacement residual K x_u − b equals D M⁻¹ r_q − r_u, so a residual
    # r_q of the second block is amplified by |D M⁻¹| ~ 1/h; the force-scaled
    # system is therefore solved to 1e-12 for a displacement residual of 1e-8.
    t = @elapsed (x, st) = Krylov.gmres(A, bb; N = P, atol = 0.0, rtol = 1e-12,
                                        itmax = 2000, memory = 500, history = true)
    xu = x[1:n]
    Kκ = Ku + κs * G
    record!("gmres-block", κs, st.niter, t, norm(Kκ * xu - b) / norm(b))
end

only_minres = length(ARGS) >= 3 && ARGS[3] == "minres-only"

@info "V-cycle on K(κ')"
Vp = VCycle(Kp, B)

for κs in (1e2 * μ, 1e3 * μ, κ, 1e5 * μ)
    Kκ = sparse(Ku + κs * G)
    if !only_minres
        @info "κ/μ = $(κs / μ): V-cycle on K(κ)"
        Vκ = VCycle(Kκ, B)
        run_cg(Kκ, b, Vκ, "cg-amg(K)", κs)
        run_cg(Kκ, b, Vp, "cg-amg(K')", κs)
        run_minres_block(Ku, D, M, Minv, b, Vp, κs, κp)
    end
    run_minres_scaled(Ku, D, M, Minv, b, Vp, κs, κp)
    run_gmres_block(Ku, D, M, Minv, b, Vp, κs, κp)
end

open(out, "w") do io
    println(io, "method\tkappa_over_mu\titerations\ttime_s\ttrue_residual")
    for r in results
        @printf(io, "%s\t%.3e\t%d\t%.3f\t%.3e\n", r.method, r.κ_over_μ, r.niter, r.t, r.resid)
    end
end
@info "Written $out"
