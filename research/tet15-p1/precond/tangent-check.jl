# Consistency of the tangent with the residual at the converged state of a
# Carina deck, on the CPU: the assembled stiffness K (FEC.stiffness) and the
# matrix-free action (stiffness_action, the operator of the device solves)
# against central differences of the residual,
#
#     K v  ≈  −[R_eff(U + ε v) − R_eff(U − ε v)] / (2 ε),
#
# for random directions v and for the direction of the displacement itself
# (the direction of further loading, along which the plastic tangent is
# small).  A consistent tangent gives an error that decreases with ε until
# rounding; an inconsistent one gives an error independent of ε.
#
#   julia --project research/tet15-p1/precond/tangent-check.jl deck1.yaml [deck2.yaml ...]

using Carina, SparseArrays, LinearAlgebra, Random, Printf
const FEC = Carina.FEC

function check(deck)
    sim = Carina.run(deck)
    p = sim.params; asm = sim.asm_cpu; ig = sim.integrator; dof = asm.dof
    Uu0 = copy(ig.U); n = length(Uu0)
    function Reff(Uu)
        copyto!(ig.U, Uu)
        FEC._update_for_assembly!(p, dof, ig.U)
        Carina.evaluate!(ig, p) || error("The residual is not finite.")
        return copy(Carina.residual(ig))
    end
    FEC._update_for_assembly!(p, dof, Uu0)
    FEC.assemble_stiffness!(asm, FEC.stiffness, Uu0, p)
    K = sparse(FEC.stiffness(asm))
    asym = norm(K - K') / norm(K)
    println("\n== ", basename(deck), ": ", n, " unknowns, |U| = ", @sprintf("%.3e", norm(Uu0)),
            ", |K − Kᵀ|/|K| = ", @sprintf("%.2e", asym))
    println(@sprintf("%-10s %-8s %12s %12s %12s", "direction", "ε/|U|", "|Kv−fd|/|fd|", "|Av−fd|/|fd|", "|Kv−Av|/|Av|"))
    rng = MersenneTwister(1)
    dirs = [("random $i", normalize(randn(rng, n))) for i in 1:2]
    push!(dirs, ("loading", normalize(Uu0)))
    for (name, v) in dirs
        Kv = K * v
        Av = zeros(n)
        FEC._update_for_assembly!(p, dof, Uu0)
        Carina._stiffness_matvec_qs!(Av, v, asm, Uu0, p)
        for εr in (1e-4, 1e-5, 1e-6, 1e-7, 1e-8)
            ε = εr * norm(Uu0)
            fd = -(Reff(Uu0 + ε * v) - Reff(Uu0 - ε * v)) / (2ε)
            println(@sprintf("%-10s %-8.0e %12.3e %12.3e %12.3e", name, εr,
                             norm(Kv - fd) / norm(fd), norm(Av - fd) / norm(fd), norm(Kv - Av) / norm(Av)))
        end
    end
    copyto!(ig.U, Uu0)
    return nothing
end

for deck in ARGS
    check(deck)
end
