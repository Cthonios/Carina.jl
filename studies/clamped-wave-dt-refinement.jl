# Δt-refinement convergence study for the clamped-bar Gaussian-pulse test
# (Mota-Tezaur-Phlipot 2022 IJNME §3.3).
#
# Mesh is fixed at h = 1 mm (1×1 mm² cross-section, 1000 elements along z).
# For each Δt in the sweep we run Carina to t_final = 1e-5 s and measure the
# discrete relative L2 error in u_z(z, t_final) against the analytical
# solution.
#
# Both Newmark γ=0.5, β=0.25 (consistent mass) and central difference
# (lumped mass) are 2nd-order in time, so the error should look like
#     e(Δt) = e_space + C·Δt² + o(Δt²)
# i.e. quadratic decay until it saturates at the spatial-dispersion floor
# e_space.
#
# Findings (2026-06-07, h = 1 mm):
#   Explicit (lumped mass) — monotone approach to floor:
#     Δt      rel L2 err
#     2.0e-7  5.625e-5
#     1.0e-7  5.801e-5
#     5.0e-8  5.845e-5
#     2.5e-8  5.856e-5     ← spatial floor ≈ 5.86e-5
#
#   Implicit (consistent mass) — non-monotone, sign-cancelling temporal
#   error sweeps through a minimum and settles slightly below the explicit
#   floor:
#     Δt      rel L2 err
#     2.0e-6  4.076e-4
#     1.0e-6  5.856e-5
#     5.0e-7  2.929e-5     ← accidental "sweet spot" (temporal ⊥ spatial)
#     2.5e-7  5.128e-5
#     1.25e-7 5.679e-5     ← spatial floor ≈ 5.7e-5
#
# Interpretation: both schemes' spatial dispersion floors agree to within
# ~3 %.  Consistent mass is slightly more accurate than lumped mass, as
# predicted by classical dispersion analysis (φ_consistent ≈ +k²h²/12,
# φ_lumped ≈ -k²h²/6).  The previously-suspected "17× implicit/explicit
# gap" is NOT present in the L2-displacement metric at this mesh — both
# integrators are mathematically consistent and agree on the same physical
# solution to 6e-5 relative.  The checked-in test rtol = 5e-3 has ~100×
# headroom over the actual error at fixed mesh.
#
# Usage (from the Carina.jl root):
#     julia --project -e 'include("studies/clamped-wave-dt-refinement.jl"); main()'

using Carina
using TOML
using Printf

# -------------- analytical solution (eqs. from mechanics-clamped-wave.jl) -----
const _a = 0.01
const _s = 0.02
const _c = 1000.0
const _L = 1.0
const _T = _L / _c    # 1e-3 s

@inline _f(ξ)    = (_a / 2) * exp(-ξ^2 / (2 * _s^2))
@inline function _uz(z, t)
    c = _c; T = _T
    return _f(z - c*t) + _f(z + c*t) -
           _f(z - c*(T - t)) - _f(z + c*(T - t))
end

# -------------- discrete L2 error over the FEM mesh --------------------------
# For the clamped bar u_x = u_y = 0 by symmetry; we measure only the z-component.
# The mesh is uniform, so the nodal-L2 norm is the volume-L2 norm up to a
# constant (h^3) that drops out of the relative error.
function _l2_error(sim)
    coords = reshape(Vector(sim.params_cpu.coords.data), 3, :)
    U_full = Vector(sim.params.field.data)
    t_f    = sim.controller.time
    n      = size(coords, 2)
    err2 = 0.0
    ref2 = 0.0
    @inbounds for i in 1:n
        z   = coords[3, i]
        u_h = U_full[3 * i]
        u_e = _uz(z, t_f)
        err2 += (u_h - u_e)^2
        ref2 += u_e^2
    end
    return sqrt(err2 / ref2)
end

# -------------- run one Carina simulation with a per-Δt toml -----------------
function _run_at_dt(template_dir::String, template_toml::String, dt::Float64)
    dict = TOML.parsefile(joinpath(template_dir, template_toml))
    dict["time_integrator"]["time_step"] = dt
    # Coarsen the output cadence so we don't write 80+ frames for the
    # finest Δt.  We only need the final state.
    dict["output_interval"] = dict["time_integrator"]["final_time"]
    mktempdir() do dir
        # Example mesh files are symlinks into examples/meshes/; resolve.
        src_mesh = realpath(joinpath(template_dir, dict["input_mesh_file"]))
        cp(src_mesh, joinpath(dir, dict["input_mesh_file"]))
        toml_path = joinpath(dir, "case.toml")
        open(toml_path, "w") do io
            TOML.print(io, dict)
        end
        sim = Carina.run(toml_path)
        return _l2_error(sim)
    end
end

# -------------- print a tidy table -------------------------------------------
function _print_table(label::String, dts::Vector{Float64}, errs::Vector{Float64})
    println()
    println("="^64)
    println(label)
    println("="^64)
    @printf("  %-12s  %-14s  %-12s  %-12s\n",
            "Δt [s]", "rel L2 err", "ratio", "order")
    for (i, dt) in enumerate(dts)
        e = errs[i]
        if i == 1
            @printf("  %-12.3e  %-14.6e  %-12s  %-12s\n", dt, e, "—", "—")
        else
            ratio = errs[i - 1] / e
            order = log2(ratio) / log2(dts[i - 1] / dt)
            @printf("  %-12.3e  %-14.6e  %-12.4f  %-12.4f\n",
                    dt, e, ratio, order)
        end
    end
    println("="^64)
end

# -------------- main study ---------------------------------------------------
function main()
    repo_root = abspath(joinpath(@__DIR__, ".."))

    # ------- implicit Newmark sweep -------
    impl_dir  = joinpath(repo_root, "examples", "mechanics",
                         "implicit-dynamic", "clamped")
    impl_dts  = [2.0e-6, 1.0e-6, 5.0e-7, 2.5e-7, 1.25e-7]
    impl_errs = Float64[]
    for dt in impl_dts
        @info "implicit  dt = $dt"
        push!(impl_errs, _run_at_dt(impl_dir, "clamped.toml", dt))
    end
    _print_table("Implicit Newmark (γ=0.5, β=0.25)  —  consistent mass",
                 impl_dts, impl_errs)

    # ------- explicit central-difference sweep -------
    expl_dir  = joinpath(repo_root, "examples", "mechanics",
                         "explicit-dynamic", "clamped")
    # CFL = c·Δt/h ≤ 1, so Δt ≤ 1e-6.  Top of the sweep is 2e-7 (matches the
    # checked-in toml).  Going below 2.5e-8 burns wall time for diminishing
    # returns.
    expl_dts  = [2.0e-7, 1.0e-7, 5.0e-8, 2.5e-8]
    expl_errs = Float64[]
    for dt in expl_dts
        @info "explicit  dt = $dt"
        push!(expl_errs, _run_at_dt(expl_dir, "clamped.toml", dt))
    end
    _print_table("Central difference  —  lumped mass",
                 expl_dts, expl_errs)

    return (impl_dts = impl_dts, impl_errs = impl_errs,
            expl_dts = expl_dts, expl_errs = expl_errs)
end
