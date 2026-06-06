# Regression test: dropped cross-term in implicit Newmark with time-varying
# Dirichlet BC.
#
# Reuses the BC-driven clamped-bar Gaussian-pulse setup
# (test/mechanics-clamped-wave-bc.jl), but samples the FEM state at the
# *first interior node next to the driver BC* (z = -L/2 + h, h = 1 mm).
# At t_f = 7.5e-4 the pulse has long since departed (peak at z = 0); the
# analytical solution at z = -L/2 + h is g(t_f - h/c) ≈ 1e-25 m, i.e.,
# effectively zero.  Any nonzero displacement/velocity/acceleration at this
# node is either bulk dispersion noise or the cross-term defect.
#
# The defect: src/integrators.jl evaluate!(::NewmarkIntegrator, p) calls
#   FEC.assemble_matrix_free_action!(asm, FEC.mass_action, U, dU, p)
# with v_full = [dU; 0_BC] in p.hvp_scratch_field.  The semantically correct
# value at that call site is v_full = [dU; dU_BC], so the inertial residual
# is missing  c_M * M_{fB} * dU_BC  — supported only at rows adjacent to the
# Dirichlet boundary.  Explicit central difference is structurally immune
# because lumped mass has M_{fB} ≡ 0.
#
# Expected outcome with current code:
#   Explicit (lumped):                 PASSES — dispersion noise only.
#   Implicit (consistent, defective):  FAILS  — cross-term offset.
# After the fix (FEC 5-arg matrix_free_action! + Carina dU_BC plumbing):
#   Both pass.

const _adj_a   = 1.0e-3
const _adj_tc  = 2.5e-4
const _adj_tau = 5.0e-5
const _adj_c   = 1000.0
const _adj_L   = 1.0
const _adj_h   = 1.0e-3   # mesh node spacing in z

@inline _adj_g(t)   = _adj_a * exp(-(t - _adj_tc)^2 / (2 * _adj_tau^2))
@inline _adj_gp(t)  = -((t - _adj_tc) / _adj_tau^2) * _adj_g(t)
@inline _adj_gpp(t) = ((t - _adj_tc)^2 / _adj_tau^4 - 1 / _adj_tau^2) * _adj_g(t)

function _adj_full(sim, sym::Symbol)
    asm = sim.integrator.asm
    n_total = length(sim.params.field.data)
    if sym === :displacement
        return Vector(sim.params.field.data)
    else
        src = sym === :velocity ? sim.integrator.V : sim.integrator.A
        full = zeros(n_total)
        full[Vector(asm.dof.unknown_dofs)] = Vector(src)
        return full
    end
end

# Snap z_target to the closest mesh node; assert the snap is exact within
# 1 nm (mesh h = 1 mm).
function _adj_node_at(z_coords, z_target)
    idx = argmin(abs.(z_coords .- z_target))
    @assert abs(z_coords[idx] - z_target) < 1.0e-9 "no node at z = $z_target (closest $(z_coords[idx]))"
    return idx
end

function _adj_run_and_report(label, example_dir; bug_active::Bool)
    mktempdir() do dir
        cp_example(joinpath(example_dir, "clamped-bc.g"),    joinpath(dir, "clamped-bc.g"))
        cp_example(joinpath(example_dir, "clamped-bc.toml"), joinpath(dir, "clamped-bc.toml"))
        sim = Carina.run(joinpath(dir, "clamped-bc.toml"))

        coords = reshape(Vector(sim.params_cpu.coords.data), 3, :)
        z_coords = coords[3, :]
        t_f = sim.controller.time

        U_full = _adj_full(sim, :displacement)
        V_full = _adj_full(sim, :velocity)
        A_full = _adj_full(sim, :acceleration)
        uz(i) = U_full[3 * i]; vz(i) = V_full[3 * i]; az(i) = A_full[3 * i]

        # Sample at z = -L/2 + k*h for k = 1, 2, 3.  All three lie in the
        # immediate boundary layer; M_{fB} is supported at k = 1 only for a
        # tet4 1mm mesh, but k = 2, 3 give us a decay profile to inspect.
        ks = (1, 2, 3)
        println("\n=== BC-adjacent diagnostics ($label) at t_f = $t_f s ===")
        println(rpad("z (m)", 14), rpad("uz_fem", 16), rpad("uz_anal", 16),
                rpad("vz_fem", 16), rpad("az_fem", 16))
        for k in ks
            z_target = -_adj_L / 2 + k * _adj_h
            i = _adj_node_at(z_coords, z_target)
            η = t_f - (z_coords[i] + _adj_L / 2) / _adj_c
            u_anal = _adj_g(η)
            println(rpad(round(z_coords[i]; digits=6), 14),
                    rpad(round(uz(i); sigdigits=6), 16),
                    rpad(round(u_anal; sigdigits=6), 16),
                    rpad(round(vz(i); sigdigits=6), 16),
                    rpad(round(az(i); sigdigits=6), 16))
        end

        # Assertions at z = -L/2 + h.  Analytical u, v, a are all ~ 1e-20
        # (pulse departed); any nonzero FEM value is either bulk dispersion
        # noise or the cross-term defect.
        #
        # Empirically on this mesh, the explicit (lumped, M_{fB} ≡ 0) floors:
        #   |vz| ≲ 1.0e-7 m/s,  |az| ≲ 0.3 m/s².
        # The implicit (consistent mass, missing c_M·M_{fB}·dU_BC term)
        # currently sits ~7× and ~17× above these on vz and az.  Tolerances
        # set midway: explicit passes; defective implicit must violate them
        # — that is what `bug_active = true` pins as @test_broken so the
        # suite stays green today, and so the moment the fix lands and
        # implicit drops to explicit's floor the test framework will flag
        # the broken markers as "unexpectedly pass".  Convert to @test then.
        i_adj = _adj_node_at(z_coords, -_adj_L / 2 + _adj_h)
        if bug_active
            @test_broken abs(vz(i_adj)) < 2.0e-7
            @test_broken abs(az(i_adj)) < 1.0
        else
            @test abs(vz(i_adj)) < 2.0e-7
            @test abs(az(i_adj)) < 1.0
        end
    end
end

@testset "Clamped Wave BC, BC-Adjacent Node (Explicit)" begin
    example_dir = joinpath(@__DIR__, "..", "examples", "mechanics",
                           "explicit-dynamic", "clamped-bc")
    _adj_run_and_report("explicit", example_dir; bug_active = false)
end

@testset "Clamped Wave BC, BC-Adjacent Node (Implicit)" begin
    example_dir = joinpath(@__DIR__, "..", "examples", "mechanics",
                           "implicit-dynamic", "clamped-bc")
    _adj_run_and_report("implicit", example_dir; bug_active = true)
end
