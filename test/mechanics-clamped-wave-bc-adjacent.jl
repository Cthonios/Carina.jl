# Diagnostic test: BC-adjacent residual at z = -L/2 + h in the BC-driven
# clamped-bar Gaussian-pulse setup.  Originally written under the
# hypothesis that the dropped Newmark inertial cross-term
# (c_M·M_{f,BC}·g''(t_{n+1})) accounted for the ~7-17× implicit/explicit
# gap in |vz| and |az| seen at z = -L/2 + h.  Empirically that hypothesis
# is WRONG for this problem:
#
#   1. The cross-term fix landed in FEC v0.14+ as
#      assemble_matrix_free_action_full!, validated end-to-end against an
#      assembled K_full · v_full reference (FEC TestAssemblers.jl).
#   2. Carina's NewmarkIntegrator.evaluate! now calls that primitive with
#      v_full[BC] = g''(t_{n+1})/c_M, so the inertial residual carries
#      M_{f,BC}·g''(t_{n+1}) exactly as Norma's full-DOF M·a product does.
#   3. After the fix, |vz| at z = -L/2 + h shifts only by ~0.04% from
#      the pre-fix value (6.39945e-7 → 6.40442e-7).  The dominant noise
#      at this node is Newmark consistent-mass *dispersion* of the bulk
#      pulse — an intrinsic property of the time integrator, not a
#      cross-term defect — and the cross-term contribution sits 3-4
#      orders of magnitude below it.
#
# So the implicit assertions below still fail today, but NOT for the
# reason this test was written to expose.  They are pinned @test_broken
# as documentation: a reminder that the cross-term has been addressed and
# anyone investigating the BC-adjacent gap further should look at
# Newmark dispersion (e.g., higher-order time stepping, mass scaling, or
# specially-tuned β/γ) rather than the inertial cross-term.

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
    # ig.U/V/A are full-DOF in the Norma-shape integrator state; BC slots
    # carry g, g', g'' at the current time.
    if sym === :displacement
        return Vector(sim.params.field.data)
    elseif sym === :velocity
        return Vector(sim.integrator.V)
    else
        return Vector(sim.integrator.A)
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
        # (pulse departed); any nonzero FEM value is either bulk-dispersion
        # noise or the cross-term defect.
        #
        # Empirically on this mesh, the explicit (lumped, M_{fB} ≡ 0) floor:
        #   |vz| ≲ 1.0e-7 m/s,  |az| ≲ 0.3 m/s².
        # The implicit floor sits ~7× and ~17× above these on vz and az —
        # see the top-of-file note for why the cross-term fix doesn't
        # bridge this gap.  Tolerances chosen midway so that:
        #   - explicit passes;
        #   - implicit (today's) fails as documented;
        #   - any future change that DOES close the gap (e.g., higher-order
        #     time stepping, mass scaling) will flag the @test_broken
        #     markers as unexpectedly passing and we revisit then.
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
