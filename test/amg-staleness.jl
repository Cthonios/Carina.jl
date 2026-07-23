# Regression tests for the AMG hierarchy staleness detector.
#
# The detector is the ONLY thing that triggers a rebuild on the quasi-static
# path: `_build_precond_op` passes c_M = 0.0 there, so the `c_M_changed` test
# in `_update_amg_precond_assembled!` compares 0.0 > 0.0 and never fires.
#
# It was originally called only from the Newmark `_linear_solve!` method, so on
# quasi-static runs the hierarchy was built once at the reference configuration
# and reused for the entire load history -- CG iteration counts drifted from 63
# to 400 on a 528k-DOF twisted-bar run while `nbuilds` stayed at 1, and the
# current-configuration near-nullspace never got a chance to matter.

@testset "AMG staleness detector" begin

    amg() = Carina.AMGPreconditioner([1, 2, 3])

    @testset "baseline latches on the first solve after a build" begin
        p = amg()
        @test p.base_iters == 0
        Carina._amg_track_iters!(p, 40)
        @test p.base_iters == 40
        @test p.rebuild == false

        # Later solves must not move the baseline.
        Carina._amg_track_iters!(p, 55)
        @test p.base_iters == 40
    end

    @testset "growth past 3x baseline requests a rebuild" begin
        p = amg()
        Carina._amg_track_iters!(p, 40)
        Carina._amg_track_iters!(p, 119)          # < 3x
        @test p.rebuild == false
        Carina._amg_track_iters!(p, 121)          # > 3x
        @test p.rebuild == true
    end

    @testset "small counts do not thrash the hierarchy" begin
        # 3x a tiny baseline is still tiny; the floor of 30 keeps a hierarchy
        # that is working well from being rebuilt over noise.
        p = amg()
        Carina._amg_track_iters!(p, 5)
        Carina._amg_track_iters!(p, 29)
        @test p.rebuild == false
        Carina._amg_track_iters!(p, 31)
        @test p.rebuild == true
    end

    @testset "no-op for every other preconditioner" begin
        # The quasi-static path calls this unguarded, so the fallback has to
        # exist for the preconditioners that have no hierarchy.
        for pc in (Carina.NoPreconditioner(),
                   Carina.ICPreconditioner(),
                   Carina.JacobiPreconditioner(Float64[1.0, 1.0]))
            @test Carina._amg_track_iters!(pc, 10_000) === nothing
        end
    end

    @testset "both linear-solve paths feed the detector" begin
        # Guards against the call being dropped from one path again. Checked by
        # reading the source: a bare `_amg_track_iters!` call must appear inside
        # each of the two `_linear_solve!(::KrylovLinearSolver, ...)` methods.
        src = read(joinpath(@__DIR__, "..", "src", "linear_solvers.jl"), String)
        lines = split(src, '\n')
        starts = findall(l -> occursin(r"^function _linear_solve!\(ls::KrylovLinearSolver", l), lines)
        @test length(starts) == 2
        for (i, s) in enumerate(starts)
            stop = i < length(starts) ? starts[i+1] - 1 : length(lines)
            body = join(lines[s:stop], '\n')
            @test occursin("_amg_track_iters!(", body)
        end
    end

end
