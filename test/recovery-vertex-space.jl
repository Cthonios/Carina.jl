# Carina: Copyright 2026 National Technology & Engineering Solutions of
# Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software. This software
# is released under the BSD license detailed in the file LICENSE.md in the
# top-level Carina.jl directory.

@testset "Nodal recovery on the vertex space" begin
    # The nodal recovery projects the quadrature-point fields onto the
    # continuous piecewise-linear space on the vertices and interpolates the
    # other nodes.  A homogeneous state is reproduced exactly by any such
    # projection: a cube under a uniform uniaxial stretch with the J2 model
    # beyond yield has one stress and one plastic strain at every quadrature
    # point, so every recovered nodal value must equal them, on TETRA15 (edge,
    # face and interior nodes) and on TETRA10, with the lumped and the
    # consistent recovery.  Poisson's ratio 0.3: at ν = 0.49995 the bulk
    # modulus turns the round-off of the converged displacement into stress
    # differences of 1e-4 relative that vary with the thread count and the
    # Julia version (CI, 2026-10-15), which is not what the test measures.  Before 2026-10-15 the recovery used the full
    # basis of these elements: on TETRA15 the lumped mass was zero or negative
    # at some nodes and the consistent mass failed to factorize.
    examples = joinpath(@__DIR__, "..", "examples", "mechanics", "quasistatic")
    deck(recovery) = """
type: single
input mesh file: cube.g
output mesh file: cube-recovery-$recovery.e
output:
  stress: true
  internal variables: true
  recovery: $recovery
model:
  type: solid mechanics
  material:
    blocks:
      cube: j2 plasticity
    j2 plasticity:
      elastic modulus: 2.9999e9
      Poisson's ratio: 0.3
      density: 8930.0
      yield stress: 1.0e6
      hardening modulus: 1.0e7
time integrator:
  type: quasi static
  time step: 0.5
  final time: 1.0
  initial time: 0.0
  predictor: tangent
boundary conditions:
  dirichlet:
    - function: "0.0"
      side set: ssx-
      component: x
    - function: "0.0"
      side set: ssy-
      component: y
    - function: "0.0"
      side set: ssz-
      component: z
    - function: "2.0e-3 * t"
      side set: ssz+
      component: z
solver:
  type: newton
  termination:
    fail when any:
      - maximum iterations: 30
    converge when any:
      - absolute residual: 1.0e-2
      - relative residual: 1.0e-8
  linear solver:
    type: direct
"""
    function recovered(mesh_dir, recovery)
        mktempdir() do dir
            cp_example(joinpath(mesh_dir, "cube.g"), joinpath(dir, "cube.g"))
            path = joinpath(dir, "cube-recovery-$recovery.yaml")
            write(path, deck(recovery))
            dict = Carina.YAML.load_file(path; dicttype = Dict{String, Any})
            sim  = Carina.create_simulation(dict, dir)
            Carina.evolve!(sim)
            Carina.FEC.close(sim.post_processor)
            exo = Carina.Exodus.ExodusDatabase(joinpath(dir, "cube-recovery-$recovery.e"), "r")
            try
                last_step = length(Carina.Exodus.read_times(exo))
                rd(name) = Carina.Exodus.read_values(exo, Carina.Exodus.NodalVariable, last_step, name)
                return sim, rd("sigma_zz_n"), rd("eqps_n")
            finally
                Carina.close(exo)
            end
        end
    end
    for (mesh, elem) in (("cube-tet15", "TETRA15"), ("cube-tet10", "TETRA10"))
        for recovery in ("lumped", "consistent")
            @testset "$elem, $recovery" begin
                sim, σzz, eqps = recovered(joinpath(examples, mesh), recovery)
                T = recovery == "lumped" ? Carina.LumpedRecovery : Carina.ConsistentRecovery
                @test sim.recovery_data isa T
                # Every node has a value, the vertices and the interpolated nodes alike.
                @test length(σzz) == size(sim.params_cpu.coords, 2)
                @test all(isfinite, σzz) && all(isfinite, eqps)
                # Homogeneous state: one value at every node, to the
                # non-uniformity left by the Newton tolerance, and in the
                # plastic range.
                @test maximum(σzz) - minimum(σzz) <= 1e-5 * abs(sum(σzz) / length(σzz))
                @test maximum(eqps) - minimum(eqps) <= 1e-5 * (sum(eqps) / length(eqps))
                @test sum(eqps) / length(eqps) > 1e-4
                @test sum(σzz) / length(σzz) > 1e6
                # Interpolation: the values at the non-vertex nodes lie between
                # the vertex values (trivially here; the test is that they are set).
                @test count(iszero, σzz) == 0
            end
        end
    end
end
