# Regression test for `output.internal variables: true`.
#
# `_write_element_fields!` looked up per-block physics by NAME:
#
#     block_physics = p_cpu.physics[block_name]
#
# `block_name` comes from `fspace.ref_fes`, which keeps the Exodus block names.
# But `FEC.create_parameters` used to re-key physics and properties POSITIONALLY
# as `region_1..N` whenever it was handed a bare `AbstractPhysics` -- which was
# always, because Carina applied a single material to the whole mesh.  The two
# NamedTuples therefore shared no key names unless the mesh block was literally
# called "region_1", so the lookup threw
#
#     FieldError: type NamedTuple has no field `tension`, available fields: `region_1`
#
# for every other mesh.  Nothing caught it: the only bundled inputs that set
# `internal variables: true` are the two tension-specimen-j2 examples, and
# neither is in the test suite.
#
# Note the throw happens before anything that depends on the number of state
# variables, so a model with no state variables reproduces it just as well as a
# plastic one -- which keeps this test cheap.

@testset "Internal variables output" begin

    example_dir = joinpath(@__DIR__, "..", "examples", "mechanics", "quasistatic", "cube")

    # The cube mesh's block is named "cube", not "region_1" -- that mismatch is
    # the whole point.
    function run_with_internal_variables(flag)
        mktempdir() do dir
            cp_example(joinpath(example_dir, "cube.g"), joinpath(dir, "cube.g"))
            dict = Carina.YAML.load_file(joinpath(example_dir, "cube.yaml");
                                         dicttype = Dict{String, Any})
            out = get!(dict, "output", Dict{String, Any}())
            out["internal variables"] = flag
            dict["input mesh file"]  = joinpath(dir, "cube.g")
            dict["output mesh file"] = joinpath(dir, "cube.e")
            sim = Carina.create_simulation(dict, dir)
            Carina.evolve!(sim)
            return sim
        end
    end

    @testset "writing state output does not throw on a normally-named block" begin
        sim = run_with_internal_variables(true)
        @test sim isa Carina.SingleDomainSimulation
    end

    @testset "still fine with the flag off" begin
        sim = run_with_internal_variables(false)
        @test sim isa Carina.SingleDomainSimulation
    end

    @testset "physics is keyed by mesh block name, in block order" begin
        # Carina hands FEC a physics NamedTuple keyed by the real mesh block
        # names, and FEC now matches those names against the function space's
        # blocks and reorders to block order, so the mismatch that caused this
        # bug no longer exists.  The output path still indexes by position,
        # which is what `foreach_block` guarantees -- assert the names line up
        # too, because a change that reintroduced a separate key space would
        # bring the original failure mode back with it.
        mktempdir() do dir
            cp_example(joinpath(example_dir, "cube.g"), joinpath(dir, "cube.g"))
            dict = Carina.YAML.load_file(joinpath(example_dir, "cube.yaml");
                                         dicttype = Dict{String, Any})
            dict["input mesh file"]  = joinpath(dir, "cube.g")
            dict["output mesh file"] = joinpath(dir, "cube.e")
            sim = Carina.create_simulation(dict, dir)

            fspace      = Carina.FEC.function_space(sim.integrator.asm.dof)
            block_names = collect(keys(fspace.ref_fes))
            phys_keys   = collect(keys(sim.params.physics))

            @test length(block_names) == length(phys_keys)
            @test Symbol("cube") in block_names
            # Same names, same order -- physics entry k really is block k.
            @test phys_keys == block_names
            # Properties are a flat `PropertyField` now -- no keys, so assert
            # the block count instead.
            @test Carina.FEC.num_blocks(sim.params.properties) == length(block_names)
            @test values(sim.params.physics)[1] isa Carina.SolidMechanics
        end
    end

end

# --------------------------------------------------------------------------- #
# Nodal recovery (L2 projection of quadrature fields to nodes).
#
# The recovery machinery had three untested regions: internal-variable
# accumulation and write-out (a state-carrying material never ran with
# `recovery` enabled), the consistent-mass variant (nothing ever requested
# it, so the scalar mass assembly + Cholesky in input parsing and the
# factor-solve in io.jl were both dead), and the interaction of the two.
# Drive a J2-plasticity cube past yield so eqps is nonzero and both
# projections have something real to recover.
# --------------------------------------------------------------------------- #
@testset "Nodal recovery of stress and internal variables" begin
    example_dir = joinpath(@__DIR__, "..", "examples", "mechanics",
                           "quasistatic", "cube")

    # E = 1 GPa, yield at 1 MPa => first yield at eps_z ~ 1e-3; the prescribed
    # stretch of 4e-3 is far beyond it, so eqps > 0 over the whole cube.
    recovery_yaml(recovery) = """
type: single
input mesh file: cube.g
output mesh file: cube_recovery.e
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
      elastic modulus: 1.0e9
      Poisson's ratio: 0.25
      density: 1000.0
      yield stress: 1.0e6
      hardening modulus: 1.0e8
time integrator:
  type: quasi static
  initial time: 0.0
  final time: 1.0
  time step: 1.0
boundary conditions:
  dirichlet:
    - side set: ssx-
      component: x
      function: "0.0"
    - side set: ssy-
      component: y
      function: "0.0"
    - side set: ssz-
      component: z
      function: "0.0"
    - side set: ssz+
      component: z
      function: "4.0e-3 * t"
solver:
  type: newton
  termination:
    fail when any:
      - maximum iterations: 30
    converge when any:
      - absolute residual: 1.0e-6
      - relative residual: 1.0e-10
  linear solver:
    type: direct
"""

    function run_recovery(recovery)
        mktempdir() do dir
            cp_example(joinpath(example_dir, "cube.g"), joinpath(dir, "cube.g"))
            path = joinpath(dir, "cube_recovery.yaml")
            open(io -> write(io, recovery_yaml(recovery)), path, "w")
            dict = Carina.YAML.load_file(path; dicttype = Dict{String, Any})
            sim  = Carina.create_simulation(dict, dir)
            Carina.evolve!(sim)
            Carina.FEC.close(sim.post_processor)

            out = joinpath(dir, "cube_recovery.e")
            exo = Carina.Exodus.ExodusDatabase(out, "r")
            try
                names = Carina.Exodus.read_names(exo, Carina.Exodus.NodalVariable)
                last_step = length(Carina.Exodus.read_times(exo))
                read(name) = Carina.Exodus.read_values(
                    exo, Carina.Exodus.NodalVariable, last_step, name)
                return sim, names, Dict(
                    "sigma_zz_n" => read("sigma_zz_n"),
                    "eqps_n"     => read("eqps_n"))
            finally
                Carina.close(exo)
            end
        end
    end

    @testset "lumped recovery includes internal variables" begin
        sim, names, vals = run_recovery("lumped")
        @test sim.recovery_data isa Carina.LumpedRecovery
        @test "sigma_zz_n" in names
        @test "eqps_n" in names
        @test "Fp_xx_n" in names       # tensor-valued state recovers too

        σ, eqps = vals["sigma_zz_n"], vals["eqps_n"]
        @test all(isfinite, σ) && all(isfinite, eqps)
        # Past yield: the axial stress sits near the (hardened) flow stress,
        # far below the 4 MPa an elastic response would carry.
        @test maximum(abs, σ) > 0.5e6
        @test maximum(abs, σ) < 4.0e6
        # eqps really accumulated, everywhere in this homogeneous stretch.
        @test minimum(eqps) > 0.0
    end

    @testset "consistent recovery solves the scalar mass system" begin
        sim, names, vals = run_recovery("L2")
        @test sim.recovery_data isa Carina.ConsistentRecovery
        @test "sigma_zz_n" in names
        @test "eqps_n" in names

        σ, eqps = vals["sigma_zz_n"], vals["eqps_n"]
        @test all(isfinite, σ) && all(isfinite, eqps)
        @test minimum(eqps) > 0.0
    end
end

# --------------------------------------------------------------------------- #
# Velocity / acceleration nodal output under Dirichlet BCs.
#
# The only run that ever wrote velocity output was the free-flying explicit
# cube, whose unknown_dofs are the identity map.  That hid a scatter bug in
# _full_dof_to_h1field: the integrator's V and A have been full-DOF since the
# Norma-shape state rework, but the scatter still treated its input as a
# reduced (free-DOF) vector, permuting values into the wrong nodes whenever
# constraints exist.  A BC-driven run makes the written field equal the
# integrator state only if the layout is honored.
# --------------------------------------------------------------------------- #
@testset "Velocity and acceleration output under BCs" begin
    example_dir = joinpath(@__DIR__, "..", "examples", "mechanics",
                           "quasistatic", "cube")
    yaml = """
type: single
input mesh file: cube.g
output mesh file: cube_va.e
output:
  velocity: true
  acceleration: true
model:
  type: solid mechanics
  material:
    blocks:
      cube: neohookean
    neohookean:
      elastic modulus: 1.0e9
      Poisson's ratio: 0.25
      density: 1000.0
time integrator:
  type: central difference
  initial time: 0.0
  final time: 2.0e-7
  time step: 1.0e-7
  gamma: 0.5
boundary conditions:
  dirichlet:
    - side set: ssz-
      component: z
      function: "0.0"
    - side set: ssz+
      component: z
      function: "1.0e-3 * t"
"""
    mktempdir() do dir
        cp_example(joinpath(example_dir, "cube.g"), joinpath(dir, "cube.g"))
        path = joinpath(dir, "cube_va.yaml")
        open(io -> write(io, yaml), path, "w")
        dict = Carina.YAML.load_file(path; dicttype = Dict{String, Any})
        sim  = Carina.create_simulation(dict, dir)
        Carina.evolve!(sim)
        Carina.FEC.close(sim.post_processor)

        out = joinpath(dir, "cube_va.e")
        exo = Carina.Exodus.ExodusDatabase(out, "r")
        V_file = A_file = nothing
        try
            last_step = length(Carina.Exodus.read_times(exo))
            rd(name) = Carina.Exodus.read_values(
                exo, Carina.Exodus.NodalVariable, last_step, name)
            V_file = vcat((rd("velo_$c")' for c in ("x", "y", "z"))...)
            A_file = vcat((rd("acce_$c")' for c in ("x", "y", "z"))...)
        finally
            Carina.close(exo)
        end

        # The written nodal field must BE the integrator state, node for node
        # (BC slots carry g' and g'', free slots the solved values).
        V_ig = reshape(Vector(sim.integrator.V), 3, :)
        A_ig = reshape(Vector(sim.integrator.A), 3, :)
        @test V_file ≈ V_ig atol = 1.0e-14 * max(1.0, maximum(abs, V_ig))
        @test A_file ≈ A_ig atol = 1.0e-14 * max(1.0, maximum(abs, A_ig))
    end
end
