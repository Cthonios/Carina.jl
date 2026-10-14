# Punch indentation: frictionless indentation of a cylinder by a rigid flat
# circular punch, quasi-static finite-deformation J2 plasticity, nearly
# incompressible (README.md).  TET15-P1 and plain TETRA15 in Carina.
#
# Problem (units m): cylinder of radius R = 4a and height H = 4a, a = 1 mm,
# axis along z, base at z = 0; the punch footprint is the disk r <= a on the
# top face.  u_z = 0 on the base (node set base); u_x = u_y = 0 at the base
# node on the axis (axis-base) and u_y = 0 at the base node (R, 0, 0)
# (pin-base), which remove the rigid-body modes the base plane leaves;
# u_z = -0.2a t on every node of the footprint (node set punch), u_x, u_y free.
# Twenty output steps, t = 0.05k, depth d = 0.01a k.  J2 plasticity with
# linear isotropic hardening (Carina's Simo-Hughes model): mu = 1e9 Pa,
# kappa = 1e13 Pa, sigma_y = 1e6 Pa, H = 1e7 Pa (case A) or 0 (case B).
#
# Elements:
#   tet15-p1   TET15-P1, TETRA15 mesh, volumetric projection linear
#   tetra15    TETRA15, no projection (locks)
#
# Usage (from the Carina root; the threads serve Carina's element loops):
#   julia -t 32 --project=. benchmark/tet15-p1/punch/run.jl --h 4,8 --case A,B
#        [--stages mesh,convert,run] [--elements tet15-p1,tetra15] [--device cpu|cuda]
#        [--preconditioner amg|jacobi] [--first-step 1.220703125e-5] [--ti "key: value"]...
#        [--constraints point|line]
# --h takes the divisors of a (h = a/4, a/8, a/16).  A CUDA run needs the
# vendor package loaded in the calling session, as for the Taylor driver:
#   julia --project=bin -e 'using CUDA; append!(ARGS, [...]);
#       include("benchmark/tet15-p1/punch/run.jl")'
# with JULIA_LOAD_PATH="@:$PWD:@stdlib"; backend.jl of the Taylor driver then
# includes bin/cuda_always_inline.jl, as bin/carina.jl does.
# --ti adds a line to the time-integrator section of the deck (for a
# predictor key); it may be repeated.
#
# Each run writes runs/<element>-<case>-h<d>[-<device>]/ with the deck, the
# Exodus output (stress and internal variables at every output step), the
# Carina log, force.tsv (step, depth, punch force: the sum over the nodes of
# node set punch of the z component of the assembled internal force, positive
# in compression), reactions.tsv (the reactions at the point constraints) and
# run.log (steps, Newton and CG iterations, wall time per output step).

using Carina
using Exodus
using Printf

const DIR    = @__DIR__
const MESHES = joinpath(DIR, "meshes")
const RUNS   = joinpath(DIR, "runs")
const CUBIT  = get(ENV, "PUNCH_CUBIT", "/usr/local/cubit/cubit")
const A_MM   = 1.0
const MU     = 1.0e9
const KAPPA  = 1.0e13
const E_MOD  = 9 * KAPPA * MU / (3 * KAPPA + MU)
const NU     = (3 * KAPPA - 2 * MU) / (2 * (3 * KAPPA + MU))
const CASES  = Dict("A" => 1.0e7, "B" => 0.0)

const DEVICE  = Ref("cpu")
const PRECOND = Ref("amg")
const FIRST   = Ref(0.05 / 4096)
const TI_EXTRA = String[]
const CONSTRAINTS = Ref("point")   # point: axis-base, pin-base; line: axis, pin
const SA = Ref("axis-base")
const SP = Ref("pin-base")

include(joinpath(DIR, "..", "taylor", "backend.jl"))

tet4_file(d)  = joinpath(MESHES, "punch-h$d-tet4.g")
tet10_file(d) = joinpath(MESHES, "punch-h$d-tet10.g")
tet15_file(d) = joinpath(MESHES, "punch-h$d-tet15.g")

function mesh(d)
    mkpath(MESHES)
    log = joinpath(MESHES, "cubit-h$d.log")
    Base.run(pipeline(`$CUBIT -batch -nographics -nojournal -noecho -information off
                       "h=$(A_MM / d)" "out='$(tet4_file(d))'" $(joinpath(DIR, "punch.jou"))`;
                      stdout = log, stderr = log))
end

function convert(d)
    jl = Base.julia_cmd()
    Base.run(`$jl --project=$(pkgdir(Carina)) $(joinpath(DIR, "convert.jl")) $(tet4_file(d)) $(tet10_file(d))`)
    isfile(tet15_file(d)) && rm(tet15_file(d))
    Carina.tetra15_mesh(tet10_file(d), tet15_file(d))
    Base.run(`$jl --project=$(pkgdir(Carina)) $(joinpath(DIR, "check_tetra15.jl")) $(tet15_file(d))`)
    for t in (10, 15)
        f = t == 10 ? tet10_file(d) : tet15_file(d)
        Base.run(`$jl --project=$(pkgdir(Carina)) $(joinpath(DIR, "..", "taylor", "sierra", "rename-sidesets.jl"))
                  $f $(replace(f, ".g" => "-sierra.g"))`)
    end
end

function deck(element, case, d, out)
    projection = element == "tet15-p1" ? "  volumetric projection: linear\n" :
                 element == "tetra15" ? "" : error("unknown element $element")
    extra = join(("  $l\n" for l in TI_EXTRA), "")
    return """
type: single
device: $(DEVICE[])
input mesh file: $(tet15_file(d))
output mesh file: $out
output interval: 0.05
output:
  stress: true
  internal variables: true
model:
  type: solid mechanics
$(projection)  material:
    blocks:
      cylinder: j2 plasticity
    j2 plasticity:
      elastic modulus: $(repr(E_MOD))
      Poisson's ratio: $(repr(NU))
      density: 8930.0
      yield stress: 1.0e6
      hardening modulus: $(repr(CASES[case]))
time integrator:
  type: quasi static
  initial time: 0.0
  final time: 1.0
  time step: $(repr(FIRST[]))
  minimum time step: 1.0e-9
  maximum time step: 0.05
  decrease factor: 0.5
  increase factor: 2.0
$(extra)boundary conditions:
  dirichlet:
    - node set: base
      component: z
      function: "0.0"
    - node set: $(SA[])
      component: x
      function: "0.0"
    - node set: $(SA[])
      component: y
      function: "0.0"
    - node set: $(SP[])
      component: y
      function: "0.0"
    - node set: punch
      component: z
      function: "-2.0e-4 * t"
solver:
  type: newton
  termination:
    fail when any:
      - maximum iterations: 30
    converge when any:
      - absolute residual: 1.0e-6
      - relative residual: 1.0e-8
  linear solver:
    type: iterative
    tolerance: 1.0e-8
    maximum iterations: 1000
    preconditioner:
      type: $(PRECOND[])
"""
end

# Node indices (1-based, in the mesh order) of a node set.
function node_set(file, name)
    exo = ExodusDatabase(file, "r")
    i = findfirst(==(name), read_names(exo, NodeSet))
    i === nothing && error("node set $name missing in $file")
    nodes = Int.(read_sets(exo, NodeSet)[i].nodes)
    close(exo)
    return nodes
end

# The assembled internal force at the current displacement, as a 3 x nnodes
# array on the host, at all DOFs, Dirichlet ones included: at a constrained
# DOF it is the reaction.  Assembled here rather than through the integrator,
# because FEC's residual(asm) zeroes the constrained entries of the storage
# in place on a condensed DOF layout.  At the converged state the material
# update reproduces the converged internal variables.
function internal_force(sim)
    ig, p = sim.integrator, sim.params
    Carina.FEC.assemble_vector!(ig.asm, Carina.FEC.residual, ig.U, p)
    F = Array(ig.asm.residual_storage.data)
    length(F) == 3 * size(p.field.data, 2) || length(F) == length(p.field.data) ||
        error("residual storage is not a full-DOF field")
    return reshape(F, 3, :)
end

function count_log(logfile)
    newton = cg = 0
    isfile(logfile) || return (0, 0)
    for line in eachline(logfile)
        occursin(r"Iter \[[1-9][0-9]*\]", line) && (newton += 1)
        m = match(r"CG: (\d+) iters", line)
        m === nothing || (cg += parse(Int, m[1]))
    end
    return newton, cg
end

function run_case(element, case, d)
    SA[], SP[] = CONSTRAINTS[] == "line" ? ("axis", "pin") : ("axis-base", "pin-base")
    tag = "$element-$case-h$d" * (DEVICE[] == "cpu" ? "" : "-" * DEVICE[]) * (CONSTRAINTS[] == "line" ? "-lc" : "")
    dir = joinpath(RUNS, tag)
    mkpath(dir)
    deckfile = joinpath(dir, "punch.yaml")
    open(io -> write(io, deck(element, case, d, joinpath(dir, "punch.e"))), deckfile, "w")
    punch = node_set(tet15_file(d), "punch")
    na, np = node_set(tet15_file(d), SA[]), node_set(tet15_file(d), SP[])
    backend = backend_of(DEVICE[])
    Carina.open_log_file(deckfile)
    t0 = time()
    rows = Tuple{Int, Float64, Float64, Float64, Float64, Float64, Float64, Int, Int}[]
    status = "completed"
    try
        sim = Base.invokelatest(Carina.create_simulation,
                                Carina.YAML.load_file(deckfile; dicttype = Dict{String, Any}), dir;
                                backend = backend)
        (; params, post_processor, controller) = sim
        n_steps = controller.num_stops - 1
        for k in 1:n_steps
            tk = time()
            Base.invokelatest(Carina._advance_controller!, controller)
            params.times.time_current = controller.prev_time
            Base.invokelatest(Carina._subcycle!, sim, controller.time, false)
            Base.invokelatest(Carina.write_output!, sim, k + 1)
            F = Base.invokelatest(internal_force, sim)
            fz = sum(F[3, n] for n in punch)
            newton, cg = count_log(first(splitext(deckfile)) * ".log")
            push!(rows, (k, 2.0e-4 * controller.time, -fz, sum(F[1, na]), sum(F[2, na]), sum(F[2, np]),
                         time() - tk, newton, cg))
            @printf("%s step %2d t = %.3f force %.6e N  wall %.1f s\n", tag, k, controller.time, -fz, time() - tk)
        end
        Carina.FEC.close(post_processor)
    catch err
        status = "failed: " * sprint(showerror, err)[1:min(end, 300)]
        @warn "$tag failed" exception = (err, catch_backtrace())
    finally
        Carina.close_log_file()
    end
    wall = time() - t0
    open(joinpath(dir, "force.tsv"), "w") do io
        println(io, "step\tdepth\tforce")
        println(io, "0\t0.000000e+00\t0.000000000e+00")
        for r in rows
            @printf(io, "%d\t%.6e\t%.9e\n", r[1], r[2], r[3])
        end
    end
    open(joinpath(dir, "reactions.tsv"), "w") do io
        println(io, "step\t$(SA[])_rx\t$(SA[])_ry\t$(SP[])_ry\tpunch_force")
        for r in rows
            @printf(io, "%d\t%.6e\t%.6e\t%.6e\t%.6e\n", r[1], r[4], r[5], r[6], r[3])
        end
    end
    open(joinpath(dir, "run.log"), "w") do io
        println(io, "run\t$tag")
        println(io, "host\t$(gethostname())")
        println(io, "device\t$(DEVICE[])\nthreads\t$(Threads.nthreads())")
        println(io, "preconditioner\t$(PRECOND[])\nfirst step\t$(FIRST[])\nconstraints\t$(CONSTRAINTS[])")
        println(io, "status\t$status\nwall_s\t$(round(wall; digits = 1))")
        newton, cg = count_log(first(splitext(deckfile)) * ".log")
        nsteps = isfile(first(splitext(deckfile)) * ".log") ?
                 count(l -> occursin("[ADVANCE]", l), eachline(first(splitext(deckfile)) * ".log")) : 0
        cuts = isfile(first(splitext(deckfile)) * ".log") ?
               count(l -> occursin("[RECOVER]", l), eachline(first(splitext(deckfile)) * ".log")) : 0
        println(io, "time steps\t$nsteps\nfailed steps (cut)\t$cuts\nnewton iterations\t$newton\ncg iterations\t$cg")
        println(io, "output_step\tt\tforce_N\twall_s\tnewton_cumulative\tcg_cumulative")
        for r in rows
            @printf(io, "%d\t%.3f\t%.6e\t%.1f\t%d\t%d\n", r[1], r[2] / 2.0e-4, r[3], r[7], r[8], r[9])
        end
    end
    @printf("%s: %s, %.1f s\n", tag, status, wall)
end

function parse_args(args)
    opts = Dict("--h" => "4", "--case" => "A,B", "--stages" => "run",
                "--elements" => "tet15-p1,tetra15")
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--device"
            DEVICE[] = args[i + 1]
        elseif a == "--preconditioner"
            PRECOND[] = args[i + 1]
        elseif a == "--first-step"
            FIRST[] = parse(Float64, args[i + 1])
        elseif a == "--constraints"
            CONSTRAINTS[] = args[i + 1]
        elseif a == "--ti"
            push!(TI_EXTRA, args[i + 1])
        elseif haskey(opts, a)
            opts[a] = args[i + 1]
        else
            error("unknown option $a")
        end
        i += 2
    end
    return opts
end

function main(args)
    opts = parse_args(args)
    levels = parse.(Int, split(opts["--h"], ","))
    stages = split(opts["--stages"], ",")
    for d in levels
        "mesh" in stages && mesh(d)
        "convert" in stages && convert(d)
        if "run" in stages
            for case in split(opts["--case"], ","), el in split(opts["--elements"], ",")
                run_case(String(el), String(case), d)
            end
        end
    end
end

main(ARGS)
