# Punch indentation, quarter model: frictionless indentation of a cylinder by
# a rigid flat circular punch, quasi-static finite-deformation J2 plasticity,
# nearly incompressible (README.md).  TET15-P1 and plain TETRA15 in Carina.
#
# Problem (units m): the quarter x >= 0, y >= 0 of the cylinder of radius
# R = 4a and height H = 4a, a = 1 mm, axis along z, base at z = 0; the punch
# footprint is the quarter disk r <= a on the top face.  u_x = 0 on the
# symmetry plane x = 0 (node set symm-x), u_y = 0 on y = 0 (symm-y), u_z = 0 on
# the base (base), u_z = -0.2a t on every node of the footprint (punch), u_x
# and u_y free there; no other constraint.  Twenty output steps, t = 0.05k,
# depth d = 0.01a k.  J2 plasticity with linear isotropic hardening (Carina's
# Simo-Hughes model): mu = 1e9 Pa, kappa = 1e13 Pa, sigma_y = 1e6 Pa,
# H = 1e7 Pa (case A) or 0 (case B).
#
# Elements:
#   tet15-p1   TET15-P1, TETRA15 mesh, volumetric projection linear
#   tetra15    TETRA15, no projection (locks)
#
# Time stepping (README, both codes): Δt = 0.05/4096 at the start, doubled
# after each converged step up to the maximum step of the level, 0.00625 (4/d)
# for h = a/d; a failed step is halved, down to the maximum step / 1024 (ten
# halvings); after the first cut of a run the step regrows by 1.5 instead of
# 2.  The output times t = 0.05k are stops that every step lands on.  Carina's
# controller has one increase factor, so the driver steps the controller
# itself, as Carina's _subcycle! does, with the integrator's increase factor
# set to 1 and the growth applied here.  The quasi-static tangent predictor
# (predictor: tangent) starts every step; inexact Newton with Eisenstat-Walker
# forcing, floor 1e-8; CG with the AMG preconditioner, at most 1000 iterations
# (--cg-cap).
#
# Usage (from the Carina root; the threads serve Carina's element loops):
#   julia -t 32 --project=. benchmark/tet15-p1/punch/run.jl --h 4,8 --case A,B
#        [--stages mesh,convert,run] [--elements tet15-p1,tetra15] [--device cpu|cuda]
#        [--preconditioner amg|jacobi] [--amg-operator tangent|compressible]
#        [--cg-cap 1000] [--tag <suffix>]
#        [--ti "key: value"]...
# --h takes the divisors of a (h = a/4, a/8, a/16, a/32).  A CUDA run needs the
# vendor package loaded in the calling session, as for the Taylor driver:
#   julia --project=bin -e 'using CUDA; append!(ARGS, [...]);
#       include("benchmark/tet15-p1/punch/run.jl")'
# with JULIA_LOAD_PATH="@:$PWD:@stdlib"; backend.jl of the Taylor driver then
# includes bin/cuda_always_inline.jl, as bin/carina.jl does.
# --ti adds a line to the time-integrator section of the deck; it may be
# repeated.  --tag appends a suffix to the run directory.  --amg-operator sets
# the operator the AMG hierarchy is built on (tangent, the default, or
# compressible: the tangent at the bulk modulus set to the shear modulus).
#
# Each run writes runs/<element>-<case>-h<d>[-<device>]/ with the deck, the
# Exodus output (stress and internal variables at every output step, at the
# quadrature points and as their consistent-mass L2 projection onto the nodes,
# sigma_*_n and eqps_n), the
# Carina log, force.tsv (step, depth, full punch force: four times the sum over
# the nodes of node set punch of the z component of the assembled internal
# force of the quarter, positive in compression) and run.log (steps, cuts,
# Newton and CG iterations, CG solves at the cap, the predictor, wall time per
# output step).

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
const FIRST  = 0.05 / 4096
const SYMMETRY = 4                  # full force = SYMMETRY x the quarter's

const DEVICE  = Ref("cpu")
const PRECOND = Ref("amg")
const AMG_OP  = Ref("tangent")
const TI_EXTRA = String[]
const CG_CAP  = Ref(1000)
const TAG     = Ref("")

include(joinpath(DIR, "..", "taylor", "backend.jl"))

tet4_file(d)  = joinpath(MESHES, "punch-h$d-tet4.g")
tet10_file(d) = joinpath(MESHES, "punch-h$d-tet10.g")
tet15_file(d) = joinpath(MESHES, "punch-h$d-tet15.g")
max_step(d)   = 0.00625 * 4 / d

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
  recovery: consistent
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
  time step: $(repr(FIRST))
  minimum time step: $(repr(max_step(d) / 1024 * (1 - 1.0e-12)))
  maximum time step: $(repr(max_step(d)))
  decrease factor: 0.5
  increase factor: 2.0
  predictor: tangent
$(extra)boundary conditions:
  dirichlet:
    - node set: symm-x
      component: x
      function: "0.0"
    - node set: symm-y
      component: y
      function: "0.0"
    - node set: base
      component: z
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
    maximum iterations: $(CG_CAP[])
    forcing term:
      type: eisenstat-walker
    preconditioner:
      type: $(PRECOND[])
$(PRECOND[] == "amg" ? "      operator: $(AMG_OP[])\n" : "")"""
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

# Steps the controller from the current time to `target` as Carina's
# _subcycle! does, with the growth of the step applied here: 2 until the
# first cut of the run, 1.5 after it, up to the maximum step.  The
# integrator's own increase factor is 1 during the step, so that a step
# returns from _advance_one_step! with time_step below the one it started
# with exactly when it was cut.  Returns (steps, cuts).
function subcycle!(sim, target, growth)
    (; params, integrator) = sim
    ig = integrator
    steps = cuts = 0
    while true
        t  = Carina.FEC.current_time(params.times)
        dt = Carina._adjusted_step(t, ig.time_step, target)
        params.times.Δt = dt
        Carina._carina_logf(4, :advance, "[%.2e, %.2e] : Δt = %.2e", t, t + dt, dt)
        ig.increase_factor = 1.0
        ts0 = ig.time_step
        Carina._advance_one_step!(sim)
        steps += 1
        # A cut leaves the step at half the step tried or less, below ts0
        # (the step tried is at most 1.5 ts0 when it closes the gap to the
        # stop).
        if ig.time_step < ts0 * (1 - 1.0e-12)
            cuts += 1
            growth[] = 1.5
        end
        ig.time_step = min(ig.time_step * growth[], ig.max_time_step)
        isapprox(Carina.FEC.current_time(params.times), target; rtol = 1e-6, atol = 1e-12) && break
    end
    return steps, cuts
end

# Counts from the Carina log: Newton iterations, CG iterations, CG solves and
# those that ended unconverged at the cap, and the predictor lines (kept,
# reverted, and the residual reduction of each), and the AMG hierarchy builds.
function count_log(logfile)
    c = (newton = 0, cg = 0, solves = 0, capped = 0, kept = 0, reverted = 0, skipped = 0, builds = 0)
    red = Float64[]
    isfile(logfile) || return c, red
    newton = cg = solves = capped = kept = reverted = skipped = builds = 0
    for line in eachline(logfile)
        occursin(r"Iter \[[1-9][0-9]*\]", line) && (newton += 1)
        m = match(r"CG: (\d+) iters", line)
        if m !== nothing
            cg += parse(Int, m[1]); solves += 1
            occursin("[STALL]", line) && (capped += 1)
        end
        m = match(r"Predictor: \|R\| = (\S+) → (\S+)", line)
        if m !== nothing
            occursin("[KEPT]", line) ? (kept += 1) : (reverted += 1)
            b, a = tryparse(Float64, m[1]), tryparse(Float64, m[2])
            b !== nothing && a !== nothing && push!(red, b / a)
        end
        occursin("Predictor: tangent solve failed", line) && (skipped += 1)
        occursin("AMG hierarchy build #", line) && (builds += 1)
    end
    return (; newton, cg, solves, capped, kept, reverted, skipped, builds), red
end

function run_case(element, case, d)
    tag = "$element-$case-h$d" * (DEVICE[] == "cpu" ? "" : "-" * DEVICE[]) * TAG[]
    dir = joinpath(RUNS, tag)
    mkpath(dir)
    deckfile = joinpath(dir, "punch.yaml")
    logfile = first(splitext(deckfile)) * ".log"
    open(io -> write(io, deck(element, case, d, joinpath(dir, "punch.e"))), deckfile, "w")
    punch = node_set(tet15_file(d), "punch")
    backend = backend_of(DEVICE[])
    Carina.open_log_file(deckfile)
    t0 = time()
    rows = Tuple{Int, Float64, Float64, Float64, Int, Int, Int, Int}[]
    status = "completed"
    steps = cuts = 0
    try
        sim = Base.invokelatest(Carina.create_simulation,
                                Carina.YAML.load_file(deckfile; dicttype = Dict{String, Any}), dir;
                                backend = backend)
        (; params, post_processor, controller) = sim
        growth = Ref(2.0)
        n_steps = controller.num_stops - 1
        for k in 1:n_steps
            tk = time()
            Base.invokelatest(Carina._advance_controller!, controller)
            params.times.time_current = controller.prev_time
            s, c = Base.invokelatest(subcycle!, sim, controller.time, growth)
            steps += s; cuts += c
            Base.invokelatest(Carina.write_output!, sim, k + 1)
            F = Base.invokelatest(internal_force, sim)
            fz = SYMMETRY * sum(F[3, n] for n in punch)
            cnt, _ = count_log(logfile)
            push!(rows, (k, 2.0e-4 * controller.time, -fz, time() - tk, s, c, cnt.newton, cnt.cg))
            @printf("%s step %2d t = %.3f force %.6e N  steps %d cuts %d  wall %.1f s\n",
                    tag, k, controller.time, -fz, s, c, time() - tk)
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
        println(io, "# full punch force: $SYMMETRY x the sum of the z reactions on the footprint nodes of the quarter")
        println(io, "step\tdepth\tforce")
        println(io, "0\t0.000000e+00\t0.000000000e+00")
        for r in rows
            @printf(io, "%d\t%.6e\t%.9e\n", r[1], r[2], r[3])
        end
    end
    cnt, red = count_log(logfile)
    open(joinpath(dir, "run.log"), "w") do io
        println(io, "run\t$tag")
        println(io, "host\t$(gethostname())")
        println(io, "device\t$(DEVICE[] == "cpu" ? "cpu" : device_name())\nthreads\t$(Threads.nthreads())")
        println(io, "preconditioner\t$(PRECOND[])" * (PRECOND[] == "amg" ? " (operator $(AMG_OP[]))" : "") * "\ncg cap\t$(CG_CAP[])\nforcing\teisenstat-walker (floor 1e-8)\npredictor\ttangent")
        println(io, "first step\t$FIRST\nmaximum step\t$(max_step(d))")
        println(io, "status\t$status\nwall_s\t$(round(wall; digits = 1))")
        println(io, "time steps\t$steps\ncuts\t$cuts")
        println(io, "newton iterations\t$(cnt.newton)\ncg iterations\t$(cnt.cg)")
        @printf(io, "cg solves\t%d\ncg solves at the cap\t%d (%.1f%%)\n", cnt.solves, cnt.capped,
                cnt.solves == 0 ? 0.0 : 100 * cnt.capped / cnt.solves)
        println(io, "amg hierarchy builds\t$(cnt.builds)")
        println(io, "predictor kept\t$(cnt.kept)\npredictor reverted\t$(cnt.reverted)\npredictor solve failed\t$(cnt.skipped)")
        if !isempty(red)
            s = sort(red)
            @printf(io, "predictor residual reduction (before/after)\tmin %.3e  median %.3e  max %.3e\n",
                    s[1], s[(length(s) + 1) ÷ 2], s[end])
        end
        println(io, "output_step\tt\tforce_N\twall_s\tsteps\tcuts\tnewton_cumulative\tcg_cumulative")
        for r in rows
            @printf(io, "%d\t%.3f\t%.6e\t%.1f\t%d\t%d\t%d\t%d\n", r[1], r[2] / 2.0e-4, r[3], r[4], r[5], r[6], r[7], r[8])
        end
    end
    @printf("%s: %s, %.1f s\n", tag, status, wall)
end

device_name() = DEVICE[] == "cuda" ? string(Base.invokelatest(() -> Main.CUDA.name(Main.CUDA.device()))) : DEVICE[]

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
        elseif a == "--amg-operator"
            AMG_OP[] = args[i + 1]
        elseif a == "--cg-cap"
            CG_CAP[] = parse(Int, args[i + 1])
        elseif a == "--tag"
            TAG[] = args[i + 1]
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
                run_case(el, case, d)
            end
        end
    end
end

main(ARGS)
