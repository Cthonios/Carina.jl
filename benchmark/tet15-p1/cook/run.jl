# Cook's membrane: the projected TETRA15 element of Carina against the
# composite tetrahedron of Albany-LCM, on the same TETRA10 meshes.
#
# Geometry and loading: cook.jou.  The membrane is clamped on x = 0 and
# loaded by a uniform shear traction q in y on x = 48, applied over ten
# equal steps (in Albany the traction component is the natural continuation
# parameter of LOCA, with the step halved on a failed solve (Failed Step
# Reduction Factor) and grown back after successful ones (Aggressiveness,
# the growth rate: with 0 the step never grows back and the h = 4 elastic
# run needed 1280 steps), as in the LCM ACE tests).  Near incompressibility makes the residual grow quadratically
# with the volumetric strain of a Newton step, so Carina's steps are cut in
# half on a failed solve (down to 1/1000 of the nominal step) and grown back
# after a successful one.  With κ of order 1e5 the assembled residual has a
# rounding floor near 1e-8, so the residual tolerances are 1e-6 absolute and
# 1e-8 relative in both codes.  Two materials, both Simo-Hughes J2 in both codes:
#   elastic:  E = 240.565, ν = 0.4999, no yield (σ_y = 1e10), q = 6.25
#             (the near-incompressible case: E and ν of the plane-strain
#             benchmark, q = F/16 with F = 100 per unit thickness)
#   plastic:  E = 206.9, ν = 0.29, σ_y = 0.45, K = 0.12924, q = 0.14
#             (the material of the elastoplastic case of Simo and Armero,
#             1992; their traction F/16 = 0.1125 leaves the three-dimensional
#             membrane with free faces almost elastic, the pointwise TETRA10
#             collapses between 0.16 and 0.20, and Albany's composite
#             tetrahedron stops converging at 0.158, so the traction is 0.14)
# Reported: the mean and the maximum of u_y over the nodes of the loaded face
# at full load, per element and mesh size h.
#
# Elements:
#   tet10        Carina, TETRA10, pointwise (locks)
#   tet15        Carina, TETRA15, pointwise
#   tet15-p1     Carina, TET15-P1: TETRA15 mesh, volumetric projection linear
#   tet15-p0     Carina, TET15-P0: TETRA15 mesh, volumetric projection constant
#   lcm-tet10    Albany-LCM, TETRA10, pointwise
#   lcm-ct       Albany-LCM, composite tetrahedron with volume-averaged J and
#                pressure (Use Composite Tet 10, Weighted Volume Average J,
#                Volume Average Pressure)
#
# Usage (from the Carina root; the threads serve Carina's element loops):
#   julia -t 12 --project=. benchmark/tet15-p1/cook/run.jl [--h 8,4] [--cases elastic,plastic]
#        [--elements tet10,tet15,tet15-p1,tet15-p0,lcm-tet10,lcm-ct] [--stress] [--no-line-search] [--report]
# Results are appended to results.tsv; --report writes RESULTS.md from it.

using Carina
using Exodus
using Printf
using Statistics
using YAML

const DIR    = @__DIR__
const MESHES = joinpath(DIR, "meshes")
const RUNS   = joinpath(DIR, "runs")
const CUBIT  = "/usr/local/cubit/cubit"
# Host-specific paths and settings; the environment variables override them
# on another host (Rigel):
#   COOK_ALBANY         Albany executable
#   COOK_DECOMP         SEACAS decomp script
#   COOK_MPI_BIN        directory of mpirun;  COOK_MPI_LIB  its libraries
#   COOK_RANKS_ELASTIC  MPI ranks for Albany in the elastic case (default 1)
#   COOK_RANKS_PLASTIC  MPI ranks for Albany in the plastic case (default 12)
#   COOK_ALBANY_DIRECT  Amesos2 solver type for Albany on any rank count,
#                       e.g. SuperLU_DIST or MUMPS; without it, one rank uses
#                       KLU2 and several ranks use GMRES with MueLu
const ALBANY = expanduser(get(ENV, "COOK_ALBANY", "~/LCM/lcm-build-serial-gcc-release/src/Albany"))
# SEACAS decomp (nem_slice + nem_spread), built from ~/Repos/seacas without
# Ioss, and the OpenMPI of the system (module mpi/openmpi-x86_64).
const DECOMP = expanduser(get(ENV, "COOK_DECOMP", "~/LCM/seacas-tools/bin/decomp"))
const MPI_BIN = get(ENV, "COOK_MPI_BIN", "/usr/lib64/openmpi/bin")
const MPI_LIB = get(ENV, "COOK_MPI_LIB", "/usr/lib64/openmpi/lib")
# MPI ranks for Albany in the plastic case.  The elastic case runs Albany on
# one rank with the KLU2 direct solver: KLU2 fails on a distributed matrix in
# this build, and the multigrid-preconditioned GMRES degrades Newton at
# ν = 0.4999 (207 iterations against 76 for the coarse mesh) so that eight
# ranks are slower than one.  In the plastic case (ν = 0.29) GMRES with the
# multigrid preconditioner keeps Newton unchanged and eight ranks were six
# times faster than the serial direct solver.
const ALBANY_RANKS = Dict("elastic" => parse(Int, get(ENV, "COOK_RANKS_ELASTIC", "1")),
                          "plastic" => parse(Int, get(ENV, "COOK_RANKS_PLASTIC", "12")))
# --stress writes the Cauchy stress at the quadrature points (Carina: frames
# every tenth of the load; Albany: every continuation step), for the pressure
# figures.  Off by default: at h = 2 the output grows by gigabytes.
const STRESS = Ref(false)
# --no-line-search turns off Carina's backtracking line search.  The line
# search requires the residual norm to decrease; at ν = 0.4999 a converging
# Newton step first raises it by about three orders of magnitude (κ times the
# square of the volume error the step leaves), so the search cuts every step
# to α ≈ 0.004.  At h = 8 without it: 89 Newton iterations instead of 6366
# (TETRA10), same result to nine digits; Albany takes full steps as well.
const LINE_SEARCH = Ref(true)
const THICKNESS = 10.0
const STEPS = 10

const CASES = Dict(
    "elastic" => (E = 240.565, ν = 0.4999, σ_y = 1.0e10, K = 0.0, q = 6.25),
    "plastic" => (E = 206.9,   ν = 0.29,   σ_y = 0.45,   K = 0.12924, q = 0.14),
)

# ---------------------------------------------------------------------------
# Meshes
# ---------------------------------------------------------------------------

function tet10_mesh(h)
    path = joinpath(MESHES, "cook-h$h.g")
    isfile(path) && return path
    mkpath(MESHES)
    Base.run(`$CUBIT -batch -nographics -nojournal -noecho -information off "h=$h" "t=$THICKNESS"
         "out='$path'" $(joinpath(DIR, "cook.jou"))`)
    isfile(path) || error("Cubit did not write $path")
    return path
end

function tet15_mesh(h)
    path = joinpath(MESHES, "cook-h$h-tet15.g")
    isfile(path) || Carina.tetra15_mesh(tet10_mesh(h), path)
    return path
end

# The mesh decomposed into `np` pieces for Albany (files <mesh>.np.k).
function decomposed_mesh(h, np)
    path = tet10_mesh(h)
    # decomp zero-pads the piece index to the width of np: <mesh>.12.00 ... .11
    pieces() = filter(f -> startswith(f, "$(basename(path)).$np."), readdir(MESHES))
    if length(pieces()) != np
        env = copy(ENV); env["ACCESS"] = dirname(dirname(DECOMP))
        env["PATH"] = dirname(DECOMP) * ":" * env["PATH"]
        Base.run(pipeline(Cmd(`$DECOMP -p $np $(basename(path))`; dir = MESHES, env = env);
                     stdout = "$path.decomp.log", stderr = "$path.decomp.log"))
        length(pieces()) == np || error("decomp did not write the $np pieces of $path")
    end
    return path
end

# Node ids of the loaded face and the element count, from the TETRA10 mesh.
# Ids, not indices: Albany renumbers the nodes of its output file, and both
# output files carry a node id map.
function load_nodes(h)
    exo = ExodusDatabase(tet10_mesh(h), "r")
    ids = read_ids(exo, NodeSet)
    names = read_names(exo, NodeSet)
    id = ids[findfirst(==("load_nodes"), names)]
    node_map = Int.(read_id_map(exo, NodeMap))
    nodes = node_map[Int.(NodeSet(exo, id).nodes)]
    n_elems = sum(size(b.conn, 2) for b in read_sets(exo, Block))
    close(exo)
    return nodes, n_elems
end

# ---------------------------------------------------------------------------
# Carina
# ---------------------------------------------------------------------------

function carina_deck(case, element, mesh_file, out_file)
    m = CASES[case]
    projection = element == "tet15-p1" ? "  volumetric projection: linear\n" :
                 element == "tet15-p0" ? "  volumetric projection: constant\n" : ""
    return """
type: single
input mesh file: $mesh_file
output mesh file: $out_file
$(STRESS[] ? "output interval: 0.1\noutput:\n  stress: true" : "output:\n  stress: false")
model:
  type: solid mechanics
$(projection)  material:
    blocks:
      membrane: j2 plasticity
    j2 plasticity:
      elastic modulus: $(m.E)
      Poisson's ratio: $(m.ν)
      density: 1.0
      yield stress: $(m.σ_y)
      hardening modulus: $(m.K)
time integrator:
  type: quasi static
  initial time: 0.0
  final time: 1.0
  time step: $(1.0 / STEPS)
  minimum time step: $(1.0e-3 / STEPS)
  maximum time step: $(1.0 / STEPS)
  decrease factor: 0.5
  increase factor: 1.5
boundary conditions:
  dirichlet:
    - node set: clamp
      component: x
      function: "0.0"
    - node set: clamp
      component: y
      function: "0.0"
    - node set: clamp
      component: z
      function: "0.0"
  neumann:
    - side set: load
      component: y
      function: "$(m.q) * t"
solver:
  type: newton
$(LINE_SEARCH[] ? "" : "  use line search: false\n")  linear solver:
    type: direct
  termination:
    fail when any:
      - maximum iterations: 50
    converge when any:
      - absolute residual: 1.0e-6
      - relative residual: 1.0e-8
"""
end

function run_carina(case, element, h)
    dir = joinpath(RUNS, "$case-$element-h$h")
    mkpath(dir)
    mesh_file = element == "tet10" ? tet10_mesh(h) : tet15_mesh(h)
    out_file = joinpath(dir, "cook.e")
    deck = joinpath(dir, "cook.yaml")
    open(io -> write(io, carina_deck(case, element, mesh_file, out_file)), deck, "w")
    t0 = time()
    sim = Carina.run(deck)
    wall = time() - t0
    n_dofs = length(sim.params_cpu.field.data)
    return [out_file], "displ_y", n_dofs, wall
end

# ---------------------------------------------------------------------------
# Albany-LCM
# ---------------------------------------------------------------------------

function lcm_materials(case, composite::Bool)
    m = CASES[case]
    block_flags = composite ? """
      Weighted Volume Average J: true
      Average J Stabilization Parameter: 0.0
      Volume Average Pressure: true
      Use Composite Tet 10: true
""" : ""
    return """
LCM:
  ElementBlocks:
    membrane:
      material: J2Mat
$(block_flags)  Materials:
    J2Mat:
      Material Model:
        Model Name: J2
      Elastic Modulus:
        Elastic Modulus Type: Constant
        Value: $(m.E)
      Poissons Ratio:
        Poissons Ratio Type: Constant
        Value: $(m.ν)
      Hardening Modulus:
        Hardening Modulus Type: Constant
        Value: $(m.K)
      Yield Strength:
        Yield Strength Type: Constant
        Value: $(m.σ_y)
      Output eqps: true
$(STRESS[] ? "      Output Cauchy Stress: true\n" : "")"""
end

# Stratimikos block: the KLU2 direct solver on one rank, GMRES with a
# smoothed-aggregation multigrid preconditioner (tolerance 1e-10, at which
# Newton is unchanged) on several.
function lcm_linear_solver(np)
    direct = get(ENV, "COOK_ALBANY_DIRECT", np == 1 ? "KLU2" : "")
    !isempty(direct) && return """
              Linear Solver Type: Amesos2
              Linear Solver Types:
                Amesos2:
                  Solver Type: $direct
              Preconditioner Type: None
"""
    return """
              Linear Solver Type: Belos
              Linear Solver Types:
                Belos:
                  VerboseObject:
                    Verbosity Level: none
                  Solver Type: Block GMRES
                  Solver Types:
                    Block GMRES:
                      Output Frequency: 10
                      Output Style: 1
                      Verbosity: 0
                      Flexible Gmres: false
                      Maximum Iterations: 2000
                      Num Blocks: 200
                      Convergence Tolerance: 1.0e-10
              Preconditioner Type: MueLu
              Preconditioner Types:
                MueLu:
                  multigrid algorithm: sa
                  verbosity: none
                  cycle type: V
                  max levels: 4
                  'smoother: type': CHEBYSHEV
                  'smoother: params':
                    'chebyshev: degree': 3
                    'chebyshev: ratio eigenvalue': 30.0
                  'smoother: pre or post': both
                  'coarse: max size': 1500
                  number of equations: 3
"""
end

function lcm_problem(case, mesh_file, out_file, composite::Bool, np::Int)
    m = CASES[case]
    return """
LCM:
  Problem:
    Name: Mechanics 3D
    Solution Method: Continuation
    MaterialDB Filename: materials.yaml
    Dirichlet BCs:
      DBC on NS clamp for DOF X: 0.0
      DBC on NS clamp for DOF Y: 0.0
      DBC on NS clamp for DOF Z: 0.0
    Neumann BCs:
      'NBC on SS load for DOF all set (t_x, t_y, t_z)': [0.0, 0.0, 0.0]
    Parameters:
      Number: 1
      Parameter 0: 'NBC on SS load for DOF all set (t_x, t_y, t_z)[1]'
    Response Functions:
      Number: 1
      Response 0: Solution Average
  Discretization:
    Method: Exodus
    Exodus Input File Name: $mesh_file
    Exodus Output File Name: $out_file
    Use Serial Mesh: false
    Cubature Degree: $(composite ? 3 : 2)
    Exodus Solution Name: disp
    Exodus Residual Name: resid
  Piro:
    LOCA:
      Bifurcation: { }
      Constraints: { }
      Predictor:
        Method: Tangent
      Stepper:
        Continuation Method: Natural
        Initial Value: 0.0
        Continuation Parameter: 'NBC on SS load for DOF all set (t_x, t_y, t_z)[1]'
        Max Steps: 5000
        Max Value: $(m.q)
        Min Value: 0.0
        Compute Eigenvalues: false
      Step Size:
        Method: Adaptive
        Initial Step Size: $(m.q / STEPS)
        Min Step Size: $(m.q / (1000 * STEPS))
        Max Step Size: $(m.q / STEPS)
        Failed Step Reduction Factor: 0.5
        Aggressiveness: 1.0
    NOX:
      Direction:
        Method: Newton
        Newton:
          Linear Solver:
            Tolerance: 1.0e-8
          Forcing Term Method: Constant
          Rescue Bad Newton Solve: true
          Stratimikos Linear Solver:
            NOX Stratimikos Options: { }
            Stratimikos:
$(lcm_linear_solver(np))      Line Search:
        Backtrack:
          Full Step: 1.0
        Method: Backtrack
      Nonlinear Solver: Line Search Based
      Printing:
        Output Precision: 3
        Output Processor: 0
        Output Information:
          Error: true
          Warning: true
          Outer Iteration: true
          Parameters: false
          Details: false
          Linear Solver Details: false
          Stepper Iteration: true
          Stepper Details: false
          Stepper Parameters: false
      Solver Options:
        Status Test Check Type: Complete
      Status Tests:
        Test Type: Combo
        Combo Type: OR
        Number of Tests: 2
        Test 0:
          Test Type: Combo
          Combo Type: AND
          Number of Tests: 2
          Test 0:
            Test Type: NStep
            Number of Nonlinear Iterations: 0
          Test 1:
            Test Type: NormF
            Scale Type: Unscaled
            Tolerance: 1.0e-6
        Test 1:
          Test Type: Combo
          Combo Type: OR
          Number of Tests: 2
          Test 0:
            Test Type: MaxIters
            Maximum Iterations: 30
          Test 1:
            Test Type: FiniteValue
"""
end

function run_lcm(case, element, h)
    composite = element == "lcm-ct"
    np = ALBANY_RANKS[case]
    dir = joinpath(RUNS, "$case-$element-h$h")
    mkpath(dir)
    mesh_file = np == 1 ? tet10_mesh(h) : decomposed_mesh(h, np)
    out_file = joinpath(dir, "cook.e")
    # remove the output of any earlier run, serial (cook.e) or per rank (cook.e.np.k)
    foreach(f -> rm(joinpath(dir, f)), filter(startswith("cook.e"), readdir(dir)))
    open(io -> write(io, lcm_materials(case, composite)), joinpath(dir, "materials.yaml"), "w")
    open(io -> write(io, lcm_problem(case, mesh_file, out_file, composite, np)), joinpath(dir, "cook.yaml"), "w")
    t0 = time()
    log = joinpath(dir, "albany.log")
    env = copy(ENV)
    env["PATH"] = MPI_BIN * ":" * env["PATH"]
    env["LD_LIBRARY_PATH"] = MPI_LIB * ":" * get(env, "LD_LIBRARY_PATH", "")
    cmd = np == 1 ? `$ALBANY cook.yaml` : `$MPI_BIN/mpirun -np $np $ALBANY cook.yaml`
    ok = success(pipeline(Cmd(cmd; dir = dir, env = env); stdout = log, stderr = log))
    wall = time() - t0
    ok || error("Albany failed for $case $element h=$h; see $log")
    # The output holds one record per continuation step; the last one must be
    # at the full traction (LOCA's default, arc-length continuation, stops at
    # Max Steps short of the target when the structure softens).
    final = [parse(Float64, m.captures[1]) for m in eachmatch(r"End of Continuation Step \d+ : Parameter: .* = ([-+0-9.eE]+)", read(log, String))]
    isempty(final) && error("Albany log $log has no continuation record")
    isapprox(final[end], CASES[case].q; rtol = 1e-6) ||
        error("Albany reached traction $(final[end]) of $(CASES[case].q) for $case $element h=$h")
    exo = ExodusDatabase(mesh_file, "r"); n_nodes = exo.init.num_nodes; close(exo)
    # on several ranks Albany writes one file per rank, cook.e.np.k
    files = np == 1 ? [out_file] :
        [joinpath(dir, f) for f in sort(filter(startswith("cook.e.$np."), readdir(dir)))]
    length(files) == np || error("expected $np output files for $case $element h=$h, found $(length(files))")
    return files, "disp_y", 3 * n_nodes, wall
end

# ---------------------------------------------------------------------------
# Measurement and driver
# ---------------------------------------------------------------------------

# `files` are the output files of one run: one file, or one per MPI rank, each
# carrying the global node ids of its nodes in its node map.
function tip_displacement(files, var, node_ids)
    u_of = Dict{Int, Float64}()
    for f in files
        exo = ExodusDatabase(f, "r")
        n_steps = read_number_of_time_steps(exo)
        u = read_values(exo, NodalVariable, n_steps, var)
        for (k, id) in enumerate(Int.(read_id_map(exo, NodeMap)))
            u_of[id] = u[k]
        end
        close(exo)
    end
    vals = [u_of[id] for id in node_ids]
    return mean(vals), maximum(vals)
end

function run_case(case, element, h)
    nodes, n_elems = load_nodes(h)
    files, var, n_dofs, wall = startswith(element, "lcm") ? run_lcm(case, element, h) :
                                                             run_carina(case, element, h)
    u_mean, u_max = tip_displacement(files, var, nodes)
    commit = strip(read(`git -C $DIR rev-parse --short HEAD`, String))
    rec = (case = case, element = element, h = Float64(h), elements = n_elems, dofs = n_dofs,
           u_y_mean = u_mean, u_y_max = u_max, wall = wall, commit = commit)
    path = joinpath(DIR, "results.tsv")
    fresh = !isfile(path)
    open(path, "a") do io
        fresh && println(io, join(string.(keys(rec)), "\t"))
        println(io, join(string.(values(rec)), "\t"))
    end
    @printf("%-8s %-10s h=%-3g elements=%-6d dofs=%-7d u_y mean=%.4f max=%.4f wall=%.1fs\n",
            case, element, h, n_elems, n_dofs, u_mean, u_max, wall)
    return rec
end

function report()
    lines = readlines(joinpath(DIR, "results.tsv"))
    header = split(lines[1], '\t')
    recs = [Dict(zip(header, split(l, '\t'))) for l in lines[2:end] if !isempty(strip(l))]
    latest = Dict{Tuple{String, String, Float64}, Any}()
    for r in recs
        latest[(r["case"], r["element"], parse(Float64, r["h"]))] = r   # last record per cell
    end
    elements = ["tet10", "tet15", "tet15-p1", "tet15-p0", "lcm-tet10", "lcm-ct"]
    open(joinpath(DIR, "RESULTS.md"), "w") do io
        println(io, "# Cook's membrane: mean u_y of the loaded face at full load\n")
        println(io, "Rows: mesh size h (TETRA10 element count in parentheses).  ",
                    "Columns: element (see run.jl).  Each entry is the newest record of ",
                    "results.tsv for that cell.\n")
        for case in ("elastic", "plastic")
            hs = sort(unique([k[3] for k in keys(latest) if k[1] == case]); rev = true)
            isempty(hs) && continue
            m = CASES[case]
            println(io, "## $case: E = $(m.E), ν = $(m.ν), σ_y = $(m.σ_y), K = $(m.K), q = $(m.q)\n")
            println(io, "| h | ", join(elements, " | "), " |")
            println(io, "|---|", join(fill("---", length(elements)), "|"), "|")
            for h in hs
                cells = map(elements) do el
                    r = get(latest, (case, el, h), nothing)
                    r === nothing ? "" : @sprintf("%.4f", parse(Float64, r["u_y_mean"]))
                end
                ne = first(r["elements"] for r in values(latest)
                           if r["case"] == case && parse(Float64, r["h"]) == h)
                println(io, "| $(h) ($(ne)) | ", join(cells, " | "), " |")
            end
            println(io)
        end
    end
    println("wrote RESULTS.md")
end

function main(args)
    opts = Dict("--h" => "8,4", "--cases" => "elastic,plastic",
                "--elements" => "tet10,tet15,tet15-p1,tet15-p0,lcm-tet10,lcm-ct")
    i = 1
    while i <= length(args)
        if args[i] == "--report"
            report(); return
        elseif args[i] == "--stress"
            STRESS[] = true; i += 1; continue
        elseif args[i] == "--no-line-search"
            LINE_SEARCH[] = false; i += 1; continue
        end
        haskey(opts, args[i]) || error("unknown option $(args[i])")
        opts[args[i]] = args[i + 1]; i += 2
    end
    hs = [parse(Float64, s) for s in split(opts["--h"], ",")]
    hs = [isinteger(h) ? Int(h) : h for h in hs]
    for case in split(opts["--cases"], ","), h in hs, element in split(opts["--elements"], ",")
        run_case(String(case), String(element), h)
    end
    report()
end

main(ARGS)
