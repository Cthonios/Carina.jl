# Taylor bar impact: TET15-P1 in Carina, on the meshes of the composite
# tetrahedron study of Foulk et al. (2021), Sec. 4.4.
#
# Problem (Foulk et al. 2021; Simo 1992, Sec. 7.2): a copper bar of length
# 32.4 mm and radius 3.2 mm, axis along z, strikes a rigid frictionless wall
# at z = 0 with velocity 227 m/s.  Density 8930 kg/m^3, E = 117 GPa,
# nu = 0.35, J2 plasticity with linear hardening, yield stress 400 MPa,
# hardening modulus 100 MPa.  Explicit central differences to 80 us, no bulk
# viscosity.  The wall is the condition u_z = 0 on the impact face: the face
# is in contact at t = 0 and, with the bar moving toward the wall, remains in
# contact (Simo 1992 constrains nodes that touch the wall to stay there).
#
# Meshes (taylor.jou, Cubit): full bar for h = 1.5, 0.75, 0.38, 0.19 mm;
# quarter bar (x >= 0, y >= 0, symmetry planes x = 0 and y = 0) for
# h = 0.094 and 0.047 mm, as in the paper.  The Cubit size is adjusted until
# the element count is within 3% of the paper's count (Figure 14), since the
# paper's h is not Cubit's size parameter.  Each mesh is smoothed with
# Norma's energetic mesh smoothing (four-node tetrahedra, surface nodes held
# on the cylinder, the end planes and the symmetry planes), converted to
# TETRA10 with the lateral edge midpoints on the cylinder (convert.jl, in m),
# and to TETRA15 (Carina.tetra15_mesh).
#
# Elements (Carina):
#   tet15-p1   TET15-P1, volumetric projection linear
#   tet15-p0   TET15-P0, volumetric projection constant
# The composite tetrahedron runs in Sierra/SM on the TETRA10 meshes (README).
#
# Reported per run (history.tsv in the run directory, results.tsv here): the
# radius of the impact face, max over its nodes of the distance from the
# axis, and the length of the bar, max z of the nodes of the free face, at
# every output time (1 us); the wall time and the number of steps.
#
# Usage (from the Carina root):
#   julia -t 16 --project=. benchmark/tet15-p1/taylor/run.jl --h 1.5,0.75
#        [--stages mesh,smooth,convert,run] [--elements tet15-p1,tet15-p0]
#        [--final-time 8.0e-5] [--cfl 0.8] [--internal-variables]
# Environment: TAYLOR_CUBIT (default /usr/local/cubit/cubit), TAYLOR_NORMA
# (default ~/Repos/Norma.jl/bin/norma), TAYLOR_NORMA_THREADS (default 8).
# --internal-variables also writes the internal variables (eqps) at every
# quadrature point and output time: 844 MB per run at h = 1.5 mm, about
# 370 GB at h = 0.19 mm; without it the output holds the displacement.

using Carina
using Exodus
using Printf

const DIR    = @__DIR__
const MESHES = joinpath(DIR, "meshes")
const SMOOTH = joinpath(DIR, "smooth")
const RUNS   = joinpath(DIR, "runs")
const CUBIT  = get(ENV, "TAYLOR_CUBIT", "/usr/local/cubit/cubit")
const NORMA  = expanduser(get(ENV, "TAYLOR_NORMA", "~/Repos/Norma.jl/bin/norma"))
const NORMA_THREADS = get(ENV, "TAYLOR_NORMA_THREADS", "8")

const R_MM = 3.2
const L_MM = 32.4
# Levels: quarter bar or full, and the paper's element count (Figure 14).
const LEVELS = Dict(1.5 => (false, 3_495), 0.75 => (false, 24_739),
                    0.38 => (false, 187_819), 0.19 => (false, 1_533_115),
                    0.094 => (true, 2_712_078), 0.047 => (true, 20_521_040))

const FINAL_TIME = Ref(8.0e-5)
const CFL = Ref(0.8)
const INTERNAL = Ref(false)

tag(h) = "h$(h)"
tet4_file(h)  = joinpath(MESHES, "taylor-$(tag(h))-tet4.g")
tet10_file(h) = joinpath(MESHES, "taylor-$(tag(h))-tet10.g")
tet15_file(h) = joinpath(MESHES, "taylor-$(tag(h))-tet15.g")
smooth_file(h) = joinpath(SMOOTH, "taylor-$(tag(h))-smooth.e")

function count_tets(path)
    e = ExodusDatabase(path, "r")
    n = sum(size(b.conn, 2) for b in read_sets(e, Block))
    close(e)
    return n
end

# Cubit mesh with the element count of the paper within 3%.
function mesh(h)
    quarter, target = LEVELS[h]
    mkpath(MESHES)
    sizes_file = joinpath(MESHES, "sizes.tsv")
    s = h * 0.9
    n = 0
    for attempt in 1:5
        log = joinpath(MESHES, "cubit-$(tag(h)).log")
        Base.run(pipeline(`$CUBIT -batch -nographics -nojournal -noecho -information off
                           "h=$s" "quarter=$(Int(quarter))" "out='$(tet4_file(h))'"
                           $(joinpath(DIR, "taylor.jou"))`; stdout = log, stderr = log))
        n = count_tets(tet4_file(h))
        @printf("mesh h=%g: Cubit size %.4f -> %d elements (paper %d)\n", h, s, n, target)
        abs(n / target - 1) < 0.03 && break
        s *= (n / target)^(1 / 3)
    end
    open(io -> println(io, "$h\t$s\t$n\t$target"), sizes_file, "a")
end

function smooth_deck(h)
    quarter = LEVELS[h][1]
    surfaces = ["lateral" => "x^2 + y^2 - $(R_MM^2)", "impact" => "z", "free" => "z - $L_MM"]
    quarter && append!(surfaces, ["symx" => "x", "symy" => "y"])
    bcs = join(("    - side set: $s\n      function: \"$f\"\n" for (s, f) in surfaces), "")
    return """
type: single
input mesh file: $(tet4_file(h))
output mesh file: $(smooth_file(h))
model:
  type: mesh smoothing
  smooth reference: max
  material:
    blocks:
      bar: elastic
    elastic:
      model: seth-hill
      m: 2
      n: 2
      bulk modulus: 1.0e+03
      shear modulus: 1.0e+03
      density: 1000.0
time integrator:
  type: quasi static
  initial time: 0.0
  final time: 10.0
  time step: 1.0
boundary conditions:
  Surface:
$bcs
solver:
  type: steepest descent
  step: lbfgs
  memory: 10
  minimum iterations: 1
  maximum iterations: 64
  relative tolerance: 1.0e-12
  absolute tolerance: 1.0e-08
  step length: 1.0e-3
  use line search: true
  line search backtrack factor: 0.5
  line search decrease factor: 1.0e-04
  line search maximum iterations: 16
"""
end

function smooth(h)
    mkpath(SMOOTH)
    deck = joinpath(SMOOTH, "taylor-$(tag(h)).yaml")
    open(io -> write(io, smooth_deck(h)), deck, "w")
    log = joinpath(SMOOTH, "taylor-$(tag(h)).out")
    Base.run(pipeline(Cmd(`$NORMA $deck --threads $NORMA_THREADS`; dir = SMOOTH);
                      stdout = log, stderr = log))
end

function convert(h)
    Base.run(`$(Base.julia_cmd()) --project=$(pkgdir(Carina)) $(joinpath(DIR, "convert.jl"))
              $(tet4_file(h)) $(smooth_file(h)) $(tet10_file(h))`)
    isfile(tet15_file(h)) && rm(tet15_file(h))
    Carina.tetra15_mesh(tet10_file(h), tet15_file(h))
end

function carina_deck(element, h, out_file)
    quarter = LEVELS[h][1]
    projection = element == "tet15-p1" ? "linear" : element == "tet15-p0" ? "constant" :
                 error("unknown element $element")
    sym = quarter ? """
    - node set: symx
      component: x
      function: "0.0"
    - node set: symy
      component: y
      function: "0.0"
""" : ""
    return """
type: single
input mesh file: $(tet15_file(h))
output mesh file: $out_file
output interval: 1.0e-6
output:
  stress: false
  internal variables: $(INTERNAL[])
model:
  type: solid mechanics
  volumetric projection: $projection
  material:
    blocks:
      bar: j2 plasticity
    j2 plasticity:
      elastic modulus: 117.0e9
      Poisson's ratio: 0.35
      density: 8930.0
      yield stress: 400.0e6
      hardening modulus: 100.0e6
time integrator:
  type: central difference
  initial time: 0.0
  final time: $(FINAL_TIME[])
  time step: 1.0e-8
  cfl: $(CFL[])
  stable time step interval: 10
  stable time step method: global
  stable time step eigenvalue interval: 200
initial conditions:
  velocity:
    - node set: all
      component: z
      function: "-227.0"
boundary conditions:
  dirichlet:
    - node set: impact
      component: z
      function: "0.0"
$sym"""
end

include(joinpath(@__DIR__, "history.jl"))

function run_carina(element, h)
    dir = joinpath(RUNS, "$element-$(tag(h))")
    mkpath(dir)
    out_file = joinpath(dir, "taylor.e")
    deck = joinpath(dir, "taylor.yaml")
    open(io -> write(io, carina_deck(element, h, out_file)), deck, "w")
    t0 = time()
    sim = Carina.run(deck)
    wall = time() - t0
    rows = taylor_history(out_file)
    write_history(joinpath(dir, "history.tsv"), rows)
    t, r, len = rows[end]
    commit = strip(read(`git -C $DIR rev-parse --short HEAD`, String))
    rec = (element = element, h = h, elements = count_tets(tet10_file(h)),
           dofs = length(sim.params_cpu.field.data), time = t, radius_mm = 1e3 * r,
           length_mm = 1e3 * len, wall = wall, commit = commit)
    path = joinpath(DIR, "results.tsv")
    fresh = !isfile(path)
    open(path, "a") do io
        fresh && println(io, join(string.(keys(rec)), "\t"))
        println(io, join(string.(values(rec)), "\t"))
    end
    @printf("%-9s h=%-6g t=%.2e radius=%.4f mm length=%.4f mm wall=%.1f s\n",
            element, h, t, 1e3 * r, 1e3 * len, wall)
end

function main(args)
    opts = Dict("--h" => "1.5", "--stages" => "mesh,smooth,convert,run",
                "--elements" => "tet15-p1,tet15-p0", "--final-time" => "8.0e-5", "--cfl" => "0.8")
    i = 1
    while i <= length(args)
        if args[i] == "--internal-variables"
            INTERNAL[] = true; i += 1; continue
        end
        haskey(opts, args[i]) || error("unknown option $(args[i])")
        opts[args[i]] = args[i + 1]; i += 2
    end
    FINAL_TIME[] = parse(Float64, opts["--final-time"])
    CFL[] = parse(Float64, opts["--cfl"])
    stages = split(opts["--stages"], ",")
    all(in(("mesh", "smooth", "convert", "run")), stages) || error("unknown stage in $stages")
    for s in split(opts["--h"], ",")
        h = parse(Float64, s)
        haskey(LEVELS, h) || error("h = $h is not a level of the study: $(sort(collect(keys(LEVELS))))")
        "mesh" in stages && mesh(h)
        "smooth" in stages && smooth(h)
        "convert" in stages && convert(h)
        if "run" in stages
            for el in split(opts["--elements"], ",")
                run_carina(String(el), h)
            end
        end
    end
end

main(ARGS)
