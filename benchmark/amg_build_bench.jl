# Splits the host-side GPU AMG hierarchy build into stages.
#
# Section 4 of benchmark_report.md now shows this build is the largest single
# line item in a large GPU AMG run: 12.1 s at 530k DOF, 20.0 s at 823k, 45.5 s
# at 1.57M, against solve phases of 57.7 / 16.3 / 29.4 s.  It is also the same
# host assembly whose COO sparsity pattern is what OOMs the box above ~3M DOF.
#
# What it does not say is WHERE that time goes, and the two candidate
# remediations pull in different directions: shrinking the COO pattern (Int32
# indices, dedup-to-CSR) attacks host assembly and memory, whereas the smoothed
# aggregation setup is a different cost entirely and would be untouched by
# either.  Measure before choosing, as with benchmark/vcycle_bench.jl.
#
# Stages, in the order `_build_gpu_amg_hierarchy!` runs them:
#
#   assemble   -- FEC.assemble_stiffness! on the host assembler
#   extract    -- FEC.stiffness(asm), COO -> SparseMatrixCSC
#   symmetrize -- sparse((K + K')/2)
#   nullspace  -- _rigid_body_modes at the current configuration
#   sa_setup   -- _sa_hierarchy_lowmem (strength, aggregation, slab Galerkin,
#                 then stock AMG.jl on the coarse problem)
#   lambda_max -- host power iteration for the fine smoother bound
#   upload     -- DeviceAMGHierarchy: CSR conversion + device allocation
#
# Usage:  julia --project=bin benchmark/amg_build_bench.jl <case> [nreps]
#         case in {torsion-qs, cube64-qs, cube80-qs}

import AMDGPU
using Carina
import Carina: FEC
import KernelAbstractions as KA
import SparseArrays
import LinearAlgebra
using Printf

AMDGPU.functional() || error("no functional AMD GPU; this benchmark is GPU-only")

const CASE  = length(ARGS) >= 1 ? ARGS[1] : "cube64-qs"
const NREPS = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
const REPO  = normpath(joinpath(@__DIR__, ".."))
const DECK  = joinpath(REPO, "benchmark", "inputs", "$(CASE)-gpu-cg-amg.yaml")
isfile(DECK) || error("no deck at $DECK")

dict = Carina.YAML.load_file(DECK; dicttype = Dict{String, Any})
# Decks reference meshes relative to benchmark/inputs; resolve from there.
dict["input mesh file"]  = normpath(joinpath(REPO, "benchmark", "inputs",
                                             dict["input mesh file"]))
dict["output mesh file"] = tempname() * ".e"
# The hierarchy is built at the first Newton iteration of the first step, so a
# single step is enough and avoids paying for the rest of the ramp.
ti = dict["time integrator"]
ti["final time"] = ti["time step"]

sim = Carina.create_simulation(dict, mktempdir(); backend = AMDGPU.ROCBackend())

asm_cpu = Carina._cpu_asm_ref[]
p_cpu   = Carina._cpu_params_ref[]
backend = Carina._backend_ref[]
pc      = sim.integrator.nonlinear_solver.linear_solver.precond
U_dev   = sim.integrator.U

asm_cpu === nothing && error("CPU assembler reference not set")

# `inv_diag` is normally filled by the per-Newton diagonal refresh before the
# first build; do it here so the upload stage sees a realistic fine level.
Carina.FEC.assemble_diagonal!(sim.integrator.asm, FEC.stiffness, U_dev, sim.params)
d = FEC.diagonal(sim.integrator.asm)
@. pc.inv_diag = 1.0 / max(abs(d), eps(Float64))

U_cpu = Vector{Float64}(Array(U_dev))
FEC._update_for_assembly!(p_cpu, asm_cpu.dof, U_cpu)

nzcount = 0
ncoo    = 0

function stages()
    t = Dict{String, Float64}()
    t["assemble"] = @elapsed FEC.assemble_stiffness!(asm_cpu, FEC.stiffness, U_cpu, p_cpu)
    local K_raw, A, B, ml, dinv, lmax
    t["extract"]    = @elapsed K_raw = FEC.stiffness(asm_cpu)
    t["symmetrize"] = @elapsed A = SparseArrays.sparse((K_raw + K_raw') / 2)
    t["nullspace"]  = @elapsed B = Carina._rigid_body_modes(
        Carina._current_coords(p_cpu), pc.udofs)
    t["sa_setup"]   = @elapsed ml = Carina._sa_hierarchy_lowmem(A, B)
    t["lambda_max"] = @elapsed begin
        dinv = 1.0 ./ Vector(LinearAlgebra.diag(A))
        lmax = Carina._host_lambda_max(A, dinv)
    end
    t["upload"] = @elapsed Carina.DeviceAMGHierarchy(backend, ml, pc.inv_diag, lmax)
    global nzcount = SparseArrays.nnz(A)
    global ncoo    = length(asm_cpu.matrix_pattern.Is)
    return t
end

stages()   # warm-up: compilation, and it is the first build that is timed below
GC.gc(true)

acc = Dict{String, Float64}()
for _ in 1:NREPS
    t = stages()
    for (k, v) in t
        acc[k] = get(acc, k, 0.0) + v / NREPS
    end
    GC.gc(true)
end

order = ["assemble", "extract", "symmetrize", "nullspace",
         "sa_setup", "lambda_max", "upload"]
total = sum(acc[k] for k in order)

@printf("\n===== AMG HIERARCHY BUILD SPLIT: %s =====\n", CASE)
@printf("free DOF %d, COO entries %d, nnz(A) %d (COO/nnz = %.2f), reps %d\n\n",
        length(U_dev), ncoo, nzcount, ncoo / nzcount, NREPS)
for k in order
    @printf("%-12s %8.3f s   %5.1f%%\n", k, acc[k], 100 * acc[k] / total)
end
@printf("%-12s %8.3f s\n", "TOTAL", total)

# The COO pattern is what gates problem size; report what it costs to hold.
coo_gb = ncoo * 8 / 1e9
@printf("\nCOO pattern: %d entries\n", ncoo)
@printf("  Is + Js + unknown_dofs + csrcolval + permutation (Int64) %6.2f GB\n", 5 * coo_gb)
@printf("  csrnzval (Float64)                                       %6.2f GB\n", coo_gb)
@printf("  -> Int32 indices would save                              %6.2f GB\n", 2.5 * coo_gb)
@printf("  -> dedup to CSR (%.2fx) would leave one Int32 map +      %6.2f GB\n",
        ncoo / nzcount, ncoo * 4 / 1e9 + nzcount * 12 / 1e9)
@printf("=========================================\n")
