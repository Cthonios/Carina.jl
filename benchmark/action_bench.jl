# Follow-on to action_bench.jl.  The atomic ablation came back flat, so this
# separates the two remaining explanations for the action's cost:
#
#   stiffness_action -- gather + deformation gradient + constitutive tangent
#                       + B^T C B v, then scatter
#   mass_action      -- gather + N^T N v, then scatter
#
# Identical connectivity gather, identical 24-DOF scatter, identical element
# count and launch geometry.  The ONLY difference is arithmetic per element.
# If mass is much cheaper, the kernel is FP64-compute-bound (this part runs
# FP64 at 1/32 rate) and the bandwidth roofline in benchmark_report.md §4 is
# the wrong yardstick.  If the two are close, the cost is in the memory
# movement and the bandwidth framing stands.
#
# Usage:  julia --project=bin benchmark/action_bench.jl [nreps]

import AMDGPU
using Carina
import Carina: FEC
import KernelAbstractions as KA
using Printf
using Statistics
using Random

AMDGPU.functional() || error("no functional AMD GPU; this benchmark is GPU-only")

const NREPS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 50
const REPO  = normpath(joinpath(@__DIR__, ".."))
const DECK  = joinpath(REPO, "benchmark", "inputs", "torsion-qs-gpu-cg-jacobi.yaml")

dict = Carina.YAML.load_file(DECK; dicttype = Dict{String, Any})
dict["input mesh file"]  = joinpath(REPO, "examples", "meshes", "torsion", "torsion.g")
dict["output mesh file"] = tempname() * ".e"

sim = Carina.create_simulation(dict, mktempdir(); backend = AMDGPU.ROCBackend())

ig, asm, p = sim.integrator, sim.integrator.asm, sim.params
U = ig.U; n = length(U)
fspace  = FEC.function_space(asm.dof)
nblocks = length(fspace.ref_fes)
nelems  = sum(size(FEC.block_view(p.state_old, b), 3) for b in 1:nblocks)
backend = KA.get_backend(asm.stiffness_action_storage.data)

v = similar(U)
copyto!(v, 1.0e-6 .* randn(MersenneTwister(20260806), n))

function timeit(action, label)
    for _ in 1:5
        action()
    end
    KA.synchronize(backend)
    ts = Float64[]
    for _ in 1:NREPS
        t0 = time_ns()
        action()
        KA.synchronize(backend)
        push!(ts, (time_ns() - t0) * 1e-9)
    end
    tmed = median(ts)
    chk  = sum(abs, Array(asm.stiffness_action_storage.data))
    @printf("%-18s med %8.3f ms   min %8.3f ms   %8.2f ns/elem   %6.2f GB/s   chk %.6e\n",
            label, tmed * 1e3, minimum(ts) * 1e3,
            tmed * 1e9 / nelems, 856.0 * nelems / tmed / 1e9, chk)
    return tmed
end

@printf("\n===== ACTION COMPARISON =====\n")
@printf("elements %d, free DOF %d, reps %d\n\n", nelems, n, NREPS)

t_stiff = timeit(() -> FEC.assemble_matrix_free_action!(asm, FEC.stiffness_action, U, v, p),
                 "stiffness_action")
t_mass  = timeit(() -> FEC.assemble_matrix_free_action!(asm, FEC.mass_action, U, v, p),
                 "mass_action")

@printf("\nstiffness / mass  = %.2fx\n", t_stiff / t_mass)
@printf("arithmetic-attributable share of stiffness = %.1f%%\n",
        100 * (t_stiff - t_mass) / t_stiff)
@printf("=============================\n")
