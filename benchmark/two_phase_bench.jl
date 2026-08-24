# A/B the two matrix-free action assemblies on a GPU:
#
#   fused      FEC's one-thread-per-element kernel (gather, serial qp loop,
#              atomic scatter)
#   two-phase  Carina's per-(element,qp) staging + node-parallel gather
#              (src/two_phase_action.jl)
#
# Same functor (NewmarkAction), same deck, same deformed state; the checksum
# printed for each row must agree to ~1e-12 (summation order differs).
#
# Usage:  julia --project=bin benchmark/two_phase_bench.jl [cuda|rocm] [nreps]

const DEVICE = length(ARGS) >= 1 ? ARGS[1] : "rocm"
DEVICE in ("cuda", "rocm") || error("device must be cuda or rocm, got \"$DEVICE\"")
const NREPS = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 50

if DEVICE == "cuda"
    import CUDA
    CUDA.functional() || error("no functional NVIDIA GPU")
    const BACKEND = CUDA.CUDABackend()
else
    import AMDGPU
    AMDGPU.functional() || error("no functional AMD GPU")
    const BACKEND = AMDGPU.ROCBackend()
end

using Carina
import Carina: FEC
import KernelAbstractions as KA
using Printf
using Statistics
using Random

const REPO = normpath(joinpath(@__DIR__, ".."))
const DECK = joinpath(REPO, "benchmark", "inputs", "torsion-newmark-gpu-cg-jacobi.yaml")

dict = Carina.YAML.load_file(DECK; dicttype = Dict{String, Any})
dict["input mesh file"]  = joinpath(REPO, "examples", "meshes", "torsion", "torsion.g")
dict["output mesh file"] = tempname() * ".e"
dict["output interval"]  = 1.0
# One step deforms the state so the geometric tangent is nonzero.
dict["time integrator"]["final time"] = 5.0e-5

sim = Carina.create_simulation(dict, mktempdir(); backend = BACKEND)
Carina.evolve!(sim)

ig, asm, p = sim.integrator, sim.integrator.asm, sim.params
U = ig.U; n = length(U)
fspace  = FEC.function_space(asm.dof)
nelems  = sum(fspace.elem_conns.nelems[b] for b in 1:length(fspace.ref_fes))
backend = KA.get_backend(asm.stiffness_action_storage.data)

v = similar(U)
copyto!(v, 1.0e-6 .* randn(MersenneTwister(20260824), n))
action = Carina.NewmarkAction(ig.c_M)

function timeit(f, label)
    for _ in 1:5
        f()
    end
    KA.synchronize(backend)
    ts = Float64[]
    for _ in 1:NREPS
        t0 = time_ns()
        f()
        KA.synchronize(backend)
        push!(ts, (time_ns() - t0) * 1e-9)
    end
    tmed = median(ts)
    chk = sum(abs, Array(asm.stiffness_action_storage.data))
    @printf("%-10s med %8.3f ms   min %8.3f ms   %8.2f ns/elem   chk %.15e\n",
            label, tmed * 1e3, minimum(ts) * 1e3, tmed * 1e9 / nelems, chk)
    return tmed
end

@printf("\n===== TWO-PHASE ACTION A/B (%s) =====\n", DEVICE)
@printf("elements %d, DOF %d, reps %d, |U|_max = %.3e\n\n",
        nelems, n, NREPS, maximum(abs, Array(U)))

t_fused = timeit(() -> FEC.assemble_matrix_free_action!(asm, action, U, v, p),
                 "fused")
t_two   = timeit(() -> Carina._assemble_action_two_phase!(asm, action, U, v, p),
                 "two-phase")

@printf("\nfused / two-phase = %.2fx\n", t_fused / t_two)
@printf("=====================================\n")
