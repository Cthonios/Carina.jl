# Assembled SpMV vs matrix-free action, on CPU, at several thread counts.
#
# benchmark/crosscode/README.md §3 measured Carina's CPU implicit path
# plateauing at 1.25x over 24 threads, against 9.9x for the explicit kernel
# (benchmark_report.md §5).  The two paths differ in one line --
# `input_parsing.jl:977`, `assembled = backend isa KA.CPU` -- so CPU runs apply
# the operator as `SparseArrays.mul!` on an assembled K_eff, which is
# single-threaded (measured: 0.033 s at both 1 and 24 threads on 43M nonzeros),
# while GPU and explicit runs use the matrix-free element loop.
#
# That says the present path cannot scale.  It does NOT say matrix-free wins:
# a matrix-free application does far more arithmetic than an SpMV does, and on
# CPU it has to beat a 33 ms memory-bound kernel from a standing start.  This
# measures both operators at the same linearization point so the crossover is a
# number rather than an assumption.
#
# Both operators are exact and interchangeable -- the same K_eff = K + c_M*M
# applied to the same vector -- so the agreement check at the end is also a
# correctness check on the matrix-free path.
#
# Usage:  julia --project=bin -t <threads> benchmark/cpu_operator_bench.jl [nreps]

using Carina
import Carina: FEC
import KernelAbstractions as KA
using LinearAlgebra
using SparseArrays
using Printf
using Statistics
using Random

const NREPS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 20
const REPO  = normpath(joinpath(@__DIR__, ".."))
const DECK  = joinpath(REPO, "benchmark", "inputs",
                       "torsion-newmark-cpu-cg-jacobi.yaml")

dict = Carina.YAML.load_file(DECK; dicttype = Dict{String, Any})
dict["input mesh file"]  = joinpath(REPO, "examples", "meshes", "torsion", "torsion.g")
dict["output mesh file"] = tempname() * ".e"
# One step is enough: the operator is what is being timed, not the trajectory.
dict["time integrator"]["final time"] = dict["time integrator"]["time step"]

sim = Carina.create_simulation(dict, mktempdir(); backend = KA.CPU())
ig  = sim.integrator
asm = ig.asm
p   = sim.params
ls  = ig.nonlinear_solver.linear_solver

# Advance into the step so c_M and the linearization point are the real ones.
Carina.predict!(ig, p)
Carina.evaluate!(ig, p)
Carina.setup_jacobian!(ig, p)

Uu  = Carina._displacement(ig)
c_M = ig.c_M
n   = length(Uu)

# Assembled operator, exactly as the CPU path builds it: setup_jacobian! has
# already folded K_eff = K + c_M*M into stiffness_storage.
K = FEC.stiffness(asm)
@printf("\n===== CPU OPERATOR COMPARISON =====\n")
@printf("free DOF %d, nnz(K_eff) %d, c_M %.3e, threads %d, reps %d\n\n",
        n, nnz(K), c_M, Threads.nthreads(), NREPS)

v = similar(Uu); copyto!(v, 1.0e-6 .* randn(MersenneTwister(20260809), n))
y_spmv = similar(v); fill!(y_spmv, 0.0)
y_mf   = similar(v); fill!(y_mf, 0.0)

# Matrix-free counterpart.  `_action_scratch!` sizes the all-DOF buffer the
# action path needs; `ls.scratch` is free-DOF sized and would throw.
sc = Carina._action_scratch!(ls, asm)

function timeit(f, label)
    for _ in 1:3; f(); end
    ts = Float64[]
    for _ in 1:NREPS
        t0 = time_ns(); f(); push!(ts, (time_ns() - t0) * 1e-9)
    end
    t = median(ts)
    @printf("%-22s med %8.2f ms   min %8.2f ms\n", label, t * 1e3, minimum(ts) * 1e3)
    return t
end

t_spmv = timeit(() -> mul!(y_spmv, K, v), "assembled SpMV")
t_mf   = timeit(() -> Carina._eff_stiffness_matvec!(y_mf, v, asm, Uu, c_M, p, sc),
                "matrix-free action")

# Same operator, so the two results must agree; this is what makes the timing
# comparison meaningful rather than a comparison of two different things.
rel = norm(y_mf - y_spmv) / norm(y_spmv)
@printf("\nagreement |mf - spmv| / |spmv| = %.3e %s\n", rel,
        rel < 1e-10 ? "(same operator)" : "*** MISMATCH ***")

@printf("\nmatrix-free / SpMV = %.2fx  (%s)\n", t_mf / t_spmv,
        t_mf < t_spmv ? "matrix-free wins" : "SpMV wins")
# benchmark_report.md §3 records 2,558 CG iterations over 4 steps for this
# variant: ~640 applications per step, which is what turns a per-application
# difference into a per-step one.
@printf("at ~640 CG iters/step: SpMV %6.1f s/step   matrix-free %6.1f s/step\n",
        640 * t_spmv, 640 * t_mf)
@printf("===================================\n")
