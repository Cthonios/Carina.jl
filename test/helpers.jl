using Statistics

# GPU vendor packages are test-only extras (Project.toml [targets].test):
# present under `Pkg.test()`, absent under a plain `julia --project=. test/runtests.jl`.
# Load them when available so the GPU verification can run; otherwise CPU-only.
try; @eval import CUDA;   catch; end
try; @eval import AMDGPU; catch; end

const _TEST_CUDA   = isdefined(@__MODULE__, :CUDA)   && CUDA.functional()
const _TEST_AMDGPU = isdefined(@__MODULE__, :AMDGPU) && AMDGPU.functional()

# Best available compute backend for tests.  Replaces Carina.best_device(),
# which moved to the bin/carina launcher when CUDA/AMDGPU left the library.
function test_best_device()
    _TEST_AMDGPU && return AMDGPU.ROCBackend()
    _TEST_CUDA   && return CUDA.CUDABackend()
    return Carina.KA.CPU()
end

# Copy a file from the examples tree into a temp directory, following symlinks
# so that mesh files (which are symlinks into examples/meshes/) are copied as
# real files rather than broken relative symlinks.
cp_example(src, dst) = cp(src, dst; follow_symlinks=true)

# ---------------------------------------------------------------------------
# Field extraction helpers for Carina test assertions.
#
# Carina's H1Field stores displacement as a flat vector laid out as:
#   [ux_1, uy_1, uz_1,  ux_2, uy_2, uz_2,  ...]
# so component i (1-based) is at data[i:NF:end].
#
# All helpers accept a SingleDomainSimulation and work on the final state of
# params.field, which is always a CPU array in test runs.
# ---------------------------------------------------------------------------

function _field_matrix(sim::Carina.SingleDomainSimulation)
    d  = sim.params.field.data
    NF = 3   # number of displacement components (3D)
    return reshape(d, NF, :)   # NF × n_nodes
end

"""
    average_components(sim) -> Vector{Float64}

Return the mean displacement in each spatial direction [ux, uy, uz],
averaged over all nodes.
"""
function average_components(sim::Carina.SingleDomainSimulation)
    u = _field_matrix(sim)
    return [mean(u[i, :]) for i in 1:size(u, 1)]
end

"""
    maximum_components(sim) -> Vector{Float64}

Return the maximum displacement in each direction.
"""
function maximum_components(sim::Carina.SingleDomainSimulation)
    u = _field_matrix(sim)
    return [maximum(u[i, :]) for i in 1:size(u, 1)]
end

"""
    minimum_components(sim) -> Vector{Float64}

Return the minimum displacement in each direction.
"""
function minimum_components(sim::Carina.SingleDomainSimulation)
    u = _field_matrix(sim)
    return [minimum(u[i, :]) for i in 1:size(u, 1)]
end

"""
    maximum_magnitude(sim) -> Float64

Return the maximum displacement magnitude over all nodes:
    max_i  sqrt(ux_i² + uy_i² + uz_i²)
"""
function maximum_magnitude(sim::Carina.SingleDomainSimulation)
    u = _field_matrix(sim)   # 3 × n_nodes
    return maximum(sqrt.(sum(u.^2, dims=1)))
end
