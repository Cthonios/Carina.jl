# CLI launcher for Carina.
#
# This script — not the Carina library — owns the GPU vendor packages
# (CUDA.jl, AMDGPU.jl).  It runs in its own environment (bin/Project.toml),
# selects a KernelAbstractions backend, and hands it to Carina.run.  This is
# what keeps the Carina library itself free of CUDA/AMDGPU dependencies.
#
# Invoked by the `bin/carina` shell wrapper:
#   carina <input.yaml> [--device cpu|cuda|rocm|auto]

import CUDA
import AMDGPU
import KernelAbstractions as KA
import YAML
import Carina

# Best available compute backend: ROCm > CUDA > CPU.
function best_device()
    try
        AMDGPU.functional() && return AMDGPU.ROCBackend()
    catch
    end
    try
        CUDA.functional() && return CUDA.CUDABackend()
    catch
    end
    return KA.CPU()
end

# Resolve a device string to a KernelAbstractions backend.
function resolve_backend(device::AbstractString)
    s = lowercase(strip(device))
    if s == "cpu"
        return KA.CPU()
    elseif s == "auto"
        return best_device()
    elseif s == "cuda"
        CUDA.functional() || error("--device cuda: no functional NVIDIA GPU found.")
        return CUDA.CUDABackend()
    elseif s == "rocm"
        AMDGPU.functional() || error("--device rocm: no functional AMD GPU found.")
        return AMDGPU.ROCBackend()
    else
        error("Unknown device \"$device\". Expected cpu, cuda, rocm, or auto.")
    end
end

const _USAGE = "Usage: carina <input.yaml> [--device cpu|cuda|rocm|auto] [--threads N]"

function parse_cli(args)
    yaml_file = nothing
    device    = nothing
    i = 1
    while i <= length(args)
        if args[i] == "--device" && i < length(args)
            device = args[i + 1]
            i += 2
        elseif yaml_file === nothing && !startswith(args[i], "-")
            yaml_file = args[i]
            i += 1
        else
            println(stderr, _USAGE)
            exit(1)
        end
    end
    if yaml_file === nothing
        println(stderr, _USAGE)
        exit(1)
    end
    return yaml_file, device
end

function main(args)
    yaml_file, device = parse_cli(args)
    # Device precedence: --device flag > YAML `device:` key > cpu.
    if device === nothing
        dict   = YAML.load_file(yaml_file; dicttype = Dict{String,Any})
        device = lowercase(get(dict, "device", "cpu"))
    end
    Carina.run(yaml_file; backend = resolve_backend(device))
end

main(ARGS)
