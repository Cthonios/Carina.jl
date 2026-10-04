# The backend of a device name for the Taylor scripts (run.jl,
# kernel_timing.jl), resolved as bin/carina.jl does.

# The backend of a device name, as bin/carina.jl resolves it: the vendor
# package must have been loaded by the caller (the Carina library does not
# depend on it), and a ROCm run attaches the workgroup-size bound of
# bin/rocm_workgroup_bound.jl before any kernel is compiled.
function backend_of(device)
    device == "cpu" && return Carina.KA.CPU()
    pkg = device == "rocm" ? :AMDGPU : device == "cuda" ? :CUDA :
          error("unknown device $device; expected cpu, rocm or cuda")
    isdefined(Main, pkg) || error("--device $device needs `using $pkg` in the calling session")
    mod = getfield(Main, pkg)
    mod.functional() || error("--device $device: no functional GPU found")
    if device == "rocm"
        include(joinpath(pkgdir(Carina), "bin", "rocm_workgroup_bound.jl"))
        return mod.ROCBackend()
    end
    return mod.CUDABackend()
end
