# Largest pressure magnitude at a quadrature point, per element, at the output times
# of punch runs: julia --project=. punch_pressure.jl <run>/punch.e ...
using Exodus, Printf
for f in ARGS
    e = ExodusDatabase(f, "r"); ts = read_times(e)
    for (k, t) in enumerate(ts)
        k == 1 && continue
        pm = nothing
        for q in 1:14
            s = sum(read_values(e, ElementVariable, k, 1, "sigma_$(c)_$q") for c in ("xx", "yy", "zz")) ./ 3
            pm = pm === nothing ? abs.(s) : max.(pm, abs.(s))
        end
        o = sortperm(pm; rev = true)
        @printf("%s t=%.4f  max |p|/σy %.2f (e %d), next %.2f %.2f %.2f; elem 1976 %.2f\n", basename(dirname(f)), t, pm[o[1]] / 1e6, o[1], (pm[o[2:4]] ./ 1e6)..., pm[1976] / 1e6)
        k > 6 && break
    end
    close(e)
end
