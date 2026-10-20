# Referee round 2, m28: convergence of the penalized eigensolver in gamma at
# N = 6 and 8, uniaxial flow direction, beta = 1.
ENV["PE_NOMAIN"] = "1"
include("plastic_ext.jl")
using Printf
for (label, m, bubble) in (("P2+int+face/P1disc", 4, :full), ("P2/P0", 1, :none)), N in (6, 8)
    coords, conn = mesh_of(N, 2)
    ap = assemble_all(coords, conn, 2, m; bubble, flow = N_AXIAL, beta = 1.0)
    for g in (1e8, 1e9, 1e10)
        lp, vp, nit = kernel_modes(ap, ap.Kdev ./ 2; gamma = g)
        @printf("%-20s N = %d  gamma = %.0e  lam_1 %.6f  r_v,1 %.6f  lam_5 %.6f  (%d it)\n",
                label, N, g, lp[1], dilatation(ap, vp[:, 1]), lp[5], nit)
        flush(stdout)
    end
end
