# Choice of a quadrature rule for the consistent mass of the fifteen-node
# tetrahedron, evaluated once in the reference configuration.  The integrand
# N_a N_b is of degree 8.  Candidates: the collapsed (conical) Gauss--Jacobi
# product rules with n points per direction, degree 2n - 1 and n^3 points,
# all weights positive; and the 14-point rule of the element (rank <= 14).
# Reported for the nodal basis on the reference element: zero eigenvalues,
# condition number, largest relative entry error against exact integration,
# row-sum error; and the zero eigenvalues of the assembled scalar mass on
# Freudenthal meshes with N = 2 and 3.
ENV["DUMMY"] = "1"
include("common.jl")
include("element.jl")
using Printf
include("mass14.jl")                       # scalar_mass, report (runs its own main first)

function elem_mass(pts, wts)
    A = nodal_transform(); M = zeros(15, 15)
    for q in eachindex(wts)
        N, _ = enriched_shape(pts[:, q]); Nn = A * N
        M .+= wts[q] .* (Nn * Nn')
    end
    return M
end

println("\nRules for the consistent mass (nodal basis, reference element):")
Mex = elem_mass(tet_rule(15)...)
for (name, (p, w)) in (("14-point, degree 5", keast14()), ("conical n=3, degree 5, 27 points", tet_rule(5)),
                       ("conical n=4, degree 7, 64 points", tet_rule(7)), ("conical n=5, degree 9, 125 points", tet_rule(9)))
    M = elem_mass(p, w); ev = eigvals(Symmetric(M))
    nz = count(<(1e-12 * ev[end]), ev)
    err = maximum(abs.(M - Mex)) / maximum(abs.(Mex))
    rs = maximum(abs.(sum(M, dims = 2) - sum(Mex, dims = 2))) / maximum(abs.(sum(Mex, dims = 2)))
    c = nz > 0 ? Inf : cond(M)
    z = [count(<(1e-12 * eigmax(Matrix(scalar_mass(mesh_of(N, 2)..., p, w)))), eigvals(scalar_mass(mesh_of(N, 2)..., p, w))) for N in (2, 3)]
    @printf("  %-36s zero eigenvalues %d  cond %-8.1f  max entry error %.1e  row-sum error %.1e  assembled zeros N=2,3: %d, %d\n",
            name, nz, c, err, rs, z[1], z[2])
end
