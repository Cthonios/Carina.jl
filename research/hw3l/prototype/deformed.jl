# The inf-sup constant on a deformed configuration.
#
# Every stability result in beta.jl is taken at F = I, where the pairing of
# the formulation reduces to int p div(psi) dV.  At a finite deformation the
# pairing is int varpi F^-T : Grad(psi) dV = int varpi div_x(psi) dV, the
# divergence pairing on the deformed configuration with the pressure scaled
# by 1/J, and whether beta_h stays bounded below under a large, non-affine
# deformation is what claim 1 of the note leaves open.
#
# The measurement assembles the same three operators as beta.jl on the
# deformed mesh: the H1 seminorm int grad_x N : grad_x N dv, the pairing
# int mu div_x N dv and the pressure mass int mu mu dv, with the pressure
# basis defined on the reference cell as the element uses it.  The deformed
# mesh is the isoparametric image of the P2 mesh under a prescribed map, so
# F is what the element would compute.  Boundary conditions are those of the
# reference cube.  The pressure scaling by 1/J changes beta_h by at most the
# ratio of extreme Jacobians and is not applied.
#
# Four maps on the unit cube, centered at (1/2, 1/2, 1/2):
#   twist 90 / 180   rotation about the z axis by alpha * X3, J = 1
#   shear            x1 = X1 + c X3^2, non-affine, J = 1
#   inflate          x = c + (X - c)(1 + k |X - c|^2), J from 1 to about 2.5
#   compress         x3 = 0.5 X3, affine, J = 0.5 (a control: an affine map
#                    changes beta_h only through the aspect ratio)
#
#   julia --project=. research/hw3l/prototype/deformed.jl

include("common.jl")

const CENTER = [0.5, 0.5, 0.5]

function twist(alpha)
    return X -> begin
        a = alpha * X[3]; c, s = cos(a), sin(a)
        d = X .- CENTER
        CENTER .+ [c * d[1] - s * d[2], s * d[1] + c * d[2], d[3]]
    end
end
shear(c) = X -> [X[1] + c * X[3]^2, X[2], X[3]]
inflate(k) = X -> (d = X .- CENTER; CENTER .+ d .* (1 + k * dot(d, d)))
compress(l) = X -> [X[1], X[2], l * X[3]]

const MAPS = (("reference",  X -> X),
              ("twist 90",   twist(pi / 2)),
              ("twist 180",  twist(pi)),
              ("shear",      shear(0.5)),
              ("inflate",    inflate(0.3)),
              ("compress",   compress(0.5)))

const PAIRS = (("P2/P1c (TH)",        _P1C, :none, 2:6),
               ("P2/P0",              1,    :none, 2:6),
               ("P2+face/P0",         1,    :face, 2:6),
               ("P2+int+face/P1disc", 4,    :full, 2:5))

"beta_h from the assembled operators, as in beta.jl."
function beta_from(a, m)
    F  = cholesky(Symmetric(a.Kh1))
    Gt = Matrix(a.G')
    Z  = Matrix(F.L \ Gt[F.p, :])
    W  = m == 1 ? Z ./ Vector(sqrt.(diag(a.M)))' :
                  Matrix(Z / cholesky(Symmetric(Matrix(a.M))).U)
    sv = sort(svdvals(W))
    r  = rank(Matrix(a.G); rtol = 1e-10)
    idx = length(sv) - r + 1
    return (1 <= idx <= length(sv)) ? sv[idx] : NaN, a.np - r
end

"Extreme Jacobians of the map over the P2 mesh's quadrature points."
function jacobian_range(coords_ref, coords_def, conn)
    el = ref_element(2)
    qpts, _ = tet_rule(2)
    lo, hi = Inf, -Inf
    for e in axes(conn, 2), q in axes(qpts, 2)
        dN = RFE.shape_function_gradient(el, qpts[:, q])
        J = det(coords_def[:, conn[:, e]] * dN) / det(coords_ref[:, conn[:, e]] * dN)
        lo = min(lo, J); hi = max(hi, J)
    end
    return lo, hi
end

# `julia deformed.jl extend` runs only the Crouzeix-Raviart pair at N = 6 on
# the three non-affine isochoric maps, where the local rate at N = 5 had not
# yet flattened; the rows are appended to the main table's output.
function extend()
    println("Crouzeix-Raviart pair at N = 6, non-affine maps (extension of the table above)")
    println("  ", rpad("map", 11), rpad("J min", 8), rpad("J max", 8), rpad("N=6", 10), "null")
    for (mname, map) in MAPS
        mname in ("twist 90", "twist 180", "shear") || continue
        coords, conn = mesh_of(6, 2)
        def = reduce(hcat, (map(coords[:, i]) for i in axes(coords, 2)))
        jr = jacobian_range(coords, def, conn)
        a = assemble_all(def, conn, 2, 4; bubble = :full, bc_coords = coords)
        b, nn = beta_from(a, 4)
        println("  ", rpad(mname, 11), rpad(string(round(jr[1], sigdigits = 3)), 8),
                rpad(string(round(jr[2], sigdigits = 3)), 8),
                rpad(string(round(b, sigdigits = 4)), 10), nn)
        flush(stdout)
    end
end

function main()
    if "extend" in ARGS
        extend(); return
    end
    println("beta_h on deformed configurations: the divergence pairing assembled on")
    println("the isoparametric image of the P2 mesh, boundary conditions of the")
    println("reference cube, whole boundary fixed.  `null` is n_p - rank G and must")
    println("be 1.  The reference row is beta.jl's measurement, repeated here.\n")
    for (label, m, bubble, Ns) in PAIRS
        println(label)
        println("  ", rpad("map", 11), rpad("J min", 8), rpad("J max", 8),
                join((rpad("N=$N", 10) for N in Ns)), "null")
        for (mname, map) in MAPS
            row = String[]; nulls = Int[]; jr = (1.0, 1.0)
            for N in Ns
                coords, conn = mesh_of(N, 2)
                nvert = m == _P1C ? (N + 1)^3 : 0
                def = reduce(hcat, (map(coords[:, i]) for i in axes(coords, 2)))
                jr = jacobian_range(coords, def, conn)
                # A coarse P2 mesh can fold under a large map even though the
                # map itself is a diffeomorphism; that row is reported as
                # inverted rather than ending the sweep.
                if jr[1] <= 0
                    push!(row, rpad("inverted", 10)); continue
                end
                a = assemble_all(def, conn, 2, m; bubble, nvert, bc_coords = coords)
                b, nn = beta_from(a, m)
                push!(row, rpad(string(round(b, sigdigits = 4)), 10)); push!(nulls, nn)
            end
            println("  ", rpad(mname, 11), rpad(string(round(jr[1], sigdigits = 3)), 8),
                    rpad(string(round(jr[2], sigdigits = 3)), 8), join(row),
                    join(unique(nulls), ","))
            flush(stdout)
        end
        println()
    end
end

main()
