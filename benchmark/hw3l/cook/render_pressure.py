"""Pressure on the front face (z = t) of the deformed Cook's membrane.

    python3 benchmark/hw3l/cook/render_pressure.py <data-dir> <out.png> <limit> <ncol> \
        <label>=<case>-<element>-h<h> ...

One panel per argument <label>=<run> (split at the last "="), in the order given, <ncol> panels per
row; "\n" in a label breaks the line.  <data-dir> holds faces-h<h>.csv
(surface_faces.jl) and, per run, <run>-pressure_nodes.csv and
<run>-displacement.csv (extract_pressure.jl).  The pressure inside each
element is the polynomial fitted to its quadrature-point values, given by its
values at the ten nodes of a quadratic tetrahedron; on each triangle of the
front face it is evaluated at the vertices of 16 subtriangles and interpolated
linearly between them.  Triangles are drawn at their deformed position.
<limit> bounds the color scale, symmetric about zero; "auto:<case>" takes the
99th percentile of |p| over the node values of every run of <case> in
<data-dir>, so that all figures of one case share one scale.  Color map:
ParaView's Rainbow Uniform.
"""
import csv, glob, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.tri import Triangulation

# ParaView's "Rainbow Uniform" preset: (position, r, g, b), exported with
# pvpython (ApplyPreset on a color transfer function, RGBPoints normalized).
RAINBOW_UNIFORM = LinearSegmentedColormap.from_list("rainbow_uniform", [(x, (r, g, b)) for x, r, g, b in [
    (0.0000, 0.0200, 0.3813, 0.9981),
    (0.0238, 0.0200, 0.4243, 0.9691),
    (0.0476, 0.0200, 0.4672, 0.9400),
    (0.0714, 0.0200, 0.5102, 0.9110),
    (0.0952, 0.0200, 0.5464, 0.8727),
    (0.1190, 0.0200, 0.5826, 0.8343),
    (0.1429, 0.0200, 0.6188, 0.7960),
    (0.1667, 0.0200, 0.6525, 0.7498),
    (0.1905, 0.0200, 0.6863, 0.7036),
    (0.2143, 0.0200, 0.7200, 0.6574),
    (0.2381, 0.0200, 0.7570, 0.6037),
    (0.2619, 0.0200, 0.7941, 0.5501),
    (0.2857, 0.0200, 0.8311, 0.4964),
    (0.3095, 0.0214, 0.8645, 0.4286),
    (0.3333, 0.0233, 0.8980, 0.3607),
    (0.3571, 0.0160, 0.9310, 0.2926),
    (0.3810, 0.2742, 0.9526, 0.1536),
    (0.4048, 0.4934, 0.9619, 0.1112),
    (0.4286, 0.6439, 0.9773, 0.0469),
    (0.4524, 0.7624, 0.9847, 0.0346),
    (0.4762, 0.8809, 0.9920, 0.0223),
    (0.5000, 0.9995, 0.9995, 0.0135),
    (0.5238, 0.9994, 0.9550, 0.0791),
    (0.5476, 0.9994, 0.9107, 0.1481),
    (0.5714, 0.9994, 0.8663, 0.2172),
    (0.5952, 0.9993, 0.8180, 0.2172),
    (0.6190, 0.9991, 0.7698, 0.2172),
    (0.6429, 0.9990, 0.7215, 0.2172),
    (0.6667, 0.9991, 0.6734, 0.2172),
    (0.6905, 0.9993, 0.6254, 0.2172),
    (0.7143, 0.9994, 0.5773, 0.2172),
    (0.7381, 0.9994, 0.5211, 0.2172),
    (0.7619, 0.9994, 0.4648, 0.2172),
    (0.7857, 0.9994, 0.4086, 0.2172),
    (0.8095, 0.9948, 0.3318, 0.2112),
    (0.8333, 0.9867, 0.2595, 0.1901),
    (0.8571, 0.9912, 0.1480, 0.2108),
    (0.8810, 0.9499, 0.1169, 0.2529),
    (0.9048, 0.9032, 0.0784, 0.2918),
    (0.9286, 0.8565, 0.0400, 0.3307),
    (0.9524, 0.7989, 0.0433, 0.3584),
    (0.9762, 0.7413, 0.0467, 0.3862),
    (1.0000, 0.6837, 0.0500, 0.4139)
]])

# Local node of the quadratic tetrahedron at the midpoint of each edge.
MID = {(1, 2): 5, (2, 3): 6, (3, 1): 7, (1, 4): 8, (2, 4): 9, (3, 4): 10}
MID.update({(b, a): k for (a, b), k in list(MID.items())})
N_SUB = 4


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def subdivision(n):
    """Barycentric points (λ1, λ2, λ3) of the n-fold subdivision of a triangle,
    and its subtriangles as index triples."""
    idx, pts = {}, []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            idx[(i, j)] = len(pts)
            pts.append(((n - i - j) / n, i / n, j / n))
    tris = []
    for i in range(n):
        for j in range(n - i):
            tris.append((idx[(i, j)], idx[(i + 1, j)], idx[(i, j + 1)]))
            if i + j < n - 1:
                tris.append((idx[(i + 1, j)], idx[(i + 1, j + 1)], idx[(i, j + 1)]))
    return np.array(pts), np.array(tris)


def quadratic_on_triangle(lam):
    """Quadratic Lagrange basis of a triangle at barycentric points: vertex
    functions 1-3, then edge midpoints 1-2, 2-3, 3-1."""
    l1, l2, l3 = lam.T
    return np.stack([l1 * (2 * l1 - 1), l2 * (2 * l2 - 1), l3 * (2 * l3 - 1),
                     4 * l1 * l2, 4 * l2 * l3, 4 * l3 * l1], axis=1)


def panel(d, run, faces, lam, tris, phi):
    nodes = {int(r["element"]): ([int(r[f"n{k}"]) for k in range(1, 5)],
                                 [float(r[f"p{k}"]) for k in range(1, 11)])
             for r in read(os.path.join(d, f"{run}-pressure_nodes.csv"))}
    u = {int(r["node"]): (float(r["ux"]), float(r["uy"]))
         for r in read(os.path.join(d, f"{run}-displacement.csv"))}
    xs, vs, ts = [], [], []
    for f in faces:
        e = int(f["element"])
        ids, p = nodes[e]
        loc = [ids.index(int(f[f"n{k}"])) + 1 for k in (1, 2, 3)]
        six = [p[loc[0] - 1], p[loc[1] - 1], p[loc[2] - 1],
               p[MID[(loc[0], loc[1])] - 1], p[MID[(loc[1], loc[2])] - 1],
               p[MID[(loc[2], loc[0])] - 1]]
        X = np.array([[float(f[f"x{k}"]) + u[int(f[f"n{k}"])][0],
                       float(f[f"y{k}"]) + u[int(f[f"n{k}"])][1]] for k in (1, 2, 3)])
        ts.append(tris + len(xs) * len(lam))
        xs.append(lam @ X)
        vs.append(phi @ np.array(six))
    return np.vstack(xs), np.concatenate(vs), np.vstack(ts)


def main(d, out, limit, ncol, *specs):
    lam, tris = subdivision(N_SUB)
    phi = quadratic_on_triangle(lam)
    runs = [(lab.replace("\\n", "\n"), run) for lab, run in (s.rsplit("=", 1) for s in specs)]
    faces = {}
    panels = []
    for lab, run in runs:
        h = run.rsplit("-h", 1)[1]
        if h not in faces:
            faces[h] = read(os.path.join(d, f"faces-h{h}.csv"))
        panels.append((lab,) + panel(d, run, faces[h], lam, tris, phi))
    if limit.startswith("auto:"):
        case = limit[5:]
        vals = np.concatenate([[float(r[f"p{k}"]) for r in read(g) for k in range(1, 11)]
                               for g in glob.glob(os.path.join(d, f"{case}-*-pressure_nodes.csv"))])
        lim = np.percentile(np.abs(vals), 99)
    else:
        lim = float(limit)
    ncol = int(ncol)
    allx = np.vstack([x for _, x, _, _ in panels])
    lo, hi = allx.min(axis=0) - 1, allx.max(axis=0) + 1
    n = len(panels)
    nrow = (n + ncol - 1) // ncol
    w = 3.3 if ncol >= 3 else 4.5
    fig, axes = plt.subplots(nrow, ncol, figsize=(w * ncol, w * (hi[1] - lo[1]) / (hi[0] - lo[0]) * nrow + 1.0), dpi=200)
    axes = np.atleast_1d(axes).ravel()
    for ax, (lab, x, v, t) in zip(axes, panels):
        tc = ax.tripcolor(Triangulation(x[:, 0], x[:, 1], t), v, shading="gouraud",
                          cmap=RAINBOW_UNIFORM, vmin=-lim, vmax=lim)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
        ax.set_aspect("equal"); ax.set_axis_off()
        ax.set_title(lab, fontsize=15)
        print(f"  {lab!r}: front face p from {v.min():.4g} to {v.max():.4g}")
    for ax in axes[n:]:
        ax.set_axis_off()
    cb = fig.colorbar(tc, ax=axes.tolist(), orientation="horizontal", fraction=0.04, pad=0.04, shrink=0.6)
    cb.set_label("pressure  p = −tr σ / 3", fontsize=16)
    cb.ax.tick_params(labelsize=14)
    fig.savefig(out, bbox_inches="tight", transparent=True)
    print(out, "panels:", n, "color limit:", lim)


main(*sys.argv[1:])
