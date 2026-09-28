"""Pressure on the front face of the deformed Cook's membrane, one panel per
element, for one case and mesh size.

    python3 benchmark/hw3l/cook/render_pressure.py <case> <h> <data-dir> <out.png>

<data-dir> holds faces-h<h>.csv (surface_faces.jl) and, for each element,
<case>-<element>-h<h>-pressure.csv and -displacement.csv (extract_pressure.jl).
Each triangle of the front face is colored by the mean pressure of the
element it belongs to and drawn at its deformed position.  One color scale
(ParaView's Rainbow Uniform)
serves all panels: symmetric about zero, bounded by the 99th percentile of
|p| over all elements of all panels.
"""
import csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap

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

ELEMENTS = [("tet10", "TETRA10\n(Carina)"), ("lcm-tet10", "TETRA10\n(Albany)"),
            ("tet15", "TETRA15\n(pointwise)"), ("tet15-p1", "TETRA15\nlinear projection"),
            ("tet15-p0", "TETRA15\nconstant projection"),
            ("lcm-ct", "Composite tetrahedron\n(Albany)")]


def read(path, cols):
    with open(path) as f:
        r = csv.DictReader(f)
        return [{k: row[k] for k in cols} for row in r]


def main(case, h, d, out):
    faces = read(os.path.join(d, f"faces-h{h}.csv"),
                 ["element", "n1", "n2", "n3", "x1", "y1", "x2", "y2", "x3", "y3"])
    fe = np.array([int(r["element"]) for r in faces])
    fn = np.array([[int(r["n1"]), int(r["n2"]), int(r["n3"])] for r in faces])
    fx = np.array([[[float(r["x1"]), float(r["y1"])], [float(r["x2"]), float(r["y2"])],
                    [float(r["x3"]), float(r["y3"])]] for r in faces])
    panels = []
    for key, label in ELEMENTS:
        pf = os.path.join(d, f"{case}-{key}-h{h}-pressure.csv")
        uf = os.path.join(d, f"{case}-{key}-h{h}-displacement.csv")
        if not (os.path.exists(pf) and os.path.exists(uf)):
            continue
        p = {int(r["element"]): float(r["p"]) for r in read(pf, ["element", "p"])}
        u = {int(r["node"]): (float(r["ux"]), float(r["uy"])) for r in read(uf, ["node", "ux", "uy"])}
        pos = fx + np.array([[u[n] for n in tri] for tri in fn])
        val = np.array([p[e] for e in fe])
        panels.append((label, pos, val))
    allv = np.concatenate([v for _, _, v in panels])
    lim = np.percentile(np.abs(allv), 99)
    allx = np.concatenate([q.reshape(-1, 2) for _, q, _ in panels])
    lo, hi = allx.min(axis=0) - 1, allx.max(axis=0) + 1
    n = len(panels)
    ncol = 3
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 3.3 * (hi[1] - lo[1]) / (hi[0] - lo[0]) * nrow + 1.0), dpi=200)
    axes = np.atleast_1d(axes).ravel()
    for ax, (label, pos, val) in zip(axes, panels):
        pc = PolyCollection(pos, array=val, cmap=RAINBOW_UNIFORM, edgecolors="none")
        pc.set_clim(-lim, lim)
        ax.add_collection(pc)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
        ax.set_aspect("equal"); ax.set_axis_off()
        ax.set_title(label, fontsize=15)
        ax.text(0.0, -0.02, f"p: {val.min():.3g} to {val.max():.3g}", transform=ax.transAxes,
                fontsize=12, color="#555555")
    for ax in axes[n:]:
        ax.set_axis_off()
    cb = fig.colorbar(pc, ax=axes.tolist(), orientation="horizontal", fraction=0.04, pad=0.04, shrink=0.6)
    cb.set_label("pressure  p = −tr σ / 3  (element mean)", fontsize=16)
    cb.ax.tick_params(labelsize=14)
    fig.savefig(out, bbox_inches="tight", transparent=True)
    print(out, "panels:", n, "color limit:", lim)


main(*sys.argv[1:])
