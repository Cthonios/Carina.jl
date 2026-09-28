"""Pressure on the front face of the deformed Cook's membrane, one panel per
element, for one case and mesh size.

    python3 benchmark/hw3l/cook/render_pressure.py <case> <h> <data-dir> <out.png>

<data-dir> holds faces-h<h>.csv (surface_faces.jl) and, for each element,
<case>-<element>-h<h>-pressure.csv and -displacement.csv (extract_pressure.jl).
Each triangle of the front face is colored by the mean pressure of the
element it belongs to and drawn at its deformed position.  One color scale
serves all panels: symmetric about zero, bounded by the 99th percentile of
|p| over all elements of all panels.
"""
import csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

ELEMENTS = [("tet10", "TETRA10 (Carina)"), ("lcm-tet10", "TETRA10 (Albany)"),
            ("tet15", "TETRA15"), ("tet15-p1", "TETRA15, linear projection"),
            ("tet15-p0", "TETRA15, constant projection"),
            ("lcm-ct", "Composite tet. (Albany)")]


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
        pc = PolyCollection(pos, array=val, cmap="RdBu_r", edgecolors="none")
        pc.set_clim(-lim, lim)
        ax.add_collection(pc)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
        ax.set_aspect("equal"); ax.set_axis_off()
        ax.set_title(label, fontsize=12)
        ax.text(0.0, -0.02, f"front face: p from {val.min():.3g} to {val.max():.3g}", transform=ax.transAxes,
                fontsize=8, color="#555555")
    for ax in axes[n:]:
        ax.set_axis_off()
    cb = fig.colorbar(pc, ax=axes.tolist(), orientation="horizontal", fraction=0.04, pad=0.04, shrink=0.6)
    cb.set_label("pressure  p = −tr σ / 3  (element mean)", fontsize=11)
    fig.savefig(out, bbox_inches="tight", transparent=True)
    print(out, "panels:", n, "color limit:", lim)


main(*sys.argv[1:])
