# Figures of the Taylor bar study from the histories in data/:
#   taylor-convergence.png  final radius of the impact face against the
#                           number of elements, for each element
#   taylor-history.png      radius of the impact face against time on the
#                           finest mesh of each element
# Usage: python3 plot.py [output directory]   (default: this directory)
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = sys.argv[1] if len(sys.argv) > 1 else HERE

# Element counts of the meshes (Cubit, then Norma smoothing), per level h in mm.
ELEMENTS = {1.5: 3522, 0.75: 25029, 0.38: 190727, 0.19: 1521212}
REFERENCE = 7.22   # converged Q1/P0 hexahedron, Foulk et al. (2021)

SERIES = [
    ("TET15-P1", "tet15-p1-h{}-history.tsv", (1.5, 0.75, 0.38), "C0", "o"),
    ("TET15-P0", "tet15-p0-h{}-history.tsv", (1.5, 0.75), "C1", "s"),
    ("composite tetrahedron", "sierra-ct-h{}-history.tsv", (0.75, 0.38, 0.19), "C2", "^"),
]


def history(name):
    a = np.loadtxt(os.path.join(DATA, name), skiprows=1)
    return 1e6 * a[:, 0], 1e3 * a[:, 1], 1e3 * a[:, 2]


plt.rcParams.update({"font.size": 11})

fig, ax = plt.subplots(figsize=(6.0, 4.2))
for label, pattern, levels, color, marker in SERIES:
    n = [ELEMENTS[h] for h in levels]
    r = [history(pattern.format(h))[1][-1] for h in levels]
    ax.plot(n, r, color=color, marker=marker, label=label)
# TET15-P0 at h = 0.38 stopped at 78.7 us; its radius at 78 us.
t, r, _ = history("tet15-p0-h0.38-history-to-78us.tsv")
ax.plot([ELEMENTS[0.75], ELEMENTS[0.38]],
        [history("tet15-p0-h0.75-history.tsv")[1][-1], r[-1]],
        color="C1", linestyle=":")
ax.plot([ELEMENTS[0.38]], [r[-1]], color="C1", marker="s", fillstyle="none",
        label="TET15-P0, at 78 μs")
ax.axhline(REFERENCE, color="0.4", linestyle="--", linewidth=1,
           label="Q1/P0 hexahedron, converged")
ax.set_xscale("log")
ax.set_xlabel("number of elements")
ax.set_ylabel("final radius of the impact face (mm)")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "taylor-convergence.png"), dpi=200)

fig, ax = plt.subplots(figsize=(6.0, 4.2))
for label, name, color, style, z in [
        ("TET15-P0, h = 0.38 mm", "tet15-p0-h0.38-history-to-78us.tsv", "C1", "-", 2),
        ("composite tetrahedron, h = 0.38 mm", "sierra-ct-h0.38-history.tsv", "C2", "--", 2),
        ("composite tetrahedron, h = 0.19 mm", "sierra-ct-h0.19-history.tsv", "C2", "-", 2),
        ("TET15-P1, h = 0.38 mm", "tet15-p1-h0.38-history.tsv", "C0", ":", 3)]:
    t, r, _ = history(name)
    ax.plot(t, r, color=color, linestyle=style, linewidth=1.8, zorder=z, label=label)
ax.axhline(REFERENCE, color="0.4", linestyle="--", linewidth=1,
           label="Q1/P0 hexahedron, final")
ax.set_xlabel("time (μs)")
ax.set_ylabel("radius of the impact face (mm)")
ax.set_xlim(0, 80)
ax.legend(frameon=False, fontsize=9, loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "taylor-history.png"), dpi=200)
