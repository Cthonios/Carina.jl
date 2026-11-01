# Figure of the single-code (Sierra/SM) Taylor bar results of the TET15-P1
# paper (tab:taylor), from the histories data/sierra-taylor-<element>-h<h>-
# history.tsv (time s, radius m, length m) and data/sierra-ct-h<h>-history.tsv:
#   taylor-sierra.png  left: final radius of the impact face against the
#                      number of elements, five elements; right: radius of the
#                      impact face against time on the finest mesh of each
# Usage: python3 plot_sierra.py [output directory]   (default: this directory)
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = sys.argv[1] if len(sys.argv) > 1 else HERE

ELEMENTS = {1.5: 3522, 0.75: 25029, 0.38: 190727, 0.19: 1521212}
REFERENCE = 7.22   # converged Q1/P0 hexahedron, Foulk et al. (2021)
T_END = 80.0       # us; the CT10 run at h = 1.5 mm stops at 41.6 us and is left out

# element: (file pattern, levels with a complete run, color, marker)
SERIES = [
    ("TET10", "sierra-taylor-tet10-h{}-history.tsv", (1.5, 0.75, 0.38), "C1", "s"),
    ("CT10", "sierra-ct-h{}-history.tsv", (0.75, 0.38, 0.19), "C2", "^"),
    ("TET15", "sierra-taylor-tet15-h{}-history.tsv", (1.5, 0.75, 0.38), "C4", "v"),
    ("TET15-SR4", "sierra-taylor-tet15-sr4-h{}-history.tsv", (1.5, 0.75, 0.38), "C3", "D"),
    ("TET15-P1", "sierra-taylor-tet15-p1-h{}-history.tsv", (1.5, 0.75, 0.38, 0.19), "C0", "o"),
]


def history(name):
    a = np.loadtxt(os.path.join(DATA, name), skiprows=1)
    return 1e6 * a[:, 0], 1e3 * a[:, 1], 1e3 * a[:, 2]


plt.rcParams.update({"font.size": 11})
fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.0, 4.2))
for label, pattern, levels, color, marker in SERIES:
    n, r = [], []
    for h in levels:
        t, rad, length = history(pattern.format(h))
        assert abs(t[-1] - T_END) < 0.5, (label, h, t[-1])
        n.append(ELEMENTS[h]); r.append(rad[-1])
        print(f"{label:10s} h = {h:<5} radius {rad[-1]:.4f} length {length[-1]:.3f}")
    ax.plot(n, r, color=color, marker=marker, label=label)
    t, rad, _ = history(pattern.format(levels[-1]))
    bx.plot(t, rad, color=color, linewidth=1.6, label=f"{label}, h = {levels[-1]} mm")
ax.axhline(REFERENCE, color="0.4", linestyle="--", linewidth=1,
           label="Q1/P0 hexahedron, converged")
ax.set_xscale("log")
ax.set_xlabel("number of elements")
ax.set_ylabel("final radius of the impact face (mm)")
ax.legend(frameon=False, fontsize=9)
bx.axhline(REFERENCE, color="0.4", linestyle="--", linewidth=1,
           label="Q1/P0 hexahedron, final")
bx.set_xlabel("time (μs)")
bx.set_ylabel("radius of the impact face (mm)")
bx.set_xlim(0, T_END)
bx.legend(frameon=False, fontsize=9, loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "taylor-sierra.png"), dpi=200)
