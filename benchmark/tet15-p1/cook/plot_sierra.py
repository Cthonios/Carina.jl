# Figures of the single-code (Sierra/SM) Cook's membrane results of the
# TET15-P1 paper, drawn from the values of its tables tab:cook and
# tab:cookpressure (handoff of 2026-11-01); the limit band 8.476 to 8.496 of
# the text is drawn in the elastic panel; Okabe-Ito colors (colorblind-safe):
#   cook-tip.png             tip displacement against the number of elements,
#                            five elements, nu = 0.4999 and nu = 0.29
#   cook-pressure-range.png  smallest and largest pressure on the face z = 10
#                            at h = 2, nu = 0.4999, at the quadrature points
#                            that enter the internal force and as element means
# Usage: python3 plot_sierra.py [output directory]   (default: this directory)
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else HERE

ELEMENTS = [327, 1860, 14475]          # h = 8, 4, 2

# element: (color, marker, tip displacement at nu = 0.4999, at nu = 0.29)
TIP = {
    "TET10":     ("#E69F00", "s", (8.2062, 8.3278, 8.4222), (0.2695, 0.2713, 0.2722)),
    "CT10":      ("#009E73", "^", (8.4589, 8.4635, 8.4763), (0.2712, 0.2721, 0.2725)),
    "TET15":     ("#CC79A7", "v", (8.2278, 8.3511, 8.4315), (0.2701, 0.2718, 0.2723)),
    "TET15-SR4": ("#D55E00", "D", (8.4115, 8.4549, 8.4719), (0.2706, 0.2721, 0.2725)),
    "TET15-P1":  ("#0072B2", "o", (8.4139, 8.4555, 8.4722), (0.2707, 0.2722, 0.2725)),
}

# element: (points min, points max, element means min, element means max)
PRESSURE = {
    "TET10":     (-150.3, 134.5, -12.21, 37.21),
    "CT10":      (-9.59, 33.56, -9.59, 33.56),
    "TET15":     (-205.7, 320.1, -8.71, 36.18),
    "TET15-SR4": (-9.48, 44.32, -7.43, 26.04),
    "TET15-P1":  (-18.3, 53.67, -7.43, 26.97),
}

plt.rcParams.update({"font.size": 11})

fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.0, 4.2))
for name, (color, marker, elastic, plastic) in TIP.items():
    ax.plot(ELEMENTS, elastic, color=color, marker=marker, label=name)
    bx.plot(ELEMENTS, plastic, color=color, marker=marker, label=name)
ax.axhspan(8.476, 8.496, color="0.88", linewidth=0, label="limit, between 8.476 and 8.496")
for axis, title in ((ax, "ν = 0.4999, elastic"), (bx, "ν = 0.29, elastoplastic")):
    axis.set_xscale("log")
    axis.set_xlabel("number of elements")
    axis.set_ylabel("tip displacement")
    axis.set_title(title, fontsize=11)
ax.legend(frameon=False, fontsize=9, loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "cook-tip.png"), dpi=200)

fig, ax = plt.subplots(figsize=(6.0, 4.2))
names = list(PRESSURE)
x = range(len(names))
w = 0.18
for i, name in enumerate(names):
    pmin, pmax, mmin, mmax = PRESSURE[name]
    ax.bar(i - w / 2 - 0.02, pmax - pmin, width=w, bottom=pmin, color="#0072B2",
           label="quadrature points" if i == 0 else None)
    ax.bar(i + w / 2 + 0.02, mmax - mmin, width=w, bottom=mmin, color="#E69F00",
           label="element means" if i == 0 else None)
ax.axhline(0.0, color="0.4", linewidth=0.8)
ax.set_ylim(-230, 350)
ax.set_xticks(list(x))
ax.set_xticklabels(names)
ax.set_ylabel("pressure on the face z = 10")
ax.legend(frameon=False, fontsize=9, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "cook-pressure-range.png"), dpi=200)
