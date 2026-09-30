# Assemble the renders of render_pressure.py into two grids, rows the
# elements and columns the mesh levels, with one color bar:
#
#   python3 compose_pressure.py <dir> <p min> <p max> <cmap.json> <out prefix>
#
# <dir>/<key>/pressure-{side,face}.png for key ct-h<h> (composite
# tetrahedron) and tet15-p1-h<h>; <cmap.json> the RGB points of ParaView's
# Rainbow Uniform (x, r, g, b, ... with x in [0, 1]).  Writes
# <out prefix>-side.png and <out prefix>-face.png.
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from PIL import Image

d, lo, hi, cmap_file, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4], sys.argv[5]
pts = np.array(json.load(open(cmap_file))).reshape(-1, 4)
x = (pts[:, 0] - pts[0, 0]) / (pts[-1, 0] - pts[0, 0])
cmap = LinearSegmentedColormap.from_list("rainbow_uniform", list(zip(x, pts[:, 1:])))

LEVELS = ("1.5", "0.75", "0.38", "0.19")
ROWS = (("ct", "composite tetrahedron"), ("tet15-p1", "TET15-P1"))


def crop(path):
    im = Image.open(path).convert("RGB")
    a = np.asarray(im)
    mask = (a < 250).any(axis=2)
    ys, xs = np.nonzero(mask)
    pad = 8
    return im.crop((max(xs.min() - pad, 0), max(ys.min() - pad, 0),
                    min(xs.max() + pad, a.shape[1]), min(ys.max() + pad, a.shape[0])))


for view in ("side", "face"):
    fig, axes = plt.subplots(2, 4, figsize=(13.0, 6.4 if view == "side" else 7.0))
    for i, (key, label) in enumerate(ROWS):
        for j, h in enumerate(LEVELS):
            ax = axes[i, j]
            ax.set_axis_off()
            path = os.path.join(d, f"{key}-h{h}", f"pressure-{view}.png")
            if os.path.exists(path):
                ax.imshow(crop(path))
            else:
                ax.text(0.5, 0.5, "not run", ha="center", va="center", fontsize=13, color="0.5",
                        transform=ax.transAxes)
            if i == 0:
                ax.set_title(f"h = {h} mm", fontsize=14)
        axes[i, 0].text(-0.06, 0.5, label, rotation=90, ha="right", va="center", fontsize=14,
                        transform=axes[i, 0].transAxes)
    fig.subplots_adjust(left=0.05, right=0.9, top=0.93, bottom=0.02, wspace=0.04, hspace=0.06)
    cax = fig.add_axes([0.925, 0.12, 0.015, 0.72])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(lo, hi), cmap=cmap), cax=cax)
    cb.set_label("p = −tr σ / 3 (GPa)", fontsize=12)
    ticks = np.arange(np.ceil(lo / 2e8) * 2e8, hi + 1, 2e8)
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{t / 1e9:.1f}" for t in ticks])
    fig.savefig(f"{out}-{view}.png", dpi=150)
