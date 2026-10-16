# Assemble the renders of the punch benchmark into grids, rows the elements
# and columns the mesh levels, with one color bar, in the style of
# benchmark/tet15-p1/taylor/compose_pressure.py:
#
#   python3 compose_fields.py <dir> <quantity> <view> <case> <lo> <hi> <cmap.json> <out.png>
#        [<key>=<label>,...] [--bar-crop <pixels>]
#
# <dir>/punch-<key>-<case>-h<h>-<quantity>[-nodal]-<view>.png, with key the
# code, element and model, e.g. sierra-ct-a0.1-hypo, sierra-tet15-p1-hypo,
# carina-tet15-p1-carina, carina-tetra15-carina; quantity p or eqps, or
# p-nodal or eqps-nodal for the projected nodal fields; view meridian or
# top; case A or B; h in {4, 8}.  <cmap.json> holds the RGB points of the
# ParaView preset (x, r, g, b, ...).  A level without a render is marked
# "not run".  --bar-crop removes a strip of that width from the right edge
# of each render, for renders that carry their own color bar.
import os
import sys

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from PIL import Image

args = [a for a in sys.argv[1:]]
bar_crop = 0
suffix = ""
if "--bar-crop" in args:
    i = args.index("--bar-crop"); bar_crop = int(args[i + 1]); del args[i:i + 2]
if "--suffix" in args:        # e.g. -deck for the tiles rendered without a color bar
    i = args.index("--suffix"); suffix = args[i + 1]; del args[i:i + 2]
transpose = False             # rows the levels and columns the elements (slides)
if "--transpose" in args:
    args.remove("--transpose"); transpose = True
punch = False                 # draw the punch on the tiles (see `draw_punch`)
if "--punch" in args:
    args.remove("--punch"); punch = True
levels = None
if "--levels" in args:
    i = args.index("--levels"); levels = tuple(args[i + 1].split(",")); del args[i:i + 2]
d, quantity, view, case, lo, hi, cmap_file, out = args[:8]
lo, hi = float(lo), float(hi)
pts = np.array(json.load(open(cmap_file))).reshape(-1, 4)
x = (pts[:, 0] - pts[0, 0]) / (pts[-1, 0] - pts[0, 0])
cmap = LinearSegmentedColormap.from_list("preset", list(zip(x, pts[:, 1:])))

LEVELS = ("4", "8")           # mesh levels a/h, in order; --levels 2,4,8,16 overrides
if levels is not None:
    LEVELS = levels
ROWS = (("sierra-ct-a0.1-hypo", "Sierra/SM CT10\nα = 0.1"),
        ("sierra-ct-a0-hypo", "Sierra/SM CT10\nα = 0"),
        ("sierra-tet15-p1-hypo", "Sierra/SM TET15-P1"),
        ("carina-tet15-p1-carina", "Carina TET15-P1"),
        ("sierra-tetra15-hypo", "Sierra/SM TETRA15"))
if len(args) > 8:
    ROWS = tuple(tuple(kv.split("=", 1)) for kv in args[8].split(","))
ELEMENTS = {"2": 944, "4": 6187, "8": 38818, "16": 277530}
NODES = {"tetra10": {"2": 1566, "4": 9146, "8": 54582, "16": 380564},
         "tetra15": {"2": 4537, "4": 28129, "8": 172391, "16": 1218149}}
mesh_of = lambda key: "tetra10" if "-ct-" in key else "tetra15"
thin = lambda n: f"{n:,}".replace(",", " ") if n >= 10000 else str(n)
LABEL = {"p": "p = −tr σ / 3 (MPa)", "eqps": "equivalent plastic strain",
         "p-nodal": "p = −tr σ / 3 (MPa), nodal projection",
         "eqps-nodal": "equivalent plastic strain, nodal projection"}


# The punch, from the fixed camera of the tiles (render.py): parallel
# projection, half-height 1.05a, 1400 × 1387 px.  Meridian view: focal point
# (a, 0, H − 0.95a) seen along +y, so x from −0.0598a to 2.0598a across and z
# from H − 2a to H + 0.1a down the tile; the punch is the half-strip
# 0 ≤ x ≤ a above the indented surface z = H − 0.2a at the last output.  Top
# view: focal point (a, a, H) seen along −z, x across and y down the tile,
# the same half-widths; the footprint edge is the quarter circle r = a.
A_PUNCH = 1.0; H_PUNCH = 4.0; DEPTH = 0.2        # in units of a
def draw_punch(im, view):
    from PIL import ImageDraw
    W, Hpx = im.size
    half = 1.05 * W / Hpx                        # horizontal half-width in a
    def px(x): return (x - (A_PUNCH - half)) / (2 * half) * W
    draw = ImageDraw.Draw(im)
    if view == "meridian":
        def py(z): return (H_PUNCH + 0.1 - z) / 2.1 * Hpx
        x0, x1 = px(0.0), px(A_PUNCH)
        y1 = py(H_PUNCH - DEPTH)
        draw.rectangle([x0, -2, x1, y1], fill=(200, 200, 200), outline=(0, 0, 0), width=3)
    elif view == "top":
        # y increases up the tile; PIL measures angles clockwise from 3 o'clock,
        # so the quarter circle in x ≥ 0, y ≥ 0 spans 270° to 360°.
        def py(y): return (A_PUNCH + half - y) / (2 * half) * Hpx
        r = px(A_PUNCH) - px(0.0)
        cx, cy = px(0.0), py(0.0)
        draw.arc([cx - r, cy - r, cx + r, cy + r], 270, 360, fill=(0, 0, 0), width=4)
    return im


def crop(path):
    im = Image.open(path).convert("RGB")
    if punch:
        im = draw_punch(im, view)
    if bar_crop:
        im = im.crop((0, 0, im.size[0] - bar_crop, im.size[1]))
    a = np.asarray(im)
    mask = (a < 250).any(axis=2)
    ys, xs = np.nonzero(mask)
    pad = 8
    return im.crop((max(xs.min() - pad, 0), max(ys.min() - pad, 0),
                    min(xs.max() + pad, a.shape[1]), min(ys.max() + pad, a.shape[0])))


nr, nc = (len(LEVELS), len(ROWS)) if transpose else (len(ROWS), len(LEVELS))
fig, axes = plt.subplots(nr, nc, figsize=(3.0 * nc + 1.4, 3.1 * nr + 0.6) if transpose
                         else (3.4 * nc + 1.6, 3.2 * nr))
axes = np.atleast_2d(axes)
for i, (key, label) in enumerate(ROWS):
    for j, h in enumerate(LEVELS):
        ax = axes[j, i] if transpose else axes[i, j]
        ax.set_axis_off()
        path = os.path.join(d, f"punch-{key}-{case}-h{h}-{quantity}-{view}{suffix}.png")
        if os.path.exists(path):
            ax.imshow(crop(path))
            ax.text(0.5, -0.01, f"{thin(NODES[mesh_of(key)][h])} nodes", ha="center", va="top",
                    fontsize=10, color="0.3", transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "pending", ha="center", va="center", fontsize=13, color="0.5",
                    transform=ax.transAxes)
        if transpose:
            if j == 0:
                ax.set_title(label.replace("\n", ", "), fontsize=12)
            if i == 0:
                ax.text(-0.06, 0.5, f"h = a/{h}\n{thin(ELEMENTS[h])} elements", rotation=90,
                        ha="right", va="center", fontsize=12, transform=ax.transAxes)
        else:
            if i == 0:
                ax.set_title(f"h = a/{h}\n{thin(ELEMENTS[h])} elements", fontsize=12)
            if j == 0:
                ax.text(-0.06, 0.5, label, rotation=90, ha="right", va="center", fontsize=11,
                        transform=ax.transAxes)
if transpose:
    fig.subplots_adjust(left=0.06, right=0.91, top=0.92, bottom=0.04, wspace=0.05, hspace=0.14)
    cax = fig.add_axes([0.935, 0.12, 0.012, 0.72])
else:
    fig.subplots_adjust(left=0.09, right=0.86, top=0.93, bottom=0.03, wspace=0.04, hspace=0.16)
    cax = fig.add_axes([0.89, 0.12, 0.02, 0.72])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(lo, hi), cmap=cmap), cax=cax)
cb.set_label(LABEL.get(quantity, quantity), fontsize=12)
if quantity.startswith("p"):
    ticks = np.arange(np.ceil(lo / 1e6) * 1e6, hi + 1, 1e6)
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{t / 1e6:.1f}" for t in ticks])
fig.savefig(out, dpi=150)
print("written", out)
