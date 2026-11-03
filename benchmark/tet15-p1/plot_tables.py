# Figures of the TET15-P1 paper drawn from the tables of its manuscript, so that
# a figure and its table cannot disagree and pending entries (\tbd, ---) appear
# as the tables are filled:
#
#   python3 plot_tables.py <paper sections directory> [output directory]
#
# Reads sections/{stability,sri,examples}.tex and writes, in the output
# directory (default: the current directory):
#   kappa-error.png     tab:kappa      (a) e_u against N at kappa/mu = 1 and 1e6,
#                                      (b) e_p against N at kappa/mu = 1e4,
#                                      (c) e_u at N = 6 against kappa/mu
#   beta.png            tab:beta, tab:jitter, tab:beta-sri   beta_h against N on
#                                      the Freudenthal, perturbed and Cubit meshes
#   plastic.png         tab:plastic    zeta_1 and r_v of the lowest mode against N
#   dilbound.png        tab:dilbound   zeta_*(0.1) on the four mesh families
#   taylor-pressure.png tab:taylorpmax, tab:taylorprange   largest element
#                                      pressure against elements; range at the points
#   taylor-cost.png     tab:taylorcost, tab:taylorcal   |deviation| against
#                                      element-steps and against estimated time
#   punch-force.png     tab:punchforce punch force against elements, cases A and B
#   punch-jump.png      tab:punchjump  normalized jump statistics (mean, p99)
#   punch-jump-mpa.png  tab:punchjump  p99 of the jump and largest |p| in MPa
# Style: matplotlib, font 11, two panels 11.0 x 4.2 in, three panels 11.0 x 3.6 in,
# one panel 6.0 x 4.2 in, dpi 200; one color and marker per element (Okabe-Ito
# palette, colorblind-safe); pending entries are left out.
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

SECTIONS = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "."

# Elements: color (Okabe-Ito), marker.  Same order everywhere.
ELEM = {
    "TET10":     ("#E69F00", "s"),
    "CT10":      ("#009E73", "^"),
    "TET15":     ("#CC79A7", "v"),
    "TET15-SR4": ("#D55E00", "D"),
    "TET15-P1":  ("#0072B2", "o"),
    "TET15-P0":  ("#56B4E9", "p"),
    "$P_2/P_0$": ("#000000", "x"),
}
# Pairs of the stability section: color by enrichment, line style by pressure space.
ENRICH = {"": "#56B4E9", "bint": "#E69F00", "bface": "#009E73", "P1": "#0072B2", "control": "#000000"}
REF = dict(color="0.4", linestyle="--", linewidth=1)
plt.rcParams.update({"font.size": 11})


# ---------------------------------------------------------------- parsing
HEADERS = {"Pair", "element", "mesh", "map", "configuration", "$h$ (mm)", "$h$ (elements)", "case"}

def tables():
    out = {}
    for f in ("stability.tex", "sri.tex", "examples.tex"):
        s = open(os.path.join(SECTIONS, f)).read()
        for m in re.finditer(r"\\begin\{table\}.*?\\end\{table\}", s, re.S):
            lab = re.search(r"\\label\{(tab:[^}]*)\}", m.group(0)).group(1)
            body = re.search(r"\\begin\{tabular\}\{(?:[^{}]|\{[^{}]*\})*\}(.*?)\\end\{tabular\}", m.group(0), re.S).group(1)
            rows = []
            for line in body.split("\\\\"):
                line = re.sub(r"\\(toprule|midrule|bottomrule|cmidrule\([^)]*\)\{[^}]*\})", "", line).strip()
                if not line or "\\multicolumn" in line or "&" not in line:
                    continue
                cells = [c.strip() for c in line.split("&")]
                if cells[0] in HEADERS or cells[1] in ("pair", "exact", "$J$", "$N$"):
                    continue
                rows.append(cells)
            out[lab] = rows
    return out


def num(cell):
    """A number from a table cell, or None for a pending or absent entry."""
    c = cell.replace("$", "").replace("\\,", "").replace("\\%", "").replace("~", "")
    c = re.sub(r"\^\\(dagger|ast)", "", c).strip()
    if c in ("", "---", "\\tbd") or "pend" in c or "tbd" in c:
        return None
    c = re.sub(r"\s*\(.*\)$", "", c)          # trailing "(4)" or "(-0.003)"
    m = re.match(r"([-+]?[\d.]+)\\times10\^\{([-\d]+)\}", c)
    if m:
        return float(m.group(1)) * 10.0 ** int(m.group(2))
    return float(c)


def rng(cell):
    """(lo, hi) from "a--b", "a" or "a (k)"."""
    c = re.sub(r"\s*\(\d+\)", "", cell).strip()
    if "--" in c:
        a, b = c.split("--")
        return float(a), float(b)
    return float(c), float(c)


def pair_key(name):
    n = name.replace("\\elem", "TET15-P1")
    if "TET15-P1" in n:
        return "P1", "P1", "TET15-P1"
    pressure = "P1c" if "P_1$ cont" in n or "continuous" in n else ("P1d" if "Pdisc" in n else "P0")
    enrich = "bint" if "bint" in n else ("bface" if "bface" in n else "")
    if pressure == "P1c":
        enrich = "control"
    label = {"P0": "P_0", "P1d": "P_1^{\\mathrm{disc}}", "P1c": "P_1\\ \\mathrm{cont.}"}[pressure]
    space = {"": "P_2", "bint": "P_2 \\oplus b_{\\mathrm{int}}", "bface": "P_2 \\oplus b_{\\mathrm{face}}", "control": "P_2"}[enrich]
    return pressure, enrich, f"${space}/{label}$"


PAIR_STYLE = {"P0": "-", "P1d": "--", "P1c": ":", "P1": "-"}
PAIR_MARK = {"": "o", "bint": "s", "bface": "^", "P1": "o", "control": "x"}


def slope_guide(ax, x0, x1, y0, slope, label):
    """A line of the given log-log slope between x0 and x1 starting at y0."""
    y1 = y0 * (x1 / x0) ** slope
    ax.plot([x0, x1], [y0, y1], color="0.4", linewidth=1, linestyle=":")
    ax.annotate(label, (x1, y1), textcoords="offset points", xytext=(3, -2), fontsize=9, color="0.3")


def tag(ax, text):
    ax.text(0.02, 0.97, text, transform=ax.transAxes, ha="left", va="top", fontsize=10)


T = tables()


# ---------------------------------------------------------------- tab:kappa
def fig_kappa():
    rows = T["tab:kappa"]
    data, name = {}, None
    for r in rows:
        if r[0]:
            name = r[0].replace("\\elem", "TET15-P1").replace(", pointwise", "")
            data[name] = []
        data[name].append([num(c) for c in r[1:7]])   # N, eu(1), eu(1e2), eu(1e4), eu(1e6), ep(1e4)
    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(11.0, 3.6))
    for e, rowsN in data.items():
        a = np.array(rowsN, dtype=float)
        col, mk = ELEM[e]
        N = a[:, 0]
        ax.plot(N, a[:, 1], color=col, marker=mk, label=e)
        ax.plot(N, a[:, 4], color=col, marker=mk, linestyle="--", markerfacecolor="white")
        bx.plot(N, a[:, 5], color=col, marker=mk, label=e)
        i6 = np.where(N == 6)[0][0]
        cx.plot([1, 1e2, 1e4, 1e6], a[i6, 1:5], color=col, marker=mk, label=e)
    for axis in (ax, bx):
        axis.set_xscale("log"); axis.set_yscale("log")
        axis.set_xlabel("$N$")
        axis.set_xticks([2, 3, 4, 6, 8, 12]); axis.set_xticklabels(["2", "3", "4", "6", "8", "12"])
        axis.minorticks_off()
    slope_guide(ax, 6, 12, 0.02, -2, "2")
    slope_guide(ax, 6, 12, 0.45, -1, "1")
    ax.set_ylabel("relative $H^1$ error $e_u$")
    ax.set_title("$e_u$, $\\kappa/\\mu = 1$ (solid) and $10^6$ (dashed)", fontsize=11)
    slope_guide(bx, 6, 12, 0.07, -2, "2")
    slope_guide(bx, 6, 12, 0.9, -1, "1")
    bx.set_ylabel("relative $L^2$ error of the mean stress $e_p$")
    bx.set_title("$e_p$, $\\kappa/\\mu = 10^4$", fontsize=11)
    cx.set_xscale("log")
    cx.set_xlabel("$\\kappa/\\mu$")
    cx.set_ylabel("relative $H^1$ error $e_u$")
    cx.set_title("$e_u$ at $N = 6$", fontsize=11)
    cx.set_ylim(0, 0.4)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "kappa-error.png"), dpi=200)


# ---------------------------------------------------------------- tab:beta, tab:jitter, tab:beta-sri
def fig_beta():
    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(11.0, 3.6), sharey=True)
    Ns = [2, 3, 4, 5, 6, 7, 8]
    handles = []
    for r in T["tab:beta"]:
        pressure, enrich, label = pair_key(r[0])
        vals = [num(c) for c in r[1:8]]
        x = [n for n, v in zip(Ns, vals) if v is not None]
        y = [v for v in vals if v is not None]
        h, = ax.plot(x, y, color=ENRICH[enrich], linestyle=PAIR_STYLE[pressure], marker=PAIR_MARK[enrich],
                     markersize=5, label=label)
        handles.append(h)
    # Four-point coupling of TET15-P1 (tab:beta-sri, Freudenthal rows).
    x, y = [], []
    for r in T["tab:beta-sri"]:
        m = re.match(r"Freudenthal \$N = (\d)\$", r[0])
        if m:
            x.append(int(m.group(1))); y.append(num(r[2]))
    h, = ax.plot(x, y, color=ENRICH["P1"], linestyle="--", marker="o", markersize=5, markerfacecolor="white",
                 label="TET15-P1, four-point coupling")
    handles.append(h)
    slope_guide(ax, 5, 8, 0.055, -1, "1")
    ax.set_title("Freudenthal meshes", fontsize=11)
    ax.set_ylabel("inf-sup constant $\\beta_h$")
    # Perturbed and Cubit meshes.
    fam = None
    for r in T["tab:jitter"]:
        if r[0]:
            fam = r[0]
        pressure, enrich, label = pair_key(r[1])
        col = ENRICH[enrich]
        los, his = zip(*(rng(c) for c in r[2:7]))
        x = np.arange(2, 7)
        if fam == "perturbed":
            bx.vlines(x, los, his, color=col, linewidth=1)
            bx.plot(x, (np.array(los) + np.array(his)) / 2, color=col, linestyle=PAIR_STYLE[pressure],
                    marker=PAIR_MARK[enrich], markersize=5)
        else:
            cx.plot(x, los, color=col, linestyle=PAIR_STYLE[pressure], marker=PAIR_MARK[enrich], markersize=5)
    bx.set_title("perturbed meshes, three seeds", fontsize=11)
    cx.set_title("Cubit meshes", fontsize=11)
    for axis, lab in ((ax, "$N$"), (bx, "$N$"), (cx, "$n$")):
        axis.set_xscale("log"); axis.set_yscale("log")
        axis.set_xlabel(lab)
        axis.set_xticks(Ns if axis is ax else Ns[:5]); axis.set_xticklabels([str(n) for n in (Ns if axis is ax else Ns[:5])])
        axis.minorticks_off()
        axis.set_yticks([0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5]); axis.set_yticklabels(["0.02", "0.03", "0.05", "0.1", "0.2", "0.3", "0.5"])
    ax.set_ylim(0.018, 0.55)
    fig.legend(handles=handles, frameon=False, fontsize=8, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(os.path.join(OUT, "beta.png"), dpi=200)


# ---------------------------------------------------------------- tab:plastic
def fig_plastic():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    name = None
    data = {}
    for r in T["tab:plastic"]:
        if r[0]:
            name = r[0]
            data[name] = []
        dag = "dagger" in r[4]
        data[name].append((num(r[1]), num(r[2]), num(r[3]), num(r[4]), num(r[5]), dag))
    handles = []
    for name, rowsN in data.items():
        pressure, enrich, label = pair_key(name)
        col, mk = ENRICH[enrich], PAIR_MARK[enrich]
        a = np.array([x[:5] for x in rowsN], dtype=float)
        dag = np.array([x[5] for x in rowsN])
        h, = ax.plot(a[:, 0], a[:, 1], color=col, marker=mk, markersize=5, label=label)
        handles.append(h)
        ax.plot(a[:, 0], a[:, 3], color=col, marker=mk, markersize=5, linestyle="--", markerfacecolor="white")
        bx.plot(a[:, 0], a[:, 2], color=col, marker=mk, markersize=5)
        bx.plot(a[:, 0], a[:, 4], color=col, marker=mk, markersize=5, linestyle="--", markerfacecolor="white")
        if dag.any():
            ax.plot(a[dag, 0], a[dag, 3], color=col, marker="+", markersize=9, linestyle="none")
            bx.plot(a[dag, 0], a[dag, 4], color=col, marker="+", markersize=9, linestyle="none")
    ax.axhline(0.125, **REF)
    ax.annotate("$\\zeta_\\infty = 1/8$", (8, 0.125), textcoords="offset points", xytext=(-2, 3), fontsize=9,
                color="0.3", ha="right")
    for axis in (ax, bx):
        axis.set_xscale("log"); axis.set_yscale("log")
        axis.set_xlabel("$N$")
        axis.set_xticks([3, 4, 5, 6, 7, 8]); axis.set_xticklabels(["3", "4", "5", "6", "7", "8"])
        axis.minorticks_off()
    ax.set_ylabel("$\\zeta_1$")
    bx.set_ylabel("dilatation $r_v$ of the lowest mode")
    slope_guide(bx, 6, 8, 0.0025, -2, "2")
    styles = [Line2D([], [], color="0.3", linestyle="-", marker="o", markersize=5, label="shear direction"),
              Line2D([], [], color="0.3", linestyle="--", marker="o", markersize=5, markerfacecolor="white",
                     label="uniaxial direction"),
              Line2D([], [], color="0.3", linestyle="none", marker="+", markersize=9, label="not converged")]
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="lower left")
    bx.legend(handles=styles, frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "plastic.png"), dpi=200)


# ---------------------------------------------------------------- tab:dilbound
def fig_dilbound():
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    fam = None
    FAM = {"Freudenthal": "o", "$\\delta = 0.1$": "s", "$\\delta = 0.25$": "D", "Cubit": "^"}
    FAMLABEL = {"Freudenthal": "Freudenthal", "$\\delta = 0.1$": "perturbed, $\\delta = 0.1$",
                "$\\delta = 0.25$": "perturbed, $\\delta = 0.25$", "Cubit": "Cubit"}
    pts = {k: {"P1": [], "P1z1": [], "P0": [], "bface": []} for k in FAM}
    for r in T["tab:dilbound"]:
        if r[0]:
            fam = r[0]
        n = int(re.search(r"(\d+)", r[1]).group(1))
        pts[fam]["P1"].append((n, num(r[4])))
        if num(r[3]) is not None:
            pts[fam]["P1z1"].append((n, num(r[3])))
        pts[fam]["P0"].append((n, num(r[6])))
        pts[fam]["bface"].append((n, num(r[7])))
    for fam, mk in FAM.items():
        for key, col, mfc in (("P1", ENRICH["P1"], None), ("P1z1", ENRICH["P1"], "white"),
                              ("P0", ENRICH[""], None), ("bface", ENRICH["bface"], None)):
            if not pts[fam][key]:
                continue
            a = np.array(pts[fam][key], dtype=float)
            ls = "-" if fam == "Cubit" else "none"
            ax.plot(a[:, 0], a[:, 1], color=col, marker=mk, markersize=6, linestyle=ls, linewidth=1,
                    markerfacecolor=mfc or col)
    ax.axhline(0.125, **REF)
    ax.annotate("$\\zeta_\\infty = 1/8$", (10, 0.125), textcoords="offset points", xytext=(0, 3), fontsize=9,
                color="0.3", ha="right")
    ax.set_xlabel("$N$ (Freudenthal, perturbed) or $n$ (Cubit)")
    ax.set_ylabel("$\\zeta_\\ast(0.1)$; $\\zeta_1$ of TET15-P1 (open)")
    ax.set_ylim(0, 0.2)
    pairs = [Line2D([], [], color=ENRICH["P1"], marker="o", linestyle="none", label="TET15-P1"),
             Line2D([], [], color=ENRICH[""], marker="o", linestyle="none", label="$P_2/P_0$"),
             Line2D([], [], color=ENRICH["bface"], marker="o", linestyle="none",
                    label="$P_2 \\oplus b_{\\mathrm{face}}/P_0$")]
    fams = [Line2D([], [], color="0.3", marker=mk, linestyle="none", label=FAMLABEL[f]) for f, mk in FAM.items()]
    leg = ax.legend(handles=pairs, frameon=False, fontsize=9, loc="upper right")
    ax.add_artist(leg)
    ax.legend(handles=fams, frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "dilbound.png"), dpi=200)


# ---------------------------------------------------------------- Taylor bar
ELEMENTS_TAYLOR = {1.5: 3522, 0.75: 25029, 0.38: 190727, 0.19: 1521212}
TAYLOR_COLS = ["TET10", "CT10", "TET15", "TET15-SR4", "TET15-P1"]


def fig_taylor_pressure():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    pmax = {e: [] for e in TAYLOR_COLS}
    hs = []
    for r in T["tab:taylorpmax"]:
        h = num(r[0]); hs.append(h)
        for e, c in zip(TAYLOR_COLS, r[1:6]):
            pmax[e].append(num(c))
    n = [ELEMENTS_TAYLOR[h] for h in hs]
    for e in TAYLOR_COLS:
        col, mk = ELEM[e]
        ax.plot(n, pmax[e], color=col, marker=mk, label=e)
    ax.set_xscale("log")
    ax.set_xlabel("number of elements")
    ax.set_ylabel("largest element pressure (GPa)")
    ax.set_ylim(0, 1.8)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    # Range at the integration points, grouped by level.
    width = 0.15
    for i, r in enumerate(T["tab:taylorprange"]):
        for j, (e, c) in enumerate(zip(TAYLOR_COLS, r[1:6])):
            if "tbd" in c:
                continue
            lo, hi = (float(v.replace("$", "")) for v in c.split(","))
            col, mk = ELEM[e]
            x = i + (j - 2) * width
            bx.bar(x, hi - lo, bottom=lo, width=width * 0.85, color=col, label=e if i == 0 else None)
            bx.plot([x - width * 0.42, x + width * 0.42], [pmax[e][i]] * 2, color="black", linewidth=1)
    bx.axhline(0, color="0.4", linewidth=0.8)
    bx.set_xticks(range(len(hs))); bx.set_xticklabels([f"$h = {h}$ mm" for h in hs])
    bx.set_ylabel("pressure at the integration points (GPa)")
    bx.set_ylim(top=3.4)
    bx.legend(frameon=False, fontsize=9, loc="upper right", ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "taylor-pressure.png"), dpi=200)


def fig_taylor_cost():
    cal = {}
    for r in T["tab:taylorcal"]:
        e = r[0].replace("\\elem", "TET15-P1").replace("\\CT", "CT10")
        cal[e] = (num(r[1]), num(r[2]))     # us per element-step at 1.5 and 0.75 mm
    data = {e: [] for e in TAYLOR_COLS}
    for r in T["tab:taylorcost"]:
        e = r[0].replace("\\elem", "TET15-P1").replace("\\CT", "CT10")
        if e not in data:
            continue
        h, steps, es, dev = num(r[1]), num(r[3]), num(r[4]), num(r[5])
        if None in (h, es, dev):
            continue
        t = es * 1e-6 * (cal[e][0] if h == 1.5 else cal[e][1])
        data[e].append((h, es, t, dev))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=True)
    for e in TAYLOR_COLS:
        a = np.array(data[e], dtype=float)
        col, mk = ELEM[e]
        for axis, k in ((ax, 1), (bx, 2)):
            axis.plot(a[:, k], np.abs(a[:, 3]), color=col, linewidth=1, label=e)
            pos = a[:, 3] > 0
            axis.plot(a[pos, k], a[pos, 3], color=col, marker=mk, linestyle="none")
            axis.plot(a[~pos, k], -a[~pos, 3], color=col, marker=mk, linestyle="none", markerfacecolor="white")
    for axis in (ax, bx):
        axis.set_xscale("log"); axis.set_yscale("log")
        axis.axhspan(0.001, 0.08, color="0.88", linewidth=0)
        axis.axhline(0.11, **REF)
        axis.axhline(0.3, color="0.4", linestyle=":", linewidth=1)
    ax.set_ylim(0.005, 3)
    for axis in (ax, bx):
        axis.annotate("tolerance 0.3 %", (0.98, 0.3), xycoords=("axes fraction", "data"), textcoords="offset points",
                      xytext=(0, 3), fontsize=9, color="0.3", ha="right")
        axis.annotate("tolerance 0.11 %", (0.98, 0.11), xycoords=("axes fraction", "data"), textcoords="offset points",
                      xytext=(0, 3), fontsize=9, color="0.3", ha="right")
        axis.annotate("uncertainty of the limit, 0.08 %", (0.02, 0.08), xycoords=("axes fraction", "data"),
                      textcoords="offset points", xytext=(0, -11), fontsize=9, color="0.3", ha="left")
    ax.set_yticks([0.01, 0.03, 0.1, 0.3, 1]); ax.set_yticklabels(["0.01", "0.03", "0.1", "0.3", "1"])
    ax.minorticks_off()
    ax.set_xlabel("element-steps")
    bx.set_xlabel("estimated computing time on one core (s)")
    ax.set_ylabel("|deviation| of the final radius from the extrapolation (%)")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "taylor-cost.png"), dpi=200)


# ---------------------------------------------------------------- punch
ELEMENTS_PUNCH = [944, 6187, 38818]
PUNCH_ROWS = [("\\CT, $\\alpha = 0.1$", "CT10", "CT10, $\\alpha = 0.1$", "-", None),
              ("\\CT, $\\alpha = 0$", "CT10", "CT10, $\\alpha = 0$", "--", "white"),
              ("TET10", "TET10", "TET10", "-", None),
              ("TET15", "TET15", "TET15", "-", None),
              ("TET15-SR4", "TET15-SR4", "TET15-SR4", "-", None),
              ("\\elem", "TET15-P1", "TET15-P1", "-", None)]


def punch_series(axis, vals, e, ls, mfc, label=None, **kw):
    col, mk = ELEM[e]
    x = [n for n, v in zip(ELEMENTS_PUNCH, vals) if v is not None]
    y = [v for v in vals if v is not None]
    axis.plot(x, y, color=col, marker=mk, linestyle=ls if len(x) > 1 else "none",
              markerfacecolor=mfc or col, label=label, **kw)


def fig_punch_force():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    rows = {r[0]: r for r in T["tab:punchforce"]}
    for key, e, label, ls, mfc in PUNCH_ROWS:
        r = rows[key]
        punch_series(ax, [num(c) for c in r[1:4]], e, ls, mfc, label)
        punch_series(bx, [num(c) for c in r[4:7]], e, ls, mfc)
    for axis, title in ((ax, "case A, $H = 10^7$ Pa"), (bx, "case B, $H = 0$")):
        axis.set_xscale("log")
        axis.set_xlabel("number of elements")
        axis.set_ylabel("punch force at $t = 1$ (N)")
        axis.set_title(title, fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "punch-force.png"), dpi=200)


def punch_jump_rows():
    """{(element key, case): (mean[], p99[], p99MPa[], maxp[])} from tab:punchjump."""
    out, case, key = {}, None, None
    for r in T["tab:punchjump"]:
        if r[0]:
            key = r[0]
        if "mean / p99" in r[1]:
            vals = [[num(v) for v in c.split("/")] if "tbd" not in c else [None, None] for c in r[2:5]]
            out.setdefault(key, {}).setdefault("norm", []).append(vals)
        else:
            vals = [[num(v) for v in c.split(",")] if "tbd" not in c else [None, None] for c in r[2:5]]
            out[key].setdefault("mpa", []).append(vals)
    return out


def fig_punch_jump(mpa):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=True)
    rows = punch_jump_rows()
    kind = "mpa" if mpa else "norm"
    for key, e, label, ls, mfc in PUNCH_ROWS:
        for axis, case in ((ax, 0), (bx, 1)):
            vals = rows[key][kind][case]
            punch_series(axis, [v[0] for v in vals], e, ls, mfc, label if axis is ax else None)
            punch_series(axis, [v[1] for v in vals], e, ":", "white", linewidth=1, markersize=4)
    for axis, title in ((ax, "case A, $H = 10^7$ Pa"), (bx, "case B, $H = 0$")):
        axis.set_xscale("log"); axis.set_yscale("log")
        axis.set_xlabel("number of elements")
        axis.set_title(title, fontsize=11)
    if mpa:
        ax.set_ylabel("99th percentile of the jump, largest $|p|$ (MPa)")
        for axis in (ax, bx):
            axis.axhline(1.0, **REF)
        ax.annotate("$\\sigma_y$", (ELEMENTS_PUNCH[0], 1.0), textcoords="offset points", xytext=(0, 3), fontsize=9, color="0.3")
        name = "punch-jump-mpa.png"
    else:
        ax.set_ylabel("normalized jump of the pressure")
        for axis in (ax, bx):
            slope_guide(axis, 6187, 38818, 0.005, -1 / 3, "1")
        name = "punch-jump.png"
    fig.legend(frameon=False, fontsize=9, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(OUT, name), dpi=200)


fig_kappa()
fig_beta()
fig_plastic()
fig_dilbound()
fig_taylor_pressure()
fig_taylor_cost()
fig_punch_force()
fig_punch_jump(False)
fig_punch_jump(True)
