#!/usr/bin/env python3
"""Cross-code torsion benchmark: Carina (CPU + GPU), Norma.jl, Albany/LCM.

All three codes solve the *same* problem: the 160,000-element torsion cylinder
(530,523 DOF) from `torsion.g`, neo-Hookean E = 1 GPa / nu = 0.25 / rho = 1000,
free-free with a rigid-torsion initial velocity (a = 8000 1/s), Newmark
beta = 0.25 / gamma = 0.5, dt = 5e-5, Newton to abs 1e-6 or rel 1e-10.  The mesh
is byte-identical across codes; Norma's own torsion example uses a coarser mesh
from the same journal file, so block and node-set names already match, and
Albany reads it through `Method: Ioss`.

Timing method.  Each configuration is run twice, at 4 and 8 steps, and the
per-step cost is reported as (T_8 - T_4) / 4.  This is the only measure that is
honestly comparable here: it cancels every fixed cost, which otherwise differ by
more than the thing being measured -- Julia pays ~40 s of JIT per process that
Albany does not, and Albany pays a serial Exodus read that Carina's harness
front-loads differently.  Total wall time is recorded too, but it is a statement
about startup as much as about solver speed, and is labelled as such.

Each code runs with the linear solver its own torsion test ships with.  The
point is to compare codes as they are meant to be used, not to force one code's
solver choice onto another; the solver is recorded per row.

Threading.  Carina and Norma are run at 1 thread and at 24; Albany at 1, 12 and
24 MPI ranks (the box is a 12-core / 24-thread Ryzen 9 9900X).  The 1-way column
is the core-for-core comparison; the wide column is each code at its best.

Usage:  python3 run_crosscode.py [--steps 4,8] [--only carina,norma,lcm]
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
NORMA = os.path.expanduser("~/Repos/Norma.jl")
ALBANY = os.path.expanduser(
    "~/LCM/lcm-build-serial-gcc-release/src/Albany")
MPIBIN = "/usr/lib64/openmpi/bin"
RESULTS = os.path.join(HERE, "results.jsonl")

DT = 5.0e-5


def ensure_mesh():
    """Link the canonical mesh in rather than keeping a 14 MB duplicate.

    All three codes read it from the working directory, and Albany's MPI runs
    additionally need `decomp`-produced torsion.g.<nranks>.<rank> pieces
    alongside it (see README).
    """
    dst = os.path.join(HERE, "torsion.g")
    if not os.path.exists(dst):
        os.symlink(os.path.join(REPO, "examples", "meshes", "torsion",
                                "torsion.g"), dst)


def env_with(**kw):
    e = dict(os.environ)
    e["PATH"] = MPIBIN + ":" + e.get("PATH", "")
    e.update({k: str(v) for k, v in kw.items()})
    return e


def run(cmd, cwd, env, logpath, timeout=14400):
    """Run one configuration, return (wall_seconds, ok)."""
    with open(logpath, "w") as log:
        t0 = time.monotonic()
        try:
            p = subprocess.run(cmd, cwd=cwd, env=env, stdout=log,
                               stderr=subprocess.STDOUT, timeout=timeout)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            return (float("nan"), False)
        wall = time.monotonic() - t0
    return (wall, rc == 0)


# --------------------------------------------------------------------------- #
# Per-code invocation.  Each returns (cmd, cwd, env, extra_fields).
# --------------------------------------------------------------------------- #

def carina_case(variant, nsteps, threads):
    """Carina writes its decks from cases.jl; patch final time for nsteps."""
    deck = os.path.join(HERE, "carina-%s-n%d.yaml" % (variant, nsteps))
    src = os.path.join(REPO, "benchmark", "inputs",
                       "torsion-newmark-%s.yaml" % variant)
    with open(src) as f:
        y = f.read()
    y = y.replace("final time: 2.0e-4", "final time: %.8g" % (DT * nsteps))
    y = re.sub(r"input mesh file: .*", "input mesh file: torsion.g", y)
    y = re.sub(r"output mesh file: .*",
               "output mesh file: carina-%s.e" % variant, y)
    with open(deck, "w") as f:
        f.write(y)
    # `bin/carina` is the supported entry point: it owns the launcher
    # environment that carries the GPU vendor packages, and parses --threads
    # itself.  The deck's own `device:` line selects CPU or GPU.
    cmd = [os.path.join(REPO, "bin", "carina"), deck,
           "--threads", str(threads)]
    return cmd, HERE, env_with()


def norma_case(nsteps, threads):
    deck = os.path.join(HERE, "norma-newmark-n%d.yaml" % nsteps)
    with open(os.path.join(HERE, "norma-newmark.yaml")) as f:
        y = f.read()
    y = y.replace("final time: 2.0e-04", "final time: %.8g" % (DT * nsteps))
    with open(deck, "w") as f:
        f.write(y)
    cmd = ["julia", "--project=%s" % NORMA, "-t", str(threads),
           os.path.join(NORMA, "src", "Norma.jl"), os.path.basename(deck)]
    return cmd, HERE, env_with()


def lcm_case(nsteps, ranks):
    deck = os.path.join(HERE, "lcm-newmark-n%d.yaml" % nsteps)
    with open(os.path.join(HERE, "lcm-newmark.yaml")) as f:
        y = f.read()
    y = y.replace("Final Time: 2.0e-04", "Final Time: %.8g" % (DT * nsteps))
    with open(deck, "w") as f:
        f.write(y)
    cmd = ([os.path.join(MPIBIN, "mpirun"), "-np", str(ranks),
            "--oversubscribe", ALBANY, os.path.basename(deck)]
           if ranks > 1 else [ALBANY, os.path.basename(deck)])
    return cmd, HERE, env_with()


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="4,8")
    ap.add_argument("--only", default="carina,norma,lcm")
    ap.add_argument("--variants", default="",
                    help="comma-separated variant filter; empty = all")
    ap.add_argument("--ways", default="",
                    help="comma-separated thread/rank-count filter; empty = all")
    # A discarded warm-up run before the timed pair.  Without it the FIRST
    # configuration to touch a given code path pays cold-cache costs the later
    # one inherits warm -- ROCm kernel compilation, Julia precompile, page
    # cache on a 14 MB mesh.  That breaks the difference method's premise that
    # the two runs share their fixed cost, and it showed up as a *negative*
    # per-step for the first GPU row of the first batch.
    ap.add_argument("--warmup", action="store_true")
    args = ap.parse_args()
    ensure_mesh()
    steps = [int(s) for s in args.steps.split(",")]
    only = set(args.only.split(","))
    vfilter = set(v for v in args.variants.split(",") if v)
    wfilter = set(int(w) for w in args.ways.split(",") if w)

    configs = []
    if "carina" in only:
        # GPU variants: no thread dimension.
        for v in ("gpu-cg-jacobi", "gpu-cg-chebyshev", "gpu-lbfgs"):
            configs.append(("carina", v, 1, "gpu"))
        # CPU: every variant at 24 threads, and the representative one at 1.
        for v in ("cpu-cg-jacobi", "cpu-cg-amg", "cpu-cg-ic", "cpu-direct"):
            configs.append(("carina", v, 24, "cpu"))
        # Thread-scaling sweep on the representative CPU variant.
        # Report section 5 measures 9.9x at 24 threads for the EXPLICIT
        # kernel; the implicit CPU path has never been measured, and the
        # first 1-vs-24 pair came back at only 1.25x, so the curve is
        # needed to tell a real property from a measurement error.
        for t in (1, 4, 12):
            configs.append(("carina", "cpu-cg-jacobi", t, "cpu"))
    if "norma" in only:
        configs.append(("norma", "hessian-newton", 24, "cpu"))
        configs.append(("norma", "hessian-newton", 1, "cpu"))
    if "lcm" in only:
        for n in (24, 12, 1):
            configs.append(("lcm", "belos-gmres-ilut", n, "cpu"))

    if vfilter:
        configs = [c for c in configs if c[1] in vfilter]
    if wfilter:
        configs = [c for c in configs if c[2] in wfilter]

    for (code, variant, ways, dev) in configs:
        walls = {}
        ok_all = True
        if args.warmup:
            n = steps[0]
            if code == "carina":
                cmd, cwd, env = carina_case(variant, n, ways)
            elif code == "norma":
                cmd, cwd, env = norma_case(n, ways)
            else:
                cmd, cwd, env = lcm_case(n, ways)
            tag = "%s_%s_%dway_warmup" % (code, variant, ways)
            logp = os.path.join(HERE, "logs", tag + ".log")
            os.makedirs(os.path.dirname(logp), exist_ok=True)
            print("[WARM] %-40s ..." % tag, flush=True, end=" ")
            w, _ = run(cmd, cwd, env, logp)
            print("%.1f s (discarded)" % w, flush=True)
        for n in steps:
            if code == "carina":
                cmd, cwd, env = carina_case(variant, n, ways)
            elif code == "norma":
                cmd, cwd, env = norma_case(n, ways)
            else:
                cmd, cwd, env = lcm_case(n, ways)
            tag = "%s_%s_%dway_n%d" % (code, variant, ways, n)
            logp = os.path.join(HERE, "logs", tag + ".log")
            os.makedirs(os.path.dirname(logp), exist_ok=True)
            print("[RUN] %-40s ..." % tag, flush=True, end=" ")
            wall, ok = run(cmd, cwd, env, logp)
            print("%.1f s %s" % (wall, "" if ok else "FAILED"), flush=True)
            walls[n] = wall
            ok_all = ok_all and ok
        per_step = ((walls[steps[-1]] - walls[steps[0]]) /
                    (steps[-1] - steps[0])) if len(steps) > 1 else None
        rec = {
            "code": code, "variant": variant, "device": dev,
            "ways": ways, "ok": ok_all,
            "walls": {str(k): v for k, v in walls.items()},
            "per_step_s": per_step,
            # Whether a discarded warm-up preceded the timed pair.  Rows with
            # warmup=false that duplicate a warmup=true row are superseded:
            # they are the cold-cache measurements kept for the record.
            "warmup": bool(args.warmup),
            "n_dofs": 530523, "dt": DT,
        }
        with open(RESULTS, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print("      -> per-step %.3f s" % per_step if per_step else "",
              flush=True)


if __name__ == "__main__":
    main()
