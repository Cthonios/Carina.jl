# Markdown table of the kernel timings of TET15-P1 across machines, from the
# JSON records of kernel_timing.jl:
#
#   python3 timing_table.py [kernel-timing.jsonl data/kernel-timing-*.jsonl ...]
#
# With no argument, every kernel-timing*.jsonl in this directory and in
# data/.  One row per host, device, threads, h and form: the newest record
# (by its date), so that a remeasurement replaces the earlier one in the
# table while the earlier record stays in its file; --all prints every
# record.  Rows in the order h, form, device, host, threads.  Records made
# before Carina 4300630 timed the eigenvalue estimate once, which overstated
# it; later ones take the median of five.
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ALL = "--all" in sys.argv
args = [a for a in sys.argv[1:] if a != "--all"]
files = args or sorted(glob.glob(os.path.join(HERE, "kernel-timing*.jsonl")) +
                               glob.glob(os.path.join(HERE, "data", "kernel-timing*.jsonl")))
DEVICE = {"cpu": "CPU", "rocm": "ROCm", "cuda": "CUDA"}
GPU = {("sirius", "rocm"): "AMD RX 7600", ("rigel", "cuda"): "NVIDIA L4",
       ("ascicgpu073", "cuda"): "NVIDIA A100-PCIE-40GB", ("ascicgpu22", "cuda"): "NVIDIA V100-PCIE-32GB",
       ("ascicgpu24", "cuda"): "NVIDIA V100-PCIE-32GB", ("ascicgpu080", "cuda"): "NVIDIA H100 80GB HBM3"}
CPU = {"sirius": "AMD Ryzen 9 9900X", "rigel": "2 x AMD EPYC 9634"}

rows = []
for f in files:
    for line in open(f):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
if not ALL:
    newest = {}
    for r in rows:
        k = (r["host"].split(".")[0], r["device"], r["threads"], r["h"], r["form"])
        if k not in newest or r["date"] > newest[k]["date"]:
            newest[k] = r
    rows = list(newest.values())
rows.sort(key=lambda r: (r["h"], r["form"], r["device"] != "cpu", r["host"], r["threads"]))

print("| host | device | threads | h (mm) | form | elements | residual (ms) | step (ms) | element dt (ms) | global dt (ms) | per run step (ms) | us per element-step |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    host = r["host"].split(".")[0]
    dev = GPU.get((host, r["device"])) or CPU.get(host, DEVICE.get(r["device"], r["device"])) if r["device"] != "cpu" else CPU.get(host, "CPU")
    ms = lambda k: f"{1e3 * r[k]:.2f}"
    print(f"| {host} | {dev} | {r['threads']} | {r['h']:g} | {r['form']} | {r['elements']} | {ms('residual_s')} | {ms('step_s')} | "
          f"{ms('element_dt_s')} | {ms('global_dt_s')} | {ms('run_step_s')} | {1e6 * r['run_step_s'] / r['elements']:.3f} |")
