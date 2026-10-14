# Render a punch run with ParaView, from its Exodus output, at step 20 (t = 1):
#
#   pvpython render.py <results.e> <out prefix> p=<min>:<max> eqps=<min>:<max>
#
# (pvpython takes an argument that starts with '-' for an option of its own,
# so each range is one argument.)  The element values are the element means:
# p = -tr(σ)/3 (Pa, positive in compression) from the element stress of
# Sierra/SM (stress_xx, ...) or the mean over the quadrature points of Carina
# (sigma_xx_<q>, ...), and eqps or the mean of eqps_<q>.  Writes, for each of
# p and eqps:
#   <prefix>-<q>-meridian.png  the symmetry face y = 0 of the deformed quarter,
#                              the region 0 <= x <= 2a, z >= H - 2a, seen
#                              along +y
#   <prefix>-<q>-top.png       the deformed top face, 0 <= x, y <= 2a, seen
#                              along -z
# with the element edges of the cut and of the face in black, the ParaView
# color map Rainbow Uniform between <min> and <max> (values beyond saturate),
# 1400 px wide.
import os
import re
import sys

from paraview.simple import (GetScalarBar, Calculator, Clip, WarpByVector, ColorBy, CreateRenderView, ExtractSurface,
                             GetColorTransferFunction, ExodusIIReader,
                             SaveScreenshot, Show, UpdatePipeline)

A, H = 1.0e-3, 4.0e-3
path, prefix = sys.argv[1], sys.argv[2]
ranges = dict((k, tuple(float(v) for v in r.split(":")))
              for k, r in (a.split("=", 1) for a in sys.argv[3:]))

reader = ExodusIIReader(FileName=[path])
reader.ElementVariables = reader.ElementVariables.Available
reader.PointVariables = reader.PointVariables.Available
reader.ApplyDisplacements = 0
times = reader.TimestepValues
t_end = times[-1] if hasattr(times, "__len__") else times
UpdatePipeline(time=t_end, proxy=reader)

names = list(reader.ElementVariables.Available)
if "stress_xx" in names or any(n.startswith("stress_") for n in names):
    p_expr = "-(stress_xx+stress_yy+stress_zz)/3"
else:
    nq = len([n for n in names if re.match(r"sigma_xx_\d+$", n)])
    p_expr = "-(" + "+".join("sigma_%s_%d" % (c, q) for q in range(1, nq + 1)
                             for c in ("xx", "yy", "zz")) + ")/%d" % (3 * nq)
eq = ["eqps"] if "eqps" in names else [n for n in names if re.match(r"eqps_\d+$", n)]
e_expr = "(" + "+".join(eq) + ")/%d" % len(eq)

calc = Calculator(Input=reader, AttributeType="Cell Data", ResultArrayName="p", Function=p_expr)
calc = Calculator(Input=calc, AttributeType="Cell Data", ResultArrayName="eqps_mean", Function=e_expr)

# The plane y = 0 is the symmetry face of the quarter; the clip at y = 0 in
# the reference configuration keeps the whole quarter (and would keep the half
# y >= 0 of a full mesh), and seen from y < 0 it shows the element faces on
# the plane.
pv = list(reader.PointVariables.Available)
disp = "displacement" if "displacement" in pv else [n for n in pv if n.lower().startswith("disp")][0]
clip = Clip(Input=calc, ClipType="Plane")
clip.ClipType.Origin = [0.0, 0.0, 0.0]
clip.ClipType.Normal = [0.0, 1.0, 0.0]
clip.Invert = 0
clip = WarpByVector(Input=clip, Vectors=["POINTS", disp], ScaleFactor=1.0)
surf = WarpByVector(Input=ExtractSurface(Input=calc), Vectors=["POINTS", disp], ScaleFactor=1.0)

view = CreateRenderView()
view.OrientationAxesVisibility = 0
if hasattr(view, "UseColorPaletteForBackground"):
    view.UseColorPaletteForBackground = 0
view.Background = [1.0, 1.0, 1.0]
view.CameraParallelProjection = 1
view.ViewTime = t_end


def shot(source, array, rng, focal, position, up, scale, size, out):
    for r in list(view.Representations):
        r.Visibility = 0
    d = Show(source, view)
    d.Representation = "Surface With Edges"
    d.EdgeColor = [0.0, 0.0, 0.0]
    d.Ambient, d.Diffuse, d.Specular = 1.0, 0.0, 0.0      # flat colors, no shading
    ColorBy(d, ("CELLS", array))
    lut = GetColorTransferFunction(array)
    lut.ApplyPreset("Rainbow Uniform", True)
    lut.AutomaticRescaleRangeMode = "Never"
    lut.RescaleTransferFunction(*rng)
    d.SetScalarBarVisibility(view, True)
    bar = GetScalarBar(lut, view)
    bar.TitleColor = bar.LabelColor = [0.0, 0.0, 0.0]
    bar.Title = {"p": "p (Pa)", "eqps_mean": "eqps"}[array]
    bar.ComponentTitle = ""
    view.CameraFocalPoint = focal
    view.CameraPosition = position
    view.CameraViewUp = up
    view.CameraParallelScale = scale
    view.ViewSize = size
    SaveScreenshot(out, view, ImageResolution=size)


for q, array in (("p", "p"), ("eqps", "eqps_mean")):
    rng = ranges[q]
    shot(clip, array, rng, [A, 0.0, H - 0.95 * A], [A, -1.0, H - 0.95 * A], [0.0, 0.0, 1.0],
         1.05 * A, [1400, 1400], "%s-%s-meridian.png" % (prefix, q))
    shot(surf, array, rng, [A, A, H], [A, A, H + 1.0], [0.0, 1.0, 0.0],
         1.05 * A, [1400, 1400], "%s-%s-top.png" % (prefix, q))
