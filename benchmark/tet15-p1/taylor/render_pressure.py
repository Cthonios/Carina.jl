# Render the pressure on the impact end of the Taylor bar with ParaView, from
# the surface.vtp and edges.vtp that pressure.jl writes in a run directory:
#
#   pvpython render_pressure.py <run dir> range=<p min>:<p max> [<out prefix>]
#
# (pvpython takes an argument that starts with '-' for an option of its own,
# so the range is one argument.)
#
# writes <prefix>-side.png, the deformed impact end seen obliquely from
# below, and <prefix>-face.png, the impact face seen along the axis, with the
# element edges in black and the pressure p = -tr(σ)/3 (Pa, positive in
# compression) in the ParaView color map Rainbow Uniform between p min and
# p max; values beyond them saturate.  <prefix> defaults to <run dir>/pressure.
import os
import sys

from paraview.simple import (ColorBy, CreateRenderView, GetColorTransferFunction,
                             ResetCamera, SaveScreenshot, Show, XMLPolyDataReader)

run = sys.argv[1]
lo, hi = (float(v) for v in sys.argv[2].split("=", 1)[1].split(":"))
prefix = sys.argv[3] if len(sys.argv) > 3 else os.path.join(run, "pressure")

surface = XMLPolyDataReader(FileName=[os.path.join(run, "surface.vtp")])
edges = XMLPolyDataReader(FileName=[os.path.join(run, "edges.vtp")])

view = CreateRenderView()
view.ViewSize = [1400, 1400]
view.OrientationAxesVisibility = 0
if hasattr(view, "UseColorPaletteForBackground"):
    view.UseColorPaletteForBackground = 0
view.Background = [1.0, 1.0, 1.0]

ds = Show(surface, view)
ColorBy(ds, ("POINTS", "pressure"))
lut = GetColorTransferFunction("pressure")
lut.ApplyPreset("Rainbow Uniform", True)
lut.RescaleTransferFunction(lo, hi)
lut.AutomaticRescaleRangeMode = "Never"
ds.SetScalarBarVisibility(view, False)
ds.Specular = 0.2
ds.Ambient = 0.35
ds.Diffuse = 0.75

de = Show(edges, view)
de.ColorArrayName = ["POINTS", ""]
de.AmbientColor = [0.0, 0.0, 0.0]
de.DiffuseColor = [0.0, 0.0, 0.0]
de.LineWidth = 1.0

cam = view.GetActiveCamera()
# Side: axis z up, the impact face (z = 0) at the bottom, seen from 30 degrees
# below the plane of the face.
view.CameraParallelProjection = 1
view.CameraFocalPoint = [0.0, 0.0, 3.0e-3]
view.CameraPosition = [0.0, -0.1, 3.0e-3 - 0.058]
view.CameraViewUp = [0.0, 0.0, 1.0]
ResetCamera(view)
cam.Zoom(1.05)
SaveScreenshot(prefix + "-side.png", view, ImageResolution=[1400, 1000])

# Face: along the axis from below the wall.
view.CameraFocalPoint = [0.0, 0.0, 0.0]
view.CameraPosition = [0.0, 0.0, -0.1]
view.CameraViewUp = [0.0, 1.0, 0.0]
ResetCamera(view)
cam.Zoom(1.05)
SaveScreenshot(prefix + "-face.png", view, ImageResolution=[1400, 1400])
