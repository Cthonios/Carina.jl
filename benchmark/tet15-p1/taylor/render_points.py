# Render the pressure at the integration points of the elements on the impact
# face of the Taylor bar with ParaView, from the points.vtp and points.txt
# that points_pressure.jl writes and the surface.vtp and edges.vtp that
# pressure.jl writes in a run directory:
#
#   pvpython render_points.py <run dir> range=<p min>:<p max> [<out prefix>]
#
# writes <prefix>-face.png, the impact face seen along the axis, with the
# deformed surface in white, the element edges in black and each integration point as a disk of diameter
# 0.3 h (h the mean edge length of the faces, from points.txt), colored by the
# pressure p = -tr(σ)/3 (Pa, positive in compression) in the ParaView color map
# Rainbow Uniform between p min and p max; values beyond them saturate.
# <prefix> defaults to <run dir>/pressure, so that compose_pressure.py composes
# the panels as it does for the element values.
import os
import sys

from paraview.simple import (ColorBy, CreateRenderView, GetColorTransferFunction, Hide,
                             ResetCamera, SaveScreenshot, Show, Transform, XMLPolyDataReader)

run = sys.argv[1]
lo, hi = (float(v) for v in sys.argv[2].split("=", 1)[1].split(":"))
prefix = sys.argv[3] if len(sys.argv) > 3 else os.path.join(run, "pressure")
h = float(open(os.path.join(run, "points.txt")).read().split()[0])

points = XMLPolyDataReader(FileName=[os.path.join(run, "points.vtp")])
edges = XMLPolyDataReader(FileName=[os.path.join(run, "edges.vtp")])
surface = XMLPolyDataReader(FileName=[os.path.join(run, "surface.vtp")])

view = CreateRenderView()
view.ViewSize = [1400, 1400]
view.OrientationAxesVisibility = 0
if hasattr(view, "UseColorPaletteForBackground"):
    view.UseColorPaletteForBackground = 0
view.Background = [1.0, 1.0, 1.0]

# The deformed surface in white hides the edges and points behind the face.
ds = Show(surface, view)
ds.ColorArrayName = ["POINTS", ""]
ds.AmbientColor = [1.0, 1.0, 1.0]
ds.DiffuseColor = [1.0, 1.0, 1.0]
ds.Ambient = 1.0
ds.Diffuse = 0.0

de = Show(edges, view)
de.ColorArrayName = ["POINTS", ""]
de.AmbientColor = [0.0, 0.0, 0.0]
de.DiffuseColor = [0.0, 0.0, 0.0]
de.LineWidth = 1.0

# The points lie inside the bar, behind the face as seen from the wall.  They
# are moved toward the camera, along the axis, so that the face does not hide
# them; in the parallel projection along the axis this does not move their
# image.
shifted = Transform(Input=points)
shifted.Transform.Translate = [0.0, 0.0, -0.02]
dp = Show(shifted, view)
dp.SetRepresentationType("Point Gaussian")
dp.ShaderPreset = "Plain circle"
dp.GaussianRadius = 0.15 * h
ColorBy(dp, ("POINTS", "pressure"))
lut = GetColorTransferFunction("pressure")
lut.ApplyPreset("Rainbow Uniform", True)
lut.RescaleTransferFunction(lo, hi)
lut.AutomaticRescaleRangeMode = "Never"
dp.SetScalarBarVisibility(view, False)

# Face: along the axis from below the wall (as render_pressure.py).  The
# camera is set on the surface and edges alone, so that the shifted points do
# not change the framing.
Hide(shifted, view)
view.CameraParallelProjection = 1
view.CameraFocalPoint = [0.0, 0.0, 0.0]
view.CameraPosition = [0.0, 0.0, -0.1]
view.CameraViewUp = [0.0, 1.0, 0.0]
ResetCamera(view)
view.GetActiveCamera().Zoom(1.05)
Show(shifted, view)
SaveScreenshot(prefix + "-face.png", view, ImageResolution=[1400, 1400])
