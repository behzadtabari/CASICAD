from __future__ import annotations
"""

Author: Behzad Tabari
Date: 2025-07-28
Description: This script is all about the definition of variation diminishing
property, having an arbitrary curve and using piecewise linear interpolation for
interpolating it will having a curve that will pass any arbitrary plane less 
than or equal times of that original curve.

simple_step_reader.py  –  step extractor for STEP files to generate a graph

A helper module for turning a STEP file into *trackable* Python objects and – if
you like – into a topological graph.  
The reading layer has been **simplified** to expose a single convenience
function:

    read_faces_with_labels(path: str | Path) -> list[(TopoDS_Face, str)]

It hides all XCAF boiler‑plate and hands you back plain OCC faces together with
an intelligible label.  You can then pass that list straight into
``WorkFace.from_faces_with_labels`` (see below) if you need colour & edges.

Dependencies
------------
pythonocc‑core ≥ 7.7.2, networkx ≥ 3.0

"""
import random

import numpy as np
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax3, gp_Pln
from OCC.Core.Geom import Geom_BezierCurve
from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline
from OCC.Core.TColgp import TColgp_Array1OfPnt
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace
from OCC.Display.SimpleGui import init_display

# Create an arbitrary 3D curve
functions = [np.sin, np.cos, np.tan, np.log, np.exp, np.sinh, np.cosh, np.tanh]
n_pts = 100
points = []
for i in range(n_pts):
    t = i / (n_pts - 1)
    x = t * 10
    ### uncomment to see some unusual shapes
    #y = random.choice(functions)(x)
    #z = random.choice(functions)(x)
    ###
    y = np.sin(x)
    z = np.cos(x)
    points.append(gp_Pnt(x, y, z))

# Convert to OCC array
array = TColgp_Array1OfPnt(1, len(points))
for i, p in enumerate(points):
    array.SetValue(i + 1, p)

# Create the original smooth BSpline curve
bspline_curve = GeomAPI_PointsToBSpline(array).Curve()

#  Sample 10 points along the BSpline
n_samples = 10
sampled_points = []
u_start = bspline_curve.FirstParameter()
u_end = bspline_curve.LastParameter()

for i in range(n_samples):
    u = u_start + (u_end - u_start) * (i / (n_samples - 1))
    p = bspline_curve.Value(u)
    sampled_points.append(p)

# = Create a piecewise-linear approximation (polygon)
polygon_edges = []
for i in range(len(sampled_points) - 1):
    edge = BRepBuilderAPI_MakeEdge(sampled_points[i], sampled_points[i + 1]).Edge()
    polygon_edges.append(edge)

# Create a Bézier curve using the sampled points
bezier_array = TColgp_Array1OfPnt(1, len(sampled_points))
for i, p in enumerate(sampled_points):
    bezier_array.SetValue(i + 1, p)

bezier_curve = Geom_BezierCurve(bezier_array)

# Create a visual plane
# Plane at midpoint of curve with normal in Y
midpoint = sampled_points[len(sampled_points) // 2]
plane_axis = gp_Ax3(midpoint, gp_Dir(0, 1, 0))
plane = gp_Pln(plane_axis)
plane_face = BRepBuilderAPI_MakeFace(plane, -5, 5, -5, 5).Face()

# Visualization
display, start_display, add_menu, add_function_to_menu = init_display()

# Show original smooth curve
display.DisplayShape(bspline_curve, update=True, color="BLUE1")

# Show polygonal approximation
for edge in polygon_edges:
    display.DisplayShape(edge, color="RED")

# Show Bézier curve
display.DisplayShape(bezier_curve, color="GREEN")

# Show the plane
display.DisplayShape(plane_face, color="BROWN", transparency=0.5)

start_display()
