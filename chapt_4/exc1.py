# ========================================================
# Author: Behzad Tabari
# Date: 2026-02-23
# Description: This script explains solves Exercise 1 of Chapter 4, which is about the cusp of a cubic Bézier curve.
# ========================================================
import random

from OCC.Core.gp import gp_Pnt
from OCC.Core.TColgp import TColgp_Array1OfPnt
from OCC.Core.Geom import Geom_BezierCurve
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
from OCC.Display.SimpleGui import init_display
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB


display, start_display, add_menu, add_function_to_menu = init_display()

def random_color():
    return (random.random(), random.random(), random.random())


def make_curve(poles_xyz):
    poles = TColgp_Array1OfPnt(1, len(poles_xyz))
    for i, (x, y, z) in enumerate(poles_xyz, start=1):
        poles.SetValue(i, gp_Pnt(x, y, z))
    return Geom_BezierCurve(poles)

N = 30
for k in range(5, N, 5):
    bez = make_curve([(10, 0, 0), (-k, 10, 0), (k, 10, 0), (-10, 0, 0)])
    edge = BRepBuilderAPI_MakeEdge(bez).Edge()
    r, g, b = random_color()
    color = Quantity_Color(r, g, b, Quantity_TOC_RGB)
    display.DisplayShape(edge, color=color, update=False)

display.FitAll()
display.Repaint()

start_display()  