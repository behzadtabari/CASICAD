# 3_3.py
# Demonstrates affine invariance of cubic Bézier curves using pythonocc-core
# - Build a cubic Bézier from 4 control points
# - Sample 100 points on the curve and rotate the sampled points
# - Rotate the control polygon first, rebuild the Bézier, and sample 100 points
# - Visualize both sets and report the maximum deviation
#
# Usage:
#   python 3_3.py
#
# Requirements:
#   pip install pythonocc-core

import math
from OCC.Core.gp import gp_Pnt, gp_Ax1, gp_Dir, gp_Trsf
from OCC.Core.Geom import Geom_BezierCurve
from OCC.Core.TColgp import TColgp_Array1OfPnt
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeSphere
from OCC.Core.Quantity import (
    Quantity_NOC_RED,
    Quantity_NOC_BLUE1,
    Quantity_NOC_GREEN,
    Quantity_NOC_BLACK,
)
from OCC.Display.SimpleGui import init_display


def make_bezier_from_points(p0, p1, p2, p3):
    """Create a cubic Bézier curve from 4 gp_Pnt poles."""
    arr = TColgp_Array1OfPnt(1, 4)
    arr.SetValue(1, p0)
    arr.SetValue(2, p1)
    arr.SetValue(3, p2)
    arr.SetValue(4, p3)
    return Geom_BezierCurve(arr)


def sample_curve(curve, n=100):
    """Sample n points uniformly in parameter [0,1] (inclusive)."""
    if n < 2:
        n = 2
    pts = []
    for i in range(n):
        u = i / (n - 1)
        pts.append(curve.Value(u))
    return pts


def rotate_points(points, trsf):
    return [p.Transformed(trsf) for p in points]


def rotate_curve_poles(curve, trsf):
    """Return a *new* Bézier curve whose poles are the original poles rotated by trsf."""
    nb = curve.NbPoles()
    arr = TColgp_Array1OfPnt(1, nb)
    curve.Poles(arr)
    for i in range(1, nb + 1):
        arr.SetValue(i, arr.Value(i).Transformed(trsf))
    return Geom_BezierCurve(arr)


def max_deviation(pts_a, pts_b):
    assert len(pts_a) == len(pts_b)
    return max(pts_a[i].Distance(pts_b[i]) for i in range(len(pts_a)))


def build_edge_from_curve(curve):
    return BRepBuilderAPI_MakeEdge(curve).Edge()


def build_edge_between(pA, pB):
    return BRepBuilderAPI_MakeEdge(pA, pB).Edge()


def display_points_as_spheres(display, pts, color, radius=0.06):
    """Render points as small spheres so we can control 'point size' on all OCC builds."""
    for p in pts:
        sphere = BRepPrimAPI_MakeSphere(p, radius).Shape()
        display.DisplayShape(sphere, update=False, color=color)


def main():
    # 1) Define 4 control points (change these to play around)
    P0 = gp_Pnt(0.0, 0.0, 0.0)
    P1 = gp_Pnt(1.5, 2.0, 0.0)
    P2 = gp_Pnt(3.0, -1.0, 0.0)
    P3 = gp_Pnt(5.0, 0.0, 0.0)

    # 2) Build the base Bézier curve
    curve = make_bezier_from_points(P0, P1, P2, P3)

    # 3) Sample 100 points
    N = 100
    pts = sample_curve(curve, N)

    # 4) Define a rotation transform (about Z-axis through origin)
    angle_deg = 37.0
    angle_rad = math.radians(angle_deg)
    axis = gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
    trsf = gp_Trsf()
    trsf.SetRotation(axis, angle_rad)

    # 5) Rotate the sampled points (post-sampling transform)
    pts_rot_after = rotate_points(pts, trsf)

    # 6) Rotate the control polygon first, rebuild Bézier, and resample
    curve_rot_first = rotate_curve_poles(curve, trsf)
    pts_rot_first = sample_curve(curve_rot_first, N)

    # 7) Compute the maximum pointwise deviation (should be ~ numerical noise)
    max_dev = max_deviation(pts_rot_after, pts_rot_first)
    print(
        f"Max deviation between 'rotate-then-sample' and 'sample-then-rotate': {max_dev:.6e}"
    )

    # 8) Build shapes for visualization
    edge_base = build_edge_from_curve(curve)
    edge_rot = build_edge_from_curve(curve_rot_first)

    # Control polygon (original)
    ctrl_poly_edges = [
        build_edge_between(P0, P1),
        build_edge_between(P1, P2),
        build_edge_between(P2, P3),
    ]

    # 9) Visualize
    display, start_display, add_menu, add_function = init_display()

    # Draw original control polygon and base curve
    for e in ctrl_poly_edges:
        display.DisplayShape(e, update=False, color=Quantity_NOC_BLACK)
    display.DisplayShape(edge_base, update=False, color=Quantity_NOC_BLUE1)

    # Rotated-after-sampling points: GREEN spheres
    display_points_as_spheres(display, pts_rot_after, color=Quantity_NOC_GREEN, radius=0.06)

    # Rotated-control-curve (and resampled points): RED curve + RED spheres
    display.DisplayShape(edge_rot, update=False, color=Quantity_NOC_RED)
    display_points_as_spheres(display, pts_rot_first, color=Quantity_NOC_RED, radius=0.05)

    # Fit view and annotate
    display.FitAll()
    display.DisplayMessage(
        gp_Pnt(0, 0, 0),
        f"Affine invariance:\nmax deviation = {max_dev:.3e}\nRotation = {angle_deg} deg",
    )

    start_display()


if __name__ == "__main__":
    main()
