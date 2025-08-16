# 3_3-interference_checking.py
# Demonstrates interference checking using pythonocc-core
"""
A very easy way to check for collision between robot arms could be derived from
the convex-hull property of a Bézier curve. If the circumscribed (axis-aligned)
bounding boxes are not interfering, there is no collision between the curves.
If they do interfere, a finer test is needed — but this very simple test may
help immensely with computation.

- Build two random cubic Bézier curves from 4 random control points each
- Circumscribe the two curves in boxes (AABBs)
- Visualize both sets if there is **no** collision between the two boxes

Usage:
    python 3_3-interference_checking.py

Requirements:
    pip install pythonocc-core
"""

import random
from typing import Tuple

from OCC.Display.SimpleGui import init_display
from OCC.Core.gp import gp_Pnt
from OCC.Core.TColgp import TColgp_Array1OfPnt
from OCC.Core.Geom import Geom_BezierCurve
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib_Add
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox


# ------------------------------- helpers ------------------------------------

def random_cubic_bezier(scale: float = 50.0, offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
    """Create a random cubic Bézier curve (degree 3) as a Geom_BezierCurve and its TopoDS_Edge.

    Parameters
    ----------
    scale : float
        Controls the spread of control points around the offset.
    offset : (x, y, z)
        Base translation applied to all control points.

    Returns
    -------
    (Geom_BezierCurve, TopoDS_Edge)
    """
    pts = []
    for _ in range(4):  # cubic => 4 control points
        x = offset[0] + random.uniform(-1.0, 1.0) * scale
        y = offset[1] + random.uniform(-1.0, 1.0) * scale
        # keep (mostly) planar to make visualization nicer, vary z a bit
        z = offset[2] + random.uniform(-0.2, 0.2) * scale
        pts.append(gp_Pnt(x, y, z))

    arr = TColgp_Array1OfPnt(1, 4)
    for i, p in enumerate(pts, start=1):
        arr.SetValue(i, p)

    curve = Geom_BezierCurve(arr)
    edge = BRepBuilderAPI_MakeEdge(curve).Edge()
    return curve, edge


def shape_aabb(shape) -> Tuple[Bnd_Box, Tuple[float, float, float, float, float, float]]:
    """Compute an axis-aligned bounding box (AABB) for a TopoDS_Shape.

    Returns the Bnd_Box object and its (xmin, ymin, zmin, xmax, ymax, zmax) tuple.
    """
    box = Bnd_Box()
    box.SetGap(0.0)
    # useTriangulation=True for faster/robust bounds on curves/surfaces wrapped as edges
    brepbndlib_Add(shape, box, True)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    return box, (xmin, ymin, zmin, xmax, ymax, zmax)


def boxes_intersect(b1: Tuple[float, float, float, float, float, float],
                    b2: Tuple[float, float, float, float, float, float]) -> bool:
    """Axis-aligned box overlap test in 3D.

    Overlap occurs iff the intervals on all three axes overlap.
    """
    x1min, y1min, z1min, x1max, y1max, z1max = b1
    x2min, y2min, z2min, x2max, y2max, z2max = b2

    return (
        (x1min <= x2max and x2min <= x1max) and
        (y1min <= y2max and y2min <= y1max) and
        (z1min <= z2max and z2min <= z1max)
    )


def make_box_shape(bounds) -> object:
    """Create a TopoDS_Shape box from bounds (xmin, ymin, zmin, xmax, ymax, zmax)."""
    xmin, ymin, zmin, xmax, ymax, zmax = bounds
    pmin = gp_Pnt(xmin, ymin, zmin)
    pmax = gp_Pnt(xmax, ymax, zmax)
    return BRepPrimAPI_MakeBox(pmin, pmax).Shape()


# -------------------------------- main --------------------------------------

def main():
    random.seed(42)  # reproducible demos

    # Build two random curves. Offset the second to reduce the chance of overlap.
    _, edge1 = random_cubic_bezier(scale=60.0, offset=(0.0, 0.0, 0.0))
    # toggle the offset to see the changes
    _, edge2 = random_cubic_bezier(scale=60.0, offset=(50.0, 0.0, 0.0))

    # Compute their bounding boxes
    _, bnds1 = shape_aabb(edge1)
    _, bnds2 = shape_aabb(edge2)

    # Quick interference test
    intersects = boxes_intersect(bnds1, bnds2)

    if intersects:
        print("Collision detected between bounding boxes — skipping visualization per spec.\n"
              "(Refine test or regenerate curves if you want a non-collision visualization.)")
        return

    # Visualize both curves and their boxes if there was NO collision
    box_shape_1 = make_box_shape(bnds1)
    box_shape_2 = make_box_shape(bnds2)

    display, start_display, _, _ = init_display()

    # show curves
    display.DisplayShape(edge1, update=False, color="BLUE")
    display.DisplayShape(edge2, update=False, color="GREEN")

    # show boxes (semi-transparent)
    display.DisplayShape(box_shape_1, update=False, color="RED", transparency=0.8)
    display.DisplayShape(box_shape_2, update=True, color="ORANGE", transparency=0.8)

    # add a tiny legend in the console
    print("No collision detected — displaying curves (BLUE/GREEN) and AABBs (RED/ORANGE).")

    start_display()



if __name__ == "__main__":
    main()
