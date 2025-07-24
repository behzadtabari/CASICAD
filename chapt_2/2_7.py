import random as rd
import numpy as np

from scipy.spatial import Voronoi
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeFace
from OCC.Core.BRep import BRep_Builder
from OCC.Core.TopoDS import TopoDS_Compound
from OCC.Display.SimpleGui import init_display
from OCC.Core.gp import gp_Pnt

def bounded_voronoi_polygons(vor, radius=1e6):
    # Code adapted from scipy docs to close infinite regions
    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    for pointidx, regionidx in enumerate(vor.point_region):
        region = vor.regions[regionidx]
        if -1 not in region:
            new_regions.append(region)
            continue
        ridges = [vor.ridge_vertices[i]
                  for i in range(len(vor.ridge_points))
                  if pointidx in vor.ridge_points[i]]

        new_region = [v for v in region if v != -1]
        for vpair in ridges:
            if -1 in vpair:
                i = vpair[0] if vpair[1] == -1 else vpair[1]
                t = vor.vertices[i] - center
                t = t / np.linalg.norm(t)
                far_point = vor.vertices[i] + t * radius
                new_vertices.append(far_point.tolist())
                new_region.append(len(new_vertices) - 1)
        new_regions.append(new_region)

    return new_regions, np.array(new_vertices)

# Generate sample points
points = np.array([[rd.uniform(0,1),rd.uniform(0,1)]for i in range(10)])
vor = Voronoi(points)
regions, vertices = bounded_voronoi_polygons(vor)

# Create OCC compound of all faces
compound = TopoDS_Compound()
builder = BRep_Builder()
builder.MakeCompound(compound)

for region in regions:
    polygon = BRepBuilderAPI_MakePolygon()
    for idx in region:
        x, y = vertices[idx]
        polygon.Add(gp_Pnt(x, y,0))
    polygon.Close()

    wire = polygon.Wire()
    face = BRepBuilderAPI_MakeFace(wire).Face()
    builder.Add(compound, face)

# Display
display, start_display, add_menu, add_function_to_menu = init_display()
display.DisplayShape(compound, update=True)
start_display()
