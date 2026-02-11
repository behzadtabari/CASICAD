from OCC.Display.SimpleGui import init_display
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.gp import gp_Trsf, gp_Vec, gp_Pnt
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.Graphic3d import Graphic3d_Camera


# Geometry

box = BRepPrimAPI_MakeBox(80, 80, 80).Shape()

trsf = gp_Trsf()
trsf.SetTranslation(gp_Vec(150, 0, 0))
trsf.SetScale(gp_Pnt(150, 0, 0), 0.7)
box2 = BRepBuilderAPI_Transform(box, trsf, True).Shape()

# First window: Orthographic
display1, start_display1, *_ = init_display()
view1 = display1.View
display1.DisplayShape(box, color="BLUE1", update=False)
display1.DisplayShape(box2, color="RED", update=True)
view1.Camera().SetProjectionType(Graphic3d_Camera.Projection_Orthographic)
view1.SetProj(1, 1, 1)
view1.MustBeResized()
print("Window 1: Orthographic projection")

# Second window: Perspective
display2, start_display2, *_ = init_display()
view2 = display2.View
display2.DisplayShape(box, color="BLUE1", update=False)
display2.DisplayShape(box2, color="RED", update=True)
view2.Camera().SetProjectionType(Graphic3d_Camera.Projection_Perspective)
view2.SetProj(1, 1, 1)
view2.MustBeResized()
print("Window 2: Perspective projection")


start_display1()
