import Exobjects
import GMeshTools

# create empty background grid
limits = [[-50.0,50.0], [-50.0,50.0], [-10.0,10.0]]
size = 10.0
gmesh = GMeshTools.GMesh(limits, size)
gmesh.write("background.gen")

# construct geometry on background grid

body = GMeshTools.Body()

noDisp1 = GMeshTools.Boundary()
noDisp2 = GMeshTools.Boundary()

load = GMeshTools.Boundary()

# brick
x = 96.0
y = 96.0
z = 10.0
brick = GMeshTools.Brick(x, y, z)
body.Add(brick)

# subtract bolt holes
radius = 12.0
height = 10.0
position = [-30.0,-30.0, 0.0]
cylinder = GMeshTools.Cylinder(height, radius, position)
body.Subtract(cylinder)
noDisp1.Add(cylinder.wall)

position = [-30.0, 30.0, 0.0]
cylinder = GMeshTools.Cylinder(height, radius, position)
body.Subtract(cylinder)
noDisp2.Add(cylinder.wall)

position = [ 30.0, 0.0, 0.0]
cylinder = GMeshTools.Cylinder(height, radius, position)
body.Subtract(cylinder)
load.Add(cylinder.wall)

gmesh.ImprintBody(body)
gmesh.CreateBoundary(noDisp1,100)
gmesh.CreateBoundary(noDisp2,200)

gmesh.CreateBoundary(load,300)

gmesh.write("mitchell.gen")
