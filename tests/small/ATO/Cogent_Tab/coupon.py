import Exobjects
import GMeshTools

# create empty background grid
limits = [[-2.0,2.0], [-0.75,0.75], [-0.1,0.1]]
size = 0.1
gmesh = GMeshTools.GMesh(limits, size)

# construct geometry on background grid

body = GMeshTools.Body()

surf1 = GMeshTools.Boundary()
surf2 = GMeshTools.Boundary()

# brick
x = 2.8
y = 1.2
z = 0.1
brick = GMeshTools.Brick(x, y, z)
body.Add(brick)

# add cylinder ends
height = 0.1
radius = 0.6
position = [1.4, 0.0, 0.0]
cylinder = GMeshTools.Cylinder(height, radius, position)
body.Add(cylinder)

position = [-1.4, 0.0, 0.0]
cylinder = GMeshTools.Cylinder(height, radius, position)
body.Add(cylinder)

# subtract bolt holes
radius = 0.3575
position = [1.4, 0.0, 0.0]
cylinder = GMeshTools.Cylinder(height, radius, position)
body.Subtract(cylinder)

surf1.Add(cylinder.wall)

position = [-1.4, 0.0, 0.0]
cylinder = GMeshTools.Cylinder(height, radius, position)
body.Subtract(cylinder)

surf2.Add(cylinder.wall)

gmesh.ImprintBody(body)
gmesh.CreateBoundary(surf1,100)
gmesh.CreateBoundary(surf2,200)


gmesh.write("coupon.gen")
