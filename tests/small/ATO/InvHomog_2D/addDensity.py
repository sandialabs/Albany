import Exobjects
import math

mesh = Exobjects.ExodusDB()
mesh.read("RVE.gen")

mesh.numNodeVars = 1
mesh.nodeVarNames = ["Rho_node"];
mesh.nodeVars = [[[0.0 for i in range(mesh.numNodes)]]];
mesh.varTimes = [0.0];

for nodeIndex in range(mesh.numNodes):
  X = mesh.getCoordData(nodeIndex,"index")
  val = math.cos(4.0*math.pi*X[0])*math.cos(4.0*math.pi*X[1]);
  if val < 0.0:
    val = 0.0
  mesh.nodeVars[0][0][nodeIndex] = val

mesh.write("RVE_restart.gen")
