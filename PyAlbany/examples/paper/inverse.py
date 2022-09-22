from mpi4py import MPI
import numpy as np
from PyAlbany import Utils
from PyAlbany import FEM_postprocess as fp
import os
import sys

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


parallelEnv = Utils.createDefaultParallelEnv()
myGlobalRank = MPI.COMM_WORLD.rank

# Create an Albany problem:
filename = "input_distributed.yaml"
paramList = Utils.createParameterList(
    filename, parallelEnv
)

problem = Utils.createAlbanyProblem(paramList, parallelEnv)
problem.performAnalysis()
problem.performSolve()

para_0 = problem.getParameter(0)

para_0_view = para_0.getLocalView()

print(para_0_view)

if myGlobalRank==0:
    x, y, sol, elements, triangulation = fp.readExodus("steady2d.exo", ['solution', 'thermal_conductivity', 'thermal_conductivity_sensitivity'], MPI.COMM_WORLD.Get_size())

    fp.tricontourf(x, y, sol[0,:], elements, triangulation, 'sol_inverse.jpeg', zlabel='Temperature', show_mesh=False)
    fp.tricontourf(x, y, sol[1,:], elements, triangulation, 'thermal_conductivity_inverse.jpeg', zlabel='Thermal conductivity', show_mesh=False)
    fp.tricontourf(x, y, sol[2,:], elements, triangulation, 'thermal_conductivity_sensitivity_inverse.jpeg', show_mesh=False)
