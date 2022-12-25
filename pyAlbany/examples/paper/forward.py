from mpi4py import MPI
import numpy as np
from PyAlbany import Utils
from PyAlbany import FEM_postprocess as fp
import os
import sys

import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams.update(mpl.rcParamsDefault)
import matplotlib.pyplot as plt


parallelEnv = Utils.createDefaultParallelEnv()

myGlobalRank = MPI.COMM_WORLD.rank

# Create an Albany problem:
filename = "input_scalar.yaml"
paramList = Utils.createParameterList(
    filename, parallelEnv
)

problem = Utils.createAlbanyProblem(paramList, parallelEnv)
problem.performSolve()

problem.reportTimers()
stackedTimer = problem.getStackedTimer()

total_time = stackedTimer.baseTimerAccumulatedTime("PyAlbany Total Time")
setup_time = stackedTimer.baseTimerAccumulatedTime("PyAlbany Total Time@PyAlbany: Setup Time")
perform_solve = stackedTimer.baseTimerAccumulatedTime("PyAlbany Total Time@PyAlbany: performSolve")
linear_solve = stackedTimer.baseTimerAccumulatedTime("PyAlbany Total Time@PyAlbany: performSolve@Piro::NOXSolver::evalModelImpl::solve@Thyra::NOXNonlinearSolver::solve@NOX Total Linear Solve")
print(linear_solve)

if myGlobalRank==0:
    x, y, sol, elements, triangulation = fp.readExodus("steady2d.exo", ['solution', 'thermal_conductivity', 'thermal_conductivity_sensitivity'], MPI.COMM_WORLD.Get_size())

    fp.tricontourf(x, y, sol[0,:], elements, triangulation, 'sol.jpeg', zlabel='Temperature', show_mesh=False)
    fp.tricontourf(x, y, sol[1,:], elements, triangulation, 'thermal_conductivity.jpeg', zlabel='Thermal conductivity')
    fp.tricontourf(x, y, sol[2,:], elements, triangulation, 'thermal_conductivity_sensitivity.jpeg', zlabel='Thermal conductivity sensitivity', show_mesh=False)

    # plot the mesh
    plt.figure()
    fp.plot_fem_mesh(x, y, elements)
    plt.axis([np.amin(x), np.amax(x), np.amin(y), np.amax(y)])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.plot([0,0],[0,1],'g', linewidth=5)
    plt.plot([0,1,1],[1,1,0],'r', linewidth=5)
    plt.plot([0,1],[0,0],'b', linewidth=5)

    plt.rcParams['text.usetex'] = True

    plt.text(1.1, 0.5, r'$T=0$', fontsize=14, color='r')
    plt.text(-0.2, 0.5, r'$T=1$', fontsize=14, color='g')
    plt.text(0.45, 1.1, r'$T=0$', fontsize=14, color='r')
    plt.text(0.45, -0.2, r'$T=p$', fontsize=14, color='b')

    

    plt.savefig('mesh.jpeg', dpi=800, bbox_inches='tight',pad_inches = 0)

