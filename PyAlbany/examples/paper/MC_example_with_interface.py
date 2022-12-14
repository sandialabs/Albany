from mpi4py import MPI
import numpy as np
from numpy.random import default_rng
from PyAlbany import Utils
import os
import sys
import time

start = time.time()

parallelEnv = Utils.createDefaultParallelEnv()

comm = MPI.COMM_WORLD
myGlobalRank = comm.rank

# Create an Albany problem:
filename = "input_scalar.yaml"
paramList = Utils.createParameterList(
    filename, parallelEnv
)

problem = Utils.createAlbanyProblem(paramList, parallelEnv)

parameter_map_0 = problem.getParameterMap(0)
parameter_0 = Utils.createVector(parameter_map_0)

parameter_0_view = parameter_0.getLocalView()

N = 50
p_min = -2.
p_max = 2.

# Generate N samples randomly chosen in [p_min, p_max]:
rng = default_rng(seed=42)
p = rng.uniform(p_min, p_max, N)
QoI = np.empty((N,))
timers = np.empty((N,))
timers_setup = np.empty((N,))

# Loop over the N samples and evaluate the quantity of interest:
for i in range(0, N):
    parameter_0_view[0] = p[i]
    parameter_0.setLocalView(parameter_0_view)
    problem.setParameter(0, parameter_0)

    problem.performSolve()

    response = problem.getResponse(0)
    QoI[i] = response.getLocalView()[0]
    timers[i] = problem.getStackedTimer().baseTimerAccumulatedTime('PyAlbany Total Time')
    timers_setup[i] = problem.getStackedTimer().baseTimerAccumulatedTime('PyAlbany Total Time@PyAlbany: Setup Time')

print(p)
print(QoI)
print(timers)

end = time.time()
elapsed_time = end-start
print(str(timers[-1])+" "+str(elapsed_time)+" "+str(100*timers[-1]/elapsed_time)+"%")
np.savetxt('timers_with_interface.txt', timers)
np.savetxt('timers_setup_with_interface.txt', timers_setup)
