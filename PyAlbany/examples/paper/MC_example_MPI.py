from mpi4py import MPI
import numpy as np
from numpy.random import default_rng
from PyAlbany import Utils
from PyAlbany import AlbanyInterface as pa
import os
import sys

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


group_size = 2

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
group_ID = np.floor(rank*1./group_size)
n_groups = np.ceil(size*1./group_size)

nComm = comm.Split(group_ID)

parallelEnv = Utils.createDefaultParallelEnv(pa.getTeuchosComm(nComm))

comm = MPI.COMM_WORLD
myGlobalRank = comm.rank

# Create an Albany problem:
filename = "input_scalar.yaml"
paramList = Utils.createParameterList(
    filename, parallelEnv
)

paramList.sublist("Discretization").set("Exodus Output File Name", "steady2d_color_"+str(group_ID)+".exo")

problem = Utils.createAlbanyProblem(paramList, parallelEnv)

parameter_map_0 = problem.getParameterMap(0)
parameter_0 = Utils.createVector(parameter_map_0)

parameter_0_view = parameter_0.getLocalView()

N = int(np.ceil(200/n_groups))
p_min = -2.
p_max = 2.

# Generate N samples randomly chosen in [p_min, p_max]:
rng = default_rng()
p = rng.uniform(p_min, p_max, N)
QoI = np.empty((N,))

# Loop over the N samples and evaluate the quantity of interest:
for i in range(0, N):
    parameter_0_view[0] = p[i]
    parameter_0.setLocalView(parameter_0_view)
    problem.setParameter(0, parameter_0)

    problem.performSolve()

    response = problem.getResponse(0)
    QoI[i] = response.getLocalView()[0]
