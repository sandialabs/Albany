from mpi4py import MPI
import numpy as np
from PyAlbany import Utils
from PyAlbany import AlbanyInterface as pa
import os
import sys
import argparse

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


parallelEnv = Utils.createDefaultParallelEnv()

comm = MPI.COMM_WORLD
nMaxProcs = comm.Get_size()
myGlobalRank = comm.rank

parser = argparse.ArgumentParser(description='Select the scaling.')
parser.add_argument('-w', action="store_true", default=False)
args = parser.parse_args()

weak_scaling = args.w

timerNames = ["PyAlbany: Create Albany Problem", 
            "PyAlbany: Perform Solve",
            "PyAlbany: Total"]

nTimers = len(timerNames)

timers_sec = np.empty((nMaxProcs,nTimers))

efficiency = np.empty((nMaxProcs,nTimers))

for nProcs in range(1, nMaxProcs+1):
    newGroup = comm.group.Incl(np.arange(0, nProcs))
    newComm = comm.Create_group(newGroup)

    if myGlobalRank < nProcs:
        parallelEnv.setComm(pa.getTeuchosComm(newComm))

        timers = Utils.createTimers(timerNames)
        timers[2].start()
        timers[0].start()

        filename = "input_scalar.yaml"
        paramList = Utils.createParameterList(
            filename, parallelEnv
        )

        if weak_scaling:
            paramList.sublist("Discretization").set("2D Elements", 40*nProcs)
        problem = Utils.createAlbanyProblem(paramList, parallelEnv)
        timers[0].stop()

        timers[1].start()
        problem.performSolve()
        timers[1].stop()
        timers[2].stop()

        if myGlobalRank == 0:
            for j in range(0, nTimers):
                timers_sec[nProcs-1,j] = timers[j].totalElapsedTime()

if myGlobalRank == 0:
    for i in range(0, nMaxProcs):
        efficiency[i,:] = timers_sec[0,:]/(timers_sec[i,:])
        if not weak_scaling:
            efficiency[i,:] /= (i+1)

    fig = plt.figure(figsize=(10,6))
    plt.plot([1, nMaxProcs+1], [1., 1.], '--')
    for j in range(0, nTimers):
        plt.plot(np.arange(1, nMaxProcs+1), efficiency[:,j], 'o-', label=timerNames[j])
    plt.ylabel('efficiency')
    plt.xlabel('number of MPI processes')
    plt.grid(True)
    plt.legend()
    if weak_scaling:
        plt.savefig('weak_scaling.jpeg', dpi=800)
    else:
        plt.savefig('strong_scaling.jpeg', dpi=800)
    plt.close()

