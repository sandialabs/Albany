from mpi4py import MPI
import numpy as np
from PyAlbany import Utils
import os
import sys

try:
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    printPlot = True
except:
    printPlot = False

def main(parallelEnv):
    comm = MPI.COMM_WORLD
    myGlobalRank = comm.rank

    # Create an Albany problem:
    filename = "input_dirichletT.yaml"
    parameter = Utils.createParameterList(
        filename, parallelEnv
    )

    problem = Utils.createAlbanyProblem(parameter, parallelEnv)

    parameter_map_0 = problem.getParameterMap(0)
    parameter_0 = Utils.createVector(parameter_map_0)

    parameter_0_view = parameter_0.getLocalView()

    N = 200
    p_min = -2.
    p_max = 2.

    # Generate N samples randomly chosen in [p_min, p_max]:
    p = np.random.uniform(p_min, p_max, N)
    QoI = np.zeros((N,))

    # Loop over the N samples and evaluate the quantity of interest:
    for i in range(0, N):
        parameter_0_view[0] = p[i]
        parameter_0.setLocalView(parameter_0_view)
        problem.setParameter(0, parameter_0)

        problem.performSolve()

        response = problem.getResponse(0)
        QoI[i] = response.getLocalView()[0]

    if myGlobalRank == 0:
        if printPlot:
            f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16,6))

            ax1.hist(p)
            ax1.set_ylabel('Counts')
            ax1.set_xlabel('Random parameter')

            ax2.scatter(p, QoI)
            ax2.set_ylabel('Quantity of interest')
            ax2.set_xlabel('Random parameter')

            ax3.hist(QoI)
            ax3.set_ylabel('Counts')
            ax3.set_xlabel('Quantity of interest')

            plt.savefig('UQ.jpeg', dpi=800)
            plt.close()

if __name__ == "__main__":
    parallelEnv = Utils.createDefaultParallelEnv()
    main(parallelEnv)
