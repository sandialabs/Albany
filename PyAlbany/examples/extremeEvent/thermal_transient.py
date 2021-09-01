from PyTrilinos import Tpetra
from PyTrilinos import Teuchos

from mpi4py import MPI
import numpy as np
from PyAlbany import Utils
from PyAlbany import ExtremeEvent as ee
import os
import sys

try:
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    printPlot = True
except:
    printPlot = False


def main(parallelEnv):
    comm = MPI.COMM_WORLD
    myGlobalRank = comm.rank

    # Create an Albany problem:

    n_params = 2
    filename = "thermal_transient.yaml"

    parameter = Utils.createParameterList(
        filename, parallelEnv
    )
    problem = Utils.createAlbanyProblem(parameter, parallelEnv)

    # ----------------------------------------------
    #
    #      1. Evaluation of the theta star
    #
    # ----------------------------------------------

    l_min = 0.
    l_max = 2.
    n_l = 5

    l = np.linspace(l_min, l_max, n_l)

    theta_star, I_star, F_star, P_star = ee.evaluateThetaStar(l, problem, n_params)

    np.savetxt('theta_star_transient.txt', theta_star)
    np.savetxt('I_star_transient.txt', I_star)
    np.savetxt('P_star_transient.txt', P_star)
    np.savetxt('F_star_transient.txt', F_star)

    # ----------------------------------------------
    #
    #   2. Evaluation of the prefactor using IS
    #
    # ----------------------------------------------

    N_samples = 100

    mean = np.array([1., 1.])
    cov = np.array([[1., 0.], [0., 1.]])

    samples = np.random.multivariate_normal(mean, cov, N_samples)

    P = ee.importanceSamplingEstimator(mean, cov, theta_star, F_star, P_star, samples, problem)

    np.savetxt('P_transient.txt', I_star)

    problem.reportTimers()

    # ----------------------------------------------
    #
    #   3.               Plots
    #
    # ----------------------------------------------

    if myGlobalRank == 0:
        if printPlot:
            plt.figure()
            plt.semilogy(F_star, P_star, '*-')
            plt.semilogy(F_star, P, '*-')

            plt.savefig('extreme_transient.jpeg', dpi=800)
            plt.close()


if __name__ == "__main__":
    comm = Teuchos.DefaultComm.getComm()
    parallelEnv = Utils.createDefaultParallelEnv(comm)
    main(parallelEnv)
