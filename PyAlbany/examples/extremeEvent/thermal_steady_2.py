from mpi4py import MPI
import numpy as np
from PyAlbany import Utils
from PyAlbany import ExtremeEvent as ee
from PyAlbany import AlbanyInterface as pa
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


def evaluate_responses(X, Y, problem, recompute=False):
    if not recompute and os.path.isfile('Z1_2.txt'):
        Z1 = np.loadtxt('Z1_2.txt')
        Z2 = np.loadtxt('Z2_2.txt')
    else:
        comm = MPI.COMM_WORLD
        myGlobalRank = comm.rank

        parameter_map = problem.getParameterMap(0)
        parameter = Utils.createVector(parameter_map)
        para_view = parameter.getLocalView()

        n_x = len(X)
        n_y = len(Y)
        Z1 = np.zeros((n_y, n_x))
        Z2 = np.zeros((n_y, n_x))

        for i in range(n_x):
            para_view[0] = X[i]
            for j in range(n_y):
                para_view[1] = Y[j]
                parameter.setLocalView(para_view)
                problem.setParameter(0, parameter)

                problem.performSolve()

                Z1[j, i] = problem.getCumulativeResponseContribution(0, 0)
                Z2[j, i] = problem.getCumulativeResponseContribution(0, 1)

        np.savetxt('Z1_2.txt', Z1)
        np.savetxt('Z2_2.txt', Z2)
    return Z1, Z2


def main(parallelEnv):
    comm = MPI.COMM_WORLD
    myGlobalRank = comm.rank

    # Create an Albany problem:

    n_params = 2
    filename = "thermal_steady_2.yaml"

    parameter = Utils.createParameterList(
        filename, parallelEnv
    )
    problem = Utils.createAlbanyProblem(parameter, parallelEnv)

    # ----------------------------------------------
    #
    #      1. Evaluation of the theta star
    #
    # ----------------------------------------------

    l_min = 8.
    l_max = 20.
    n_l = 5

    p = 1.

    l = l_min + np.power(np.linspace(0.0, 1.0, n_l), p) * (l_max-l_min)

    theta_star, I_star, F_star, P_star = ee.evaluateThetaStar(l, problem, n_params)

    np.savetxt('theta_star_steady_2.txt', theta_star)
    np.savetxt('I_star_steady_2.txt', I_star)
    np.savetxt('P_star_steady_2.txt', P_star)
    np.savetxt('F_star_steady_2.txt', F_star)

    # ----------------------------------------------
    #
    #   2. Evaluation of the prefactor using IS
    #
    # ----------------------------------------------

    N_samples = 10

    mean = np.array([1., 1.])
    cov = np.array([[1., 0.], [0., 1.]])

    samples = np.random.multivariate_normal(mean, cov, N_samples)

    angle_1 = 0.49999*np.pi
    angle_2 = np.pi - angle_1

    P_IS = ee.importanceSamplingEstimator(mean, cov, theta_star, F_star, P_star, samples, problem)
    P_mixed = ee.mixedImportanceSamplingEstimator(mean, cov, theta_star, F_star, P_star, samples, problem, angle_1, angle_2)
    P_SO = ee.secondOrderEstimator(mean, cov, l, theta_star, I_star, F_star, P_star, problem)

    np.savetxt('P_steady_IS_2.txt', P_IS)
    np.savetxt('P_steady_mixed_2.txt', P_mixed)
    np.savetxt('P_steady_SO_2.txt', P_SO)

    problem.reportTimers()

    # ----------------------------------------------
    #
    #   3.               Plots
    #
    # ----------------------------------------------
    if n_params == 2:
        X = np.arange(1, 7, 0.2)
        Y = np.arange(1, 7, 0.25)

        Z1, Z2 = evaluate_responses(X, Y, problem, True)

        X, Y = np.meshgrid(X, Y)

    if myGlobalRank == 0:
        if printPlot:
            plt.figure()
            plt.semilogy(F_star, P_star, 'k*-')
            plt.semilogy(F_star, P_IS, 'b*-')
            plt.semilogy(F_star, P_mixed, 'r*--')
            plt.semilogy(F_star, P_SO, 'g*-')

            plt.savefig('extreme_steady_2.jpeg', dpi=800)
            plt.close()

            if n_params == 2:
                plt.figure()
                plt.plot(theta_star[:, 0], theta_star[:, 1], '*-')
                plt.contour(X, Y, Z1, levels=I_star, colors='g')
                plt.contour(X, Y, Z2, levels=F_star, colors='r')
                plt.savefig('theta_star_2.jpeg', dpi=800)
                plt.close()

                fig = plt.figure()
                ax = fig.gca(projection='3d')
                ax.plot_surface(X, Y, Z1)

                plt.savefig('Z1_2.jpeg', dpi=800)
                plt.close()

                fig = plt.figure()
                ax = fig.gca(projection='3d')
                ax.plot_surface(X, Y, Z2)

                plt.savefig('Z2_2.jpeg', dpi=800)
                plt.close()


if __name__ == "__main__":
    comm = pa.getTeuchosComm(MPI.COMM_WORLD)
    parallelEnv = Utils.createDefaultParallelEnv(comm)
    main(parallelEnv)
