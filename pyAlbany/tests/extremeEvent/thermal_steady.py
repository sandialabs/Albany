import numpy as np
from mpi4py import MPI
from PyAlbany import Utils
from PyAlbany import ExtremeEvent as ee
import os
import sys

import unittest

class TestExtremeEvent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parallelEnv = Utils.createDefaultParallelEnv()
        cls.comm = cls.parallelEnv.getComm()

    def test_all(self):
        debug = True
        cls = self.__class__
        myGlobalRank = cls.comm.getRank()
        nproc = cls.comm.getSize()

        # Create an Albany problem:

        n_params = 2
        filename = "thermal_steady.yaml"

        parameter = Utils.createParameterList(
            filename, cls.parallelEnv
        )
        problem = Utils.createAlbanyProblem(parameter, cls.parallelEnv)

        # ----------------------------------------------
        #
        #      1. Evaluation of the theta star
        #
        # ----------------------------------------------

        l_min = 0.
        l_max = 2.
        n_l = 3

        l = np.linspace(l_min, l_max, n_l)

        theta_star, I_star, F_star, P_star = ee.evaluateThetaStar(l, problem, n_params)

        # ----------------------------------------------
        #
        #   2. Evaluation of the prefactor using IS
        #
        # ----------------------------------------------

        N_samples = 10

        mean = np.array([1., 1.])
        cov = np.array([[1., 0.], [0., 1.]])

        np.random.seed(41)
        samples = np.random.multivariate_normal(mean, cov, N_samples)

        angle_1 = 0.49999*np.pi
        angle_2 = np.pi - angle_1

        P_IS = ee.importanceSamplingEstimator(mean, cov, theta_star, F_star, P_star, samples, problem)
        P_mixed = ee.mixedImportanceSamplingEstimator(mean, cov, theta_star, F_star, P_star, samples, problem, angle_1, angle_2)

        if myGlobalRank == 0:
            expected_theta_star = np.loadtxt('expected_theta_star_steady_'+str(nproc)+'.txt')
            expected_I_star = np.loadtxt('expected_I_star_steady_'+str(nproc)+'.txt')
            expected_P_star = np.loadtxt('expected_P_star_steady_'+str(nproc)+'.txt')
            expected_F_star = np.loadtxt('expected_F_star_steady_'+str(nproc)+'.txt')
            expected_P_IS = np.loadtxt('expected_P_steady_IS_'+str(nproc)+'.txt')
            expected_P_mixed = np.loadtxt('expected_P_steady_mixed_'+str(nproc)+'.txt')

            tol = 5e-8
            tol_F = 5e-5

            if debug:
                for i in range(0, len(expected_theta_star)):
                    print('i = ' + str(i) + ': theta star: expected value = ' + str(expected_theta_star[i]) + ', computed value = ' + str(theta_star[i]) + ', and diff = ' + str(expected_theta_star[i]-theta_star[i]))
                    print('i = ' + str(i) + ': I star: expected value = ' + str(expected_I_star[i]) + ', computed value = ' + str(I_star[i]) + ', and diff = ' + str(expected_I_star[i]-I_star[i]))
                    print('i = ' + str(i) + ': P star: expected value = ' + str(expected_P_star[i]) + ', computed value = ' + str(P_star[i]) + ', and diff = ' + str(expected_P_star[i]-P_star[i]))
                    print('i = ' + str(i) + ': F star: expected value = ' + str(expected_F_star[i]) + ', computed value = ' + str(F_star[i]) + ', and diff = ' + str(expected_F_star[i]-F_star[i]))
                    print('i = ' + str(i) + ': P IS: expected value = ' + str(expected_P_IS[i]) + ', computed value = ' + str(P_IS[i]) + ', and diff = ' + str(expected_P_IS[i]-P_IS[i]))
                    print('i = ' + str(i) + ': P mixed: expected value = ' + str(expected_P_mixed[i]) + ', computed value = ' + str(P_mixed[i]) + ', and diff = ' + str(expected_P_mixed[i]-P_mixed[i]))

            self.assertLess(np.amax(np.abs(expected_theta_star - theta_star)), tol)
            self.assertLess(np.amax(np.abs(expected_I_star - I_star)), tol)
            self.assertLess(np.amax(np.abs(expected_P_star - P_star)), tol)
            self.assertLess(np.amax(np.abs(expected_F_star - F_star)), tol_F)
            self.assertLess(np.amax(np.abs(expected_P_IS - P_IS)), tol)
            self.assertLess(np.amax(np.abs(expected_P_mixed - P_mixed)), tol)

    @classmethod
    def tearDownClass(cls):
        cls.parallelEnv = None
        cls.comm = None


if __name__ == "__main__":
    unittest.main()
