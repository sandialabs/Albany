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
        filename = "thermal_steady_hessian.yaml"

        parameter = Utils.createParameterList(
            filename, cls.parallelEnv
        )
        problem = Utils.createAlbanyProblem(parameter, cls.parallelEnv)

        # ----------------------------------------------
        #
        #      1. Evaluation of the theta star
        #
        # ----------------------------------------------

        l_min = 1.
        l_max = 2.
        n_l = 4

        p = 0.25

        l = l_min + np.power(np.linspace(0.0, 1.0, n_l), p) * (l_max-l_min)

        theta_star, I_star, F_star, P_star = ee.evaluateThetaStar(l, problem, n_params)

        # ----------------------------------------------
        #
        #   2. Evaluation of the prefactor using SO
        #
        # ----------------------------------------------

        mean = np.array([1., 1.])
        cov = np.array([[1., 0.], [0., 1.]])
        
        P_SO = ee.secondOrderEstimator(mean, cov, l, theta_star, I_star, F_star, P_star, problem)

        if myGlobalRank == 0:
            expected_theta_star = np.loadtxt('expected_theta_star_steady_hessian_'+str(nproc)+'.txt')
            expected_I_star = np.loadtxt('expected_I_star_steady_hessian_'+str(nproc)+'.txt')
            expected_P_star = np.loadtxt('expected_P_star_steady_hessian_'+str(nproc)+'.txt')
            expected_F_star = np.loadtxt('expected_F_star_steady_hessian_'+str(nproc)+'.txt')
            expected_P_SO = np.loadtxt('expected_P_steady_hessian_SO_'+str(nproc)+'.txt')

            tol = 5e-6

            if debug:
                for i in range(0, len(expected_theta_star)):
                    print('i = ' + str(i) + ': theta star: expected value = ' + str(expected_theta_star[i]) + ', computed value = ' + str(theta_star[i]) + ', and diff = ' + str(expected_theta_star[i]-theta_star[i]))
                    print('i = ' + str(i) + ': I star: expected value = ' + str(expected_I_star[i]) + ', computed value = ' + str(I_star[i]) + ', and diff = ' + str(expected_I_star[i]-I_star[i]))
                    print('i = ' + str(i) + ': P star: expected value = ' + str(expected_P_star[i]) + ', computed value = ' + str(P_star[i]) + ', and diff = ' + str(expected_P_star[i]-P_star[i]))
                    print('i = ' + str(i) + ': F star: expected value = ' + str(expected_F_star[i]) + ', computed value = ' + str(F_star[i]) + ', and diff = ' + str(expected_F_star[i]-F_star[i]))
                    print('i = ' + str(i) + ': P SO: expected value = ' + str(expected_P_SO[i]) + ', computed value = ' + str(P_SO[i]) + ', and diff = ' + str(expected_P_SO[i]-P_SO[i]))

            self.assertTrue(np.amax(np.abs(expected_theta_star - theta_star)) < tol)
            self.assertTrue(np.amax(np.abs(expected_I_star - I_star)) < tol)
            self.assertTrue(np.amax(np.abs(expected_P_star - P_star)) < tol)
            self.assertTrue(np.amax(np.abs(expected_F_star - F_star)) < tol)
            self.assertTrue(np.amax(np.abs(expected_P_SO - P_SO)) < tol)

    @classmethod
    def tearDownClass(cls):
        cls.parallelEnv = None
        cls.comm = None


if __name__ == "__main__":
    unittest.main()
