import unittest
import numpy as np
from mpi4py import MPI
from PyAlbany import Utils
import os


class TestSteadyHeat(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parallelEnv = Utils.createDefaultParallelEnv()
        cls.comm = cls.parallelEnv.getComm()

    def test_all(self):
        cls = self.__class__
        rank = cls.comm.getRank()

        file_dir = os.path.dirname(__file__)

        # Create an Albany problem:
        filename = "input_dirichlet_mixed_paramsT.yaml"
        parameter = Utils.createParameterList(
            file_dir + "/" + filename, cls.parallelEnv
        )

        parameter.sublist("Discretization").set("1D Elements", 10)
        parameter.sublist("Discretization").set("2D Elements", 10)

        problem = Utils.createAlbanyProblem(parameter, cls.parallelEnv)

        g_target_before = 0.35681844
        g_target_after = 0.17388281
        g_target_2 = 0.1957233
        p_0_target = 0.39855677
        p_1_norm_target = 5.366834867170422
        tol = 5e-8

        problem.performSolve()

        response_before_analysis = problem.getResponse(0)
        response_before_analysis_view = response_before_analysis.getLocalView()

        problem.performAnalysis()

        para_0 = problem.getParameter(0)
        para_1 = problem.getParameter(1)

        para_0_view = para_0.getLocalView()
        para_1_view = para_1.getLocalView()

        print(para_0_view)
        print(para_1_view)

        para_1_norm = Utils.norm(para_1)
        print(para_1_norm)

        if rank == 0:
            self.assertTrue(np.abs(para_0_view[0] - p_0_target) < tol)
            self.assertTrue(np.abs(para_1_norm - p_1_norm_target) < tol)

        problem.performSolve()

        response_after_analysis = problem.getResponse(0)
        response_after_analysis_view = response_after_analysis.getLocalView()

        print("Response before analysis " + str(response_before_analysis_view))
        print("Response after analysis " + str(response_after_analysis_view))
        if rank == 0:
            self.assertTrue(np.abs(response_before_analysis_view[0] - g_target_before) < tol)
            self.assertTrue(np.abs(response_after_analysis_view[0] - g_target_after) < tol)

        parameter_map_0 = problem.getParameterMap(0)
        para_0_new = Utils.createVector(parameter_map_0)
        problem.setParameter(0, para_0_new)

        problem.performSolve()

        response = problem.getResponse(0)
        response_view = response.getLocalView()
        print("Response after setParameter " + str(response_view))
        if rank == 0:
            self.assertTrue(np.abs(response_view[0] - g_target_2) < tol)

    @classmethod
    def tearDownClass(cls):
        cls.parallelEnv = None
        cls.comm = None


if __name__ == "__main__":
    unittest.main()
