from PyTrilinos import Tpetra
from PyTrilinos import Teuchos

import unittest
import numpy as np

try:
    from PyAlbany import Utils
except:
    import Utils
import os


class TestSteadyHeat(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.comm = Teuchos.DefaultComm.getComm()
        cls.parallelEnv = Utils.createDefaultParallelEnv(cls.comm)

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
        g_target_after = 0.17388298
        g_target_2 = 0.19570272
        p_0_target = 0.39886689
        p_1_norm_target = 5.37319376038225
        tol = 1e-8

        problem.performSolve()

        response_before_analysis = problem.getResponse(0)

        problem.performAnalysis()

        para_0 = problem.getParameter(0)
        para_1 = problem.getParameter(1)

        print(para_0.getData())
        print(para_1.getData())

        para_1_norm = Utils.norm(para_1.getData(), cls.comm)
        print(para_1_norm)

        if rank == 0:
            self.assertTrue(np.abs(para_0[0] - p_0_target) < tol)
            self.assertTrue(np.abs(para_1_norm - p_1_norm_target) < tol)

        problem.performSolve()

        response_after_analysis = problem.getResponse(0)

        print("Response before analysis " + str(response_before_analysis.getData()))
        print("Response after analysis " + str(response_after_analysis.getData()))
        if rank == 0:
            self.assertTrue(np.abs(response_before_analysis[0] - g_target_before) < tol)
            self.assertTrue(np.abs(response_after_analysis[0] - g_target_after) < tol)

        parameter_map_0 = problem.getParameterMap(0)
        para_0_new = Tpetra.Vector(parameter_map_0, dtype="d")
        para_0_new[:] = 0.0
        problem.setParameter(0, para_0_new)

        problem.performSolve()

        response = problem.getResponse(0)
        print("Response after setParameter " + str(response.getData()))
        if rank == 0:
            self.assertTrue(np.abs(response[0] - g_target_2) < tol)

    @classmethod
    def tearDownClass(cls):
        cls.parallelEnv = None
        cls.comm = None


if __name__ == "__main__":
    unittest.main()