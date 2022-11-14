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

        parameter_map_0 = problem.getParameterMap(0)
        para_0_new = Utils.createVector(parameter_map_0)

        para_0_new_view = para_0_new.getLocalView()

        parameter_map_1 = problem.getParameterMap(1)
        para_1_new = Utils.createVector(parameter_map_1)

        para_1_new_view = para_1_new.getLocalView()
        para_1_new_view[:] = 0.333333
        para_1_new.setLocalView(para_1_new_view)


        n_values = 5
        para_0_values = np.linspace(-1, 1, n_values)
        responses = np.zeros((n_values,))

        responses_target = np.array(
            [0.69247527, 0.48990929, 0.35681844, 0.29320271, 0.2990621]
        )
        tol = 1e-8

        for i in range(0, n_values):
            para_0_new_view[0] = para_0_values[i]
            para_0_new.setLocalView(para_0_new_view)
            problem.setParameter(0, para_0_new)

            problem.performSolve()

            responses[i] = problem.getResponse(0).getLocalView()[0]

        print("p = " + str(para_0_values))
        print("QoI = " + str(responses))

        if rank == 0:
            self.assertTrue(np.abs(np.amax(responses - responses_target)) < tol)

    @classmethod
    def tearDownClass(cls):
        cls.parallelEnv = None
        cls.comm = None


if __name__ == "__main__":
    unittest.main()