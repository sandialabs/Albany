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
        filename = 'input_conductivity_dist_paramT.yaml'
        problem = Utils.createAlbanyProblem(file_dir+'/'+filename, cls.parallelEnv)

        n_directions = 4
        parameter_map = problem.getParameterMap(0)
        directions = Tpetra.MultiVector(parameter_map, n_directions, dtype="d")

        directions[0,:] = 1.
        directions[1,:] = -1.
        directions[2,:] = 3.
        directions[3,:] = -3.

        problem.setDirections(0, directions)

        problem.performSolve()

        response = problem.getResponse(0)
        sensitivity = problem.getSensitivity(0, 0)
        hessian = problem.getReducedHessian(0, 0)

        g_target = 3.23754626955999991e-01
        norm_target = 8.94463776843999921e-03
        h_target = np.array([0.009195356672103817, 0.009195356672103817, 0.027586070971800013, 0.027586070971800013])

        g_data = response.getData()
        norm = Utils.norm(sensitivity.getData(0), cls.comm)

        print("g_target = " + str(g_target))
        print("g_data[0] = " + str(g_data[0]))
        print("norm = " + str(norm))
        print("norm_target = " + str(norm_target))

        hessian_norms = np.zeros((n_directions,))
        for i in range(0,n_directions):
            hessian_norms[i] = Utils.norm(hessian.getData(i), cls.comm)

        tol = 1e-8
        if rank == 0:
            self.assertTrue(np.abs(g_data[0]-g_target) < tol)
            self.assertTrue(np.abs(norm-norm_target) < tol)
            for i in range(0,n_directions):
                self.assertTrue(np.abs(hessian_norms[i]-h_target[i]) < tol)

    @classmethod
    def tearDownClass(cls):
        cls.parallelEnv = None
        cls.comm = None

if __name__ == '__main__':
    unittest.main()