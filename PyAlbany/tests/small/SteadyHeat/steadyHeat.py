from PyTrilinos import Tpetra
from PyTrilinos import Teuchos

import unittest
import numpy as np
from PyAlbany import Utils
import os

class TestSteadyHeat(unittest.TestCase):
    def test_all(self):
        comm = Teuchos.DefaultComm.getComm()
        rank = comm.getRank()

        file_dir = os.path.dirname(__file__)

        # Create an Albany problem:
        filename = 'input_conductivity_dist_paramT.yaml'
        problem = Utils.createAlbanyProblem(file_dir+'/'+filename)

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
        h_target = np.array([4.2121719763904516e-05, -4.21216874727712e-05, 0.00012636506241831498, -0.00012636506241831496])

        g_data = response.getData()
        norm = Utils.norm(sensitivity.getData(0), comm)

        print("g_target = " + str(g_target))
        print("g_data[0] = " + str(g_data[0]))
        print("norm = " + str(norm))
        print("norm_target = " + str(norm_target))

        tol = 1e-8
        if rank == 0:
            self.assertTrue(np.abs(g_data[0]-g_target) < tol)
            self.assertTrue(np.abs(norm-norm_target) < tol)
            for i in range(0,n_directions):
                self.assertTrue(np.abs(hessian[i,0]-h_target[i]) < tol)

if __name__ == '__main__':
    unittest.main()