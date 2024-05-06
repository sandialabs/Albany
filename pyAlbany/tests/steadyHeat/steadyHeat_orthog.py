import unittest
import numpy as np
from mpi4py import MPI
from PyAlbany import Utils
from PyAlbany import AlbanyInterface as pa
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
        filename = 'input_conductivity_dist_param.yaml'
        problem = Utils.createAlbanyProblem(file_dir+'/'+filename, cls.parallelEnv)

        n_vecs = 4
        parameter_map = problem.getParameterMap(0)
        num_elems     = parameter_map.getLocalNumElements()
        
        # generate vectors with random entries
        omega = Utils.createMultiVector(parameter_map, n_vecs)
        omega_view = omega.getLocalView()
        for i in range(n_vecs):
            omega_view[:,i] = np.random.randn(num_elems)
        omega.setLocalView(omega_view)
        
        # call the orthonormalization method
        pa.orthogTpMVecs(omega, 2)
        
        # check that the vectors are now orthonormal
        tol = 1.e-12
        for i in range(n_vecs):
            for j in range(i+1):
                omegaiTomegaj = Utils.inner(omega.getVector(i), omega.getVector(j))
                if rank == 0:
                    if i == j:
                        self.assertLess(abs(omegaiTomegaj - 1.0), tol)
                    else:
                        self.assertLess(abs(omegaiTomegaj-0.0), tol)

    @classmethod
    def tearDownClass(cls):
        cls.parallelEnv = None
        cls.comm = None

if __name__ == '__main__':
    unittest.main()
