from PyTrilinos import Tpetra
from PyTrilinos import Teuchos

import unittest
import numpy as np
try:
    from PyAlbany import Utils
except:
    import Utils
try:
    from PyAlbany import wpyalbany as wpa
except:
    import wpyalbany as wpa
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

        n_vecs = 4
        parameter_map = problem.getParameterMap(0)
        num_elems     = parameter_map.getLocalNumElements()
        
        # generate vectors with random entries
        omega = Tpetra.MultiVector(parameter_map, n_vecs, dtype="d")
        for i in range(n_vecs):
            omega[i,:] = np.random.randn(num_elems)
        
        # call the orthonormalization method
        wpa.orthogTpMVecs(omega, 2)
        
        # check that the vectors are now orthonormal
        tol = 1.e-12
        for i in range(n_vecs):
            for j in range(i+1):
                omegaiTomegaj = Utils.inner(omega[i,:], omega[j,:], cls.comm)
                if rank == 0:
                    if i == j:
                        self.assertTrue(abs(omegaiTomegaj - 1.0) < tol)
                    else:
                        self.assertTrue(abs(omegaiTomegaj-0.0) < tol)

    @classmethod
    def tearDownClass(cls):
        cls.parallelEnv = None
        cls.comm = None

if __name__ == '__main__':
    unittest.main()
