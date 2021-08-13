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

        parameter_map    = problem.getParameterMap(0)
        parameter        = Tpetra.MultiVector(parameter_map, 1, dtype="d")
        num_elems        = parameter_map.getNodeNumElements()
        parameter[0, :]  = 2.0*np.ones(num_elems)
    

        problem.performSolve()
        state_map    = problem.getStateMap()
        state        = Tpetra.MultiVector(state_map, 1, dtype="d")
        state[0, :]  = problem.getState()
        state_ref    = Utils.loadMVector('state_ref', 1, state_map, distributedFile=False, useBinary=False, readOnRankZero=True)
        

        stackedTimer = problem.getStackedTimer()
        setup_time = stackedTimer.accumulatedTime("PyAlbany: Setup Time")
        print("setup_time = " + str(setup_time))
        tol = 1.e-8
        self.assertTrue(np.linalg.norm(state_ref[0, :] - state[0,:]) < tol)


    @classmethod
    def tearDownClass(cls):
        cls.parallelEnv = None
        cls.comm = None

if __name__ == '__main__':
    unittest.main()
