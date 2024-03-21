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

        file_dir = os.path.dirname(__file__)

        # Create an Albany problem:
        filename = 'input_conductivity_dist_paramT.yaml'
        problem = Utils.createAlbanyProblem(file_dir+'/'+filename, cls.parallelEnv)

        parameter_map    = problem.getParameterMap(0)
        parameter        = Utils.createMultiVector(parameter_map, 1)
        num_elems        = parameter_map.getLocalNumElements()

        parameter_view = parameter.getLocalView()

        parameter_view[:] = 2.0*np.ones((num_elems, 1))

        parameter.setLocalView(parameter_view)
    

        problem.performSolve()
        state_map    = problem.getStateMap()
        state        = problem.getState()
        state_ref    = Utils.loadMVector('state_ref', 1, state_map, distributedFile=False, useBinary=False).getVector(0)
        

        stackedTimer = problem.getStackedTimer()
        setup_time = stackedTimer.accumulatedTime("PyAlbany: Setup Time")
        print("setup_time = " + str(setup_time))
        tol = 1.e-8
        state_view = state.getLocalView()
        state_ref_view = state_ref.getLocalView()
        self.assertTrue(np.linalg.norm(state_ref_view - state_view) < tol)


    @classmethod
    def tearDownClass(cls):
        cls.parallelEnv = None
        cls.comm = None

if __name__ == '__main__':
    unittest.main()
