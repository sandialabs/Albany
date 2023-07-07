import numpy as np
from mpi4py import MPI
from PyAlbany import Utils
from PyAlbany import AlbanyInterface as pa
from PyAlbany.RandomizedCompression import singlePass


class Hessian:
   def __init__(me, problem, parameterIndex, responseIndex):
       me.problem = problem
       me.parameterIndex = parameterIndex
       me.responseIndex  = responseIndex
       me.Map            = me.problem.getParameterMap(me.parameterIndex)
   def dot(me, x):
       me.problem.setDirections(me.parameterIndex, x)
       me.problem.performSolve()
       return me.problem.getReducedHessian(me.responseIndex, me.parameterIndex)

import unittest

class TestAdjointSolve(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parallelEnv = Utils.createDefaultParallelEnv()
        cls.comm = cls.parallelEnv.getComm()

    def test_all(self):
        debug = True
        cls = self.__class__
        myGlobalRank = cls.comm.getRank()
        iAmRoot = myGlobalRank == 0

        np.random.seed(42)

        # Create an Albany problem:
        fileName = 'inputT_MueLu.yaml'
        pList    = Utils.createParameterList(fileName, cls.parallelEnv)
        problem = Utils.createAlbanyProblem(pList, cls.parallelEnv)
        problem.performSolve()

        parameterIndex = 0
        responseIndex  = 0
    
        Hess = Hessian(problem, parameterIndex, responseIndex)
        k = 1
        p = 0
        r = k + p
        eigVals, eigVecs = singlePass(Hess, r)

        if myGlobalRank == 0:
            tol = 5e-7

            print(eigVals)
            print(eigVecs.getLocalView())

            expected_eigVals = 0.85552097
            expected_eigVecs = np.array([[0.43910025], [0.36382469], [0.38869418], [0.44107581], [0.57375215]])

            self.assertTrue(np.abs(eigVals - expected_eigVals) < tol)
            self.assertTrue(np.amax(np.abs(eigVecs.getLocalView() - expected_eigVecs)) < tol)

    @classmethod
    def tearDownClass(cls):
        cls.parallelEnv = None
        cls.comm = None


if __name__ == "__main__":
    unittest.main()
