import unittest
import numpy as np
import os
from mpi4py import MPI
from PyAlbany import Utils
from PyAlbany import AlbanyInterface as pa
from PyAlbany.RandomizedCompression import doublePass


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

class TestDoublePass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parallelEnv = Utils.createDefaultParallelEnv()
        cls.comm = cls.parallelEnv.getComm()

    def test_all(self):
        cls = self.__class__
        rank    = cls.comm.getRank()
        nProcs  = cls.comm.getSize()
        iAmRoot = rank == 0
        fileDir = os.path.dirname(__file__)

        # Create an Albany problem:
        fileName = 'input_conductivity_dist_param.yaml'
        pList    = Utils.createParameterList(fileDir+'/'+fileName, cls.parallelEnv)
        pList.sublist("Problem").sublist("Response Functions").sublist("Response 0").sublist("Response 1").set("Scaling", 0.e0)
        pList.sublist("Discretization").set("1D Elements", 15)
        pList.sublist("Discretization").set("2D Elements", 15)
        problem = Utils.createAlbanyProblem(pList, cls.parallelEnv)
        problem.performSolve()
 
        parameterIndex = 0
        responseIndex  = 0
        parameterMap = problem.getParameterMap(parameterIndex)
        N            = parameterMap.getGlobalNumElements()
        h            = Utils.loadMVector(fileDir+'/Hess', N, parameterMap, distributedFile=False, useBinary=True)

        Hess = Hessian(problem, parameterIndex, responseIndex)
        k = 10
        p = 10
        r = k + p
        u, sigVals, v = doublePass(Hess, r, symmetric=False)
        H = pa.gatherMVector(h, parameterMap)
        U = pa.gatherMVector(u, parameterMap)
        V = pa.gatherMVector(v, parameterMap)
        # H \approx U \Lambda U^T
        if iAmRoot:
            Htilde = U.getLocalView().dot(np.diag(sigVals).dot(V.getLocalView().T))
            error  = np.linalg.norm(Htilde[:,:] - H.getLocalView())
            sigValsTrue = np.loadtxt(fileDir+"/singularvaluesTrue.txt")
            # see equation (5) of "Compressing rank-structured matrices via randomized sampling" Martinsson (2016)
            # for the error bound 
            errorBound = (1. + 11.*np.sqrt(k+p)*np.sqrt(N))*sigValsTrue[k+1]
            self.assertLessEqual(error, errorBound)


        stackedTimer = problem.getStackedTimer()
        setup_time = stackedTimer.accumulatedTime("PyAlbany: Setup Time")
        print("setup_time = " + str(setup_time))


    @classmethod
    def tearDownClass(cls):
        cls.parallelEnv = None
        cls.comm = None

if __name__ == '__main__':
    unittest.main()
