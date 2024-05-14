import unittest
import numpy as np
import os
from mpi4py import MPI
from PyAlbany import Utils
from PyAlbany import AlbanyInterface as pa
from PyAlbany.RandomizedCompression import HODLR, Hpartition


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

class TestHODLR(unittest.TestCase):
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
        H = pa.gatherMVector(h, parameterMap)

        Hess = Hessian(problem, parameterIndex, responseIndex)
        k = 10
        p = 10
        r = k + p
        L = 2
        us, sigs, vs = HODLR(Hess, L , r)
        UL0J0 = pa.gatherMVector(us[0][0], parameterMap)
        VL0J0 = pa.gatherMVector(vs[0][0], parameterMap)
        UL1J0 = pa.gatherMVector(us[1][0], parameterMap)
        VL1J0 = pa.gatherMVector(vs[1][0], parameterMap)
        Us = [UL0J0, UL1J0]
        Vs = [VL0J0, VL1J0]
        
        idxSet = Hpartition(N, L) 
        if iAmRoot:
            for l in range(L):
                sigValsTrue = np.loadtxt(fileDir+"/SigTrueL"+str(l)+"J0.txt")
                idx0 = idxSet[l][0][0]
                idx1 = idxSet[l][0][1]
                idx2 = idxSet[l][1][1]
                H12tilde = Us[l].getLocalView()[idx0:idx1,:].dot(np.diag(sigs[l][0]).dot(Vs[l].getLocalView()[idx1:idx2,:].T))
                error12 = np.linalg.norm(H12tilde - H.getLocalView()[idx1:idx2, idx0:idx1])
                # see equation (5) of "Compressing rank-structured matrices via randomized sampling" Martinsson (2016)
                # for the error bound, this error does not take into account errors
                # due to peeling
                M12 = idx1-idx0 # rows of OD-block
                N12 = idx2-idx1 # columns of OD-block
                errorBound12 = (1. + 11.*np.sqrt(k+p)*np.sqrt(min(M12, N12)))*sigValsTrue[k+1]
                self.assertLessEqual(error12, errorBound12)


        stackedTimer = problem.getStackedTimer()
        setup_time = stackedTimer.accumulatedTime("PyAlbany: Setup Time")
        print("setup_time = " + str(setup_time))


    @classmethod
    def tearDownClass(cls):
        cls.parallelEnv = None
        cls.comm = None

if __name__ == '__main__':
    unittest.main()
