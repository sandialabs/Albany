import unittest
import numpy as np
from mpi4py import MPI
from PyAlbany import Utils
from PyAlbany import AlbanyInterface as pa
import os

class TestMatrixOperations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parallelEnv = Utils.createDefaultParallelEnv()
        cls.comm = cls.parallelEnv.getComm()

    def test_all(self):
        cls = self.__class__
        rank = cls.comm.getRank()

        file_dir = os.path.dirname(__file__)

        # Create an Albany problem:
        filename = 'input_steadyHeat.yaml'
        problem = Utils.createAlbanyProblem(file_dir+'/'+filename, cls.parallelEnv)

        pMap = problem.getParameterMap(0)
        A = Utils.createCrsMatrix(pMap, "A.mm")

        #generate a multivector with random reproducible entries
        nVecs = 2;
        pMapRoot = pa.getRankZeroMap(pMap)
        vRoot = Utils.createMultiVector(pMapRoot, nVecs)
        if rank == 0:
           rng = np.random.default_rng(20240321)
           N = pMap.getGlobalNumElements()
           randVecs = rng.normal(size=(N, nVecs))
           vRoot.setLocalView(randVecs)
        v = pa.scatterMVector(vRoot, pMap)
        v0 = v.getVector(0)
        v1 =  v.getVector(1)

        tol0 = 1e-13
        tol = 1e-6

        #test mat product 
        #test 0: v1' A v0 = v0' A' v1
        Av0 = Utils.matVecProduct(A, v0)
        Atv1 = Utils.matVecProduct(A, v1, trans=True)
        err0 = Av0.dot(v1) - Atv1.dot(v0)
        if(rank == 0 and err0 >= tol0):
            print('ERROR!, test 0 error = ', err0, 'is greater than ', tol)


        #test solve and mat product:  

        w = Utils.createMultiVector(pMap, nVecs)
        w0 = Utils.createVector(pMap)
        w1 = Utils.createVector(pMap)
        solverOptions = Utils.createParameterList("solverOptions.yaml", cls.parallelEnv)

        #test 1: A^{-1} A v0 = v0
        Utils.solve(A,w0,Av0,solverOptions)
        #w0 = w0-v0
        w0.update(-1.0, v0, 1.0)
        err1 = Utils.norm(w0)
        if(rank ==0 and err1 >= tol):
            print('ERROR!, test 1 error = ', err1, 'is greater than ', tol)


        #test 2: (A')^{-1} A' v0 = v0
        Utils.solve(A,w1,Atv1,solverOptions,trans=True)
        #w1 = w1-v1
        w1.update(-1.0, v1, 1.0)
        err2 = Utils.norm(w1)
        if(rank ==0 and err2 >= tol):
            print('ERROR!, test 2 error = ', err2, 'is greater than ', tol)


        #test 3: A^{-1} A v = v   (multivector version)
        Av = Utils.matVecProduct(A, v)
        Utils.solve(A,w,Av,solverOptions,trans=False)
        #w=w-v
        w.update(-1.0, v, 1.0)
        err3 = max(Utils.norm(w.getVector(0)),Utils.norm(w.getVector(1)))
        if(rank ==0 and err3 >= tol):
            print('ERROR!, test 3 error = ', err3, 'is greater than ', tol)


        #test 4: (A')^{-1} A' v = v
        Atv = Utils.matVecProduct(A, v,trans=True)
        Utils.solve(A,w,Atv,solverOptions,trans=True,zeroInitGuess=False)
        #w=w-v
        w.update(-1.0, v, 1.0)
        err4 = max(Utils.norm(w.getVector(0)),Utils.norm(w.getVector(1)))
        if(rank == 0 and err4 >= tol):
            print('ERROR!, test 4 error = ', err4, 'is greater than ', tol)


        if rank == 0:
            self.assertLess(err0, tol0)
            self.assertLess(err1, tol)
            self.assertLess(err2, tol)
            self.assertLess(err3, tol)
            self.assertLess(err4, tol)

    @classmethod
    def tearDownClass(cls):
        cls.parallelEnv = None
        cls.comm = None

if __name__ == '__main__':
    unittest.main()
