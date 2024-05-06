import unittest
import numpy as np
from mpi4py import MPI
from PyAlbany import Utils
import os

class TestIO(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parallelEnv = Utils.createDefaultParallelEnv()
        cls.comm = cls.parallelEnv.getComm()

    def test_write_distributed_npy(self):
        cls = self.__class__
        rank = cls.comm.getRank()
        nproc = cls.comm.getSize()
        
        if nproc == 1:
            return
        
        mvector_filename = 'out_mvector_write_test_' + str(nproc)
        file_dir = os.path.dirname(__file__)

        filename = 'input.yaml'
        problem = Utils.createAlbanyProblem(file_dir+'/'+filename, cls.parallelEnv)

        n_cols = 4
        parameter_map = problem.getParameterMap(0)
        mvector = Utils.createMultiVector(parameter_map, n_cols)

        mvector_view = mvector.getLocalView()
        mvector_view[:,0] = 1.*(rank+1)
        mvector_view[:,1] = -1.*(rank+1)
        mvector_view[:,2] = 3.26*(rank+1)
        mvector_view[:,3] = -3.1*(rank+1)
        mvector.setLocalView(mvector_view)

        Utils.writeMVector(file_dir+'/'+mvector_filename, mvector, distributedFile = True, useBinary = True)

    def test_write_distributed_txt(self):
        cls = self.__class__
        rank = cls.comm.getRank()
        nproc = cls.comm.getSize()
        
        if nproc == 1:
            return
        
        mvector_filename = 'out_mvector_write_test_' + str(nproc)
        file_dir = os.path.dirname(__file__)

        filename = 'input.yaml'
        problem = Utils.createAlbanyProblem(file_dir+'/'+filename, cls.parallelEnv)

        n_cols = 4
        parameter_map = problem.getParameterMap(0)
        mvector = Utils.createMultiVector(parameter_map, n_cols)

        mvector_view = mvector.getLocalView()
        mvector_view[:,0] = 1.*(rank+1)
        mvector_view[:,1] = -1.*(rank+1)
        mvector_view[:,2] = 3.26*(rank+1)
        mvector_view[:,3] = -3.1*(rank+1)
        mvector.setLocalView(mvector_view)

        Utils.writeMVector(file_dir+'/'+mvector_filename, mvector, distributedFile = True, useBinary = False)

    def test_write_non_distributed_npy(self):
        cls = self.__class__
        rank = cls.comm.getRank()
        nproc = cls.comm.getSize()
        if nproc > 1:
            mvector_filename = 'out_mvector_write_test_' + str(nproc)
        else:
            mvector_filename ='out_mvector_write_test'

        file_dir = os.path.dirname(__file__)

        filename = 'input.yaml'
        problem = Utils.createAlbanyProblem(file_dir+'/'+filename, cls.parallelEnv)

        n_cols = 4
        parameter_map = problem.getParameterMap(0)
        mvector = Utils.createMultiVector(parameter_map, n_cols)

        mvector_view = mvector.getLocalView()
        mvector_view[:,0] = 1.*(rank+1)
        mvector_view[:,1] = -1.*(rank+1)
        mvector_view[:,2] = 3.26*(rank+1)
        mvector_view[:,3] = -3.1*(rank+1)
        mvector.setLocalView(mvector_view)

        Utils.writeMVector(file_dir+'/'+mvector_filename, mvector, distributedFile = False, useBinary = True)

    def test_write_non_distributed_txt(self):
        cls = self.__class__
        rank = cls.comm.getRank()
        nproc = cls.comm.getSize()
        if nproc > 1:
            mvector_filename = 'out_mvector_write_test_' + str(nproc)
        else:
            mvector_filename ='out_mvector_write_test'

        file_dir = os.path.dirname(__file__)

        filename = 'input.yaml'
        problem = Utils.createAlbanyProblem(file_dir+'/'+filename, cls.parallelEnv)

        n_cols = 4
        parameter_map = problem.getParameterMap(0)
        mvector = Utils.createMultiVector(parameter_map, n_cols)

        mvector_view = mvector.getLocalView()
        mvector_view[:,0] = 1.*(rank+1)
        mvector_view[:,1] = -1.*(rank+1)
        mvector_view[:,2] = 3.26*(rank+1)
        mvector_view[:,3] = -3.1*(rank+1)
        mvector.setLocalView(mvector_view)

        Utils.writeMVector(file_dir+'/'+mvector_filename, mvector, distributedFile = False, useBinary = False)

    def test_read_distributed_npy(self):
        cls = self.__class__
        rank = cls.comm.getRank()
        nproc = cls.comm.getSize()
        
        if nproc == 1:
            return

        mvector_filename = 'in_mvector_read_test_' + str(nproc)
        file_dir = os.path.dirname(__file__)

        filename = 'input.yaml'
        problem = Utils.createAlbanyProblem(file_dir+'/'+filename, cls.parallelEnv)

        n_cols = 4
        parameter_map = problem.getParameterMap(0)

        mvector = Utils.loadMVector(file_dir+'/'+mvector_filename, n_cols, parameter_map, distributedFile = True)
        mvector_view = mvector.getLocalView()

        tol = 1e-8
        mvector_target = np.array([1., -1, 3.26, -3.1])*(rank+1)

        for i in range(0, n_cols):
            self.assertLess(np.abs(mvector_view[0,i]-mvector_target[i]), tol)

    def test_read_distributed_txt(self):
        cls = self.__class__
        rank = cls.comm.getRank()
        nproc = cls.comm.getSize()
        
        if nproc == 1:
            return

        mvector_filename = 'in_mvector_read_test_' + str(nproc)
        file_dir = os.path.dirname(__file__)

        filename = 'input.yaml'
        problem = Utils.createAlbanyProblem(file_dir+'/'+filename, cls.parallelEnv)

        n_cols = 4
        parameter_map = problem.getParameterMap(0)

        mvector = Utils.loadMVector(file_dir+'/'+mvector_filename, n_cols, parameter_map, distributedFile = True, useBinary = False)
        mvector_view = mvector.getLocalView()

        tol = 1e-8
        mvector_target = np.array([1., -1, 3.26, -3.1])*(rank+1)
        for i in range(0, n_cols):
            self.assertLess(np.abs(mvector_view[0,i]-mvector_target[i]), tol)

    def test_read_non_distributed_npy(self):
        cls = self.__class__
        rank = cls.comm.getRank()
        nproc = cls.comm.getSize()
        if nproc > 1:
            mvector_filename = 'in_mvector_read_test_' + str(nproc)
        else:
            mvector_filename ='in_mvector_read_test'

        file_dir = os.path.dirname(__file__)

        filename = 'input.yaml'
        problem = Utils.createAlbanyProblem(file_dir+'/'+filename, cls.parallelEnv)

        n_cols = 4
        parameter_map = problem.getParameterMap(0)

        mvector = Utils.loadMVector(file_dir+'/'+mvector_filename, n_cols, parameter_map, distributedFile = False)
        mvector_view = mvector.getLocalView()

        tol = 1e-8
        mvector_target = np.array([1., -1, 3.26, -3.1])*(rank+1)
        for i in range(0, n_cols):
            self.assertLess(np.abs(mvector_view[0,i]-mvector_target[i]), tol)

    def test_read_non_distributed_txt(self):
        cls = self.__class__
        rank = cls.comm.getRank()
        nproc = cls.comm.getSize()
        if nproc > 1:
            mvector_filename = 'in_mvector_read_test_' + str(nproc)
        else:
            mvector_filename ='in_mvector_read_test'

        file_dir = os.path.dirname(__file__)

        filename = 'input.yaml'
        problem = Utils.createAlbanyProblem(file_dir+'/'+filename, cls.parallelEnv)

        n_cols = 4
        parameter_map = problem.getParameterMap(0)

        mvector = Utils.loadMVector(file_dir+'/'+mvector_filename, n_cols, parameter_map, distributedFile = False, useBinary = False)
        mvector_view = mvector.getLocalView()

        tol = 1e-8
        mvector_target = np.array([1., -1, 3.26, -3.1])*(rank+1)
        for i in range(0, n_cols):
            self.assertLess(np.abs(mvector_view[0,i]-mvector_target[i]), tol)

    @classmethod
    def tearDownClass(cls):
        cls.parallelEnv = None
        cls.comm = None

if __name__ == '__main__':
    unittest.main()
