from PyTrilinos import Tpetra
from PyTrilinos import Teuchos

import unittest
import numpy as np
from PyAlbany import Utils
import os

class TestIO(unittest.TestCase):
    def test_write_distributed_npy(self):
        comm = Teuchos.DefaultComm.getComm()
        rank = comm.getRank()
        nproc = comm.getSize()
        if nproc > 1:
            mvector_filename = 'out_mvector_write_test_' + str(nproc)
        else:
            mvector_filename ='out_mvector_write_test'

        file_dir = os.path.dirname(__file__)

        filename = 'input.yaml'
        problem = Utils.createAlbanyProblem(file_dir+'/'+filename)

        n_cols = 4
        parameter_map = problem.getParameterMap(0)
        mvector = Tpetra.MultiVector(parameter_map, n_cols, dtype="d")

        mvector[0,:] = 1.*(rank+1)
        mvector[1,:] = -1.*(rank+1)
        mvector[2,:] = 3.26*(rank+1)
        mvector[3,:] = -3.1*(rank+1)

        Utils.writeMVector(file_dir+'/'+mvector_filename, mvector, distributedFile = True, useBinary = True)

    def test_write_distributed_txt(self):
        comm = Teuchos.DefaultComm.getComm()
        rank = comm.getRank()
        nproc = comm.getSize()
        if nproc > 1:
            mvector_filename = 'out_mvector_write_test_' + str(nproc)
        else:
            mvector_filename ='out_mvector_write_test'

        file_dir = os.path.dirname(__file__)

        filename = 'input.yaml'
        problem = Utils.createAlbanyProblem(file_dir+'/'+filename)

        n_cols = 4
        parameter_map = problem.getParameterMap(0)
        mvector = Tpetra.MultiVector(parameter_map, n_cols, dtype="d")

        mvector[0,:] = 1.*(rank+1)
        mvector[1,:] = -1.*(rank+1)
        mvector[2,:] = 3.26*(rank+1)
        mvector[3,:] = -3.1*(rank+1)

        Utils.writeMVector(file_dir+'/'+mvector_filename, mvector, distributedFile = True, useBinary = False)

    def test_write_non_distributed_npy(self):
        comm = Teuchos.DefaultComm.getComm()
        rank = comm.getRank()
        nproc = comm.getSize()
        if nproc > 1:
            mvector_filename = 'out_mvector_write_test_' + str(nproc)
        else:
            mvector_filename ='out_mvector_write_test'

        file_dir = os.path.dirname(__file__)

        filename = 'input.yaml'
        problem = Utils.createAlbanyProblem(file_dir+'/'+filename)

        n_cols = 4
        parameter_map = problem.getParameterMap(0)
        mvector = Tpetra.MultiVector(parameter_map, n_cols, dtype="d")

        mvector[0,:] = 1.*(rank+1)
        mvector[1,:] = -1.*(rank+1)
        mvector[2,:] = 3.26*(rank+1)
        mvector[3,:] = -3.1*(rank+1)

        Utils.writeMVector(file_dir+'/'+mvector_filename, mvector, distributedFile = False, useBinary = True)

    def test_write_non_distributed_txt(self):
        comm = Teuchos.DefaultComm.getComm()
        rank = comm.getRank()
        nproc = comm.getSize()
        if nproc > 1:
            mvector_filename = 'out_mvector_write_test_' + str(nproc)
        else:
            mvector_filename ='out_mvector_write_test'

        file_dir = os.path.dirname(__file__)

        filename = 'input.yaml'
        problem = Utils.createAlbanyProblem(file_dir+'/'+filename)

        n_cols = 4
        parameter_map = problem.getParameterMap(0)
        mvector = Tpetra.MultiVector(parameter_map, n_cols, dtype="d")

        mvector[0,:] = 1.*(rank+1)
        mvector[1,:] = -1.*(rank+1)
        mvector[2,:] = 3.26*(rank+1)
        mvector[3,:] = -3.1*(rank+1)

        Utils.writeMVector(file_dir+'/'+mvector_filename, mvector, distributedFile = False, useBinary = False)

    def test_read_distributed_npy(self):
        comm = Teuchos.DefaultComm.getComm()
        rank = comm.getRank()
        nproc = comm.getSize()
        if nproc > 1:
            mvector_filename = 'in_mvector_read_test_' + str(nproc)
        else:
            mvector_filename ='in_mvector_read_test'

        file_dir = os.path.dirname(__file__)

        filename = 'input.yaml'
        problem = Utils.createAlbanyProblem(file_dir+'/'+filename)

        n_cols = 4
        parameter_map = problem.getParameterMap(0)

        mvector = Utils.loadMVector(file_dir+'/'+mvector_filename, n_cols, parameter_map, distributedFile = True)

        tol = 1e-8
        mvector_target = np.array([1., -1, 3.26, -3.1])*(rank+1)
        for i in range(0, n_cols):
            self.assertTrue(np.abs(mvector[i,0]-mvector_target[i]) < tol)

    def test_read_distributed_txt(self):
        comm = Teuchos.DefaultComm.getComm()
        rank = comm.getRank()
        nproc = comm.getSize()
        if nproc > 1:
            mvector_filename = 'in_mvector_read_test_' + str(nproc)
        else:
            mvector_filename ='in_mvector_read_test'

        file_dir = os.path.dirname(__file__)

        filename = 'input.yaml'
        problem = Utils.createAlbanyProblem(file_dir+'/'+filename)

        n_cols = 4
        parameter_map = problem.getParameterMap(0)

        mvector = Utils.loadMVector(file_dir+'/'+mvector_filename, n_cols, parameter_map, distributedFile = True, useBinary = False)

        tol = 1e-8
        mvector_target = np.array([1., -1, 3.26, -3.1])*(rank+1)
        for i in range(0, n_cols):
            self.assertTrue(np.abs(mvector[i,0]-mvector_target[i]) < tol)

    def test_read_non_distributed_npy(self):
        comm = Teuchos.DefaultComm.getComm()
        rank = comm.getRank()
        nproc = comm.getSize()
        if nproc > 1:
            mvector_filename = 'in_mvector_read_test_' + str(nproc)
        else:
            mvector_filename ='in_mvector_read_test'

        file_dir = os.path.dirname(__file__)

        filename = 'input.yaml'
        problem = Utils.createAlbanyProblem(file_dir+'/'+filename)

        n_cols = 4
        parameter_map = problem.getParameterMap(0)

        mvector = Utils.loadMVector(file_dir+'/'+mvector_filename, n_cols, parameter_map, distributedFile = False)

        tol = 1e-8
        mvector_target = np.array([1., -1, 3.26, -3.1])*(rank+1)
        for i in range(0, n_cols):
            self.assertTrue(np.abs(mvector[i,0]-mvector_target[i]) < tol)

    def test_read_non_distributed_txt(self):
        comm = Teuchos.DefaultComm.getComm()
        rank = comm.getRank()
        nproc = comm.getSize()
        if nproc > 1:
            mvector_filename = 'in_mvector_read_test_' + str(nproc)
        else:
            mvector_filename ='in_mvector_read_test'

        file_dir = os.path.dirname(__file__)

        filename = 'input.yaml'
        problem = Utils.createAlbanyProblem(file_dir+'/'+filename)

        n_cols = 4
        parameter_map = problem.getParameterMap(0)

        mvector = Utils.loadMVector(file_dir+'/'+mvector_filename, n_cols, parameter_map, distributedFile = False, useBinary = False)

        tol = 1e-8
        mvector_target = np.array([1., -1, 3.26, -3.1])*(rank+1)
        for i in range(0, n_cols):
            self.assertTrue(np.abs(mvector[i,0]-mvector_target[i]) < tol)

    def test_read_non_distributed_non_scattered_npy(self):
        comm = Teuchos.DefaultComm.getComm()
        rank = comm.getRank()
        nproc = comm.getSize()
        if nproc > 1:
            mvector_filename = 'in_mvector_read_test_' + str(nproc)
        else:
            mvector_filename ='in_mvector_read_test'

        file_dir = os.path.dirname(__file__)

        filename = 'input.yaml'
        problem = Utils.createAlbanyProblem(file_dir+'/'+filename)

        n_cols = 4
        parameter_map = problem.getParameterMap(0)

        mvector = Utils.loadMVector(file_dir+'/'+mvector_filename, n_cols, parameter_map, distributedFile = False, readOnRankZero = False)

        tol = 1e-8
        mvector_target = np.array([1., -1, 3.26, -3.1])*(rank+1)
        for i in range(0, n_cols):
            self.assertTrue(np.abs(mvector[i,0]-mvector_target[i]) < tol)

    def test_read_non_distributed_non_scattered_txt(self):
        comm = Teuchos.DefaultComm.getComm()
        rank = comm.getRank()
        nproc = comm.getSize()
        if nproc > 1:
            mvector_filename = 'in_mvector_read_test_' + str(nproc)
        else:
            mvector_filename ='in_mvector_read_test'

        file_dir = os.path.dirname(__file__)

        filename = 'input.yaml'
        problem = Utils.createAlbanyProblem(file_dir+'/'+filename)

        n_cols = 4
        parameter_map = problem.getParameterMap(0)

        mvector = Utils.loadMVector(file_dir+'/'+mvector_filename, n_cols, parameter_map, distributedFile = False, useBinary = False, readOnRankZero = False)

        tol = 1e-8
        mvector_target = np.array([1., -1, 3.26, -3.1])*(rank+1)
        for i in range(0, n_cols):
            self.assertTrue(np.abs(mvector[i,0]-mvector_target[i]) < tol)


if __name__ == '__main__':
    unittest.main()