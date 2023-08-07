import numpy as np
from scipy.sparse import csr_matrix
import unittest
import subprocess
import os.path
from matrix_reader import fuse_blocks, read_matrix


class TestBlockJacobian(unittest.TestCase):
    def test_graph(self):
        n_blocks = 5
        # Remove previously computed files:
        if os.path.isfile('jac.txt'):
            os.remove('jac.txt')
        for i in range(0, n_blocks):
            for j in range(0, n_blocks):
                if os.path.isfile('b_jac_'+str(i)+'_'+str(j)+'.txt'):
                    os.remove('b_jac_'+str(i)+'_'+str(j)+'.txt')

        subprocess.call('./blockJacobian_unit_tester')

        row_indices, col_indices, values, nrows, ncols, nrows_per_block, ncols_per_block = fuse_blocks(
            n_blocks, n_blocks, "b_jac_")

        nnz_total = len(col_indices)

        A = csr_matrix(
            (values+1, (row_indices, col_indices)), shape=(nrows, ncols))

        row_indices, col_indices, values, nrows, ncols = read_matrix("jac.txt")

        jac = csr_matrix(
            (values+1, (row_indices, col_indices)), shape=(nrows, ncols))

        nnz_jac = len(col_indices)

        tol = 1e-8

        self.assertEqual(nnz_jac, nnz_total)
        self.assertTrue(np.amax(np.abs(A-jac)) < tol)


if __name__ == '__main__':
    unittest.main()
