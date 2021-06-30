import numpy as np
import re
from scipy.sparse import csr_matrix
import unittest
import subprocess
import os.path


def remove_all_non_num(line):
    return re.sub('[^0-9.]', ' ', line)


def read_matrix(name):
    read_data = False

    row_ptr = np.zeros((1,), dtype=int)
    row_indices = np.zeros((1,), dtype=int)
    col_indices = np.zeros((1,), dtype=int)
    values = np.zeros((1,), dtype=float)

    tmp_col_indices = np.zeros((1,), dtype=int)
    tmp_values = np.zeros((1,), dtype=float)

    nrows_read = False
    nnz_read = False
    max_nnz_read = False
    tmp_resize = False

    with open(name, 'r') as bfile:
        lines = bfile.read().splitlines()
        for l in lines:
            if read_data:
                data = np.fromstring(remove_all_non_num(l),
                                     dtype=float, sep=' ')

                current_row = int(data[1])
                nnz_current_row = int(data[2])

                row_ptr[current_row+1] = nnz_current_row

                for i in range(0, nnz_current_row):
                    tmp_col_indices[current_row, i] = int(data[3 + 2*i])
                    tmp_values[current_row, i] = data[4 + 2*i]

            if 'Entries(Index,Value)' in l:
                read_data = True

            if 'Global dimensions' in l and not nrows_read:
                data = np.fromstring(remove_all_non_num(l), dtype=int, sep=' ')
                nrows = data[0]
                ncols = data[1]
                row_ptr = np.resize(row_ptr, (nrows+1,))
                row_ptr[0] = 0
                nrows_read = True

            if 'Global number of entries' in l and not nnz_read:
                data = np.fromstring(remove_all_non_num(l), dtype=int, sep=' ')
                nnz = data[0]
                row_indices = np.resize(col_indices, (nnz,))
                col_indices = np.resize(col_indices, (nnz,))
                values = np.resize(values, (nnz,))
                nnz_read = True

            if 'Max number of entries per row' in l and not max_nnz_read:
                data = np.fromstring(remove_all_non_num(l), dtype=int, sep=' ')
                max_nnz = data[0]
                max_nnz_read = True

            if max_nnz_read and nrows_read and not tmp_resize:
                tmp_col_indices = np.resize(tmp_col_indices, (nrows, max_nnz))
                tmp_values = np.resize(tmp_values, (nrows, max_nnz))
                tmp_resize = True

    for current_row in range(0, nrows):
        nnz_current_row = row_ptr[current_row+1]
        row_ptr[current_row+1] = row_ptr[current_row] + nnz_current_row

        for i in range(0, nnz_current_row):
            row_indices[row_ptr[current_row] +
                        i] = current_row
            col_indices[row_ptr[current_row] +
                        i] = tmp_col_indices[current_row, i]
            values[row_ptr[current_row] + i] = tmp_values[current_row, i]

    return row_indices, col_indices, values, nrows, ncols


def read_block(i_block, j_block, base_dir, extension=".txt"):
    return read_matrix(base_dir+str(i_block)+"_"+str(j_block)+extension)


def fuse_blocks(n_blocks, m_blocks, base_dir, extension=".txt"):
    row_indices = np.array((), dtype=int)
    col_indices = np.array((), dtype=int)
    values = np.array((), dtype=float)

    nrows_per_block = np.zeros((n_blocks+1,), dtype=int)
    ncols_per_block = np.zeros((m_blocks+1,), dtype=int)

    for i_block in range(0, n_blocks):
        for j_block in range(0, m_blocks):
            tmp_row_indices, tmp_col_indices, tmp_values, nrows, ncols = read_block(
                i_block, j_block, base_dir)
            nrows_per_block[i_block+1] = nrows_per_block[i_block] + nrows
            ncols_per_block[j_block+1] = ncols_per_block[j_block] + ncols
            row_indices = np.append(
                row_indices, nrows_per_block[i_block] + tmp_row_indices)
            col_indices = np.append(
                col_indices, ncols_per_block[j_block] + tmp_col_indices)
            values = np.append(values, tmp_values)
    total_nrows = nrows_per_block[-1]
    total_ncols = ncols_per_block[-1]
    return row_indices, col_indices, values, total_nrows, total_ncols, nrows_per_block, ncols_per_block


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
