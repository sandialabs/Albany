import numpy as np
import re
from scipy.sparse import csr_matrix


def remove_all_non_num(line, is_int=False):
    if is_int:
        return re.sub('[^0-9.+-]', ' ', line)
    else:
        return re.sub('[^eE0-9.+-]', ' ', line)


def read_mm(filename, is_sparse=True, add=0.0, use_map=False, map_inverse=None):
    X_dim = np.genfromtxt(filename, skip_header=1, max_rows=1)
    m = X_dim[0].astype(int)
    n = X_dim[1].astype(int)

    if is_sparse:
        from scipy.sparse import csr_matrix
        X = np.loadtxt(filename, skiprows=2)

        row = X[:, 0].astype(int)-1
        col = X[:, 1].astype(int)-1
        data = X[:, 2]+add

        if use_map:
            row = map_inverse[X[:, 0].astype(int)-1]
            col = map_inverse[X[:, 1].astype(int)-1]

        mask = data != 0.
        m = np.amax(row)+1
        n = np.amax(col)+1
        A = csr_matrix((data[mask], (row[mask], col[mask])), shape=(m, n))
        B = A.tocsc()
    else:
        X = np.loadtxt(filename, skiprows=2)
        if n != 1:
            B = X.reshape((m, n)).T
        else:
            B = X.reshape((m,))

    return B


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
                data = np.fromstring(remove_all_non_num(l, True), dtype=int, sep=' ')
                nrows = data[0]
                ncols = data[1]
                row_ptr = np.resize(row_ptr, (nrows+1,))
                row_ptr[0] = 0
                nrows_read = True

            if 'Global number of entries' in l and not nnz_read:
                data = np.fromstring(remove_all_non_num(l, True), dtype=int, sep=' ')
                nnz = data[0]
                row_indices = np.resize(col_indices, (nnz,))
                col_indices = np.resize(col_indices, (nnz,))
                values = np.resize(values, (nnz,))
                nnz_read = True

            if 'Max number of entries per row' in l and not max_nnz_read:
                data = np.fromstring(remove_all_non_num(l, True), dtype=int, sep=' ')
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

