//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "MOR_ExtendedEpetraLapack.hpp"

#include "Epetra_SerialSymDenseMatrix.h"

#include <vector>
#include <cstddef>
#include <cassert>

namespace MOR {

double frobeniusNorm(const Epetra_SerialSymDenseMatrix &matrix) {
  const char norm = 'F';
  const char uplo = matrix.Upper() ? 'U' : 'L';

  return Extended_Epetra_LAPACK().LANSY(norm, uplo, matrix.N(), matrix.A(), matrix.LDA(), /*WORK = */ NULL);
}

double smallestEigenvalue(const Epetra_SerialSymDenseMatrix &m) {
  Epetra_SerialSymDenseMatrix matrix(m);
  const int matrixSize = matrix.N();

  if (matrixSize > 0) {
    const char noVectors = 'N';
    const char range = 'I';
    const char uplo = matrix.Upper() ? 'U' : 'L';
    const int eigenRank = 1;
    const double defaultTol = 0.0;

    int eigenValueCount;
    std::vector<double> eigenValues(matrixSize);

    int info;

    const int query = -1;
    double workQueriedSize;
    int iworkQueriedSize;

    Extended_Epetra_LAPACK().SYEVR(
        noVectors, range, uplo,
        matrixSize, matrix.A(), matrix.LDA(),
        /* vl = */ NULL, /* vu = */ NULL, &eigenRank, &eigenRank,
        defaultTol,
        &eigenValueCount, &eigenValues[0], /* z = */ NULL, /* ldz = */ 1 /* (dummy) */, /* isuppz = */ NULL,
        &workQueriedSize, /* lwork = */ query, &iworkQueriedSize, /* liwork = */ query,
        &info);
    assert(info == 0);

    std::vector<double> work(static_cast<int>(workQueriedSize));
    std::vector<int> iwork(iworkQueriedSize);

    // Workaround for bug in DSYEVR in early releases of the Lapack 3 reference implementation:
    // The input arguments VU and VL are accessed even when RANGE != 'V'
    const double v_workaround = 0.0;

    Epetra_LAPACK().SYEVR(
        noVectors, range, uplo,
        matrixSize, matrix.A(), matrix.LDA(),
        /* vl = */ &v_workaround, /* vu = */ &v_workaround, &eigenRank, &eigenRank,
        defaultTol,
        &eigenValueCount, &eigenValues[0], /* z = */ NULL, /* ldz = */ 1 /* (dummy) */, /* isuppz = */ NULL,
        &work[0], /* lwork = */ work.size(), &iwork[0], iwork.size(),
        &info);
    assert(info == 0);

    assert(eigenValueCount == 1);
    return eigenValues[0];
  } else {
    return 0.0; // Dummy value
  }
}

} // end namespace MOR
