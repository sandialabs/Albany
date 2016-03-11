//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "MOR_ExtendedEpetraLapack.hpp"

#include "Epetra_LAPACK_wrappers.h"

#define DLANSY_F77 F77_BLAS_MANGLE(dlansy, DLANSY)

#ifdef __cplusplus
extern "C" {
#endif
double PREFIX DLANSY_F77(Epetra_fcd norm, Epetra_fcd uplo, const int* n, const double* a, const int* lda, double * work);
#ifdef __cplusplus
}
#endif

#ifdef CHAR_MACRO
#undef CHAR_MACRO
#endif
#if defined (INTEL_CXML)
#define CHAR_MACRO(char_var) &char_var, 1
#else
#define CHAR_MACRO(char_var) &char_var
#endif

namespace MOR {

double
Extended_Epetra_LAPACK::LANSY(const char NORM, const char UPLO, const int N, const double *A, const int LDA, double *WORK) const {
  return DLANSY_F77(CHAR_MACRO(NORM), CHAR_MACRO(UPLO), &N, A, &LDA, WORK);
}

} // end namespace MOR
