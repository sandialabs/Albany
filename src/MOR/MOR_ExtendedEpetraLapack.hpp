//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef MOR_EXTENDEDEPETRALAPACK_HPP
#define MOR_EXTENDEDEPETRALAPACK_HPP

#include "Epetra_LAPACK.h"

namespace MOR {

class Extended_Epetra_LAPACK : public Epetra_LAPACK {
public:
  double LANSY(const char NORM, const char UPLO, const int N, const double *A, const int LDA, double *WORK) const;
};

} // end namespace MOR

#endif /* MOR_EXTENDEDEPETRALAPACK_HPP */
