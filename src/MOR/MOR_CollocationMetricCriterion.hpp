//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
class Epetra_SerialSymDenseMatrix;

namespace MOR {

double frobeniusNorm(const Epetra_SerialSymDenseMatrix &);
double smallestEigenvalue(const Epetra_SerialSymDenseMatrix &);

} // end namespace MOR
