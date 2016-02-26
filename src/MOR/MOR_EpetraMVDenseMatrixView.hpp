//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef MOR_EPETRAMVDENSEMATRIXVIEW_HPP
#define MOR_EPETRAMVDENSEMATRIXVIEW_HPP

class Epetra_MultiVector;
class Epetra_SerialDenseMatrix;
class Epetra_SerialSymDenseMatrix;

namespace MOR {

Epetra_SerialDenseMatrix localDenseMatrixView(Epetra_MultiVector &mv);
Epetra_SerialSymDenseMatrix localSymDenseMatrixView(Epetra_MultiVector &mv);

} // namespace MOR

#endif /* MOR_EPETRAMVDENSEMATRIXVIEW_HPP */
