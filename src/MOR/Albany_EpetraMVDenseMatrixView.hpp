//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef ALBANY_EPETRAMVDENSEMATRIXVIEW_HPP
#define ALBANY_EPETRAMVDENSEMATRIXVIEW_HPP

class Epetra_MultiVector;
class Epetra_SerialDenseMatrix;
class Epetra_SerialSymDenseMatrix;

namespace Albany {

Epetra_SerialDenseMatrix localDenseMatrixView(Epetra_MultiVector &mv);
Epetra_SerialSymDenseMatrix localSymDenseMatrixView(Epetra_MultiVector &mv);

} // namespace Albany

#endif /* ALBANY_EPETRAMVDENSEMATRIXVIEW_HPP */
