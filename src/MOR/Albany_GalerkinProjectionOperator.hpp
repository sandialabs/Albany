//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef ALBANY_GALERKINPROJECTIONOPERATOR_HPP
#define ALBANY_GALERKINPROJECTIONOPERATOR_HPP

#include "Epetra_Operator.h"

#include "Teuchos_RCP.hpp"

namespace Albany {

class ReducedSpace; 

class GalerkinProjectionOperator : public Epetra_Operator {
public:
  GalerkinProjectionOperator(const Teuchos::RCP<Epetra_Operator> &fullOperator,
                             const Teuchos::RCP<const ReducedSpace> &reducedSpace);
    
  // Overriden
  virtual const char *Label() const;
  
  virtual const Epetra_Comm & Comm() const;
  virtual const Epetra_Map & OperatorDomainMap() const;
  virtual const Epetra_Map & OperatorRangeMap() const;
  
  virtual bool HasNormInf() const;
  virtual double NormInf() const;
  
  virtual bool UseTranspose() const;
  virtual int SetUseTranspose(bool UseTranspose);
  
  virtual int Apply(const Epetra_MultiVector &X, Epetra_MultiVector &Y) const;
  virtual int ApplyInverse(const Epetra_MultiVector &X, Epetra_MultiVector &Y) const;

  // Direct access to internals
  Teuchos::RCP<Epetra_Operator> fullOperator() { return fullOperator_; }
  Teuchos::RCP<const Epetra_Operator> fullOperator() const { return fullOperator_; }

private:
  Teuchos::RCP<Epetra_Operator> fullOperator_;
  Teuchos::RCP<const ReducedSpace> reducedSpace_;
};

} // end namespace Albany

#endif /* ALBANY_GALERKINPROJECTIONOPERATOR_HPP */
