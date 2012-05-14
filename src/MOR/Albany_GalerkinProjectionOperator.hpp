/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/

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
