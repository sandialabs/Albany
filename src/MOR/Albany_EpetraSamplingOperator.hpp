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

#ifndef ALBANY_EPETRASAMPLINGOPERATOR_HPP
#define ALBANY_EPETRASAMPLINGOPERATOR_HPP

#include "Epetra_Operator.h"
#include "Epetra_Map.h"

#include "Teuchos_Array.hpp"
#include "Teuchos_ArrayView.hpp"

namespace Albany {

class EpetraSamplingOperator : public Epetra_Operator {
public:
  typedef int GlobalIndex;

  EpetraSamplingOperator(const Epetra_Map &map, const Teuchos::ArrayView<const GlobalIndex> &sampleGIDs);

  // Overriden from Epetra_Operator
  virtual const char *Label() const;

  virtual const Epetra_Map &OperatorDomainMap() const;
  virtual const Epetra_Map &OperatorRangeMap() const;
  virtual const Epetra_Comm &Comm() const;

  virtual bool UseTranspose() const;
  virtual int SetUseTranspose(bool UseTranspose);

  virtual int Apply(const Epetra_MultiVector &X, Epetra_MultiVector &Y) const;
  virtual int ApplyInverse(const Epetra_MultiVector &X, Epetra_MultiVector &Y) const;

  virtual bool HasNormInf() const;
  virtual double NormInf() const;

private:
  Epetra_Map map_;
  Teuchos::Array<int> sampleLIDs_;

  bool useTranspose_;
};

} // namespace Albany

#endif /* ALBANY_EPETRASAMPLINGOPERATOR_HPP */
