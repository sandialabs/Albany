//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef MOR_EPETRASAMPLINGOPERATOR_HPP
#define MOR_EPETRASAMPLINGOPERATOR_HPP

#include "Epetra_Operator.h"
#include "Epetra_Map.h"

#include "Teuchos_Array.hpp"
#include "Teuchos_ArrayView.hpp"

namespace MOR {

class EpetraSamplingOperator : public Epetra_Operator {
public:
#ifndef EPETRA_NO_32BIT_GLOBAL_INDICES
  typedef int GlobalIndex;
#else
  typedef long long GlobalIndex;
#endif

  enum FromGIDsTag {
    fromGIDs
  };

  EpetraSamplingOperator(const Epetra_Map &map, const Teuchos::ArrayView<const GlobalIndex> &sampleLIDs);
  EpetraSamplingOperator(const Epetra_Map &map, FromGIDsTag, const Teuchos::ArrayView<const GlobalIndex> &sampleGIDs);

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

} // namespace MOR

#endif /* MOR_EPETRASAMPLINGOPERATOR_HPP */
