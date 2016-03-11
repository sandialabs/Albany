//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef ALBANY_STKEPETRAMVSOURCE_HPP
#define ALBANY_STKEPETRAMVSOURCE_HPP

#include "MOR_EpetraMVSource.hpp"

#include "Albany_STKDiscretization.hpp"

namespace Albany {

class StkEpetraMVSource : public MOR::EpetraMVSource {
public:
  explicit StkEpetraMVSource(const Teuchos::RCP<STKDiscretization> &disc);

  virtual int vectorCount() const;
  virtual Epetra_Map vectorMap() const;

  virtual Teuchos::RCP<Epetra_MultiVector> multiVectorNew();
  virtual Teuchos::RCP<Epetra_MultiVector> truncatedMultiVectorNew(int vectorCountMax);
  virtual const Epetra_MultiVector &filledMultiVector(Epetra_MultiVector &result);

private:
  Teuchos::RCP<STKDiscretization> disc_;
};

} // end namespace Albany

#endif /*ALBANY_STKEPETRAMVSOURCE_HPP*/
