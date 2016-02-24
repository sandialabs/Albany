//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef MOR_EPETRAMVSOURCE_HPP
#define MOR_EPETRAMVSOURCE_HPP

#include "Epetra_Map.h"
#include "Epetra_MultiVector.h"

#include "Teuchos_RCP.hpp"

namespace MOR {

class BasicEpetraMVSource {
public:
  virtual int vectorCount() const = 0;
  virtual Epetra_Map vectorMap() const = 0;

  virtual Teuchos::RCP<Epetra_MultiVector> multiVectorNew() = 0;
  virtual Teuchos::RCP<Epetra_MultiVector> truncatedMultiVectorNew(int vectorCountMax);

  virtual ~BasicEpetraMVSource() {}
};

class EpetraMVSource : public BasicEpetraMVSource {
public:
  virtual const Epetra_MultiVector &filledMultiVector(Epetra_MultiVector &result) = 0;
};

} // end namespace MOR

#endif /*MOR_EPETRAMVSOURCE_HPP*/
