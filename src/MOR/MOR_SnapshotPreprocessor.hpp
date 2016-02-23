//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef MOR_SNAPSHOTPREPROCESSOR_HPP
#define MOR_SNAPSHOTPREPROCESSOR_HPP

#include "Epetra_MultiVector.h"
#include "Epetra_Vector.h"

#include "Teuchos_RCP.hpp"

namespace MOR {

class SnapshotPreprocessor {
public:
  virtual Teuchos::RCP<const Epetra_MultiVector> modifiedSnapshotSet() const = 0;
  virtual Teuchos::RCP<const Epetra_Vector> origin() const = 0;

  virtual void rawSnapshotSetIs(const Teuchos::RCP<Epetra_MultiVector> &) = 0;

  virtual ~SnapshotPreprocessor() {}

protected:
  SnapshotPreprocessor() {}

private:
  // Disallow copy and assigment
  SnapshotPreprocessor(const SnapshotPreprocessor &);
  SnapshotPreprocessor& operator=(const SnapshotPreprocessor &);
};

} // namespace MOR

#endif /* MOR_SNAPSHOTPREPROCESSOR_HPP */
