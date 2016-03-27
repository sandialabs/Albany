//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef MOR_FIRSTVECTORSUBSTRACTINGSNAPSHOTPREPROCESSOR_HPP
#define MOR_FIRSTVECTORSUBSTRACTINGSNAPSHOTPREPROCESSOR_HPP

#include "MOR_SnapshotPreprocessor.hpp"

namespace MOR {

class FirstVectorSubstractingSnapshotPreprocessor : public SnapshotPreprocessor {
public:
  FirstVectorSubstractingSnapshotPreprocessor();

  virtual Teuchos::RCP<const Epetra_MultiVector> modifiedSnapshotSet() const;
  virtual Teuchos::RCP<const Epetra_Vector> origin() const;

  virtual void rawSnapshotSetIs(const Teuchos::RCP<Epetra_MultiVector> &);

private:
  Teuchos::RCP<const Epetra_MultiVector> modifiedSnapshots_;
  Teuchos::RCP<const Epetra_Vector> origin_;
};

} // namespace MOR

#endif /* MOR_FIRSTVECTORSUBSTRACTINGSNAPSHOTPREPROCESSOR_HPP */
