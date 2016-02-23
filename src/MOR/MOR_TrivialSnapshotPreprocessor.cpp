//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "MOR_TrivialSnapshotPreprocessor.hpp"

namespace MOR {

TrivialSnapshotPreprocessor::TrivialSnapshotPreprocessor() :
  rawSnaphots_()
{}

Teuchos::RCP<const Epetra_MultiVector>
TrivialSnapshotPreprocessor::modifiedSnapshotSet() const
{
  return rawSnaphots_;
}

Teuchos::RCP<const Epetra_Vector>
TrivialSnapshotPreprocessor::origin() const
{
  return Teuchos::null;
}

void
TrivialSnapshotPreprocessor::rawSnapshotSetIs(const Teuchos::RCP<Epetra_MultiVector> &rs)
{
  rawSnaphots_ = rs;
}

} // namespace MOR
