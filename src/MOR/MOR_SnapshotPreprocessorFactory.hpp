//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef MOR_SNAPSHOTPREPROCESSORFACTORY_HPP
#define MOR_SNAPSHOTPREPROCESSORFACTORY_HPP

#include "MOR_SnapshotPreprocessor.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"

#include "Epetra_Vector.h"

namespace MOR {

class SnapshotPreprocessorFactory {
public:
  Teuchos::RCP<SnapshotPreprocessor> instanceNew(const Teuchos::RCP<Teuchos::ParameterList> &params);

  Teuchos::RCP<const Epetra_Vector> userProvidedBaseVector() const;
  void userProvidedBaseVectorIs(const Teuchos::RCP<const Epetra_Vector> &baseVector);

private:
  Teuchos::RCP<const Epetra_Vector> userProvidedBaseVector_;
};

} // namespace MOR

#endif /* MOR_SNAPSHOTPREPROCESSORFACTORY_HPP */
