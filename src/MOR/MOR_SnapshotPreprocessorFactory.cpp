//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "MOR_SnapshotPreprocessorFactory.hpp"

#include "MOR_TrivialSnapshotPreprocessor.hpp"
#include "MOR_FirstVectorSubstractingSnapshotPreprocessor.hpp"
#include "MOR_MeanSubstractingSnapshotPreprocessor.hpp"
#include "MOR_SubstractingSnapshotPreprocessor.hpp"

#include "Teuchos_TestForException.hpp"

#include <string>
#include <stdexcept>

namespace MOR {

Teuchos::RCP<SnapshotPreprocessor>
SnapshotPreprocessorFactory::instanceNew(const Teuchos::RCP<Teuchos::ParameterList> &params)
{
  const std::string typeToken = params->get("Type", "None");

  if (typeToken == "None") {
    return Teuchos::rcp(new TrivialSnapshotPreprocessor);
  } else if (typeToken == "Substract First Vector") {
    return Teuchos::rcp(new FirstVectorSubstractingSnapshotPreprocessor);
  } else if (typeToken == "Substract Arithmetic Mean") {
    return Teuchos::rcp(new MeanSubstractingSnapshotPreprocessor);
  } else if (typeToken == "Substract Provided Origin") {
    return Teuchos::rcp(new SubstractingSnapshotPreprocessor(userProvidedOrigin_));
  }

  TEUCHOS_TEST_FOR_EXCEPTION(
      true,
      std::invalid_argument,
      typeToken + " is not a valid snapshot preprocessor type.");
  return Teuchos::null;
}

Teuchos::RCP<const Epetra_Vector>
SnapshotPreprocessorFactory::userProvidedOrigin() const
{
  return userProvidedOrigin_;
}

void
SnapshotPreprocessorFactory::userProvidedOriginIs(
    const Teuchos::RCP<const Epetra_Vector> &origin)
{
  userProvidedOrigin_ = origin;
}

} // namespace MOR
