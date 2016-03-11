//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "MOR_InputFileEpetraMVSource.hpp"

namespace MOR {

InputFileEpetraMVSource::InputFileEpetraMVSource(
    const Epetra_Map &vectorMap,
    const Teuchos::RCP<MultiVectorInputFile> &inputFile) :
  vectorMap_(vectorMap),
  inputFile_(inputFile),
  vectorCount_(inputFile->readVectorCount(vectorMap.Comm()))
{
}

int
InputFileEpetraMVSource::vectorCount() const
{
  return vectorCount_;
}

Epetra_Map
InputFileEpetraMVSource::vectorMap() const
{
  return vectorMap_;
}

Teuchos::RCP<Epetra_MultiVector>
InputFileEpetraMVSource::multiVectorNew()
{
  return inputFile_->read(vectorMap_);
}

} // end namespace MOR

