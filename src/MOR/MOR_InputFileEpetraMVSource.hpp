//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef MOR_INPUTFILEEPETRAMVSOURCE_HPP
#define MOR_INPUTFILEEPETRAMVSOURCE_HPP

#include "MOR_EpetraMVSource.hpp"

#include "MOR_MultiVectorInputFile.hpp"

namespace MOR {

class InputFileEpetraMVSource : public BasicEpetraMVSource {
public:
  InputFileEpetraMVSource(
      const Epetra_Map &vectorMap,
      const Teuchos::RCP<MultiVectorInputFile> &inputFile);

  virtual int vectorCount() const;
  virtual Epetra_Map vectorMap() const;

  virtual Teuchos::RCP<Epetra_MultiVector> multiVectorNew();

private:
  const Epetra_Map vectorMap_;
  Teuchos::RCP<MultiVectorInputFile> inputFile_;

  int vectorCount_;
};

} // end namespace MOR

#endif /*MOR_INPUTFILEEPETRAMVSOURCE_HPP*/
