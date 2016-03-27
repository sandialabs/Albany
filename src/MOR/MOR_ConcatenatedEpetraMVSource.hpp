//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef MOR_CONCATENATEDEPETRAMVSOURCE_HPP
#define MOR_CONCATENATEDEPETRAMVSOURCE_HPP

#include "MOR_EpetraMVSource.hpp"

#include "Teuchos_ArrayView.hpp"
#include "Teuchos_Array.hpp"

namespace MOR {

class ConcatenatedEpetraMVSource : public BasicEpetraMVSource {
public:
  ConcatenatedEpetraMVSource(
      const Epetra_Map &vectorMap,
      const Teuchos::ArrayView<const Teuchos::RCP<EpetraMVSource> > &sources);

  virtual int vectorCount() const;
  virtual Epetra_Map vectorMap() const;

  virtual Teuchos::RCP<Epetra_MultiVector> multiVectorNew();

private:
  const Epetra_Map vectorMap_;

  typedef Teuchos::Array<Teuchos::RCP<EpetraMVSource> > SourceList;
  SourceList sources_;

  int vectorCount_;
};

} // end namespace MOR

#endif /*MOR_CONCATENATEDEPETRAMVSOURCE_HPP*/
