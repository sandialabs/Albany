//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "MOR_ConcatenatedEpetraMVSource.hpp"

#include "Teuchos_TestForException.hpp"

#include <algorithm>

namespace MOR {

ConcatenatedEpetraMVSource::ConcatenatedEpetraMVSource(
    const Epetra_Map &vectorMap,
    const Teuchos::ArrayView<const Teuchos::RCP<EpetraMVSource> > &sources) :
  vectorMap_(vectorMap),
  sources_(sources),
  vectorCount_(0)
{
  for (SourceList::const_iterator it = sources_.begin(), it_end = sources_.end(); it != it_end; ++it) {
    const EpetraMVSource &source = **it;
    TEUCHOS_TEST_FOR_EXCEPT(!vectorMap_.SameAs(source.vectorMap()));
    vectorCount_ += source.vectorCount();
  }
}

int
ConcatenatedEpetraMVSource::vectorCount() const
{
  return vectorCount_;
}

Epetra_Map
ConcatenatedEpetraMVSource::vectorMap() const
{
  return vectorMap_;
}

Teuchos::RCP<Epetra_MultiVector>
ConcatenatedEpetraMVSource::multiVectorNew()
{
  const Teuchos::RCP<Epetra_MultiVector> result(
      new Epetra_MultiVector(vectorMap_, vectorCount_, /*zeroOut =*/ false));

  int firstVectorRank = 0;
  for (SourceList::iterator it = sources_.begin(), it_end = sources_.end(); it != it_end; ++it) {
    EpetraMVSource &source = **it;
    const int vectorSubcount = source.vectorCount();
    Epetra_MultiVector buffer(View, *result, firstVectorRank, vectorSubcount);
    source.filledMultiVector(buffer);
    firstVectorRank += vectorSubcount;
  }

  return result;
}

} // end namespace MOR
