//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_StkEpetraMVSource.hpp"

namespace Albany {

StkEpetraMVSource::StkEpetraMVSource(const Teuchos::RCP<STKDiscretization> &disc) :
  disc_(disc)
{}

int
StkEpetraMVSource::vectorCount() const
{
  return disc_->getSolutionFieldHistoryDepth();
}

Epetra_Map
StkEpetraMVSource::vectorMap() const
{
  return *disc_->getMap();
}

Teuchos::RCP<Epetra_MultiVector>
StkEpetraMVSource::multiVectorNew()
{
  return disc_->getSolutionFieldHistory();
}

Teuchos::RCP<Epetra_MultiVector>
StkEpetraMVSource::truncatedMultiVectorNew(int vectorCountMax)
{
  return disc_->getSolutionFieldHistory(vectorCountMax);
}

const Epetra_MultiVector &
StkEpetraMVSource::filledMultiVector(Epetra_MultiVector &result)
{
  disc_->getSolutionFieldHistory(result);
  return result;
}

} // end namespace Albany
