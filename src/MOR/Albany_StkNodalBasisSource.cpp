//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_StkNodalBasisSource.hpp"

#include "Albany_AbstractMeshStruct.hpp"

#include "Teuchos_Assert.hpp"

namespace Albany {

StkNodalBasisSource::StkNodalBasisSource(const Teuchos::RCP<STKDiscretization> &disc) :
  disc_(disc),
  currentVectorRank_(-1),
  currentVector_(new Epetra_Vector(*disc->getMap(), /*zeroOut =*/ false))
{
  // Nothing to do
}

Epetra_Map
StkNodalBasisSource::atomMap() const
{
  return Epetra_Map(*disc_->getNodeMap());
}

int
StkNodalBasisSource::entryCount(int /*localAtomRank*/) const
{
  return disc_->getNumEq();
}

int
StkNodalBasisSource::entryCountMax() const
{
  return disc_->getNumEq();
}

int
StkNodalBasisSource::vectorCount() const
{
  return disc_->getSTKMeshStruct()->getSolutionFieldHistoryDepth();
}

int
StkNodalBasisSource::currentVectorRank() const
{
  return currentVectorRank_;
}

void
StkNodalBasisSource::currentVectorRankIs(int vr)
{
  if (vr != currentVectorRank_) {
    disc_->getSTKMeshStruct()->loadSolutionFieldHistory(vr);
    currentVector_ = disc_->getSolutionField();
    currentVectorRank_ = vr;
  }
}

Teuchos::ArrayView<const double>
StkNodalBasisSource::atomData(int localAtomRank, const Teuchos::ArrayView<double> &result) const
{
  TEUCHOS_ASSERT(result.size() <= this->entryCount(localAtomRank));
  const int dofCount = this->entryCount(localAtomRank);
  for (int dofRank = 0; dofRank < dofCount; ++dofRank) {
    const int localEntryIndex = disc_->getOwnedDOF(localAtomRank, dofRank);
    result[dofRank] = (*currentVector_)[localEntryIndex];
  }
  return result;
}

} // end namespace Albany
