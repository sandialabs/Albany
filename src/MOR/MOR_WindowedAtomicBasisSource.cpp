//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "MOR_WindowedAtomicBasisSource.hpp"

#include <algorithm>

namespace MOR {

WindowedAtomicBasisSource::WindowedAtomicBasisSource(
    const Teuchos::RCP<AtomicBasisSource> &delegate,
    int firstVectorRank) :
  delegate_(delegate),
  firstVectorRank_(firstVectorRank),
  vectorCount_(delegate->vectorCount() - firstVectorRank)
{
  // Nothing to do
}

WindowedAtomicBasisSource::WindowedAtomicBasisSource(
    const Teuchos::RCP<AtomicBasisSource> &delegate,
    int firstVectorRank,
    int vectorCountMax) :
  delegate_(delegate),
  firstVectorRank_(firstVectorRank),
  vectorCount_(std::min(delegate->vectorCount() - firstVectorRank, vectorCountMax))
{
  // Nothing to do
}

Epetra_Map
WindowedAtomicBasisSource::atomMap() const
{
  return delegate_->atomMap();
}

int
WindowedAtomicBasisSource::entryCount(int localAtomRank) const
{
  return delegate_->entryCount(localAtomRank);
}

int
WindowedAtomicBasisSource::entryCountMax() const
{
  return delegate_->entryCountMax();
}

int
WindowedAtomicBasisSource::vectorCount() const
{
  return vectorCount_;
}

int
WindowedAtomicBasisSource::currentVectorRank() const
{
  return firstVectorRank_ + delegate_->currentVectorRank();
}

void
WindowedAtomicBasisSource::currentVectorRankIs(int vr)
{
  delegate_->currentVectorRankIs(firstVectorRank_ + vr);
}

Teuchos::ArrayView<const double>
WindowedAtomicBasisSource::atomData(int localAtomRank, const Teuchos::ArrayView<double> &result) const
{
  return delegate_->atomData(localAtomRank, result);
}

} // namespace MOR
