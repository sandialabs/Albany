//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "MOR_GreedyAtomicBasisSample.hpp"

#include "MOR_CollocationMetricCriterion.hpp"
#include "MOR_MinMaxTools.hpp"

#include "Epetra_SerialDenseMatrix.h"
#include "Epetra_SerialSymDenseMatrix.h"
#include "Epetra_Comm.h"

#include "Thyra_EpetraThyraWrappers.hpp"

#include "Teuchos_RCP.hpp"
#include "Teuchos_Array.hpp"
#include "Teuchos_Comm.hpp"

#include "Teuchos_Assert.hpp"

#include <utility>
#include <algorithm>

namespace MOR {

namespace Detail {

Teuchos::Array<Epetra_SerialDenseMatrix>
createAtomicSections(MOR::AtomicBasisSource &basisSource)
{
  const Epetra_Map atomMap = basisSource.atomMap();
  const int ownedAtomCount = atomMap.NumMyElements();
  const int vectorCount = basisSource.vectorCount();

  // Setup
  Teuchos::Array<Epetra_SerialDenseMatrix> result(ownedAtomCount);
  for (int iAtom = 0; iAtom < ownedAtomCount; ++iAtom) {
    TEUCHOS_ASSERT(atomMap.MyLID(iAtom));
    result[iAtom].Shape(basisSource.entryCount(iAtom), vectorCount);
  }

  // Fill
  for (int vectorRank = 0; vectorRank < vectorCount; ++vectorRank) {
    basisSource.currentVectorRankIs(vectorRank);
    for (int iAtom = 0; iAtom < ownedAtomCount; ++iAtom) {
      Teuchos::ArrayView<double> target(result[iAtom][vectorRank], basisSource.entryCount(iAtom));
      basisSource.atomData(iAtom, target);
    }
  }

  return result;
}

Teuchos::Array<Epetra_SerialSymDenseMatrix>
createAtomicContributions(const Teuchos::ArrayView<const Epetra_SerialDenseMatrix> &atomicSections) {
  const int sectionCount = atomicSections.size();
  Teuchos::Array<Epetra_SerialSymDenseMatrix> result(sectionCount);

  for (int i = 0; i < sectionCount; ++i) {
    const Epetra_SerialDenseMatrix &section = atomicSections[i];
    Epetra_SerialSymDenseMatrix &contribution = result[i];

    const int contributionSize = section.ColDim();
    if (contributionSize > 0) {
      contribution.Shape(contributionSize);
      contribution.Multiply('T', 'N', 1.0, section, section, 0.0);
    }
  }

  return result;
}

Teuchos::Array<Epetra_SerialSymDenseMatrix>
createAtomicContributions(MOR::AtomicBasisSource &basisSource)
{
  return createAtomicContributions(createAtomicSections(basisSource));
}

Teuchos::Array<double>
computePartialFitnesses(
    const Epetra_SerialSymDenseMatrix &reference,
    const Teuchos::ArrayView<const Epetra_SerialSymDenseMatrix> &atomicContributions,
    const CollocationMetricCriterion &criterion,
    int referenceContributionCount)
{
  const int localAtomCount = atomicContributions.size();
  Teuchos::Array<double> result(localAtomCount);

  if (!result.empty()) {
    Epetra_SerialSymDenseMatrix candidate;
    const int candidateContributionCount = referenceContributionCount + 1;

    for (int iAtom = 0; iAtom < localAtomCount; ++iAtom) {
      {
        candidate = reference;
        candidate += atomicContributions[iAtom];
      }
      result[iAtom] = criterion.partialFitness(candidate, candidateContributionCount);
    }
  }

  return result;
}

Epetra_SerialSymDenseMatrix negative_eye(int size) {
  Epetra_SerialSymDenseMatrix result;
  result.Shape(size);
  for (int i = 0; i < size; ++i) {
    result[i][i] = -1.0;
  }
  return result;
}

template <typename Ordinal>
void broadcast(const Teuchos::Comm<Ordinal> &comm, int rootRank, Epetra_SerialDenseMatrix &buffer) {
  const int count = buffer.LDA() * buffer.N();
  Teuchos::broadcast<Ordinal, double>(comm, rootRank, count, buffer.A());
}

int
bestCandidateId(
    const Epetra_Map &candidateMap,
    const Teuchos::ArrayView<const Epetra_SerialSymDenseMatrix> &candidates,
    Epetra_SerialSymDenseMatrix &reference,
    const CollocationMetricCriterion &criterion,
    int referenceContributionCount)
{
  const Teuchos::Array<double> fitnesses =
    computePartialFitnesses(reference, candidates, criterion, referenceContributionCount);

  const Teuchos::RCP<const Teuchos::Comm<Thyra::Ordinal> > comm =
    Thyra::create_Comm(Teuchos::rcpFromRef(candidateMap.Comm()));

  const Teuchos::ArrayView<int> ids(candidateMap.MyGlobalElements(), candidateMap.NumMyElements());
  return MOR::globalIdOfGlobalMinimum(*comm, ids, fitnesses);
}

void
updateReferenceAndCandidates(
    const Epetra_Map &candidateMap,
    Teuchos::ArrayView<Epetra_SerialSymDenseMatrix> candidates,
    int selectedId,
    Epetra_SerialSymDenseMatrix &reference)
{
  int selectedIdCpu;
  int selectedLocalId;
  candidateMap.RemoteIDList(1, &selectedId, &selectedIdCpu, &selectedLocalId);
  const bool selectedCandidateIsLocal = (selectedIdCpu == candidateMap.Comm().MyPID());

  if (selectedCandidateIsLocal) {
    Epetra_SerialSymDenseMatrix &selectedCandidate = candidates[selectedLocalId];
    reference += selectedCandidate;
    selectedCandidate.Scale(0.0); // Zero-out to prevent reselecting
  }
  const Teuchos::RCP<const Teuchos::Comm<Thyra::Ordinal> > comm =
    Thyra::create_Comm(Teuchos::rcpFromRef(candidateMap.Comm()));
  broadcast(*comm, selectedIdCpu, reference);
}

} // namespace Detail

GreedyAtomicBasisSample::GreedyAtomicBasisSample(
    AtomicBasisSource &basisSource,
    const Teuchos::RCP<const CollocationMetricCriterion> &criterion) :
  criterion_(criterion),
  atomMap_(basisSource.atomMap()),
  contributions_(Detail::createAtomicContributions(basisSource)),
  discrepancy_(Detail::negative_eye((contributions_.size() > 0) ? contributions_.front().RowDim() : 0)),
  sample_()
{
  // Nothing to do
}

void
GreedyAtomicBasisSample::sampleSizeInc(int incr) {
  for (int iter = 0; iter < incr; ++iter) {
    const int selectedId = Detail::bestCandidateId(atomMap_, contributions_, discrepancy_, *criterion_, iter);
    Detail::updateReferenceAndCandidates(atomMap_, contributions_, selectedId, discrepancy_);;
    sample_.push_back(selectedId);
  }
}

} // namespace MOR
