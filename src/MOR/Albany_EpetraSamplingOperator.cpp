//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Albany_EpetraSamplingOperator.hpp"

#include "Epetra_MultiVector.h"
#include "Epetra_Comm.h"

#include "Teuchos_Assert.hpp"
#include "Teuchos_TypeNameTraits.hpp"

#include <string>
#include <algorithm>
#include <iterator>
#include <functional>

namespace Albany {

using ::Teuchos::Array;
using ::Teuchos::ArrayView;

EpetraSamplingOperator::EpetraSamplingOperator(const Epetra_Map &map,
                                               const ArrayView<const GlobalIndex> &sampleGIDs) :
  map_(map)
{
  Array<GlobalIndex> sortedMyGIDs(map_.MyGlobalElements(), map_.MyGlobalElements() + map_.NumMyElements());
  std::sort(sortedMyGIDs.begin(), sortedMyGIDs.end());

  Array<GlobalIndex> sortedSampleGIDs(sampleGIDs);
  std::sort(sortedSampleGIDs.begin(), sortedSampleGIDs.end());

  Array<GlobalIndex> mySampleGIDs;
  std::set_intersection(sortedMyGIDs.begin(), sortedMyGIDs.end(),
                        sortedSampleGIDs.begin(), sortedSampleGIDs.end(),
                        std::back_inserter(mySampleGIDs));

  sampleLIDs_.reserve(mySampleGIDs.size());
  std::transform(mySampleGIDs.begin(), mySampleGIDs.end(),
                 std::back_inserter(sampleLIDs_),
                 std::bind1st(std::mem_fun_ref(static_cast<int(Epetra_Map::*)(GlobalIndex) const>(&Epetra_Map::LID)), map_));

  std::sort(sampleLIDs_.begin(), sampleLIDs_.end());
}

const char *EpetraSamplingOperator::Label() const
{
  static const std::string label = Teuchos::TypeNameTraits<EpetraSamplingOperator>::name();
  return label.c_str();
}

const Epetra_Map &EpetraSamplingOperator::OperatorDomainMap() const
{
  return map_;
}

const Epetra_Map &EpetraSamplingOperator::OperatorRangeMap() const
{
  return map_;
}

const Epetra_Comm &EpetraSamplingOperator::Comm() const
{
  return map_.Comm();
}

int EpetraSamplingOperator::SetUseTranspose(bool UseTranspose)
{
  useTranspose_ = UseTranspose;
  return 0;
}

bool EpetraSamplingOperator::UseTranspose() const
{
  return useTranspose_;
}

int EpetraSamplingOperator::Apply(const Epetra_MultiVector &X, Epetra_MultiVector &Y) const
{
  TEUCHOS_ASSERT(map_.PointSameAs(X.Map()) && map_.PointSameAs(Y.Map()));
  TEUCHOS_ASSERT(X.NumVectors() == Y.NumVectors());

  Y.PutScalar(0.0);

  for (int iVec = 0; iVec < X.NumVectors(); ++iVec) {
    const ArrayView<const double> sourceVec(X[iVec], X.MyLength());
    const ArrayView<double> targetVec(Y[iVec], Y.MyLength());
    for (Array<GlobalIndex>::const_iterator it = sampleLIDs_.begin(), it_end = sampleLIDs_.end(); it != it_end; ++it) {
       targetVec[*it] = sourceVec[*it];
    }
  }

  return 0;
}

int EpetraSamplingOperator::ApplyInverse(const Epetra_MultiVector &X, Epetra_MultiVector &Y) const
{
  // Not supported (rank-deficient operator)
  return -1;
}

bool EpetraSamplingOperator::HasNormInf() const
{
  return true;
}

double EpetraSamplingOperator::NormInf() const
{
  // Using long because of Epetra_Comm::SumAll (should be Array<GlobalIndex>::size_type)
  long mySampleCount = sampleLIDs_.size(); // Nonconst because of Epetra_Comm::SumAll
  long sampleCount;
  this->Comm().SumAll(&mySampleCount, &sampleCount, 1);
  return sampleCount != 0l ? 1.0 : 0.0;
}

} // namespace Albany
