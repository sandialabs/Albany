//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef MOR_GREEDYATOMICBASISSAMPLE_HPP
#define MOR_GREEDYATOMICBASISSAMPLE_HPP

#include "MOR_AtomicBasisSource.hpp"
#include "MOR_CollocationMetricCriterion.hpp"

#include "Epetra_SerialSymDenseMatrix.h"
#include "Epetra_Map.h"

#include "Teuchos_RCP.hpp"
#include "Teuchos_Array.hpp"
#include "Teuchos_ArrayView.hpp"

namespace MOR {

class GreedyAtomicBasisSample {
public:
  GreedyAtomicBasisSample(
      AtomicBasisSource &basisFile,
      const Teuchos::RCP<const CollocationMetricCriterion> &criterion);

  int sampleSize() const { return sample_.size(); }
  Teuchos::ArrayView<const int> sample() const { return sample_; }
  const Epetra_SerialSymDenseMatrix &discrepancy() const { return discrepancy_; }
  double sampleFitness() const { return criterion_->fitness(discrepancy_); }

  void sampleSizeInc(int incr);

private:
  Teuchos::RCP<const CollocationMetricCriterion> criterion_;

  Epetra_Map atomMap_;

  Teuchos::Array<Epetra_SerialSymDenseMatrix> contributions_;
  Epetra_SerialSymDenseMatrix discrepancy_;

  Teuchos::Array<int> sample_;
};

} // end namespace MOR

#endif /* MOR_GREEDYATOMICBASISSAMPLE_HPP */
