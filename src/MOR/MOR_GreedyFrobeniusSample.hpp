//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef MOR_GREEDYFROBENIUSSAMPLE_HPP
#define MOR_GREEDYFROBENIUSSAMPLE_HPP

#include "MOR_AtomicBasisSource.hpp"

#include "Epetra_SerialSymDenseMatrix.h"
#include "Epetra_Map.h"

#include "Teuchos_RCP.hpp"
#include "Teuchos_Array.hpp"
#include "Teuchos_ArrayView.hpp"

namespace MOR {

class GreedyFrobeniusSample {
public:
  explicit GreedyFrobeniusSample(AtomicBasisSource &basisFile);
  GreedyFrobeniusSample(AtomicBasisSource &basisFile, int firstVectorRank);
  GreedyFrobeniusSample(AtomicBasisSource &basisFile, int firstVectorRank, int vectorCountMax);

  int sampleSize() const { return sample_.size(); }
  Teuchos::ArrayView<const int> sample() const { return sample_; }

  void sampleSizeInc(int incr);

  double fitness() const;
  const Epetra_SerialSymDenseMatrix &discrepancy() const { return discrepancy_; }

private:
  Epetra_Map atomMap_;

  Teuchos::Array<Epetra_SerialSymDenseMatrix> contributions_;
  Epetra_SerialSymDenseMatrix discrepancy_;

  Teuchos::Array<int> sample_;
};

} // namespace MOR

#endif /* MOR_GREEDYFROBENIUSSAMPLE_HPP */
