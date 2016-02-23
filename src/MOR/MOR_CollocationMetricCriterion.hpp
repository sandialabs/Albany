//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef MOR_COLLOCATIONMETRICCRITERION_HPP
#define MOR_COLLOCATIONMETRICCRITERION_HPP

#include "Epetra_SerialSymDenseMatrix.h"

namespace MOR {

class CollocationMetricCriterion {
public:
  CollocationMetricCriterion() {}
  virtual ~CollocationMetricCriterion() {}

  virtual double fitness(const Epetra_SerialSymDenseMatrix &discrepancy) const = 0;
  virtual double partialFitness(const Epetra_SerialSymDenseMatrix &discrepancy, int contributionCount) const = 0;

private:
  // Disallow copy and assignment
  CollocationMetricCriterion(const CollocationMetricCriterion &);
  CollocationMetricCriterion &operator=(const CollocationMetricCriterion &);
};


class FrobeniusNormCriterion : public CollocationMetricCriterion {
public:
  virtual double fitness(const Epetra_SerialSymDenseMatrix &discrepancy) const;
  virtual double partialFitness(const Epetra_SerialSymDenseMatrix &discrepancy, int contributionCount) const;

private:
  static double normalizedFrobeniusNorm(const Epetra_SerialSymDenseMatrix &discrepancy);
};


class TwoNormCriterion : public CollocationMetricCriterion {
public:
  explicit TwoNormCriterion(int eigenRankStride);

  virtual double fitness(const Epetra_SerialSymDenseMatrix &discrepancy) const;
  virtual double partialFitness(const Epetra_SerialSymDenseMatrix &discrepancy, int contributionCount) const;

private:
  int eigenRankStride_;
};

} // end namespace MOR

#endif /* MOR_COLLOCATIONMETRICCRITERION_HPP */
