/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/

#include "Albany_GalerkinProjectionOperator.hpp"

#include "Albany_ReducedSpace.hpp"

#include "Epetra_MultiVector.h"

#include "Teuchos_Assert.hpp"

namespace Albany {

using Teuchos::RCP;

GalerkinProjectionOperator::GalerkinProjectionOperator(const RCP<Epetra_Operator> &fullOperator,
                                                       const RCP<const ReducedSpace> &reducedSpace) :
  fullOperator_(fullOperator),
  reducedSpace_(reducedSpace)
{
  TEUCHOS_ASSERT(fullOperator_->OperatorDomainMap().SameAs(reducedSpace_->basisMap()));
  TEUCHOS_ASSERT(fullOperator_->OperatorRangeMap().SameAs(reducedSpace_->basisMap()));
}

const char *GalerkinProjectionOperator::Label() const
{
  return fullOperator_->Label();
}

const Epetra_Comm &GalerkinProjectionOperator::Comm() const
{
  return reducedSpace_->comm();
}

const Epetra_Map &GalerkinProjectionOperator::OperatorDomainMap() const
{
  return reducedSpace_->componentMap();
}

const Epetra_Map &GalerkinProjectionOperator::OperatorRangeMap() const
{
  return reducedSpace_->componentMap();
}

bool GalerkinProjectionOperator::HasNormInf() const
{
  return false;
}

double GalerkinProjectionOperator::NormInf() const
{
  return 0.0;
}

bool GalerkinProjectionOperator::UseTranspose() const
{
  return fullOperator_->UseTranspose();
}

int GalerkinProjectionOperator::SetUseTranspose(bool UseTranspose)
{
  return fullOperator_->SetUseTranspose(UseTranspose);
}

int GalerkinProjectionOperator::Apply(const Epetra_MultiVector &X, Epetra_MultiVector &Y) const
{
  const int vectorCount = X.NumVectors();

  // full_X <- basis * X
  Epetra_MultiVector full_X(fullOperator_->OperatorDomainMap(), vectorCount, false);
  reducedSpace_->linearExpansion(X, full_X);

  // full_Y <- full_operator * full_X
  Epetra_MultiVector full_Y(fullOperator_->OperatorRangeMap(), vectorCount, false);
  const int err = fullOperator_->Apply(full_X, full_Y);
  if (err != 0) {
    return err;
  }

  // Phi^T * A * (Phi * v)
  reducedSpace_->linearReduction(full_Y, Y);
  return 0;
}

int GalerkinProjectionOperator::ApplyInverse(const Epetra_MultiVector &/*X*/, Epetra_MultiVector &/*Y*/) const
{
  return -1;
}

} // end namespace Albany
