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

#include "Albany_PetrovGalerkinOperatorFactory.hpp"

#include "Albany_BasisOps.hpp"

#include "Epetra_Operator.h"

namespace Albany {

using Teuchos::RCP;

PetrovGalerkinOperatorFactory::PetrovGalerkinOperatorFactory(const RCP<const Epetra_MultiVector> &reducedBasis) :
  reducedBasis_(reducedBasis),
  projectionBasis_(reducedBasis),
  jacobianFactory_(reducedBasis_)
{
  // Nothing to do
}

PetrovGalerkinOperatorFactory::PetrovGalerkinOperatorFactory(const RCP<const Epetra_MultiVector> &reducedBasis,
                                                             const RCP<const Epetra_MultiVector> &projectionBasis) :
  reducedBasis_(reducedBasis),
  projectionBasis_(projectionBasis),
  jacobianFactory_(reducedBasis_)
{
  // Nothing to do
}

bool PetrovGalerkinOperatorFactory::fullJacobianRequired(bool /*residualRequested*/, bool jacobianRequested) const {
  return jacobianRequested;
}

const Epetra_MultiVector & PetrovGalerkinOperatorFactory::leftProjection(const Epetra_MultiVector &fullVec, Epetra_MultiVector &result) const {
  const int err = reduce(*projectionBasis_, fullVec, result);
  TEUCHOS_TEST_FOR_EXCEPT(err != 0);
  return result;
}

RCP<Epetra_CrsMatrix> PetrovGalerkinOperatorFactory::reducedJacobianNew() {
  return jacobianFactory_.reducedMatrixNew();
}

const Epetra_CrsMatrix & PetrovGalerkinOperatorFactory::reducedJacobian(Epetra_CrsMatrix &result) const {
  return jacobianFactory_.reducedMatrix(*projectionBasis_, result);
}

void PetrovGalerkinOperatorFactory::fullJacobianIs(const Epetra_Operator &op) {
  jacobianFactory_.fullJacobianIs(op);
}

} // namespace Albany
