//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "MOR_PetrovGalerkinOperatorFactory.hpp"

#include "MOR_BasisOps.hpp"

#include "Epetra_Operator.h"

#include "Epetra_CrsMatrix.h"

namespace MOR {

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

bool PetrovGalerkinOperatorFactory::fullJacobianRequired(bool residualRequested, bool jacobianRequested) const {
  return residualRequested || jacobianRequested;
  //return jacobianRequested;
}

const Epetra_MultiVector & PetrovGalerkinOperatorFactory::rightProjection(const Epetra_MultiVector &fullVec, Epetra_MultiVector &result) const {
  const int err = reduce(*projectionBasis_, fullVec, result);
  TEUCHOS_TEST_FOR_EXCEPT(err != 0);
  return result;
}

const Epetra_MultiVector & PetrovGalerkinOperatorFactory::leftProjection(const Epetra_MultiVector &fullVec, Epetra_MultiVector &result) const {
  const int err = reduce(*projectionBasis_, fullVec, result);
  TEUCHOS_TEST_FOR_EXCEPT(err != 0);
  return result;
}

RCP<const Epetra_MultiVector> PetrovGalerkinOperatorFactory::getRightBasis() const {
  return jacobianFactory_.rightProjector();
}

RCP<const Epetra_MultiVector> PetrovGalerkinOperatorFactory::getPremultipliedReducedBasis() const {
	return jacobianFactory_.premultipliedRightProjector();
}

RCP<const Epetra_MultiVector> PetrovGalerkinOperatorFactory::getLeftBasisCopy() const {
	return projectionBasis_;
}

RCP<Epetra_CrsMatrix> PetrovGalerkinOperatorFactory::reducedJacobianNew() {
  return jacobianFactory_.reducedMatrixNew();
}

const Epetra_CrsMatrix & PetrovGalerkinOperatorFactory::reducedJacobianL(Epetra_CrsMatrix &result) const {
  return jacobianFactory_.reducedMatrix(*projectionBasis_, result);
}

const Epetra_CrsMatrix & PetrovGalerkinOperatorFactory::reducedJacobianR(Epetra_CrsMatrix &result) const {
  return jacobianFactory_.reducedMatrix(*projectionBasis_, result);
}

void PetrovGalerkinOperatorFactory::fullJacobianIs(const Epetra_Operator &op) {
  jacobianFactory_.fullJacobianIs(op);
}

} // namespace MOR
