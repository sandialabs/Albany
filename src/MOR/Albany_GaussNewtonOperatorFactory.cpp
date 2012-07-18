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

#include "Albany_GaussNewtonOperatorFactory.hpp"

#include "Albany_BasisOps.hpp"

#include "Epetra_Operator.h"

namespace Albany {

using Teuchos::RCP;

GaussNewtonOperatorFactory::GaussNewtonOperatorFactory(const RCP<const Epetra_MultiVector> &reducedBasis) :
  reducedBasis_(reducedBasis),
  jacobianFactory_(reducedBasis_)
{
  // Nothing to do
}

bool GaussNewtonOperatorFactory::fullJacobianRequired(bool residualRequested, bool jacobianRequested) const {
  return residualRequested || jacobianRequested;
}

const Epetra_MultiVector & GaussNewtonOperatorFactory::leftProjection(const Epetra_MultiVector &fullVec, Epetra_MultiVector &result) const {
  const int err = reduce(*jacobianFactory_.premultipliedRightProjector(), fullVec, result);
  TEUCHOS_TEST_FOR_EXCEPT(err != 0);
  return result;
}

RCP<Epetra_CrsMatrix> GaussNewtonOperatorFactory::reducedJacobianNew() {
  return jacobianFactory_.reducedMatrixNew();
}

const Epetra_CrsMatrix & GaussNewtonOperatorFactory::reducedJacobian(Epetra_CrsMatrix &result) const {
  return jacobianFactory_.reducedMatrix(*jacobianFactory_.premultipliedRightProjector(), result);
}

void GaussNewtonOperatorFactory::fullJacobianIs(const Epetra_Operator &op) {
  jacobianFactory_.fullJacobianIs(op);
}

} // namespace Albany
