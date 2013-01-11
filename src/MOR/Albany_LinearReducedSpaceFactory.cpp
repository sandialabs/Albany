//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Albany_LinearReducedSpaceFactory.hpp"

#include "Albany_ReducedSpace.hpp"

#include "Epetra_MultiVector.h"

namespace Albany {

LinearReducedSpaceFactory::LinearReducedSpaceFactory(const Teuchos::RCP<ReducedBasisFactory> &basisFactory) :
  basisRepository_(basisFactory)
{
  // Nothing to do
}

Teuchos::RCP<LinearReducedSpace>
LinearReducedSpaceFactory::create(const Teuchos::RCP<Teuchos::ParameterList> &params)
{
  const Teuchos::RCP<const Epetra_MultiVector> basis = this->getBasis(params);
  return Teuchos::nonnull(basis) ? Teuchos::rcp(new LinearReducedSpace(*basis)) : Teuchos::null;
}

Teuchos::RCP<const Epetra_MultiVector>
LinearReducedSpaceFactory::getBasis(const Teuchos::RCP<Teuchos::ParameterList> &params)
{
  return basisRepository_.get(params);
}

} // end namespace Albany
