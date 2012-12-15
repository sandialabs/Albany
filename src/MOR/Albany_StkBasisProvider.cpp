//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_StkBasisProvider.hpp"

#include "Albany_STKDiscretization.hpp"

namespace Albany {

StkBasisProvider::StkBasisProvider(const Teuchos::RCP<STKDiscretization> &disc) :
  disc_(disc)
{
  // Nothing to do
}

Teuchos::RCP<Epetra_MultiVector> StkBasisProvider::operator()(const Teuchos::RCP<Teuchos::ParameterList> &/*params*/)
{
  return disc_->getSolutionFieldHistory();
}

} // end namepsace Albany
