//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_StkBasisProvider.hpp"

#include "Teuchos_Ptr.hpp"

namespace Albany {

StkBasisProvider::StkBasisProvider(const Teuchos::RCP<STKDiscretization> &disc) :
  disc_(disc)
{
  // Nothing to do
}

MOR::ReducedBasisElements
StkBasisProvider::operator()(const Teuchos::RCP<Teuchos::ParameterList> &params)
{
  const Teuchos::Ptr<const int> maxVecCount(params->getPtr<int>("Basis Size Max"));
  if (Teuchos::nonnull(maxVecCount)) {
    return disc_->getSolutionFieldHistory(*maxVecCount);
  } else {
    return disc_->getSolutionFieldHistory();
  }
}

} // end namepsace Albany
