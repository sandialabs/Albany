//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef ALBANY_LINEARREDUCEDSPACEFACTORY_HPP
#define ALBANY_LINEARREDUCEDSPACEFACTORY_HPP

#include "Albany_ReducedBasisRepository.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"

#include <string>
#include <map>

namespace Albany {

class ReducedBasisFactory;
class LinearReducedSpace;

class LinearReducedSpaceFactory {
public:
  explicit LinearReducedSpaceFactory(const Teuchos::RCP<ReducedBasisFactory> &basisFactory);

  Teuchos::RCP<LinearReducedSpace> create(const Teuchos::RCP<Teuchos::ParameterList> &params);

private:
  ReducedBasisRepository basisRepository_;
};

} // end namepsace Albany

#endif /* ALBANY_LINEARREDUCEDSPACEFACTORY_HPP */
