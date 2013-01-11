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

class Epetra_Operator;
class Epetra_Map;

namespace Albany {

class ReducedBasisFactory;
class SampleDofListFactory;
class LinearReducedSpace;

class LinearReducedSpaceFactory {
public:
  LinearReducedSpaceFactory(
      const Teuchos::RCP<ReducedBasisFactory> &basisFactory,
      const Teuchos::RCP<SampleDofListFactory> &samplingFactory);

  Teuchos::RCP<LinearReducedSpace> create(const Teuchos::RCP<Teuchos::ParameterList> &params);

  Teuchos::RCP<const Epetra_MultiVector> getBasis(const Teuchos::RCP<Teuchos::ParameterList> &params);
  Teuchos::RCP<const Epetra_MultiVector> getProjector(const Teuchos::RCP<Teuchos::ParameterList> &params);
  Teuchos::RCP<const Epetra_Operator> getSamplingOperator(
      const Teuchos::RCP<Teuchos::ParameterList> &params,
      const Epetra_Map &stateMap);

private:
  ReducedBasisRepository basisRepository_;
  Teuchos::RCP<SampleDofListFactory> samplingFactory_;
};

} // end namepsace Albany

#endif /* ALBANY_LINEARREDUCEDSPACEFACTORY_HPP */
