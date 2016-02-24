//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef MOR_REDUCEDSPACEFACTORY_HPP
#define MOR_REDUCEDSPACEFACTORY_HPP

#include "MOR_ReducedBasisRepository.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"

#include <string>
#include <map>

class Epetra_Operator;
class Epetra_Map;

namespace MOR {

class ReducedBasisFactory;
class SampleDofListFactory;
class ReducedSpace;

class ReducedSpaceFactory {
public:
  ReducedSpaceFactory(
      const Teuchos::RCP<ReducedBasisFactory> &basisFactory,
      const Teuchos::RCP<SampleDofListFactory> &samplingFactory);

  Teuchos::RCP<ReducedSpace> create(const Teuchos::RCP<Teuchos::ParameterList> &params);

  Teuchos::RCP<const Epetra_MultiVector> getBasis(const Teuchos::RCP<Teuchos::ParameterList> &params);

  Teuchos::RCP<const Epetra_MultiVector> getProjector(const Teuchos::RCP<Teuchos::ParameterList> &params);

  Teuchos::RCP<const Epetra_Operator> getSamplingOperator(
      const Teuchos::RCP<Teuchos::ParameterList> &params,
      const Epetra_Map &stateMap);

private:
  ReducedBasisRepository basisRepository_;
  Teuchos::RCP<SampleDofListFactory> samplingFactory_;

  Teuchos::RCP<const Epetra_Vector> getOrigin(const Teuchos::RCP<Teuchos::ParameterList> &params);
};

} // end namepsace Albany

#endif /* MOR_REDUCEDSPACEFACTORY_HPP */
