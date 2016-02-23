//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef MOR_REDUCEDORDERMODELFACTORY_HPP
#define MOR_REDUCEDORDERMODELFACTORY_HPP

#include "EpetraExt_ModelEvaluator.h"

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

class Epetra_MultiVector;
class Epetra_Map;

namespace MOR {

class ReducedSpaceFactory;

class ReducedOrderModelFactory {
public:
  ReducedOrderModelFactory(
      const Teuchos::RCP<ReducedSpaceFactory> &spaceFactory,
      const Teuchos::RCP<Teuchos::ParameterList> &parentParams);

  Teuchos::RCP<EpetraExt::ModelEvaluator> create(const Teuchos::RCP<EpetraExt::ModelEvaluator> &child);

private:
  Teuchos::RCP<ReducedSpaceFactory> spaceFactory_;
  Teuchos::RCP<Teuchos::ParameterList> params_;

  static Teuchos::RCP<Teuchos::ParameterList> extractModelOrderReductionParams(const Teuchos::RCP<Teuchos::ParameterList> &source);
  static Teuchos::RCP<Teuchos::ParameterList> extractReducedOrderModelParams(const Teuchos::RCP<Teuchos::ParameterList> &source);

  bool useReducedOrderModel() const;

  // Disallow copy & assignment
  ReducedOrderModelFactory(const ReducedOrderModelFactory &);
  ReducedOrderModelFactory &operator=(const ReducedOrderModelFactory &);
};

} // namespace MOR

#endif /* MOR_REDUCEDORDERMODELFACTORY_HPP */
