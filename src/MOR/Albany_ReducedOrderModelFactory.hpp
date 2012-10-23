//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef ALBANY_REDUCEDORDERMODELFACTORY_HPP
#define ALBANY_REDUCEDORDERMODELFACTORY_HPP

#include "EpetraExt_ModelEvaluator.h"

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

class Epetra_MultiVector;
class Epetra_Map;

namespace Albany {

class ReducedOrderModelFactory {
public:
  explicit ReducedOrderModelFactory(const Teuchos::RCP<Teuchos::ParameterList> &parentParams);

  Teuchos::RCP<EpetraExt::ModelEvaluator> create(const Teuchos::RCP<EpetraExt::ModelEvaluator> &child);

private:
  Teuchos::RCP<Teuchos::ParameterList> params_;

  static Teuchos::RCP<Teuchos::ParameterList> extractModelOrderReductionParams(const Teuchos::RCP<Teuchos::ParameterList> &source);
  static Teuchos::RCP<Teuchos::ParameterList> extractReducedOrderModelParams(const Teuchos::RCP<Teuchos::ParameterList> &source);

  bool useReducedOrderModel() const;

  static Teuchos::RCP<Epetra_MultiVector> createOrthonormalBasis(const Epetra_Map &fullStateMap,
                                                                 const Teuchos::RCP<Teuchos::ParameterList> &params);

  // Disallow copy & assignment
  ReducedOrderModelFactory(const ReducedOrderModelFactory &);
  ReducedOrderModelFactory &operator=(const ReducedOrderModelFactory &);
};

} // end namespace Albany

#endif /* ALBANY_REDUCEDORDERMODELFACTORY_HPP */
