//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef AADAPT_ADAPTIVEMODELFACTORY_HPP
#define AADAPT_ADAPTIVEMODELFACTORY_HPP

#include "EpetraExt_ModelEvaluator.h"

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

class Epetra_MultiVector;
class Epetra_Map;

namespace AAdapt {

class AdaptiveModelFactory {
  public:
    AdaptiveModelFactory(
      const Teuchos::RCP<Teuchos::ParameterList>& parentParams);

    Teuchos::RCP<EpetraExt::ModelEvaluator> create(const Teuchos::RCP<EpetraExt::ModelEvaluator>& child);

  private:

    Teuchos::RCP<Teuchos::ParameterList> params;

    static Teuchos::RCP<Teuchos::ParameterList> extractAdaptiveModelParams(const Teuchos::RCP<Teuchos::ParameterList>& source);

    bool useAdaptiveModel() const;

    // Disallow copy & assignment
    AdaptiveModelFactory(const AdaptiveModelFactory&);
    AdaptiveModelFactory& operator=(const AdaptiveModelFactory&);
};

} // namespace AAdapt

#endif /* ALBANY_ADAPTIVEMODELFACTORY_HPP */
