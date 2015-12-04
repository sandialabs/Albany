//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "AAdapt_AdaptiveModelFactory.hpp"
#include "AAdapt_ThyraAdaptiveModelEvaluator.hpp"

#include <string>
#include <stdexcept>

namespace AAdapt {

using ::Teuchos::RCP;
using ::Teuchos::rcp;
using ::Teuchos::nonnull;
using ::Teuchos::ParameterList;
using ::Teuchos::sublist;
using ::Teuchos::Tuple;
using ::Teuchos::tuple;

AdaptiveModelFactory::AdaptiveModelFactory(
  const RCP<ParameterList>& parentParams) :
  params(extractAdaptiveModelParams(parentParams)) {
  // Nothing to do
}

AdaptiveModelFactory::~AdaptiveModelFactory(){
   // Prevent circular RCP references by releasing RCP on destruction
   thyra_model = Teuchos::null;
#ifdef ALBANY_DEBUG
  std::cout << "Calling destructor for Albany_AdaptiveModelFactory" << std::endl;
#endif
}

RCP<ParameterList> AdaptiveModelFactory::extractAdaptiveModelParams(const RCP<ParameterList>& params_) {

  const Teuchos::RCP<Teuchos::ParameterList>& problemParams =
    Teuchos::sublist(params_, "Problem", true);

  if(problemParams->isSublist("Adaptation")) { // If the user has specified adaptation on input, grab the sublist

    return sublist(problemParams, "Adaptation");

  }

  return Teuchos::null;

}

Teuchos::RCP<Thyra::ModelEvaluator<ST> > 
AdaptiveModelFactory::create(const Teuchos::RCP<EpetraExt::ModelEvaluator>& child,
         const Teuchos::RCP<Thyra::LinearOpWithSolveFactoryBase<ST> > &W_factory){
  Teuchos::RCP<Thyra::ModelEvaluator<ST> > result;
  if(useAdaptiveModel()) {
      result = Teuchos::rcp(new ThyraAdaptiveModelEvaluator(child, W_factory));
  }
  else {
      result = Thyra::epetraModelEvaluator(child, W_factory);
  }

  // Keep only a weak pointer as member to avoid circular references
//  thyra_model = result.create_weak();
  thyra_model = result;

  return result;
}

bool AdaptiveModelFactory::useAdaptiveModel() const {

  if(Teuchos::nonnull(params)){

    return true;

  }

  return false;

}

} // namespace AAdapt
