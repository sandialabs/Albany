//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/12/14: Epetra ifdef'ed out!

#include "Albany_ModelFactory.hpp"

#include "Albany_Application.hpp"
#if defined(ALBANY_EPETRA)
#include "Albany_ModelEvaluator.hpp"
#endif
#include "Albany_ModelEvaluatorT.hpp"


#ifdef ALBANY_MOR
#if defined(ALBANY_EPETRA)
#include "MOR_ReducedOrderModelFactory.hpp"
#endif
#endif

namespace Albany {

using Teuchos::RCP;
using Teuchos::ParameterList;

ModelFactory::ModelFactory(const RCP<ParameterList> &params,
                           const RCP<Application> &app) :
  params_(params),
  app_(app)
{
  // Nothing to do
}

#if defined(ALBANY_EPETRA)
RCP<EpetraExt::ModelEvaluator> ModelFactory::create() const
{
  RCP<EpetraExt::ModelEvaluator> model(new Albany::ModelEvaluator(app_, params_));

  // Wrap a decorator around the original model when an adaptive computation is requested.
//  const RCP<AAdapt::AdaptiveModelFactory> adaptMdlFactory = app_->getAdaptSolMgr()->modelFactory();
//  model = adaptMdlFactory->create(model);

#ifdef ALBANY_MOR
  if(app_->getDiscretization()->supportsMOR()){
    // Wrap a decorator around the original model when a reduced-order computation is requested.
    const RCP<MOR::ReducedOrderModelFactory> romFactory = app_->getMorFacade()->modelFactory();
    model = romFactory->create(model);
  }
#endif

  return model;
}
#endif

RCP<Thyra::ModelEvaluatorDefaultBase<ST> > ModelFactory::createT() const
{
  RCP<Thyra::ModelEvaluatorDefaultBase<ST> > modelT(new Albany::ModelEvaluatorT(app_, params_)); 
  
  // Wrap a decorator around the original model when a reduced-order computation is requested.
  const RCP<ParameterList> problemParams = Teuchos::sublist(params_, "Problem", true);
  //WILL NEED TO CONVERT ROM STUFF TO THYRA!
  //ReducedOrderModelFactory romFactory(problemParams);
  //model = romFactory.create(model);
  
  return modelT;
}

} // end namespace Albany
