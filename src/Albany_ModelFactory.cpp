//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Albany_ModelFactory.hpp"

#include "Albany_Application.hpp"
#include "Albany_ModelEvaluator.hpp"

#include "MOR/Albany_ReducedOrderModelFactory.hpp"

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

RCP<EpetraExt::ModelEvaluator> ModelFactory::create() const
{
  RCP<EpetraExt::ModelEvaluator> model(new Albany::ModelEvaluator(app_, params_));

#if !defined(ALBANY_SCOREC)
  //FIXME getMorFacade() returns Facade (not) created on line 264 of Albany_Application.cpp as it assumes disc is STK
  // Wrap a decorator around the original model when a reduced-order computation is requested.
  const RCP<ReducedOrderModelFactory> romFactory = app_->getMorFacade()->modelFactory();
  model = romFactory->create(model);
#endif

  return model;
}

} // end namespace Albany
