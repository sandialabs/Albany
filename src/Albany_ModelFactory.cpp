/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/

#include "Albany_ModelFactory.hpp"

#include "Albany_Application.hpp"
#include "Albany_ModelEvaluator.hpp"
#include "Albany_ModelEvaluatorT.hpp"

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
  
  // Wrap a decorator around the original model when a reduced-order computation is requested.
  const RCP<ParameterList> problemParams = Teuchos::sublist(params_, "Problem", true);
  ReducedOrderModelFactory romFactory(problemParams);
  model = romFactory.create(model);
  
  return model;
}

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
