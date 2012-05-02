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

namespace Albany {

using Teuchos::RCP;
using Teuchos::rcp;
using Teuchos::ParameterList;

ModelFactory::ModelFactory(const Teuchos::RCP<Teuchos::ParameterList> &params,
                           const Teuchos::RCP<Application> &app) :
  params_(params),
  app_(app)
{
  // Nothing to do
}

RCP<EpetraExt::ModelEvaluator> ModelFactory::create() const
{
  return rcp(new Albany::ModelEvaluator(app_, params_));
}

} // end namespace Albany
