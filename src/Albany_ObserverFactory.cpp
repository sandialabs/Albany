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

#include "Albany_ObserverFactory.hpp"

#include "Albany_NOXObserver.hpp"
#include "Albany_RythmosObserver.hpp"

#include "MOR/Albany_MORObserverFactory.hpp"

#include "Teuchos_ParameterList.hpp"

#include <string>

namespace Albany {

using ::Teuchos::RCP;
using ::Teuchos::rcp;
using ::Teuchos::null;
using ::Teuchos::ParameterList;

ObserverFactory::ObserverFactory(const RCP<ParameterList> &params,
                                 const RCP<Application> &app) :
  params_(params),
  app_(app)
{
  // Nothing to do
}

RCP<NOX::Epetra::Observer> ObserverFactory::createNoxObserver()
{
  if (useNOX()) {
    RCP<NOX::Epetra::Observer> observer(new Albany_NOXObserver(app_)); 
    MORObserverFactory morFactory(params_);  
    return morFactory.create(observer);
  }
  return null;
}

RCP<Rythmos::IntegrationObserverBase<double> > ObserverFactory::createRythmosObserver() {
  if (useRythmos()) {
    return rcp(new Albany_RythmosObserver(app_));
  }
  return null;
}

bool ObserverFactory::useRythmos() const {
  const std::string solutionMethod = params_->get("Solution Method", "Steady");
  const std::string secondOrder = params_->get("Second Order", "No");

  return (solutionMethod == "Transient") && (secondOrder == "No");
}

bool ObserverFactory::useNOX() const {
  return !useRythmos();
}

}
