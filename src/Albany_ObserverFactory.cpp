//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
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
    const RCP<NOX::Epetra::Observer> observer(new Albany_NOXObserver(app_));
    const RCP<MORObserverFactory> morObserverFactory = app_->getMorFacade()->observerFactory();
    return morObserverFactory->create(observer);
  }
  return null;
}

RCP<Rythmos::IntegrationObserverBase<double> > ObserverFactory::createRythmosObserver() {
  if (useRythmos()) {
    const RCP<Rythmos::IntegrationObserverBase<double> > observer(new Albany_RythmosObserver(app_));
    const RCP<MORObserverFactory> morObserverFactory = app_->getMorFacade()->observerFactory();
    return morObserverFactory->create(observer);
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
