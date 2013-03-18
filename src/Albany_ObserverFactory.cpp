//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Albany_ObserverFactory.hpp"

#include "Albany_NOXObserver.hpp"
#include "Albany_RythmosObserver.hpp"

#ifdef ALBANY_MOR
#include "MOR/MOR_ObserverFactory.hpp"
#endif

#include "Teuchos_ParameterList.hpp"

#include <string>

namespace Albany {

NOXObserverFactory::NOXObserverFactory(const Teuchos::RCP<Application> &app) :
  app_(app)
{}

Teuchos::RCP<NOX::Epetra::Observer>
NOXObserverFactory::operator()()
{
  Teuchos::RCP<NOX::Epetra::Observer> result(new Albany_NOXObserver(app_));
#ifdef ALBANY_MOR
  const Teuchos::RCP<MOR::ObserverFactory> morObserverFactory = app_->getMorFacade()->observerFactory();
  result = morObserverFactory->create(result);
#endif
  return result;
}

RythmosObserverFactory::RythmosObserverFactory(const Teuchos::RCP<Application> &app) :
  app_(app)
{}

Teuchos::RCP<Rythmos::IntegrationObserverBase<double> >
RythmosObserverFactory::operator()()
{
  Teuchos::RCP<Rythmos::IntegrationObserverBase<double> > result(new Albany_RythmosObserver(app_));
#ifdef ALBANY_MOR
  const Teuchos::RCP<MOR::ObserverFactory> morObserverFactory = app_->getMorFacade()->observerFactory();
  result = morObserverFactory->create(result);
#endif
  return result;
}

ObserverFactory::ObserverFactory(
    const Teuchos::RCP<Teuchos::ParameterList> &params,
    const Teuchos::RCP<Application> &app) :
  params_(params),
  app_(app)
{}

Teuchos::RCP<NOX::Epetra::Observer> ObserverFactory::createNoxObserver()
{
  if (useNOX()) {
    return NOXObserverFactory(app_)();
  } else {
    return Teuchos::null;
  }
}

Teuchos::RCP<Rythmos::IntegrationObserverBase<double> > ObserverFactory::createRythmosObserver() {
  if (useRythmos()) {
    return RythmosObserverFactory(app_)();
  } else {
    return Teuchos::null;
  }
}

bool ObserverFactory::useRythmos() const {
  const std::string solutionMethod = params_->get("Solution Method", "Steady");
  const std::string secondOrder = params_->get("Second Order", "No");

  return (solutionMethod == "Transient") && (secondOrder == "No");
}

bool ObserverFactory::useNOX() const {
  return !useRythmos();
}

} // namespace Albany
