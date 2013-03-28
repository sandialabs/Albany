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
NOXObserverFactory::createInstance()
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
RythmosObserverFactory::createInstance()
{
  Teuchos::RCP<Rythmos::IntegrationObserverBase<double> > result(new Albany_RythmosObserver(app_));
#ifdef ALBANY_MOR
  const Teuchos::RCP<MOR::ObserverFactory> morObserverFactory = app_->getMorFacade()->observerFactory();
  result = morObserverFactory->create(result);
#endif
  return result;
}

} // namespace Albany
