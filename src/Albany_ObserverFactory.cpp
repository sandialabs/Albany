//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Albany_ObserverFactory.hpp"

#include "Albany_NOXObserver.hpp"
#include "Albany_NOXStatelessObserver.hpp"
#include "Albany_RythmosObserver.hpp"

#ifdef ALBANY_MOR
#if defined(ALBANY_EPETRA)
#include "MOR_ObserverFactory.hpp"
#endif
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
#if defined(ALBANY_EPETRA)
  if(app_->getDiscretization()->supportsMOR()){
    const Teuchos::RCP<MOR::ObserverFactory> morObserverFactory = app_->getMorFacade()->observerFactory();
    result = morObserverFactory->create(result);
  }
#endif
#endif
  return result;
}

NOXStatelessObserverFactory::
NOXStatelessObserverFactory (const Teuchos::RCP<Application> &app)
  : app_(app)
{}

Teuchos::RCP<NOX::Epetra::Observer>
NOXStatelessObserverFactory::createInstance () {
  Teuchos::RCP<NOX::Epetra::Observer> result(new NOXStatelessObserver(app_));
#ifdef ALBANY_MOR
#if defined(ALBANY_EPETRA)
  if (app_->getDiscretization()->supportsMOR()) {
    const Teuchos::RCP<MOR::ObserverFactory>
      morObserverFactory = app_->getMorFacade()->observerFactory();
    result = morObserverFactory->create(result);
  }
#endif
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
#if defined(ALBANY_EPETRA)
  if(app_->getDiscretization()->supportsMOR()){
    const Teuchos::RCP<MOR::ObserverFactory> morObserverFactory = app_->getMorFacade()->observerFactory();
    result = morObserverFactory->create(result);
  }
#endif
#endif
  return result;
}

} // namespace Albany
