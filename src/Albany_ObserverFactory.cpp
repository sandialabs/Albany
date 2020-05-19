//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Albany_ObserverFactory.hpp"

#if defined(ALBANY_EPETRA)
#include "Albany_NOXObserver.hpp"
#include "Albany_NOXStatelessObserver.hpp"
#endif

#include "Teuchos_ParameterList.hpp"

#include <string>

namespace Albany {

NOXObserverFactory::NOXObserverFactory(const Teuchos::RCP<Application> &app) :
  app_(app)
{}

#if defined(ALBANY_EPETRA)
Teuchos::RCP<NOX::Epetra::Observer>
NOXObserverFactory::createInstance()
{
  Teuchos::RCP<NOX::Epetra::Observer> result(new NOXObserver(app_));
  return result;
}

NOXStatelessObserverFactory::
NOXStatelessObserverFactory (const Teuchos::RCP<Application> &app)
  : app_(app)
{}

Teuchos::RCP<NOX::Epetra::Observer>
NOXStatelessObserverFactory::createInstance () {
  Teuchos::RCP<NOX::Epetra::Observer> result(new NOXStatelessObserver(app_));
  return result;
}
#endif

} // namespace Albany
