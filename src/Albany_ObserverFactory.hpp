//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef ALBANY_OBSERVERFACTORY_HPP
#define ALBANY_OBSERVERFACTORY_HPP

#include "NOX_Epetra_Observer.H"
#include "Rythmos_IntegrationObserverBase.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

namespace Albany {

class Application;

class NOXObserverFactory {
public:
  explicit NOXObserverFactory(const Teuchos::RCP<Application> &app);

  Teuchos::RCP<NOX::Epetra::Observer> createInstance();

private:
  Teuchos::RCP<Application> app_;
};

class NOXStatelessObserverFactory {
public:
  explicit NOXStatelessObserverFactory(const Teuchos::RCP<Application> &app);

  Teuchos::RCP<NOX::Epetra::Observer> createInstance();

private:
  Teuchos::RCP<Application> app_;
};

class RythmosObserverFactory {
public:
  explicit RythmosObserverFactory(const Teuchos::RCP<Application> &app);

  Teuchos::RCP<Rythmos::IntegrationObserverBase<double> > createInstance();

private:
  Teuchos::RCP<Application> app_;
};

} // namespace Albany

#endif /* ALBANY_OBSERVERFACTORY_HPP */
