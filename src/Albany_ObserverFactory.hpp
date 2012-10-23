//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
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

class ObserverFactory {
public:
  ObserverFactory(const Teuchos::RCP<Teuchos::ParameterList> &params,
                  const Teuchos::RCP<Application> &app);

  Teuchos::RCP<NOX::Epetra::Observer> createNoxObserver();
  Teuchos::RCP<Rythmos::IntegrationObserverBase<double> > createRythmosObserver();

private:
  bool useNOX() const;
  bool useRythmos() const;

  Teuchos::RCP<Teuchos::ParameterList> params_;
  Teuchos::RCP<Application> app_;

  // Disallow copy & assignment
  ObserverFactory(const ObserverFactory &);
  ObserverFactory &operator=(const ObserverFactory &);
};

}

#endif /* ALBANY_OBSERVERFACTORY_HPP */
