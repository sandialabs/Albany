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

#ifndef ALBANY_OBSERVERFACTORY_HPP
#define ALBANY_OBSERVERFACTORY_HPP

#include "NOX_Epetra_Observer.H"
#include "Rythmos_IntegrationObserverBase.hpp"

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
};

}

#endif /* ALBANY_OBSERVERFACTORY_HPP */
