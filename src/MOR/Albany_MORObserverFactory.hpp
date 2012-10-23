//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef ALBANY_MOROBSERVERFACTORY_HPP
#define ALBANY_MOROBSERVERFACTORY_HPP

#include "NOX_Epetra_Observer.H"
#include "Rythmos_IntegrationObserverBase.hpp"

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Epetra_Map.h"

namespace Albany {

class MORObserverFactory {
public:
  MORObserverFactory(const Teuchos::RCP<Teuchos::ParameterList> &parentParams,
                     const Epetra_Map &applicationMap);

  Teuchos::RCP<NOX::Epetra::Observer> create(const Teuchos::RCP<NOX::Epetra::Observer> &child);
  Teuchos::RCP<Rythmos::IntegrationObserverBase<double> > create(const Teuchos::RCP<Rythmos::IntegrationObserverBase<double> > &child);

private:
  bool collectSnapshots() const;
  bool computeProjectionError() const;
  bool useReducedOrderModel() const;

  Teuchos::RCP<Teuchos::ParameterList> getSnapParameters() const;
  Teuchos::RCP<Teuchos::ParameterList> getErrorParameters() const;
  Teuchos::RCP<Teuchos::ParameterList> getReducedOrderModelParameters() const;

  Teuchos::RCP<Teuchos::ParameterList> params_;
  Epetra_Map applicationMap_;

  // Disallow copy & assignment
  MORObserverFactory(const MORObserverFactory &);
  MORObserverFactory &operator=(const MORObserverFactory &);
};

} // end namespace Albany

#endif /* ALBANY_MOROBSERVERFACTORY_HPP */
