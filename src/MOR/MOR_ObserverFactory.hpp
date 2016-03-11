//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef MOR_MOROBSERVERFACTORY_HPP
#define MOR_MOROBSERVERFACTORY_HPP

#include "NOX_Epetra_Observer.H"
#include "Rythmos_IntegrationObserverBase.hpp"

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Epetra_Map.h"

namespace MOR {

class ReducedSpaceFactory;

class ObserverFactory {
public:
  ObserverFactory(
      const Teuchos::RCP<ReducedSpaceFactory> &spaceFactory,
      const Teuchos::RCP<Teuchos::ParameterList> &parentParams);

  Teuchos::RCP<NOX::Epetra::Observer> create(const Teuchos::RCP<NOX::Epetra::Observer> &child);
  Teuchos::RCP<Rythmos::IntegrationObserverBase<double> > create(const Teuchos::RCP<Rythmos::IntegrationObserverBase<double> > &child);

private:
  bool collectSnapshots() const;
  bool computeProjectionError() const;
  bool useReducedOrderModel() const;
  bool observeGeneralizedCoordinates() const;

  Teuchos::RCP<Teuchos::ParameterList> getSnapParameters() const;
  Teuchos::RCP<Teuchos::ParameterList> getErrorParameters() const;
  Teuchos::RCP<Teuchos::ParameterList> getReducedOrderModelParameters() const;
  Teuchos::RCP<Teuchos::ParameterList> getGeneralizedCoordinatesParameters() const;

  Teuchos::RCP<ReducedSpaceFactory> spaceFactory_;
  Teuchos::RCP<Teuchos::ParameterList> params_;

  // Disallow copy & assignment
  ObserverFactory(const ObserverFactory &);
  ObserverFactory &operator=(const ObserverFactory &);
};

} // namespace MOR

#endif /* MOR_MOROBSERVERFACTORY_HPP */
