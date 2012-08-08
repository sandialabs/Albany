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
