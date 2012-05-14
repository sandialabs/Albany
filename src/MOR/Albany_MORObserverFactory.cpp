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

#include "Albany_MORObserverFactory.hpp"

#include "Albany_SnapshotCollectionObserver.hpp"
#include "Albany_ProjectionErrorObserver.hpp"
#include "Albany_FullStateReconstructor.hpp"

#include "Teuchos_ParameterList.hpp"

#include <string>

namespace Albany {

using ::Teuchos::RCP;
using ::Teuchos::rcp;
using ::Teuchos::ParameterList;
using ::Teuchos::sublist;

MORObserverFactory::MORObserverFactory(const RCP<ParameterList> &parentParams,
                                       const Epetra_Map &applicationMap) :
  params_(sublist(parentParams, "Model Order Reduction")),
  applicationMap_(applicationMap)
{
  // Nothing to do
}

RCP<NOX::Epetra::Observer> MORObserverFactory::create(const RCP<NOX::Epetra::Observer> &child)
{
  RCP<NOX::Epetra::Observer> result = child;
  
  if (collectSnapshots()) {
    result = rcp(new SnapshotCollectionObserver(getSnapParameters(), result));
  }
  
  if (computeProjectionError()) {
    result = rcp(new ProjectionErrorObserver(getErrorParameters(), result, rcp(new Epetra_Map(applicationMap_))));
  }

  if (useReducedOrderModel()) {
    result = rcp(new FullStateReconstructor(getReducedOrderModelParameters(), result, applicationMap_));
  }

  return result;
}

bool MORObserverFactory::collectSnapshots() const
{
  return getSnapParameters()->get("Activate", false);
}

bool MORObserverFactory::computeProjectionError() const
{
  return getErrorParameters()->get("Activate", false);
}

bool MORObserverFactory::useReducedOrderModel() const
{
  return getReducedOrderModelParameters()->get("Activate", false);
}

RCP<ParameterList> MORObserverFactory::getSnapParameters() const
{
  return sublist(params_, "Snapshot Collection");
}

RCP<ParameterList> MORObserverFactory::getErrorParameters() const
{
  return sublist(params_, "Projection Error");
}

RCP<ParameterList> MORObserverFactory::getReducedOrderModelParameters() const
{
  return sublist(params_, "Reduced-Order Model");
}

} // end namespace Albany
