//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Albany_MORObserverFactory.hpp"

#include "Albany_SnapshotCollectionObserver.hpp"
#include "Albany_ProjectionErrorObserver.hpp"
#include "Albany_FullStateReconstructor.hpp"

#include "Albany_RythmosSnapshotCollectionObserver.hpp"
#include "Albany_RythmosProjectionErrorObserver.hpp"
#include "Albany_RythmosFullStateReconstructor.hpp"

#include "Rythmos_CompositeIntegrationObserver.hpp"

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

RCP<Rythmos::IntegrationObserverBase<double> > MORObserverFactory::create(const RCP<Rythmos::IntegrationObserverBase<double> > &child) {
  RCP<Rythmos::IntegrationObserverBase<double> > result = child;

  if (collectSnapshots()) {
    const RCP<Rythmos::CompositeIntegrationObserver<double> > composite = Rythmos::createCompositeIntegrationObserver<double>();
    composite->addObserver(result);
    composite->addObserver(rcp(new RythmosSnapshotCollectionObserver(getSnapParameters(), rcp(new Epetra_Map(applicationMap_)))));
    result = composite;
  }

  if (computeProjectionError()) {
    const RCP<Rythmos::CompositeIntegrationObserver<double> > composite = Rythmos::createCompositeIntegrationObserver<double>();
    composite->addObserver(result);
    composite->addObserver(rcp(new RythmosProjectionErrorObserver(getErrorParameters(), rcp(new Epetra_Map(applicationMap_)))));
    result = composite;
  }

  if (useReducedOrderModel()) {
    result = rcp(new RythmosFullStateReconstructor(getReducedOrderModelParameters(), result, rcp(new Epetra_Map(applicationMap_))));
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
