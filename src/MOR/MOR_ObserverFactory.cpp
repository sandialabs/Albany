//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "MOR_ObserverFactory.hpp"

#include "MOR_MultiVectorOutputFile.hpp"
#include "MOR_MultiVectorOutputFileFactory.hpp"
#include "MOR_ReducedSpace.hpp"
#include "MOR_ReducedSpaceFactory.hpp"

#include "MOR_NOXEpetraCompositeObserver.hpp"
#include "MOR_SnapshotCollectionObserver.hpp"
#include "MOR_ProjectionErrorObserver.hpp"
#include "MOR_FullStateReconstructor.hpp"
#include "MOR_GeneralizedCoordinatesNOXObserver.hpp"

#include "MOR_RythmosSnapshotCollectionObserver.hpp"
#include "MOR_RythmosProjectionErrorObserver.hpp"
#include "MOR_RythmosFullStateReconstructor.hpp"
#include "MOR_GeneralizedCoordinatesRythmosObserver.hpp"

#include "Rythmos_CompositeIntegrationObserver.hpp"

#include "Teuchos_ParameterList.hpp"

#include <string>

namespace MOR {

using ::Teuchos::RCP;
using ::Teuchos::rcp;
using ::Teuchos::ParameterList;
using ::Teuchos::sublist;

namespace { // anonymous

RCP<ParameterList> fillDefaultOutputParams(const RCP<ParameterList> &params, std::string defaultValue)
{
  params->get("Output File Group Name", defaultValue);
  params->get("Output File Default Base File Name", defaultValue);
  return params;
}

RCP<ParameterList> fillDefaultProjectionErrorOutputParams(const RCP<ParameterList> &params)
{
  return fillDefaultOutputParams(params, "proj_error");
}

RCP<ParameterList> fillDefaultSnapshotOutputParams(const RCP<ParameterList> &params)
{
  return fillDefaultOutputParams(params, "snapshots");
}

RCP<MultiVectorOutputFile> createOutputFile(const RCP<ParameterList> &params)
{
  MultiVectorOutputFileFactory factory(params);
  return factory.create();
}

RCP<MultiVectorOutputFile> createProjectionErrorOutputFile(const RCP<ParameterList> &params)
{
  return createOutputFile(fillDefaultProjectionErrorOutputParams(params));
}

RCP<MultiVectorOutputFile> createSnapshotOutputFile(const RCP<ParameterList> &params)
{
  return createOutputFile(fillDefaultSnapshotOutputParams(params));
}

int getSnapshotPeriod(const RCP<ParameterList> &params)
{
  return params->get("Period", 1);
}

std::string getGeneralizedCoordinatesFilename(const RCP<ParameterList> &params)
{
  return params->get("Generalized Coordinates Output File Name", "generalized_coordinates.mtx");
}

std::string getGeneralizedCoordinatesStampsFilename(const RCP<ParameterList> &params)
{
  const std::string defaultValue = "stamps_" + getGeneralizedCoordinatesFilename(params);
  return params->get("Generalized Coordinates Stamps Output File Name", defaultValue);
}

} // end anonymous namespace

ObserverFactory::ObserverFactory(
      const Teuchos::RCP<ReducedSpaceFactory> &spaceFactory,
      const Teuchos::RCP<Teuchos::ParameterList> &parentParams) :
  spaceFactory_(spaceFactory),
  params_(sublist(parentParams, "Model Order Reduction"))
{
  // Nothing to do
}

RCP<NOX::Epetra::Observer> ObserverFactory::create(const RCP<NOX::Epetra::Observer> &child)
{
  RCP<NOX::Epetra::Observer> fullOrderObserver;
  {
    const RCP<NOXEpetraCompositeObserver> composite(new NOXEpetraCompositeObserver);

    if (this->collectSnapshots()) {
      const RCP<ParameterList> params = this->getSnapParameters();
      const RCP<MultiVectorOutputFile> snapOutputFile = createSnapshotOutputFile(params);
      const int period = getSnapshotPeriod(params);
      composite->addObserver(rcp(new SnapshotCollectionObserver(period, snapOutputFile)));
    }

    if (this->computeProjectionError()) {
      const RCP<ParameterList> params = this->getErrorParameters();
      const RCP<MultiVectorOutputFile> errorOutputFile = createProjectionErrorOutputFile(params);
      const RCP<ReducedSpace> projectionSpace = spaceFactory_->create(params);
      composite->addObserver(rcp(new ProjectionErrorObserver(projectionSpace, errorOutputFile)));
    }

    if (composite->observerCount() > 0) {
      composite->addObserver(child);
      fullOrderObserver = composite;
    } else {
      fullOrderObserver = child;
    }
  }

  if (this->useReducedOrderModel()) {
    const RCP<ParameterList> params = this->getReducedOrderModelParameters();
    const RCP<ReducedSpace> projectionSpace = spaceFactory_->create(params);
    const RCP<NOX::Epetra::Observer> reducedOrderObserver =
      rcp(new FullStateReconstructor(projectionSpace, fullOrderObserver));

    if (this->observeGeneralizedCoordinates()) {
      const RCP<NOXEpetraCompositeObserver> composite(new NOXEpetraCompositeObserver);
      const RCP<ParameterList> genCoordParams = this->getGeneralizedCoordinatesParameters();
      const std::string filename = getGeneralizedCoordinatesFilename(genCoordParams);
      const std::string stampsFilename = getGeneralizedCoordinatesStampsFilename(genCoordParams);
      composite->addObserver(rcp(new GeneralizedCoordinatesNOXObserver(filename, stampsFilename)));
      composite->addObserver(reducedOrderObserver);
      return composite;
    } else {
      return reducedOrderObserver;
    }
  } else {
    return fullOrderObserver;
  }
}

RCP<Rythmos::IntegrationObserverBase<double> > ObserverFactory::create(const RCP<Rythmos::IntegrationObserverBase<double> > &child) {
  RCP<Rythmos::IntegrationObserverBase<double> > fullOrderObserver;
  {
    const RCP<Rythmos::CompositeIntegrationObserver<double> > composite =
      Rythmos::createCompositeIntegrationObserver<double>();
    int observersInComposite = 0;

    if (this->collectSnapshots()) {
      const RCP<ParameterList> params = this->getSnapParameters();
      const RCP<MultiVectorOutputFile> snapOutputFile = createSnapshotOutputFile(params);
      const int period = getSnapshotPeriod(params);
      composite->addObserver(rcp(new RythmosSnapshotCollectionObserver(period, snapOutputFile)));
      ++observersInComposite;
    }

    if (this->computeProjectionError()) {
      const RCP<ParameterList> params = this->getErrorParameters();
      const RCP<MultiVectorOutputFile> errorOutputFile = createProjectionErrorOutputFile(params);
      const RCP<ReducedSpace> projectionSpace = spaceFactory_->create(params);
      composite->addObserver(rcp(new RythmosProjectionErrorObserver(projectionSpace, errorOutputFile)));
      ++observersInComposite;
    }

    if (observersInComposite > 0) {
      composite->addObserver(child);
      fullOrderObserver = composite;
    } else {
      fullOrderObserver = child;
    }
  }

  if (useReducedOrderModel()) {
    const RCP<ParameterList> params = this->getReducedOrderModelParameters();
    const RCP<ReducedSpace> projectionSpace = spaceFactory_->create(params);
    const RCP<Rythmos::IntegrationObserverBase<double> > reducedOrderObserver =
      rcp(new RythmosFullStateReconstructor(projectionSpace, fullOrderObserver));

    if (this->observeGeneralizedCoordinates()) {
      const RCP<Rythmos::CompositeIntegrationObserver<double> > composite =
        Rythmos::createCompositeIntegrationObserver<double>();
      const RCP<ParameterList> genCoordParams = this->getGeneralizedCoordinatesParameters();
      const std::string filename = getGeneralizedCoordinatesFilename(genCoordParams);
      const std::string stampsFilename = getGeneralizedCoordinatesStampsFilename(genCoordParams);
      composite->addObserver(rcp(new GeneralizedCoordinatesRythmosObserver(filename, stampsFilename)));
      composite->addObserver(reducedOrderObserver);
      return composite;
    } else {
      return reducedOrderObserver;
    }
  } else {
    return fullOrderObserver;
  }
}

bool ObserverFactory::collectSnapshots() const
{
  return getSnapParameters()->get("Activate", false);
}

bool ObserverFactory::computeProjectionError() const
{
  return getErrorParameters()->get("Activate", false);
}

bool ObserverFactory::useReducedOrderModel() const
{
  return getReducedOrderModelParameters()->get("Activate", false);
}

bool ObserverFactory::observeGeneralizedCoordinates() const
{
  return Teuchos::isParameterType<std::string>(
      *this->getGeneralizedCoordinatesParameters(),
      "Generalized Coordinates Output File Name");
}

RCP<ParameterList> ObserverFactory::getSnapParameters() const
{
  return sublist(params_, "Snapshot Collection");
}

RCP<ParameterList> ObserverFactory::getErrorParameters() const
{
  return sublist(params_, "Projection Error");
}

RCP<ParameterList> ObserverFactory::getReducedOrderModelParameters() const
{
  return sublist(params_, "Reduced-Order Model");
}

RCP<ParameterList> ObserverFactory::getGeneralizedCoordinatesParameters() const
{
  return this->getReducedOrderModelParameters();
}

} // namespace MOR
