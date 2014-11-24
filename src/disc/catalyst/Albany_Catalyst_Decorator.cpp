//*****************************************************************//
//    Albany 2.0:  Copyright 2013 Kitware Inc.                     //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_Catalyst_Adapter.hpp"
#include "Albany_Catalyst_Decorator.hpp"
#include "Albany_Catalyst_Grid.hpp"
#include "Albany_Catalyst_TeuchosArrayRCPDataArray.hpp"

#include <Shards_BasicTopologies.hpp>
#include <Shards_CellTopology.hpp>
#include <Shards_CellTopologyData.h>

#include <vtkNew.h>
#include <vtkPoints.h>

namespace Albany {
namespace Catalyst {

Decorator::Decorator(
    Teuchos::RCP<AbstractDiscretization> discretization_,
    const Teuchos::RCP<Teuchos::ParameterList>& catalystParams_)
  : discretization(discretization_),
    timestep(0)
{
  Adapter::initialize(catalystParams_);
}

Decorator::~Decorator()
{
  Adapter::cleanup();
}

Teuchos::RCP<const Epetra_Map> Decorator::getMap() const
{
  return discretization->getMap();
}

Teuchos::RCP<const Epetra_Map> Decorator::getOverlapMap() const
{
  return discretization->getOverlapMap();
}

Teuchos::RCP<const Epetra_CrsGraph> Decorator::getJacobianGraph() const
{
  return discretization->getJacobianGraph();
}

Teuchos::RCP<const Epetra_CrsGraph>
Decorator::getOverlapJacobianGraph() const
{
  return discretization->getOverlapJacobianGraph();
}

Teuchos::RCP<const Epetra_Map> Decorator::getNodeMap() const
{
  return discretization->getNodeMap();
}

const NodeSetList &Decorator::getNodeSets() const
{
  return discretization->getNodeSets();
}

const NodeSetCoordList &Decorator::getNodeSetCoords() const
{
  return discretization->getNodeSetCoords();
}

const SideSetList &Decorator::getSideSets(const int workset) const
{
  return discretization->getSideSets(workset);
}

WsLIDList &Decorator::getElemGIDws()
{
  return discretization->getElemGIDws();
}

const WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > > >::type &
Decorator::getWsElNodeEqID() const
{
  return discretization->getWsElNodeEqID();
}

Teuchos::ArrayRCP<double> &Decorator::getCoordinates() const
{
  return discretization->getCoordinates();
}

const WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >::type &Decorator::getCoords() const
{
  return discretization->getCoords();
}


void Decorator::printCoords() const
{
  discretization->printCoords();
}

Teuchos::RCP<AbstractMeshStruct> Decorator::getMeshStruct() const
{
  return discretization->getMeshStruct();
}

StateArrays &Decorator::getStateArrays()
{
  return discretization->getStateArrays();
}

const WorksetArray<std::string>::type &Decorator::getWsEBNames() const
{
  return discretization->getWsEBNames();
}

const WorksetArray<int>::type &Decorator::getWsPhysIndex() const
{
  return discretization->getWsPhysIndex();
}

void Decorator::writeSolution(
    const Epetra_Vector &soln, const double time, const bool overlapped)
{
  Adapter *adapter = Adapter::get();
  if (adapter)
    adapter->update(this->timestep++, time, *this, soln);

  discretization->writeSolution(soln, time, overlapped);
}

vtkUnstructuredGridBase *Decorator::newVtkUnstructuredGrid()
{
  vtkNew<TeuchosArrayRCPDataArray<double> > coords;
  coords->SetArrayRCP(this->getCoordinates(), 3);

  vtkNew<vtkPoints> points;
  points->SetData(coords.GetPointer());

  Grid *grid = Grid::New();
  grid->SetPoints(points.GetPointer());
  grid->GetImplementation()->SetDecorator(this);

  return grid;
}

Teuchos::RCP<Epetra_Vector> Decorator::getSolutionField() const
{
  return discretization->getSolutionField();
}

void Decorator::setResidualField(const Epetra_Vector &residual)
{
  discretization->setResidualField(residual);
}

bool Decorator::hasRestartSolution() const
{
  return discretization->hasRestartSolution();
}

bool Decorator::supportsMOR() const
{
// Currently, the MOR library will dynamic cast the discretication to the STK
// implementation. Since this decorator cannot be cast to the STK class, the
// code will crash. For now, just return false.
//  return discretization->supportsMOR();
  return false;
}

double Decorator::restartDataTime() const
{
  return discretization->restartDataTime();
}

int Decorator::getNumDim() const
{
  return discretization->getNumDim();
}

int Decorator::getNumEq() const
{
  return discretization->getNumEq();
}

} // end namespace Catalyst
} // end namespace Albany
