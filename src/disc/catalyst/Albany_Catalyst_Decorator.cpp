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

Teuchos::RCP<const Tpetra_Map> Decorator::getMapT() const {
  return discretization->getMapT();
}

Teuchos::RCP<const Tpetra_Map> Decorator::getMapT(const std::string& field_name) const {
  return discretization->getMapT(field_name);
}

Teuchos::RCP<const Tpetra_Map> Decorator::getOverlapMapT() const {
  return discretization->getOverlapMapT();
}

Teuchos::RCP<const Tpetra_Map> Decorator::getOverlapMapT(
    const std::string& field_name) const {
  return discretization->getOverlapMapT();
}

Teuchos::RCP<const Epetra_Map> Decorator::getOverlapMap() const
{
  return discretization->getOverlapMap();
}

Teuchos::RCP<const Epetra_CrsGraph> Decorator::getJacobianGraph() const
{
  return discretization->getJacobianGraph();
}

Teuchos::RCP<const Tpetra_CrsGraph> Decorator::getJacobianGraphT() const {
  return discretization->getJacobianGraphT();
}

#ifdef ALBANY_AERAS 
Teuchos::RCP<const Tpetra_CrsGraph> Decorator::getImplicitJacobianGraphT() const {
  return discretization->getImplicitJacobianGraphT();
}
#endif

Teuchos::RCP<const Epetra_CrsGraph>
Decorator::getOverlapJacobianGraph() const
{
  return discretization->getOverlapJacobianGraph();
}

Teuchos::RCP<const Tpetra_CrsGraph> Decorator::getOverlapJacobianGraphT() const {
  return discretization->getOverlapJacobianGraphT();
}

#ifdef ALBANY_AERAS 
Teuchos::RCP<const Tpetra_CrsGraph> Decorator::getImplicitOverlapJacobianGraphT() const {
  return discretization->getImplicitOverlapJacobianGraphT();
}
#endif

Teuchos::RCP<const Epetra_Map> Decorator::getNodeMap() const
{
  return discretization->getNodeMap();
}

Teuchos::RCP<const Tpetra_Map> Decorator::getNodeMapT() const {
  return discretization->getNodeMapT();
}

Teuchos::RCP<const Tpetra_Map> Decorator::getNodeMapT(
    const std::string& field_name) const {
  return discretization->getNodeMapT();
}

#if defined(ALBANY_EPETRA)
Teuchos::RCP<const Epetra_Map> Decorator::getOverlapNodeMap() const {
  return discretization->getOverlapNodeMap();
}
#endif

Teuchos::RCP<const Tpetra_Map> Decorator::getOverlapNodeMapT() const {
  return discretization->getOverlapNodeMapT();
}

bool Decorator::isExplicitScheme() const {
  return discretization->isExplicitScheme();
}

Teuchos::RCP<const Tpetra_Map> Decorator::getOverlapNodeMapT(
    const std::string& field_name) const {
  return discretization->getOverlapNodeMapT();
}

const NodeSetList &Decorator::getNodeSets() const
{
  return discretization->getNodeSets();
}

const NodeSetGIDsList& Decorator::getNodeSetGIDs() const {
  return discretization->getNodeSetGIDs();
}

const NodeSetCoordList &Decorator::getNodeSetCoords() const
{
  return discretization->getNodeSetCoords();
}

const SideSetList &Decorator::getSideSets(const int workset) const
{
  return discretization->getSideSets(workset);
}

const Decorator::Conn&
Decorator::getWsElNodeEqID() const
{
  return discretization->getWsElNodeEqID();
}

const WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> > >::type&
Decorator::getWsElNodeID() const {
  return discretization->getWsElNodeID();
}

const std::vector<IDArray>& Decorator::getElNodeEqID(const std::string& field_name) const {
  return discretization->getElNodeEqID(field_name);
}

const NodalDOFManager& Decorator::getDOFManager(
    const std::string& field_name) const {
  return discretization->getDOFManager(field_name);
}

const NodalDOFManager& Decorator::getOverlapDOFManager(
    const std::string& field_name) const {
  return discretization->getOverlapDOFManager(field_name);
}

const WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >::type &Decorator::getCoords() const
{
  return discretization->getCoords();
}

const Teuchos::ArrayRCP<double>& Decorator::getCoordinates() const
{
  return discretization->getCoordinates();
}

void Decorator::setCoordinates(const Teuchos::ArrayRCP<const double>& c) {
  /* TODO: probably should react to this... */
  discretization->setCoordinates(c);
}

void Decorator::setReferenceConfigurationManager(
    const Teuchos::RCP<AAdapt::rc::Manager>& rcm) {
  discretization->setReferenceConfigurationManager(rcm);
}

#ifdef ALBANY_CONTACT
Teuchos::RCP<const Albany::ContactManager> Decorator::getContactManager() const {
  return discretization->getContactManager();
}
#endif

const WorksetArray<Teuchos::ArrayRCP<double> >::type& Decorator::getSphereVolume() const {
  return discretization->getSphereVolume();
}

const WorksetArray<Teuchos::ArrayRCP<double*> >::type& Decorator::getLatticeOrientation() const {
  return discretization->getLatticeOrientation();
}

void Decorator::printCoords() const
{
  discretization->printCoords();
}

const Decorator::SideSetDiscretizationsType& Decorator::getSideSetDiscretizations() const {
  return discretization->getSideSetDiscretizations();
}

const std::map<std::string,std::map<GO,GO>>& Decorator::getSideToSideSetCellMap() const {
  return discretization->getSideToSideSetCellMap();
}

const std::map<std::string,std::map<GO,std::vector<int> > >&
Decorator::getSideNodeNumerationMap() const {
  return discretization->getSideNodeNumerationMap();
}

Teuchos::RCP<AbstractMeshStruct> Decorator::getMeshStruct() const
{
  return discretization->getMeshStruct();
}

StateArrays& Decorator::getStateArrays()
{
  return discretization->getStateArrays();
}

const Albany::StateInfoStruct& Decorator::getNodalParameterSIS() const {
  return discretization->getNodalParameterSIS();
}

const WorksetArray<std::string>::type& Decorator::getWsEBNames() const
{
  return discretization->getWsEBNames();
}

const WorksetArray<int>::type& Decorator::getWsPhysIndex() const
{
  return discretization->getWsPhysIndex();
}

WsLIDList& Decorator::getElemGIDws() {
  return discretization->getElemGIDws();
}

const WsLIDList& Decorator::getElemGIDws() const {
  return discretization->getElemGIDws();
}

#ifdef ALBANY_EPETRA
void Decorator::writeSolution(
    const Epetra_Vector &soln, const double time, const bool overlapped)
{
  Adapter *adapter = Adapter::get();
  if (adapter)
    adapter->update(this->timestep++, time, *this, soln);

  discretization->writeSolution(soln, time, overlapped);
}
#endif

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

Teuchos::RCP<Epetra_Vector> Decorator::getSolutionField(bool overlapped) const
{
  return discretization->getSolutionField(overlapped);
}

Teuchos::RCP<Tpetra_Vector> Decorator::getSolutionFieldT(bool overlapped) const {
  return discretization->getSolutionFieldT(overlapped);
}

Teuchos::RCP<Tpetra_MultiVector> Decorator::getSolutionMV(bool overlapped) const {
  return discretization->getSolutionMV(overlapped);
}

void Decorator::getFieldT(
    Tpetra_Vector &field_vector, const std::string& field_name) const {
  discretization->getFieldT(field_vector, field_name);
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

void Decorator::setFieldT(const Tpetra_Vector &field_vector,
    const std::string& field_name, bool overlapped) {
  /* TODO: should react to this */
  discretization->setFieldT(field_vector, field_name, overlapped);
}

void Decorator::setResidualFieldT(const Tpetra_Vector& residual) {
  /* TODO: should react to this */
  discretization->setResidualFieldT(residual);
}

#if defined(ALBANY_EPETRA)
void Decorator::writeSolution(
    const Epetra_Vector& solution, const Epetra_Vector& solution_dot, 
    const double time, const bool overlapped) {
  /* TODO: should react to this */
  discretization->writeSolution(solution, solution_dot, time, overlapped);
}
#endif

void Decorator::writeSolutionT(
    const Tpetra_Vector &solutionT, const double time, const bool overlapped) {
  /* TODO: should react to this */
  discretization->writeSolutionT(solutionT, time, overlapped);
}

void Decorator::writeSolutionT(
    const Tpetra_Vector &solutionT, const Tpetra_Vector &solution_dotT, 
    const double time, const bool overlapped) {
  /* TODO: should react to this */
  discretization->writeSolutionT(solutionT, solution_dotT, time, overlapped);
}

void Decorator::writeSolutionT(
    const Tpetra_Vector &solutionT, const Tpetra_Vector &solution_dotT, 
    const Tpetra_Vector &solution_dotdotT, 
    const double time, const bool overlapped) {
  /* TODO: should react to this */
  discretization->writeSolutionT(
      solutionT, solution_dotT, solution_dotdotT, time, overlapped);
}

void Decorator::writeSolutionMV(
    const Tpetra_MultiVector &solutionT, const double time,
    const bool overlapped) {
  /* TODO: should react to this */
  discretization->writeSolutionMV(solutionT, time, overlapped);
}

void Decorator::writeSolutionToMeshDatabaseT(
    const Tpetra_Vector &solutionT, const double time, const bool overlapped) {
  /* TODO: should react to this */
  discretization->writeSolutionToMeshDatabaseT(solutionT, time, overlapped);
}

void Decorator::writeSolutionToMeshDatabaseT(
    const Tpetra_Vector &solutionT, const Tpetra_Vector &solution_dotT, 
    const double time, const bool overlapped) {
  /* TODO: should react to this */
  discretization->writeSolutionToMeshDatabaseT(
      solutionT, solution_dotT, time, overlapped);
}

void Decorator::writeSolutionToMeshDatabaseT(
    const Tpetra_Vector &solutionT, const Tpetra_Vector &solution_dotT, 
    const Tpetra_Vector &solution_dotdotT, 
    const double time, const bool overlapped) {
  /* TODO: should react to this */
  discretization->writeSolutionToMeshDatabaseT(
      solutionT, solution_dotT, solution_dotdotT, time, overlapped);
}

void Decorator::writeSolutionMVToMeshDatabase(
    const Tpetra_MultiVector &solutionT, const double time,
    const bool overlapped) {
  /* TODO: should react to this */
  discretization->writeSolutionMVToMeshDatabase(solutionT, time, overlapped);
}

void Decorator::writeSolutionToFileT(const Tpetra_Vector& solutionT, const double time,
    const bool overlapped) {
  /* TODO: should react to this */
  discretization->writeSolutionToFileT(solutionT, time, overlapped);
}

void Decorator::writeSolutionMVToFile(const Tpetra_MultiVector& solutionT,
    const double time, const bool overlapped) {
  /* TODO: should react to this */
  discretization->writeSolutionMVToFile(solutionT, time, overlapped);
}

Teuchos::RCP<LayeredMeshNumbering<LO> > Decorator::getLayeredMeshNumbering() {
  return discretization->getLayeredMeshNumbering();
}

} // end namespace Catalyst
} // end namespace Albany
