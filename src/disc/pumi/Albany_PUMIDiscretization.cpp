//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#include <limits>
#ifdef ALBANY_EPETRA
#include "Epetra_Export.h"
#endif

#include "Albany_Utils.hpp"
#include "Petra_Converters.hpp"
#include "Albany_PUMIDiscretization.hpp"
#include "Albany_PUMIOutput.hpp"
#include <string>
#include <iostream>
#include <fstream>

#include <Shards_BasicTopologies.hpp>
#include "Shards_CellTopology.hpp"
#include "Shards_CellTopologyData.h"

#include <PHAL_Dimension.hpp>

#include <apfMesh.h>
#include <apfShape.h>
#include <PCU.h>

#ifdef ALBANY_EPETRA
// Some integer-type converter helpers for Epetra_Map so that we can compile
// the Epetra_Map file regardless of the value of ALBANY_64BIT_INT.
namespace {
typedef int EpetraInt;
#ifdef ALBANY_64BIT_INT
Teuchos::RCP< Teuchos::Array<int> >
convert (const Teuchos::Array<GO>& indicesAV) {
  Teuchos::RCP< Teuchos::Array<int> > ind = Teuchos::rcp(
    new Teuchos::Array<int>(indicesAV.size()));
  for (std::size_t i = 0; i < indicesAV.size(); ++i)
    (*ind)[i] = Teuchos::as<int>(indicesAV[i]);
  return ind;
};
#else // not ALBANY_64BIT_INT
Teuchos::RCP< Teuchos::Array<GO> >
convert (Teuchos::Array<GO>& indicesAV) {
  return Teuchos::rcp(&indicesAV, false);
}
#endif // not ALBANY_64BIT_INT
} // namespace
#endif // ALBANY_EPETRA

Albany::PUMIDiscretization::PUMIDiscretization(Teuchos::RCP<Albany::PUMIMeshStruct> pumiMeshStruct_,
            const Teuchos::RCP<const Teuchos_Comm>& commT_,
            const Teuchos::RCP<Albany::RigidBodyModes>& rigidBodyModes_) :
  out(Teuchos::VerboseObjectBase::getDefaultOStream()),
  previous_time_label(-1.0e32),
  commT(commT_),
  rigidBodyModes(rigidBodyModes_),
  neq(pumiMeshStruct_->neq),
  pumiMeshStruct(pumiMeshStruct_),
  interleavedOrdering(pumiMeshStruct_->interleavedOrdering),
  outputInterval(0)
{
  meshOutput = PUMIOutput::create(pumiMeshStruct_, commT_);
#ifdef ALBANY_EPETRA
  comm = Albany::createEpetraCommFromTeuchosComm(commT_);
#endif
  globalNumbering = 0;
  elementNumbering = 0;

  // Initialize the mesh and all data structures
  bool shouldTransferIPData = false;
  Albany::PUMIDiscretization::updateMesh(shouldTransferIPData);

  Teuchos::Array<std::string> layout = pumiMeshStruct->solVectorLayout;
  int index;

  for (std::size_t i=0; i < layout.size(); i+=2) {
    solNames.push_back(layout[i]);
    resNames.push_back(layout[i].append("Res"));
    if (layout[i+1] == "S") {
      index = 1;
      solIndex.push_back(index);
    }
    else if (layout[i+1] == "V") {
      index = getNumDim();
      solIndex.push_back(index);
    }
  }

  // zero the residual field for Rhythmos
  if (solNames.size())
    for (size_t i = 0; i < solNames.size(); ++i)
      apf::zeroField(pumiMeshStruct->getMesh()->findField(solNames[i].c_str()));
  else
    apf::zeroField(
      pumiMeshStruct->getMesh()->findField(PUMIMeshStruct::residual_name));
}

Albany::PUMIDiscretization::~PUMIDiscretization()
{
  delete meshOutput;
  apf::destroyGlobalNumbering(globalNumbering);
  apf::destroyGlobalNumbering(elementNumbering);
}

Teuchos::RCP<const Tpetra_Map>
Albany::PUMIDiscretization::getMapT() const
{
  return mapT;
}

Teuchos::RCP<const Tpetra_Map>
Albany::PUMIDiscretization::getOverlapMapT() const
{
  return overlap_mapT;
}

#ifdef ALBANY_EPETRA
Teuchos::RCP<const Epetra_Map>
Albany::PUMIDiscretization::getOverlapNodeMap() const
{
  return Petra::TpetraMap_To_EpetraMap(overlap_node_mapT, comm);
}
#endif

Teuchos::RCP<const Tpetra_CrsGraph>
Albany::PUMIDiscretization::getJacobianGraphT() const
{
  return graphT;
}

Teuchos::RCP<const Tpetra_CrsGraph>
Albany::PUMIDiscretization::getOverlapJacobianGraphT() const
{
  return overlap_graphT;
}

Teuchos::RCP<const Tpetra_Map>
Albany::PUMIDiscretization::getNodeMapT() const
{
  return node_mapT;
}

Teuchos::RCP<const Tpetra_Map>
Albany::PUMIDiscretization::getOverlapNodeMapT() const
{
  return overlap_node_mapT;
}

const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<LO> > > >::type&
Albany::PUMIDiscretization::getWsElNodeEqID() const
{
  return wsElNodeEqID;
}

const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> > >::type&
Albany::PUMIDiscretization::getWsElNodeID() const
{
  return wsElNodeID;
}

const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >::type&
Albany::PUMIDiscretization::getCoords() const
{
  return coords;
}

void
Albany::PUMIDiscretization::printCoords() const
{
  int mesh_dim = pumiMeshStruct->getMesh()->getDimension();

  std::cout << "Processor " << PCU_Comm_Self() << " has " << coords.size()
      << " worksets." << std::endl;

  for (int ws=0; ws<coords.size(); ws++)  //workset
    for (int e=0; e<coords[ws].size(); e++) //cell
      for (int j=0; j<coords[ws][e].size(); j++) //node
        for (int d=0; d<mesh_dim; d++) //dim
          std::cout << "Coord for workset: " << ws << " element: " << e
              << " node: " << j << " DOF: " << d << " is: " <<
              coords[ws][e][j][d] << std::endl;
}

const Teuchos::ArrayRCP<double>&
Albany::PUMIDiscretization::getCoordinates() const
{
  coordinates.resize(3 * numOverlapNodes);
  apf::Field* f = pumiMeshStruct->getMesh()->getCoordinateField();
  for (size_t i=0; i < nodes.getSize(); ++i)
    apf::getComponents(f,nodes[i].entity,nodes[i].node,&(coordinates[3*i]));
  return coordinates;
}

void
Albany::PUMIDiscretization::setCoordinates(
    const Teuchos::ArrayRCP<const double>& c)
{
  apf::Field* f = pumiMeshStruct->getMesh()->getCoordinateField();
  for (size_t i=0; i < nodes.getSize(); ++i)
    apf::setComponents(f,nodes[i].entity,nodes[i].node,&(c[3*i]));
}

void Albany::PUMIDiscretization::
setReferenceConfigurationManager(const Teuchos::RCP<AAdapt::rc::Manager>& ircm)
{ rcm = ircm; }

const Albany::WorksetArray<Teuchos::ArrayRCP<double> >::type&
Albany::PUMIDiscretization::getSphereVolume() const
{
  return sphereVolume;
}

double mean (const double* x, const int n,
             const Teuchos::RCP<const Tpetra_Map>& map) {
  Teuchos::ArrayView<const double> xav = Teuchos::arrayView(x, n);
  Tpetra_Vector xv(map, xav);
  return xv.meanValue();
}

/* DAI: this function also has to change for high-order fields */
void Albany::PUMIDiscretization::setupMLCoords()
{
  if (rigidBodyModes.is_null()) return;
  if (!rigidBodyModes->isMLUsed() && !rigidBodyModes->isMueLuUsed()) return;

  // get mesh dimension and part handle
  const int mesh_dim = getNumDim();
  rigidBodyModes->resize(mesh_dim, numOwnedNodes);
  apf::Mesh* m = pumiMeshStruct->getMesh();
  apf::Field* f = pumiMeshStruct->getMesh()->getCoordinateField();

  double* const coords = rigidBodyModes->getCoordArray();
  for (std::size_t i = 0; i < nodes.getSize(); ++i) {
    apf::Node node = nodes[i];
    if ( ! m->isOwned(node.entity)) continue; // Skip nodes that are not local

    const GO node_gid = apf::getNumber(globalNumbering, node);
    const LO node_lid = node_mapT->getLocalElement(node_gid);
    double lcoords[3];
    apf::getComponents(f, nodes[i].entity, nodes[i].node, lcoords);
    for (std::size_t j = 0; j < mesh_dim; ++j)
      coords[j*numOwnedNodes + node_lid] = lcoords[j];
  }

  if (pumiMeshStruct->useNullspaceTranslationOnly)
    rigidBodyModes->setCoordinates(node_mapT);
  else
    rigidBodyModes->setCoordinatesAndNullspace(node_mapT, mapT);
}

const Albany::WorksetArray<std::string>::type&
Albany::PUMIDiscretization::getWsEBNames() const
{
  return wsEBNames;
}

const Albany::WorksetArray<int>::type&
Albany::PUMIDiscretization::getWsPhysIndex() const
{
  return wsPhysIndex;
}

void Albany::PUMIDiscretization::setField(
  const char* name, const ST* data, bool overlapped, int offset, int nentries)
{
  apf::Mesh* m = pumiMeshStruct->getMesh();
  apf::Field* f = m->findField(name);
  for (size_t i=0; i < nodes.getSize(); ++i) {
    apf::Node node = nodes[i];
    GO node_gid = apf::getNumber(globalNumbering,node);
    int node_lid;
    if (overlapped)
      node_lid = overlap_node_mapT->getLocalElement(node_gid);
    else {
      if ( ! m->isOwned(node.entity)) continue;
      node_lid = node_mapT->getLocalElement(node_gid);
    }
    int firstDOF = getDOF(node_lid,offset,nentries);
    apf::setComponents(f,node.entity,node.node,&(data[firstDOF]));
  }
  if ( ! overlapped)
    apf::synchronize(f);
}

void Albany::PUMIDiscretization::setSplitFields(
  const std::vector<std::string>& names, const std::vector<int>& indices,
  const ST* data, bool overlapped)
{
  apf::Mesh* m = pumiMeshStruct->getMesh();
  int offset = 0;
  int indexSum = 0;
  for (std::size_t i=0; i < names.size(); ++i) {
    assert(indexSum==offset);
    this->setField(names[i].c_str(),data,overlapped,offset);
    offset += apf::countComponents(m->findField(names[i].c_str()));
    indexSum += indices[i];
  }
}

void Albany::PUMIDiscretization::getField(
  const char* name, ST* data, bool overlapped, int offset, int nentries) const
{
  apf::Mesh* m = pumiMeshStruct->getMesh();
  apf::Field* f = m->findField(name);
  for (size_t i=0; i < nodes.getSize(); ++i) {
    apf::Node node = nodes[i];
    GO node_gid = apf::getNumber(globalNumbering,node);
    int node_lid;
    if (overlapped)
      node_lid = overlap_node_mapT->getLocalElement(node_gid);
    else {
      if ( ! m->isOwned(node.entity))
        continue;
      node_lid = node_mapT->getLocalElement(node_gid);
    }
    int firstDOF = getDOF(node_lid,offset,nentries);
    apf::getComponents(f,node.entity,node.node,&(data[firstDOF]));
  }
}

void Albany::PUMIDiscretization::getSplitFields(
  const std::vector<std::string>& names, const std::vector<int>& indices, ST* data,
  bool overlapped) const
{
  apf::Mesh* m = pumiMeshStruct->getMesh();
  int offset = 0;
  int indexSum = 0;
  for (std::size_t i=0; i < names.size(); ++i) {
    assert(indexSum==offset);

    this->getField(names[i].c_str(),data,overlapped,offset);
    offset += apf::countComponents(m->findField(names[i].c_str()));
    indexSum += indices[i];
  }
}

void Albany::PUMIDiscretization::
createField(const char* name, int value_type)
{
  apf::createFieldOn(pumiMeshStruct->getMesh(), name, value_type);
  apf::zeroField(pumiMeshStruct->getMesh()->findField(name));
}

void Albany::PUMIDiscretization::reNameExodusOutput(
    const std::string& str)
{
  meshOutput->setFileName(str);
}

void Albany::PUMIDiscretization::writeSolutionT(
  const Tpetra_Vector& solnT, const double time_value, const bool overlapped)
{
  Teuchos::ArrayRCP<const ST> data = solnT.get1dView();
  writeAnySolutionToMeshDatabase(&(data[0]),time_value,overlapped);
  writeAnySolutionToFile(&(data[0]),time_value,overlapped);
}

void Albany::PUMIDiscretization::writeSolutionToMeshDatabaseT(
  const Tpetra_Vector& solnT, const double time_value, const bool overlapped)
{
  Teuchos::ArrayRCP<const ST> data = solnT.get1dView();
  writeAnySolutionToMeshDatabase(&(data[0]),time_value,overlapped);
}

void Albany::PUMIDiscretization::writeSolutionToFileT(
  const Tpetra_Vector& solnT, const double time_value, const bool overlapped)
{
  Teuchos::ArrayRCP<const ST> data = solnT.get1dView();
  writeAnySolutionToFile(&(data[0]),time_value,overlapped);
}

#ifdef ALBANY_EPETRA
void Albany::PUMIDiscretization::writeSolution(const Epetra_Vector& soln, const double time_value,
      const bool overlapped)
{
  writeAnySolutionToMeshDatabase(&(soln[0]),time_value,overlapped);
  writeAnySolutionToFile(&(soln[0]),time_value,overlapped);
}
#endif

void Albany::PUMIDiscretization::writeAnySolutionToMeshDatabase(
      const ST* soln, const double time_value,
      const bool overlapped)
{
  if (solNames.size() == 0)
    this->setField(PUMIMeshStruct::solution_name,soln,overlapped);
  else
    this->setSplitFields(solNames,solIndex,soln,overlapped);

  pumiMeshStruct->solutionInitialized = true;
}

void Albany::PUMIDiscretization::writeAnySolutionToFile(
      const ST* soln, const double time_value,
      const bool overlapped)
{
  // Skip this write unless the proper interval has been reached.
  if (outputInterval++ % pumiMeshStruct->outputInterval) return;

  if (pumiMeshStruct->outputFileName.empty()) return;

  double time_label = monotonicTimeLabel(time_value);
  int out_step = 0;

  if (mapT->getComm()->getRank()==0) {
    *out << "Albany::PUMIDiscretization::writeSolution: writing time " << time_value;
    if (time_label != time_value) *out << " with label " << time_label;
    *out << " to index " << out_step << " in file "
         << pumiMeshStruct->outputFileName << std::endl;
  }

  apf::Field* f;
  int dim = getNumDim();
  apf::FieldShape* fs = apf::getIPShape(dim, pumiMeshStruct->cubatureDegree);
  copyQPStatesToAPF(f,fs,false);

  meshOutput->writeFile(time_label);

  removeQPStatesFromAPF();
}

void
Albany::PUMIDiscretization::writeMeshDebug (const std::string& filename) {
  apf::Field* f;
  apf::FieldShape* fs = apf::getIPShape(getNumDim(),
                                        pumiMeshStruct->cubatureDegree);
  copyQPStatesToAPF(f, fs, true);
  apf::writeVtkFiles(filename.c_str(), getPUMIMeshStruct()->getMesh());
  removeQPStatesFromAPF();
}

double
Albany::PUMIDiscretization::monotonicTimeLabel(const double time)
{
  // If increasing, then all is good
  if (time > previous_time_label) {
    previous_time_label = time;
    return time;
  }
// Try absolute value
  double time_label = fabs(time);
  if (time_label > previous_time_label) {
    previous_time_label = time_label;
    return time_label;
  }

  // Try adding 1.0 to time
  if (time_label+1.0 > previous_time_label) {
    previous_time_label = time_label+1.0;
    return time_label+1.0;
  }

  // Otherwise, just add 1.0 to previous
  previous_time_label += 1.0;
  return previous_time_label;
}

void
Albany::PUMIDiscretization::setResidualFieldT(const Tpetra_Vector& residualT)
{
  Teuchos::ArrayRCP<const ST> data = residualT.get1dView();
  if (solNames.size() == 0)
    this->setField(PUMIMeshStruct::residual_name,&(data[0]),/*overlapped=*/false);
  else
    this->setSplitFields(resNames,solIndex,&(data[0]),/*overlapped=*/false);

  pumiMeshStruct->residualInitialized = true;
}

#ifdef ALBANY_EPETRA
void
Albany::PUMIDiscretization::setResidualField(const Epetra_Vector& residual)
{
  if (solNames.size() == 0)
    this->setField(PUMIMeshStruct::residual_name,&(residual[0]),/*overlapped=*/false);
  else
    this->setSplitFields(resNames,solIndex,&(residual[0]),/*overlapped=*/false);

  pumiMeshStruct->residualInitialized = true;
}
#endif

Teuchos::RCP<Tpetra_Vector>
Albany::PUMIDiscretization::getSolutionFieldT(bool overlapped) const
{
  // Copy soln vector into solution field, one node at a time
  Teuchos::RCP<Tpetra_Vector> solnT = Teuchos::rcp(
    new Tpetra_Vector(overlapped ? overlap_mapT : mapT));
  {
    Teuchos::ArrayRCP<ST> data = solnT->get1dViewNonConst();

    if (pumiMeshStruct->solutionInitialized) {
      if (solNames.size() == 0)
        this->getField(PUMIMeshStruct::solution_name,&(data[0]),overlapped);
      else
        this->getSplitFields(solNames,solIndex,&(data[0]),overlapped);
    }
    else if ( ! PCU_Comm_Self())
      *out <<__func__<<": uninit field" << std::endl;
  }
  return solnT;
}

#ifdef ALBANY_EPETRA
Teuchos::RCP<Epetra_Vector>
Albany::PUMIDiscretization::getSolutionField(bool overlapped) const
{
  // Copy soln vector into solution field, one node at a time
  Teuchos::RCP<Epetra_Vector> soln = Teuchos::rcp(
    new Epetra_Vector(overlapped ? *overlap_map : *map));

  if (pumiMeshStruct->solutionInitialized) {
    if (solNames.size() == 0)
      this->getField(PUMIMeshStruct::solution_name,&((*soln)[0]),overlapped);
    else
      this->getSplitFields(solNames,solIndex,&((*soln)[0]),overlapped);
  }
  else if ( ! PCU_Comm_Self())
    *out <<__func__<<": uninit field" << std::endl;

  return soln;
}
#endif

int Albany::PUMIDiscretization::nonzeroesPerRow(const int neq) const
{
  int numDim = getNumDim();

  /* DAI: this function should be revisited for overall correctness,
     especially in the case of higher-order fields */
  int estNonzeroesPerRow;
  switch (numDim) {
  case 0: estNonzeroesPerRow=1*neq; break;
  case 1: estNonzeroesPerRow=3*neq; break;
  case 2: estNonzeroesPerRow=9*neq; break;
  case 3: estNonzeroesPerRow=27*neq; break;
  default: TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
			      "PUMIDiscretization:  Bad numDim"<< numDim);
  }
  return estNonzeroesPerRow;
}

void Albany::PUMIDiscretization::computeOwnedNodesAndUnknowns()
{
  apf::Mesh* m = pumiMeshStruct->getMesh();
  if (globalNumbering) apf::destroyGlobalNumbering(globalNumbering);
  globalNumbering = apf::makeGlobal(apf::numberOwnedNodes(m,"owned"));
  apf::DynamicArray<apf::Node> ownedNodes;
  apf::getNodes(globalNumbering,ownedNodes);
  numOwnedNodes = ownedNodes.getSize();
  apf::synchronize(globalNumbering);
  Teuchos::Array<GO> indices(numOwnedNodes);
  for (int i=0; i < numOwnedNodes; ++i)
    indices[i] = apf::getNumber(globalNumbering,ownedNodes[i]);
  node_mapT = Tpetra::createNonContigMap<LO, GO>(indices, commT);
  numGlobalNodes = node_mapT->getMaxAllGlobalIndex() + 1;
  if(Teuchos::nonnull(pumiMeshStruct->nodal_data_base))
    pumiMeshStruct->nodal_data_base->resizeLocalMap(indices, commT);
  indices.resize(numOwnedNodes*neq);
  for (int i=0; i < numOwnedNodes; ++i)
    for (int j=0; j < neq; ++j) {
      GO gid = apf::getNumber(globalNumbering,ownedNodes[i]);
      indices[getDOF(i,j)] = getDOF(gid,j);
    }
  mapT = Tpetra::createNonContigMap<LO, GO>(indices, commT);
#ifdef ALBANY_EPETRA
  map = Teuchos::rcp(
    new Epetra_Map(-1, indices.size(), convert(indices)->getRawPtr(), 0,
                   *comm));
#endif
}

void Albany::PUMIDiscretization::computeOverlapNodesAndUnknowns()
{
  apf::Mesh* m = pumiMeshStruct->getMesh();
  apf::Numbering* overlap = m->findNumbering("overlap");
  if (overlap) apf::destroyNumbering(overlap);
  overlap = apf::numberOverlapNodes(m,"overlap");
  apf::getNodes(overlap,nodes);
  numOverlapNodes = nodes.getSize();
  Teuchos::Array<GO> nodeIndices(numOverlapNodes);
  Teuchos::Array<GO> dofIndices(numOverlapNodes*neq);
  for (int i=0; i < numOverlapNodes; ++i) {
    GO global = apf::getNumber(globalNumbering,nodes[i]);
    nodeIndices[i] = global;
    for (int j=0; j < neq; ++j)
      dofIndices[getDOF(i,j)] = getDOF(global,j);
  }
  overlap_node_mapT = Tpetra::createNonContigMap<LO, GO>(nodeIndices, commT);
  overlap_mapT = Tpetra::createNonContigMap<LO, GO>(dofIndices, commT);
#ifdef ALBANY_EPETRA
  overlap_map = Teuchos::rcp(
    new Epetra_Map(-1, dofIndices.size(), convert(dofIndices)->getRawPtr(), 0,
                   *comm));
#endif
  if(Teuchos::nonnull(pumiMeshStruct->nodal_data_base))
    pumiMeshStruct->nodal_data_base->resizeOverlapMap(nodeIndices, commT);
}

void Albany::PUMIDiscretization::computeGraphs()
{

  apf::Mesh* m = pumiMeshStruct->getMesh();
  int numDim = m->getDimension();
  std::vector<apf::MeshEntity*> cells;
  std::vector<int> n_nodes_in_elem;
  cells.reserve(m->count(numDim));
  apf::MeshIterator* it = m->begin(numDim);
  apf::MeshEntity* e;
  GO node_sum = 0;
  while ((e = m->iterate(it))){
    cells.push_back(e);
    int nnodes = apf::countElementNodes(m->getShape(),m->getType(e));
    n_nodes_in_elem.push_back(nnodes);
    node_sum += nnodes;
  }
  m->end(it);
  int nodes_per_element = std::ceil((double)node_sum / (double)cells.size());
  /* construct the overlap graph of all local DOFs as they
     are coupled by element-node connectivity */
  overlap_graphT = Teuchos::rcp(new Tpetra_CrsGraph(
                 overlap_mapT, neq*nodes_per_element));
#ifdef ALBANY_EPETRA
  overlap_graph =
    Teuchos::rcp(new Epetra_CrsGraph(Copy, *overlap_map,
                                     neq*nodes_per_element, false));
#endif
  for (size_t i=0; i < cells.size(); ++i) {
    apf::NewArray<long> cellNodes;
    apf::getElementNumbers(globalNumbering,cells[i],cellNodes);
    for (int j=0; j < n_nodes_in_elem[i]; ++j) {
      for (int k=0; k < neq; ++k) {
        GO row = getDOF(cellNodes[j],k);
        for (int l=0; l < n_nodes_in_elem[i]; ++l) {
          for (int m=0; m < neq; ++m) {
            GO col = getDOF(cellNodes[l],m);
            Teuchos::ArrayView<GO> colAV = Teuchos::arrayView(&col, 1);
            overlap_graphT->insertGlobalIndices(row, colAV);
#ifdef ALBANY_EPETRA
            EpetraInt ecol = Teuchos::as<EpetraInt>(col);
            overlap_graph->InsertGlobalIndices(row,1,&ecol);
#endif
          }
        }
      }
    }
  }
  overlap_graphT->fillComplete();
#ifdef ALBANY_EPETRA
  overlap_graph->FillComplete();
#endif

  // Create Owned graph by exporting overlap with known row map
  graphT = Teuchos::rcp(new Tpetra_CrsGraph(mapT, nonzeroesPerRow(neq)));
#ifdef ALBANY_EPETRA
  graph = Teuchos::rcp(new Epetra_CrsGraph(Copy, *map, nonzeroesPerRow(neq), false));
#endif

  // Create non-overlapped matrix using two maps and export object
  Teuchos::RCP<Tpetra_Export> exporterT = Teuchos::rcp(new Tpetra_Export(
                                                       overlap_mapT, mapT));
  graphT->doExport(*overlap_graphT, *exporterT, Tpetra::INSERT);
  graphT->fillComplete();

#ifdef ALBANY_EPETRA
  Epetra_Export exporter(*overlap_map, *map);
  graph->Export(*overlap_graph, exporter, Insert);
  graph->FillComplete();
#endif
}

static apf::StkModel* findElementBlock(
    apf::Mesh* m,
    apf::StkModels& sets,
    apf::ModelEntity* me)
{
  int tag = m->getModelTag(me);
  int d = m->getDimension();
  for (size_t i = 0; i < sets[d].getSize(); ++i)
    if (sets[d][i].apfTag == tag)
      return &sets[d][i];
  return 0;
}

void Albany::PUMIDiscretization::computeWorksetInfo()
{
  apf::Mesh* m = pumiMeshStruct->getMesh();
  int numDim = m->getDimension();
  if (elementNumbering) apf::destroyGlobalNumbering(elementNumbering);
  elementNumbering = apf::makeGlobal(apf::numberElements(m,"element"));

/*
 * Note: Max workset size is given in input file, or set to a default in Albany_PUMIMeshStruct.cpp
 * The workset size is set in Albany_PUMIMeshStruct.cpp to be the maximum number in an element block if
 * the element block size < Max workset size.
 * STK bucket size is set to the workset size. We will "chunk" the elements into worksets here.
 */

  // This function is called each adaptive cycle. Need to reset the 2D array "buckets" back to the initial size.
  for(int i = 0; i < buckets.size(); i++)
    buckets[i].clear();

  buckets.clear();

  std::map<apf::ModelEntity*, int> bucketMap;
  std::map<apf::ModelEntity*, int>::iterator buck_it;
  apf::StkModels& sets = pumiMeshStruct->getSets();
  int bucket_counter = 0;

  int worksetSize = pumiMeshStruct->worksetSize;

  // iterate over all elements
  apf::MeshIterator* it = m->begin(numDim);
  apf::MeshEntity* element;
  while ((element = m->iterate(it)))
  {
    apf::ModelEntity* block = m->toModel(element);
    // find which bucket holds the elements for the element block
    buck_it = bucketMap.find(block);
    if((buck_it == bucketMap.end()) ||  // Make a new bucket to hold the new element block's elements
       (buckets[buck_it->second].size() >= worksetSize)){ // old bucket is full, put the element in a new one
      // Associate this elem_blk with a new bucket
      bucketMap[block] = bucket_counter;
      // resize the bucket array larger by one
      buckets.resize(bucket_counter + 1);
      wsEBNames.resize(bucket_counter + 1);
      // save the element in the bucket
      buckets[bucket_counter].push_back(element);
      // save the name of the new element block
      apf::StkModel* set = findElementBlock(m, sets, block);
      TEUCHOS_TEST_FOR_EXCEPTION(!set, std::logic_error,
			   "Error: findElementBlock() failed on line " << __LINE__ << " of file " << __FILE__ << std::endl);
      std::string EB_name = set->stkName;
      wsEBNames[bucket_counter] = EB_name;
      bucket_counter++;
    }
    else { // put the element in the proper bucket
      buckets[buck_it->second].push_back(element);
    }
  }
  m->end(it);

  int numBuckets = bucket_counter;

  wsPhysIndex.resize(numBuckets);

  if (pumiMeshStruct->allElementBlocksHaveSamePhysics)
    for (int i=0; i<numBuckets; i++) wsPhysIndex[i]=0;
  else
    for (int i=0; i<numBuckets; i++) wsPhysIndex[i]=pumiMeshStruct->ebNameToIndex[wsEBNames[i]];

  // Fill  wsElNodeEqID(workset, el_LID, local node, Eq) => unk_LID

  wsElNodeEqID.resize(numBuckets);
  wsElNodeID.resize(numBuckets);
  coords.resize(numBuckets);
  sphereVolume.resize(numBuckets);

  // Clear map if remeshing
  if(!elemGIDws.empty()) elemGIDws.clear();

  /* this block of code creates the wsElNodeEqID,
     wsElNodeID, and coords structures.
     These are (bucket, element, element_node, dof)-indexed
     structures to get numbers or coordinates */
  for (int b=0; b < numBuckets; b++) {

    std::vector<apf::MeshEntity*>& buck = buckets[b];
    wsElNodeEqID[b].resize(buck.size());
    wsElNodeID[b].resize(buck.size());
    coords[b].resize(buck.size());

    // i is the element index within bucket b

    for (std::size_t i=0; i < buck.size(); i++) {

      // Traverse all the elements in this bucket
      element = buck[i];
      apf::Node node(element,0);

      // Now, save a map from element GID to workset on this PE
      elemGIDws[apf::getNumber(elementNumbering,node)].ws = b;

      // Now, save a map element GID to local id on this workset on this PE
      elemGIDws[apf::getNumber(elementNumbering,node)].LID = i;

      // get global node numbers
      apf::NewArray<long> nodeIDs;
      apf::getElementNumbers(globalNumbering,element,nodeIDs);

      int nodes_per_element = apf::countElementNodes(
          m->getShape(),m->getType(element));
      wsElNodeEqID[b][i].resize(nodes_per_element);
      wsElNodeID[b][i].resize(nodes_per_element);
      coords[b][i].resize(nodes_per_element);

      // loop over local nodes

      for (int j=0; j < nodes_per_element; j++) {
        const GO node_gid = nodeIDs[j];
        const LO node_lid = overlap_node_mapT->getLocalElement(node_gid);

        TEUCHOS_TEST_FOR_EXCEPTION(node_lid<0, std::logic_error,
			   "PUMI_Disc: node_lid out of range " << node_lid << std::endl);

        coords[b][i][j] = &coordinates[node_lid * 3];
        wsElNodeEqID[b][i][j].resize(neq);
        wsElNodeID[b][i][j] = node_gid;

        for (std::size_t eq=0; eq < neq; eq++)
          wsElNodeEqID[b][i][j][eq] = getDOF(node_lid,eq);
      }
    }
  }

  // (Re-)allocate storage for element data
  //
  // For each state, create storage for the data for on processor elements
  // elemGIDws.size() is the number of elements on this processor ...
  // Note however that Intrepid will stride over numBuckets * worksetSize
  // so we must allocate enough storage for that

  std::size_t numElementsAccessed = numBuckets * worksetSize;

  for (std::size_t i=0; i<pumiMeshStruct->qpscalar_states.size(); i++)
      pumiMeshStruct->qpscalar_states[i]->reAllocateBuffer(numElementsAccessed);
  for (std::size_t i=0; i<pumiMeshStruct->qpvector_states.size(); i++)
      pumiMeshStruct->qpvector_states[i]->reAllocateBuffer(numElementsAccessed);
  for (std::size_t i=0; i<pumiMeshStruct->qptensor_states.size(); i++)
      pumiMeshStruct->qptensor_states[i]->reAllocateBuffer(numElementsAccessed);
  for (std::size_t i=0; i<pumiMeshStruct->scalarValue_states.size(); i++)
      // special case : need to store one double value that represents all the elements in the workset (time)
      // numBuckets are the number of worksets
      pumiMeshStruct->scalarValue_states[i]->reAllocateBuffer(numBuckets);

  // Pull out pointers to shards::Arrays for every bucket, for every state

  // Note that numBuckets is typically larger each time the mesh is adapted

  stateArrays.elemStateArrays.resize(numBuckets);

  for (std::size_t b=0; b < buckets.size(); b++) {
    std::vector<apf::MeshEntity*>& buck = buckets[b];
    for (std::size_t i=0; i<pumiMeshStruct->qpscalar_states.size(); i++)
      stateArrays.elemStateArrays[b][pumiMeshStruct->qpscalar_states[i]->name] =
                 pumiMeshStruct->qpscalar_states[i]->getMDA(buck.size());
    for (std::size_t i=0; i<pumiMeshStruct->qpvector_states.size(); i++)
      stateArrays.elemStateArrays[b][pumiMeshStruct->qpvector_states[i]->name] =
                 pumiMeshStruct->qpvector_states[i]->getMDA(buck.size());
    for (std::size_t i=0; i<pumiMeshStruct->qptensor_states.size(); i++)
      stateArrays.elemStateArrays[b][pumiMeshStruct->qptensor_states[i]->name] =
                 pumiMeshStruct->qptensor_states[i]->getMDA(buck.size());
    for (std::size_t i=0; i<pumiMeshStruct->scalarValue_states.size(); i++)
      stateArrays.elemStateArrays[b][pumiMeshStruct->scalarValue_states[i]->name] =
                 pumiMeshStruct->scalarValue_states[i]->getMDA(1);
  }

// Process node data sets if present

  if(Teuchos::nonnull(pumiMeshStruct->nodal_data_base) &&
    pumiMeshStruct->nodal_data_base->isNodeDataPresent()) {

    std::vector< std::vector<apf::Node> > nbuckets; // bucket of nodes
    int numNodeBuckets =  (int)ceil((double)numOwnedNodes / (double)worksetSize);

    nbuckets.resize(numNodeBuckets);
    int node_bucket_counter = 0;
    int node_in_bucket = 0;

    // iterate over all nodes and save the owned ones into buckets
    for (size_t i=0; i < nodes.getSize(); ++i)
      if (m->isOwned(nodes[i].entity))
      {
        nbuckets[node_bucket_counter].push_back(nodes[i]);
        node_in_bucket++;
        if (node_in_bucket >= worksetSize) {
          ++node_bucket_counter;
          node_in_bucket = 0;
        }
      }

    Teuchos::RCP<Albany::NodeFieldContainer> node_states = pumiMeshStruct->nodal_data_base->getNodeContainer();

    stateArrays.nodeStateArrays.resize(numNodeBuckets);

    // Loop over all the node field containers
    for (Albany::NodeFieldContainer::iterator nfs = node_states->begin();
                nfs != node_states->end(); ++nfs){
      Teuchos::RCP<Albany::AbstractPUMINodeFieldContainer> nodeContainer =
             Teuchos::rcp_dynamic_cast<Albany::AbstractPUMINodeFieldContainer>((*nfs).second);

      // resize the container to hold all the owned node's data
      nodeContainer->resize(node_mapT);

      // Now, loop over each workset to get a reference to each workset collection of nodes
      for (std::size_t b=0; b < nbuckets.size(); b++) {
         std::vector<apf::Node>& buck = nbuckets[b];
         stateArrays.nodeStateArrays[b][(*nfs).first] = nodeContainer->getMDA(buck);
      }
    }
  }
}

void Albany::PUMIDiscretization::copyQPTensorToAPF(
    unsigned nqp,
    PUMIQPData<double, 4>& state,
    apf::Field* f)
{
  for (std::size_t b=0; b < buckets.size(); ++b) {
    std::vector<apf::MeshEntity*>& buck = buckets[b];
    Albany::MDArray& ar = stateArrays.elemStateArrays[b][state.name];
    for (std::size_t e=0; e < buck.size(); ++e) {
      apf::Matrix3x3 v;
      for (std::size_t p=0; p < nqp; ++p) {
        for (std::size_t i=0; i < 3; ++i)
        for (std::size_t j=0; j < 3; ++j)
          v[i][j] = ar(e,p,i,j);
        apf::setMatrix(f,buck[e],p,v);
      }
    }
  }
}

void Albany::PUMIDiscretization::copyQPScalarToAPF(
    unsigned nqp,
    PUMIQPData<double, 2>& state,
    apf::Field* f)
{
  for (std::size_t b=0; b < buckets.size(); ++b) {
    std::vector<apf::MeshEntity*>& buck = buckets[b];
    Albany::MDArray& ar = stateArrays.elemStateArrays[b][state.name];
    for (std::size_t e=0; e < buck.size(); ++e)
      for (std::size_t p=0; p < nqp; ++p)
        apf::setScalar(f,buck[e],p,ar(e,p));
  }
}

void Albany::PUMIDiscretization::copyQPVectorToAPF(
    unsigned nqp,
    PUMIQPData<double, 3>& state,
    apf::Field* f)
{
  for (std::size_t b=0; b < buckets.size(); ++b) {
    std::vector<apf::MeshEntity*>& buck = buckets[b];
    Albany::MDArray& ar = stateArrays.elemStateArrays[b][state.name];
    for (std::size_t e=0; e < buck.size(); ++e) {
      apf::Vector3 v;
      for (std::size_t p=0; p < nqp; ++p) {
        for (std::size_t i=0; i < 3; ++i)
          v[i] = ar(e,p,i);
        apf::setVector(f,buck[e],p,v);
      }
    }
  }
}

void Albany::PUMIDiscretization::copyQPStatesToAPF(
    apf::Field* f,
    apf::FieldShape* fs,
    bool copyAll)
{
  apf::Mesh2* m = pumiMeshStruct->getMesh();
  for (std::size_t i=0; i < pumiMeshStruct->qpscalar_states.size(); ++i) {
    PUMIQPData<double, 2>& state = *(pumiMeshStruct->qpscalar_states[i]);
    if (!copyAll && !state.output)
      continue;
    int nqp = state.dims[1];
    f = apf::createField(m,state.name.c_str(),apf::SCALAR,fs);
    copyQPScalarToAPF(nqp,state,f);
  }
  for (std::size_t i=0; i < pumiMeshStruct->qpvector_states.size(); ++i) {
    PUMIQPData<double, 3>& state = *(pumiMeshStruct->qpvector_states[i]);
    if (!copyAll && !state.output)
      continue;
    int nqp = state.dims[1];
    f = apf::createField(m,state.name.c_str(),apf::VECTOR,fs);
    copyQPVectorToAPF(nqp,state,f);
  }
  for (std::size_t i=0; i < pumiMeshStruct->qptensor_states.size(); ++i) {
    PUMIQPData<double, 4>& state = *(pumiMeshStruct->qptensor_states[i]);
    if (!copyAll && !state.output)
      continue;
    int nqp = state.dims[1];
    f = apf::createField(m,state.name.c_str(),apf::MATRIX,fs);
    copyQPTensorToAPF(nqp,state,f);
  }
}

void Albany::PUMIDiscretization::removeQPStatesFromAPF()
{
  apf::Mesh2* m = pumiMeshStruct->getMesh();
  for (std::size_t i=0; i < pumiMeshStruct->qpscalar_states.size(); ++i) {
    PUMIQPData<double, 2>& state = *(pumiMeshStruct->qpscalar_states[i]);
    apf::destroyField(m->findField(state.name.c_str()));
  }
  for (std::size_t i=0; i < pumiMeshStruct->qpvector_states.size(); ++i) {
    PUMIQPData<double, 3>& state = *(pumiMeshStruct->qpvector_states[i]);
    apf::destroyField(m->findField(state.name.c_str()));
  }
  for (std::size_t i=0; i < pumiMeshStruct->qptensor_states.size(); ++i) {
    PUMIQPData<double, 4>& state = *(pumiMeshStruct->qptensor_states[i]);
    apf::destroyField(m->findField(state.name.c_str()));
  }
}

void Albany::PUMIDiscretization::copyQPScalarFromAPF(
    unsigned nqp,
    PUMIQPData<double, 2>& state,
    apf::Field* f)
{
  apf::Mesh2* m = pumiMeshStruct->getMesh();
  for (std::size_t b=0; b < buckets.size(); ++b) {
    std::vector<apf::MeshEntity*>& buck = buckets[b];
    Albany::MDArray& ar = stateArrays.elemStateArrays[b][state.name];
    for (std::size_t e=0; e < buck.size(); ++e) {
      for (std::size_t p = 0; p < nqp; ++p)
        ar(e,p) = apf::getScalar(f,buck[e],p);
    }
  }
}

void Albany::PUMIDiscretization::copyQPVectorFromAPF(
    unsigned nqp,
    PUMIQPData<double, 3>& state,
    apf::Field* f)
{
  apf::Mesh2* m = pumiMeshStruct->getMesh();
  for (std::size_t b=0; b < buckets.size(); ++b) {
    std::vector<apf::MeshEntity*>& buck = buckets[b];
    Albany::MDArray& ar = stateArrays.elemStateArrays[b][state.name];
    for (std::size_t e=0; e < buck.size(); ++e) {
      apf::Vector3 v;
      for (std::size_t p=0; p < nqp; ++p) {
        apf::getVector(f,buck[e],p,v);
        for (std::size_t i=0; i < 3; ++i)
          ar(e,p,i) = v[i];
      }
    }
  }
}

void Albany::PUMIDiscretization::copyQPTensorFromAPF(
    unsigned nqp,
    PUMIQPData<double, 4>& state,
    apf::Field* f)
{
  apf::Mesh2* m = pumiMeshStruct->getMesh();
  for (std::size_t b = 0; b < buckets.size(); ++b) {
    std::vector<apf::MeshEntity*>& buck = buckets[b];
    Albany::MDArray& ar = stateArrays.elemStateArrays[b][state.name];
    for (std::size_t e=0; e < buck.size(); ++e) {
      apf::Matrix3x3 v;
      for (std::size_t p=0; p < nqp; ++p) {
        apf::getMatrix(f,buck[e],p,v);
        for (std::size_t i=0; i < 3; ++i) {
          for (std::size_t j=0; j < 3; ++j)
            ar(e,p,i,j) = v[i][j];
        }
      }
    }
  }
}

void Albany::PUMIDiscretization::copyQPStatesFromAPF()
{
  apf::Mesh2* m = pumiMeshStruct->getMesh();
  apf::Field* f;
  for (std::size_t i=0; i < pumiMeshStruct->qpscalar_states.size(); ++i) {
    PUMIQPData<double, 2>& state = *(pumiMeshStruct->qpscalar_states[i]);
    int nqp = state.dims[1];
    f = m->findField(state.name.c_str());
    copyQPScalarFromAPF(nqp,state,f);
  }
  for (std::size_t i=0; i < pumiMeshStruct->qpvector_states.size(); ++i) {
    PUMIQPData<double, 3>& state = *(pumiMeshStruct->qpvector_states[i]);
    int nqp = state.dims[1];
    f = m->findField(state.name.c_str());
    copyQPVectorFromAPF(nqp,state,f);
  }
  for (std::size_t i=0; i < pumiMeshStruct->qptensor_states.size(); ++i) {
    PUMIQPData<double, 4>& state = *(pumiMeshStruct->qptensor_states[i]);
    int nqp = state.dims[1];
    f = m->findField(state.name.c_str());
    copyQPTensorFromAPF(nqp,state,f);
  }
}

void Albany::PUMIDiscretization::computeSideSets()
{
  apf::Mesh* m = pumiMeshStruct->getMesh();
  apf::StkModels& sets = pumiMeshStruct->getSets();

  // need a sideset list per workset
  int num_buckets = wsEBNames.size();
  sideSets.resize(num_buckets);

  // loop over side sets
  int d = m->getDimension();
  for (size_t i = 0; i < sets[d - 1].getSize(); ++i) {
    apf::StkModel& ss = sets[d - 1][i];

    // get the name of this side set
    std::string const& ss_name = ss.stkName;

    apf::ModelEntity* me = m->findModelEntity(d - 1, ss.apfTag);
    apf::MeshIterator* it = m->begin(d - 1);
    apf::MeshEntity* side;
    // loop over the sides in this side set
    while ((side = m->iterate(it))) {
      if (m->toModel(side) != me)
        continue;

      // get the elements adjacent to this side
      apf::Up side_elems;
      m->getUp(side, side_elems);

      // we are not yet considering non-manifold side sets !
      TEUCHOS_TEST_FOR_EXCEPTION(side_elems.n != 1, std::logic_error,
		   "PUMIDisc: cannot figure out side set topology for side set "<<ss_name<<std::endl);

      apf::MeshEntity* elem = side_elems.e[0];

      // fill in the data holder for a side struct

      Albany::SideStruct sstruct;

      sstruct.elem_GID = apf::getNumber(elementNumbering, apf::Node(elem, 0));
      int workset = elemGIDws[sstruct.elem_GID].ws; // workset ID that this element lives in
      sstruct.elem_LID = elemGIDws[sstruct.elem_GID].LID; // local element id in this workset
      sstruct.elem_ebIndex = pumiMeshStruct->ebNameToIndex[wsEBNames[workset]]; // element block that workset lives in

      sstruct.side_local_id = apf::getLocalSideId(m, elem, side);

      Albany::SideSetList& ssList = sideSets[workset]; // Get a ref to the side set map for this ws

      // Get an iterator to the correct sideset (if it exists)
      Albany::SideSetList::iterator it = ssList.find(ss_name);

      if(it != ssList.end()) // The sideset has already been created
        it->second.push_back(sstruct); // Save this side to the vector that belongs to the name ss->first
      else { // Add the key ss_name to the map, and the side vector to that map
        std::vector<Albany::SideStruct> tmpSSVec;
        tmpSSVec.push_back(sstruct);
        ssList.insert(Albany::SideSetList::value_type(ss_name, tmpSSVec));
      }
    }
  }
}

void Albany::PUMIDiscretization::computeNodeSets()
{
  // Make sure all the maps are allocated
  for (std::vector<std::string>::iterator ns_iter = pumiMeshStruct->nsNames.begin();
        ns_iter != pumiMeshStruct->nsNames.end(); ++ns_iter )
  { // Iterate over Node Sets
    nodeSets[*ns_iter].resize(0);
    nodeSetCoords[*ns_iter].resize(0);
    nodeset_node_coords[*ns_iter].resize(0);
  }
  //grab the node set geometric objects
  apf::StkModels sets = pumiMeshStruct->getSets();
  apf::Mesh* m = pumiMeshStruct->getMesh();
  int mesh_dim = m->getDimension();
  for (size_t i = 0; i < sets[0].getSize(); ++i)
  {
    apf::StkModel& ns = sets[0][i];
    apf::ModelEntity* me = m->findModelEntity(ns.dim, ns.apfTag);
    apf::DynamicArray<apf::Node> nodesInSet;
    apf::getNodesOnClosure(m, me, nodesInSet);
    std::vector<apf::Node> owned_ns_nodes;
    for (size_t i=0; i < nodesInSet.getSize(); ++i)
      if (m->isOwned(nodesInSet[i].entity))
        owned_ns_nodes.push_back(nodesInSet[i]);
    std::string const& NS_name = ns.stkName;
    nodeSets[NS_name].resize(owned_ns_nodes.size());
    nodeSetCoords[NS_name].resize(owned_ns_nodes.size());
    nodeset_node_coords[NS_name].resize(owned_ns_nodes.size() * mesh_dim);
    for (std::size_t i=0; i < owned_ns_nodes.size(); i++)
    {
      apf::Node node = owned_ns_nodes[i];
      nodeSets[NS_name][i].resize(neq);
      GO node_gid = apf::getNumber(globalNumbering,node);
      int node_lid = node_mapT->getLocalElement(node_gid);
      assert(node_lid >= 0);
      assert(node_lid < numOwnedNodes);
      for (std::size_t eq=0; eq < neq; eq++)
        nodeSets[NS_name][i][eq] = getDOF(node_lid, eq);
      double* node_coords = &(nodeset_node_coords[NS_name][i*mesh_dim]);
      apf::getComponents(m->getCoordinateField(),node.entity,node.node,node_coords);
      nodeSetCoords[NS_name][i] = node_coords;
    }
  }
}

void
Albany::PUMIDiscretization::updateMesh(bool shouldTransferIPData)
{
  // This function is called both to initialize the mesh at the beginning of the simulation
  // and then each time the mesh is adapted (called from AAdapt_MeshAdapt_Def.hpp - afterAdapt())

  computeOwnedNodesAndUnknowns();
  computeOverlapNodesAndUnknowns();
  setupMLCoords();
  computeGraphs();
  getCoordinates(); //fill the coordinates array
  computeWorksetInfo();
  computeNodeSets();
  computeSideSets();
  // transfer of internal variables
  if (shouldTransferIPData)
    copyQPStatesFromAPF();
}

void
Albany::PUMIDiscretization::attachQPData() {
  apf::Field* f;
  int order = pumiMeshStruct->cubatureDegree;
  int dim = pumiMeshStruct->getMesh()->getDimension();
  apf::FieldShape* fs = apf::getVoronoiShape(dim,order);
  copyQPStatesToAPF(f,fs);
}

void
Albany::PUMIDiscretization::detachQPData() {
  removeQPStatesFromAPF();
}

void Albany::PUMIDiscretization::releaseMesh () {
  if (globalNumbering) {
    apf::destroyGlobalNumbering(globalNumbering);
    globalNumbering = 0;
  }
  if (elementNumbering) {
    apf::destroyGlobalNumbering(elementNumbering);  
    elementNumbering = 0;
  }
}
