//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <limits>
#include "Epetra_Export.h"

#include "Albany_Utils.hpp"
#include "AlbPUMI_FMDBDiscretization.hpp"
#include <string>
#include <iostream>
#include <fstream>

#include <Shards_BasicTopologies.hpp>
#include "Shards_CellTopology.hpp"
#include "Shards_CellTopologyData.h"

#include <PHAL_Dimension.hpp>

#include <apfMesh.h>
#include <apfShape.h>

template<class Output>
AlbPUMI::FMDBDiscretization<Output>::FMDBDiscretization(Teuchos::RCP<AlbPUMI::FMDBMeshStruct> fmdbMeshStruct_,
            const Teuchos::RCP<const Epetra_Comm>& comm_,
            const Teuchos::RCP<Piro::MLRigidBodyModes>& rigidBodyModes_) :
  out(Teuchos::VerboseObjectBase::getDefaultOStream()),
  previous_time_label(-1.0e32),
  comm(comm_),
  rigidBodyModes(rigidBodyModes_),
  neq(fmdbMeshStruct_->neq),
  fmdbMeshStruct(fmdbMeshStruct_),
  interleavedOrdering(fmdbMeshStruct_->interleavedOrdering),
  outputInterval(0),
  meshOutput(*fmdbMeshStruct_, comm_)
{
  bool shouldTransferIPData = false;
  AlbPUMI::FMDBDiscretization<Output>::updateMesh(shouldTransferIPData);

  Teuchos::Array<std::string> layout = fmdbMeshStruct->solVectorLayout;
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

}

template<class Output>
AlbPUMI::FMDBDiscretization<Output>::~FMDBDiscretization()
{
}

template<class Output>
Teuchos::RCP<const Epetra_Map>
AlbPUMI::FMDBDiscretization<Output>::getMap() const
{
  return map;
}

template<class Output>
Teuchos::RCP<const Epetra_Map>
AlbPUMI::FMDBDiscretization<Output>::getOverlapMap() const
{
  return overlap_map;
}

template<class Output>
Teuchos::RCP<const Epetra_CrsGraph>
AlbPUMI::FMDBDiscretization<Output>::getJacobianGraph() const
{
  return graph;
}

template<class Output>
Teuchos::RCP<const Epetra_CrsGraph>
AlbPUMI::FMDBDiscretization<Output>::getOverlapJacobianGraph() const
{
  return overlap_graph;
}

template<class Output>
Teuchos::RCP<const Epetra_Map>
AlbPUMI::FMDBDiscretization<Output>::getNodeMap() const
{
  return node_map;
}

template<class Output>
Teuchos::RCP<const Epetra_Map>
AlbPUMI::FMDBDiscretization<Output>::getOverlapNodeMap() const
{
  return overlap_node_map;
}

template<class Output>
const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > > >::type&
AlbPUMI::FMDBDiscretization<Output>::getWsElNodeEqID() const
{
  return wsElNodeEqID;
}

template<class Output>
const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > >::type&
AlbPUMI::FMDBDiscretization<Output>::getWsElNodeID() const
{
  return wsElNodeID;
}

template<class Output>
const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >::type&
AlbPUMI::FMDBDiscretization<Output>::getCoords() const
{
  return coords;
}

template<class Output>
void
AlbPUMI::FMDBDiscretization<Output>::printCoords() const
{
  int mesh_dim;
  FMDB_Mesh_GetDim(fmdbMeshStruct->getMesh(), &mesh_dim);

std::cout << "Processor " << SCUTIL_CommRank() << " has " << coords.size() << " worksets." << std::endl;

       for (int ws=0; ws<coords.size(); ws++) {  //workset
         for (int e=0; e<coords[ws].size(); e++) { //cell
           for (int j=0; j<coords[ws][e].size(); j++) { //node
             for (int d=0; d<mesh_dim; d++){  //node
std::cout << "Coord for workset: " << ws << " element: " << e << " node: " << j << " DOF: " << d << " is: " <<
                coords[ws][e][j][d] << std::endl;
       } } } }

}

template<class Output>
Teuchos::ArrayRCP<double>&
AlbPUMI::FMDBDiscretization<Output>::getCoordinates() const
{
  coordinates.resize(3 * numOverlapNodes);
  apf::Field* f = fmdbMeshStruct->apfMesh->getCoordinateField();
  for (size_t i=0; i < nodes.getSize(); ++i)
    apf::getComponents(f,nodes[i].entity,nodes[i].node,&(coordinates[3*i]));
  return coordinates;
}

// FELIX uninitialized variables (FIXME)
template<class Output>
const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> > >::type&
AlbPUMI::FMDBDiscretization<Output>::getSurfaceHeight() const
{
  return sHeight;
}

template<class Output>
const Albany::WorksetArray<Teuchos::ArrayRCP<double> >::type&
AlbPUMI::FMDBDiscretization<Output>::getTemperature() const
{
  return temperature;
}

template<class Output>
const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> > >::type&
AlbPUMI::FMDBDiscretization<Output>::getBasalFriction() const
{
  return basalFriction;
}

template<class Output>
const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> > >::type&
AlbPUMI::FMDBDiscretization<Output>::getThickness() const
{
  return thickness;
}

template<class Output>
const Albany::WorksetArray<Teuchos::ArrayRCP<double> >::type&
AlbPUMI::FMDBDiscretization<Output>::getFlowFactor() const
{
  return flowFactor;
}

template<class Output>
const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >::type&
AlbPUMI::FMDBDiscretization<Output>::getSurfaceVelocity() const
{
  return surfaceVelocity;
}

template<class Output>
const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >::type&
AlbPUMI::FMDBDiscretization<Output>::getVelocityRMS() const
{
  return velocityRMS;
}

template<class Output>
const Albany::WorksetArray<Teuchos::ArrayRCP<double> >::type&
AlbPUMI::FMDBDiscretization<Output>::getSphereVolume() const
{
  return sphereVolume;
}

//The function transformMesh() maps a unit cube domain by applying the transformation
//x = L*x
//y = L*y
//z = s(x,y)*z + b(x,y)*(1-z)
//where b(x,y) and s(x,y) are curves specifying the bedrock and top surface
//geometries respectively.
//Currently this function is only needed for some FELIX problems.

template<class Output>
void
AlbPUMI::FMDBDiscretization<Output>::setupMLCoords()
{

  // Function to return x,y,z at owned nodes as double*, specifically for ML

  // if ML is not used, return

  if(rigidBodyModes.is_null()) return;

  if(!rigidBodyModes->isMLUsed()) return;

  // get mesh dimension and part handle
  int mesh_dim, counter=0;
  FMDB_Mesh_GetDim(fmdbMeshStruct->getMesh(), &mesh_dim);
  pPart part;
  FMDB_Mesh_GetPart(fmdbMeshStruct->getMesh(), 0, part);

  rigidBodyModes->resize(mesh_dim, numOwnedNodes);

  double *xx;
  double *yy;
  double *zz;

  rigidBodyModes->getCoordArrays(&xx, &yy, &zz);

  double* node_coords=new double[3];

  pPartEntIter node_it;
  pMeshEnt node;

  int owner_partid, iterEnd = FMDB_PartEntIter_Init(part, FMDB_VERTEX, FMDB_ALLTOPO, node_it);

  /* DAI: this function also has to change for high-order fields */
  while (!iterEnd)
  {
    iterEnd = FMDB_PartEntIter_GetNext(node_it, node);
    if (iterEnd) break;

    FMDB_Ent_GetOwnPartID(node, part, &owner_partid);
    if (owner_partid!=FMDB_Part_ID(part)) continue; // skip un-owned entity

    FMDB_Vtx_GetCoord (node, node_coords);
    xx[counter]=node_coords[0];
    yy[counter]=node_coords[1];
    if (mesh_dim>2) zz[counter]=node_coords[2];
    ++counter;
  }

  FMDB_PartEntIter_Del (node_it);
  delete [] node_coords;

  rigidBodyModes->informML();

}


template<class Output>
const Albany::WorksetArray<std::string>::type&
AlbPUMI::FMDBDiscretization<Output>::getWsEBNames() const
{
  return wsEBNames;
}

template<class Output>
const Albany::WorksetArray<int>::type&
AlbPUMI::FMDBDiscretization<Output>::getWsPhysIndex() const
{
  return wsPhysIndex;
}

template<class Output>
void AlbPUMI::FMDBDiscretization<Output>::setField(
    const char* name,
    const Epetra_Vector& data,
    bool overlapped,
    int offset)
{
  apf::Mesh* m = fmdbMeshStruct->apfMesh;
  apf::Field* f = m->findField(name);
  apf::Numbering* n = m->findNumbering("apf_owned_numbering");
  for (size_t i=0; i < nodes.getSize(); ++i)
  {
    apf::Node node = nodes[i];
    int node_gid = apf::getNumber(n,node.entity,node.node,0);
    int node_lid;
    if (overlapped)
      node_lid = overlap_node_map->LID(node_gid);
    else
    {
      if ( ! m->isOwned(node.entity)) continue;
      node_lid = node_map->LID(node_gid);
    }
    int firstDOF = getDOF(node_lid,offset);
    apf::setComponents(f,node.entity,node.node,&(data[firstDOF]));
  }
  if ( ! overlapped)
    apf::synchronize(f);
}

template<class Output>
void AlbPUMI::FMDBDiscretization<Output>::setSplitFields(std::vector<std::string> names,
    std::vector<int> indices, const Epetra_Vector& data, bool overlapped)
{
  apf::Mesh* m = fmdbMeshStruct->apfMesh;
  int offset = 0;
  int indexSum = 0;
  for (std::size_t i=0; i < names.size(); ++i)
  {
    assert(indexSum==offset);
    this->setField(names[i].c_str(),data,overlapped,offset);
    offset += apf::countComponents(m->findField(names[i].c_str()));
    indexSum += indices[i];
  }
}

template<class Output>
void AlbPUMI::FMDBDiscretization<Output>::getField(
    const char* name,
    Epetra_Vector& data,
    bool overlapped,
    int offset) const
{
  apf::Mesh* m = fmdbMeshStruct->apfMesh;
  apf::Field* f = m->findField(name);
  apf::Numbering* n = m->findNumbering("apf_owned_numbering");
  for (size_t i=0; i < nodes.getSize(); ++i)
  {
    apf::Node node = nodes[i];
    int node_gid = apf::getNumber(n,node.entity,node.node,0);
    int node_lid = node_map->LID(node_gid);
    if (overlapped)
      node_lid = overlap_node_map->LID(node_gid);
    else
    {
      if ( ! m->isOwned(node.entity)) continue;
      node_lid = node_map->LID(node_gid);
    }
    int firstDOF = getDOF(node_lid,offset);
/* the only difference between setField and getField is this one char:
         V                                                           */
    apf::getComponents(f,node.entity,node.node,&(data[firstDOF]));
  }
}

template<class Output>
void AlbPUMI::FMDBDiscretization<Output>::getSplitFields(std::vector<std::string> names,
   std::vector<int> indices, Epetra_Vector& data, bool overlapped) const
{
  apf::Mesh* m = fmdbMeshStruct->apfMesh;
  int offset = 0;
  int indexSum = 0;
  for (std::size_t i=0; i < names.size(); ++i)
  {
    assert(indexSum==offset);
 
    this->getField(names[i].c_str(),data,overlapped,offset);
    offset += apf::countComponents(m->findField(names[i].c_str()));
    indexSum += indices[i];
  }
}


template<class Output>
void AlbPUMI::FMDBDiscretization<Output>::writeSolution(const Epetra_Vector& soln, const double time_value,
       const bool overlapped){

  if (fmdbMeshStruct->outputFileName.empty())
    return;

  // Skip this write unless the proper interval has been reached
  if(outputInterval++ % fmdbMeshStruct->outputInterval)
    return;

  double time_label = monotonicTimeLabel(time_value);
  int out_step = 0;

  if (map->Comm().MyPID()==0) {
    *out << "AlbPUMI::FMDBDiscretization::writeSolution: writing time " << time_value;
    if (time_label != time_value) *out << " with label " << time_label;
    *out << " to index " <<out_step<<" in file "<<fmdbMeshStruct->outputFileName<< std::endl;
  }

  if (solNames.size() == 0)
    this->setField("solution",soln,overlapped);
  else
    this->setSplitFields(solNames,solIndex,soln,overlapped);

  fmdbMeshStruct->solutionInitialized = true;

  outputInterval = 0;

  apf::Field* f;
  int order = fmdbMeshStruct->cubatureDegree;
  int dim = fmdbMeshStruct->apfMesh->getDimension();
  apf::FieldShape* fs = apf::getIPShape(dim,order);
  copyQPStatesToAPF(f,fs);
  meshOutput.writeFile(time_label);
  removeQPStatesFromAPF();

}

template<class Output>
void
AlbPUMI::FMDBDiscretization<Output>::debugMeshWriteNative(const Epetra_Vector& soln, const char* filename){

  if (solNames.size() == 0 )
    this->setField("solution",soln,/*overlapped=*/false);
  else
    this->setSplitFields(solNames,solIndex,soln,/*overlapped=*/false);

  fmdbMeshStruct->solutionInitialized = true;
  fmdbMeshStruct->apfMesh->writeNative(filename);

}

template<class Output>
void
AlbPUMI::FMDBDiscretization<Output>::debugMeshWrite(const Epetra_Vector& soln, const char* filename){

  if (solNames.size() == 0 )
    this->setField("solution",soln,/*overlapped=*/false);
  else
    this->setSplitFields(solNames,solIndex,soln,/*overlapped=*/false);

  if (! PCU_Comm_Self()) {
    std::cout << "************************************************" << std::endl;
    std::cout << "Writing mesh debug output! " << std::endl;
    std::cout << "************************************************" << std::endl;
    std::cout << std::endl;
  }

  fmdbMeshStruct->solutionInitialized = true;

  apf::Field* f;
  int order = fmdbMeshStruct->cubatureDegree;
  int dim = fmdbMeshStruct->apfMesh->getDimension();
  apf::FieldShape* fs = apf::getIPShape(dim,order);
  copyQPStatesToAPF(f,fs);
  meshOutput.debugMeshWrite(filename);
  removeQPStatesFromAPF();

}

template<class Output>
double
AlbPUMI::FMDBDiscretization<Output>::monotonicTimeLabel(const double time)
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

template<class Output>
void
AlbPUMI::FMDBDiscretization<Output>::setResidualField(const Epetra_Vector& residual)
{
  if (solNames.size() == 0)
    this->setField("residual",residual,/*overlapped=*/false);
  else
    this->setSplitFields(resNames,solIndex,residual,/*overlapped=*/false);

  fmdbMeshStruct->residualInitialized = true;
}

template<class Output>
Teuchos::RCP<Epetra_Vector>
AlbPUMI::FMDBDiscretization<Output>::getSolutionField() const
{
  // Copy soln vector into solution field, one node at a time
  Teuchos::RCP<Epetra_Vector> soln = Teuchos::rcp(new Epetra_Vector(*map));

  if (fmdbMeshStruct->solutionInitialized) {
    if (solNames.size() == 0)
      this->getField("solution",*soln,/*overlapped=*/false);
    else
      this->getSplitFields(solNames,solIndex,*soln,/*overlapped=*/false);
  }
  else if ( ! PCU_Comm_Self())
    *out <<__func__<<": uninit field" << std::endl;

  return soln;
}

template<class Output>
void
AlbPUMI::FMDBDiscretization<Output>::setSolutionField(const Epetra_Vector& soln)
{
  if (solNames.size() == 0)
    this->setField("solution",soln,/*overlapped=*/false);
  else
    this->setSplitFields(solNames,solIndex,soln,/*overlapped=*/false);

  fmdbMeshStruct->solutionInitialized = true;
}


template<class Output>
int AlbPUMI::FMDBDiscretization<Output>::nonzeroesPerRow(const int neq) const
{
  int numDim;
  FMDB_Mesh_GetDim(fmdbMeshStruct->getMesh(), &numDim);

  /* DAI: this function should be revisited for overall correctness,
     especially in the case of higher-order fields */
  int estNonzeroesPerRow;
  switch (numDim) {
  case 0: estNonzeroesPerRow=1*neq; break;
  case 1: estNonzeroesPerRow=3*neq; break;
  case 2: estNonzeroesPerRow=9*neq; break;
  case 3: estNonzeroesPerRow=27*neq; break;
  default: TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
			      "FMDBDiscretization:  Bad numDim"<< numDim);
  }
  return estNonzeroesPerRow;
}

template<class Output>
void AlbPUMI::FMDBDiscretization<Output>::computeOwnedNodesAndUnknowns()
{
  apf::Mesh* m = fmdbMeshStruct->apfMesh;
  apf::Numbering* n = m->findNumbering("apf_owned_numbering");
  if (n) apf::destroyNumbering(n);
  n = apf::numberOwnedNodes(m,"apf_owned_numbering");
  apf::globalize(n); /* turn local numbering to global numbering */
  apf::DynamicArray<apf::Node> ownedNodes;
  apf::getNodes(n,ownedNodes);
  numOwnedNodes = ownedNodes.getSize();
  apf::synchronize(n); /* this puts owned node numbers on non-owned overlap nodes */
  std::vector<int> indices(numOwnedNodes);
  for (int i=0; i < numOwnedNodes; ++i)
    indices[i] = getNumber(n,ownedNodes[i].entity,ownedNodes[i].node,0);
  node_map = Teuchos::rcp(new Epetra_Map(-1, numOwnedNodes,
					 &(indices[0]), 0, *comm));
  numGlobalNodes = node_map->MaxAllGID() + 1;
  if(Teuchos::nonnull(fmdbMeshStruct->nodal_data_block))
    fmdbMeshStruct->nodal_data_block->resizeLocalMap(indices, *comm);
  indices.resize(numOwnedNodes*neq);
  for (int i=0; i < numOwnedNodes; ++i)
    for (int j=0; j < neq; ++j)
      indices[getDOF(i,j)] = 
        getDOF(getNumber(n,ownedNodes[i].entity,ownedNodes[i].node,0),j);
  map = Teuchos::rcp(new Epetra_Map(-1, indices.size(), &(indices[0]), 0, *comm));
}

template <class Output>
void AlbPUMI::FMDBDiscretization<Output>::computeOverlapNodesAndUnknowns()
{
  apf::Mesh* m = fmdbMeshStruct->apfMesh;
  apf::Numbering* overlap = m->findNumbering("apf_overlap_numbering");
  if (overlap) apf::destroyNumbering(overlap);
  overlap = apf::numberOverlapNodes(m,"apf_overlap_numbering");
  apf::getNodes(overlap,nodes);
  numOverlapNodes = nodes.getSize();
  std::vector<int> nodeIndices(numOverlapNodes);
  std::vector<int> dofIndices(numOverlapNodes*neq);
  apf::Numbering* owned = m->findNumbering("apf_owned_numbering");
  for (int i=0; i < numOverlapNodes; ++i)
  {
    int global = apf::getNumber(owned,nodes[i].entity,nodes[i].node,0);
    nodeIndices[i] = global;
    for (int j=0; j < neq; ++j)
      dofIndices[getDOF(i,j)] = getDOF(global,j);
  }
  overlap_node_map = Teuchos::rcp(new Epetra_Map(-1, nodeIndices.size(),
						 &(nodeIndices[0]), 0, *comm));
  overlap_map = Teuchos::rcp(new Epetra_Map(-1, dofIndices.size(),
					    &(dofIndices[0]), 0, *comm));
  if(Teuchos::nonnull(fmdbMeshStruct->nodal_data_block))
    fmdbMeshStruct->nodal_data_block->resizeOverlapMap(nodeIndices, *comm);
}

template<class Output>
void AlbPUMI::FMDBDiscretization<Output>::computeGraphs()
{
  // GAH: the following assumes all element blocks in the problem have the same
  // number of nodes per element and that the cell topologies are the same.
  apf::Mesh* m = fmdbMeshStruct->apfMesh;
  int numDim = m->getDimension();
  std::vector<apf::MeshEntity*> cells;
  cells.reserve(m->count(numDim));
  apf::MeshIterator* it = m->begin(numDim);
  apf::MeshEntity* e;
  while ((e = m->iterate(it)))
    cells.push_back(e);
  m->end(it);
  //got cells, count the nodes on the first one
  apf::Numbering* n = m->findNumbering("apf_owned_numbering");
  int nodes_per_element = apf::countElementNodes(
      apf::getShape(n),m->getType(cells[0]));
  /* construct the overlap graph of all local DOFs as they
     are coupled by element-node connectivity */
  overlap_graph =
    Teuchos::rcp(new Epetra_CrsGraph(Copy, *overlap_map,
                                     neq*nodes_per_element, false));
  for (size_t i=0; i < cells.size(); ++i)
  {
    apf::NewArray<int> cellNodes;
    apf::getElementNumbers(n,cells[i],cellNodes);
    for (int j=0; j < nodes_per_element; ++j)
    {
      for (int k=0; k < neq; ++k)
      {
        int row = getDOF(cellNodes[j],k);
        for (int l=0; l < nodes_per_element; ++l)
        {
          for (int m=0; m < neq; ++m)
          {
            int col = getDOF(cellNodes[l],m);
            overlap_graph->InsertGlobalIndices(row,1,&col);
          }
        }
      }
    }
  }
  overlap_graph->FillComplete();

  // Create Owned graph by exporting overlap with known row map

  graph = Teuchos::rcp(new Epetra_CrsGraph(Copy, *map, nonzeroesPerRow(neq), false));

  // Create non-overlapped matrix using two maps and export object
  Epetra_Export exporter(*overlap_map, *map);
  graph->Export(*overlap_graph, exporter, Insert);
  graph->FillComplete();
}

template<class Output>
void AlbPUMI::FMDBDiscretization<Output>::computeWorksetInfo()
{
  apf::Mesh* m = fmdbMeshStruct->apfMesh;
  int numDim = m->getDimension();
  apf::Numbering* en = m->findNumbering("apf_element_numbering");
  if (en) apf::destroyNumbering(en);
  en = apf::numberElements(m,"apf_element_numbering");
  apf::globalize(en);
  apf::Numbering* nn = m->findNumbering("apf_owned_numbering");

/*
   * Note: Max workset size is given in input file, or set to a default in AlbPUMI_FMDBMeshStruct.cpp
   * The workset size is set in AlbPUMI_FMDBMeshStruct.cpp to be the maximum number in an element block if
   * the element block size < Max workset size.
   * STK bucket size is set to the workset size. We will "chunk" the elements into worksets here.
   *
*/

  // This function is called each adaptive cycle. Need to reset the 2D array "buckets" back to the initial size.

  for(int i = 0; i < buckets.size(); i++)
    buckets[i].clear();

  buckets.clear();

  pGeomEnt elem_blk;
  std::map<pElemBlk, int> bucketMap;
  std::map<pElemBlk, int>::iterator buck_it;
  int bucket_counter = 0;

  int worksetSize = fmdbMeshStruct->worksetSize;

  // iterate over all elements
  apf::MeshIterator* it = m->begin(numDim);
  apf::MeshEntity* element;
  while ((element = m->iterate(it)))
  {

    // skip owned elements
    if ( ! m->isOwned(element))
      continue;

    // Get the element block that the element is in
    elem_blk = reinterpret_cast<pGeomEnt>(m->toModel(element));

    // find which bucket holds the elements for the element block
    buck_it = bucketMap.find(elem_blk);

    if((buck_it == bucketMap.end()) ||  // Make a new bucket to hold the new element block's elements
       (buckets[buck_it->second].size() >= worksetSize)){ // old bucket is full, put the element in a new one

      // Associate this elem_blk with a new bucket
      bucketMap[elem_blk] = bucket_counter;

      // resize the bucket array larger by one
      buckets.resize(bucket_counter + 1);
      wsEBNames.resize(bucket_counter + 1);

      // save the element in the bucket
      buckets[bucket_counter].push_back(element);

      // save the name of the new element block
      std::string EB_name;
      PUMI_ElemBlk_GetName(elem_blk, EB_name);
      wsEBNames[bucket_counter] = EB_name;

      bucket_counter++;

    }
    else { // put the element in the proper bucket

      buckets[buck_it->second].push_back(element);

    }

  }

  int numBuckets = bucket_counter;

  wsPhysIndex.resize(numBuckets);

  if (fmdbMeshStruct->allElementBlocksHaveSamePhysics)
    for (int i=0; i<numBuckets; i++) wsPhysIndex[i]=0;
  else
    for (int i=0; i<numBuckets; i++) wsPhysIndex[i]=fmdbMeshStruct->ebNameToIndex[wsEBNames[i]];

  // Fill  wsElNodeEqID(workset, el_LID, local node, Eq) => unk_LID

  wsElNodeEqID.resize(numBuckets);
  wsElNodeID.resize(numBuckets);
  coords.resize(numBuckets);
  sHeight.resize(numBuckets);
  temperature.resize(numBuckets);
  basalFriction.resize(numBuckets);
  thickness.resize(numBuckets);
  surfaceVelocity.resize(numBuckets);
  velocityRMS.resize(numBuckets);
  flowFactor.resize(numBuckets);
  surfaceVelocity.resize(numBuckets);
  velocityRMS.resize(numBuckets);

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

      // Now, save a map from element GID to workset on this PE
      elemGIDws[apf::getNumber(en,element,0,0)].ws = b;

      // Now, save a map element GID to local id on this workset on this PE
      elemGIDws[apf::getNumber(en,element,0,0)].LID = i;

      // get global node numbers
      apf::NewArray<int> nodeIDs;
      apf::getElementNumbers(nn,element,nodeIDs);

      int nodes_per_element = apf::countElementNodes(apf::getShape(nn),m->getType(element));
      wsElNodeEqID[b][i].resize(nodes_per_element);
      wsElNodeID[b][i].resize(nodes_per_element);
      coords[b][i].resize(nodes_per_element);

      // loop over local nodes

      for (int j=0; j < nodes_per_element; j++) {

        int node_gid = nodeIDs[j];
        int node_lid = overlap_node_map->LID(node_gid);

        TEUCHOS_TEST_FOR_EXCEPTION(node_lid<0, std::logic_error,
			   "FMDB1D_Disc: node_lid out of range " << node_lid << std::endl);

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

  for (std::size_t i=0; i<fmdbMeshStruct->qpscalar_states.size(); i++) {
      fmdbMeshStruct->qpscalar_states[i]->reAllocateBuffer(numElementsAccessed);
  }
  for (std::size_t i=0; i<fmdbMeshStruct->qpvector_states.size(); i++) {
      fmdbMeshStruct->qpvector_states[i]->reAllocateBuffer(numElementsAccessed);
  }
  for (std::size_t i=0; i<fmdbMeshStruct->qptensor_states.size(); i++) {
      fmdbMeshStruct->qptensor_states[i]->reAllocateBuffer(numElementsAccessed);
  }
  for (std::size_t i=0; i<fmdbMeshStruct->scalarValue_states.size(); i++) {
      // special case : need to store one double value that represents all the elements in the workset (time)
      // numBuckets are the number of worksets
      fmdbMeshStruct->scalarValue_states[i]->reAllocateBuffer(numBuckets);
  }

  // Pull out pointers to shards::Arrays for every bucket, for every state

  // Note that numBuckets is typically larger each time the mesh is adapted

  stateArrays.elemStateArrays.resize(numBuckets);

  for (std::size_t b=0; b < buckets.size(); b++) {

    std::vector<apf::MeshEntity*>& buck = buckets[b];

    for (std::size_t i=0; i<fmdbMeshStruct->qpscalar_states.size(); i++) {
      stateArrays.elemStateArrays[b][fmdbMeshStruct->qpscalar_states[i]->name] =
                 fmdbMeshStruct->qpscalar_states[i]->getMDA(buck.size());
    }
    for (std::size_t i=0; i<fmdbMeshStruct->qpvector_states.size(); i++) {
      stateArrays.elemStateArrays[b][fmdbMeshStruct->qpvector_states[i]->name] =
                 fmdbMeshStruct->qpvector_states[i]->getMDA(buck.size());
    }
    for (std::size_t i=0; i<fmdbMeshStruct->qptensor_states.size(); i++) {
      stateArrays.elemStateArrays[b][fmdbMeshStruct->qptensor_states[i]->name] =
                 fmdbMeshStruct->qptensor_states[i]->getMDA(buck.size());
    }
    for (std::size_t i=0; i<fmdbMeshStruct->scalarValue_states.size(); i++) {
      // Store one double precision value per workset
      const int size = 1;
      stateArrays.elemStateArrays[b][fmdbMeshStruct->scalarValue_states[i]->name] =
                 fmdbMeshStruct->scalarValue_states[i]->getMDA(size);
    }
  }

// Process node data sets if present

  if(Teuchos::nonnull(fmdbMeshStruct->nodal_data_block)){

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

    Teuchos::RCP<Albany::NodeFieldContainer> node_states = fmdbMeshStruct->nodal_data_block->getNodeContainer();

    stateArrays.nodeStateArrays.resize(numNodeBuckets);

    // Loop over all the node field containers
    for (Albany::NodeFieldContainer::iterator nfs = node_states->begin();
                nfs != node_states->end(); ++nfs){
      Teuchos::RCP<AlbPUMI::AbstractPUMINodeFieldContainer> nodeContainer =
             Teuchos::rcp_dynamic_cast<AlbPUMI::AbstractPUMINodeFieldContainer>((*nfs).second);

      // resize the container to hold all the owned node's data
      nodeContainer->resize(node_map);

      // Now, loop over each workset to get a reference to each workset collection of nodes
      for (std::size_t b=0; b < nbuckets.size(); b++) {
         std::vector<apf::Node>& buck = nbuckets[b];
         stateArrays.nodeStateArrays[b][(*nfs).first] = nodeContainer->getMDA(buck);
      }
    }
  }
}

template<class Output>
void AlbPUMI::FMDBDiscretization<Output>::copyQPScalarToAPF(
    unsigned nqp,
    QPData<double, 2>& state,
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

template<class Output>
void AlbPUMI::FMDBDiscretization<Output>::copyQPVectorToAPF(
    unsigned nqp,
    QPData<double, 3>& state,
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

template <class Output>
void AlbPUMI::FMDBDiscretization<Output>::copyQPTensorToAPF(
    unsigned nqp,
    QPData<double, 4>& state,
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

template<class Output>
void AlbPUMI::FMDBDiscretization<Output>::copyQPStatesToAPF(apf::Field* f, apf::FieldShape* fs) {
  apf::Mesh2* m = fmdbMeshStruct->apfMesh;
  for (std::size_t i=0; i < fmdbMeshStruct->qpscalar_states.size(); ++i) {
    QPData<double, 2>& state = *(fmdbMeshStruct->qpscalar_states[i]);
    int nqp = state.dims[1];
    f = apf::createField(m,state.name.c_str(),apf::SCALAR,fs);
    copyQPScalarToAPF(nqp,state,f);
  }
  for (std::size_t i=0; i < fmdbMeshStruct->qpvector_states.size(); ++i) {
    QPData<double, 3>& state = *(fmdbMeshStruct->qpvector_states[i]);
    int nqp = state.dims[1];
    f = apf::createField(m,state.name.c_str(),apf::VECTOR,fs);
    copyQPVectorToAPF(nqp,state,f);
  }
  for (std::size_t i=0; i < fmdbMeshStruct->qptensor_states.size(); ++i) {
    QPData<double, 4>& state = *(fmdbMeshStruct->qptensor_states[i]);
    int nqp = state.dims[1];
    f = apf::createField(m,state.name.c_str(),apf::MATRIX,fs);
    copyQPTensorToAPF(nqp,state,f);
  }
}

template<class Output>
void AlbPUMI::FMDBDiscretization<Output>::removeQPStatesFromAPF() {
  apf::Mesh2* m = fmdbMeshStruct->apfMesh;
  for (std::size_t i=0; i < fmdbMeshStruct->qpscalar_states.size(); ++i) {
    QPData<double, 2>& state = *(fmdbMeshStruct->qpscalar_states[i]);
    apf::destroyField(m->findField(state.name.c_str()));
  }
  for (std::size_t i=0; i < fmdbMeshStruct->qpvector_states.size(); ++i) {
    QPData<double, 3>& state = *(fmdbMeshStruct->qpvector_states[i]);
    apf::destroyField(m->findField(state.name.c_str()));
  }
  for (std::size_t i=0; i < fmdbMeshStruct->qptensor_states.size(); ++i) {
    QPData<double, 4>& state = *(fmdbMeshStruct->qptensor_states[i]);
    apf::destroyField(m->findField(state.name.c_str()));
  }
}

template<class Output>
void AlbPUMI::FMDBDiscretization<Output>::copyQPScalarFromAPF(
    unsigned nqp,
    QPData<double, 2>& state,
    apf::Field* f) 
{
  apf::Mesh2* m = fmdbMeshStruct->apfMesh;
  for (std::size_t b=0; b < buckets.size(); ++b) {
    std::vector<apf::MeshEntity*>& buck = buckets[b];
    Albany::MDArray& ar = stateArrays.elemStateArrays[b][state.name];
    for (std::size_t e=0; e < buck.size(); ++e) {
      for (std::size_t p = 0; p < nqp; ++p) {
        ar(e,p) = apf::getScalar(f,buck[e],p);
  } } }
}

template<class Output>
void AlbPUMI::FMDBDiscretization<Output>::copyQPVectorFromAPF(
    unsigned nqp,
    QPData<double, 3>& state,
    apf::Field* f) 
{
  apf::Mesh2* m = fmdbMeshStruct->apfMesh;
  for (std::size_t b=0; b < buckets.size(); ++b) {
    std::vector<apf::MeshEntity*>& buck = buckets[b];
    Albany::MDArray& ar = stateArrays.elemStateArrays[b][state.name];
    for (std::size_t e=0; e < buck.size(); ++e) {
      apf::Vector3 v;
      for (std::size_t p=0; p < nqp; ++p) {
        apf::getVector(f,buck[e],p,v);
        for (std::size_t i=0; i < 3; ++i) {
          ar(e,p,i) = v[i];
  } } } }
}

template <class Output>
void AlbPUMI::FMDBDiscretization<Output>::copyQPTensorFromAPF(
    unsigned nqp,
    QPData<double, 4>& state,
    apf::Field* f)
{
  apf::Mesh2* m = fmdbMeshStruct->apfMesh;
  for (std::size_t b = 0; b < buckets.size(); ++b) {
    std::vector<apf::MeshEntity*>& buck = buckets[b];
    Albany::MDArray& ar = stateArrays.elemStateArrays[b][state.name];
    for (std::size_t e=0; e < buck.size(); ++e) {
      apf::Matrix3x3 v;
      for (std::size_t p=0; p < nqp; ++p) {
        apf::getMatrix(f,buck[e],p,v);
        for (std::size_t i=0; i < 3; ++i) {
          for (std::size_t j=0; j < 3; ++j) {
            ar(e,p,i,j) = v[i][j];
  } } } } }
}

template<class Output>
void AlbPUMI::FMDBDiscretization<Output>::copyQPStatesFromAPF() {
  apf::Mesh2* m = fmdbMeshStruct->apfMesh;
  apf::Field* f;
  for (std::size_t i=0; i < fmdbMeshStruct->qpscalar_states.size(); ++i) {
    QPData<double, 2>& state = *(fmdbMeshStruct->qpscalar_states[i]);
    int nqp = state.dims[1];
    f = m->findField(state.name.c_str());
    copyQPScalarFromAPF(nqp,state,f);
  }
  for (std::size_t i=0; i < fmdbMeshStruct->qpvector_states.size(); ++i) {
    QPData<double, 3>& state = *(fmdbMeshStruct->qpvector_states[i]);
    int nqp = state.dims[1];
    f = m->findField(state.name.c_str());
    copyQPVectorFromAPF(nqp,state,f);
  }
  for (std::size_t i=0; i < fmdbMeshStruct->qptensor_states.size(); ++i) {
    QPData<double, 4>& state = *(fmdbMeshStruct->qptensor_states[i]);
    int nqp = state.dims[1];
    f = m->findField(state.name.c_str());
    copyQPTensorFromAPF(nqp,state,f);
  }
}

template<class Output>
void AlbPUMI::FMDBDiscretization<Output>::computeSideSets() {

  pMeshMdl mesh = fmdbMeshStruct->getMesh();
  pPart part;
  FMDB_Mesh_GetPart(mesh, 0, part);
  apf::Mesh* m = fmdbMeshStruct->apfMesh;
  apf::Numbering* en = m->findNumbering("apf_element_numbering");

  // need a sideset list per workset
  int num_buckets = wsEBNames.size();
  sideSets.resize(num_buckets);

  // get side sets
  std::vector<pSideSet> side_sets;
  PUMI_Exodus_GetSideSet(mesh, side_sets);

  std::vector<pMeshEnt> side_elems;
  std::vector<pMeshEnt> ss_sides;

  // loop over side sets
  for (std::vector<pSideSet>::iterator ss = side_sets.begin();
       ss != side_sets.end(); ++ss) {

    // get the name of this side set
    std::string ss_name;
    PUMI_SideSet_GetName(*ss, ss_name);

    // get sides on side the side set
    ss_sides.clear();
    PUMI_SideSet_GetSide(mesh, *ss, ss_sides);

    // loop over the sides in this side set
    for (std::vector<pMeshEnt>::iterator side = ss_sides.begin();
	 side != ss_sides.end(); ++side) {

      // get the elements adjacent to this side
      // note - if the side is internal, it will show up twice in the element list,
      // once for each element that contains it

      side_elems.clear();
      int side_dim;
      FMDB_Ent_GetType(*side, &side_dim);
      FMDB_Ent_GetAdj(*side, side_dim+1, 1, side_elems);

      // according to template below - we are not yet considering non-manifold side sets?
      // i.e. side_elems.size() > 1
      TEUCHOS_TEST_FOR_EXCEPTION(side_elems.size() != 1, std::logic_error,
		   "FMDBDisc: cannot figure out side set topology for side set "<<ss_name<<std::endl);

      pMeshEnt elem = side_elems[0];

      // fill in the data holder for a side struct

      Albany::SideStruct sstruct;

      sstruct.elem_GID = apf::getNumber(en,apf::castEntity(elem),0,0);
      int workset = elemGIDws[sstruct.elem_GID].ws; // workset ID that this element lives in
      sstruct.elem_LID = elemGIDws[sstruct.elem_GID].LID; // local element id in this workset
      sstruct.elem_ebIndex = fmdbMeshStruct->ebNameToIndex[wsEBNames[workset]]; // element block that workset lives in

      int side_exodus_order;
      PUMI_MeshEnt_GetExodusOrder(elem, *side, &side_exodus_order);
      sstruct.side_local_id = side_exodus_order-1; // local id of side wrt element

      Albany::SideSetList& ssList = sideSets[workset]; // Get a ref to the side set map for this ws

      Albany::SideSetList::iterator it = ssList.find(ss_name); // Get an iterator to the correct sideset (if
                                                       // it exists)

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

template<class Output>
void AlbPUMI::FMDBDiscretization<Output>::computeNodeSets()
{
  // Make sure all the maps are allocated
  for (std::vector<std::string>::iterator ns_iter = fmdbMeshStruct->nsNames.begin();
        ns_iter != fmdbMeshStruct->nsNames.end(); ++ns_iter )
  { // Iterate over Node Sets
    nodeSets[*ns_iter].resize(0);
    nodeSetCoords[*ns_iter].resize(0);
    nodeset_node_coords[*ns_iter].resize(0);
  }
  //grab the node set geometric objects
  std::vector<pNodeSet> node_set;
  PUMI_Exodus_GetNodeSet(fmdbMeshStruct->getMesh(), node_set);
  apf::Mesh* m = fmdbMeshStruct->apfMesh;
  apf::Numbering* n = m->findNumbering("apf_owned_numbering");
  int mesh_dim = m->getDimension();
  for (std::vector<pNodeSet>::iterator node_set_it=node_set.begin(); node_set_it!=node_set.end(); ++node_set_it)
  {
    apf::DynamicArray<apf::Node> nodesInSet;
    apf::ModelEntity* me = reinterpret_cast<apf::ModelEntity*>(*node_set_it);
    apf::getNodesOnClosure(m,me,nodesInSet);
    std::vector<apf::Node> owned_ns_nodes;
    for (size_t i=0; i < nodesInSet.getSize(); ++i)
      if (m->isOwned(nodesInSet[i].entity))
        owned_ns_nodes.push_back(nodesInSet[i]);
    std::string NS_name;
    PUMI_NodeSet_GetName(*node_set_it, NS_name);
    nodeSets[NS_name].resize(owned_ns_nodes.size());
    nodeSetCoords[NS_name].resize(owned_ns_nodes.size());
    nodeset_node_coords[NS_name].resize(owned_ns_nodes.size() * mesh_dim);
    for (std::size_t i=0; i < owned_ns_nodes.size(); i++)
    {
      apf::Node node = owned_ns_nodes[i];
      nodeSets[NS_name][i].resize(neq);
      int node_gid = apf::getNumber(n,node.entity,node.node,0);
      int node_lid = node_map->LID(node_gid);
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

template<class Output>
void
AlbPUMI::FMDBDiscretization<Output>::updateMesh(bool shouldTransferIPData)
{
  computeOwnedNodesAndUnknowns();
#ifdef ALBANY_DEBUG
  std::cout<<"["<<SCUTIL_CommRank()<<"] "<<__func__<<": computeOwnedNodesAndUnknowns() completed\n";
#endif

  computeOverlapNodesAndUnknowns();
#ifdef ALBANY_DEBUG
  std::cout<<"["<<SCUTIL_CommRank()<<"] "<<__func__<<": computeOverlapNodesAndUnknowns() completed\n";
#endif

  computeGraphs();
#ifdef ALBANY_DEBUG
  std::cout<<"["<<SCUTIL_CommRank()<<"] "<<__func__<<": computeGraphs() completed\n";
#endif

  getCoordinates(); //fill the coordinates array

  computeWorksetInfo();
#ifdef ALBANY_DEBUG
  std::cout<<"["<<SCUTIL_CommRank()<<"] "<<__func__<<": computeWorksetInfo() completed\n";
#endif

  computeNodeSets();
#ifdef ALBANY_DEBUG
  std::cout<<"["<<SCUTIL_CommRank()<<"] "<<__func__<<": computeNodeSets() completed\n";
#endif

  computeSideSets();
#ifdef ALBANY_DEBUG
  std::cout<<"["<<SCUTIL_CommRank()<<"] "<<__func__<<": computeSideSets() completed\n";
#endif

  // transfer of internal variables
  if (shouldTransferIPData)  {
    copyQPStatesFromAPF();
    writeVtkFiles("after-solution-transfer",fmdbMeshStruct->apfMesh);
  }
}

template<class Output>
void
AlbPUMI::FMDBDiscretization<Output>::attachQPData() {
  apf::Field* f;
  int order = fmdbMeshStruct->cubatureDegree;
  int dim = fmdbMeshStruct->apfMesh->getDimension();
  apf::FieldShape* fs = apf::getVoronoiShape(dim,order); 
  copyQPStatesToAPF(f,fs);
}

template<class Output>
void
AlbPUMI::FMDBDiscretization<Output>::detachQPData() {
  removeQPStatesFromAPF();
}


