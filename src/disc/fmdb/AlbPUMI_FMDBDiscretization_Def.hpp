//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <limits>
#include "Epetra_Export.h"

#include "Albany_Utils.hpp"
#include "Albany_FMDBDiscretization.hpp"
#include "Petra_Converters.hpp"
#include <string>
#include <iostream>
#include <fstream>

#include "apfMesh.h"
//#include "apfField.h"

#include <Shards_BasicTopologies.hpp>
#include "Shards_CellTopology.hpp"
#include "Shards_CellTopologyData.h"

#include <PHAL_Dimension.hpp>
#define DEBUG 1

template<class Output>
AlbPUMI::FMDBDiscretization<Output>::FMDBDiscretization(Teuchos::RCP<AlbPUMI::FMDBMeshStruct> fmdbMeshStruct_,
            const Teuchos::RCP<const Epetra_Comm>& comm_,
            const Teuchos::RCP<Piro::MLRigidBodyModes>& rigidBodyModes_) :
  out(Teuchos::VerboseObjectBase::getDefaultOStream()),
  previous_time_label(-1.0e32),
  comm(comm_),
  commT(Albany::createTeuchosCommFromMpiComm(Albany::getMpiCommFromEpetraComm(*comm_))),
  rigidBodyModes(rigidBodyModes_),
  neq(fmdbMeshStruct_->neq),
  fmdbMeshStruct(fmdbMeshStruct_),
  interleavedOrdering(fmdbMeshStruct_->interleavedOrdering),
  outputInterval(0),
  meshOutput(*fmdbMeshStruct_, comm_)
{

   //Ultimately Tpetra comm needs to be passed in to this constructor like Epetra comm...
   //Create the Kokkos Node instance to pass into Tpetra::Map constructors.
  Teuchos::ParameterList kokkosNodeParams;
  nodeT = Teuchos::rcp(new KokkosNode (kokkosNodeParams));

  AlbPUMI::FMDBDiscretization<Output>::updateMesh();

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
  Teuchos::RCP<const Epetra_Map> map = Petra::TpetraMap_To_EpetraMap(mapT, comm);
  return map;
}

Teuchos::RCP<const Tpetra_Map>
Albany::FMDBDiscretization::getMapT() const
{
  return mapT;
}

template<class Output>
Teuchos::RCP<const Epetra_Map>
AlbPUMI::FMDBDiscretization<Output>::getOverlapMap() const
{
  Teuchos::RCP<const Epetra_Map> overlap_map = Petra::TpetraMap_To_EpetraMap(overlap_mapT, comm);
  return overlap_map;
}

Teuchos::RCP<const Tpetra_Map>
Albany::FMDBDiscretization::getOverlapMapT() const
{
  return overlap_mapT;
}

template<class Output>
Teuchos::RCP<const Epetra_CrsGraph>
AlbPUMI::FMDBDiscretization<Output>::getJacobianGraph() const
{
  Teuchos::RCP<const Epetra_CrsGraph> graph= Petra::TpetraCrsGraph_To_EpetraCrsGraph(graphT, comm);
  return graph;
}

Teuchos::RCP<const Tpetra_CrsGraph>
Albany::FMDBDiscretization::getJacobianGraphT() const
{
  return graphT;
}

template<class Output>
Teuchos::RCP<const Epetra_CrsGraph>
AlbPUMI::FMDBDiscretization<Output>::getOverlapJacobianGraph() const
{
  Teuchos::RCP<const Epetra_CrsGraph> overlap_graph= Petra::TpetraCrsGraph_To_EpetraCrsGraph(overlap_graphT, comm);
  return overlap_graph;
}

Teuchos::RCP<const Tpetra_CrsGraph>

Albany::FMDBDiscretization::getOverlapJacobianGraphT() const
{
  return overlap_graphT;
}

template<class Output>
Teuchos::RCP<const Epetra_Map>
AlbPUMI::FMDBDiscretization<Output>::getNodeMap() const
{
  Teuchos::RCP<const Epetra_Map> node_map = Petra::TpetraMap_To_EpetraMap(node_mapT, comm);
  return node_map;
}

Teuchos::RCP<const Tpetra_Map>
Albany::FMDBDiscretization::getNodeMapT() const
{
  return node_mapT;
}

/*
template<class Output>
Teuchos::RCP<const Epetra_Map>
AlbPUMI::FMDBDiscretization<Output>::getOverlapNodeMap() const
{
  return overlap_node_map;
}
*/

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
  // get mesh dimension and part handle
  int mesh_dim, counter=0;
  FMDB_Mesh_GetDim(fmdbMeshStruct->getMesh(), &mesh_dim);
  pPart part;
  FMDB_Mesh_GetPart(fmdbMeshStruct->getMesh(), 0, part);

  // iterate over all vertices (nodes)
  pMeshEnt node;
  pPartEntIter node_it;

  int iterEnd = FMDB_PartEntIter_Init(part, FMDB_VERTEX, FMDB_ALLTOPO, node_it);
  while (!iterEnd)
  {
    iterEnd = FMDB_PartEntIter_GetNext(node_it, node);
    if (iterEnd) break;
    int node_gid = FMDB_Ent_ID(node);
    int node_lid = overlap_node_mapT->getLocalElement(node_gid);
    // get vertex(node) coords
    FMDB_Vtx_GetCoord (node,
       &coordinates[3 * node_lid]); // Extract a pointer to the correct spot to begin placing coordinates
  }

  FMDB_PartEntIter_Del (node_it);

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

  while (!iterEnd)
  {
    iterEnd = FMDB_PartEntIter_GetNext(node_it, node);
    if (iterEnd) break;

    FMDB_Ent_GetOwnPartID(node, part, &owner_partid);
    if (owner_partid!=FMDB_Part_ID(part)) continue; // skip un-owned entity

//    FMDB_Vtx_GetCoord (node, &node_coords);
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
void AlbPUMI::FMDBDiscretization<Output>::setField(const char* name, const Epetra_Vector& data, bool overlapped)
{
  apf::Mesh* m = fmdbMeshStruct->apfMesh;
  apf::Field* f = m->findField(name);
  apf::MeshIterator* it = m->begin(0);
  apf::MeshEntity* e;
  while ((e = m->iterate(it)))
  {
    int node_gid = FMDB_Ent_ID(reinterpret_cast<pMeshEnt>(e));
    int node_lid = node_mapT->getLocalElement(node_gid);
    if (overlapped)
      node_lid = overlap_node_mapT->getLocalElement(node_gid);
    else
    {
      if ( ! m->isOwned(e)) continue;
      node_lid = node_mapT->getLocalElement(node_gid);
    }
    if (neq==1)
    {
      int dofId = getOverlapDOF(node_lid,0);
      apf::setScalar(f,e,0,data[dofId]);
    }
    else
    { assert(neq==3);
      apf::Vector3 v;
      for (size_t i=0; i < neq; ++i)
      {
        int dofId = getOverlapDOF(node_lid,i);
        v[i] = data[dofId];
      }
      apf::setVector(f,e,0,v);
    }
  }
  m->end(it);
  if ( ! overlapped)
    apf::synchronize(f);
}

template<class Output>
void AlbPUMI::FMDBDiscretization<Output>::setFieldT(const char* name, const Tpetra_Vector& dataT, bool overlapped)
{
  Teuchos::ArrayRCP<const ST> dataT_constView = dataT.get1dView();

  apf::Mesh* m = fmdbMeshStruct->apfMesh;
  apf::Field* f = m->findField(name);
  apf::MeshIterator* it = m->begin(0);
  apf::MeshEntity* e;
  while ((e = m->iterate(it)))
  {
    int node_gid = FMDB_Ent_ID(reinterpret_cast<pMeshEnt>(e));
    int node_lid = node_mapT->getLocalElement(node_gid);
    if (overlapped)
      node_lid = overlap_node_mapT->getLocalElement(node_gid);
    else
    {
      if ( ! m->isOwned(e)) continue;
      node_lid = node_mapT->getLocalElement(node_gid);
    }
    if (neq==1)
    {
      int dofId = getOverlapDOF(node_lid,0);
      apf::setScalar(f,e,0,dataT_constView[dofId]);
    }
    else
    { assert(neq==3);
      apf::Vector3 v;
      for (size_t i=0; i < neq; ++i)
      {
        int dofId = getOverlapDOF(node_lid,i);
        v[i] = dataT_constView[dofId];
      }
      apf::setVector(f,e,0,v);
    }
  }
  m->end(it);
  if ( ! overlapped)
    apf::synchronize(f);
}

template<class Output>
void AlbPUMI::FMDBDiscretization<Output>::setSplitFields(std::vector<std::string> names,
    std::vector<int> indices, const Epetra_Vector& data, bool overlapped)
{
  apf::Mesh* m = fmdbMeshStruct->apfMesh;
  apf::MeshIterator* it = m->begin(0);
  apf::MeshEntity* e;
  while ((e = m->iterate(it)))
  {
    int node_gid = FMDB_Ent_ID(reinterpret_cast<pMeshEnt>(e));
    int node_lid = node_mapT->getLocalElement(node_gid);
    if (overlapped)
      node_lid = overlap_node_mapT->getLocalElement(node_gid);
    else
    {
      if ( ! m->isOwned(e)) continue;
      node_lid = node_mapT->getLocalElement(node_gid);
    }
    int eq = 0;
    for (std::size_t i=0; i < names.size(); ++i)
    {
      apf::Field* f = m->findField(names[i].c_str());
      if (indices[i] == 1)
      {
        int dofId = getOverlapDOF(node_lid,eq);
        apf::setScalar(f,e,0,data[dofId]);
        eq += 1;
      }
      else
      {
        apf::Vector3 v;
        for (std::size_t j=0; j < indices[i]; ++j)
        {
          int dofId = getOverlapDOF(node_lid,eq);
          v[j] = data[dofId];
          eq += 1;
        }
        apf::setVector(f,e,0,v);
      }
    }
  }
  m->end(it);
  if  (!overlapped)
  {
    for (std::size_t i=0; i < names.size(); ++i)
    {
      apf::Field* f = m->findField(names[i].c_str());
      apf::synchronize(f);
    }
  }
}

template<class Output>
void AlbPUMI::FMDBDiscretization<Output>::setSplitFieldsT(std::vector<std::string> names,
    std::vector<int> indices, const Tpetra_Vector& dataT, bool overlapped)
{
  Teuchos::ArrayRCP<const ST> dataT_constView = dataT.get1dView();
  apf::Mesh* m = fmdbMeshStruct->apfMesh;
  apf::MeshIterator* it = m->begin(0);
  apf::MeshEntity* e;
  while ((e = m->iterate(it)))
  {
    int node_gid = FMDB_Ent_ID(reinterpret_cast<pMeshEnt>(e));
    int node_lid = node_mapT->getLocalElement(node_gid);
    if (overlapped)
      node_lid = overlap_node_mapT->getLocalElement(node_gid);
    else
    {
      if ( ! m->isOwned(e)) continue;
      node_lid = node_mapT->getLocalElement(node_gid);
    }
    int eq = 0;
    for (std::size_t i=0; i < names.size(); ++i)
    {
      apf::Field* f = m->findField(names[i].c_str());
      if (indices[i] == 1)
      {
        int dofId = getOverlapDOF(node_lid,eq);
        apf::setScalar(f,e,0,dataT_constView[dofId]);
        eq += 1;
      }
      else
      {
        apf::Vector3 v;
        for (std::size_t j=0; j < indices[i]; ++j)
        {
          int dofId = getOverlapDOF(node_lid,eq);
          v[j] = dataT_constView[dofId];
          eq += 1;
        }
        apf::setVector(f,e,0,v);
      }
    }
  }
  m->end(it);
  if  (!overlapped)
  {
    for (std::size_t i=0; i < names.size(); ++i)
    {
      apf::Field* f = m->findField(names[i].c_str());
      apf::synchronize(f);
    }
  }
}

template<class Output>
void AlbPUMI::FMDBDiscretization<Output>::getField(const char* name, Epetra_Vector& data, bool overlapped) const
{
  apf::Mesh* m = fmdbMeshStruct->apfMesh;
  apf::Field* f = m->findField(name);
  apf::MeshIterator* it = m->begin(0);
  apf::MeshEntity* e;
  while ((e = m->iterate(it)))
  {
    int node_gid = FMDB_Ent_ID(reinterpret_cast<pMeshEnt>(e));
    int node_lid = node_mapT->getLocalElement(node_gid);
    if (overlapped)
      node_lid = overlap_node_mapT->getLocalElement(node_gid);
    else
    {
      if ( ! m->isOwned(e)) continue;
      node_lid = node_mapT->getLocalElement(node_gid);
    }
    if (neq==1)
    {
      int dofId = getOverlapDOF(node_lid,0);
      data[dofId] = apf::getScalar(f,e,0);
    }
    else
    { assert(neq==3);
      apf::Vector3 v;
      apf::getVector(f,e,0,v);
      for (size_t i=0; i < neq; ++i)
      {
        int dofId = getOverlapDOF(node_lid,i);
        data[dofId] = v[i];
      }
    }
  }
  m->end(it);
}

template<class Output>
void AlbPUMI::FMDBDiscretization<Output>::getFieldT(const char* name, Tpetra_Vector& dataT, bool overlapped) const
{
  Teuchos::ArrayRCP<const ST> dataT_nonConstView = dataT.get1dViewNonConst();
  apf::Mesh* m = fmdbMeshStruct->apfMesh;
  apf::Field* f = m->findField(name);
  apf::MeshIterator* it = m->begin(0);
  apf::MeshEntity* e;
  while ((e = m->iterate(it)))
  {
    int node_gid = FMDB_Ent_ID(reinterpret_cast<pMeshEnt>(e));
    int node_lid = node_mapT->getLocalElement(node_gid);
    if (overlapped)
      node_lid = overlap_node_mapT->getLocalElement(node_gid);
    else
    {
      if ( ! m->isOwned(e)) continue;
      node_lid = node_mapT->getLocalElement(node_gid);
    }
    if (neq==1)
    {
      int dofId = getOverlapDOF(node_lid,0);
      dataT_nonConstView[dofId] = apf::getScalar(f,e,0);
    }
    else
    { assert(neq==3);
      apf::Vector3 v;
      apf::getVector(f,e,0,v);
      for (size_t i=0; i < neq; ++i)
      {
        int dofId = getOverlapDOF(node_lid,i);
        dataT_nonConstView[dofId] = v[i];
      }
    }
  }
  m->end(it);
}

template<class Output>
void AlbPUMI::FMDBDiscretization<Output>::getSplitFields(std::vector<std::string> names,
   std::vector<int> indices, Epetra_Vector& data, bool overlapped) const
{
  apf::Mesh* m = fmdbMeshStruct->apfMesh;
  apf::MeshIterator* it = m->begin(0);
  apf::MeshEntity* e;
  while ((e = m->iterate(it)))
  {
    int node_gid = FMDB_Ent_ID(reinterpret_cast<pMeshEnt>(e));
    int node_lid = node_mapT->getLocalElement(node_gid);
    if (overlapped)
      node_lid = overlap_node_mapT->getLocalElement(node_gid);
    else
    {
      if ( ! m->isOwned(e)) continue;
      node_lid = node_mapT->getLocalElement(node_gid);
    }
    int eq = 0;
    for (std::size_t i=0; i < names.size(); ++i)
    {
      apf::Field* f = m->findField(names[i].c_str());
      if (indices[i] == 1)
      {
        int dofId = getOverlapDOF(node_lid,eq);
        data[dofId] = apf::getScalar(f,e,0);
        eq += 1;
      }
      else
      {
        apf::Vector3 v;
        apf::getVector(f,e,0,v);
        for (std::size_t j=0; j < indices[j]; ++j)
        {
          int dofId = getOverlapDOF(node_lid,eq);
          data[dofId] = v[j];
          eq += 1;
        }
      }
    }
    m->end(it);
  }
}

template<class Output>
void AlbPUMI::FMDBDiscretization<Output>::getSplitFieldsT(std::vector<std::string> names,
   std::vector<int> indices, Tpetra_Vector& dataT, bool overlapped) const
{
  Teuchos::ArrayRCP<const ST> dataT_nonConstView = dataT.get1dViewNonConst();
  apf::Mesh* m = fmdbMeshStruct->apfMesh;
  apf::MeshIterator* it = m->begin(0);
  apf::MeshEntity* e;
  while ((e = m->iterate(it)))
  {
    int node_gid = FMDB_Ent_ID(reinterpret_cast<pMeshEnt>(e));
    int node_lid = node_mapT->getLocalElement(node_gid);
    if (overlapped)
      node_lid = overlap_node_mapT->getLocalElement(node_gid);
    else
    {
      if ( ! m->isOwned(e)) continue;
      node_lid = node_mapT->getLocalElement(node_gid);
    }
    int eq = 0;
    for (std::size_t i=0; i < names.size(); ++i)
    {
      apf::Field* f = m->findField(names[i].c_str());
      if (indices[i] == 1)
      {
        int dofId = getOverlapDOF(node_lid,eq);
        dataT_+nonConstView[dofId] = apf::getScalar(f,e,0);
        eq += 1;
      }
      else
      {
        apf::Vector3 v;
        apf::getVector(f,e,0,v);
        for (std::size_t j=0; j < indices[j]; ++j)
        {
          int dofId = getOverlapDOF(node_lid,eq);
          dataT_nonConstView[dofId] = v[j];
          eq += 1;
        }
      }
    }
    m->end(it);
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

  if (mapT->getComm()->getRank()==0) {
    *out << "Albany::FMDBDiscretization::writeSolution: writing time " << time;
    if (time_label != time) *out << " with label " << time_label;
    *out << " to index " <<out_step<<" in file "<<fmdbMeshStruct->outputFileName<< std::endl;
  }

  if (solNames.size() == 0)
    this->setField("solution",soln,overlapped);
  else
    this->setSplitFields(solNames,solIndex,soln,overlapped);

  fmdbMeshStruct->solutionInitialized = true;

  outputInterval = 0;

  copyQPStatesToAPF();
  meshOutput.writeFile(time_label);
  removeQPStatesFromAPF();

}

//Tpetra version of above
void Albany::FMDBDiscretization::writeSolutionT(const Tpetra_Vector& solnT, const double time, const bool overlapped){


  Teuchos::ArrayRCP<const ST> solnT_constView = solnT.get1dView();
  if (fmdbMeshStruct->outputFileName.empty()) 

    return;

  // Skip this write unless the proper interval has been reached

  if(outputInterval++ % fmdbMeshStruct->outputInterval)

    return;

  double time_label = monotonicTimeLabel(time);
  int out_step = 0;

  if (mapT->getComm()->getRank()==0) {
    *out << "Albany::FMDBDiscretization::writeSolutionT: writing time " << time;
    if (time_label != time) *out << " with label " << time_label;
    *out << " to index " <<out_step<<" in file "<<fmdbMeshStruct->outputFileName<< std::endl;
  }
  
  if (solNames.size() == 0)
    this->setFieldT("solution",solnT,overlapped);
  else
    this->setSplitFieldsT(solNames,solIndex,solnT,overlapped);

  fmdbMeshStruct->solutionInitialized = true;

  outputInterval = 0;

  copyQPStatesToAPF();
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

std::cout << "************************************************" << std::endl;
std::cout << "Writing mesh debug output! " << std::endl;
std::cout << "************************************************" << std::endl;
std::cout << std::endl;

  fmdbMeshStruct->solutionInitialized = true;

  copyQPStatesToAPF();
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

void 
Albany::FMDBDiscretization::setResidualFieldT(const Tpetra_Vector& residualT) 
{
  pPart part;
  FMDB_Mesh_GetPart(fmdbMeshStruct->getMesh(), 0, part);
  //get const (read-only) view of residualT
  Teuchos::ArrayRCP<const ST> resT_constView = residualT.get1dView();

  pPartEntIter node_it;
  pMeshEnt node;
  int owner_part_id, counter=0;
  double* res = new double[neq];
  // iterate over all vertices (nodes)
  int iterEnd = FMDB_PartEntIter_Init(part, FMDB_VERTEX, FMDB_ALLTOPO, node_it);
  while (!iterEnd)
  {
    iterEnd = FMDB_PartEntIter_GetNext(node_it, node);
    if(iterEnd) break; 
    // get node's owner part id and skip if not owned
    FMDB_Ent_GetOwnPartID(node, part, &owner_part_id);
    if (FMDB_Part_ID(part)!=owner_part_id) continue; 

    for (std::size_t j=0; j<neq; j++)
      res[j] = resT_constView[getOwnedDOF(FMDB_Ent_LocalID(node),j)]; 
    FMDB_Ent_SetDblArrTag (fmdbMeshStruct->getMesh(), node, fmdbMeshStruct->residual_field_tag, res, neq);
  }
  FMDB_PartEntIter_Del (node_it);
  delete [] res;
  FMDB_Tag_SyncPtn(fmdbMeshStruct->getMesh(), fmdbMeshStruct->residual_field_tag, FMDB_VERTEX);
}

template<class Output>
Teuchos::RCP<Epetra_Vector>
AlbPUMI::FMDBDiscretization<Output>::getSolutionField() const
{

  // Copy soln vector into solution field, one node at a time
  Teuchos::ArrayView<const int> indicesAV = mapT->getNodeElementList();
  int numElements = mapT->getNodeNumElements();
  Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(new Epetra_Map(-1, numElements, indicesAV.getRawPtr(), 0, *comm));
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

Teuchos::RCP<Tpetra_Vector>
Albany::FMDBDiscretization::getSolutionFieldT() const
{
  // Copy soln vector into solution field, one node at a time
  Teuchos::RCP<Tpetra_Vector> solnT = Teuchos::rcp(new Tpetra_Vector(mapT));
  if (fmdbMeshStruct->solutionInitialized) {
    if (solNames.size() == 0)
      this->getFieldT("solution",*solnT,/*overlapped=*/false);
    else
      this->getSplitFieldsT(solNames,solIndex,*solnT,/*overlapped=*/false);
  }
  else if ( ! PCU_Comm_Self())
    *out <<__func__<<": uninit field" << std::endl;

  return solnT;
}

/*****************************************************************/
/*** Private functions follow. These are just used in above code */
/*****************************************************************/

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
  // Loads member data:  ownednodes, numOwnedNodes, node_map, numGlobalNodes, map
  // maps for owned nodes and unknowns

  // get the first (0th) part handle on local process -- assumption: single part per process/mesh_instance
  pPart part;
  FMDB_Mesh_GetPart(fmdbMeshStruct->getMesh(), 0, part);

  // compute owned nodes
  pPartEntIter node_it;
  pMeshEnt node;
  Teuchos::Array<int> indicesT;
  //copy of indicesT in std::vector form
  std::vector<int> indices; 
  int owner_part_id;
  std::vector<pMeshEnt> owned_nodes;

  // iterate over all vertices (nodes) and save owned nodes
  int iterEnd = FMDB_PartEntIter_Init(part, FMDB_VERTEX, FMDB_ALLTOPO, node_it);
  while (!iterEnd)
  {
    iterEnd = FMDB_PartEntIter_GetNext(node_it, node);
    if(iterEnd) break;
    // get node's owner part id and skip if not owned
    FMDB_Ent_GetOwnPartID(node, part, &owner_part_id);
    if (FMDB_Part_ID(part)!=owner_part_id) continue;

    owned_nodes.push_back(node); // Save the local node
    indicesT.push_back(FMDB_Ent_ID(node));  // Save the global id of the note.
                                           
  }

  FMDB_PartEntIter_Del (node_it);
  numOwnedNodes = owned_nodes.size();
  node_mapT = Tpetra::createNonContigMapWithNode<LO, GO, KokkosNode> (indicesT(), commT, nodeT);

  numGlobalNodes = node_mapT->getMaxAllGlobalIndex() + 1;

  //IK, 2/10/14: TO DO - rewrite resizeLocalMap so it takes in commT and indicesT objects
  //Then the copying below will not be necessary 
  //copy indicesT into indices
  indices.size(indicesT.size()); 
  for (int i=0; i<indicesT.size(); i++) 
    indices[i] = indicesT[i]; 

  if(Teuchos::nonnull(fmdbMeshStruct->nodal_data_block))
    fmdbMeshStruct->nodal_data_block->resizeLocalMap(indices, *comm);

  indices.resize(numOwnedNodes * neq);
  //copy indices into indicesT 
  for (int i=0; i<indices.size(); i++) 
    indicesT[i] = indices[i]; 

  for (int i=0; i < numOwnedNodes; ++i)
    for (std::size_t j=0; j < neq; ++j)
      indicesT[getOwnedDOF(i,j)] = getGlobalDOF(FMDB_Ent_ID(owned_nodes[i]),j);

  mapT = Tpetra::createNonContigMapWithNode<LO, GO, KokkosNode> (indicesT(), commT, nodeT);

}

template<class Output>
void AlbPUMI::FMDBDiscretization<Output>::computeOverlapNodesAndUnknowns()
{

  Teuchos::Array<int> indicesT;
  std::vector<int> indices;

  // get the first (0th) part handle on local process -- assumption: single part per process/mesh_instance
  pPart part;
  FMDB_Mesh_GetPart(fmdbMeshStruct->getMesh(), 0, part);

  pPartEntIter node_it;
  pMeshEnt node;

  // get # all (owned, duplicate copied on part boundary and ghosted) nodes
  FMDB_Part_GetNumEnt (part, FMDB_VERTEX, FMDB_ALLTOPO, &numOverlapNodes);
  indicesT.resize(numOverlapNodes * neq);
  indices.resize(numOverlapNodes * neq); 

  // Allocate an array to hold the node coordinates for the nodes visible to this PE
  int mesh_dim;
  FMDB_Mesh_GetDim(fmdbMeshStruct->getMesh(), &mesh_dim);

  // get global id of all nodes
  int i=0, iterEnd = FMDB_PartEntIter_Init(part, FMDB_VERTEX, FMDB_ALLTOPO, node_it);
  while (!iterEnd)
  {
    iterEnd = FMDB_PartEntIter_GetNext(node_it, node);
    if(iterEnd) break;
    for (std::size_t j=0; j < neq; j++)
      indicesT[getOverlapDOF(i,j)] = getGlobalDOF(FMDB_Ent_ID(node),j);
    ++i;
  }

  overlap_mapT = Tpetra::createNonContigMapWithNode<LO, GO, KokkosNode> (indicesT(), commT, nodeT);


  // Set up tpetra map of node IDs
  indicesT.resize(numOverlapNodes);
//  iterEnd = FMDB_PartEntIter_Reset(node_it);
  iterEnd = PUMI_PartEntIter_Reset(node_it);
  i=0;
  while (!iterEnd)
  {
    iterEnd = FMDB_PartEntIter_GetNext(node_it, node);
    if(iterEnd) break; 
    indicesT[i] = FMDB_Ent_ID(node);
    i++;
  }
  FMDB_PartEntIter_Del (node_it);

  overlap_node_mapT = Tpetra::createNonContigMapWithNode<LO, GO, KokkosNode> (indicesT(), commT, nodeT);

  //IK, 2/10/14: TO DO -- rewrite resizeOverlapMap to take in commT and indicesT 
  for (int i=0; i<indicesT.size(); i++) 
    indices[i] = indicesT[i]; 

  if(Teuchos::nonnull(fmdbMeshStruct->nodal_data_block))
    fmdbMeshStruct->nodal_data_block->resizeOverlapMap(indices, *comm);

  for (int i=0; i<indices.size(); i++) 
    indicesT[i] = indices[i]; 

  coordinates.resize(3 * numOverlapNodes);

}


template<class Output>
void AlbPUMI::FMDBDiscretization<Output>::computeGraphs()
{
  // GAH: the following assumes all element blocks in the problem have the same
  // number of nodes per element and that the cell topologies are the same.

  // get mesh dimension and part handle
  int mesh_dim, counter=0;
  FMDB_Mesh_GetDim(fmdbMeshStruct->getMesh(), &mesh_dim);
  pPart part;
  FMDB_Mesh_GetPart(fmdbMeshStruct->getMesh(), 0, part);

  // let's get an element (region) to query element topology
  pPartEntIter elem_it;
  pMeshEnt elem;
  FMDB_PartEntIter_Init(part, FMDB_REGION, FMDB_ALLTOPO, elem_it);
  FMDB_PartEntIter_GetNext(elem_it, elem);
  FMDB_PartEntIter_Del(elem_it);

  // query element topology
  int elem_topology;
  FMDB_Ent_GetTopo(elem, &elem_topology);

  // query # nodes per element topology
  int nodes_per_element = FMDB_Topo_NumDownAdj(elem_topology, FMDB_VERTEX);

  // Loads member data:  overlap_graph, numOverlapodes, overlap_node_map, coordinates, graphs

  overlap_graphT = Teuchos::rcp(new Tpetra_CrsGraph(overlap_mapT, neq*nodes_per_element));

  // get cells
  std::vector<pMeshEnt> cells;
  pPartEntIter cell_it;
  pMeshEnt cell;
  int iterEnd = FMDB_PartEntIter_Init(part, FMDB_REGION, FMDB_ALLTOPO, cell_it);
  while (!iterEnd)
  {
    iterEnd = FMDB_PartEntIter_GetNext(cell_it, cell);
    if(iterEnd) break;
    cells.push_back(cell);
  }
  FMDB_PartEntIter_Del (cell_it);

  if (SCUTIL_CommRank()==0)
    *out <<__func__<<": "<<cells.size() << " elements on Proc 0 " << std::endl;

  int row, col;
  std::vector<pMeshEnt> rel;
  Teuchos::ArrayView<int> colAV;

  for (std::size_t i=0; i < cells.size(); i++) 
  {
    pMeshEnt current_element = cells[i];
    rel.clear();
    FMDB_Ent_GetAdj(current_element, FMDB_VERTEX, 1, rel);

    // loop over local nodes
    for (std::size_t j=0; j < rel.size(); j++)
    {
      pMeshEnt rowNode = rel[j];

      // loop over eqs
      for (std::size_t k=0; k < neq; k++)
      {
        row = getGlobalDOF(FMDB_Ent_ID(rowNode), k);
        for (std::size_t l=0; l < rel.size(); l++)
        {
          pMeshEnt colNode = rel[l];
          for (std::size_t m=0; m < neq; m++)
          {
            col = getGlobalDOF(FMDB_Ent_ID(colNode), m);
            colAV = Teuchos::arrayView(&col, 1);
            overlap_graphT->insertGlobalIndices(row, colAV);
          }
        }
      }
    }
  }

  overlap_graphT->fillComplete();

  // Create Owned graph by exporting overlap with known row map

  graphT = Teuchos::rcp(new Tpetra_CrsGraph(mapT, nonzeroesPerRow(neq)));

  // Create non-overlapped matrix using two maps and export object
  Teuchos::RCP<Tpetra_Export> exporterT = Teuchos::rcp(new Tpetra_Export(overlap_mapT, mapT));
  graphT->doExport(*overlap_graphT, *exporterT, Tpetra::INSERT);
  graphT->fillComplete();

}

template<class Output>
void AlbPUMI::FMDBDiscretization<Output>::computeWorksetInfo()
{
  int mesh_dim;
  FMDB_Mesh_GetDim(fmdbMeshStruct->getMesh(), &mesh_dim);

  pPart part;
  FMDB_Mesh_GetPart(fmdbMeshStruct->getMesh(), 0, part);

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
  pPartEntIter element_it;
  pMeshEnt element;
  int owner_part_id;
  std::map<pElemBlk, int> bucketMap;
  std::map<pElemBlk, int>::iterator buck_it;
  int bucket_counter = 0;

  int worksetSize = fmdbMeshStruct->worksetSize;

  // iterate over all elements
  int iterEnd = FMDB_PartEntIter_Init(part, mesh_dim, FMDB_ALLTOPO, element_it);

  while (!iterEnd) {

    iterEnd = FMDB_PartEntIter_GetNext(element_it, element);
    if(iterEnd) break;

    // get element owner's part id and skip element if not owned
    FMDB_Ent_GetOwnPartID(element, part, &owner_part_id);
    if (FMDB_Part_ID(part)!=owner_part_id) continue;

    // Get the element block that the element is in
    FMDB_Ent_GetGeomClas (element, elem_blk);

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

  for (int b=0; b < numBuckets; b++) {

    std::vector<pMeshEnt>& buck = buckets[b];
    wsElNodeEqID[b].resize(buck.size());
    wsElNodeID[b].resize(buck.size());
    coords[b].resize(buck.size());

    // i is the element index within bucket b

    for (std::size_t i=0; i < buck.size(); i++) {

      // Traverse all the elements in this bucket
      element = buck[i];

      // Now, save a map from element GID to workset on this PE
      elemGIDws[FMDB_Ent_ID(element)].ws = b;

      // Now, save a map from element GID to local id on this workset on this PE
      elemGIDws[FMDB_Ent_ID(element)].LID = i;

      // get adj nodes per element

      std::vector<pMeshEnt> rel;
      FMDB_Ent_GetAdj(element, FMDB_VERTEX, 1, rel);

      int owner_part_id, nodes_per_element = rel.size();
      wsElNodeEqID[b][i].resize(nodes_per_element);
      wsElNodeID[b][i].resize(nodes_per_element);
      coords[b][i].resize(nodes_per_element);

      // loop over local nodes

      for (int j=0; j < nodes_per_element; j++) {

        pMeshEnt rowNode = rel[j];

        int node_gid = FMDB_Ent_ID(rowNode);
        int node_lid = overlap_node_mapT->getLocalElement(node_gid);
        
        TEUCHOS_TEST_FOR_EXCEPTION(node_lid<0, std::logic_error,
			   "FMDB1D_Disc: node_lid out of range " << node_lid << std::endl);
        FMDB_Vtx_GetCoord (rowNode,
            &coordinates[node_lid * 3]); // Extract a pointer to the correct spot to begin placing coordinates

        coords[b][i][j] = &coordinates[node_lid * 3];
        wsElNodeEqID[b][i][j].resize(neq);
        wsElNodeID[b][i][j] = node_gid;

        for (std::size_t eq=0; eq < neq; eq++)
          wsElNodeEqID[b][i][j][eq] = getOverlapDOF(node_lid,eq);

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

    std::vector<pMeshEnt>& buck = buckets[b];

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

    std::vector< std::vector<pMeshEnt> > nbuckets; // bucket of nodes
    int numNodeBuckets =  (int)ceil((double)numOwnedNodes / (double)worksetSize);

    nbuckets.resize(numNodeBuckets);
    int node_bucket_counter = 0;
    int node_in_bucket = 0;

    // get the first (0th) part handle on local process -- assumption: single part per process/mesh_instance
    pPart part;
    FMDB_Mesh_GetPart(fmdbMeshStruct->getMesh(), 0, part);

    // compute buckets of nodes, each workset size
    pPartEntIter node_it;
    pMeshEnt node;
    int owner_part_id;

    // iterate over all vertices (nodes) and save owned nodes into buckets
    int iterEnd = FMDB_PartEntIter_Init(part, FMDB_VERTEX, FMDB_ALLTOPO, node_it);
    while (!iterEnd)
    {
      iterEnd = FMDB_PartEntIter_GetNext(node_it, node);
      if(iterEnd) break;
      // get node's owner part id and skip if not owned
      FMDB_Ent_GetOwnPartID(node, part, &owner_part_id);
      if (FMDB_Part_ID(part) != owner_part_id) continue;

      nbuckets[node_bucket_counter].push_back(node);
      node_in_bucket++;
      if(node_in_bucket >= worksetSize){
        node_bucket_counter++;
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
         std::vector<pMeshEnt>& buck = nbuckets[b];
         stateArrays.nodeStateArrays[b][(*nfs).first] = nodeContainer->getMDA(buck);
      }
    }
  }
}

static int getQPOrder(apf::Mesh* m, int nqp)
{
  apf::MeshIterator* it = m->begin(m->getDimension());
  apf::MeshEntity* e = m->iterate(it);
  m->end(it);
  int type = m->getType(e);
  if (type==apf::Mesh::TET)
  {
    if (nqp==5) return 3;
    if (nqp==1) return 1;
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
			"AlbPUMI::FMDBDiscretization getQPOrder unsupported type " << type << " and nqp " << nqp << std::endl);
}

template<class Output>
void AlbPUMI::FMDBDiscretization<Output>::copyQPScalarToAPF(
    unsigned nqp,
    QPData<double, 2>& state,
    apf::Field* f) {
  for (std::size_t b=0; b < buckets.size(); ++b) {
    std::vector<pMeshEnt>& buck = buckets[b];
    Albany::MDArray& ar = stateArrays.elemStateArrays[b][state.name];
    for (std::size_t e=0; e < buck.size(); ++e)
      for (std::size_t p=0; p < nqp; ++p)
        apf::setScalar(f,apf::castEntity(buck[e]),p,ar(e,p));
  }
}

template<class Output>
void AlbPUMI::FMDBDiscretization<Output>::copyQPVectorToAPF(
    unsigned nqp,
    QPData<double, 3>& state,
    apf::Field* f) {
  for (std::size_t b=0; b < buckets.size(); ++b) {
    std::vector<pMeshEnt>& buck = buckets[b];
    Albany::MDArray& ar = stateArrays.elemStateArrays[b][state.name];
    for (std::size_t e=0; e < buck.size(); ++e) {
      apf::Vector3 v;
      for (std::size_t p=0; p < nqp; ++p) {
        for (std::size_t i=0; i < 3; ++i)
          v[i] = ar(e,p,i);
        apf::setVector(f,apf::castEntity(buck[e]),p,v);
      }
    }
  }
}

template<class Output>
void AlbPUMI::FMDBDiscretization<Output>::copyQPTensorToAPF(
    unsigned nqp,
    QPData<double, 4>& state,
    apf::Field* f) {
  for (std::size_t b=0; b < buckets.size(); ++b) {
    std::vector<pMeshEnt>& buck = buckets[b];
    Albany::MDArray& ar = stateArrays.elemStateArrays[b][state.name];
    for (std::size_t e=0; e < buck.size(); ++e) {
      apf::Matrix3x3 v;
      for (std::size_t p=0; p < nqp; ++p) {
        for (std::size_t i=0; i < 3; ++i)
        for (std::size_t j=0; j < 3; ++j)
          v[i][j] = ar(e,p,i,j);
        apf::setMatrix(f,apf::castEntity(buck[e]),p,v);
      }
    }
  }
}

template<class Output>
void AlbPUMI::FMDBDiscretization<Output>::copyQPStatesToAPF() {
  apf::Mesh2* m = fmdbMeshStruct->apfMesh;
  int nqp = -1;
  int order;
  for (std::size_t i=0; i < fmdbMeshStruct->qpscalar_states.size(); ++i) {
    QPData<double, 2>& state = *(fmdbMeshStruct->qpscalar_states[i]);
    if (nqp == -1)
    {
      nqp = state.dims[1];
      order = getQPOrder(m,nqp);
    }
    apf::Field* f = m->findField(state.name.c_str());
    if(!f)
       f = apf::createIPField(m,state.name.c_str(),apf::SCALAR,order);
 //   else if(f->getValueType() != apf::SCALAR)
 //      f = apf::createIPField(m,state.name.c_str(),apf::SCALAR,order);
    copyQPScalarToAPF(nqp,state,f);
  }
  for (std::size_t i=0; i < fmdbMeshStruct->qpvector_states.size(); ++i) {
    QPData<double, 3>& state = *(fmdbMeshStruct->qpvector_states[i]);
    if (nqp == -1)
    {
      nqp = state.dims[1];
      order = getQPOrder(m,nqp);
    }
    apf::Field* f = m->findField(state.name.c_str());
    if(!f)
       f = apf::createIPField(m,state.name.c_str(),apf::VECTOR,order);
//    else if(f->getValueType() != apf::SCALAR)
//       f = apf::createIPField(m,state.name.c_str(),apf::VECTOR,order);
    copyQPVectorToAPF(nqp,state,f);
  }
  for (std::size_t i=0; i < fmdbMeshStruct->qptensor_states.size(); ++i) {
    QPData<double, 4>& state = *(fmdbMeshStruct->qptensor_states[i]);
    if (nqp == -1)
    {
      nqp = state.dims[1];
      order = getQPOrder(m,nqp);
    }
    apf::Field* f = m->findField(state.name.c_str());
    if(!f)
       f = apf::createIPField(m,state.name.c_str(),apf::MATRIX,order);
//    else if(f->getValueType() != apf::SCALAR)
//       f = apf::createIPField(m,state.name.c_str(),apf::MATRIX,order);
    copyQPTensorToAPF(nqp,state,f);
  }
}

template<class Output>
void AlbPUMI::FMDBDiscretization<Output>::removeQPStatesFromAPF() {
  apf::Mesh2* m = fmdbMeshStruct->apfMesh;
  for (std::size_t i=0; i < fmdbMeshStruct->qpscalar_states.size(); ++i) {
    QPData<double, 2>& state = *(fmdbMeshStruct->qpscalar_states[i]);
    apf::Field* f = m->findField(state.name.c_str());
    if(f)
      apf::destroyField(f);
  }
  for (std::size_t i=0; i < fmdbMeshStruct->qpvector_states.size(); ++i) {
    QPData<double, 3>& state = *(fmdbMeshStruct->qpvector_states[i]);
    apf::Field* f = m->findField(state.name.c_str());
    if(f)
      apf::destroyField(f);
  }
  for (std::size_t i=0; i < fmdbMeshStruct->qptensor_states.size(); ++i) {
    QPData<double, 4>& state = *(fmdbMeshStruct->qptensor_states[i]);
    apf::Field* f = m->findField(state.name.c_str());
    if(f)
      apf::destroyField(f);
  }
}

template<class Output>
void AlbPUMI::FMDBDiscretization<Output>::computeSideSets() {

  pMeshMdl mesh = fmdbMeshStruct->getMesh();
  pPart part;
  FMDB_Mesh_GetPart(mesh, 0, part);

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
#ifdef ALBANY_DEBUG
    std::cout<<"FMDBDisc: nodeset "<<ss_name<<" has size "<<ss_sides.size()<<"  on Proc "<<SCUTIL_CommRank()<<std::endl;
#endif

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

      sstruct.elem_GID = FMDB_Ent_ID(elem); // Global element ID
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
#ifdef ALBANY_DEBUG
    std::cout<<"["<<SCUTIL_CommRank()<<"] node set "<<*ns_iter<<std::endl;
#endif
    nodeSets[*ns_iter].resize(0);
    nodeSetCoords[*ns_iter].resize(0);
    nodeset_node_coords[*ns_iter].resize(0);
  }

  int mesh_dim;
  FMDB_Mesh_GetDim(fmdbMeshStruct->getMesh(), &mesh_dim);

  pPart part;
  FMDB_Mesh_GetPart(fmdbMeshStruct->getMesh(), 0, part);

  int owner_part_id;
  std::vector<pNodeSet> node_set;

  PUMI_Exodus_GetNodeSet (fmdbMeshStruct->getMesh(), node_set);

  for (std::vector<pNodeSet>::iterator node_set_it=node_set.begin(); node_set_it!=node_set.end(); ++node_set_it)
  {
    std::vector<pMeshEnt> node_set_nodes;
    PUMI_NodeSet_GetNode(fmdbMeshStruct->getMesh(), *node_set_it, node_set_nodes);
    // compute owned nodes
    std::vector<pMeshEnt> owned_ns_nodes;
    for (std::vector<pMeshEnt>::iterator node_it=node_set_nodes.begin(); node_it!=node_set_nodes.end(); ++node_it)
    {
      FMDB_Ent_GetOwnPartID(*node_it, part, &owner_part_id);

      // if the node is owned by the local part, save it
      if (FMDB_Part_ID(part)==owner_part_id)
        owned_ns_nodes.push_back(*node_it);
    }

    std::string NS_name;
    PUMI_NodeSet_GetName(*node_set_it, NS_name);
    nodeSets[NS_name].resize(owned_ns_nodes.size());
    nodeSetCoords[NS_name].resize(owned_ns_nodes.size());
    nodeset_node_coords[NS_name].resize(owned_ns_nodes.size() * mesh_dim);

#ifdef ALBANY_DEBUG
    std::cout << "FMDBDisc: nodeset "<< NS_name <<" has size " << owned_ns_nodes.size() << "  on Proc "<<SCUTIL_CommRank()<< std::endl;
#endif

    for (std::size_t i=0; i < owned_ns_nodes.size(); i++)
    {
      nodeSets[NS_name][i].resize(neq);
      for (std::size_t eq=0; eq < neq; eq++)
        nodeSets[NS_name][i][eq] = getOwnedDOF(FMDB_Ent_LocalID(owned_ns_nodes[i]), eq);
      FMDB_Vtx_GetCoord (owned_ns_nodes[i],
          &nodeset_node_coords[NS_name][i * mesh_dim]); // Extract a pointer to the correct spot to begin placing coordinates
      nodeSetCoords[NS_name][i] = &nodeset_node_coords[NS_name][i * mesh_dim];
    }
  }
}

template<class Output>
void
AlbPUMI::FMDBDiscretization<Output>::updateMesh()
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

}
