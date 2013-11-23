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
#define DEBUG 1

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
  meshOutput(fmdbMeshStruct_->outputFileName, fmdbMeshStruct_->getMesh(), comm_)
{
  int Count=1, PartialMins=SCUTIL_CommRank(), GlobalMins;

  MPI_Allreduce(&PartialMins, &GlobalMins, Count, MPI_INT, MPI_MIN, Albany::getMpiCommFromEpetraComm(*comm));

  std::cout<<"["<<SCUTIL_CommRank()<<"] PartialMins="<<PartialMins<<", GlobalMins="<<GlobalMins<<std::endl;

  AlbPUMI::FMDBDiscretization<Output>::updateMesh();

}

template<class Output>
AlbPUMI::FMDBDiscretization<Output>::~FMDBDiscretization()
{

  for (int i=0; i< toDelete.size(); i++) delete [] toDelete[i];

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
    int node_lid = overlap_node_map->LID(node_gid);
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

  // get the first (0th) part handle on local process -- assumption: single part per process/mesh_instance
  pPart part;
  FMDB_Mesh_GetPart(fmdbMeshStruct->getMesh(), 0, part);

  pPartEntIter node_it;
  pMeshEnt node;
  int owner_part_id, counter=0;
  double* sol = new double[neq];
  // iterate over all vertices (nodes)
//std::cout << " Writing solution for time step: " << time_label << std::endl;
  int iterEnd = FMDB_PartEntIter_Init(part, FMDB_VERTEX, FMDB_ALLTOPO, node_it);
  while (!iterEnd)
  {
    iterEnd = FMDB_PartEntIter_GetNext(node_it, node);
    if(iterEnd) break; 

    // get node's owner part id and skip if not owned
//    FMDB_Ent_GetOwnPartID(node, part, &owner_part_id);
//    if (FMDB_Part_ID(part)!=owner_part_id) continue; 

    for (std::size_t j=0; j<neq; j++){
      int local_id = overlap_map->LID(getOverlapDOF(FMDB_Ent_ID(node),j));
      sol[j] = soln[local_id];

    }

    FMDB_Ent_SetDblArrTag (fmdbMeshStruct->getMesh(), node, fmdbMeshStruct->solution_field_tag, sol, neq);
    ++counter;
  }

  FMDB_PartEntIter_Del (node_it);
  delete [] sol;
//  FMDB_Tag_SyncPtn(fmdbMeshStruct->getMesh(), fmdbMeshStruct->solution_field_tag, FMDB_VERTEX);

  outputInterval = 0;

  meshOutput.writeFile(time_label);

}

template<class Output>
void
AlbPUMI::FMDBDiscretization<Output>::debugMeshWrite(const Epetra_Vector& soln, const char* filename){

  // get the first (0th) part handle on local process -- assumption: single part per process/mesh_instance
  pPart part;
  FMDB_Mesh_GetPart(fmdbMeshStruct->getMesh(), 0, part);

  pPartEntIter node_it;
  pMeshEnt node;
  int owner_part_id, counter=0;
  double* sol = new double[neq];
  // iterate over all vertices (nodes)
//std::cout << " Writing solution for time step: " << time_label << std::endl;
  int iterEnd = FMDB_PartEntIter_Init(part, FMDB_VERTEX, FMDB_ALLTOPO, node_it);
  while (!iterEnd)
  {
    iterEnd = FMDB_PartEntIter_GetNext(node_it, node);
    if(iterEnd) break;

    // get node's owner part id and skip if not owned
    FMDB_Ent_GetOwnPartID(node, part, &owner_part_id);
    if (FMDB_Part_ID(part)!=owner_part_id) continue;

    for (std::size_t j=0; j<neq; j++){
//      int local_id = overlap_map->LID(getOverlapDOF(FMDB_Ent_ID(node),j));
//      indices[getOwnedDOF(i,j)] = getGlobalDOF(FMDB_Ent_ID(owned_nodes[i]),j);
      int local_id = map->LID(getOwnedDOF(FMDB_Ent_ID(node),j));
//std::cout << FMDB_Ent_ID(node) << " " << local_id << " " << soln[local_id] << std::endl;
      sol[j] = soln[local_id];
    }

    FMDB_Ent_SetDblArrTag (fmdbMeshStruct->getMesh(), node, fmdbMeshStruct->solution_field_tag, sol, neq);
    ++counter;
  }

  FMDB_PartEntIter_Del (node_it);
  FMDB_Tag_SyncPtn(fmdbMeshStruct->getMesh(), fmdbMeshStruct->solution_field_tag, FMDB_VERTEX);

  FMDB_Mesh_WriteToFile (fmdbMeshStruct->getMesh(), filename,  (SCUTIL_CommSize()>1?1:0));

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
  pPart part;
  FMDB_Mesh_GetPart(fmdbMeshStruct->getMesh(), 0, part);

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
      res[j] = residual[getOwnedDOF(FMDB_Ent_LocalID(node),j)]; 
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
  Teuchos::RCP<Epetra_Vector> soln = Teuchos::rcp(new Epetra_Vector(*map));

  // get the first (0th) part handle on local process -- assumption: single part per process/mesh_instance
  pPart part;
  FMDB_Mesh_GetPart(fmdbMeshStruct->getMesh(), 0, part);

  pPartEntIter node_it;
  pMeshEnt node;
  int owner_part_id, sol_size;
  double* sol = new double[neq];
  // iterate over all vertices (nodes)
  int iterEnd = FMDB_PartEntIter_Init(part, FMDB_VERTEX, FMDB_ALLTOPO, node_it);
  while (!iterEnd)
  {
    iterEnd = FMDB_PartEntIter_GetNext(node_it, node);
    if(iterEnd) break; 
    // get node's owner part id and skip if not owned
    FMDB_Ent_GetOwnPartID(node, part, &owner_part_id);
    if (FMDB_Part_ID(part)!=owner_part_id) continue; 

    FMDB_Ent_GetDblArrTag (fmdbMeshStruct->getMesh(), node, fmdbMeshStruct->solution_field_tag, &sol, &sol_size);
    for (std::size_t j=0; j<neq; j++)
      (*soln)[getOwnedDOF(FMDB_Ent_LocalID(node),j)] = sol[j]; 
  }
  FMDB_PartEntIter_Del (node_it);
  delete [] sol;

  return soln;
}

/*****************************************************************/
/*** Private functions follow. These are just used in above code */
/*****************************************************************/

template<class Output>
void 
AlbPUMI::FMDBDiscretization<Output>::setSolutionField(const Epetra_Vector& soln) 
{
  // get the first (0th) part handle on local process -- assumption: single part per process/mesh_instance
  pPart part;
  FMDB_Mesh_GetPart(fmdbMeshStruct->getMesh(), 0, part);

  pPartEntIter node_it;
  pMeshEnt node;
  int owner_part_id;
  double* sol = new double[neq];
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
      sol[j] = soln[getOwnedDOF(FMDB_Ent_LocalID(node),j)];  
    FMDB_Ent_SetDblArrTag (fmdbMeshStruct->getMesh(), node, fmdbMeshStruct->solution_field_tag, sol, neq);
  }
  FMDB_PartEntIter_Del (node_it);
  delete [] sol;
  FMDB_Tag_SyncPtn(fmdbMeshStruct->getMesh(), fmdbMeshStruct->solution_field_tag, FMDB_VERTEX);
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
    indices.push_back(FMDB_Ent_ID(node));  // Save the global id of the note.
                                           
  }

  FMDB_PartEntIter_Del (node_it);

  numOwnedNodes = owned_nodes.size();
  node_map = Teuchos::rcp(new Epetra_Map(-1, numOwnedNodes,
					 &(indices[0]), 0, *comm));

  numGlobalNodes = node_map->MaxAllGID() + 1;

  indices.resize(numOwnedNodes * neq);
  for (int i=0; i < numOwnedNodes; ++i)
    for (std::size_t j=0; j < neq; ++j)
      indices[getOwnedDOF(i,j)] = getGlobalDOF(FMDB_Ent_ID(owned_nodes[i]),j);

  map = Teuchos::rcp(new Epetra_Map(-1, indices.size(), &(indices[0]), 0, *comm));
}

template<class Output>
void AlbPUMI::FMDBDiscretization<Output>::computeOverlapNodesAndUnknowns()
{
  std::vector<int> indices;

  // get the first (0th) part handle on local process -- assumption: single part per process/mesh_instance
  pPart part;
  FMDB_Mesh_GetPart(fmdbMeshStruct->getMesh(), 0, part);

  pPartEntIter node_it;
  pMeshEnt node;

  // get # all (owned, duplicate copied on part boundary and ghosted) nodes
  FMDB_Part_GetNumEnt (part, FMDB_VERTEX, FMDB_ALLTOPO, &numOverlapNodes);
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
      indices[getOverlapDOF(i,j)] = getGlobalDOF(FMDB_Ent_ID(node),j);
    ++i;
  }

  overlap_map = Teuchos::rcp(new Epetra_Map(-1, indices.size(),
					    &(indices[0]), 0, *comm));


  // Set up epetra map of node IDs
  indices.resize(numOverlapNodes);
//  iterEnd = FMDB_PartEntIter_Reset(node_it);
  iterEnd = PUMI_PartEntIter_Reset(node_it);
  i=0;
  while (!iterEnd)
  {
    iterEnd = FMDB_PartEntIter_GetNext(node_it, node);
    if(iterEnd) break; 
    indices[i] = FMDB_Ent_ID(node);
    i++;
  }
  FMDB_PartEntIter_Del (node_it);

  overlap_node_map = Teuchos::rcp(new Epetra_Map(-1, indices.size(),
						 &(indices[0]), 0, *comm));
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

  overlap_graph =
    Teuchos::rcp(new Epetra_CrsGraph(Copy, *overlap_map,
                                     neq*nodes_per_element, false));

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
            overlap_graph->InsertGlobalIndices(row, 1, &col);
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

// GAH here: need to recalculate workset size now that the mesh has changed???

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
  coords.resize(numBuckets);
  sHeight.resize(numBuckets);
  temperature.resize(numBuckets);
  basalFriction.resize(numBuckets);
  thickness.resize(numBuckets);
  flowFactor.resize(numBuckets);
  surfaceVelocity.resize(numBuckets);
  velocityRMS.resize(numBuckets);

  for (int b=0; b < numBuckets; b++) {

    std::vector<pMeshEnt>& buck = buckets[b];
    wsElNodeEqID[b].resize(buck.size());
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
      coords[b][i].resize(nodes_per_element);

      // loop over local nodes

      for (int j=0; j < nodes_per_element; j++) {

        pMeshEnt rowNode = rel[j];

        int node_gid = FMDB_Ent_ID(rowNode);
        int node_lid = overlap_node_map->LID(node_gid);
        
        TEUCHOS_TEST_FOR_EXCEPTION(node_lid<0, std::logic_error,
			   "FMDB1D_Disc: node_lid out of range " << node_lid << std::endl);
        FMDB_Vtx_GetCoord (rowNode, 
            &coordinates[node_lid * 3]); // Extract a pointer to the correct spot to begin placing coordinates

        coords[b][i][j] = &coordinates[node_lid * 3];
        wsElNodeEqID[b][i][j].resize(neq);

        for (std::size_t eq=0; eq < neq; eq++) 
          wsElNodeEqID[b][i][j][eq] = getOverlapDOF(node_lid,eq);

      }
    }
  }

  // Pull out pointers to shards::Arrays for every bucket, for every state

  stateArrays.resize(numBuckets);

  for (std::size_t b=0; b < buckets.size(); b++) {

    std::vector<pMeshEnt>& buck = buckets[b];

    for (std::size_t i=0; i<fmdbMeshStruct->qpscalar_states.size(); i++) {
//      stk::mesh::BucketArray<AlbPUMI::FMDBMeshStruct::QPScalarFieldType> array(*fmdbMeshStruct->qpscalar_states[i], buck);
//      MDArray ar = array;
//      MDArray ar = *fmdbMeshStruct->qpscalar_states[i];
//      stateArrays[b][fmdbMeshStruct->qpscalar_name[i]] = ar;
//      stateArrays[b][fmdbMeshStruct->qpscalar_states[i]->name] = fmdbMeshStruct->qpscalar_states[i]->allocateArray(buck.size());
      Albany::MDArray ar = *fmdbMeshStruct->qpscalar_states[i]->allocateArray(buck.size());
      stateArrays[b][fmdbMeshStruct->qpscalar_states[i]->name] = ar;
    }
    for (std::size_t i=0; i<fmdbMeshStruct->qpvector_states.size(); i++) {
//      stk::mesh::BucketArray<AlbPUMI::FMDBMeshStruct::QPVectorFieldType> array(*fmdbMeshStruct->qpvector_states[i], buck);
//      MDArray ar = array;
//      MDArray ar = *fmdbMeshStruct->qpvector_states[i];
//      stateArrays[b][fmdbMeshStruct->qpvector_name[i]] = ar;
//      stateArrays[b][fmdbMeshStruct->qpvector_states[i]->name] = fmdbMeshStruct->qpvector_states[i]->allocateArray(buck.size());
      Albany::MDArray ar = *fmdbMeshStruct->qpvector_states[i]->allocateArray(buck.size());
      stateArrays[b][fmdbMeshStruct->qpvector_states[i]->name] = ar;
    }
    for (std::size_t i=0; i<fmdbMeshStruct->qptensor_states.size(); i++) {
//      stk::mesh::BucketArray<AlbPUMI::FMDBMeshStruct::QPTensorFieldType> array(*fmdbMeshStruct->qptensor_states[i], buck);
//      MDArray ar = array;
//      MDArray ar = *fmdbMeshStruct->qptensor_states[i];
//      stateArrays[b][fmdbMeshStruct->qptensor_name[i]] = ar;
//      stateArrays[b][fmdbMeshStruct->qptensor_states[i]->name] = fmdbMeshStruct->qptensor_states[i]->allocateArray(buck.size());
      Albany::MDArray ar = *fmdbMeshStruct->qptensor_states[i]->allocateArray(buck.size());
      stateArrays[b][fmdbMeshStruct->qptensor_states[i]->name] = ar;
    }    
    for (std::size_t i=0; i<fmdbMeshStruct->scalarValue_states.size(); i++) {      
      const int size = 1;
//      shards::Array<double, shards::NaturalOrder, Cell> array(&(fmdbMeshStruct->time), size);
//      MDArray ar = array;
//      stateArrays[b][fmdbMeshStruct->scalarValue_states[i]] = ar;
//      stateArrays[b][fmdbMeshStruct->scalarValue_states[i]->name] = fmdbMeshStruct->scalarValue_states[i]->allocateArray(size);
      Albany::MDArray ar = *fmdbMeshStruct->scalarValue_states[i]->allocateArray(size);
      stateArrays[b][fmdbMeshStruct->scalarValue_states[i]->name] = ar;
    }
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
    std::cout<<"FMDBDisc: nodeset "<<ss_name<<" has size "<<ss_sides.size()<<"  on Proc "<<SCUTIL_CommRank()<<std::endl;
    
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
    std::cout<<"["<<SCUTIL_CommRank()<<"] node set "<<*ns_iter<<std::endl;
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

    std::cout << "FMDBDisc: nodeset "<< NS_name <<" has size " << owned_ns_nodes.size() << "  on Proc "<<SCUTIL_CommRank()<< std::endl;
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
  std::cout<<"["<<SCUTIL_CommRank()<<"] "<<__func__<<": computeOwnedNodesAndUnknowns() completed\n";

  computeOverlapNodesAndUnknowns();
  std::cout<<"["<<SCUTIL_CommRank()<<"] "<<__func__<<": computeOverlapNodesAndUnknowns() completed\n";

  computeGraphs();
  std::cout<<"["<<SCUTIL_CommRank()<<"] "<<__func__<<": computeGraphs() completed\n";

  computeWorksetInfo();
  std::cout<<"["<<SCUTIL_CommRank()<<"] "<<__func__<<": computeWorksetInfo() completed\n";

  computeNodeSets();
  std::cout<<"["<<SCUTIL_CommRank()<<"] "<<__func__<<": computeNodeSets() completed\n";

  computeSideSets();
  std::cout<<"["<<SCUTIL_CommRank()<<"] "<<__func__<<": computeSideSets() completed\n";

}
