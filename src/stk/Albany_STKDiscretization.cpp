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


#include <limits>
#include "Epetra_Export.h"

#include "Albany_Utils.hpp"
#include "Albany_STKDiscretization.hpp"
#include <iostream>

#include <Shards_BasicTopologies.hpp>
#include "Shards_CellTopology.hpp"
#include "Shards_CellTopologyData.h"

#include <stk_util/parallel/Parallel.hpp>

#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/FieldData.hpp>
#include <stk_mesh/base/Selector.hpp>

#include <stk_mesh/fem/FEMHelpers.hpp>

#ifdef ALBANY_IOSS
  #include <Ionit_Initializer.h>
#endif 

Albany::STKDiscretization::STKDiscretization(
     Teuchos::RCP<Albany::AbstractSTKMeshStruct> stkMeshStruct_,
     const Teuchos::RCP<const Epetra_Comm>& comm) :
  neq(stkMeshStruct_->neq),
  stkMeshStruct(stkMeshStruct_),
  time(0.0)
{
  //Unpack mesh data from struct container
  stk::mesh::fem::FEMMetaData& metaData = *stkMeshStruct->metaData;
  stk::mesh::BulkData& bulkData = *stkMeshStruct->bulkData;
  //Albany::AbstractSTKMeshStruct::VectorFieldType& coordinates_field
  //      = *stkMeshStruct->coordinates_field;
  //Albany::AbstractSTKMeshStruct::VectorFieldType& solution_field
  //      = *stkMeshStruct->solution_field;

  stk::mesh::Part& universalPart = metaData.universal_part();

  //Teuchos::RCP<Epetra_Map>& elem_map = stkMeshStruct->elem_map;
  int& numDim = stkMeshStruct->numDim;

  if (stkMeshStruct->useElementAsTopRank) {
    nodes_per_element =  metaData.get_cell_topology(*(stkMeshStruct->partVec[0])).getNodeCount(); 
    cout << " SSS " <<   metaData.get_cell_topology(*(stkMeshStruct->partVec[0])).getName() << endl;
   } 
  else {  // comes from cubit
    if (numDim==1)  nodes_per_element = 2;   //can't get topology from Cubit
    else if (numDim==2)  nodes_per_element = 4;   //can't get topology from Cubit
    else             nodes_per_element = 8;
  }

 // Constructs overlap_map, overlap_graph, and graph.
  int row, col;
  //STK version
  // maps for owned nodes and unknowns
  stk::mesh::Selector select_owned_in_part =
    stk::mesh::Selector( universalPart ) &
    stk::mesh::Selector( metaData.locally_owned_part() );

  stk::mesh::get_selected_entities( select_owned_in_part ,
                           bulkData.buckets( metaData.node_rank() ) ,
                           ownednodes );

  int numOwned = ownednodes.size();
  std::vector<int> indices(numOwned);
  for (int i=0; i < numOwned; i++) indices[i] = ownednodes[i]->identifier() - 1;

  node_map = Teuchos::rcp(new Epetra_Map(-1, numOwned,
                              &(indices[0]), 0, *comm));

  indices.resize(numOwned * neq);
  for (int i=0; i < numOwned; i++)
    for (unsigned int j=0; j < neq; j++)
      indices[i*neq+j] = getDOF(*ownednodes[i],j);

  map = Teuchos::rcp(new Epetra_Map(-1, indices.size(),
                              &(indices[0]), 0, *comm));


  // maps for overlap unknowns
  stk::mesh::Selector select_overlap_in_part =
    stk::mesh::Selector( universalPart ) &
    ( stk::mesh::Selector( metaData.locally_owned_part() )
     | stk::mesh::Selector( metaData.globally_shared_part() ) );


  //  overlapnodes used for overlap map -- stored for changing coords
  stk::mesh::get_selected_entities( select_overlap_in_part ,
                           bulkData.buckets( metaData.node_rank() ) ,
                           overlapnodes );

  indices.resize(overlapnodes.size() * neq);
  for (unsigned int i=0; i < overlapnodes.size(); i++)
    for (unsigned int j=0; j < neq; j++)
      indices[i*neq+j] = getDOF(*overlapnodes[i],j);

  overlap_map = Teuchos::rcp(new Epetra_Map(-1, indices.size(),
                              &(indices[0]), 0, *comm));

  coordinates.resize(3*overlapnodes.size());

  // create overlap graph
  overlap_graph =
    Teuchos::rcp(new Epetra_CrsGraph(Copy, *overlap_map,
                                     neq*nodes_per_element, false));

  //std::vector< stk::mesh::Entity * > cells ;
  stk::mesh::get_selected_entities( select_owned_in_part ,
                           bulkData.buckets( metaData.element_rank() ) ,
                           cells );


  std::vector< stk::mesh::Bucket * > buckets ;
  stk::mesh::get_buckets( select_owned_in_part ,
                          bulkData.buckets( metaData.element_rank() ) ,
                          buckets);

  int numBuckets =  buckets.size();


  if (comm->MyPID()==0)
    cout << "STKDisc: " << cells.size() << " elements on Proc 0 " << endl;

  for (unsigned int i=0; i < cells.size(); i++) {
    stk::mesh::Entity& e = *cells[i];
    stk::mesh::PairIterRelation rel = e.relations();

    // loop over local nodes
    for (unsigned int j=0; j < rel.size(); j++) {
      stk::mesh::Entity& rowNode = * rel[j].entity();

      // loop over eqs
      for (unsigned int k=0; k < neq; k++) {
        row = getDOF(rowNode, k);
        for (unsigned int l=0; l < rel.size(); l++) {
          stk::mesh::Entity& colNode = * rel[l].entity();
          for (unsigned int m=0; m < neq; m++) {
            col = getDOF(colNode, m);
            overlap_graph->InsertGlobalIndices(row, 1, &col);
          }
        }
      }
    }
  }
  overlap_graph->FillComplete();

  // Fill  elNodeID(el_LID, local node) => node_LID
  elNodeID.resize(cells.size());
  int el_lid=0;
for (unsigned int b=0; b < buckets.size(); b++) {
  stk::mesh::Bucket& buck = *buckets[b];
  for (unsigned int i=0; i < buck.size(); i++, el_lid++) {

    stk::mesh::Entity& e = buck[i];

    stk::mesh::PairIterRelation rel = e.relations();

    elNodeID[el_lid].resize(nodes_per_element);
    // loop over local nodes
    for (unsigned int j=0; j < rel.size(); j++) {
      stk::mesh::Entity& rowNode = * rel[j].entity();
      int node_gid = rowNode.identifier() - 1;
      int node_lid = overlap_map->LID(node_gid * neq) / neq;
      TEST_FOR_EXCEPTION(node_lid<0, std::logic_error,
          "STK1D_Disc: node_lid out of range " << node_lid << endl);
      elNodeID[el_lid][j] = node_lid;
    }
  }
}

  int estNonzeroesPerRow;
  switch (numDim) {
    case 0: estNonzeroesPerRow=1*neq; break;
    case 1: estNonzeroesPerRow=3*neq; break;
    case 2: estNonzeroesPerRow=9*neq; break;
    case 3: estNonzeroesPerRow=27*neq; break;
    default: TEST_FOR_EXCEPTION(true, std::logic_error,
        "STKDiscretization:  Bad numDim"<< numDim);
  }
  graph = Teuchos::rcp(new Epetra_CrsGraph(Copy, *map, estNonzeroesPerRow, false));

  // Create non-overlapped matrix using two maps and export object
  Epetra_Export exporter(*overlap_map, *map);
  graph->Export(*overlap_graph, exporter, Insert);
  graph->FillComplete();

  // STK: NodeSets

  std::map<std::string, stk::mesh::Part*>::iterator ns = stkMeshStruct->nsPartVec.begin();
  while ( ns != stkMeshStruct->nsPartVec.end() ) { // Iterate over Node Sets
    // Get all owned nodes in this node set
    stk::mesh::Selector select_owned_in_nspart =
      stk::mesh::Selector( *(ns->second) ) &
      stk::mesh::Selector( metaData.locally_owned_part() );

    std::vector< stk::mesh::Entity * > nodes ;
    stk::mesh::get_selected_entities( select_owned_in_nspart ,
                             bulkData.buckets( metaData.node_rank() ) ,
                             nodes );

    nodeSets[ns->first].resize(nodes.size());
    nodeSetIDs.push_back(ns->first); // Grab string ID
    if (comm->MyPID()==0)
      cout << "STKDisc: nodeset "<< ns->first <<" has size " << nodes.size() << "  on Proc 0." << endl;
    for (unsigned int i=0; i < nodes.size(); i++) nodeSets[ns->first][i] = nodes[i]->identifier() - 1;
    ns++;
  }

#ifdef ALBANY_IOSS
  if (stkMeshStruct->exoOutput) {
    if (stkMeshStruct->numDim > 1) {

      Ioss::Init::Initializer io;
      mesh_data = new stk::io::util::MeshData();
      stk::io::util::create_output_mesh(
            stkMeshStruct->exoOutFile, "", "",
            Albany::getMpiCommFromEpetraComm(*comm),
            bulkData, metaData, *mesh_data);
    }
    else {
      cout << "\nWARNING: Exodus output for 1D Meshes not implemented:"
           << " Disabling output \n" << endl;
      stkMeshStruct->exoOutput = false;
    }
  }
#else
  if (stkMeshStruct->exoOutput) 
    cout << "\nWARNING: exodus output requested but IOSS not compiled in:"
         << " disabling exodus output \n" << endl;
  
#endif


  const stk::mesh::PartVector& pv = metaData.get_parts();
  for (unsigned int i=0; i < pv.size(); i++)
    if (pv[i]->primary_entity_rank() == metaData.element_rank())
     if (pv[i]->name()[0] != '{')
      cout << "QQQQ " << pv[i]->name() << endl;

}

Albany::STKDiscretization::~STKDiscretization()
{
#ifdef ALBANY_IOSS
  if (stkMeshStruct->exoOutput) delete mesh_data;
#endif
}

	    
Teuchos::RCP<const Epetra_Map>
Albany::STKDiscretization::getMap() const
{
  return map;
}

Teuchos::RCP<const Epetra_Map>
Albany::STKDiscretization::getOverlapMap() const
{
  return overlap_map;
}

Teuchos::RCP<const Epetra_CrsGraph>
Albany::STKDiscretization::getJacobianGraph() const
{
  return graph;
}

Teuchos::RCP<const Epetra_CrsGraph>
Albany::STKDiscretization::getOverlapJacobianGraph() const
{
  return overlap_graph;
}

Teuchos::RCP<const Epetra_Map>
Albany::STKDiscretization::getNodeMap() const
{
  return node_map;
}

const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >&
Albany::STKDiscretization::getElNodeID() const
{
  return elNodeID;
}

Teuchos::ArrayRCP<double>& 
Albany::STKDiscretization::getCoordinates() const
{
  // Coordinates are computed here, and not precomputed,
  // since the mesh can move in shape opt problems
  for (unsigned int i=0; i < overlapnodes.size(); i++)  {
      int node_gid = overlapnodes[i]->identifier() - 1;
      int node_lid = overlap_map->LID(node_gid * neq) / neq;
    double* x = stk::mesh::field_data(*stkMeshStruct->coordinates_field, *overlapnodes[i]);
    for (int dim=0; dim<stkMeshStruct->numDim; dim++)
      coordinates[3*node_lid + dim] = x[dim];
  }

  return coordinates;
}

inline int Albany::STKDiscretization::getDOF(stk::mesh::Entity& node, int eq) const
{ return (node.identifier()-1)*neq + eq; }

const std::vector<std::string>&
Albany::STKDiscretization::getNodeSetIDs() const
{
  return nodeSetIDs;
}

const CellTopologyData&
Albany::STKDiscretization::getCellTopologyData() const
{
  if (stkMeshStruct->useElementAsTopRank)
    return *(stkMeshStruct->metaData->get_cell_topology(*(stkMeshStruct->partVec[0])).getCellTopologyData()); 
  else { // havn't figured out how to get shards topo from Cubit
   if (stkMeshStruct->numDim==1) return *(shards::getCellTopologyData<shards::Line<2> >());
   else if (stkMeshStruct->numDim==2) return *(shards::getCellTopologyData<shards::Quadrilateral<4> >());
   else                          return *(shards::getCellTopologyData<shards::Hexahedron<8> >());
  }
}

void Albany::STKDiscretization::outputToExodus(const Epetra_Vector& soln,
					       const std::vector<std::vector<double> > states)
{
  // Put solution as Epetra_Vector into STK Mesh
  setSolutionField(soln);
 
  if (states.size()>0) setStateField(states);

#ifdef ALBANY_IOSS
  if (stkMeshStruct->exoOutput) {

    time+=1.0;

    int out_step = stk::io::util::process_output_request(
                   *mesh_data, *stkMeshStruct->bulkData, time);

    if (map->Comm().MyPID()==0)
      cout << "Albany::STKDiscretization::outputToExodus: writing time " << time 
           << " index " <<out_step<<" to file "<<stkMeshStruct->exoOutFile<< endl;
  }
#endif
}

void 
Albany::STKDiscretization::setStateField(const std::vector<std::vector<double> > states) 
{
  for (unsigned int i=0; i < cells.size(); i++)  {
    double* st = stk::mesh::field_data(*stkMeshStruct->state_field, *cells[i]);
    for (unsigned int j=0; j<stkMeshStruct->nstates; j++)
      st[j] = states[i][j];
  }
}

void 
Albany::STKDiscretization::setSolutionField(const Epetra_Vector& soln) 
{
  // Copy soln vector into solution field, one node at a time
  for (unsigned int i=0; i < ownednodes.size(); i++)  {
    int soln_gid = getDOF(*ownednodes[i], 0);;
    int soln_lid = soln.Map().LID(soln_gid);
    double* sol = stk::mesh::field_data(*stkMeshStruct->solution_field, *ownednodes[i]);
    for (unsigned int j=0; j<neq; j++)
      sol[j] = soln[soln_lid + j];
  }
}

void 
Albany::STKDiscretization::setResidualField(const Epetra_Vector& residual) 
{
  // Copy residual vector into residual field, one node at a time
  for (unsigned int i=0; i < ownednodes.size(); i++)  
  {
    int res_gid = getDOF(*ownednodes[i], 0);
    int res_lid = residual.Map().LID(res_gid);
    double* res = stk::mesh::field_data(*stkMeshStruct->residual_field, *ownednodes[i]);
    for (unsigned int j=0; j<neq; j++)
      res[j] = residual[res_lid + j];
  }
}

Teuchos::RCP<Epetra_Vector>
Albany::STKDiscretization::getSolutionField() const
{
  // Copy soln vector into solution field, one node at a time
  Teuchos::RCP<Epetra_Vector> soln = Teuchos::rcp(new Epetra_Vector(*map));
  for (unsigned int i=0; i < ownednodes.size(); i++)  {
    const int soln_gid = getDOF(*ownednodes[i], 0);;
    const int soln_lid = soln->Map().LID(soln_gid);
    const double* sol = stk::mesh::field_data(*stkMeshStruct->solution_field, *ownednodes[i]);
    for (unsigned int j=0; j<neq; j++) (*soln)[soln_lid + j] = sol[j];
  }
  return soln;
}
