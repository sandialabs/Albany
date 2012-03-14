/********************************************************************\
*
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
#include <string>
#include <iostream>
#include <fstream>

#include <Shards_BasicTopologies.hpp>
#include "Shards_CellTopology.hpp"
#include "Shards_CellTopologyData.h"

#include <stk_util/parallel/Parallel.hpp>

#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/FieldData.hpp>
#include <stk_mesh/base/Selector.hpp>

//#include <Intrepid_FieldContainer.hpp>
#include <PHAL_Dimension.hpp>

#include <stk_mesh/fem/FEMHelpers.hpp>

#ifdef ALBANY_SEACAS
#include <Ionit_Initializer.h>
#endif 

Albany::STKDiscretization::STKDiscretization(Teuchos::RCP<Albany::AbstractSTKMeshStruct> stkMeshStruct_,
					     const Teuchos::RCP<const Epetra_Comm>& comm_) :
  out(Teuchos::VerboseObjectBase::getDefaultOStream()),
  previous_time_label(-1.0e32),
  metaData(*stkMeshStruct_->metaData),
  bulkData(*stkMeshStruct_->bulkData),
  comm(comm_),
  neq(stkMeshStruct_->neq),
  stkMeshStruct(stkMeshStruct_),
  interleavedOrdering(stkMeshStruct_->interleavedOrdering),
  allocated_xyz(false)
{
  Albany::STKDiscretization::updateMesh(stkMeshStruct,comm);
}

Albany::STKDiscretization::~STKDiscretization()
{
#ifdef ALBANY_SEACAS
  if (stkMeshStruct->exoOutput) delete mesh_data;
#endif
  if (allocated_xyz) { delete [] xx; delete [] yy; delete [] zz; delete [] rr; allocated_xyz=false;} 
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

const Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > > >&
Albany::STKDiscretization::getWsElNodeEqID() const
{
  return wsElNodeEqID;
}

const Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >&
Albany::STKDiscretization::getCoords() const
{
  return coords;
}

Teuchos::ArrayRCP<double>& 
Albany::STKDiscretization::getCoordinates() const
{
  // Coordinates are computed here, and not precomputed,
  // since the mesh can move in shape opt problems
  for (int i=0; i < numOverlapNodes; i++)  {
    int node_gid = gid(overlapnodes[i]);
    int node_lid = overlap_node_map->LID(node_gid);

    double* x = stk::mesh::field_data(*stkMeshStruct->coordinates_field, *overlapnodes[i]);
    for (int dim=0; dim<stkMeshStruct->numDim; dim++)
      coordinates[3*node_lid + dim] = x[dim];
  }

  return coordinates;
}

void
Albany::STKDiscretization::getOwned_xyz(double** x, double** y, double** z,
                                        double **rbm, int& nNodes, int numPDEs, int numScalar,  int nullSpaceDim)
{
  // Function to return x,y,z at owned nodes as double*, specifically for ML
  int numDim = stkMeshStruct->numDim;
  nNodes = numOwnedNodes;

  if (allocated_xyz) { delete [] xx; delete [] yy; delete [] zz;} 
  xx = new double[numOwnedNodes];
  yy = new double[numOwnedNodes];
  zz = new double[numOwnedNodes];
  if (nullSpaceDim>0) rr = new double[(nullSpaceDim + numScalar)*numPDEs*nNodes];
  else                rr = new double[1]; // Just so there is something to delete in destructor
  allocated_xyz = true;

  for (int i=0; i < numOwnedNodes; i++)  {
    int node_gid = gid(ownednodes[i]);
    int node_lid = node_map->LID(node_gid);

    double* X = stk::mesh::field_data(*stkMeshStruct->coordinates_field, *overlapnodes[i]);
    if (numDim > 0) xx[node_lid] = X[0];
    if (numDim > 1) yy[node_lid] = X[1];
    if (numDim > 2) zz[node_lid] = X[2];
  }

  // Leave unused dim as null pointers.
  if (numDim > 0) *x = xx;
  if (numDim > 1) *y = yy;
  if (numDim > 2) *z = zz;
  *rbm = rr;
}


const Teuchos::ArrayRCP<std::string>& 
Albany::STKDiscretization::getWsEBNames() const
{
  return wsEBNames;
}

const Teuchos::ArrayRCP<int>& 
Albany::STKDiscretization::getWsPhysIndex() const
{
  return wsPhysIndex;
}

/*
const std::vector<std::string>&
Albany::STKDiscretization::getNodeSetIDs() const
{
  return nodeSetIDs;
}
*/

void Albany::STKDiscretization::outputToExodus(const Epetra_Vector& soln, const double time)
{
  // Put solution as Epetra_Vector into STK Mesh
  setSolutionField(soln);

#ifdef ALBANY_SEACAS
  if (stkMeshStruct->exoOutput) {

    double time_label = monotonicTimeLabel(time);

    int out_step = stk::io::process_output_request(*mesh_data, *stkMeshStruct->bulkData, time_label);

    if (map->Comm().MyPID()==0) {
      *out << "Albany::STKDiscretization::outputToExodus: writing time " << time;
      if (time_label != time) *out << " with label " << time_label;
      *out << " to index " <<out_step<<" in file "<<stkMeshStruct->exoOutFile<< endl;
    }
  }
#endif
}

double
Albany::STKDiscretization::monotonicTimeLabel(const double time) 
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
Albany::STKDiscretization::setResidualField(const Epetra_Vector& residual) 
{
#ifdef ALBANY_LCM
  // Copy residual vector into residual field, one node at a time
  for (std::size_t i=0; i < ownednodes.size(); i++)  
  {
    double* res = stk::mesh::field_data(*stkMeshStruct->residual_field, *ownednodes[i]);
    for (std::size_t j=0; j<neq; j++)
      res[j] = residual[getOwnedDOF(i,j)];
  }
#endif
}

Teuchos::RCP<Epetra_Vector>
Albany::STKDiscretization::getSolutionField() const
{
  // Copy soln vector into solution field, one node at a time
  Teuchos::RCP<Epetra_Vector> soln = Teuchos::rcp(new Epetra_Vector(*map));
  for (std::size_t i=0; i < ownednodes.size(); i++)  {
    const double* sol = stk::mesh::field_data(*stkMeshStruct->solution_field, *ownednodes[i]);
    for (std::size_t j=0; j<neq; j++)
      (*soln)[getOwnedDOF(i,j)] = sol[j];
  }
  return soln;
}

/*****************************************************************/
/*** Private functions follow. These are just used in above code */
/*****************************************************************/

void 
Albany::STKDiscretization::setSolutionField(const Epetra_Vector& soln) 
{
  // Copy soln vector into solution field, one node at a time
  for (std::size_t i=0; i < ownednodes.size(); i++)  {
    double* sol = stk::mesh::field_data(*stkMeshStruct->solution_field, *ownednodes[i]);
    for (std::size_t j=0; j<neq; j++)
      sol[j] = soln[getOwnedDOF(i,j)];
  }
}

inline int Albany::STKDiscretization::gid(const stk::mesh::Entity& node) const
{ return node.identifier()-1; }

inline int Albany::STKDiscretization::gid(const stk::mesh::Entity* node) const
{ return gid(*node); }

inline int Albany::STKDiscretization::getOwnedDOF(const int inode, const int eq) const
{
  if (interleavedOrdering) return inode*neq + eq;
  else  return inode + numOwnedNodes*eq;
}

inline int Albany::STKDiscretization::getOverlapDOF(const int inode, const int eq) const
{
  if (interleavedOrdering) return inode*neq + eq;
  else  return inode + numOverlapNodes*eq;
}

inline int Albany::STKDiscretization::getGlobalDOF(const int inode, const int eq) const
{
  if (interleavedOrdering) return inode*neq + eq;
  else  return inode + numGlobalNodes*eq;
}

int Albany::STKDiscretization::nonzeroesPerRow(const int neq) const
{
  int numDim = stkMeshStruct->numDim;
  int estNonzeroesPerRow;
  switch (numDim) {
  case 0: estNonzeroesPerRow=1*neq; break;
  case 1: estNonzeroesPerRow=3*neq; break;
  case 2: estNonzeroesPerRow=9*neq; break;
  case 3: estNonzeroesPerRow=27*neq; break;
  default: TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
			      "STKDiscretization:  Bad numDim"<< numDim);
  }
  return estNonzeroesPerRow;
}

void Albany::STKDiscretization::computeOwnedNodesAndUnknowns()
{
  // Loads member data:  ownednodes, numOwnedNodes, node_map, numGlobalNodes, map
  // maps for owned nodes and unknowns
  stk::mesh::Selector select_owned_in_part =
    stk::mesh::Selector( metaData.universal_part() ) &
    stk::mesh::Selector( metaData.locally_owned_part() );

  stk::mesh::get_selected_entities( select_owned_in_part ,
				    bulkData.buckets( metaData.node_rank() ) ,
				    ownednodes );

  numOwnedNodes = ownednodes.size();
  std::vector<int> indices(numOwnedNodes);
  for (int i=0; i < numOwnedNodes; i++) indices[i] = gid(ownednodes[i]);

  node_map = Teuchos::rcp(new Epetra_Map(-1, numOwnedNodes,
					 &(indices[0]), 0, *comm));

  numGlobalNodes = node_map->NumGlobalElements();
  indices.resize(numOwnedNodes * neq);
  for (int i=0; i < numOwnedNodes; i++)
    for (std::size_t j=0; j < neq; j++)
      indices[getOwnedDOF(i,j)] = getGlobalDOF(gid(ownednodes[i]),j);

  map = Teuchos::rcp(new Epetra_Map(-1, indices.size(), &(indices[0]), 0, *comm));

}

void Albany::STKDiscretization::computeOverlapNodesAndUnknowns()
{
  // Loads member data:  overlapodes, numOverlapodes, overlap_node_map, coordinates
  std::vector<int> indices;
  // maps for overlap unknowns
  stk::mesh::Selector select_overlap_in_part =
    stk::mesh::Selector( metaData.universal_part() ) &
    ( stk::mesh::Selector( metaData.locally_owned_part() )
      | stk::mesh::Selector( metaData.globally_shared_part() ) );

  //  overlapnodes used for overlap map -- stored for changing coords
  stk::mesh::get_selected_entities( select_overlap_in_part ,
				    bulkData.buckets( metaData.node_rank() ) ,
				    overlapnodes );

  numOverlapNodes = overlapnodes.size();
  indices.resize(numOverlapNodes * neq);
  for (int i=0; i < numOverlapNodes; i++)
    for (std::size_t j=0; j < neq; j++)
      indices[getOverlapDOF(i,j)] = getGlobalDOF(gid(overlapnodes[i]),j);

  overlap_map = Teuchos::rcp(new Epetra_Map(-1, indices.size(),
					    &(indices[0]), 0, *comm));

  // Set up epetra map of node IDs
  indices.resize(numOverlapNodes);
  for (int i=0; i < numOverlapNodes; i++)
    indices[i] = gid(overlapnodes[i]);

  overlap_node_map = Teuchos::rcp(new Epetra_Map(-1, indices.size(),
						 &(indices[0]), 0, *comm));

  coordinates.resize(3*numOverlapNodes);
 
}


void Albany::STKDiscretization::computeGraphs()
{

  // FIXME - GAH: the following assumes all element blocks in the problem have the same
  // number of nodes per element and that the cell topologies are the same.
  std::map<int, stk::mesh::Part*>::iterator pv = stkMeshStruct->partVec.begin();
  int nodes_per_element =  metaData.get_cell_topology(*(pv->second)).getNodeCount(); 
// int nodes_per_element_est =  metaData.get_cell_topology(*(stkMeshStruct->partVec[0])).getNodeCount();

  // Loads member data:  overlap_graph, numOverlapodes, overlap_node_map, coordinates, graphs
  overlap_graph =
    Teuchos::rcp(new Epetra_CrsGraph(Copy, *overlap_map,
                                     neq*nodes_per_element, false));

  stk::mesh::Selector select_owned_in_part =
    stk::mesh::Selector( metaData.universal_part() ) &
    stk::mesh::Selector( metaData.locally_owned_part() );

  stk::mesh::get_selected_entities( select_owned_in_part ,
				    bulkData.buckets( metaData.element_rank() ) ,
				    cells );


  if (comm->MyPID()==0)
    *out << "STKDisc: " << cells.size() << " elements on Proc 0 " << endl;

  int row, col;

  for (std::size_t i=0; i < cells.size(); i++) {
    stk::mesh::Entity& e = *cells[i];
    stk::mesh::PairIterRelation rel = e.relations(metaData.NODE_RANK);

    // loop over local nodes
    for (std::size_t j=0; j < rel.size(); j++) {
      stk::mesh::Entity& rowNode = * rel[j].entity();

      // loop over eqs
      for (std::size_t k=0; k < neq; k++) {
        row = getGlobalDOF(gid(rowNode), k);
        for (std::size_t l=0; l < rel.size(); l++) {
          stk::mesh::Entity& colNode = * rel[l].entity();
          for (std::size_t m=0; m < neq; m++) {
            col = getGlobalDOF(gid(colNode), m);
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

void Albany::STKDiscretization::computeWorksetInfo()
{

  stk::mesh::Selector select_owned_in_part =
    stk::mesh::Selector( metaData.universal_part() ) &
    stk::mesh::Selector( metaData.locally_owned_part() );

  std::vector< stk::mesh::Bucket * > buckets ;
  stk::mesh::get_buckets( select_owned_in_part ,
                          bulkData.buckets( metaData.element_rank() ) ,
                          buckets);

  int numBuckets =  buckets.size();

  wsEBNames.resize(numBuckets);
  for (int i=0; i<numBuckets; i++) {
    std::vector< stk::mesh::Part * >  bpv;
    buckets[i]->supersets(bpv);
    for (std::size_t j=0; j<bpv.size(); j++) {
      if (bpv[j]->primary_entity_rank() == metaData.element_rank()) {
	if (bpv[j]->name()[0] != '{') {
	  // *out << "Bucket " << i << " is in Element Block:  " << bpv[j]->name() 
	  //      << "  and has " << buckets[i]->size() << " elements." << endl;
	  wsEBNames[i]=bpv[j]->name();
	}
      }
    }
  }

  wsPhysIndex.resize(numBuckets);
  if (stkMeshStruct->allElementBlocksHaveSamePhysics)
    for (int i=0; i<numBuckets; i++) wsPhysIndex[i]=0;
  else 
    for (int i=0; i<numBuckets; i++) wsPhysIndex[i]=stkMeshStruct->ebNameToIndex[wsEBNames[i]];

  // Fill  wsElNodeEqID(workset, el_LID, local node, Eq) => unk_LID

  wsElNodeEqID.resize(numBuckets);
  coords.resize(numBuckets);

  for (int b=0; b < numBuckets; b++) {

    stk::mesh::Bucket& buck = *buckets[b];
    wsElNodeEqID[b].resize(buck.size());
    coords[b].resize(buck.size());

    // i is the element index within bucket b

    for (std::size_t i=0; i < buck.size(); i++) {
  
      // Traverse all the elements in this bucket
      stk::mesh::Entity& element = buck[i];

      // Now, save a map from element GID to workset on this PE
      elemGIDws[gid(element)].ws = b;

      // Now, save a map from element GID to local id on this workset on this PE
      elemGIDws[gid(element)].LID = i;

      stk::mesh::PairIterRelation rel = element.relations(metaData.NODE_RANK);

      int nodes_per_element = rel.size();
      wsElNodeEqID[b][i].resize(nodes_per_element);
      coords[b][i].resize(nodes_per_element);
      // loop over local nodes
      for (int j=0; j < nodes_per_element; j++) {
        stk::mesh::Entity& rowNode = * rel[j].entity();
        int node_gid = gid(rowNode);
        int node_lid = overlap_node_map->LID(node_gid);
        
        TEUCHOS_TEST_FOR_EXCEPTION(node_lid<0, std::logic_error,
			   "STK1D_Disc: node_lid out of range " << node_lid << endl);
        coords[b][i][j] = stk::mesh::field_data(*stkMeshStruct->coordinates_field, rowNode);
        wsElNodeEqID[b][i][j].resize(neq);
        for (std::size_t eq=0; eq < neq; eq++) 
          wsElNodeEqID[b][i][j][eq] = getOverlapDOF(node_lid,eq);
      }
    }
  }

  // Pull out pointers to shards::Arrays for every bucket, for every state
  // Code is data-type dependent
  stateArrays.resize(numBuckets);
  for (std::size_t b=0; b < buckets.size(); b++) {
    stk::mesh::Bucket& buck = *buckets[b];
    for (std::size_t i=0; i<stkMeshStruct->qpscalar_states.size(); i++) {
      stk::mesh::BucketArray<Albany::AbstractSTKMeshStruct::QPScalarFieldType> array(*stkMeshStruct->qpscalar_states[i], buck);
      MDArray ar = array;
      stateArrays[b][stkMeshStruct->qpscalar_states[i]->name()] = ar;
    }
    for (std::size_t i=0; i<stkMeshStruct->qpvector_states.size(); i++) {
      stk::mesh::BucketArray<Albany::AbstractSTKMeshStruct::QPVectorFieldType> array(*stkMeshStruct->qpvector_states[i], buck);
      MDArray ar = array;
      stateArrays[b][stkMeshStruct->qpvector_states[i]->name()] = ar;
    }
    for (std::size_t i=0; i<stkMeshStruct->qptensor_states.size(); i++) {
      stk::mesh::BucketArray<Albany::AbstractSTKMeshStruct::QPTensorFieldType> array(*stkMeshStruct->qptensor_states[i], buck);
      MDArray ar = array;
      stateArrays[b][stkMeshStruct->qptensor_states[i]->name()] = ar;
    }    
    for (std::size_t i=0; i<stkMeshStruct->scalarValue_states.size(); i++) {      
      const int size = 1;
      shards::Array<double, shards::NaturalOrder, Cell> array(&(stkMeshStruct->time), size);
      MDArray ar = array;
      stateArrays[b][stkMeshStruct->scalarValue_states[i]] = ar;
    }
  }
}

void Albany::STKDiscretization::computeSideSets(){

  const stk::mesh::EntityRank element_rank = metaData.element_rank();

  // iterator over all side_rank parts found in the mesh
  std::map<std::string, stk::mesh::Part*>::iterator ss = stkMeshStruct->ssPartVec.begin();

  int numBuckets = wsEBNames.size();

  sideSets.resize(numBuckets); // Need a sideset list per workset

  while ( ss != stkMeshStruct->ssPartVec.end() ) { 

    // Get all owned sides in this side set
    stk::mesh::Selector select_owned_in_sspart =

      // get only entities in the ss part (ss->second is the current sideset part)
      stk::mesh::Selector( *(ss->second) ) &
      // and only if the part is local
      stk::mesh::Selector( metaData.locally_owned_part() );

    std::vector< stk::mesh::Entity * > sides ;
    stk::mesh::get_selected_entities( select_owned_in_sspart , // sides local to this processor
				      bulkData.buckets( metaData.side_rank() ) ,
				      sides ); // store the result in "sides"

    *out << "STKDisc: sideset "<< ss->first <<" has size " << sides.size() << "  on Proc 0." << endl;

 //   sideSets[ss->first].resize(sides.size()); // build the data holder

    // loop over the sides to see what they are, then fill in the data holder
    // for side set options, look at $TRILINOS_DIR/packages/stk/stk_usecases/mesh/UseCase_13.cpp

    for (std::size_t localSideID=0; localSideID < sides.size(); localSideID++) {

      stk::mesh::Entity &sidee = *sides[localSideID];

      const stk::mesh::PairIterRelation side_elems = sidee.relations(element_rank); // get the elements
            // containing the side. Should be only one for external boundaries. Need to fix for internal
            // flux conditions

      TEUCHOS_TEST_FOR_EXCEPTION(side_elems.size() != 1, std::logic_error,
			   "STKDisc: cannot figure out side set topology for side set " << ss->first << endl);

      const stk::mesh::Entity & elem = *side_elems[0].entity();

      SideStruct sStruct;

      // Save elem id. This is the global element id
      sStruct.elem_GID = gid(elem);

      int workset = elemGIDws[sStruct.elem_GID].ws; // Get the ws that this element lives in

      // Save elem id. This is the local element id within the workset
      sStruct.elem_LID = elemGIDws[sStruct.elem_GID].LID;

      // Save the side identifier inside of the element. This starts at zero here.
      sStruct.side_local_id = determine_local_side_id(elem, sidee);

      SideSetList& ssList = sideSets[workset];   // Get a ref to the side set map for this ws
      SideSetList::iterator it = ssList.find(ss->first); // Get an iterator to the correct sideset (if
                                                                // it exists)

//std::cout << "Workset = " << workset << " egid = " << sStruct.elem_GID << " eLID = " 
//    << sStruct.elem_LID << " eside = " << sStruct.side_local_id << std::endl;


      if(it != ssList.end()) // The sideset has already been created

        it->second.push_back(sStruct); // Save this side to the vector that belongs to the name ss->first

      else { // Add the key ss->first to the map, and the side vector to that map

        std::vector<SideStruct> tmpSSVec;
        tmpSSVec.push_back(sStruct);
        
        ssList.insert(SideSetList::value_type(ss->first, tmpSSVec));

      }

    }

    ss++;
  }
}

// From $TRILINOS_DIR/packages/stk/stk_usecases/mesh/UseCase_13.cpp (GAH)

unsigned 
Albany::STKDiscretization::determine_local_side_id( const stk::mesh::Entity & elem , stk::mesh::Entity & side ) {

  using namespace stk;

  const CellTopologyData * const elem_top = mesh::fem::get_cell_topology( elem ).getCellTopologyData();

  const mesh::PairIterRelation elem_nodes = elem.relations( mesh::fem::FEMMetaData::NODE_RANK );
  const mesh::PairIterRelation side_nodes = side.relations( mesh::fem::FEMMetaData::NODE_RANK );

  int side_id = -1 ;

  for ( unsigned i = 0 ; side_id == -1 && i < elem_top->side_count ; ++i ) {
    const CellTopologyData & side_top = * elem_top->side[i].topology ;
    const unsigned     * side_map =   elem_top->side[i].node ;

    if ( side_nodes.size() == side_top.node_count ) {

      side_id = i ;

      for ( unsigned j = 0 ;
            side_id == static_cast<int>(i) && j < side_top.node_count ; ++j ) {

        mesh::Entity * const elem_node = elem_nodes[ side_map[j] ].entity();

        bool found = false ;

        for ( unsigned k = 0 ; ! found && k < side_top.node_count ; ++k ) {
          found = elem_node == side_nodes[k].entity();
        }

        if ( ! found ) { side_id = -1 ; }
      }
    }
  }

  if ( side_id < 0 ) {
    std::ostringstream msg ;
    msg << "determine_local_side_id( " ;
    msg << elem_top->name ;
    msg << " , Element[ " ;
    msg << elem.identifier();
    msg << " ]{" ;
    for ( unsigned i = 0 ; i < elem_nodes.size() ; ++i ) {
      msg << " " << elem_nodes[i].entity()->identifier();
    }
    msg << " } , Side[ " ;
    msg << side.identifier();
    msg << " ]{" ;
    for ( unsigned i = 0 ; i < side_nodes.size() ; ++i ) {
      msg << " " << side_nodes[i].entity()->identifier();
    }
    msg << " } ) FAILED" ;
    throw std::runtime_error( msg.str() );
  }

  return static_cast<unsigned>(side_id) ;
}


void Albany::STKDiscretization::computeNodeSets()
{

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
    nodeSetCoords[ns->first].resize(nodes.size());
//    nodeSetIDs.push_back(ns->first); // Grab string ID
    *out << "STKDisc: nodeset "<< ns->first <<" has size " << nodes.size() << "  on Proc 0." << endl;
    for (std::size_t i=0; i < nodes.size(); i++) {
      int node_gid = gid(nodes[i]);
      int node_lid = node_map->LID(node_gid);
      nodeSets[ns->first][i].resize(neq);
      for (std::size_t eq=0; eq < neq; eq++)  nodeSets[ns->first][i][eq] = getOwnedDOF(node_lid,eq);
      nodeSetCoords[ns->first][i] = stk::mesh::field_data(*stkMeshStruct->coordinates_field, *nodes[i]);
    }
    ns++;
  }
}

void Albany::STKDiscretization::setupExodusOutput()
{
#ifdef ALBANY_SEACAS
  if (stkMeshStruct->exoOutput) {

    Ioss::Init::Initializer io;
    mesh_data = new stk::io::MeshData();
    stk::io::create_output_mesh(stkMeshStruct->exoOutFile,
		  Albany::getMpiCommFromEpetraComm(*comm),
		  bulkData, *mesh_data);

    stk::io::define_output_fields(*mesh_data, metaData);

  }
#else
  if (stkMeshStruct->exoOutput) 
    *out << "\nWARNING: exodus output requested but SEACAS not compiled in:"
         << " disabling exodus output \n" << endl;
  
#endif
}

void
Albany::STKDiscretization::updateMesh(Teuchos::RCP<Albany::AbstractSTKMeshStruct> stkMeshStruct,
				      const Teuchos::RCP<const Epetra_Comm>& comm)
{
  computeOwnedNodesAndUnknowns();

  computeOverlapNodesAndUnknowns();

  computeGraphs();

  computeWorksetInfo();

  computeNodeSets();

  computeSideSets();

  setupExodusOutput();
}
