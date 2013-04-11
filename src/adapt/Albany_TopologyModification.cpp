//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_TopologyModification.hpp"
#include "Albany_StressFracture.hpp"
#include "Teuchos_TimeMonitor.hpp"
#include <stk_util/parallel/ParallelReduce.hpp>

#include <boost/foreach.hpp>

typedef stk::mesh::Entity Entity;
typedef stk::mesh::EntityRank EntityRank;
typedef stk::mesh::RelationIdentifier EdgeId;
typedef stk::mesh::EntityKey EntityKey;

Albany::TopologyMod::
TopologyMod(const Teuchos::RCP<Teuchos::ParameterList>& params_,
                     const Teuchos::RCP<ParamLib>& paramLib_,
                     Albany::StateManager& StateMgr_,
                     const Teuchos::RCP<const Epetra_Comm>& comm_) :
    Albany::AbstractAdapter(params_, paramLib_, StateMgr_, comm_),
    remeshFileIndex(1)
{

  disc = StateMgr.getDiscretization();

	stk_discretization = static_cast<Albany::STKDiscretization *>(disc.get());

	stkMeshStruct = stk_discretization->getSTKMeshStruct();

	bulkData = stkMeshStruct->bulkData;
	metaData = stkMeshStruct->metaData;

	// The entity ranks
	nodeRank = metaData->NODE_RANK;
	edgeRank = metaData->EDGE_RANK;
	faceRank = metaData->FACE_RANK;
	elementRank = metaData->element_rank();

	numDim = stkMeshStruct->numDim;

  // Save the initial output file name
  baseExoFileName = stkMeshStruct->exoOutFile;

  double crit_stress = params->get<double>("Fracture Stress");

  // Not used yet
  std::vector<std::vector<double> > avg_stresses;

  sfcriterion = 
    Teuchos::rcp(new LCM::StressFracture(numDim, elementRank, avg_stresses, crit_stress, 
    *stk_discretization));

// Modified by GAH from LCM::NodeUpdate.cc

  topology =
    Teuchos::rcp(new LCM::topology(disc, sfcriterion));


}

Albany::TopologyMod::
~TopologyMod()
{
}

bool
Albany::TopologyMod::queryAdaptationCriteria(){


// FIXME Dumb criteria

   if(iter == 5 || iter == 10 || iter == 15){ // fracture at iter = 5, 10, 15

    // First, check and see if the mesh fracture criteria is met anywhere before messing with things.

    // Get a vector containing the edge set of the mesh where fractures can occur

    std::vector<stk::mesh::Entity*> edge_lst;

// Get the faces owned by this processor
    stk::mesh::Selector select_owned = metaData->locally_owned_part();

    // get all the faces local to this processor
    stk::mesh::get_selected_entities( select_owned,
				    bulkData->buckets( numDim - 1 ) ,
				    edge_lst );

    *out << "Num edges : " << edge_lst.size() << std::endl;

    // Probability that fracture_criterion will return true.
    double p = iter;
    int total_fractured;

    // Iterate over the boundary entities
    for (int i = 0; i < edge_lst.size(); ++i){

      stk::mesh::Entity& edge = *(edge_lst[i]);

      if(sfcriterion->fracture_criterion(edge, p))

        fractured_edges.push_back(edge_lst[i]);

    }

    if((total_fractured = accumulateFractured(fractured_edges.size())) == 0) {

       fractured_edges.clear();

       return false; // nothing to do

    }


    *out << "TopologyModification: Need to split \"" 
         << total_fractured << "\" mesh elements." << std::endl;


    return true;

  }

  return false; 
 
}

bool
Albany::TopologyMod::adaptMesh(){

  *out << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
  *out << "Adapting mesh using Albany::TopologyMod method      " << std::endl;
  *out << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;

  // Save the current results and close the exodus file

  // Create a remeshed output file naming convention by adding the remeshFileIndex ahead of the period
  std::ostringstream ss;
  std::string str = baseExoFileName;
  ss << "_" << remeshFileIndex << ".";
  str.replace(str.find('.'), 1, ss.str());

  *out << "Remeshing: renaming output file to - " << str << endl;

  // Open the new exodus file for results
  stk_discretization->reNameExodusOutput(str);

  remeshFileIndex++;

  // Print element connectivity before the mesh topology is modified

//  *out << "*************************\n"
//	   << "Before element separation\n"
//	   << "*************************\n";
  
  // Start the mesh update process

  // Modifies mesh for graph algorithm
  // Function must be called each time before there are changes to the mesh

//topology->disp_connectivity();

  topology->remove_element_to_node_relations();

  // Check for failure criterion

  std::map<EntityKey, bool> local_entity_open;
  std::map<EntityKey, bool> global_entity_open;
  topology->set_entities_open(fractured_edges, local_entity_open);

  getGlobalOpenList(local_entity_open, global_entity_open);

  // begin mesh update

  topology->fracture_boundary(global_entity_open);

  // Clear the list of fractured edges in preparation for the next
  // fracture event
  fractured_edges.clear();

  //std::string gviz_output = "output.dot";
  //topology.output_to_graphviz(gviz_output,entity_open);

  // Recreates connectivity in stk mesh expected by Albany_STKDiscretization
  // Must be called each time at conclusion of mesh modification

  topology->restore_element_to_node_relations();

  // Output the surface mesh as an exodus file
//  topology->output_surface_mesh();


//  *out << "*************************\n"
//	   << "After element separation\n"
//	   << "*************************\n";

//  topology->disp_connectivity();

  // Throw away all the Albany data structures and re-build them from the mesh

  stk_discretization->updateMesh();

  return true;

}

//! Transfer solution between meshes.
// THis is a no-op as the solution is copied to the newly created nodes by the topology->fracture_boundary() function.
void
Albany::TopologyMod::
solutionTransfer(const Epetra_Vector& oldSolution,
        Epetra_Vector& newSolution){}


Teuchos::RCP<const Teuchos::ParameterList>
Albany::TopologyMod::getValidAdapterParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericAdapterParams("ValidTopologyModificationParams");

  validPL->set<double>("Fracture Stress", 1.0, "Fracture stress value at which two elements separate");

  return validPL;
}

void
Albany::TopologyMod::showRelations(){

	std::vector<Entity*> element_lst;
	stk::mesh::get_entities(*(bulkData),elementRank,element_lst);

	// Remove extra relations from element
	for (int i = 0; i < element_lst.size(); ++i){
		Entity & element = *(element_lst[i]);
		stk::mesh::PairIterRelation relations = element.relations();
    std::cout << "Element " << element_lst[i]->identifier() << " relations are :" << std::endl;

		for (int j = 0; j < relations.size(); ++j){
			cout << "entity:\t" << relations[j].entity()->identifier() << ","
				 << relations[j].entity()->entity_rank() << "\tlocal id: "
				 << relations[j].identifier() << "\n";
		}
  }
}

#ifdef ALBANY_MPI
int
Albany::TopologyMod::accumulateFractured(int num_fractured){

  int total_fractured;

  stk::all_reduce_sum(bulkData->parallel(), &num_fractured, &total_fractured, 1);

  return total_fractured;
}

// Parallel all-gatherv function. Communicates local open list to all processors to form global open list.
void 
Albany::TopologyMod::getGlobalOpenList( std::map<EntityKey, bool>& local_entity_open,  
        std::map<EntityKey, bool>& global_entity_open){

   // Make certain that we can send keys as MPI_UINT64_T types
   assert(sizeof(EntityKey::raw_key_type) >= sizeof(uint64_t));

   const unsigned parallel_size = bulkData->parallel_size();

// Build local vector of keys
   std::pair<EntityKey,bool> me; // what a map<EntityKey, bool> is made of
   std::vector<EntityKey::raw_key_type> v;     // local vector of open keys

   BOOST_FOREACH(me, local_entity_open) {
       v.push_back(me.first.raw_key());

// Debugging
/*
      const unsigned entity_rank = stk::mesh::entity_rank( me.first);
      const stk::mesh::EntityId entity_id = stk::mesh::entity_id( me.first );
      const std::string & entity_rank_name = metaData->entity_rank_name( entity_rank );
      Entity *entity = bulkData->get_entity(me.first);
      std::cout<<"Single proc fracture list contains "<<" "<<entity_rank_name<<" ["<<entity_id<<"] Proc:"
         <<entity->owner_rank() <<std::endl;
*/

   }

   int num_open_on_pe = v.size();

// Perform the allgatherv

   // gather the number of open entities on each processor
   int *sizes = new int[parallel_size];
   MPI_Allgather(&num_open_on_pe, 1, MPI_INT, sizes, 1, MPI_INT, bulkData->parallel()); 

   // Loop over each processor and calculate the array offset of its entities in the receive array
   int *offsets = new int[parallel_size];
   int count = 0; 

   for (int i = 0; i < parallel_size; i++){
     offsets[i] = count; 
     count += sizes[i];
   } 

   int total_number_of_open_entities = count;

   EntityKey::raw_key_type *result_array = new EntityKey::raw_key_type[total_number_of_open_entities];
   MPI_Allgatherv(&v[0], num_open_on_pe, MPI_UINT64_T, result_array, 
       sizes, offsets, MPI_UINT64_T, bulkData->parallel());

   // Save the global keys
   for(int i = 0; i < total_number_of_open_entities; i++){

      EntityKey key = EntityKey(&result_array[i]);
      global_entity_open[key] = true;

// Debugging
/*
      const unsigned entity_rank = stk::mesh::entity_rank( key);
      const stk::mesh::EntityId entity_id = stk::mesh::entity_id( key );
      const std::string & entity_rank_name = metaData->entity_rank_name( entity_rank );
      Entity *entity = bulkData->get_entity(key);
      if(!entity) { std::cout << "Error on this processor: Entity not addressible!!!!!!!!!!!!!" << std::endl;

      std::cout<<"Global proc fracture list contains "<<" "<<entity_rank_name<<" ["<<entity_id<<"]" << std::endl;
	std::vector<Entity*> element_lst;
	stk::mesh::get_entities(*(bulkData),elementRank,element_lst);
	for (int i = 0; i < element_lst.size(); ++i){
      std::cout << element_lst[i]->identifier() << std::endl;
  }
	std::vector<Entity*> entity_lst;
	stk::mesh::get_entities(*(bulkData),entity_rank,entity_lst);
	for (int i = 0; i < entity_lst.size(); ++i){
      std::cout << entity_lst[i]->identifier() << std::endl;
  }
      }
      else {
      std::cout<<"Global proc fracture list contains "<<" "<<entity_rank_name<<" ["<<entity_id<<"] Proc:"
         <<entity->owner_rank() <<std::endl;
      }
*/
    }

   delete [] sizes;
   delete [] offsets;
   delete [] result_array;
}

#else
int
Albany::TopologyMod::accumulateFractured(int num_fractured){
  return num_fractured;
}

// Parallel all-gatherv function. Communicates local open list to all processors to form global open list.
void 
Albany::TopologyMod::getGlobalOpenList( std::map<EntityKey, bool>& local_entity_open,  
        std::map<EntityKey, bool>& global_entity_open){

   global_entity_open = local_entity_open;
}
#endif
