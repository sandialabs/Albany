//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_RandomFracture.hpp"
#include "Albany_RandomCriterion.hpp"
#include "Teuchos_TimeMonitor.hpp"
#include <stk_util/parallel/ParallelReduce.hpp>

#include <boost/foreach.hpp>

using stk::mesh::EntityKey;
using stk::mesh::Entity;

Albany::RandomFracture::
RandomFracture(const Teuchos::RCP<Teuchos::ParameterList>& params_,
               const Teuchos::RCP<ParamLib>& paramLib_,
               Albany::StateManager& StateMgr_,
               const Teuchos::RCP<const Epetra_Comm>& comm_) :
  Albany::AbstractAdapter(params_, paramLib_, StateMgr_, comm_),
  remeshFileIndex(1),
  fracture_interval_(params_->get<int>("Adaptivity Step Interval", 1)),
  fracture_probability_(params_->get<double>("Fracture Probability", 1.0))
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


  sfcriterion = 
    Teuchos::rcp(new LCM::RandomCriterion(numDim, 
                                          elementRank,
                                          *stk_discretization));

  // Modified by GAH from LCM::NodeUpdate.cc
  topology =
    Teuchos::rcp(new LCM::topology(disc, sfcriterion));

}

Albany::RandomFracture::
~RandomFracture()
{
}

bool
Albany::RandomFracture::queryAdaptationCriteria()
{
  // iter is a member variable elsewhere, NOX::Epetra::AdaptManager.H
  if ( iter % fracture_interval_ == 0) {

    // Get a vector containing the face set of the mesh where
    // fractures can occur
    std::vector<Entity*> face_list;

// Get the faces owned by this processor
    stk::mesh::Selector select_owned = metaData->locally_owned_part();


    // get all the faces owned by this processor
    stk::mesh::get_selected_entities( select_owned,
				    bulkData->buckets( numDim - 1 ) ,
				    face_list );


#ifdef ALBANY_VERBOSE
    std::cout << "Num faces owned by PE " << bulkData->parallel_rank() << " is: " << face_list.size() << std::endl;
#endif

    // keep count of total fractured faces
    int total_fractured;


    // Iterate over the boundary entities
    for (int i = 0; i < face_list.size(); ++i){

      Entity & face = *(face_list[i]);

      if(sfcriterion->fracture_criterion(face, fracture_probability_)) {
        fractured_edges.push_back(face_list[i]);
      }
    }

    if((total_fractured = accumulateFractured(fractured_edges.size())) == 0) {

      fractured_edges.clear();

      return false; // nothing to do

    }


    *out << "RandomFractureification: Need to split \"" 
              << total_fractured << "\" mesh elements." << std::endl;


    return true;

  }

  return false; 
 
}

bool
Albany::RandomFracture::adaptMesh(){

  *out << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
  *out << "Adapting mesh using Albany::RandomFracture method   " << std::endl;
  *out << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;

  // Save the current results and close the exodus file

  // Create a remeshed output file naming convention by adding the
  // remeshFileIndex ahead of the period
  std::ostringstream ss;
  std::string str = baseExoFileName;
  ss << "_" << remeshFileIndex << ".";
  str.replace(str.find('.'), 1, ss.str());

  *out << "Remeshing: renaming output file to - " << str << endl;

  // Open the new exodus file for results
  stk_discretization->reNameExodusOutput(str);

  // increment name index
  remeshFileIndex++;

  // perform topology operations
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

  // Recreates connectivity in stk mesh expected by
  // Albany_STKDiscretization Must be called each time at conclusion
  // of mesh modification
  topology->restore_element_to_node_relations();

showTopLevelRelations();
  // end mesh update

  // Throw away all the Albany data structures and re-build them from
  // the mesh
  stk_discretization->updateMesh();

  *out << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
  *out << "Completed mesh adaptation                           " << std::endl;
  *out << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;

  return true;

}

//! Transfer solution between meshes.
// THis is a no-op as the solution is copied to the newly created nodes by the topology->fracture_boundary() function.
void
Albany::RandomFracture::
solutionTransfer(const Epetra_Vector& oldSolution,
                 Epetra_Vector& newSolution){}


Teuchos::RCP<const Teuchos::ParameterList>
Albany::RandomFracture::getValidAdapterParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericAdapterParams("ValidRandomFractureificationParams");

  validPL->set<double>("Fracture Probability", 
                       1.0, 
                       "Probability of fracture");
  validPL->set<double>("Adaptivity Step Interval", 
                       1, 
                       "Interval to check for fracture");

  return validPL;
}

void
Albany::RandomFracture::showTopLevelRelations(){

  std::vector<Entity*> element_lst;
  stk::mesh::get_entities(*(bulkData),elementRank,element_lst);

  // Remove extra relations from element
  for (int i = 0; i < element_lst.size(); ++i){
    Entity & element = *(element_lst[i]);
    stk::mesh::PairIterRelation relations = element.relations();
    std::cout << "Entitiy " << element.identifier() << " relations are :" << std::endl;

    for (int j = 0; j < relations.size(); ++j){
      cout << "entity:\t" << relations[j].entity()->identifier() << ","
           << relations[j].entity()->entity_rank() << "\tlocal id: "
           << relations[j].identifier() << "\n";
    }
  }
}

void
Albany::RandomFracture::showRelations(){

  std::vector<Entity*> element_lst;
  stk::mesh::get_entities(*(bulkData),elementRank,element_lst);

  // Remove extra relations from element
  for (int i = 0; i < element_lst.size(); ++i){
    Entity & element = *(element_lst[i]);
    showRelations(0, element);
  }
}

// Recursive print function
void
Albany::RandomFracture::showRelations(int level, const Entity& ent){

    stk::mesh::PairIterRelation relations = ent.relations();

    for(int i = 0; i < level; i++)
      std::cout << "     ";

    std::cout << metaData->entity_rank_name( ent.entity_rank()) <<
          " " << ent.identifier() << " relations are :" << std::endl;

    for (int j = 0; j < relations.size(); ++j){
      for(int i = 0; i < level; i++)
        std::cout << "     ";
      cout << "  " << metaData->entity_rank_name( relations[j].entity()->entity_rank()) << ":\t" 
           << relations[j].entity()->identifier() << ","
           << relations[j].entity()->entity_rank() << "\tlocal id: "
           << relations[j].identifier() << "\n";
    }
    for (int j = 0; j < relations.size(); ++j){
      if(relations[j].entity()->entity_rank() <= ent.entity_rank())
        showRelations(level + 1, *relations[j].entity());
  }
}


#ifdef ALBANY_MPI
int
Albany::RandomFracture::accumulateFractured(int num_fractured){

  int total_fractured;

  stk::all_reduce_sum(bulkData->parallel(), &num_fractured, &total_fractured, 1);

  return total_fractured;
}

// Parallel all-gatherv function. Communicates local open list to all processors to form global open list.
void 
Albany::RandomFracture::getGlobalOpenList( std::map<EntityKey, bool>& local_entity_open,  
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
      const unsigned entity_rank = stk::mesh::entity_rank( key);
      const stk::mesh::EntityId entity_id = stk::mesh::entity_id( key );
      const std::string & entity_rank_name = metaData->entity_rank_name( entity_rank );
      Entity *entity = bulkData->get_entity(key);
      std::cout<<"Global proc fracture list contains "<<" "<<entity_rank_name<<" ["<<entity_id<<"] Proc:"
         <<entity->owner_rank() <<std::endl;
    }

   delete [] sizes;
   delete [] offsets;
   delete [] result_array;
}

#else
int
Albany::RandomFracture::accumulateFractured(int num_fractured){
  return num_fractured;
}

// Parallel all-gatherv function. Communicates local open list to all processors to form global open list.
void 
Albany::RandomFracture::getGlobalOpenList( std::map<EntityKey, bool>& local_entity_open,  
        std::map<EntityKey, bool>& global_entity_open){

   global_entity_open = local_entity_open;
}
#endif

