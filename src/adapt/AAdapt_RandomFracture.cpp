//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "AAdapt_RandomFracture.hpp"
#include "AAdapt_RandomCriterion.hpp"
#include "Teuchos_TimeMonitor.hpp"
#include <stk_util/parallel/ParallelReduce.hpp>

#include <boost/foreach.hpp>

using stk_classic::mesh::EntityKey;
using stk_classic::mesh::Entity;

namespace AAdapt {

typedef stk_classic::mesh::Entity Entity;
typedef stk_classic::mesh::EntityRank EntityRank;
typedef stk_classic::mesh::RelationIdentifier EdgeId;
typedef stk_classic::mesh::EntityKey EntityKey;

//----------------------------------------------------------------------------
AAdapt::RandomFracture::
RandomFracture(const Teuchos::RCP<Teuchos::ParameterList>& params,
               const Teuchos::RCP<ParamLib>& param_lib,
               Albany::StateManager& state_mgr,
               const Teuchos::RCP<const Epetra_Comm>& comm) :
  AAdapt::AbstractAdapter(params, param_lib, state_mgr, comm),

  remesh_file_index_(1),
  fracture_interval_(params->get<int>("Adaptivity Step Interval", 1)),
  fracture_probability_(params->get<double>("Fracture Probability", 1.0)) {

  discretization_ = state_mgr_.getDiscretization();

  stk_discretization_ =
    static_cast<Albany::STKDiscretization*>(discretization_.get());

  stk_mesh_struct_ = stk_discretization_->getSTKMeshStruct();

  bulk_data_ = stk_mesh_struct_->bulkData;
  meta_data_ = stk_mesh_struct_->metaData;

  // The entity ranks
  node_rank_ = meta_data_->NODE_RANK;
  edge_rank_ = meta_data_->EDGE_RANK;
  face_rank_ = meta_data_->FACE_RANK;
  element_rank_ = meta_data_->element_rank();

  fracture_criterion_ =
    Teuchos::rcp(new AAdapt::RandomCriterion(num_dim_,
                                          element_rank_,
                                          *stk_discretization_));

  num_dim_ = stk_mesh_struct_->numDim;

  // Save the initial output file name
  base_exo_filename_ = stk_mesh_struct_->exoOutFile;

  // Modified by GAH from LCM::NodeUpdate.cc
  topology_ =
    Teuchos::rcp(new LCM::Topology(discretization_, fracture_criterion_));

}

//----------------------------------------------------------------------------
AAdapt::RandomFracture::
~RandomFracture() {
}

//----------------------------------------------------------------------------
bool
AAdapt::RandomFracture::queryAdaptationCriteria() {
  // iter is a member variable elsewhere, NOX::Epetra::AdaptManager.H
  if(iter % fracture_interval_ == 0) {

    // Get a vector containing the face set of the mesh where
    // fractures can occur
    std::vector<stk_classic::mesh::Entity*> face_list;

    // get all the faces owned by this processor
    stk_classic::mesh::Selector select_owned = meta_data_->locally_owned_part();

    // get all the faces owned by this processor
    stk_classic::mesh::get_selected_entities(select_owned,
                                     bulk_data_->buckets(num_dim_ - 1) ,
                                     face_list);

#ifdef ALBANY_VERBOSE
    std::cout << "Num faces : " << face_list.size() << std::endl;
#endif

    // keep count of total fractured faces
    int total_fractured;

    // Iterate over the boundary entities
    for(int i(0); i < face_list.size(); ++i) {

      stk_classic::mesh::Entity& face = *(face_list[i]);

      if(fracture_criterion_->
          computeFractureCriterion(face, fracture_probability_)) {
        fractured_faces_.push_back(face_list[i]);
      }
    }

    // if(fractured_edges.size() == 0) return false; // nothing to
    // do
    if((total_fractured =
          accumulateFractured(fractured_faces_.size())) == 0) {

      fractured_faces_.clear();

      return false; // nothing to do
    }

    *output_stream_ << "RandomFractureification: Need to split \""
                    << total_fractured << "\" mesh elements." << std::endl;

    return true;
  }

  return false;
}

//----------------------------------------------------------------------------
bool
AAdapt::RandomFracture::adaptMesh(const Epetra_Vector& solution, const Epetra_Vector& ovlp_solution) {
  *output_stream_ << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
  *output_stream_ << "Adapting mesh using AAdapt::RandomFracture method   " << std::endl;
  *output_stream_ << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;

  // Create a remeshed output file naming convention by adding the
  // remeshFileIndex ahead of the period
  std::ostringstream ss;
  std::string str = base_exo_filename_;
  ss << "_" << remesh_file_index_ << ".";
  str.replace(str.find('.'), 1, ss.str());

  *output_stream_ << "Remeshing: renaming output file to - " << str << std::endl;

  // Open the new exodus file for results
  stk_discretization_->reNameExodusOutput(str);

  // increment name index
  remesh_file_index_++;

  // perform topology operations
  topology_->removeElementToNodeConnectivity(old_elem_to_node_);

  // Check for failure criterion
  std::map<EntityKey, bool> local_entity_open;
  std::map<EntityKey, bool> global_entity_open;
  topology_->setEntitiesOpen(fractured_faces_, local_entity_open);
  getGlobalOpenList(local_entity_open, global_entity_open);

  // begin mesh update
  bulk_data_->modification_begin();

  // FIXME parallel bug lies in here
  topology_->splitOpenFaces(global_entity_open);

  // Clear the list of fractured faces in preparation for the next
  // fracture event
  fractured_faces_.clear();

  // Recreates connectivity in stk mesh expected by
  // Albany_STKDiscretization Must be called each time at conclusion
  // of mesh modification
  topology_->restoreElementToNodeConnectivity(new_elem_to_node_);

  showTopLevelRelations();

  // end mesh update
  bulk_data_->modification_end();

  // Throw away all the Albany data structures and re-build them from
  // the mesh
  stk_discretization_->updateMesh();


  *output_stream_ << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
  *output_stream_ << "Completed mesh adaptation                           " << std::endl;
  *output_stream_ << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;

  return true;
}

//----------------------------------------------------------------------------
//
// Transfer solution between meshes.
//
// currently a no-op as the solution is copied to the newly created
// nodes by the topology->splitOpenFaces() function
void
AAdapt::RandomFracture::
solutionTransfer(const Epetra_Vector& oldSolution,
                 Epetra_Vector& newSolution)
{}


//----------------------------------------------------------------------------
Teuchos::RCP<const Teuchos::ParameterList>
AAdapt::RandomFracture::getValidAdapterParameters() const {
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

//----------------------------------------------------------------------------
void
AAdapt::RandomFracture::
showTopLevelRelations() {
  std::vector<Entity*> element_list;
  stk_classic::mesh::get_entities(*(bulk_data_), element_rank_, element_list);

  // Remove extra relations from element
  for(int i = 0; i < element_list.size(); ++i) {
    Entity& element = *(element_list[i]);
    stk_classic::mesh::PairIterRelation relations = element.relations();
    std::cout << "Entitiy " << element.identifier() << " relations are :" << std::endl;

    for(int j = 0; j < relations.size(); ++j) {
      std::cout << "entity:\t" << relations[j].entity()->identifier() << ","
                << relations[j].entity()->entity_rank() << "\tlocal id: "
                << relations[j].identifier() << "\n";
    }
  }
}

//----------------------------------------------------------------------------
void
AAdapt::RandomFracture::showRelations() {
  std::vector<Entity*> element_list;
  stk_classic::mesh::get_entities(*(bulk_data_), element_rank_, element_list);

  // Remove extra relations from element
  for(int i = 0; i < element_list.size(); ++i) {
    Entity& element = *(element_list[i]);
    showRelations(0, element);
  }
}

//----------------------------------------------------------------------------
void
AAdapt::RandomFracture::showRelations(int level, const Entity& entity) {
  stk_classic::mesh::PairIterRelation relations = entity.relations();

  for(int i = 0; i < level; i++) {
    std::cout << "     ";
  }

  std::cout << meta_data_->entity_rank_name(entity.entity_rank()) <<
            " " << entity.identifier() << " relations are :" << std::endl;

  for(int j = 0; j < relations.size(); ++j) {
    for(int i = 0; i < level; i++) {
      std::cout << "     ";
    }

    std::cout << "  " << meta_data_->entity_rank_name(relations[j].entity()->entity_rank()) << ":\t"
              << relations[j].entity()->identifier() << ","
              << relations[j].entity()->entity_rank() << "\tlocal id: "
              << relations[j].identifier() << "\n";
  }

  for(int j = 0; j < relations.size(); ++j) {
    if(relations[j].entity()->entity_rank() <= entity.entity_rank())
      showRelations(level + 1, *relations[j].entity());
  }
}

#ifdef ALBANY_MPI
//----------------------------------------------------------------------------
int
AAdapt::RandomFracture::accumulateFractured(int num_fractured) {
  int total_fractured;

  stk_classic::all_reduce_sum(bulk_data_->parallel(), &num_fractured, &total_fractured, 1);

  return total_fractured;
}

//----------------------------------------------------------------------------
// Parallel all-gatherv function. Communicates local open list to all processors to form global open list.
void
AAdapt::RandomFracture::getGlobalOpenList(std::map<EntityKey, bool>& local_entity_open,
    std::map<EntityKey, bool>& global_entity_open) {
  // Make certain that we can send keys as MPI_UINT64_T types
  assert(sizeof(EntityKey::raw_key_type) >= sizeof(uint64_t));

  const unsigned parallel_size = bulk_data_->parallel_size();

  // Build local vector of keys
  std::pair<EntityKey, bool> me; // what a map<EntityKey, bool> is made of
  std::vector<EntityKey::raw_key_type> v;     // local vector of open keys

  BOOST_FOREACH(me, local_entity_open) {
    v.push_back(me.first.raw_key());

    // Debugging
    /*
      const unsigned entity_rank = stk_classic::mesh::entity_rank( me.first);
      const stk_classic::mesh::EntityId entity_id = stk_classic::mesh::entity_id( me.first );
      const std::string & entity_rank_name = meta_data_->entity_rank_name( entity_rank );
      Entity *entity = bulk_data_->get_entity(me.first);
      std::cout<<"Single proc fracture list contains "<<" "<<entity_rank_name<<" ["<<entity_id<<"] Proc:"
      <<entity->owner_rank() <<std::endl;
    */

  }

  int num_open_on_pe = v.size();

  // Perform the allgatherv

  // gather the number of open entities on each processor
  int* sizes = new int[parallel_size];
  MPI_Allgather(&num_open_on_pe, 1, MPI_INT, sizes, 1, MPI_INT, bulk_data_->parallel());

  // Loop over each processor and calculate the array offset of its entities in the receive array
  int* offsets = new int[parallel_size];
  int count = 0;

  for(int i = 0; i < parallel_size; i++) {
    offsets[i] = count;
    count += sizes[i];
  }

  int total_number_of_open_entities = count;

  EntityKey::raw_key_type* result_array = new EntityKey::raw_key_type[total_number_of_open_entities];
  MPI_Allgatherv(&v[0], num_open_on_pe, MPI_UINT64_T, result_array,
                 sizes, offsets, MPI_UINT64_T, bulk_data_->parallel());

  // Save the global keys
  for(int i = 0; i < total_number_of_open_entities; i++) {

    EntityKey key = EntityKey(&result_array[i]);
    global_entity_open[key] = true;

    // Debugging
    const unsigned entity_rank = stk_classic::mesh::entity_rank(key);
    const stk_classic::mesh::EntityId entity_id = stk_classic::mesh::entity_id(key);
    const std::string& entity_rank_name = meta_data_->entity_rank_name(entity_rank);
    Entity* entity = bulk_data_->get_entity(key);
    std::cout << "Global proc fracture list contains " << " " << entity_rank_name << " [" << entity_id << "] Proc:"
              << entity->owner_rank() << std::endl;
  }

  delete [] sizes;
  delete [] offsets;
  delete [] result_array;
}

#else
int
AAdapt::RandomFracture::accumulateFractured(int num_fractured) {
  return num_fractured;
}

// Parallel all-gatherv function. Communicates local open list to all processors to form global open list.
void
AAdapt::RandomFracture::getGlobalOpenList(std::map<EntityKey, bool>& local_entity_open,
    std::map<EntityKey, bool>& global_entity_open) {
  global_entity_open = local_entity_open;
}
#endif
}
