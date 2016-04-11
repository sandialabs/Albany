//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <Teuchos_TimeMonitor.hpp>

#include <stk_util/parallel/ParallelReduce.hpp>

#include "AAdapt_TopologyModification.hpp"
#include "LCM/utils/topology/Topology_FractureCriterion.h"

namespace AAdapt {

typedef stk::mesh::Entity Entity;
typedef stk::mesh::EntityRank EntityRank;
typedef stk::mesh::RelationIdentifier EdgeId;
typedef stk::mesh::EntityKey EntityKey;

//
//
//
AAdapt::TopologyMod::TopologyMod(
    Teuchos::RCP<Teuchos::ParameterList> const & params,
    Teuchos::RCP<ParamLib> const & param_lib,
    Albany::StateManager & state_mgr,
    const Teuchos::RCP<const Teuchos_Comm> & commT) :
  AAdapt::AbstractAdapter(params, param_lib, state_mgr, commT),
  remesh_file_index_(1) {

  discretization_ = state_mgr_.getDiscretization();

  stk_discretization_ =
    static_cast<Albany::STKDiscretization*>(discretization_.get());

  stk_mesh_struct_ = stk_discretization_->getSTKMeshStruct();

  bulk_data_ = stk_mesh_struct_->bulkData;
  meta_data_ = stk_mesh_struct_->metaData;

  num_dim_ = stk_mesh_struct_->numDim;

  // Save the initial output file name
  base_exo_filename_ = stk_mesh_struct_->exoOutFile;

  std::string const
  bulk_block_name = params->get<std::string>("Bulk Block Name");

  std::string const
  interface_block_name = params->get<std::string>("Interface Block Name");

  std::string const
  stress_name = "nodal_FirstPK";

  double const
  critical_traction = params->get<double>("Critical Traction");

  double const
  beta = params->get<double>("beta");

  topology_ =
    Teuchos::rcp(new LCM::Topology(
        discretization_,
        bulk_block_name,
        interface_block_name));

  fracture_criterion_ =
    Teuchos::rcp(
        new LCM::FractureCriterionTraction(
            *topology_,
            stress_name,
            critical_traction,
            beta));

  topology_->set_fracture_criterion(fracture_criterion_);
}

//
//
//
AAdapt::TopologyMod::~TopologyMod() {
}

//
//
//
bool
AAdapt::TopologyMod::queryAdaptationCriteria() {
  size_t
  number_fractured_faces = topology_->setEntitiesOpen();

  return number_fractured_faces > 0;
}

//
//
//
bool
AAdapt::TopologyMod::adaptMesh(
    Epetra_Vector const & solution,
    Epetra_Vector const & ovlp_solution) {

  *output_stream_
      << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
      << "Adapting mesh using AAdapt::TopologyMod method      \n"
      << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";

  // Save the current results and close the exodus file

  // Create a remeshed output file naming convention by
  // adding the remesh_file_index_ ahead of the period
  std::ostringstream ss;
  std::string str = base_exo_filename_;
  ss << "_" << remesh_file_index_ << ".";
  str.replace(str.find('.'), 1, ss.str());

  *output_stream_ << "Remeshing: renaming output file to - " << str << '\n';

  // Open the new exodus file for results
  stk_discretization_->reNameExodusOutput(str);

  remesh_file_index_++;

  // Print element connectivity before the mesh topology is modified

  //  *output_stream_
  //   << "*************************\n"
  //   << "Before element separation\n"
  //   << "*************************\n";

  // Start the mesh update process

  // Modifies mesh for graph algorithm

  // begin mesh update

  topology_->splitOpenFaces();

  // Throw away all the Albany data structures and re-build them from the mesh

  stk_discretization_->updateMesh();

  return true;
}

//----------------------------------------------------------------------------
// Transfer solution between meshes.  This is a no-op as the
// solution is copied to the newly created nodes by the
// topology->fracture_boundary() function.
void
AAdapt::TopologyMod::
solutionTransfer(const Epetra_Vector& oldSolution,
                 Epetra_Vector& newSolution) {}

//----------------------------------------------------------------------------
Teuchos::RCP<const Teuchos::ParameterList>
AAdapt::TopologyMod::getValidAdapterParameters() const {
  Teuchos::RCP<Teuchos::ParameterList> valid_pl_ =
    this->getGenericAdapterParams("ValidTopologyModificationParams");

  valid_pl_->
  set<double>("Critical Traction",
              1.0,
              "Critical traction at which two elements separate t_eff >= t_cr");

  valid_pl_->
  set<double>("beta",
              1.0,
              "Weight factor t_eff = sqrt[(t_s/beta)^2 + t_n^2]");

  return valid_pl_;
}


//----------------------------------------------------------------------------
void
AAdapt::TopologyMod::showRelations() {
  std::vector<Entity> element_list;
  stk::mesh::get_entities(*(bulk_data_), stk::topology::ELEMENT_RANK, element_list);

  // Remove extra relations from element
  for(int i = 0; i < element_list.size(); ++i) {
    Entity element = element_list[i];

    for (EntityRank rank = stk::topology::NODE_RANK; rank < meta_data_->entity_rank_count(); ++rank) {

      Entity const* relations = bulk_data_->begin(element, rank);
      stk::mesh::ConnectivityOrdinal const* ords = bulk_data_->begin_ordinals(element, rank);
      size_t const num_relations = bulk_data_->num_connectivity(element, rank);

      std::cout << "Element " << bulk_data_->identifier(element_list[i])
                << " relations are :" << std::endl;

      for(int j = 0; j < num_relations; ++j) {
        std::cout << "entity:\t" << bulk_data_->identifier(relations[j]) << ","
                  << bulk_data_->entity_rank(relations[j]) << "\tlocal id: "
                  << ords[j] << "\n";
      }
    }
  }
}

#ifdef ALBANY_MPI
//----------------------------------------------------------------------------
int
AAdapt::TopologyMod::accumulateFractured(int num_fractured) {
  int total_fractured;

  stk::all_reduce_sum(bulk_data_->parallel(), &num_fractured, &total_fractured, 1);

  return total_fractured;
}

//----------------------------------------------------------------------------
// Parallel all-gatherv function. Communicates local open list to
// all processors to form global open list.
void
AAdapt::TopologyMod::
getGlobalOpenList(std::map<EntityKey, bool>& local_entity_open,
                  std::map<EntityKey, bool>& global_entity_open) {

  // Make certain that we can send keys as MPI_UINT64_T types
  assert(sizeof(EntityKey::entity_key_t) >= sizeof(uint64_t));

  const unsigned parallel_size = bulk_data_->parallel_size();

  // Build local vector of keys
  std::pair<EntityKey, bool> me; // what a map<EntityKey, bool> is made of
  std::vector<EntityKey::entity_key_t> v;     // local vector of open keys

  BOOST_FOREACH(me, local_entity_open) {
    v.push_back(EntityKey::entity_key_t(me.first));

    // Debugging
    /*
      const unsigned entity_rank = stk::mesh::entity_rank( me.first);
      const stk::mesh::EntityId entity_id = stk::mesh::entity_id( me.first );
      const std::string & entity_rank_name = metaData->entity_rank_name( entity_rank );
      Entity entity = bulk_data_->get_entity(me.first);
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

// Needed for backward compatibility with older MPI versions.
#ifndef MPI_UINT64_T
#define MPI_UINT64_T MPI_UNSIGNED_LONG_LONG
#endif
  EntityKey::entity_key_t* result_array = new EntityKey::entity_key_t[total_number_of_open_entities];
  MPI_Allgatherv((void *)&v[0], num_open_on_pe, MPI_UINT64_T, (void *)result_array,
                 sizes, offsets, MPI_UINT64_T, bulk_data_->parallel());

  // Save the global keys
  for(int i = 0; i < total_number_of_open_entities; i++) {

    EntityKey key = EntityKey(result_array[i]);
    global_entity_open[key] = true;

    // Debugging
    /*
      const unsigned entity_rank = stk::mesh::entity_rank( key);
      const stk::mesh::EntityId entity_id = stk::mesh::entity_id( key );
      const std::string & entity_rank_name = metaData->entity_rank_name( entity_rank );
      Entity entity = bulk_data_->get_entity(key);
      if(!entity) { std::cout << "Error on this processor: Entity not addressible!!!!!!!!!!!!!" << std::endl;

      std::cout<<"Global proc fracture list contains "<<" "<<entity_rank_name<<" ["<<entity_id<<"]" << std::endl;
    std::vector<Entity> element_lst;
    stk::mesh::get_entities(*(bulk_data_),elementRank,element_lst);
    for (int i = 0; i < element_lst.size(); ++i){
      std::cout << element_lst[i]->identifier() << std::endl;
      }
    std::vector<Entity> entity_lst;
    stk::mesh::get_entities(*(bulk_data_),entity_rank,entity_lst);
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
//----------------------------------------------------------------------------
int
AAdapt::TopologyMod::accumulateFractured(int num_fractured) {
  return num_fractured;
}

//----------------------------------------------------------------------------
// Parallel all-gatherv function. Communicates local open list to
// all processors to form global open list.
void
AAdapt::TopologyMod::getGlobalOpenList(std::map<EntityKey, bool>& local_entity_open,
                                       std::map<EntityKey, bool>& global_entity_open) {

  global_entity_open = local_entity_open;
}
#endif

}

