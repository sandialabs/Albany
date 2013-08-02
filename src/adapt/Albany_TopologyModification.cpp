//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <Teuchos_TimeMonitor.hpp>

#include <stk_util/parallel/ParallelReduce.hpp>

#include "Albany_TopologyModification.hpp"
#include "Albany_StressFracture.hpp"

namespace Albany {

  typedef stk::mesh::Entity Entity;
  typedef stk::mesh::EntityRank EntityRank;
  typedef stk::mesh::RelationIdentifier EdgeId;
  typedef stk::mesh::EntityKey EntityKey;

  //----------------------------------------------------------------------------
  Albany::TopologyMod::
  TopologyMod(const Teuchos::RCP<Teuchos::ParameterList>& params,
              const Teuchos::RCP<ParamLib>& param_lib,
              Albany::StateManager& state_mgr,
              const Teuchos::RCP<const Epetra_Comm>& comm) :
    Albany::AbstractAdapter(params, param_lib, state_mgr, comm),
    remesh_file_index_(1)
  {

    discretization_ = state_mgr_.getDiscretization();

    stk_discretization_ =
      static_cast<Albany::STKDiscretization *>(discretization_.get());

    stk_mesh_struct_ = stk_discretization_->getSTKMeshStruct();

    bulk_data_ = stk_mesh_struct_->bulkData;
    meta_data_ = stk_mesh_struct_->metaData;

    // The entity ranks
    node_rank_ = meta_data_->NODE_RANK;
    edge_rank_ = meta_data_->EDGE_RANK;
    face_rank_ = meta_data_->FACE_RANK;
    element_rank_ = meta_data_->element_rank();

    num_dim_ = stk_mesh_struct_->numDim;

    // Save the initial output file name
    base_exo_filename_ = stk_mesh_struct_->exoOutFile;

    double critical_stress_ = params->get<double>("Fracture Stress");

    fracture_criterion_ =
      Teuchos::rcp(new LCM::StressFracture(num_dim_,
                                           element_rank_,
                                           avg_stresses_,
                                           critical_stress_,
                                           *stk_discretization_));

    // Modified by GAH from LCM::NodeUpdate.cc
    topology_ =
      Teuchos::rcp(new LCM::Topology(discretization_, fracture_criterion_));
  }

  //----------------------------------------------------------------------------
  Albany::TopologyMod::
  ~TopologyMod()
  {
  }

  //----------------------------------------------------------------------------
  bool
  Albany::TopologyMod::queryAdaptationCriteria()
  {
    // FIXME Dumb criteria

    if(iter == 5 || iter == 10 || iter == 15){ // fracture at iter = 5, 10, 15

      // First, check and see if the mesh fracture criteria is met
      // anywhere before messing with things.

      // Get a vector containing the edge set of the mesh where
      // fractures can occur

      std::vector<stk::mesh::Entity*> face_list;
      stk::mesh::Selector select_owned = meta_data_->locally_owned_part();
      stk::mesh::get_selected_entities(select_owned,
                                       bulk_data_->buckets(num_dim_-1),
                                       face_list);

      //    *out << "Num faces : " << face_list.size() << std::endl;
      std::cout << "Num faces : " << face_list.size() << std::endl;

      // Probability that fracture_criterion will return true.
      double p = iter;
      int total_fractured;

      // Iterate over the boundary entities
      for (int i = 0; i < face_list.size(); ++i){

        stk::mesh::Entity& face = *(face_list[i]);

        if(fracture_criterion_->computeFractureCriterion(face, p))

          fractured_faces_.push_back(face_list[i]);

      }

      //    if(fractured_edges.size() == 0) return false; // nothing to do
      if((total_fractured = accumulateFractured(fractured_faces_.size())) == 0) {

        fractured_faces_.clear();

        return false; // nothing to do
      }

      *output_stream_ << "TopologyModification: Need to split \""
                      << total_fractured << "\" mesh elements." << std::endl;


      return true;
    }
    return false;
  }

  //----------------------------------------------------------------------------
  bool
  Albany::TopologyMod::adaptMesh(const Epetra_Vector& solution, const Epetra_Vector& ovlp_solution){

    *output_stream_
      << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
      << "Adapting mesh using Albany::TopologyMod method      \n"
      << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";

    // Save the current results and close the exodus file

    // Create a remeshed output file naming convention by adding the remesh_file_index_ ahead of the period
    std::ostringstream ss;
    std::string str = base_exo_filename_;
    ss << "_" << remesh_file_index_ << ".";
    str.replace(str.find('.'), 1, ss.str());

    *output_stream_ << "Remeshing: renaming output file to - " << str << std::endl;

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
    // Function must be called each time before there are changes to the mesh

    topology_->removeElementToNodeConnectivity(old_elem_to_node_);

    // Check for failure criterion
    std::map<EntityKey, bool> local_entity_open;
    std::map<EntityKey, bool> global_entity_open;
    topology_->setEntitiesOpen(fractured_faces_, local_entity_open);

    getGlobalOpenList(local_entity_open, global_entity_open);

    // begin mesh update

    topology_->splitOpenFaces(global_entity_open);

    // Clear the list of fractured edges in preparation for the next
    // fracture event
    fractured_faces_.clear();

    topology_->restoreElementToNodeConnectivity(new_elem_to_node_);
    // Throw away all the Albany data structures and re-build them from the mesh

    stk_discretization_->updateMesh();

    return true;
  }

  //----------------------------------------------------------------------------
  // Transfer solution between meshes.  This is a no-op as the
  // solution is copied to the newly created nodes by the
  // topology->fracture_boundary() function.
  void
  Albany::TopologyMod::
  solutionTransfer(const Epetra_Vector& oldSolution,
                   Epetra_Vector& newSolution){}

  //----------------------------------------------------------------------------
  Teuchos::RCP<const Teuchos::ParameterList>
  Albany::TopologyMod::getValidAdapterParameters() const
  {
    Teuchos::RCP<Teuchos::ParameterList> valid_pl_ =
      this->getGenericAdapterParams("ValidTopologyModificationParams");

    valid_pl_->
      set<double>("Fracture Stress",
                  1.0,
                  "Fracture stress value at which two elements separate");

    return valid_pl_;
  }


  //----------------------------------------------------------------------------
  void
  Albany::TopologyMod::showRelations()
  {
    std::vector<Entity*> element_list;
    stk::mesh::get_entities(*(bulk_data_),element_rank_,element_list);

    // Remove extra relations from element
    for (int i = 0; i < element_list.size(); ++i){
      Entity & element = *(element_list[i]);
      stk::mesh::PairIterRelation relations = element.relations();
      std::cout << "Element " << element_list[i]->identifier()
                << " relations are :" << std::endl;

      for (int j = 0; j < relations.size(); ++j){
        std::cout << "entity:\t" << relations[j].entity()->identifier() << ","
             << relations[j].entity()->entity_rank() << "\tlocal id: "
             << relations[j].identifier() << "\n";
      }
    }
  }

#ifdef ALBANY_MPI
  //----------------------------------------------------------------------------
  int
  Albany::TopologyMod::accumulateFractured(int num_fractured)
  {
    int total_fractured;

    stk::all_reduce_sum(bulk_data_->parallel(), &num_fractured, &total_fractured, 1);

    return total_fractured;
  }

  //----------------------------------------------------------------------------
  // Parallel all-gatherv function. Communicates local open list to
  // all processors to form global open list.
  void
  Albany::TopologyMod::
  getGlobalOpenList( std::map<EntityKey, bool>& local_entity_open,
                     std::map<EntityKey, bool>& global_entity_open) {

    // Make certain that we can send keys as MPI_UINT64_T types
    assert(sizeof(EntityKey::raw_key_type) >= sizeof(uint64_t));

    const unsigned parallel_size = bulk_data_->parallel_size();

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
        Entity *entity = bulk_data_->get_entity(me.first);
        std::cout<<"Single proc fracture list contains "<<" "<<entity_rank_name<<" ["<<entity_id<<"] Proc:"
        <<entity->owner_rank() <<std::endl;
      */

    }

    int num_open_on_pe = v.size();

    // Perform the allgatherv

    // gather the number of open entities on each processor
    int *sizes = new int[parallel_size];
    MPI_Allgather(&num_open_on_pe, 1, MPI_INT, sizes, 1, MPI_INT, bulk_data_->parallel());

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
                   sizes, offsets, MPI_UINT64_T, bulk_data_->parallel());

    // Save the global keys
    for(int i = 0; i < total_number_of_open_entities; i++){

      EntityKey key = EntityKey(&result_array[i]);
      global_entity_open[key] = true;

      // Debugging
      /*
        const unsigned entity_rank = stk::mesh::entity_rank( key);
        const stk::mesh::EntityId entity_id = stk::mesh::entity_id( key );
        const std::string & entity_rank_name = metaData->entity_rank_name( entity_rank );
        Entity *entity = bulk_data_->get_entity(key);
        if(!entity) { std::cout << "Error on this processor: Entity not addressible!!!!!!!!!!!!!" << std::endl;

        std::cout<<"Global proc fracture list contains "<<" "<<entity_rank_name<<" ["<<entity_id<<"]" << std::endl;
	std::vector<Entity*> element_lst;
	stk::mesh::get_entities(*(bulk_data_),elementRank,element_lst);
	for (int i = 0; i < element_lst.size(); ++i){
        std::cout << element_lst[i]->identifier() << std::endl;
        }
	std::vector<Entity*> entity_lst;
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
  Albany::TopologyMod::accumulateFractured(int num_fractured){
    return num_fractured;
  }

  //----------------------------------------------------------------------------
  // Parallel all-gatherv function. Communicates local open list to
  // all processors to form global open list.
  void
  Albany::TopologyMod::getGlobalOpenList( std::map<EntityKey, bool>& local_entity_open,
                                          std::map<EntityKey, bool>& global_entity_open){

    global_entity_open = local_entity_open;
  }
#endif

}

