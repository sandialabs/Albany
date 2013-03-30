//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_CopyRemesh.hpp"

#include "Teuchos_TimeMonitor.hpp"

namespace Albany {

  typedef stk::mesh::Entity Entity;
  typedef stk::mesh::EntityRank EntityRank;
  typedef stk::mesh::RelationIdentifier EdgeId;
  typedef stk::mesh::EntityKey EntityKey;

  //----------------------------------------------------------------------------
  Albany::CopyRemesh::
  CopyRemesh(const Teuchos::RCP<Teuchos::ParameterList>& params,
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

  }

  //----------------------------------------------------------------------------
  Albany::CopyRemesh::
  ~CopyRemesh()
  {
  }

  //----------------------------------------------------------------------------
  bool
  Albany::CopyRemesh::queryAdaptationCriteria(){

    int remesh_iter = adapt_params_->get<int>("Remesh Step Number");

    if(iter == remesh_iter)
      return true;

    return false; 
  }

  //----------------------------------------------------------------------------
  bool
  Albany::CopyRemesh::adaptMesh(){

    std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";
    std::cout << "Adapting mesh using Albany::CopyRemesh method       \n";
    std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";

    // Save the current results and close the exodus file

    // Create a remeshed output file naming convention by adding the
    // remeshFileIndex ahead of the period

    std::ostringstream ss;
    std::string str = base_exo_filename_;
    ss << "_" << remesh_file_index_ << ".";
    str.replace(str.find('.'), 1, ss.str());

    std::cout << "Remeshing: renaming output file to - " << str << endl;

    // Open the new exodus file for results
    stk_discretization_->reNameExodusOutput(str);

    remesh_file_index_++;

    // do remeshing right here if we were doing any...

    // Throw away all the Albany data structures and re-build them
    // from the mesh

    stk_discretization_->updateMesh();

    return true;

  }

  //----------------------------------------------------------------------------
  //
  // Transfer solution between meshes.
  //
  void
  Albany::CopyRemesh::
  solutionTransfer(const Epetra_Vector& oldSolution,
                   Epetra_Vector& newSolution){

    TEUCHOS_TEST_FOR_EXCEPT(oldSolution.MyLength() != newSolution.MyLength());
    newSolution = oldSolution;
  }

  //----------------------------------------------------------------------------
  Teuchos::RCP<const Teuchos::ParameterList>
  Albany::CopyRemesh::getValidAdapterParameters() const
  {
    Teuchos::RCP<Teuchos::ParameterList> valid_pl =
      this->getGenericAdapterParams("ValidCopyRemeshParameters");

    valid_pl->set<int>("Remesh Step Number",
                       1,
                       "Iteration step at which to remesh the problem");

    return valid_pl;
  }
  //----------------------------------------------------------------------------
}
