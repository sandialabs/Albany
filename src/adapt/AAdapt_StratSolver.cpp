//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "AAdapt_CopyRemesh.hpp"

#include "Teuchos_TimeMonitor.hpp"

namespace AAdapt {

typedef stk_classic::mesh::Entity Entity;
typedef stk_classic::mesh::EntityRank EntityRank;
typedef stk_classic::mesh::RelationIdentifier EdgeId;
typedef stk_classic::mesh::EntityKey EntityKey;

//----------------------------------------------------------------------------
AAdapt::CopyRemesh::
CopyRemesh(const Teuchos::RCP<Teuchos::ParameterList>& params,
           const Teuchos::RCP<ParamLib>& param_lib,
           Albany::StateManager& state_mgr,
           const Teuchos::RCP<const Epetra_Comm>& comm) :
  AAdapt::AbstractAdapter(params, param_lib, state_mgr, comm),
  remesh_file_index_(1) {

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

  num_dim_ = stk_mesh_struct_->numDim;

  // Save the initial output file name
  base_exo_filename_ = stk_mesh_struct_->exoOutFile;

}

//----------------------------------------------------------------------------
AAdapt::CopyRemesh::
~CopyRemesh() {
}

//----------------------------------------------------------------------------
bool
AAdapt::CopyRemesh::queryAdaptationCriteria() {

  if(adapt_params_->get<std::string>("Remesh Strategy", "None").compare("Continuous") == 0){

    if(iter > 1)

      return true;

    else

      return false;

  }

  Teuchos::Array<int> remesh_iter = adapt_params_->get<Teuchos::Array<int> >("Remesh Step Number");

  for(int i = 0; i < remesh_iter.size(); i++)

    if(iter == remesh_iter[i])

      return true;

  return false;

}

//----------------------------------------------------------------------------
bool
AAdapt::CopyRemesh::adaptMesh(const Epetra_Vector& solution, const Epetra_Vector& ovlp_solution) {

  std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";
  std::cout << "Adapting mesh using AAdapt::CopyRemesh method       \n";
  std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";

  // Save the current results and close the exodus file

  // Create a remeshed output file naming convention by adding the
  // remeshFileIndex ahead of the period

  std::ostringstream ss;
  std::string str = base_exo_filename_;
  ss << "_" << remesh_file_index_ << ".";
  str.replace(str.find('.'), 1, ss.str());

  std::cout << "Remeshing: renaming output file to - " << str << std::endl;

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
AAdapt::CopyRemesh::
solutionTransfer(const Epetra_Vector& oldSolution,
                 Epetra_Vector& newSolution) {

  TEUCHOS_TEST_FOR_EXCEPT(oldSolution.MyLength() != newSolution.MyLength());
  newSolution = oldSolution;
}

//----------------------------------------------------------------------------
Teuchos::RCP<const Teuchos::ParameterList>
AAdapt::CopyRemesh::getValidAdapterParameters() const {
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericAdapterParams("ValidCopyRemeshParameters");

  Teuchos::Array<int> defaultArgs;

  validPL->set<Teuchos::Array<int> >("Remesh Step Number", defaultArgs, "Iteration step at which to remesh the problem");
  validPL->set<std::string>("Remesh Strategy", "", "Strategy to use when remeshing: Continuous - remesh every step.");

  return validPL;
}
//----------------------------------------------------------------------------
}
