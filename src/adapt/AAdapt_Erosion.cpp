//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <Teuchos_TimeMonitor.hpp>
#include <stk_util/parallel/ParallelReduce.hpp>

#include "AAdapt_Erosion.hpp"
#include "Albany_Utils.hpp"
#include "LCM/utils/topology/Topology_FailureCriterion.h"

namespace AAdapt {

//
//
//
AAdapt::Erosion::Erosion(
    Teuchos::RCP<Teuchos::ParameterList> const& params,
    Teuchos::RCP<ParamLib> const&               param_lib,
    Albany::StateManager const&                 state_mgr,
    Teuchos::RCP<Teuchos_Comm const> const&     comm)
    : AAdapt::AbstractAdapter(params, param_lib, state_mgr, comm),
      remesh_file_index_(1)
{
  discretization_ = state_mgr_.getDiscretization();

  stk_discretization_ =
      static_cast<Albany::STKDiscretization*>(discretization_.get());

  stk_mesh_struct_ = stk_discretization_->getSTKMeshStruct();

  bulk_data_ = stk_mesh_struct_->bulkData;
  meta_data_ = stk_mesh_struct_->metaData;

  num_dim_ = stk_mesh_struct_->numDim;

  // Save the initial output file name
  base_exo_filename_ = stk_mesh_struct_->exoOutFile;

  topology_ = Teuchos::rcp(new LCM::Topology(discretization_));

  std::string const failure_indicator_name = "ACE Failure Indicator";

  failure_criterion_ = Teuchos::rcp(
      new LCM::BulkFailureCriterion(*topology_, failure_indicator_name));

  topology_->set_failure_criterion(failure_criterion_);
}

//
//
//
bool
AAdapt::Erosion::queryAdaptationCriteria(int)
{
  return topology_->there_are_failed_cells();
}

//
//
//
bool
AAdapt::Erosion::adaptMesh()
{
  *output_stream_ << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
                  << "Adapting mesh using AAdapt::Erosion method      \n"
                  << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";

  // Save the current results and close the exodus file

  // Create a remeshed output file naming convention by
  // adding the remesh_file_index_ ahead of the period
  std::ostringstream ss;
  std::string        str = base_exo_filename_;
  ss << "_" << remesh_file_index_ << ".";
  str.replace(str.find('.'), 1, ss.str());

  *output_stream_ << "Remeshing: renaming output file to - " << str << '\n';

  // Open the new exodus file for results
  stk_discretization_->reNameExodusOutput(str);

  remesh_file_index_++;

  std::cout << "**** BEFORE EROSION ****\n";
  Albany::printInternalElementStates(this->state_mgr_);
  topology_->printFailureState(std::cout);
  // Start the mesh update process

  topology_->erodeFailedElements();

  // Throw away all the Albany data structures and re-build them from the mesh

  stk_discretization_->updateMesh();

  std::cout << "**** AFTER EROSION ****\n";
  Albany::printInternalElementStates(this->state_mgr_);
  topology_->printFailureState(std::cout);

  return true;
}

//
//
//
Teuchos::RCP<Teuchos::ParameterList const>
AAdapt::Erosion::getValidAdapterParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> valid_pl_ =
      this->getGenericAdapterParams("ValidErosionParams");

  return valid_pl_;
}

}  // namespace AAdapt
