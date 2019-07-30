//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "AAdapt_Erosion.hpp"
#include <Teuchos_TimeMonitor.hpp>
#include <stk_util/parallel/ParallelReduce.hpp>
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
  discretization_     = state_mgr_.getDiscretization();
  auto* pdisc         = discretization_.get();
  stk_discretization_ = static_cast<Albany::STKDiscretization*>(pdisc);
  stk_mesh_struct_    = stk_discretization_->getSTKMeshStruct();
  bulk_data_          = stk_mesh_struct_->bulkData;
  meta_data_          = stk_mesh_struct_->metaData;
  num_dim_            = stk_mesh_struct_->numDim;

  // Save the initial output file name
  base_exo_filename_ = stk_mesh_struct_->exoOutFile;
  topology_          = Teuchos::rcp(new LCM::Topology(discretization_));
  std::string const failure_indicator_name = "ACE Failure Indicator";
  failure_criterion_                       = Teuchos::rcp(
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

namespace {

void
copyStateArray(
    Albany::StateArrayVec const&      src,
    Albany::StateArrayVec&            dst,
    std::vector<std::vector<double>>& store)
{
  auto const num_ws = src.size();
  dst.resize(num_ws);
  store.resize(num_ws);
  for (auto ws = 0; ws < num_ws; ++ws) {
    auto&& src_map = src[ws];
    auto&& dst_map = dst[ws];
    for (auto&& kv : src_map) {
      auto&&     state_name = kv.first;
      auto&&     src_states = kv.second;
      auto&&     dst_states = dst_map[state_name];
      auto const num_states = src_states.size();
      auto const rank       = src_states.rank();
      using DimT            = decltype(src_states.dimension(0));
      using TagT            = decltype(src_states.tag(0));
      std::vector<DimT> dims(rank);
      std::vector<TagT> tags(rank);
      for (auto i = 0; i < rank; ++i) {
        dims[i] = src_states.dimension(i);
        tags[i] = src_states.tag(i);
      }
      store[ws].resize(num_states);
      auto*   pval = &store[ws][0];
      auto*   pdim = &dims[0];
      auto*   ptag = &tags[0];
      MDArray mda(pval, rank, pdim, ptag);
      dst_states = mda;
    }
  }
}

}  // anonymous namespace

//
//
//
void
AAdapt::Erosion::copyStateArrays(Albany::StateArrays const& sa)
{
  auto&& src_esa = sa.elemStateArrays;
  auto&& dst_esa = state_arrays_.elemStateArrays;
  copyStateArray(src_esa, dst_esa, cell_state_store_);
  auto&& src_nsa = sa.nodeStateArrays;
  auto&& dst_nsa = state_arrays_.nodeStateArrays;
  copyStateArray(src_nsa, dst_nsa, node_state_store_);
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

  // AQUI
  std::cout << "**** BEFORE EROSION ****\n";
  Albany::printInternalElementStates(this->state_mgr_);
  topology_->printFailureState(std::cout);
  stk_discretization_->printElemGIDws(std::cout);
  auto&& state_arrays = stk_discretization_->getStateArrays();
  copyStateArrays(state_arrays);

  // Start the mesh update process
  topology_->erodeFailedElements();

  // Throw away all the Albany data structures and re-build them from the mesh
  stk_discretization_->updateMesh();

  std::cout << "**** AFTER EROSION ****\n";
  Albany::printInternalElementStates(this->state_mgr_);
  topology_->printFailureState(std::cout);
  stk_discretization_->printElemGIDws(std::cout);
  return true;
}

//
//
//
Teuchos::RCP<Teuchos::ParameterList const>
AAdapt::Erosion::getValidAdapterParameters() const
{
  auto valid_pl_ = this->getGenericAdapterParams("ValidErosionParams");
  return valid_pl_;
}

}  // namespace AAdapt
