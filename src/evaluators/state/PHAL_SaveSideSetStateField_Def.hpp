//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_GeneralPurposeFieldsNames.hpp"
#include "Albany_Layouts.hpp"
#include "PHAL_SaveSideSetStateField.hpp"
#include "Albany_STKDiscretization.hpp"

#include "Phalanx_DataLayout.hpp"
#include "Teuchos_TestForException.hpp"
#include <Teuchos_RCPDecl.hpp>
#include <stdexcept>

namespace PHAL
{

// **********************************************************************
template<typename Traits>
SaveSideSetStateField<PHAL::AlbanyTraits::Residual, Traits>::
SaveSideSetStateField (const Teuchos::ParameterList& p,
                       const Teuchos::RCP<Albany::Layouts>& dl)
{
  TEUCHOS_TEST_FOR_EXCEPTION (!dl->isSideLayouts, std::runtime_error,
      "Error! Input Layouts struct is not that of a sideset.\n");

  sideSetName   = p.get<std::string>("Side Set Name");

  fieldName = p.get<std::string>("Field Name");
  stateName = p.get<std::string>("State Name");

  rank = p.get<FRT>("Field Rank");
  loc  = p.get<FL>("Field Location");
  TEUCHOS_TEST_FOR_EXCEPTION (loc!=FL::Cell && loc!=FL::Node, std::runtime_error,
      "Error! Only Node and Cell field location supported.\n")
  TEUCHOS_TEST_FOR_EXCEPTION (loc==FL::Node && rank==FRT::Gradient, std::runtime_error,
      "Error! Gradient fields only supported if at Cell location.\n");

  auto layout = Albany::get_field_layout(rank,loc,dl);
  
  field = decltype(field)(fieldName, layout);

  savestate_operation = Teuchos::rcp(new PHX::Tag<ScalarT>(fieldName, dl->dummy));

  this->addDependentField (field.fieldTag());
  this->addEvaluatedField (*savestate_operation);

  if (rank==FRT::Gradient) {
    const auto tangents_name = Albany::tangents_name + "_" + sideSetName;
    const auto w_meas_name   = Albany::weighted_measure_name + "_" + sideSetName;

    tangents = decltype(tangents)(tangents_name, dl->qp_tensor_cd_sd);
    this->addDependentField(tangents);

    w_measure = decltype(w_measure)(tangents_name, dl->qp_tensor_cd_sd);
    this->addDependentField(w_measure);
  }

  numQPs = dl->qp_scalar->dimension(1);
  numNodes = dl->node_scalar->dimension(1);

  this->setName ("Save Side Set Field " + fieldName + " to Side Set State " + stateName + " <Residual>");
}

// **********************************************************************
template<typename Traits>
void SaveSideSetStateField<PHAL::AlbanyTraits::Residual, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(field,fm);
  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields());
  if (d.memoizer_active()) memoizer.enable_memoizer();
}
// **********************************************************************
template<typename Traits>
void SaveSideSetStateField<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  if (memoizer.have_saved_data(workset,this->evaluatedFields())) return;

  TEUCHOS_TEST_FOR_EXCEPTION (workset.sideSets==Teuchos::null, std::logic_error,
      "Error! The mesh does not store any side set.\n");

  if (workset.sideSetViews->find(sideSetName)==workset.sideSetViews->end())
    return; // Side set not present in this workset

  TEUCHOS_TEST_FOR_EXCEPTION (workset.disc==Teuchos::null, std::logic_error,
      "Error! The workset must store a valid discretization pointer.\n");

  const auto& ssDiscs = workset.disc->getSideSetDiscretizations();

  TEUCHOS_TEST_FOR_EXCEPTION (ssDiscs.size()==0, std::logic_error,
      "Error! The discretization must store side set discretizations.\n");

  TEUCHOS_TEST_FOR_EXCEPTION (ssDiscs.find(sideSetName)==ssDiscs.end(), std::logic_error,
      "Error! No discretization found for side set " << sideSetName << ".\n");

  TEUCHOS_TEST_FOR_EXCEPTION (ssDiscs.at(sideSetName)==Teuchos::null, std::logic_error,
        "Error! Side discretization is invalid for side set " << sideSetName << ".\n");

  if (loc==FL::Node and workset.wsIndex == (workset.numWs-1)) {
    saveNodeState(workset);
    // Transfer the (Cell,Node,...) state to the purely nodal state
    auto ss_disc = workset.disc->getSideSetDiscretizations().at(sideSetName);
    auto ss_mfa = ss_disc->getMeshStruct()->get_field_accessor();
    ss_mfa->transferElemStateToNodeState (stateName);
  } else {
    saveElemState(workset);
  }

}

template<typename Traits>
void SaveSideSetStateField<PHAL::AlbanyTraits::Residual, Traits>::
saveElemState(typename Traits::EvalData workset)
{
  const auto& ss_disc = workset.disc->getSideSetDiscretizations().at(sideSetName);
  const auto& side_to_ss_cell = workset.disc->getSideToSideSetCellMap().at(sideSetName);
  const auto  ss_cell_indexer = ss_disc->getCellsGlobalLocalIndexer();
  const auto  ss_elem_ws_idx = ss_disc->get_elements_workset_idx();

  const auto field_d_view = field.get_view();
  const auto field_h_mirror = Kokkos::create_mirror_view(field_d_view);
  Kokkos::deep_copy(field_h_mirror, field_d_view);

  auto ss_mfa = ss_disc->getMeshStruct()->get_field_accessor();
  auto& state = ss_mfa->getElemStates()[workset.wsIndex].at(stateName);
  auto state_h = state.host();

  // Loop on the sides of this sideSet that are in this workset
  auto sideSet = workset.sideSetViews->at(sideSetName);
  double tan_cell_val, meas;
  for (int sideSet_idx = 0; sideSet_idx < sideSet.size; ++sideSet_idx) {
    // Get the side GID
    const int side_GID = sideSet.side_GID.h_view(sideSet_idx);
    const GO ss_cell_GID = side_to_ss_cell.at(side_GID);
    const int ss_cell_LID = ss_cell_indexer->getLocalElement(ss_cell_GID);

    const int icell = ss_elem_ws_idx[ss_cell_LID].idx;

    switch (rank) {
      case FRT::Scalar:
        state_h(icell) = field_h_mirror(sideSet_idx);
        break;
      case FRT::Vector:
        for (int idim=0; idim<state_h.extent_int(1); ++idim) {
          state_h(icell,idim) = field_h_mirror(sideSet_idx,idim);
        }
        break;
      case FRT::Gradient:
        meas = 0;
        for (int qp=0; qp<numQPs; ++qp) {
          meas += w_measure(sideSet_idx,qp);
        }
        for (int idim=0; idim<state_h.extent_int(1); ++idim) {
          state_h(icell,idim) = 0;
          for (int itan=0; itan<state_h.extent_int(1); ++itan) {
            tan_cell_val = 0;
            for (int qp=0; qp<numQPs; ++qp) {
              tan_cell_val += tangents(sideSet_idx,qp,idim,itan)*w_measure(sideSet_idx,qp);
            }
            state_h(icell,idim) +=  (tan_cell_val/meas) * field_h_mirror(sideSet_idx,itan);
          }
        }
        break;
      case FRT::Tensor:
        for (int idim=0; idim<state_h.extent_int(1); ++idim) {
          for (int jdim=0; jdim<state_h.extent_int(2); ++jdim) {
            state_h(icell,idim,jdim) = field_h_mirror(sideSet_idx,idim,jdim);
        }}
        break;
    }
  }
  state.sync_to_dev();
}

template<typename Traits>
void SaveSideSetStateField<PHAL::AlbanyTraits::Residual, Traits>::
saveNodeState(typename Traits::EvalData workset)
{
  const auto& ss_disc = workset.disc->getSideSetDiscretizations().at(sideSetName);
  const auto& side_to_ss_cell = workset.disc->getSideToSideSetCellMap().at(sideSetName);
  const auto& side_to_node_map = workset.disc->getSideNodeNumerationMap().at(sideSetName);
  const auto  ss_cell_indexer = ss_disc->getCellsGlobalLocalIndexer();
  const auto  ss_elem_ws_idx = ss_disc->get_elements_workset_idx();

  const auto field_d_view = field.get_view();
  const auto field_h_mirror = Kokkos::create_mirror_view(field_d_view);
  Kokkos::deep_copy(field_h_mirror, field_d_view);

  auto ss_mfa = ss_disc->getMeshStruct()->get_field_accessor();
  auto& state = ss_mfa->getElemStates()[workset.wsIndex].at(stateName);
  auto state_h = state.host();

  // Loop on the sides of this sideSet that are in this workset
  auto sideSet = workset.sideSetViews->at(sideSetName);
  double tan_cell_val, meas;
  for (int sideSet_idx = 0; sideSet_idx < sideSet.size; ++sideSet_idx) {
    // Get the side GID
    const int side_GID = sideSet.side_GID.h_view(sideSet_idx);
    const GO ss_cell_GID = side_to_ss_cell.at(side_GID);
    const int ss_cell_LID = ss_cell_indexer->getLocalElement(ss_cell_GID);
    const auto& node_map = side_to_node_map.at(side_GID);

    const int icell = ss_elem_ws_idx[ss_cell_LID].idx;

    for (int side_node=0; side_node<numNodes; ++side_node) {
      auto ss_cell_node = node_map[side_node];
      switch (rank) {
        case FRT::Scalar:
          state_h(icell,ss_cell_node) = field_h_mirror(sideSet_idx,side_node);
          break;
        case FRT::Vector:
          for (int idim=0; idim<state_h.extent_int(1); ++idim) {
            state_h(icell,ss_cell_node,idim) = field_h_mirror(sideSet_idx,side_node,idim);
          }
          break;
        case FRT::Gradient:
          meas = 0;
          for (int qp=0; qp<numQPs; ++qp) {
            meas += w_measure(sideSet_idx,qp);
          }
          for (int idim=0; idim<state_h.extent_int(1); ++idim) {
            state_h(icell,ss_cell_node,idim) = 0;
            for (int itan=0; itan<state_h.extent_int(1); ++itan) {
              tan_cell_val = 0;
              for (int qp=0; qp<numQPs; ++qp) {
                tan_cell_val += tangents(sideSet_idx,qp,idim,itan)*w_measure(sideSet_idx,qp);
              }
              state_h(icell,ss_cell_node,idim) +=  (tan_cell_val/meas) * field_h_mirror(sideSet_idx,side_node,itan);
            }
          }
          break;
        case FRT::Tensor:
          for (int idim=0; idim<state_h.extent_int(1); ++idim) {
            for (int jdim=0; jdim<state_h.extent_int(2); ++jdim) {
              state_h(icell,ss_cell_node,idim,jdim) = field_h_mirror(sideSet_idx,side_node,idim,jdim);
          }}
          break;
      }
    }
  }
  state.sync_to_dev();
}

} // Namespace PHAL
