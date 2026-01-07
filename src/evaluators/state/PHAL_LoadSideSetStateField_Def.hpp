//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "PHAL_LoadSideSetStateField.hpp"

#include "Albany_STKDiscretization.hpp"

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Shards_CellTopology.hpp"

namespace PHAL
{

template<typename EvalT, typename Traits, typename ScalarType>
LoadSideSetStateFieldBase<EvalT, Traits, ScalarType>::
LoadSideSetStateFieldBase (const Teuchos::ParameterList& p)
{
  sideSetName = p.get<std::string>("Side Set Name");

  fieldName = p.get<std::string>("Field Name");
  stateName = p.get<std::string>("State Name");

  field  = PHX::MDField<ScalarType>(fieldName, p.get<Teuchos::RCP<PHX::DataLayout> >("Field Layout") );

  this->addEvaluatedField (field);

  const auto& phx_dl = field.fieldTag().dataLayout();
  TEUCHOS_TEST_FOR_EXCEPTION(phx_dl.name(0)!=PHX::print<Side>(), std::runtime_error,
      "Error! To load a side-set state, the first tag of the layout MUST be 'Side'.\n");

  const auto rank = phx_dl.rank();
  nodalState = rank>1 ? phx_dl.name(1)==PHX::print<Node>() : false;
  if (nodalState) {
    TEUCHOS_TEST_FOR_EXCEPTION (rank!=2 && rank!=3, std::runtime_error,
        "Error! Only Scalar and Vector field supported for nodal states.\n");
    TEUCHOS_TEST_FOR_EXCEPTION(
        rank>2 && phx_dl.name(2)!=PHX::print<Dim>() && phx_dl.name(2)!=PHX::print<LayerDim>(),
        std::runtime_error,
        "Error! To load a side-set nodal state, the third tag (if present) MUST be 'Dim'.\n")
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION (rank!=1 && rank!=2, std::runtime_error,
        "Error! Only Scalar and Vector field supported for elem states.\n");
    TEUCHOS_TEST_FOR_EXCEPTION(
        rank>2 && phx_dl.name(2)!=PHX::print<Dim>() && phx_dl.name(2)!=PHX::print<LayerDim>(),
         std::runtime_error,
        "Error! To save a side-set elem state, the second tag (if present) MUST be 'Dim'.\n");
  }

  this->setName ("Load Side Set Field " + fieldName + " from Side Set State " + stateName 
    + PHX::print<EvalT>());
}

// **********************************************************************
template<typename EvalT, typename Traits, typename ScalarType>
void LoadSideSetStateFieldBase<EvalT, Traits, ScalarType>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(field,fm);

  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields());
  if (d.memoizer_active()) memoizer.enable_memoizer();
}

template<typename EvalT, typename Traits, typename ScalarType>
void LoadSideSetStateFieldBase<EvalT, Traits, ScalarType>::
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

  TEUCHOS_TEST_FOR_EXCEPTION (ssDiscs.find(sideSetName)==ssDiscs.end(), std::logic_error,
      "Error! No discretization found for side set " << sideSetName << ".\n");

  TEUCHOS_TEST_FOR_EXCEPTION (ssDiscs.at(sideSetName)==Teuchos::null, std::logic_error,
        "Error! Side discretization is invalid for side set " << sideSetName << ".\n");

  if (this->nodalState)
    loadNodeState(workset);
  else
    loadElemState(workset);
}

template<typename EvalT, typename Traits, typename ScalarType>
void LoadSideSetStateFieldBase<EvalT, Traits, ScalarType>::
loadNodeState(typename Traits::EvalData workset)
{
  const auto& ss_disc = workset.disc->getSideSetDiscretizations().at(sideSetName);
  const auto& side_to_ss_cell = workset.disc->getSideToSideSetCellMap().at(sideSetName);
  const auto& side_to_node_map = workset.disc->getSideNodeNumerationMap().at(sideSetName);
  const auto  ss_cell_indexer = ss_disc->getCellsGlobalLocalIndexer();
  const auto  ss_elem_ws_idx = ss_disc->get_elements_workset_idx();

  const auto field_d_view = field.get_view();
  const auto field_h_mirror = Kokkos::create_mirror_view(field_d_view);

  auto ss_mfa = ss_disc->getMeshStruct()->get_field_accessor();
  auto& state = ss_mfa->getElemStates()[workset.wsIndex].at(stateName);
  state.sync_to_host();
  auto state_h = state.host();

  // Loop on the sides of this sideSet that are in this workset
  // TODO: use state dev view
  auto sideSet = workset.sideSetViews->at(sideSetName);
  for (int sideSet_idx = 0; sideSet_idx < sideSet.size; ++sideSet_idx) {
    // Get the side GID
    const int side_GID = sideSet.side_GID.view_host()(sideSet_idx);
    const GO ss_cell_GID = side_to_ss_cell.at(side_GID);
    const int ss_cell_LID = ss_cell_indexer->getLocalElement(ss_cell_GID);
    const auto& node_map = side_to_node_map.at(side_GID);

    const int icell = ss_elem_ws_idx[ss_cell_LID].idx;

    for (int side_node=0; side_node<state_h.extent_int(1); ++side_node) {
      auto ss_cell_node = node_map[side_node];
      switch (state_h.rank()) {
        case 2:
          field_h_mirror(sideSet_idx,side_node) = state_h(icell,ss_cell_node);
          break;
        case 3:
          for (int idim=0; idim<state_h.extent_int(2); ++idim) {
            field_h_mirror(sideSet_idx,side_node,idim) = state_h(icell,ss_cell_node,idim);
          }
          break;
        case 4:
          for (int idim=0; idim<state_h.extent_int(2); ++idim) {
            for (int jdim=0; jdim<state_h.extent_int(3); ++jdim) {
              field_h_mirror(sideSet_idx,side_node,idim,jdim) = state_h(icell,ss_cell_node,idim,jdim);
          }}
          break;
      }
    }
  }
  Kokkos::deep_copy(field.get_view(), field_h_mirror);
}

template<typename EvalT, typename Traits, typename ScalarType>
void LoadSideSetStateFieldBase<EvalT, Traits, ScalarType>::
loadElemState(typename Traits::EvalData workset)
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
  state.sync_to_host();
  auto state_h = state.host();

  // Loop on the sides of this sideSet that are in this workset
  auto sideSet = workset.sideSetViews->at(sideSetName);
  for (int sideSet_idx = 0; sideSet_idx < sideSet.size; ++sideSet_idx) {
    // Get the side GID
    const int side_GID = sideSet.side_GID.view_host()(sideSet_idx);
    const GO ss_cell_GID = side_to_ss_cell.at(side_GID);
    const int ss_cell_LID = ss_cell_indexer->getLocalElement(ss_cell_GID);

    const int icell = ss_elem_ws_idx[ss_cell_LID].idx;

    switch (state_h.rank()) {
      case 1:
        field_h_mirror(sideSet_idx) = state_h(icell);
        break;
      case 2:
        for (int idim=0; idim<state_h.extent_int(1); ++idim) {
          field_h_mirror(sideSet_idx,idim) = state_h(icell,idim);
        }
        break;
      case 3:
        for (int idim=0; idim<state_h.extent_int(1); ++idim) {
          for (int jdim=0; jdim<state_h.extent_int(2); ++jdim) {
            field_h_mirror(sideSet_idx,idim,jdim) = state_h(icell,idim,jdim);
        }}
        break;
    }
  }
}

} // Namespace PHAL
