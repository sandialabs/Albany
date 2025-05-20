//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_DataLayout_MDALayout.hpp"

#include "PHAL_SaveStateField.hpp"

#include "Albany_AbstractSTKMeshStruct.hpp"
#include "Albany_AbstractSTKFieldContainer.hpp"
#include "Albany_AbstractDiscretization.hpp"

namespace PHAL {

template<typename EvalT, typename Traits>
SaveStateField<EvalT, Traits>::
SaveStateField(const Teuchos::ParameterList& /* p */)
{
  // States Not Saved for Generic Type, only Specializations
  this->setName("Save State Field" );
}

// **********************************************************************
template<typename EvalT, typename Traits>
void SaveStateField<EvalT, Traits>::postRegistrationSetup(typename Traits::SetupData /* d */,
                      PHX::FieldManager<Traits>& /* fm */)
{
  // States Not Saved for Generic Type, only Specializations
}

// **********************************************************************
template<typename EvalT, typename Traits>
void SaveStateField<EvalT, Traits>::evaluateFields(typename Traits::EvalData /* workset */)
{
  // States Not Saved for Generic Type, only Specializations
}
// **********************************************************************
// **********************************************************************
template<typename Traits>
SaveStateField<PHAL::AlbanyTraits::Residual, Traits>::
SaveStateField(const Teuchos::ParameterList& p)
{
  fieldName =  p.get<std::string>("Field Name");
  stateName =  p.get<std::string>("State Name");

  Teuchos::RCP<PHX::DataLayout> layout = p.get<Teuchos::RCP<PHX::DataLayout> >("State Field Layout");
  field = decltype(field)(fieldName, layout );

  TEUCHOS_TEST_FOR_EXCEPTION (
      layout->name(0)!=PHX::print<Cell>() &&
      layout->name(0)!=PHX::print<Dim>() &&
      layout->name(0)!=PHX::print<Dummy>(), std::runtime_error,
      "Error! Invalid state layout. Supported cases:\n"
      " - <Cell, Node [,Dim]>\n"
      " - <Cell, QuadPoint [,Dim]>\n"
      " - <Cell [,Dim [,Dim [,Dim]]]\n"
      " - <Dim [,Dim]>\n"
      " - <Dummy]>\n");
  if (layout->name(0) != PHX::print<Cell>()) {
    worksetState = true;
    nodalState = false;
    TEUCHOS_TEST_FOR_EXCEPTION (layout->rank()>2, Teuchos::Exceptions::InvalidParameter,
        "Error! Only rank<=2 workset states supported.\n");
  } else {
    worksetState = false;
    nodalState = layout->rank()>1 && layout->name(1)==PHX::print<Node>();
    TEUCHOS_TEST_FOR_EXCEPTION (nodalState && layout->rank()>3, Teuchos::Exceptions::InvalidParameter,
        "Error! Only scalar/vector nodal states supported.\n");
    TEUCHOS_TEST_FOR_EXCEPTION (!nodalState && layout->rank()>5, Teuchos::Exceptions::InvalidParameter,
        "Error! Only rank<=4 elem states supported.\n");
  }

  Teuchos::RCP<PHX::DataLayout> dummy = Teuchos::rcp(new PHX::MDALayout<Dummy>(0));
  savestate_operation = Teuchos::rcp(new PHX::Tag<ScalarT> (fieldName, dummy));

  this->addDependentField(field.fieldTag());
  this->addEvaluatedField(*savestate_operation);

  this->setName("Save Field " + fieldName +" to State " + stateName
                + "Residual");
}

// **********************************************************************
template<typename Traits>
void SaveStateField<PHAL::AlbanyTraits::Residual, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(field,fm);

  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields());
  if (d.memoizer_active()) memoizer.enable_memoizer();
}
// **********************************************************************
template<typename Traits>
void SaveStateField<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  if (memoizer.have_saved_data(workset,this->evaluatedFields())) return;

  if (this->nodalState)
    saveNodeState(workset);
  else if (this->worksetState)
    saveWorksetState(workset);
  else
    saveElemState(workset);
}

template<typename Traits>
void SaveStateField<PHAL::AlbanyTraits::Residual, Traits>::
saveElemState(typename Traits::EvalData workset)
{
  TEUCHOS_TEST_FOR_EXCEPTION (workset.stateArrayPtr->count(stateName)!=1, std::runtime_error,
      "[SaveStateField] Error: cannot locate elem state '" << stateName << "'\n");

  const auto field_d_view = field.get_view();
  const auto field_h_mirror = Kokkos::create_mirror_view(field_d_view);
  Kokkos::deep_copy(field_h_mirror, field_d_view);

  auto& state = workset.stateArrayPtr->at(stateName);
  auto state_h = state.host();
  switch (state_h.rank()) {
  case 1:
    for (unsigned int cell = 0; cell < workset.numCells; ++cell)
      state_h(cell) = field_h_mirror(cell);
    break;
  case 2:
    for (unsigned int cell = 0; cell < workset.numCells; ++cell)
      for (unsigned int qp = 0; qp < state_h.extent(1); ++qp)
        state_h(cell, qp) = field_h_mirror(cell,qp);;
    break;
  case 3:
    for (unsigned int cell = 0; cell < workset.numCells; ++cell)
      for (unsigned int qp = 0; qp < state_h.extent(1); ++qp)
        for (unsigned int i = 0; i < state_h.extent(2); ++i)
          state_h(cell, qp, i) = field_h_mirror(cell,qp,i);
    break;
  case 4:
    for (unsigned int cell = 0; cell < workset.numCells; ++cell)
      for (unsigned int qp = 0; qp < state_h.extent(1); ++qp)
        for (unsigned int i = 0; i < state_h.extent(2); ++i)
          for (unsigned int j = 0; j < state_h.extent(3); ++j)
            state_h(cell, qp, i, j) = field_h_mirror(cell,qp,i,j);
    break;
  case 5:
    for (unsigned int cell = 0; cell < workset.numCells; ++cell)
      for (unsigned int qp = 0; qp < state_h.extent(1); ++qp)
        for (unsigned int i = 0; i < state_h.extent(2); ++i)
          for (unsigned int j = 0; j < state_h.extent(3); ++j)
            for (unsigned int k = 0; k < state_h.extent(4); ++k)
            state_h(cell, qp, i, j, k) = field_h_mirror(cell,qp,i,j,k);
    break;
  default:
    TEUCHOS_TEST_FOR_EXCEPTION(false,std::runtime_error,
                        "Unexpected state rank in SaveStateField: " << state.rank());
  }
  state.sync_to_dev();
}

template<typename Traits>
void SaveStateField<PHAL::AlbanyTraits::Residual, Traits>::
saveWorksetState(typename Traits::EvalData workset)
{
  TEUCHOS_TEST_FOR_EXCEPTION (workset.globalStates.count(stateName)==1, std::runtime_error,
      "[SaveStateField] Error: cannot locate workset state '" << stateName << "'\n");

  // Get shards Array (from STK) for this state
  // Need to check if we can just copy full size -- can assume same ordering?
  auto& state = workset.globalStates.at(stateName);

  const auto field_d_view = field.get_view();
  const auto field_h_mirror = Kokkos::create_mirror_view(field_d_view);
  Kokkos::deep_copy(field_h_mirror, field_d_view);

  auto state_h = state.host();
  switch (state.rank()) {
  case 1:
    for (unsigned int idim = 0; idim < state_h.extent(0); ++idim)
      state_h(idim) = field_h_mirror(idim);
    break;
  case 2:
    for (unsigned int idim = 0; idim < state_h.extent(0); ++idim)
      for (unsigned int jdim = 0; jdim < state_h.extent(1); ++jdim)
        state_h(idim,jdim) = field_h_mirror(idim,jdim);
    break;
  default:
    TEUCHOS_TEST_FOR_EXCEPTION(false,std::runtime_error,
                        "Unexpected state rank in SaveStateField: " << state.rank());
  }
  state.sync_to_dev();
}

template<typename Traits>
void SaveStateField<PHAL::AlbanyTraits::Residual, Traits>::
saveNodeState(typename Traits::EvalData workset)
{
  // Note: to save nodal fields, we need to open up the mesh, and work directly on it.
  //       For this reason, we assume the mesh is stk. Moreover, since the cell buckets
  //       and node buckets do not coincide (elem-bucket 12 does not necessarily contain
  //       all nodes in node-bucket 12), we need to work with stk fields. To do this we
  //       must extract entities from the bulk data and use them to access the values
  //       of the stk field.

  Teuchos::RCP<Albany::AbstractDiscretization> disc = workset.disc;
  TEUCHOS_TEST_FOR_EXCEPTION (disc==Teuchos::null, std::runtime_error, "Error! Discretization is needed to save nodal state.\n");

  Teuchos::RCP<Albany::AbstractSTKMeshStruct> mesh = Teuchos::rcp_dynamic_cast<Albany::AbstractSTKMeshStruct>(disc->getMeshStruct());
  TEUCHOS_TEST_FOR_EXCEPTION (mesh==Teuchos::null, std::runtime_error, "Error! Save nodal states available only for stk meshes.\n");

  stk::mesh::MetaData& metaData = *mesh->metaData;
  stk::mesh::BulkData& bulkData = *mesh->bulkData;

  using SFT = Albany::AbstractSTKFieldContainer::STKFieldType;
  SFT* stk_field = metaData.get_field<double> (stk::topology::NODE_RANK, stateName);
  TEUCHOS_TEST_FOR_EXCEPTION (stk_field==nullptr, std::runtime_error, "Error! Field not found.\n");
  std::vector<PHX::DataLayout::size_type> dims;
  field.dimensions(dims);

  const auto& node_dof_mgr = workset.disc->getNodeDOFManager();
  const auto& elem_lids = disc->getElementLIDs_host(workset.wsIndex);

  const auto& cell_indexer = node_dof_mgr->cell_indexer();

  const auto field_d_view = field.get_view();
  const auto field_h_mirror = Kokkos::create_mirror_view(field_d_view);
  Kokkos::deep_copy(field_h_mirror, field_d_view);

  double* values;
  stk::mesh::Entity e;
  switch (dims.size()) {
    case 2:   // node_scalar
      for (unsigned int cell=0; cell<workset.numCells; ++cell) {
        const int elem_LID = elem_lids[cell];
        const GO  elem_GID = cell_indexer->getGlobalElement(elem_LID);
        const auto& elem = bulkData.get_entity(stk::topology::ELEM_RANK,elem_GID+1);
        const auto* nodes = bulkData.begin_nodes(elem);
        for (unsigned int node=0; node<dims[1]; ++node) {
          values = stk::mesh::field_data(*stk_field, nodes[node]);
          values[0] = field_h_mirror(cell,node);
        }
      }
      break;
    case 3:   // node_vector
      for (unsigned int cell=0; cell<workset.numCells; ++cell) {
        const int elem_LID = elem_lids[cell];
        const GO  elem_GID = cell_indexer->getGlobalElement(elem_LID);
        const auto& elem = bulkData.get_entity(stk::topology::ELEM_RANK,elem_GID+1);
        const auto* nodes = bulkData.begin_nodes(elem);
        for (unsigned int node=0; node<dims[1]; ++node) {
          values = stk::mesh::field_data(*stk_field, nodes[node]);
          for (unsigned int i=0; i<dims[2]; ++i)
            values[i] = field_h_mirror(cell,node,i);
        }
      }
      break;
    default:  // error!
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Unexpected field dimension (only node_scalar/node_vector for now).\n");
  }
}

} // namespace PHAL
