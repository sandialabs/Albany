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

  if (this->worksetState) {
    saveWorksetState(workset);
  } else {
    saveElemState(workset);
    if (this->nodalState) {
      auto disc = workset.disc;
      auto last_ws = disc->getNumWorksets()-1;
      if (workset.wsIndex==last_ws) {
        auto mfa = disc->getMeshStruct()->get_field_accessor();
        mfa->transferElemStateToNodeState(stateName);
      }
    }
  }
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


} // namespace PHAL
