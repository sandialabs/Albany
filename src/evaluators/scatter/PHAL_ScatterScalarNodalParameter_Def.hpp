//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "PHAL_ScatterScalarNodalParameter.hpp"
#include "Albany_DistributedParameterLibrary.hpp"
#include "Albany_AbstractDiscretization.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_GlobalLocalIndexer.hpp"

namespace PHAL {

template<typename EvalT, typename Traits>
ScatterScalarNodalParameterBase<EvalT,Traits>::
ScatterScalarNodalParameterBase(const Teuchos::ParameterList& p,
                               const Teuchos::RCP<Albany::Layouts>& dl)
{
  param_name = p.get<std::string>("Parameter Name");
  std::string field_name = p.isParameter("Field Name") ? p.get<std::string>("Field Name") : param_name;
  val = decltype(val)(field_name,dl->node_scalar);
  numNodes = 0;

  this->addDependentField(val);

  this->setName("Scatter Nodal Parameter" );
}

// **********************************************************************
template<typename EvalT, typename Traits>
void ScatterScalarNodalParameterBase<EvalT,Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val,fm);
  numNodes = val.extent(1);

  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields());
  if (d.memoizer_active()) memoizer.enable_memoizer();
}

// **********************************************************************
template<typename EvalT, typename Traits>
void ScatterScalarNodalParameter<EvalT, Traits>::
evaluateFields(typename Traits::EvalData /* workset */)
{
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error,
    "PHAL::ScatterScalarNodalParameter is supposed to be used only for EvalT=Residual\n");
}

// **********************************************************************
template<typename EvalT, typename Traits>
void ScatterScalarExtruded2DNodalParameter<EvalT, Traits>::
evaluateFields(typename Traits::EvalData /* workset */)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "PHAL::ScatterScalarNodalParameter is supposed to be used only for Residual evaluation Type.");
}

// **********************************************************************
// Specialization: Residual
// **********************************************************************

template<typename Traits>
ScatterScalarNodalParameter<PHAL::AlbanyTraits::Residual, Traits>::
ScatterScalarNodalParameter(const Teuchos::ParameterList& p,
                            const Teuchos::RCP<Albany::Layouts>& dl)
  : ScatterScalarNodalParameterBase<PHAL::AlbanyTraits::Residual, Traits>(p,dl)
{
  // Create field tag
  nodal_field_tag =
    Teuchos::rcp(new PHX::Tag<ParamScalarT>(className, dl->dummy));

  this->addEvaluatedField(*nodal_field_tag);
}

// **********************************************************************
template<typename Traits>
void ScatterScalarNodalParameter<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // Check for early return
  if (this->memoizer.have_saved_data(workset,this->evaluatedFields()))
    return;

  constexpr auto ALL = Kokkos::ALL();
  const int ws = workset.wsIndex;

  const auto param = workset.distParamLib->get(this->param_name);
  const auto p_data = Albany::getNonconstDeviceData(param->vector());

  const auto ws_elem_lids = workset.disc->getWsElementLIDs().dev();
  const auto elem_lids = Kokkos::subview(ws_elem_lids,ws,ALL);
  const auto p_elem_dof_lids = param->get_dof_mgr()->elem_dof_lids().dev();
  const int p_local_subdim = p_data.size();

  Kokkos::parallel_for(this->getName(),RangePolicy(0,workset.numCells),
                       KOKKOS_CLASS_LAMBDA(const int& cell) {
    const auto elem_LID = elem_lids(cell);
    for (int node=0; node<this->numNodes; ++node) {
      const LO lid = p_elem_dof_lids(elem_LID,node);
      if (lid>=0 && lid<p_local_subdim) { // Exploit the fact that owned lids come before ghosted lids
        p_data(lid) = this->val(cell,node);
      }
    }
  });
}

template<typename Traits>
ScatterScalarExtruded2DNodalParameter<PHAL::AlbanyTraits::Residual, Traits>::
ScatterScalarExtruded2DNodalParameter(const Teuchos::ParameterList& p,
                           const Teuchos::RCP<Albany::Layouts>& dl)
 : ScatterScalarNodalParameterBase<PHAL::AlbanyTraits::Residual, Traits>(p,dl)
{
  fieldLevel = p.get<int>("Field Level");

  // Create field tag
  nodal_field_tag = Teuchos::rcp(new PHX::Tag<ParamScalarT>(className, dl->dummy));

  this->addEvaluatedField(*nodal_field_tag);
}

template<typename Traits>
void ScatterScalarExtruded2DNodalParameter<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // Check for early return
  if (this->memoizer.have_saved_data(workset,this->evaluatedFields()))
    return;

  const auto& param = workset.distParamLib->get(this->param_name);
  const auto& p_dof_mgr = param->get_dof_mgr();
  const auto  p_data = Albany::getNonconstLocalData(param->vector());
  const auto& p_elem_dof_lids = param->get_dof_mgr()->elem_dof_lids().host();
  const int   p_local_subdim = p_data.size();

  const auto& layers_data    = *workset.disc->getMeshStruct()->global_cell_layers_data;
  const int   top = layers_data.top_side_pos;
  const int   bot = layers_data.bot_side_pos;
  const auto elem_lids       = workset.disc->getElementLIDs_host(workset.wsIndex);

  const int fieldLayer = fieldLevel==layers_data.numLayers ? fieldLevel-1 : fieldLevel;
  const int field_pos  = fieldLayer==fieldLevel ? bot : top;

  const auto& offsets = p_dof_mgr->getGIDFieldOffsetsSide(0,field_pos);
  for (size_t cell=0; cell<workset.numCells; ++cell) {
    const auto elem_LID = elem_lids(cell);
    const auto layer = layers_data.getLayerId(elem_LID);
    if (layer==fieldLayer) {
      for (auto o : offsets) {
        const LO lid = p_elem_dof_lids(elem_LID,o);
        if (lid>=0 && lid<p_local_subdim) { // Exploit the fact that owned lids come before ghosted lids
          p_data[lid] = this->val(cell,o);
        }
      }
    }
  }
}

} // namespace PHAL
