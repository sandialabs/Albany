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
                           const Teuchos::RCP<Albany::Layouts>& dl) :
  ScatterScalarNodalParameterBase<PHAL::AlbanyTraits::Residual, Traits>(p,dl)
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

  const auto param = workset.distParamLib->get(this->param_name);
  const auto p_data = Albany::getNonconstLocalData(param->vector());

  const auto elem_lids    = workset.disc->getElementLIDs_host(workset.wsIndex);
  const auto node_dof_mgr = workset.disc->getNodeNewDOFManager();
  const auto p_dof_mgr    = workset.disc->getNewDOFManager(this->param_name);
  const auto p_indexer    = p_dof_mgr->indexer();

  std::vector<GO> node_gids;
  for (size_t cell=0; cell<workset.numCells; ++cell) {
    const auto elem_LID = elem_lids(cell);
    node_dof_mgr->getElementGIDs(elem_LID,node_gids);
    for (int node=0; node<this->numNodes; ++node) {
      const LO lid = p_indexer->getLocalElement(node_gids[node]);
      if(lid >= 0) {
       p_data[lid] = this->val(cell,node);
      }
    }
  }
}

template<typename Traits>
ScatterScalarExtruded2DNodalParameter<PHAL::AlbanyTraits::Residual, Traits>::
ScatterScalarExtruded2DNodalParameter(const Teuchos::ParameterList& p,
                           const Teuchos::RCP<Albany::Layouts>& dl) :
  ScatterScalarNodalParameterBase<PHAL::AlbanyTraits::Residual, Traits>(p,dl)
{
  fieldLevel = p.get<int>("Field Level");

  // Create field tag
  nodal_field_tag =
    Teuchos::rcp(new PHX::Tag<ParamScalarT>(className, dl->dummy));

  this->addEvaluatedField(*nodal_field_tag);
}

template<typename Traits>
void ScatterScalarExtruded2DNodalParameter<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // Check for early return
  if (this->memoizer.have_saved_data(workset,this->evaluatedFields()))
    return;

  const auto param = workset.distParamLib->get(this->param_name);
  const auto p_data = Albany::getNonconstLocalData(param->vector());

  const auto& layers_data   = *workset.disc->getLayeredMeshNumbering();
  const auto  elem_lids    = workset.disc->getElementLIDs_host(workset.wsIndex);
  const auto  node_dof_mgr = workset.disc->getNodeNewDOFManager();
  const auto  p_dof_mgr    = workset.disc->getNewDOFManager(this->param_name);
  const auto  p_indexer    = p_dof_mgr->indexer();

  std::vector<GO> node_gids;
  for (size_t cell=0; cell<workset.numCells; ++cell) {
    const auto elem_LID = elem_lids(cell);
    node_dof_mgr->getElementGIDs(elem_LID,node_gids);
    for (int node=0; node<this->numNodes; ++node) {
      const GO ilayer = layers_data.getLayerId(node_gids[node]);
      if (ilayer==fieldLevel) {
        const LO lid = p_indexer->getLocalElement(node_gids[node]);
        if (lid>=0) {
          p_data[lid] = this->val(cell,node);
        }
      }
    }
  }
}

} // namespace PHAL
