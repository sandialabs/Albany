#include "LandIce_HydrologySurfaceWaterInput.hpp"
#include "Albany_StringUtils.hpp" // for 'upper_case'

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_Print.hpp"

namespace LandIce
{

template<typename EvalT, typename Traits>
HydrologySurfaceWaterInput<EvalT,Traits>::
HydrologySurfaceWaterInput (const Teuchos::ParameterList& p,
                            const Teuchos::RCP<Albany::Layouts>& dl)
{
  // Check if it is a sideset evaluation
  eval_on_side = false;
  if (p.isParameter("Side Set Name")) {
    sideSetName = p.get<std::string>("Side Set Name");
    eval_on_side = true;
  }
  TEUCHOS_TEST_FOR_EXCEPTION (eval_on_side!=dl->isSideLayouts, std::logic_error,
      "Error! Input Layouts structure not compatible with requested field layout.\n");

  auto& plist = *p.get<Teuchos::ParameterList*>("Surface Water Input Params");
  std::string type = plist.get<std::string>("Type","Given Field");
  type = util::upper_case(type);
  if (type=="APPROXIMATE FROM SMB") {
    // Set omega=min(-smb,0);
    smb = decltype(smb)(p.get<std::string> ("Surface Mass Balance Variable Name"), dl->node_scalar);
    this->addDependentField(smb);

    omega = decltype(omega)(p.get<std::string> ("Surface Water Input Variable Name"), dl->node_scalar);
    this->addEvaluatedField(omega);

    input_type = InputType::SMB_APPROX;
  } else if (type=="GIVEN FIELD") {
    // Do nothing, we will simply use the input field
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
                                "Error! Invalid choice '" << type << "' for "
                                "surface water input type.\n");
  }

  // Get Dimensions
  if (eval_on_side) {
    sideSetName = p.get<std::string>("Side Set Name");
    numNodes = dl->node_scalar->extent(1);
  } else {
    numNodes = dl->node_scalar->extent(1);
  }

  this->setName("Surface Water Input From SMB " + PHX::print<EvalT>());
}

template<typename EvalT, typename Traits>
void HydrologySurfaceWaterInput<EvalT,Traits>::
evaluateFields (typename Traits::EvalData workset)
{
  if (eval_on_side) {
    evaluateFieldsSide(workset);
  } else {
    evaluateFieldsCell(workset);
  }
}

template<typename EvalT, typename Traits>
void HydrologySurfaceWaterInput<EvalT,Traits>::
evaluateFieldsCell (typename Traits::EvalData workset)
{
  if (input_type==InputType::SMB_APPROX) {
    ParamScalarT zero (0.0);
    for (unsigned int cell=0; cell<workset.numCells; ++cell) {
      for (unsigned int node=0; node<numNodes; ++node) {
        omega(cell,node) = -std::min(smb(cell,node),zero);
      }
    }
  }
}

template<typename EvalT, typename Traits>
void HydrologySurfaceWaterInput<EvalT,Traits>::
evaluateFieldsSide (typename Traits::EvalData workset)
{
  if (workset.sideSets->find(sideSetName)==workset.sideSets->end()) return;

  sideSet = workset.sideSetViews->at(sideSetName);
  for (int sideSet_idx = 0; sideSet_idx < sideSet.size; ++sideSet_idx)
  {
    ParamScalarT zero (0.0);
    for (unsigned int node=0; node<numNodes; ++node) {
      omega(sideSet_idx,node) = -std::min(smb(sideSet_idx,node),zero);
    }
  }
}

} // Namespace LandIce
