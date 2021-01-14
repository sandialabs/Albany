#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_Print.hpp"
#include "utility/string.hpp"

#include <math.h>

namespace LandIce
{

template<typename EvalT, typename Traits, bool OnSide>
HydrologySurfaceWaterInput<EvalT,Traits,OnSide>::
HydrologySurfaceWaterInput (const Teuchos::ParameterList& p,
                            const Teuchos::RCP<Albany::Layouts>& dl)
{
  // Safety check
  TEUCHOS_TEST_FOR_EXCEPTION (OnSide!=dl->isSideLayouts, std::logic_error,
                              "Error! Instantiation with OnSide=" << OnSide << ", requires input layouts with isSideLayouts=" << OnSide << ".\n");

  Teuchos::ParameterList& plist = *p.get<Teuchos::ParameterList*>("Surface Water Input Params");
  std::string type = plist.get<std::string>("Type","Given Field");
  type = util::upper_case(type);
  if (type=="GIVEN VALUE") {
    // Set omega=val
    omega = decltype(omega)(p.get<std::string> ("Surface Water Input Variable Name"), dl->node_scalar);
    this->addEvaluatedField(omega);

    omega_val = plist.get<double>("Given Value");

    input_type = InputType::GIVEN_VALUE;
  } else if (type=="GIVEN FIELD") {
    // Nothing to be done. The user should already be loading a surface water input nodal field.
    // If not, the Phalanx DAG will be broken and Phalanx will report it.
    input_type = InputType::GIVEN_FIELD;
  } else if (type=="APPROXIMATE FROM SMB") {
    // Set omega=min(-smb,0);
    smb = decltype(smb)(p.get<std::string> ("Surface Mass Balance Variable Name"), dl->node_scalar);
    this->addDependentField(smb);

    omega = decltype(omega)(p.get<std::string> ("Surface Water Input Variable Name"), dl->node_scalar);
    this->addEvaluatedField(omega);

    input_type = InputType::SMB_APPROX;
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
                                "Error! Invalid choice '" << type << "' for "
                                "surface water input type.\n");
  }

  // Get Dimensions
  if (OnSide) {
    sideSetName = p.get<std::string>("Side Set Name");
    numNodes = dl->node_scalar->extent(2);
  } else {
    numNodes = dl->node_scalar->extent(1);
  }

  this->setName("Surface Water Input From SMB " + PHX::print<EvalT>());
}

template<typename EvalT, typename Traits, bool OnSide>
void HydrologySurfaceWaterInput<EvalT,Traits,OnSide>::
postRegistrationSetup (typename Traits::SetupData d,
                       PHX::FieldManager<Traits>& fm)
{
  if (input_type==InputType::SMB_APPROX) {
    this->utils.setFieldData(smb,fm);
    this->utils.setFieldData(omega,fm);
  }
}

template<typename EvalT, typename Traits, bool OnSide>
void HydrologySurfaceWaterInput<EvalT,Traits,OnSide>::
evaluateFields (typename Traits::EvalData workset)
{
  if (OnSide) {
    evaluateFieldsSide(workset);
  } else {
    evaluateFieldsCell(workset);
  }
}

template<typename EvalT, typename Traits, bool OnSide>
void HydrologySurfaceWaterInput<EvalT,Traits,OnSide>::
evaluateFieldsCell (typename Traits::EvalData workset)
{
  ParamScalarT zero (0.0);
  if (input_type==InputType::SMB_APPROX) {
    for (unsigned int cell=0; cell<workset.numCells; ++cell) {
      for (unsigned int node=0; node<numNodes; ++node) {
        omega(cell,node) = -std::min(smb(cell,node),zero);
      }
    }
  }
}

template<typename EvalT, typename Traits, bool OnSide>
void HydrologySurfaceWaterInput<EvalT,Traits,OnSide>::
evaluateFieldsSide (typename Traits::EvalData workset)
{
  if (workset.sideSets->find(sideSetName)==workset.sideSets->end()) {
    return;
  }

  if (input_type==InputType::GIVEN_FIELD) {
    return;
  }

  const std::vector<Albany::SideStruct>& sideSet = workset.sideSets->at(sideSetName);
  for (auto const& it_side : sideSet)
  {
    // Get the local data of side and cell
    const int cell = it_side.elem_LID;
    const int side = it_side.side_local_id;

    ParamScalarT zero (0.0);
    for (unsigned int node=0; node<numNodes; ++node) {
      omega(cell,side,node) = -std::min(smb(cell,side,node),zero);
    }
  }
}

} // Namespace LandIce
