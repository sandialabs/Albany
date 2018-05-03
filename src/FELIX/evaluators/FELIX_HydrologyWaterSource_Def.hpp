#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

namespace FELIX
{

template<typename EvalT, typename Traits, bool OnSide>
HydrologyWaterSource<EvalT,Traits,OnSide>::
HydrologyWaterSource (const Teuchos::ParameterList& p,
                      const Teuchos::RCP<Albany::Layouts>& dl)
 : smb   (p.get<std::string> ("Surface Mass Balance Variable Name"), dl->node_scalar)
 , omega (p.get<std::string> ("Surface Water Input Variable Name"), dl->node_scalar)
{
  // Safety check
  TEUCHOS_TEST_FOR_EXCEPTION (OnSide!=dl->isSideLayouts, std::logic_error,
                              "Error! Instantiation with OnSide=" << OnSide << ", requires input layouts with isSideLayouts=" << OnSide << ".\n");

  // Get Dimensions
  if (OnSide) {
    sideSetName = p.get<std::string>("Side Set Name");
    numNodes = dl->node_scalar->dimension(2);
  } else {
    numNodes = dl->node_scalar->dimension(1);
  }

  this->addDependentField(smb);
  this->addEvaluatedField(omega);

  this->setName("Surface Water Input From SMB " + PHX::typeAsString<EvalT>());
}

template<typename EvalT, typename Traits, bool OnSide>
void HydrologyWaterSource<EvalT,Traits,OnSide>::
postRegistrationSetup (typename Traits::SetupData d,
                       PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(smb,fm);
  this->utils.setFieldData(omega,fm);
}

template<typename EvalT, typename Traits, bool OnSide>
void HydrologyWaterSource<EvalT,Traits,OnSide>::
evaluateFields (typename Traits::EvalData workset)
{
  if (OnSide) {
    evaluateFieldsSide(workset);
  } else {
    evaluateFieldsCell(workset);
  }
}

template<typename EvalT, typename Traits, bool OnSide>
void HydrologyWaterSource<EvalT,Traits,OnSide>::
evaluateFieldsCell (typename Traits::EvalData workset)
{
  for (int cell=0; cell<workset.numCells; ++cell) {
    for (int node=0; node<numNodes; ++node) {
      omega(cell,node) = -std::min(smb(cell,node),0.0);
    }
  }
}

template<typename EvalT, typename Traits, bool OnSide>
void HydrologyWaterSource<EvalT,Traits,OnSide>::
evaluateFieldsSide (typename Traits::EvalData workset)
{
  if (workset.sideSets->find(sideSetName)==workset.sideSets->end()) {
    return;
  }

  const std::vector<Albany::SideStruct>& sideSet = workset.sideSets->at(sideSetName);
  for (auto const& it_side : sideSet)
  {
    // Get the local data of side and cell
    const int cell = it_side.elem_LID;
    const int side = it_side.side_local_id;

    for (int node=0; node<numNodes; ++node) {
      omega(cell,side,node) = -std::min(smb(cell,side,node),0.0);
    }
  }
}

} // Namespace FELIX
