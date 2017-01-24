#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

namespace FELIX
{

template<typename EvalT, typename Traits>
HydrologyWaterSource<EvalT,Traits>::
HydrologyWaterSource (const Teuchos::ParameterList& p,
                      const Teuchos::RCP<Albany::Layouts>& dl):
  smb   (p.get<std::string> ("Surface Mass Balance Variable Name"), dl->node_scalar),
  omega (p.get<std::string> ("Surface Water Input Variable Name"), dl->node_scalar)
{
  // Get Dimensions
  numNodes = dl->node_qp_vector->dimension(1);

  this->addDependentField(smb.fieldTag());
  this->addEvaluatedField(omega);

  this->setName("Surface Water Input From SMB " + PHX::typeAsString<EvalT>());
}

template<typename EvalT, typename Traits>
void HydrologyWaterSource<EvalT,Traits>::
postRegistrationSetup (typename Traits::SetupData d,
                       PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(smb,fm);
  this->utils.setFieldData(omega,fm);
}

template<typename EvalT, typename Traits>
void HydrologyWaterSource<EvalT,Traits>::
evaluateFields (typename Traits::EvalData workset)
{
  for (int cell=0; cell<workset.numCells; ++cell)
    for (int node=0; node<numNodes; ++node)
      omega(cell,node) = -std::min(smb(cell,node),0.0);
}

} // Namespace FELIX
