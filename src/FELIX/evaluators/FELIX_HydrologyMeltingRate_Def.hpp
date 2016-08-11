//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

namespace FELIX {

//**************************************************************
// PARTIAL SPECIALIZATION: Hydrology ***************************
//**************************************************************
template<typename EvalT, typename Traits>
HydrologyMeltingRate<EvalT, Traits, false>::
HydrologyMeltingRate (const Teuchos::ParameterList& p,
                      const Teuchos::RCP<Albany::Layouts>& dl) :
  u_b  (p.get<std::string> ("Sliding Velocity QP Variable Name"), dl->qp_scalar),
  G    (p.get<std::string> ("Geothermal Heat Source QP Variable Name"), dl->qp_scalar),
  beta (p.get<std::string> ("Basal Friction Coefficient QP Variable Name"), dl->qp_scalar),
  m    (p.get<std::string> ("Melting Rate QP Variable Name"),dl->qp_scalar)
{
  numQPs = dl->qp_scalar->dimension(1);

  this->addDependentField(beta);
  this->addDependentField(u_b);
  this->addDependentField(G);

  this->addEvaluatedField(m);

  // Setting parameters
  Teuchos::ParameterList& physical_params  = *p.get<Teuchos::ParameterList*>("FELIX Physical Parameters");
  L = physical_params.get<double>("Ice Latent Heat");

  this->setName("HydrologyMeltingRate"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void HydrologyMeltingRate<EvalT, Traits, false>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(u_b,fm);
  this->utils.setFieldData(G,fm);
  this->utils.setFieldData(beta,fm);
  this->utils.setFieldData(m,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void HydrologyMeltingRate<EvalT, Traits, false>::evaluateFields (typename Traits::EvalData workset)
{
  // m = \frac{ G - \beta |u_b|^2 + \nabla (phiH-N)\cdot q }{L} %% The nonlinear term \nabla (phiH-N)\cdot q can be ignored

  for (int cell=0; cell < workset.numCells; ++cell)
  {
    for (int qp=0; qp < numQPs; ++qp)
    {
      m(cell,qp) = ( G(cell,qp) - beta(cell,qp) * std::pow(u_b(cell,qp),2) * 1000/31536000 ) / L;
      // The factor 1000/31536000 is to adjust from kPa and yr units (ice) to SI units (hydrology)
    }
  }
}

//**********************************************************************
// PARTIAL SPECIALIZATION: StokesFOHydrology ***************************
//**********************************************************************
template<typename EvalT, typename Traits>
HydrologyMeltingRate<EvalT, Traits, true>::
HydrologyMeltingRate (const Teuchos::ParameterList& p,
                      const Teuchos::RCP<Albany::Layouts>& dl) :
  u_b  (p.get<std::string> ("Sliding Velocity QP Variable Name"), dl->qp_scalar),
  G    (p.get<std::string> ("Geothermal Heat Source QP Variable Name"), dl->qp_scalar),
  beta (p.get<std::string> ("Basal Friction Coefficient QP Variable Name"), dl->qp_scalar),
  m    (p.get<std::string> ("Melting Rate QP Variable Name"),dl->qp_scalar)
{
  TEUCHOS_TEST_FOR_EXCEPTION (!dl->isSideLayouts, Teuchos::Exceptions::InvalidParameter,
                              "Error! The layout structure does not appear to be that of a side set.\n");

  numQPs = dl->qp_scalar->dimension(2);

  this->addDependentField(beta);
  this->addDependentField(u_b);
  this->addDependentField(G);

  this->addEvaluatedField(m);

  // Setting parameters
  Teuchos::ParameterList& physical_params  = *p.get<Teuchos::ParameterList*>("FELIX Physical Parameters");
  L = physical_params.get<double>("Ice Latent Heat");

  this->setName("HydrologyMeltingRate"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void HydrologyMeltingRate<EvalT, Traits, true>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(u_b,fm);
  this->utils.setFieldData(G,fm);
  this->utils.setFieldData(beta,fm);
  this->utils.setFieldData(m,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void HydrologyMeltingRate<EvalT, Traits, true>::evaluateFields (typename Traits::EvalData workset)
{
  // m = \frac{ G - \beta |u_b|^2 + \nabla (phiH-N)\cdot q }{L} %% The nonlinear term \nabla (phiH-N)\cdot q can be ignored

  if (workset.sideSets->find(sideSetName)==workset.sideSets->end())
    return;

  const std::vector<Albany::SideStruct>& sideSet = workset.sideSets->at(sideSetName);
  for (auto const& it_side : sideSet)
  {
    // Get the local data of side and cell
    const int cell = it_side.elem_LID;
    const int side = it_side.side_local_id;

    for (int qp=0; qp < numQPs; ++qp)
    {
      m(cell,side,qp) = ( G(cell,side,qp) - beta(cell,side,qp) * std::pow(u_b(cell,side,qp),2) * 1000/31536000 ) / L;
      // The factor 31536000 is to adjust from yr units (ice) to s units (hydrology)
    }
  }
}

} // Namespace FELIX
