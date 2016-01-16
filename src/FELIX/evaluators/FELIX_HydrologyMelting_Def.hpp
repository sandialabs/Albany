//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

namespace FELIX {

//**********************************************************************
template<typename EvalT, typename Traits>
HydrologyMelting<EvalT, Traits>::HydrologyMelting (const Teuchos::ParameterList& p,
                                                   const Teuchos::RCP<Albany::Layouts>& dl)
{
  if (p.isParameter("Stokes Coupling"))
  {
    stokes_coupling = p.get<bool>("Stokes Coupling");
  }
  else
  {
    stokes_coupling = false;
  }

  if (stokes_coupling)
  {
    sideSetName = p.get<std::string>("Side Set Name");

    u_b  = PHX::MDField<ParamScalarT>(p.get<std::string> ("Sliding Velocity Side QP Variable Name"), dl->side_qp_scalar);
    beta = PHX::MDField<ParamScalarT>(p.get<std::string> ("Basal Friction Coefficient Side QP Variable Name"), dl->side_qp_scalar);
    G    = PHX::MDField<ParamScalarT>(p.get<std::string> ("Geothermal Heat Source Side QP Variable Name"), dl->side_qp_scalar);
    m    = PHX::MDField<ParamScalarT>(p.get<std::string> ("Melting Rate Side QP Variable Name"),dl->side_qp_scalar);

    numQPs = dl->side_qp_scalar->dimension(2);
  }
  else
  {
    u_b  = PHX::MDField<ParamScalarT>(p.get<std::string> ("Sliding Velocity QP Variable Name"), dl->qp_scalar);
    beta = PHX::MDField<ParamScalarT>(p.get<std::string> ("Basal Friction Coefficient QP Variable Name"), dl->qp_scalar);
    G    = PHX::MDField<ParamScalarT>(p.get<std::string> ("Geothermal Heat Source QP Variable Name"), dl->qp_scalar);
    m    = PHX::MDField<ParamScalarT>(p.get<std::string> ("Melting Rate QP Variable Name"),dl->qp_scalar);

    numQPs = dl->qp_scalar->dimension(1);
  }

  this->addDependentField(u_b);
  this->addDependentField(beta);
  this->addDependentField(G);

  this->addEvaluatedField(m);

  // Setting parameters
  Teuchos::ParameterList& physical_params  = *p.get<Teuchos::ParameterList*>("FELIX Physical Parameters");
  L = physical_params.get<double>("Ice Latent Heat");

  this->setName("HydrologyMelting"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void HydrologyMelting<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(u_b,fm);
  this->utils.setFieldData(beta,fm);
  this->utils.setFieldData(G,fm);

  this->utils.setFieldData(m,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void HydrologyMelting<EvalT, Traits>::evaluateFields (typename Traits::EvalData workset)
{
  // m = \frac{ G - \beta |u_b|^2 + \nabla (phiH-N)\cdot q }{L} %% The nonlinear term \nabla (phiH-N)\cdot q can be ignored

  if (!stokes_coupling)
  {
    for (int cell=0; cell < workset.numCells; ++cell)
    {
      for (int qp=0; qp < numQPs; ++qp)
      {
        m(cell,qp) = (G(cell,qp) - beta(cell,qp) * std::pow(u_b(cell,qp),2) ) / L; //- nonlin_coeff * prod) / L;
      }
    }
  }
  else
  {
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
        m(cell,side,qp) = ( G(cell,side,qp) - beta(cell,side,qp) * std::pow(u_b(cell,side,qp),2) ) / L;
      }
    }
  }
}

} // Namespace FELIX
