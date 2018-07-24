//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

namespace LandIce {

//**************************************************************
// PARTIAL SPECIALIZATION: Hydrology ***************************
//**************************************************************
template<typename EvalT, typename Traits, bool IsStokes>
HydrologyMeltingRate<EvalT, Traits, IsStokes>::
HydrologyMeltingRate (const Teuchos::ParameterList& p,
                      const Teuchos::RCP<Albany::Layouts>& dl)
{
  if (IsStokes)
  {
    TEUCHOS_TEST_FOR_EXCEPTION (!dl->isSideLayouts, Teuchos::Exceptions::InvalidParameter,
                                "Error! The layout structure does not appear to be that of a side set.\n");
    numQPs   = dl->qp_scalar->dimension(2);
    numNodes = dl->node_scalar->dimension(2);
    sideSetName = p.get<std::string>("Side Set Name");
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION (dl->isSideLayouts, Teuchos::Exceptions::InvalidParameter,
                                "Error! The layout structure appears to be that of a side set.\n");
    numQPs   = dl->qp_scalar->dimension(1);
    numNodes = dl->node_scalar->dimension(1);
  }

  // Setting parameters
  Teuchos::ParameterList& physical_params  = *p.get<Teuchos::ParameterList*>("LandIce Physical Parameters");
  L = physical_params.get<double>("Ice Latent Heat");

  /*
   * Scalings, needed to account for different units: ice velocity
   * is in m/yr, the mesh is in km, and hydrology time unit is s.
   *
   * The melting rate has 2 terms (forget about signs), with the following
   * units (including the km^2 from dx):
   *
   * 1) G                 [W m^-2 km^2] = [Pa m s^-1 km^2]
   * 2) beta*|u|^2        [kPa m yr^-1 km^2]
   *
   * To sum apples to apples, we need to convert one to the units of the other.
   * We choose to keep [kPa m yr^-1 km^2], which are the units used in the residuals,
   * and we rescale G by
   *
   *  scaling_G = yr_to_s/1000
   *
   * where yr_to_s=365.25*24*3600 (the number of seconds in a year).
   * Furthermore, we scale J to be in kJ/kg = kPa*m^3/kg, so that kPa cancel out.
   *
   * With this choice, considering G~10^-1 W/m^2, |u|~1000 m/yr, beta~10 kPa*yr/m, L~10^5 J/kg
   * we get m ~ 100 kg/(m^2 yr). When multiplied by 1/rho_i in the residual, this gives a term of
   * order 10^-1
   *
   */
  scaling_G = 365.25*24*3600/1000;
  L *= 1e-3;

  nodal = p.isParameter("Nodal") ? p.get<bool>("Nodal") : false;
  Teuchos::RCP<PHX::DataLayout> layout;
  if (nodal) {
    layout = dl->node_scalar;
  } else {
    layout = dl->qp_scalar;
  }
  u_b  = PHX::MDField<const IceScalarT>(p.get<std::string> ("Sliding Velocity Variable Name"), layout);
  G    = PHX::MDField<const ParamScalarT>(p.get<std::string> ("Geothermal Heat Source Variable Name"), layout);
  beta = PHX::MDField<const ScalarT>(p.get<std::string> ("Basal Friction Coefficient Variable Name"), layout);
  m    = PHX::MDField<ScalarT>(p.get<std::string> ("Melting Rate Variable Name"),layout);

  this->addDependentField(beta);
  this->addDependentField(u_b);
  this->addDependentField(G);
  this->addEvaluatedField(m);

  this->setName("HydrologyMeltingRate"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits, bool IsStokes>
void HydrologyMeltingRate<EvalT, Traits, IsStokes>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(u_b,fm);
  this->utils.setFieldData(G,fm);
  this->utils.setFieldData(beta,fm);
  this->utils.setFieldData(m,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits, bool IsStokes>
void HydrologyMeltingRate<EvalT, Traits, IsStokes>::evaluateFields (typename Traits::EvalData workset)
{
  // m = \frac{ G - \beta |u_b|^2 + \nabla (phiH-N)\cdot q }{L} %% The nonlinear term \nabla (phiH-N)\cdot q can be ignored

  if (IsStokes)
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
        m(cell,side,qp) = ( scaling_G*G(cell,side,qp) + beta(cell,side,qp) * std::pow(u_b(cell,side,qp),2) ) / L;
      }
    }
  }
  else
  {
    int dim = nodal ? numNodes : numQPs;
    for (int cell=0; cell < workset.numCells; ++cell)
    {
      for (int i=0; i<dim; ++i)
      {
        m(cell,i) = ( scaling_G*G(cell,i) + beta(cell,i) * std::pow(u_b(cell,i),2) ) / L;
      }
    }
  }
}

} // Namespace LandIce
