//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_Print.hpp"

#include "LandIce_HydrologyMeltingRate.hpp"

namespace LandIce {

//**************************************************************
// PARTIAL SPECIALIZATION: Hydrology ***************************
//**************************************************************
template<typename EvalT, typename Traits, bool IsStokes>
HydrologyMeltingRate<EvalT, Traits, IsStokes>::
HydrologyMeltingRate (const Teuchos::ParameterList& p,
                      const Teuchos::RCP<Albany::Layouts>& dl)
{
  if (IsStokes) {
    TEUCHOS_TEST_FOR_EXCEPTION (!dl->isSideLayouts, Teuchos::Exceptions::InvalidParameter,
                                "Error! The layout structure does not appear to be that of a side set.\n");
    numQPs   = dl->qp_scalar->extent(1);
    numNodes = dl->node_scalar->extent(1);
    sideSetName = p.get<std::string>("Side Set Name");
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION (dl->isSideLayouts, Teuchos::Exceptions::InvalidParameter,
                                "Error! The layout structure appears to be that of a side set.\n");
    numQPs   = dl->qp_scalar->extent(1);
    numNodes = dl->node_scalar->extent(1);
  }

  // Setting parameters
  Teuchos::ParameterList& physical_params  = *p.get<Teuchos::ParameterList*>("LandIce Physical Parameters");
  latent_heat = physical_params.get<double>("Ice Latent Heat Of Fusion");

  /*
   * Scalings, needed to account for different units: ice velocity
   * is in m/yr, the mesh is in km, and hydrology time unit is s.
   *
   * The melting rate has 2 terms (forget about signs), with the following
   * units:
   *
   * 1) G                 [W m^-2] = [Pa m s^-1]
   * 2) beta*|u|^2        [kPa m yr^-1]
   *
   * To sum apples to apples, we need to convert one to the units of the other.
   * We choose to keep [kPa m yr^-1], which are the units used in the residuals,
   * and we rescale G by
   *
   *  scaling_G = yr_to_s/1000
   *
   * where yr_to_s=365.25*24*3600 (the number of seconds in a year).
   * Note: we will need to scale the result by 1000, since the stuff inside the parentheses
   * is in kPa m/yr while L is in Pa m^3/kg.
   *
   * With this choice, considering G~10^-1 W/m^2, |u|~1000 m/yr, beta~10 kPa*yr/m, L~10^5 J/kg
   * we get m ~ 100 kg/(m^2 yr). When multiplied by 1/rho_[i|w] in the residual, this gives a term of
   * order 10^-1
   *
   */
  scaling_G = 365.25*24*3600/1000;

  nodal = p.isParameter("Nodal") ? p.get<bool>("Nodal") : false;
  Teuchos::RCP<PHX::DataLayout> layout;
  if (nodal) {
    layout = dl->node_scalar;
  } else {
    layout = dl->qp_scalar;
  }

  m = PHX::MDField<ScalarT>(p.get<std::string> ("Melting Rate Variable Name"),layout);
  this->addEvaluatedField(m);

  Teuchos::ParameterList& hy_pl = *p.get<Teuchos::ParameterList*>("LandIce Hydrology");
  Teuchos::ParameterList& melt_pl = hy_pl.sublist("Melting Rate");

  if (melt_pl.isParameter("Given Value")) {
    m_value = melt_pl.get<double>("Given Value");
    m_given = true;
  } else {
    if (melt_pl.get("Use Friction Melt",false)) {
      u_b  = decltype(u_b) (p.get<std::string> ("Sliding Velocity Variable Name"), layout);
      beta = decltype(beta)(p.get<std::string> ("Basal Friction Coefficient Variable Name"), layout);
      this->addDependentField(beta);
      this->addDependentField(u_b);

      friction = true;
    }
    if (melt_pl.get("Use Geothermal Melt",false)) {
      if (melt_pl.isParameter("Given Geothermal Flux")) {
        G_given = true;
        G_value = melt_pl.get<double>("Given Geothermal Flux");
      } else {
        G = decltype(G)(p.get<std::string> ("Geothermal Heat Source Variable Name"), layout);
        this->addDependentField(G);
        G_field = true;
      }
    }
  }

  this->setName("HydrologyMeltingRate" + (nodal ? std::string("Nodal") : std::string("QPs")) +PHX::print<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits, bool IsStokes>
void HydrologyMeltingRate<EvalT, Traits, IsStokes>::
postRegistrationSetup(typename Traits::SetupData /* d */,
                      PHX::FieldManager<Traits>& )
{
  m.deep_copy(m_value);
}

//**********************************************************************
template<typename EvalT, typename Traits, bool IsStokes>
void HydrologyMeltingRate<EvalT, Traits, IsStokes>::evaluateFields (typename Traits::EvalData workset)
{
  // m = ( G + \beta |u_b|^2 ) / L, in kg m^-2 yr^-1

  TEUCHOS_TEST_FOR_EXCEPTION (!m_given && !friction && !G_field && !G_given, std::runtime_error,
      "Error! You did not specify how to compute the Melt Rate.\n");

  if (m_given) return;

  // Scale L so that kPa cancel out: [L/1000] = kJ/kg = kPa m^3 / kg, and [G] = kPa m/yr
  L = latent_heat*1e-3;

  dim = nodal ? numNodes : numQPs;

  if (IsStokes) {
    if (workset.sideSets->find(sideSetName)==workset.sideSets->end()) return;
    sideSet = workset.sideSetViews->at(sideSetName);
    worksetSize = sideSet.size;
  } else {
    worksetSize = workset.numCells;
  }

  for (unsigned int cell=0; cell < worksetSize; ++cell) {
    evaluateMeltingRate(cell);
  }

}

template<typename EvalT, typename Traits, bool IsStokes>
void HydrologyMeltingRate<EvalT, Traits, IsStokes>::evaluateMeltingRate (unsigned int cell)
{
  for (unsigned int i=0; i<dim; ++i) {
    ScalarT val(0.0);
    if (G_field) {
      val += scaling_G*G(cell,i);
    } else if (G_given) {
      val += G_value;
    }
    if (friction) {
      val += beta(cell,i) * std::pow(u_b(cell,i),2);
    }
    m(cell,i) = val / L;
  }
}

} // Namespace LandIce
