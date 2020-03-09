/*
 * LandIce_PressureMeltingEnthalpy_Def.hpp
 *
 *  Created on: Jun 6, 2016
 *      Author: abarone
 */

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_Print.hpp"

#include "LandIce_PressureMeltingEnthalpy.hpp"

namespace LandIce
{

template<typename EvalT, typename Traits, typename PressST>
PressureMeltingEnthalpy<EvalT,Traits,PressST>::
PressureMeltingEnthalpy(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl):
  pressure       (p.get<std::string> ("Hydrostatic Pressure Variable Name"), dl->node_scalar),
  meltingTemp    (p.get<std::string> ("Melting Temperature Variable Name"), dl->node_scalar),
  enthalpyHs     (p.get<std::string> ("Enthalpy Hs Variable Name"), dl->node_scalar)
{
  std::vector<PHX::Device::size_type> dims;
  dl->node_qp_vector->dimensions(dims);

  numNodes = dims[1];

  this->addDependentField(pressure);

  this->addEvaluatedField(meltingTemp);
  this->addEvaluatedField(enthalpyHs);
  this->setName("Pressure-melting Enthalpy");

  // Setting parameters
  Teuchos::ParameterList& physics_list = *p.get<Teuchos::ParameterList*>("LandIce Physical Parameters");
  rho_i = physics_list.get<double>("Ice Density"); //916
  c_i   = physics_list.get<double>("Heat capacity of ice");  //2009
  T0    = physics_list.get<double>("Reference Temperature"); //265
  beta =  physics_list.get<double>("Clausius-Clapeyron Coefficient");
  Tm = physics_list.get<double>("Atmospheric Pressure Melting Temperature");
}

template<typename EvalT, typename Traits, typename PressST>
void PressureMeltingEnthalpy<EvalT,Traits,PressST>::
postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
{}

template<typename EvalT, typename Traits, typename PressST>
void PressureMeltingEnthalpy<EvalT,Traits,PressST>::
evaluateFields(typename Traits::EvalData d)
{
  const double powm6 = 1e-6; // [k^2], k=1000

  for (std::size_t cell = 0; cell < d.numCells; ++cell)
    for (std::size_t node = 0; node < numNodes; ++node) {
      meltingTemp(cell,node) = - beta * pressure(cell,node) + Tm;
      enthalpyHs(cell,node) = rho_i * c_i * ( meltingTemp(cell,node) - T0 ) * powm6;
    }
}

} // namespace LandIce
