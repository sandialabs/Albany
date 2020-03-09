/*
 * LandIce_SurfaceAirEnthalpy_Def.hpp
 *
 *  Created on: March 2, 2020
 *      Author: mperego
 */

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_Print.hpp"
#include "PHAL_Utilities.hpp"

#include "LandIce_SurfaceAirEnthalpy.hpp"

namespace LandIce
{

template<typename EvalT, typename Traits, typename SurfTempST>
SurfaceAirEnthalpy<EvalT,Traits,SurfTempST>::
SurfaceAirEnthalpy(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl):
  surfaceTemp    (p.get<std::string> ("Surface Air Temperature Name"), dl->node_scalar),
  surfaceEnthalpy(p.get<std::string> ("Surface Air Enthalpy Name"), dl->node_scalar)
{
  fieldName =  p.get<std::string>("Surface Air Temperature Name");

  std::vector<PHX::Device::size_type> dims;
  dl->node_qp_vector->dimensions(dims);

  numNodes = dims[1];

  this->addDependentField(surfaceTemp);
  this->addEvaluatedField(surfaceEnthalpy);

  this->setName("Surface air Enthalpy");

  // Setting parameters
  Teuchos::ParameterList& physics_list = *p.get<Teuchos::ParameterList*>("LandIce Physical Parameters");
  rho_i = physics_list.get<double>("Ice Density"); //916
  c_i   = physics_list.get<double>("Heat capacity of ice");  //2009
  T0    = physics_list.get<double>("Reference Temperature"); //265
  Tm = physics_list.get<double>("Atmospheric Pressure Melting Temperature");
}

template<typename EvalT, typename Traits, typename SurfTempST>
void SurfaceAirEnthalpy<EvalT,Traits,SurfTempST>::
postRegistrationSetup(typename Traits::SetupData workset, PHX::FieldManager<Traits>& fm)
{}

template<typename EvalT, typename Traits, typename SurfTempST>
void SurfaceAirEnthalpy<EvalT,Traits,SurfTempST>::
evaluateFields(typename Traits::EvalData workset)
{
  const double powm6 = 1e-6; // [k^2], k=1000

  for (std::size_t cell = 0; cell < workset.numCells; ++cell)
    for (std::size_t node = 0; node < numNodes; ++node)
      surfaceEnthalpy(cell,node) = rho_i * c_i * ( std::min(surfaceTemp(cell,node),Tm) - T0 ) * powm6;
}

} // namespace LandIce
