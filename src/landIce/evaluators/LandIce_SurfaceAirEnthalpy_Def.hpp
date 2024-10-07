/*
 * LandIce_SurfaceAirEnthalpy_Def.hpp
 *
 *  Created on: March 2, 2020
 *      Author: mperego
 */

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_Print.hpp"
#include "PHAL_Utilities.hpp"
#include "Albany_KokkosUtils.hpp"

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
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& /* fm */)
{
  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields());
  if (d.memoizer_active()) memoizer.enable_memoizer();
}

template<typename EvalT, typename Traits, typename SurfTempST>
void SurfaceAirEnthalpy<EvalT,Traits,SurfTempST>::
evaluateFields(typename Traits::EvalData workset)
{
  if (memoizer.have_saved_data(workset,this->evaluatedFields())) return;

  const double powm6 = 1e-6; // [k^2], k=1000

  Kokkos::parallel_for(this->getName(),RangePolicy(0,workset.numCells),
                       KOKKOS_CLASS_LAMBDA(const int& cell) {
    for (std::size_t node = 0; node < numNodes; ++node)
      surfaceEnthalpy(cell,node) = rho_i * c_i * ( KU::min(surfaceTemp(cell,node),Tm) - T0 ) * powm6;
  });
}

} // namespace LandIce
