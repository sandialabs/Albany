/*
 * LandIce_HydrostaticPressure_Def.hpp
 *
 *  Created on: Jun 6, 2016
 *      Author: abarone
 */

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_Print.hpp"

#include "LandIce_HydrostaticPressure.hpp"

namespace LandIce
{

const double pow3 = 1.0e3;

template<typename EvalT, typename Traits, typename SurfHeightST>
HydrostaticPressure<EvalT,Traits,SurfHeightST>::
HydrostaticPressure(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl):
  s        (p.get<std::string> ("Surface Height Variable Name"), dl->node_scalar),
  z        (p.get<std::string> ("Coordinate Vector Variable Name"),dl->vertices_vector),
  pressure (p.get<std::string> ("Hydrostatic Pressure Variable Name"), dl->node_scalar)
{
  std::vector<PHX::Device::size_type> dims;
  dl->node_scalar->dimensions(dims);

  numNodes = dims[1];

  this->addDependentField(s);
  this->addDependentField(z);

  this->addEvaluatedField(pressure);
  this->setName("Hydrostatic Pressure");

  // Setting parameters
  Teuchos::ParameterList& physics  = *p.get<Teuchos::ParameterList*>("LandIce Physical Parameters");

  rho_i = physics.get<double>("Ice Density");
  g     = physics.get<double>("Gravity Acceleration");
}

template<typename EvalT, typename Traits, typename SurfHeightST>
KOKKOS_INLINE_FUNCTION
void HydrostaticPressure<EvalT, Traits, SurfHeightST>::
operator() (const int& node, const int& cell) const{

    pressure(cell,node) = pow3 * rho_i * g * ( s(cell,node) - z(cell,node,2) ); 

}

template<typename EvalT, typename Traits, typename SurfHeightST>
void HydrostaticPressure<EvalT,Traits,SurfHeightST>::
postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(s,fm);
  this->utils.setFieldData(z,fm);

  this->utils.setFieldData(pressure,fm);

  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields());
  if (d.memoizer_active()) memoizer.enable_memoizer();
}

template<typename EvalT, typename Traits, typename SurfHeightST>
void HydrostaticPressure<EvalT,Traits,SurfHeightST>::
evaluateFields(typename Traits::EvalData workset)
{
  if (memoizer.have_saved_data(workset,this->evaluatedFields())) return;

  // double pow3 = 1.0e3;
  // for (std::size_t cell = 0; cell < workset.numCells; ++cell)
  //   for (std::size_t node = 0; node < numNodes; ++node)
  //     pressure(cell,node) = pow3 * rho_i * g * ( s(cell,node) - z(cell,node,2) );
  Kokkos::parallel_for(HYDROSTATIC_PRESSURE_Policy({0,0}, {numNodes,workset.numCells}), *this);
}

} // namespace LandIce
