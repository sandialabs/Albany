/*
 * FELIX_HydrostaticPressure_Def.hpp
 *
 *  Created on: Jun 6, 2016
 *      Author: abarone
 */

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

namespace FELIX
{

  template<typename EvalT, typename Traits, typename Type>
  HydrostaticPressure<EvalT,Traits,Type>::
  HydrostaticPressure(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl):
  s        	   (p.get<std::string> ("Surface Height Variable Name"), dl->node_scalar),
  z      		   (p.get<std::string> ("Coordinate Vector Variable Name"),dl->vertices_vector),
  pressure       (p.get<std::string> ("Hydrostatic Pressure Variable Name"), dl->node_scalar)
  {
    std::vector<PHX::Device::size_type> dims;
    dl->node_scalar->dimensions(dims);

    numNodes = dims[1];

    this->addDependentField(s.fieldTag());
    this->addDependentField(z.fieldTag());

    this->addEvaluatedField(pressure);
    this->setName("Hydrostatic Pressure");

    // Setting parameters
    Teuchos::ParameterList& physics  = *p.get<Teuchos::ParameterList*>("FELIX Physical Parameters");

    rho_i = physics.get<double>("Ice Density",910);
    g     = physics.get<double>("Gravity Acceleration",9.8);
    p_atm = 101325.0; // kg * m^-1 * s^-2
  }

  template<typename EvalT, typename Traits, typename Type>
  void HydrostaticPressure<EvalT,Traits,Type>::
  postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(s,fm);
    this->utils.setFieldData(z,fm);

    this->utils.setFieldData(pressure,fm);
  }

  template<typename EvalT, typename Traits, typename Type>
  void HydrostaticPressure<EvalT,Traits,Type>::
  evaluateFields(typename Traits::EvalData d)
  {
    double pow3 = pow(10.0,3.0);
    for (std::size_t cell = 0; cell < d.numCells; ++cell)
      for (std::size_t node = 0; node < numNodes; ++node)
        pressure(cell,node) = pow3 * rho_i * g * ( s(cell,node) - z(cell,node,2) ) + p_atm;
  }


}


