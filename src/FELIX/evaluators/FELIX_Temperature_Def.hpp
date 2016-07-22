/*
 * FELIX_Temperature_Def.hpp
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
Temperature<EvalT,Traits,Type>::
Temperature(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl):
	meltingTemp    (p.get<std::string> ("Melting Temperature Variable Name"), dl->node_scalar),
	enthalpyHs	   (p.get<std::string> ("Enthalpy Hs Variable Name"), dl->node_scalar),
	enthalpy	   (p.get<std::string> ("Enthalpy Variable Name"), dl->node_scalar),
	temperature	   (p.get<std::string> ("Temperature Variable Name"), dl->node_scalar)
{
	std::vector<PHX::Device::size_type> dims;
	dl->node_qp_vector->dimensions(dims);

	numNodes = dims[1];

	this->addDependentField(meltingTemp);
	this->addDependentField(enthalpyHs);
	this->addDependentField(enthalpy);

	this->addEvaluatedField(temperature);
	this->setName("Temperature");

	// Setting parameters
	Teuchos::ParameterList& physics = *p.get<Teuchos::ParameterList*>("FELIX Physical Parameters");
	rho_i = physics.get<double>("Ice Density", 916.0);
	c_i = physics.get<double>("Heat capacity of ice", 2009.0);
	T0 = physics.get<double>("Reference Temperature", 240.0);
}

template<typename EvalT, typename Traits, typename Type>
void Temperature<EvalT,Traits,Type>::
postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(meltingTemp,fm);
  this->utils.setFieldData(enthalpyHs,fm);
  this->utils.setFieldData(enthalpy,fm);

  this->utils.setFieldData(temperature,fm);
}

template<typename EvalT, typename Traits, typename Type>
void Temperature<EvalT,Traits,Type>::
evaluateFields(typename Traits::EvalData d)
{
    for (std::size_t cell = 0; cell < d.numCells; ++cell)
    {
   		for (std::size_t node = 0; node < numNodes; ++node)
   		{
   			if ( enthalpy(cell,node) < enthalpyHs(cell,node) )
   				temperature(cell,node) = enthalpy(cell,node)/(rho_i * c_i) + T0;
   			else
   				temperature(cell,node) = meltingTemp(cell,node);
   		}
   	}
}


}

