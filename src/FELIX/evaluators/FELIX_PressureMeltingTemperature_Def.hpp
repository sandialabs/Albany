/*
 * FELIX_PressureMeltingTemperature_Def.hpp
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
PressureMeltingTemperature<EvalT,Traits,Type>::
PressureMeltingTemperature(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl):
	pressure       (p.get<std::string> ("Hydrostatic Pressure QP Variable Name"), dl->qp_scalar),
	meltingTemp    (p.get<std::string> ("Melting Temperature QP Variable Name"), dl->qp_scalar)
{
	std::vector<PHX::Device::size_type> dims;
	dl->node_qp_vector->dimensions(dims);

	numQPs   = dims[2];

	this->addDependentField(pressure);

	this->addEvaluatedField(meltingTemp);
	this->setName("Pressure-melting Temperature");

	// Setting parameters
	Teuchos::ParameterList& physics = *p.get<Teuchos::ParameterList*>("FELIX Physical Parameters");

	beta = physics.get<double>("Clausius-Clapeyron coefficient", 0.0);
}

template<typename EvalT, typename Traits, typename Type>
void PressureMeltingTemperature<EvalT,Traits,Type>::
postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(pressure,fm);

  this->utils.setFieldData(meltingTemp,fm);
}

template<typename EvalT, typename Traits, typename Type>
void PressureMeltingTemperature<EvalT,Traits,Type>::
evaluateFields(typename Traits::EvalData d)
{
    for (std::size_t cell = 0; cell < d.numCells; ++cell)
   		for (std::size_t qp = 0; qp < numQPs; ++qp)
    		meltingTemp(cell,qp) = - beta * pressure(cell,qp) + 273.158004675;
}


}
