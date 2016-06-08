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
	meltingTemp    (p.get<std::string> ("Melting Temperature QP Variable Name"), dl->qp_scalar),
	enthalpyHs	   (p.get<std::string> ("Enthalpy Hs QP Variable Name"), dl->qp_scalar),
	enthalpy	   (p.get<std::string> ("Enthalpy QP Variable Name"), dl->qp_scalar),
	temperature	   (p.get<std::string> ("Temperature QP Variable Name"), dl->qp_scalar)
{
	std::vector<PHX::Device::size_type> dims;
	dl->node_qp_vector->dimensions(dims);

	numQPs = dims[2];

	this->addDependentField(meltingTemp);
	this->addDependentField(enthalpyHs);
	this->addDependentField(enthalpy);

	this->addEvaluatedField(temperature);
	this->setName("Temperature");

	// Setting parameters
	Teuchos::ParameterList& physics = *p.get<Teuchos::ParameterList*>("FELIX Physical Parameters");

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
   		for (std::size_t qp = 0; qp < numQPs; ++qp)
   		{
   			if ( enthalpy(cell,qp) < enthalpyHs(cell,qp) )
   				temperature(cell,qp) = enthalpy(cell,qp)/c_i + T0;
   			else
   				temperature(cell,qp) = meltingTemp(cell,qp);
   		}
   	}
}


}

