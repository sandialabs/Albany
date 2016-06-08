/*
 * FELIX_PressureMeltingEnthalpy_Def.hpp
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
PressureMeltingEnthalpy<EvalT,Traits,Type>::
PressureMeltingEnthalpy(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl):
	meltingTemp    (p.get<std::string> ("Melting Temperature QP Variable Name"), dl->qp_scalar),
	enthalpyHs	   (p.get<std::string> ("Enthalpy Hs QP Variable Name"), dl->qp_scalar)
{
	std::vector<PHX::Device::size_type> dims;
	dl->node_qp_vector->dimensions(dims);

	numQPs = dims[2];

	this->addDependentField(meltingTemp);

	this->addEvaluatedField(enthalpyHs);
	this->setName("Pressure-melting Enthalpy");

	// Setting parameters
	Teuchos::ParameterList& physics = *p.get<Teuchos::ParameterList*>("FELIX Physical Parameters");

	c_i = physics.get<double>("Heat capacity of ice", 2009.0);
	T0 = physics.get<double>("Reference Temperature", 240.0);
}

template<typename EvalT, typename Traits, typename Type>
void PressureMeltingEnthalpy<EvalT,Traits,Type>::
postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(meltingTemp,fm);

  this->utils.setFieldData(enthalpyHs,fm);
}

template<typename EvalT, typename Traits, typename Type>
void PressureMeltingEnthalpy<EvalT,Traits,Type>::
evaluateFields(typename Traits::EvalData d)
{
    for (std::size_t cell = 0; cell < d.numCells; ++cell)
   		for (std::size_t qp = 0; qp < numQPs; ++qp)
   			enthalpyHs(cell,qp) = c_i * ( meltingTemp(cell,qp) - T0 );
}


}

