/*
 * FELIX_LiquidWaterFraction_Def.hpp
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
LiquidWaterFraction<EvalT,Traits,Type>::
LiquidWaterFraction(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl):
	enthalpyHs	   (p.get<std::string> ("Enthalpy Hs QP Variable Name"), dl->qp_scalar),
	enthalpy	   (p.get<std::string> ("Enthalpy QP Variable Name"), dl->qp_scalar),
	omega	   	   (p.get<std::string> ("Omega QP Variable Name"), dl->qp_scalar)
{
	std::vector<PHX::Device::size_type> dims;
	dl->node_scalar->dimensions(dims);

	numQPs = dims[2];

	this->addDependentField(enthalpyHs);
	this->addDependentField(enthalpy);

	this->addEvaluatedField(omega);
	this->setName("Omega");

	// Setting parameters
	Teuchos::ParameterList& physics = *p.get<Teuchos::ParameterList*>("FELIX Physical Parameters");

	L = physics.get<double>("Latent heat of fusion", 334000.0);
}

template<typename EvalT, typename Traits, typename Type>
void LiquidWaterFraction<EvalT,Traits,Type>::
postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(enthalpyHs,fm);
  this->utils.setFieldData(enthalpy,fm);

  this->utils.setFieldData(omega,fm);
}

template<typename EvalT, typename Traits, typename Type>
void LiquidWaterFraction<EvalT,Traits,Type>::
evaluateFields(typename Traits::EvalData d)
{
    for (std::size_t cell = 0; cell < d.numCells; ++cell)
    {
   		for (std::size_t qp = 0; qp < numQPs; ++qp)
   		{
   			if ( enthalpy(cell,qp) < enthalpyHs(cell,qp) )
   				omega(cell,qp) = 0.0;
   			else
   				omega(cell,qp) = ( enthalpy(cell,qp) - enthalpyHs(cell,qp) ) / L;
   		}
   	}
}


}





