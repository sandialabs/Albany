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
	enthalpyHs	   (p.get<std::string> ("Enthalpy Hs Variable Name"), dl->node_scalar),
	enthalpy	   (p.get<std::string> ("Enthalpy Variable Name"), dl->node_scalar),
	omega	   	   (p.get<std::string> ("Omega Variable Name"), dl->node_scalar)
{
	// Get Dimensions
	std::vector<PHX::DataLayout::size_type> dims;
	dl->node_qp_vector->dimensions(dims);
	numNodes = dims[1];

	//sideSetName = p.get<std::string> ("Side Set Name");

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
    	for (std::size_t node = 0; node < numNodes; ++node)
    	{
    		if ( enthalpy(cell,node) < enthalpyHs(cell,node) )
    			omega(cell,node) = 0.0;
    	    else
    	    	omega(cell,node) = ( enthalpy(cell,node) - enthalpyHs(cell,node) ) / L;
    	}
    }
}


}





