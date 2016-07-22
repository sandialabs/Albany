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
    homotopy	   (p.get<std::string> ("Continuation Parameter Name"), dl->shared_param),
	omega	   	   (p.get<std::string> ("Omega Variable Name"), dl->node_scalar)
{
	// Get Dimensions
	std::vector<PHX::DataLayout::size_type> dims;
	dl->node_qp_vector->dimensions(dims);
	numNodes = dims[1];

	this->addDependentField(enthalpyHs);
	this->addDependentField(enthalpy);
	this->addDependentField(homotopy);

	this->addEvaluatedField(omega);
	this->setName("Omega");

	// Setting parameters
	Teuchos::ParameterList& physics = *p.get<Teuchos::ParameterList*>("FELIX Physical Parameters");
	rho_w = physics.get<double>("Water Density", 1000.0);
	L = physics.get<double>("Latent heat of fusion", 334000.0);
}

template<typename EvalT, typename Traits, typename Type>
void LiquidWaterFraction<EvalT,Traits,Type>::
postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(enthalpyHs,fm);
  this->utils.setFieldData(enthalpy,fm);
  this->utils.setFieldData(homotopy,fm);

  this->utils.setFieldData(omega,fm);
}

template<typename EvalT, typename Traits, typename Type>
void LiquidWaterFraction<EvalT,Traits,Type>::
evaluateFields(typename Traits::EvalData d)
{
	ScalarT hom = homotopy(0);
	ScalarT alpha = pow(10.0, -8.0 + hom*10);
	double pi = atan(1.) * 4.;

	ScalarT om;

    for (std::size_t cell = 0; cell < d.numCells; ++cell)
    {
    	for (std::size_t node = 0; node < numNodes; ++node)
    	{
			ScalarT scale = - atan(alpha * (enthalpy(cell,node) - enthalpyHs(cell,node)))/pi + 0.5;

			omega(cell,node) = (1-scale) * ( enthalpy(cell,node) - enthalpyHs(cell,node) ) / (rho_w * L);
			/*
			if ( enthalpy(cell,node) < enthalpyHs(cell,node) )
    			om = 0.0;
    	    else
    	    	om = ( enthalpy(cell,node) - enthalpyHs(cell,node) ) / (rho_w * L);
			 */
    		// TODO change here
    		// this is just temporary, it has been done to not let omega be bigger than 1
    		//omega(cell,node) = om / (1 + om);
    	}
    }
}


}





