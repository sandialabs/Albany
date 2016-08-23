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
	phi	   	   	   (p.get<std::string> ("Water Content Variable Name"), dl->node_scalar)
{
	// Get Dimensions
	std::vector<PHX::DataLayout::size_type> dims;
	dl->node_qp_vector->dimensions(dims);
	numNodes = dims[1];

	this->addDependentField(enthalpyHs);
	this->addDependentField(enthalpy);
	this->addDependentField(homotopy);

	this->addEvaluatedField(phi);
	this->setName("Phi");

	// Setting parameters
	Teuchos::ParameterList& physics = *p.get<Teuchos::ParameterList*>("FELIX Physical Parameters");
	rho_w = physics.get<double>("Water Density", 1000.0);
	L = physics.get<double>("Latent heat of fusion", 334000.0);

	printedAlpha = -1.0;

}

template<typename EvalT, typename Traits, typename Type>
void LiquidWaterFraction<EvalT,Traits,Type>::
postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(enthalpyHs,fm);
  this->utils.setFieldData(enthalpy,fm);
  this->utils.setFieldData(homotopy,fm);

  this->utils.setFieldData(phi,fm);
}

template<typename EvalT, typename Traits, typename Type>
void LiquidWaterFraction<EvalT,Traits,Type>::
evaluateFields(typename Traits::EvalData d)
{
	double pow6 = pow(10.0,6.0);
	ScalarT hom = homotopy(0);
	double pi = atan(1.) * 4.;
	ScalarT phiNode;

	for (std::size_t cell = 0; cell < d.numCells; ++cell)
    {
    	for (std::size_t node = 0; node < numNodes; ++node)
    	{
			if ( enthalpy(cell,node) < enthalpyHs(cell,node) )
				phiNode = 0.0;
    	    else
    	    	phiNode = pow6 * (enthalpy(cell,node) - enthalpyHs(cell,node)) / (rho_w * L);

			if (phi(cell,node) < 0.0)
				phi(cell,node) = 0.0;
			else
				phi(cell,node) = phiNode;
    	}
    }
}


}





