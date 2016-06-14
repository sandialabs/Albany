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
LiquidWaterFraction(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl_basal):
	enthalpyHs	   (p.get<std::string> ("Enthalpy Hs Side Variable Name"), dl_basal->node_scalar),
	enthalpy	   (p.get<std::string> ("Enthalpy Side Variable Name"), dl_basal->node_scalar),
	omega	   	   (p.get<std::string> ("Omega Variable Name"), dl_basal->node_scalar)
{
	// Get Dimensions
	std::vector<PHX::DataLayout::size_type> dims;
	dl_basal->qp_gradient->dimensions(dims);
	numSideNodes = dims[2];

	sideSetName = p.get<std::string> ("Side Set Name");

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

    if (d.sideSets->find(sideSetName) != d.sideSets->end())
    {
    	const std::vector<Albany::SideStruct>& sideSet = d.sideSets->at(sideSetName);
    	for (auto const& it_side : sideSet)
    	{
    		// Get the local data of side and cell
    		const int cell = it_side.elem_LID;
    		const int side = it_side.side_local_id;

    		for (int node = 0; node < numSideNodes; ++node)
    		{
    			if ( enthalpy(cell,side,node) < enthalpyHs(cell,side,node) )
    				omega(cell,side,node) = 0.0;
    			else
    				omega(cell,side,node) = ( enthalpy(cell,side,node) - enthalpyHs(cell,side,node) ) / L;
    		}
    	}
    }
}


}





