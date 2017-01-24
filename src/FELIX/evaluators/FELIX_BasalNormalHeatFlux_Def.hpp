/*
 * FELIX_BasalNormalHeatFlux_Def.hpp
 *
 *  Created on: Jun 15, 2016
 *      Author: abarone
 */

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

namespace FELIX
{

template<typename EvalT, typename Traits>
BasalNormalHeatFlux<EvalT,Traits>::
BasalNormalHeatFlux(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl):
	gradMeltTemp   			(p.get<std::string> ("Melting Temperature Gradient QP Variable Name"), dl->qp_gradient),
	gradEnthalpy    		(p.get<std::string> ("Enthalpy Gradient QP Variable Name"), dl->qp_gradient),
	normal					(p.get<std::string> ("Basal Normal Vector Coords QP Variable Name"), dl->qp_coords),
	basalNormalHeatCold 	(p.get<std::string> ("Basal Normal Heat Flux Cold Variable Name"), dl->node_scalar),
	basalNormalHeatTemperate(p.get<std::string> ("Basal Normal Heat Flux Temperate Variable Name"), dl->node_scalar)
{
	std::vector<PHX::Device::size_type> dims;
	dl->node_qp_vector->dimensions(dims);

	numNodes = dims[1];
	numQPs   = dims[2];
	numDims  = dims[3];

	this->addDependentField(gradMeltTemp.fieldTag());
	this->addDependentField(gradEnthalpy.fieldTag());
	this->addDependentField(normal.fieldTag());

	this->addEvaluatedField(basalNormalHeatCold);
	this->addEvaluatedField(basalNormalHeatTemperate);
	this->setName("Basal Normal Heat Flux");

	Teuchos::ParameterList* physics_list = p.get<Teuchos::ParameterList*>("FELIX Physical Parameters");
	k_i = physics_list->get("Conductivity of ice", 1.0);
	k_i *= 0.001; //scaling needs to be done to match dimensions
	c_i = physics_list->get("Heat capacity of ice",2000.0);
	K_i = k_i / c_i;

	K_0 = physics_list->get("Diffusivity temperate ice", 0.001);
	K_0 *= 0.001; //scaling needs to be done to match dimensions
}

template<typename EvalT, typename Traits>
void BasalNormalHeatFlux<EvalT,Traits>::
postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(gradMeltTemp,fm);
  this->utils.setFieldData(gradEnthalpy,fm);
  this->utils.setFieldData(normal,fm);

  this->utils.setFieldData(basalNormalHeatCold,fm);
  this->utils.setFieldData(basalNormalHeatTemperate,fm);
}

template<typename EvalT, typename Traits>
void BasalNormalHeatFlux<EvalT,Traits>::
evaluateFields(typename Traits::EvalData d)
{
    for (std::size_t cell = 0; cell < d.numCells; ++cell)
    {
    	for (std::size_t node = 0; node < numNodes; ++node)
    	{
    		basalNormalHeatCold(cell,node) = 0.0;
    		basalNormalHeatTemperate(cell,node) = 0.0;
    	}
    }

    for (std::size_t cell = 0; cell < d.numCells; ++cell)
    {
   		for (std::size_t node = 0; node < numNodes; ++node)
   		{
   			for (std::size_t qp = 0; qp < numQPs; ++qp)
   			{
   				for (std::size_t dim = 0; dim < numDims; ++dim)
   				{
   					basalNormalHeatCold(cell,node) -= K_i * (gradEnthalpy(cell,qp,dim) * normal(cell,qp,dim));
   					basalNormalHeatTemperate(cell,node) -= k_i * (gradMeltTemp(cell,qp,dim) * normal(cell,qp,dim)) - K_0 * (gradEnthalpy(cell,qp,dim) * normal(cell,qp,dim));
   				}
   			}
   		}
    }
}


}
