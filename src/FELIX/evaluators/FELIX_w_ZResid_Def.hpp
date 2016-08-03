/*
 * FELIX_VelocityZ_Def.hpp
 *
 *  Created on: Jun 7, 2016
 *      Author: abarone
 */

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

namespace FELIX
{

template<typename EvalT, typename Traits, typename VelocityType>
w_ZResid<EvalT,Traits,VelocityType>::
w_ZResid(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl):
	GradVelocity   (p.get<std::string> ("Velocity Gradient QP Variable Name"), dl->qp_vecgradient),
	wBF     	   (p.get<std::string> ("Weighted BF Variable Name"), dl->node_qp_scalar),
	w_z			   (p.get<std::string> ("w_z QP Variable Name"), dl->qp_scalar),
	Residual 	   (p.get<std::string> ("Residual Variable Name"), dl->node_scalar)
{
	std::vector<PHX::Device::size_type> dims;
	dl->node_qp_vector->dimensions(dims);

	numNodes = dims[1];
	numQPs   = dims[2];

	this->addDependentField(GradVelocity);
	this->addDependentField(wBF);
	this->addDependentField(w_z);

	this->addEvaluatedField(Residual);
	this->setName("w_z Residual");
}

template<typename EvalT, typename Traits, typename VelocityType>
void w_ZResid<EvalT,Traits,VelocityType>::
postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(GradVelocity,fm);
  this->utils.setFieldData(wBF,fm);
  this->utils.setFieldData(w_z,fm);

  this->utils.setFieldData(Residual,fm);
}

template<typename EvalT, typename Traits, typename VelocityType>
void w_ZResid<EvalT,Traits,VelocityType>::
evaluateFields(typename Traits::EvalData d)
{
    for (std::size_t cell = 0; cell < d.numCells; ++cell)
    	for (std::size_t node = 0; node < numNodes; ++node)
    		Residual(cell,node) = 0.0;

    for (std::size_t cell = 0; cell < d.numCells; ++cell)
   		for (std::size_t node = 0; node < numNodes; ++node)
   			for (std::size_t qp = 0; qp < numQPs; ++qp)
   				Residual(cell,node) += ( w_z(cell,qp) + GradVelocity(cell,qp,0,0) +  GradVelocity(cell,qp,1,1) ) * wBF(cell,node,qp);
}


}








