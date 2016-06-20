/*
 * FELIX_Dissipation_Def.hpp
 *
 *  Created on: May 19, 2016
 *      Author: abarone
 */

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

namespace FELIX
{

template<typename EvalT, typename Traits>
Dissipation<EvalT,Traits>::
Dissipation(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl):
	mu        	(p.get<std::string> ("Viscosity QP Variable Name"), dl->qp_scalar),
	epsilonSq   (p.get<std::string> ("EpsilonSq QP Variable Name"), dl->qp_scalar),
	diss        (p.get<std::string> ("Dissipation QP Variable Name"), dl->qp_scalar)
{
	std::vector<PHX::Device::size_type> dims;
	dl->node_qp_vector->dimensions(dims);

	numQPs   = dims[2];

	this->addDependentField(mu);
	this->addDependentField(epsilonSq);

	this->addEvaluatedField(diss);
	this->setName("Dissipation");
}

template<typename EvalT, typename Traits>
void Dissipation<EvalT,Traits>::
postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(mu,fm);
  this->utils.setFieldData(epsilonSq,fm);

  this->utils.setFieldData(diss,fm);
}

template<typename EvalT, typename Traits>
void Dissipation<EvalT,Traits>::
evaluateFields(typename Traits::EvalData d)
{
    for (std::size_t cell = 0; cell < d.numCells; ++cell)
    	for (std::size_t qp = 0; qp < numQPs; ++qp)
    	   	diss(cell,qp) = 1.0/(3.154*pow(10.0,4.0)) * 2.0 * mu(cell,qp) * epsilonSq(cell,qp);
}


}



