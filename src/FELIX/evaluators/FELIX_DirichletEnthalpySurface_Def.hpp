/*
 * FELIX_DirichletEnthalpySurface_Def.hpp
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

template<typename EvalT, typename Traits>
DirichletEnthalpySurface<EvalT,Traits>::
DirichletEnthalpySurface(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl):
    dirTemp        	(p.get<std::string> ("Dirichlet Temperature Surface Variable Name"), dl->node_scalar),
	dirEnth   		(p.get<std::string> ("Dirichlet Enthalpy Surface Variable Name"), dl->node_scalar)
{
	std::vector<PHX::Device::size_type> dims;
	dl->node_scalar->dimensions(dims);

	numNodes = dims[1];

	this->addDependentField(dirTemp);

	this->addEvaluatedField(dirEnth);
	this->setName("Dirichlet Enthalpy Surface");

	// Setting parameters
	Teuchos::ParameterList& physics = *p.get<Teuchos::ParameterList*>("FELIX Physical Parameters");

	c_i = physics.get<double>("Heat capacity of ice", 2009.0);
	T0 = physics.get<double>("Reference Temperature", 240.0);
}

template<typename EvalT, typename Traits>
void DirichletEnthalpySurface<EvalT,Traits>::
postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(dirTemp,fm);

  this->utils.setFieldData(dirEnth,fm);
}

template<typename EvalT, typename Traits>
void DirichletEnthalpySurface<EvalT,Traits>::
evaluateFields(typename Traits::EvalData d)
{
    for (std::size_t cell = 0; cell < d.numCells; ++cell)
    {
    	for (std::size_t node = 0; node < numNodes; ++node)
    	{
    		dirEnth(cell,node) = c_i * ( dirTemp(cell,node) - T0 );
    		//std::cout << dirEnth(cell,node) << std::endl;
    	}
    }
}

}


