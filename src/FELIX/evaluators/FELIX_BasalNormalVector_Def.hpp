/*
 * FELIX_BasalNormalVector_Def.hpp
 *
 *  Created on: Jun 9, 2016
 *      Author: abarone
 */

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

namespace FELIX
{

template<typename EvalT, typename Traits>
BasalNormalVector<EvalT,Traits>::
BasalNormalVector(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl_basal):
	gradSurfHeight    (p.get<std::string> ("Surface Height Gradient Side QP Variable Name"), dl_basal->qp_gradient),
	gradThickness	  (p.get<std::string> ("Thickness Gradient Side QP Variable Name"), dl_basal->qp_gradient),
	normal	   		  (p.get<std::string> ("Basal Normal Vector Coords QP Variable Name"), dl_basal->qp_coords)
{
	std::vector<PHX::Device::size_type> dims;
	dl_basal->node_qp_vector->dimensions(dims);

	numQPs = dims[2];

	this->addDependentField(gradSurfHeight);
	this->addDependentField(gradThickness);

	this->addEvaluatedField(normal);
	this->setName("Basal Normal");
}

template<typename EvalT, typename Traits>
void BasalNormalVector<EvalT,Traits>::
postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(gradSurfHeight,fm);
  this->utils.setFieldData(gradThickness,fm);

  this->utils.setFieldData(normal,fm);
}

template<typename EvalT, typename Traits>
void BasalNormalVector<EvalT,Traits>::
evaluateFields(typename Traits::EvalData d)
{
	TEUCHOS_TEST_FOR_EXCEPTION (d.sideSets==Teuchos::null, std::runtime_error,
	                            "Side sets defined in input file but not properly specified on the mesh.\n");

	ParamScalarT normb;

	if (d.sideSets->find(sideSetName) != d.sideSets->end())
	{
		const std::vector<Albany::SideStruct>& sideSet = workset.sideSets->at(sideSetName);
	    for (auto const& it_side : sideSet)
	    {
	    	// Get the local data of side and cell
	        const int cell = it_side.elem_LID;
	        const int side = it_side.side_local_id;

	        for (int qp = 0; qp < numSideQPs; ++qp)
	        {
	        	normb = sqrt( 1 + ( gradSurfHeight(cell,side,qp,0) - gradThickness(cell,side,qp,0) )^2 + ( gradSurfHeight(cell,side,qp,1) - gradThickness(cell,side,qp,1) )^2 );
	        	normal(cell, side, qp, 0) = - ( gradSurfHeight(cell,side,qp,0) - gradThickness(cell,side,qp,0) )/normb;
	        	normal(cell, side, qp, 1) = - ( gradSurfHeight(cell,side,qp,1) - gradThickness(cell,side,qp,1) )/normb;
	        	normal(cell, side, qp, 2) = 1.0/normb;
	        }
	    }
	}
}


}
