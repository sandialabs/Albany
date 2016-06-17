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
BasalNormalVector(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl):
	normal	   		  (p.get<std::string> ("Basal Normal Vector Coords QP Variable Name"), dl->qp_coords)
{
	basalSideName = p.get<std::string>("Side Set Name");

	TEUCHOS_TEST_FOR_EXCEPTION (dl->side_layouts.find(basalSideName)==dl->side_layouts.end(), std::runtime_error, "Error! Basal side data layout not found.\n");

	Teuchos::RCP<Albany::Layouts> dl_basal = dl->side_layouts.at(basalSideName);

	gradSurfHeight    (p.get<std::string> ("Surface Height Gradient Side QP Variable Name"), dl_basal->qp_gradient),
	gradThickness	  (p.get<std::string> ("Thickness Gradient Side QP Variable Name"), dl_basal->qp_gradient),

	this->addDependentField(gradSurfHeight);
	this->addDependentField(gradThickness);

	this->addEvaluatedField(normal);
	this->setName("Basal Normal");

	std::vector<PHX::DataLayout::size_type> dims;
	dl_basal->node_qp_gradient->dimensions(dims);
	int numSides = dims[1];
	numSideNodes = dims[2];
	numSideQPs   = dims[3];
	sideDim      = dims[4];

	// Index of the nodes on the sides in the numeration of the cell
	Teuchos::RCP<shards::CellTopology> cellType;
	cellType = p.get<Teuchos::RCP <shards::CellTopology> > ("Cell Type");
	sideNodes.resize(numSides);
	for (int side=0; side<numSides; ++side)
	{
		// Need to get the subcell exact count, since different sides may have different number of nodes (e.g., Wedge)
	    int thisSideNodes = cellType->getNodeCount(sideDim,side);
	    sideNodes[side].resize(thisSideNodes);
	    for (int node=0; node<thisSideNodes; ++node)
	    {
	    	sideNodes[side][node] = cellType->getNodeMap(sideDim,side,node);
	    }
	}

	Teuchos::RCP<PHX::DataLayout> vector_dl = p.get< Teuchos::RCP<PHX::DataLayout> >("Node QP Vector Data Layout");
	vector_dl->dimensions(dims);
	numNodes = dims[1];
	numQPs   = dims[2];
	numDims  = dims[3];
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
	// Zero out, to avoid leaving stuff from previous workset!
	for (int cell = 0; cell < d.numCells; ++cell)
		for (int node = 0; node < numNodes; ++node)
			for (int qp = 0; qp < numQPs; ++qp)
				for (int dim = 0; dim < numDims; ++dim)
					normal(cell,qp,dim) = 0.;

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

	        for (int node = 0; node < numSideNodes; ++node)
	    	{
				for (int qp = 0; qp < numSideQPs; ++qp)
				{
					normb = sqrt( 1 + ( gradSurfHeight(cell,side,qp,0) - gradThickness(cell,side,qp,0) )^2 + ( gradSurfHeight(cell,side,qp,1) - gradThickness(cell,side,qp,1) )^2 );
					normal(cell, qp, 0) = - ( gradSurfHeight(cell,side,qp,0) - gradThickness(cell,side,qp,0) ) / normb;
					normal(cell, qp, 1) = - ( gradSurfHeight(cell,side,qp,1) - gradThickness(cell,side,qp,1) ) / normb;
					normal(cell, qp, 2) = 1.0/normb;
				}
	    	}
	    }
	}
}


}
