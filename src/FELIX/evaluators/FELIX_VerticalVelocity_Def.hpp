/*
 * FELIX_VerticalVelocity_Def.hpp
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


template<typename EvalT, typename Traits, typename VelocityType>
VerticalVelocity<EvalT,Traits,VelocityType>::
VerticalVelocity(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl):
	thickness				(p.get<std::string> ("Thickness Variable Name"), dl->node_scalar),
	int1Dw_z 				(p.get<std::string> ("Integral1D w_z Variable Name"),dl->node_scalar),
	w						(p.get<std::string> ("Vertical Velocity Variable Name"),dl->node_scalar)
{
	/*
	basalSideName = p.get<std::string>("Side Set Name");
	TEUCHOS_TEST_FOR_EXCEPTION (dl->side_layouts.find(basalSideName)==dl->side_layouts.end(), std::runtime_error, "Error! Basal side data layout not found.\n");
	Teuchos::RCP<Albany::Layouts> dl_basal = dl->side_layouts.at(basalSideName);

	basalMeltRate 		= PHX::MDField<ParamScalarT,Cell,Side,QuadPoint>(p.get<std::string> ("Basal Melt Rate QP Variable Name"), dl_basal->qp_scalar);
	//int1Ddrainage		= PHX::MDField<ScalarT,Cell,Side,QuadPoint>(p.get<std::string> ("Integral1D Drainage Side QP Variable Name"), dl_basal->qp_scalar);
	surface_height_grad = PHX::MDField<ParamScalarT,Cell,Side,QuadPoint,Dim>(p.get<std::string> ("Surface Height Gradient Side QP Variable Name"), dl_basal->qp_gradient);
	thickness_grad      = PHX::MDField<ParamScalarT,Cell,Side,QuadPoint,Dim>(p.get<std::string> ("Thickness Gradient Side QP Variable Name"), dl_basal->qp_gradient);
	velocity    		= PHX::MDField<VelocityType,Cell,Side,QuadPoint,Dim>(p.get<std::string> ("Velocity Side QP Variable Name"), dl->qp_vector);
*/
	//this->addDependentField(basalMeltRate);
	//this->addDependentField(int1Ddrainage);
	//this->addDependentField(surface_height_grad);
	//this->addDependentField(thickness_grad);
	//this->addDependentField(velocity);
	this->addDependentField(thickness);
	this->addDependentField(int1Dw_z);

	this->addEvaluatedField(w);
	this->setName("Vertical Velocity");
/*
	std::vector<PHX::DataLayout::size_type> dims;
	dl_basal->node_qp_gradient->dimensions(dims);
	int numSides = dims[1];
	numSideNodes = dims[2];
	numSideQPs   = dims[3];
	sideDim      = dims[4];
	numCellNodes = basalFricHeat.fieldTag().dataLayout().dimension(1);

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
*/
	std::vector<PHX::Device::size_type> dims;
	dl->node_qp_vector->dimensions(dims);

	numNodes = dims[1];
}

template<typename EvalT, typename Traits, typename VelocityType>
void VerticalVelocity<EvalT,Traits,VelocityType>::
postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
{
	//this->utils.setFieldData(basalMeltRate,fm);
	//this->utils.setFieldData(int1Ddrainage,fm);
	//this->utils.setFieldData(surface_height_grad,fm);
	//this->utils.setFieldData(thickness_grad,fm);
	//this->utils.setFieldData(velocity,fm);
	this->utils.setFieldData(thickness,fm);
	this->utils.setFieldData(int1Dw_z,fm);
	this->utils.setFieldData(w,fm);
}

template<typename EvalT, typename Traits, typename Type>
void VerticalVelocity<EvalT,Traits,Type>::
evaluateFields(typename Traits::EvalData d)
{
    for (std::size_t cell = 0; cell < d.numCells; ++cell)
    	for (std::size_t node = 0; node < numNodes; ++node)
    		w(cell,node) = thickness(cell,node) * int1Dw_z(cell,node);
}

}
