/*
 * FELIX_BasalMeltRate_Def.hpp
 *
 *  Created on: Jun 16, 2016
 *      Author: abarone
 */

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

namespace FELIX
{


template<typename EvalT, typename Traits, typename VelocityType>
BasalMeltRate<EvalT,Traits,VelocityType>::
BasalMeltRate(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl_basal):
	basalNormalHeatCold				(p.get<std::string> ("Basal Normal Heat Flux Cold Side Variable Name"),dl_basal->node_scalar),
	basalNormalHeatTemperate 		(p.get<std::string> ("Basal Normal Heat Flux Temperate Side Variable Name"),dl_basal->node_scalar),
	omega							(p.get<std::string> ("Omega Side Variable Name"),dl_basal->node_scalar),
	basal_heat_flux					(p.get<std::string> ("Geotermal Flux Side Variable Name"),dl_basal->node_scalar),
	velocity						(p.get<std::string> ("Velocity Side Variable Name"),dl_basal->node_vector),
	basal_friction					(p.get<std::string> ("Basal Friction Coefficient Side Variable Name"),dl_basal->node_scalar),
	meltEnthalpy					(p.get<std::string> ("Enthalpy Hs Side Variable Name"),dl_basal->node_scalar),
	Enthalpy						(p.get<std::string> ("Enthalpy Side Variable Name"),dl_basal->node_scalar),
	basalMeltRate					(p.get<std::string> ("Basal Melt Rate Variable Name"),dl_basal->node_scalar)
{
	basalSideName = p.get<std::string>("Side Set Name");

	basalMeltRate 		= PHX::MDField<ParamScalarT,Cell,Side,QuadPoint>(p.get<std::string> ("Basal Melt Rate QP Variable Name"), dl_basal->qp_scalar);
	//int1Ddrainage		= PHX::MDField<ScalarT,Cell,Side,QuadPoint>(p.get<std::string> ("Integral1D Drainage Side QP Variable Name"), dl_basal->qp_scalar);
	surface_height_grad = PHX::MDField<ParamScalarT,Cell,Side,QuadPoint,Dim>(p.get<std::string> ("Surface Height Gradient Side QP Variable Name"), dl_basal->qp_gradient);
	thickness_grad      = PHX::MDField<ParamScalarT,Cell,Side,QuadPoint,Dim>(p.get<std::string> ("Thickness Gradient Side QP Variable Name"), dl_basal->qp_gradient);
	velocity    		= PHX::MDField<VelocityType,Cell,Side,QuadPoint,Dim>(p.get<std::string> ("Velocity Side QP Variable Name"), dl->qp_vector);

	this->addDependentField(basalNormalHeatCold);
	this->addDependentField(basalNormalHeatTemperate);
	this->addDependentField(omega);
	this->addDependentField(basal_heat_flux);
	this->addDependentField(velocity);
	this->addDependentField(basal_friction);
	this->addDependentField(meltEnthalpy);
	this->addDependentField(Enthalpy);

	this->addEvaluatedField(basalMeltRate);
	this->setName("Basal Melt Rate");

	std::vector<PHX::DataLayout::size_type> dims;
	dl_basal->node_qp_gradient->dimensions(dims);
	int numSides = dims[1];
	numSideNodes = dims[2];
/*
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
}

template<typename EvalT, typename Traits, typename VelocityType>
void BasalMeltRate<EvalT,Traits,VelocityType>::
postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
{
	this->utils.setFieldData(basalNormalHeatCold,fm);
	this->utils.setFieldData(basalNormalHeatTemperate,fm);
	this->utils.setFieldData(omega,fm);
	this->utils.setFieldData(basal_heat_flux,fm);
	this->utils.setFieldData(velocity,fm);
	this->utils.setFieldData(basal_friction,fm);
	this->utils.setFieldData(meltEnthalpy,fm);
	this->utils.setFieldData(Enthalpy,fm);
	this->utils.setFieldData(basalMeltRate,fm);
}

template<typename EvalT, typename Traits, typename Type>
void VerticalVelocity<EvalT,Traits,Type>::
evaluateFields(typename Traits::EvalData d)
{
	if (d.sideSets->find(basalSideName)==d.sideSets->end())
		return;

	const std::vector<Albany::SideStruct>& sideSet = d.sideSets->at(basalSideName);


	for (auto const& it_side : sideSet)
	{
		// Get the local data of side and cell
		const int cell = it_side.elem_LID;
		const int side = it_side.side_local_id;

		for (int node = 0; node < numSideNodes; ++node)
		{
			for (int qp = 0; qp < numSideQPs; ++qp)
			{
				// check whether w(cell,qp) is correct
				w(cell,qp) +=  basalMeltRate(cell,side,qp) + gradb0 * velocity(cell,side,qp,0) + gradb1 * velocity(cell,side,qp,1); // + int1Ddrainage(cell,side,qp)
			}
		}
	}
}

}


