/*
 * FELIX_BasalFrictionHeat_Def.hpp
 *
 *  Created on: May 25, 2016
 *      Author: abarone
 */

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

namespace FELIX
{

template<typename EvalT, typename Traits, typename Type>
BasalFrictionHeat<EvalT,Traits,Type>::
BasalFrictionHeat(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl):
	basalFricHeat   (p.get<std::string> ("Basal Friction Heat Variable Name"), dl->node_scalar)
{
	basalSideName = p.get<std::string>("Side Set Name");

	TEUCHOS_TEST_FOR_EXCEPTION (dl->side_layouts.find(basalSideName)==dl->side_layouts.end(), std::runtime_error, "Error! Basal side data layout not found.\n");

	Teuchos::RCP<Albany::Layouts> dl_basal = dl->side_layouts.at(basalSideName);

	velocity  	= PHX::MDField<Type,Cell,Side,QuadPoint,VecDim>(p.get<std::string> ("Velocity Side QP Variable Name"), dl_basal->qp_vector);
	verticalVel	= PHX::MDField<ScalarT,Cell,Side,QuadPoint>(p.get<std::string>("Vertical Velocity Side QP Variable Name"), dl_basal->qp_scalar);
	beta 	  	= PHX::MDField<ParamScalarT,Cell,Side,QuadPoint>(p.get<std::string> ("Basal Friction Coefficient Side QP Variable Name"), dl_basal->qp_scalar);
	BF        	= PHX::MDField<RealType,Cell,Side,Node,QuadPoint>(p.get<std::string> ("BF Side Name"), dl_basal->node_qp_scalar);
    GradBF    	= PHX::MDField<RealType,Cell,Side,Node,QuadPoint,Dim>(p.get<std::string> ("Gradient BF Side Name"), dl_basal->node_qp_gradient);
	w_measure 	= PHX::MDField<MeshScalarT,Cell,Side,QuadPoint> (p.get<std::string> ("Weighted Measure Name"), dl_basal->qp_scalar);

	Teuchos::ParameterList* SUPG_list = p.get<Teuchos::ParameterList*>("SUPG Settings");
	haveSUPG = SUPG_list->get("Have SUPG Stabilization", false);

	this->addDependentField(velocity);
	this->addDependentField(verticalVel);
	this->addDependentField(beta);
	this->addDependentField(BF);
	this->addDependentField(GradBF);
	this->addDependentField(w_measure);

	this->addEvaluatedField(basalFricHeat);
	this->setName("Basal Friction Heat");

	if (haveSUPG)
	{
		basalFricHeatSUPG  = PHX::MDField<ScalarT,Cell,Node>(p.get<std::string> ("Basal Friction Heat SUPG Variable Name"), dl->node_scalar);
		this->addEvaluatedField(basalFricHeatSUPG);
		this->setName("Basal Friction Heat SUPG");
	}

	std::vector<PHX::DataLayout::size_type> dims;
	dl_basal->node_qp_gradient->dimensions(dims);
	int numSides = dims[1];
	numSideNodes = dims[2];
	numSideQPs   = dims[3];
	sideDim      = dims[4];
	numCellNodes = basalFricHeat.fieldTag().dataLayout().dimension(1);

	dl->node_vector->dimensions(dims);
	vecDimFO     = std::min((int)dims[2],2);

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
}

template<typename EvalT, typename Traits, typename Type>
void BasalFrictionHeat<EvalT,Traits,Type>::
postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
{
	this->utils.setFieldData(velocity,fm);
	this->utils.setFieldData(verticalVel,fm);
	this->utils.setFieldData(beta,fm);
	this->utils.setFieldData(BF,fm);
	this->utils.setFieldData(GradBF,fm);
	this->utils.setFieldData(w_measure,fm);
	this->utils.setFieldData(basalFricHeat,fm);

	if (haveSUPG)
		this->utils.setFieldData(basalFricHeatSUPG,fm);
}

template<typename EvalT, typename Traits, typename Type>
void BasalFrictionHeat<EvalT,Traits,Type>::
evaluateFields(typename Traits::EvalData d)
{
	// Zero out, to avoid leaving stuff from previous workset!
	for (int cell = 0; cell < d.numCells; ++cell)
		for (int node = 0; node < numCellNodes; ++node)
			basalFricHeat(cell,node) = 0.;

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
			  basalFricHeat(cell,sideNodes[side][node]) = 0.;
			  for (int qp = 0; qp < numSideQPs; ++qp)
			  {
				  for (int dim = 0; dim < vecDimFO; ++dim)
				  {
					  basalFricHeat(cell,sideNodes[side][node]) += (1./(3.154*pow(10.0,4.0))) * beta(cell,side,qp) * velocity(cell,side,qp,dim) * velocity(cell,side,qp,dim) *
					 		  	  	  	  	  	  	  	  	  	    BF(cell,side,node,qp) * w_measure(cell,side,qp);
				  }
				  basalFricHeat(cell,sideNodes[side][node]) += (1./(3.154*pow(10.0,4.0))) * beta(cell,side,qp) * verticalVel(cell,side,qp) * verticalVel(cell,side,qp) *
				  					 		  	  	  	  	  	  	  	  	  	    BF(cell,side,node,qp) * w_measure(cell,side,qp);
			  }
		  }
	  }

	  if (haveSUPG)
	  {
		  ScalarT wSUPG;

		  // Zero out, to avoid leaving stuff from previous workset!
		  for (int cell = 0; cell < d.numCells; ++cell)
			  for (int node = 0; node < numCellNodes; ++node)
				  basalFricHeatSUPG(cell,node) = 0.;

		  const std::vector<Albany::SideStruct>& sideSetSUPG = d.sideSets->at(basalSideName);

	      for (auto const& iter_side : sideSetSUPG)
		  {
			  // Get the local data of side and cell
			  const int cell = iter_side.elem_LID;
			  const int side = iter_side.side_local_id;

	    	  for (int node = 0; node < numSideNodes; ++node)
			  {
				  basalFricHeatSUPG(cell,sideNodes[side][node]) = 0.;
				  for (int qp = 0; qp < numSideQPs; ++qp)
				  {
					  wSUPG = (1/(3.154 * pow(10.0,10.0))) *
						      (velocity(cell,side,qp,0)*GradBF(cell,side,node,qp,0) + velocity(cell,side,qp,1)*GradBF(cell,side,node,qp,1) +
						       verticalVel(cell,side,qp)*GradBF(cell,side,node,qp,2))*w_measure(cell,side,qp);

					  basalFricHeatSUPG(cell,sideNodes[side][node]) += (1./(3.154*pow(10.0,4.0))) * beta(cell,side,qp) *
							  	  	  	  	  	  	  	  	  	  	   (velocity(cell,side,qp,0) * velocity(cell,side,qp,0) + velocity(cell,side,qp,1) * velocity(cell,side,qp,1) +
																		verticalVel(cell,side,qp) * verticalVel(cell,side,qp)) * wSUPG;
				  }
			  }
		  }
	  }
}


}
