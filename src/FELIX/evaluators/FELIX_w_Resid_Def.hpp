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
  w_Resid<EvalT,Traits,VelocityType>::
  w_Resid(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl):
  GradVelocity   (p.get<std::string> ("Velocity Gradient QP Variable Name"), dl->qp_vecgradient),
  basalMeltRate   (p.get<std::string>("Basal Melt Rate Variable Name"), dl->node_scalar),
  wBF     	   (p.get<std::string> ("Weighted BF Variable Name"), dl->node_qp_scalar),
  w_z			   (p.get<std::string> ("w Gradient QP Variable Name"), dl->qp_gradient),
  w        (p.get<std::string> ("w Variable Name"), dl->node_scalar),
  Residual 	   (p.get<std::string> ("Residual Variable Name"), dl->node_scalar)
  {
    Teuchos::RCP<shards::CellTopology> cellType;
    cellType = p.get<Teuchos::RCP <shards::CellTopology> > ("Cell Type");

    sideName = p.get<std::string> ("Side Set Name");

    TEUCHOS_TEST_FOR_EXCEPTION (dl->side_layouts.find(sideName)==dl->side_layouts.end(), std::runtime_error,
                                "Error! Basal side data layout not found.\n");
    Teuchos::RCP<Albany::Layouts> dl_side = dl->side_layouts.at(sideName);

    std::vector<PHX::Device::size_type> dims;
    dl->node_qp_vector->dimensions(dims);
    numNodes = dims[1];
    numQPs   = dims[2];
    numSideNodes  = dl_side->node_scalar->dimension(2);

    int numSides = dl_side->node_scalar->dimension(1);
    int sideDim  = cellType->getDimension()-1;

    sideNodes.resize(numSides);
    for (int side=0; side<numSides; ++side)
    {
      //Need to get the subcell exact count, since different sides may have different number of nodes (e.g., Wedge)
      int thisSideNodes = cellType->getNodeCount(sideDim,side);
      sideNodes[side].resize(thisSideNodes);
      for (int node=0; node<thisSideNodes; ++node)
        sideNodes[side][node] = cellType->getNodeMap(sideDim,side,node);
    }

    this->addDependentField(GradVelocity);
    this->addDependentField(basalMeltRate);
    this->addDependentField(wBF);
    this->addDependentField(w);
    this->addDependentField(w_z);

    this->addEvaluatedField(Residual);
    this->setName("w Residual");
  }

  template<typename EvalT, typename Traits, typename VelocityType>
  void w_Resid<EvalT,Traits,VelocityType>::
  postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(GradVelocity,fm);
    this->utils.setFieldData(basalMeltRate,fm);
    this->utils.setFieldData(wBF,fm);
    this->utils.setFieldData(w,fm);
    this->utils.setFieldData(w_z,fm);

    this->utils.setFieldData(Residual,fm);
  }

  template<typename EvalT, typename Traits, typename VelocityType>
  void w_Resid<EvalT,Traits,VelocityType>::
  evaluateFields(typename Traits::EvalData d)
  {
    for (std::size_t cell = 0; cell < d.numCells; ++cell)
      for (std::size_t node = 0; node < numNodes; ++node)
        Residual(cell,node) = 0.0;

    for (std::size_t cell = 0; cell < d.numCells; ++cell)
      for (std::size_t node = 0; node < numNodes; ++node)
        for (std::size_t qp = 0; qp < numQPs; ++qp)
          Residual(cell,node) += ( w_z(cell,qp,2) + GradVelocity(cell,qp,0,0) +  GradVelocity(cell,qp,1,1) ) * wBF(cell,node,qp);

    if (d.sideSets->find(sideName) != d.sideSets->end())
    {
      const std::vector<Albany::SideStruct>& sideSet = d.sideSets->at(sideName);
      for (auto const& it_side : sideSet)
      {
        // Get the local data of side and cell
        const int cell = it_side.elem_LID;
        const int side = it_side.side_local_id;
        for (int inode=0; inode<numSideNodes; ++inode)
          Residual(cell,sideNodes[side][inode]) = w(cell,sideNodes[side][inode])-basalMeltRate(cell,sideNodes[side][inode]);
      }
    }
  }


}








