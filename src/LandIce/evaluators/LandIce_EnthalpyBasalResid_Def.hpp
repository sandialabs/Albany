/*
 * LandIce_enthalpyBasalResid_Def.hpp
 *
 *  Created on: May 31, 2016
 *      Author: mperego, abarone
 */

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_Print.hpp"
#include "Shards_CellTopology.hpp"

#include "LandIce_EnthalpyBasalResid.hpp"
#include "Albany_DiscretizationUtils.hpp"

#include "Shards_CellTopology.hpp"

namespace LandIce
{

template<typename EvalT, typename Traits, typename Type>
EnthalpyBasalResid<EvalT,Traits,Type>::
EnthalpyBasalResid(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl)
 : enthalpyBasalResid(p.get<std::string> ("Enthalpy Basal Residual Variable Name"), dl->node_scalar)
{
  basalSideName = p.get<std::string>("Side Set Name");

  TEUCHOS_TEST_FOR_EXCEPTION (dl->side_layouts.find(basalSideName)==dl->side_layouts.end(), std::runtime_error, "Error! Basal side data layout not found.\n");

  Teuchos::RCP<Albany::Layouts> dl_basal = dl->side_layouts.at(basalSideName);

  BF         = decltype(BF)(p.get<std::string> ("BF Side Name"), dl_basal->node_qp_scalar);
  w_measure  = decltype(w_measure)(p.get<std::string> ("Weighted Measure Side Name"), dl_basal->qp_scalar);
  basalMeltRateQP = decltype(basalMeltRateQP)(p.get<std::string> ("Basal Melt Rate Side QP Variable Name"), dl_basal->qp_scalar);

  this->addDependentField(BF);
  this->addDependentField(w_measure);
  this->addDependentField(basalMeltRateQP);

  this->addEvaluatedField(enthalpyBasalResid);

  std::vector<PHX::DataLayout::size_type> dims;
  dl_basal->node_qp_gradient->dimensions(dims);
  int numSides = dims[1];
  numSideNodes = dims[2];
  numSideQPs   = dims[3];
  numCellNodes = enthalpyBasalResid.fieldTag().dataLayout().extent(1);

  dl->node_vector->dimensions(dims);
  vecDimFO     = std::min((int)dims[2],2);

  // Index of the nodes on the sides in the numeration of the cell
  Teuchos::RCP<shards::CellTopology> cellType;
  cellType = p.get<Teuchos::RCP <shards::CellTopology> > ("Cell Type");
  sideDim      = cellType->getDimension()-1;
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

  this->setName("Enthalpy Basal Residual" + PHX::print<EvalT>());
}

template<typename EvalT, typename Traits, typename Type>
void EnthalpyBasalResid<EvalT,Traits,Type>::
postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(BF,fm);
  this->utils.setFieldData(w_measure,fm);
  this->utils.setFieldData(basalMeltRateQP,fm);
  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields());
}

template<typename EvalT, typename Traits, typename Type>
void EnthalpyBasalResid<EvalT,Traits,Type>::
evaluateFields(typename Traits::EvalData d)
{
  // Zero out, to avoid leaving stuff from previous workset!
  for (int cell = 0; cell < d.numCells; ++cell)
    for (int node = 0; node < numCellNodes; ++node)
      enthalpyBasalResid(cell,node) = 0.;

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
      int cnode = sideNodes[side][node];
      enthalpyBasalResid(cell,cnode) = 0.;

      for (int qp = 0; qp < numSideQPs; ++qp)
      {
       enthalpyBasalResid(cell,cnode) += basalMeltRateQP(cell,side,qp) *  BF(cell,side,node,qp) * w_measure(cell,side,qp);
      }
    }
  }
}

} // namespace LandIce
