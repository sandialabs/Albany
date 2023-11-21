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
  numSideNodes = dims[1];
  numSideQPs   = dims[2];
  numCellNodes = enthalpyBasalResid.fieldTag().dataLayout().extent(1);

  dl->node_vector->dimensions(dims);
  vecDimFO     = std::min((int)dims[2],2);

  Teuchos::RCP<shards::CellTopology> cellType;
  cellType = p.get<Teuchos::RCP <shards::CellTopology> > ("Cell Type");
  unsigned int numSides = cellType->getSideCount();
  sideDim      = cellType->getDimension()-1;
  unsigned int nodeMax = 0;
  for (unsigned int side=0; side<numSides; ++side) {
    unsigned int thisSideNodes = cellType->getNodeCount(sideDim,side);
    nodeMax = std::max(nodeMax, thisSideNodes);
  }
  sideNodes = Kokkos::DualView<int**, PHX::Device>("sideNodes", numSides, nodeMax);
  for (unsigned int side=0; side<numSides; ++side) {
    unsigned int thisSideNodes = cellType->getNodeCount(sideDim,side);
    for (unsigned int node=0; node<thisSideNodes; ++node) {
      sideNodes.h_view(side,node) = cellType->getNodeMap(sideDim,side,node);
    }
  }
  sideNodes.modify_host();
  sideNodes.sync_device();

  this->setName("Enthalpy Basal Residual" + PHX::print<EvalT>());
}

template<typename EvalT, typename Traits, typename Type>
KOKKOS_INLINE_FUNCTION
void EnthalpyBasalResid<EvalT,Traits,Type>::
operator() (const Enthalpy_Basal_Residual_Tag& tag, const int& sideSet_idx) const{

  constexpr int maxNumNodesPerSide = 4;

  const int cell = sideSet.ws_elem_idx.d_view(sideSet_idx);
  const int side = sideSet.side_pos.d_view(sideSet_idx);

  ScalarT val[maxNumNodesPerSide] = {};
  for (unsigned int node = 0; node < numSideNodes; ++node) {
      val[node] = ScalarT(0);
      for (unsigned int qp = 0; qp < numSideQPs; ++qp) {
      val[node] += basalMeltRateQP(sideSet_idx,qp) 
                 * BF(sideSet_idx,node,qp) 
                 * w_measure(sideSet_idx,qp);
    }
  }
  
  for (unsigned int node = 0; node < numSideNodes; ++node) {
    enthalpyBasalResid(cell, sideNodes.d_view(side,node)) += val[node];
  }

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
  enthalpyBasalResid.deep_copy(0);

  if (d.sideSetViews->find(basalSideName)==d.sideSetViews->end())
    return;

  sideSet = d.sideSetViews->at(basalSideName);

  Kokkos::parallel_for(Enthalpy_Basal_Residual_Policy(0, sideSet.size), *this);
}

} // namespace LandIce
