//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "LandIce_ColumnCouplingTestResidual.hpp"
#include "Albany_AbstractDiscretization.hpp"

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_Print.hpp"
#include "Shards_CellTopology.hpp"

namespace LandIce {

//**********************************************************************
template<typename EvalT, typename Traits>
ColumnCouplingTestResidual<EvalT, Traits>::
ColumnCouplingTestResidual (const Teuchos::ParameterList& p,
                            const Teuchos::RCP<Albany::Layouts>& dl)
{
  sideSetName = p.get<std::string>("Side Set Name");

  auto dl_side = dl->side_layouts.at(sideSetName);

  solution = decltype(solution) (p.get<std::string> ("Solution Name"), dl_side->node_scalar);
  surf_height = decltype(surf_height) (p.get<std::string> ("Surface Height Name"), dl_side->node_scalar);
  residual = decltype(residual) (p.get<std::string> ("Residual Name"), dl->node_scalar);

  this->addDependentField(solution);
  this->addDependentField(surf_height);
  this->addEvaluatedField(residual);

  this->setName("ColumnCouplingTestResidual"+PHX::print<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ColumnCouplingTestResidual<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(solution,fm);
  this->utils.setFieldData(residual,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ColumnCouplingTestResidual<EvalT, Traits>::
evaluateFields (typename Traits::EvalData workset)
{
  TEUCHOS_TEST_FOR_EXCEPTION (workset.sideSets.is_null(), std::logic_error,
      "Side sets not properly specified on the mesh.\n");

  // Check for early return
  if (workset.sideSets->count(sideSetName)==0) {
    return;
  }

  const auto& localDOFView = workset.localDOFViews->at(sideSetName);
  const auto& sideSet = workset.sideSets->at(sideSetName);
  const int numSides = sideSet.size();
  const int numSideNodes = solution.fieldTag().dataLayout().dimension(1);
  auto ctd = workset.disc->getMeshStruct()->getMeshSpecs()[0]->ctd;
  shards::CellTopology topo(&ctd);
  const int sideDim = topo.getDimension()-1;
  for (auto iside=0; iside<numSides; ++iside) {
    const auto& side = sideSet[iside];
    const int cell = side.elem_LID;
    const int pos = side.side_local_id;
    for (int inode=0; inode<numSideNodes; ++inode) {
      const int cellNode = topo.getNodeMap(sideDim,pos,inode);
      residual (cell,cellNode) = surf_height(iside,inode) - solution(iside,inode);
    }
  }
}

} // Namespace PHAL
