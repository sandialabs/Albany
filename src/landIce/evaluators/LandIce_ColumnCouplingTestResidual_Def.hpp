//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_Print.hpp"

#include "LandIce_ColumnCouplingTestResidual.hpp"
#include "Albany_AbstractDiscretization.hpp"

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
postRegistrationSetup(typename Traits::SetupData /* d */,
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

  const auto node_offsets = workset.disc->getNodeDOFManager()->getGIDFieldOffsetsSidesKokkos(0);

  const auto& sideSet    = workset.sideSetViews->at(sideSetName);
  const auto ws_elem_idx = sideSet.ws_elem_idx.view_device();
  const auto side_pos    = sideSet.side_pos.view_device();
  const int numSideNodes = solution.extent(1);

  Kokkos::parallel_for(this->getName(),
                       Kokkos::RangePolicy<PHX::Device::execution_space>(0,sideSet.size),
                       KOKKOS_CLASS_LAMBDA(const int iside) {
    const int cell = ws_elem_idx(iside);
    const int pos  = side_pos(iside);
    for (int inode=0; inode<numSideNodes; ++inode) {
      residual(cell,node_offsets(pos,inode)) = surf_height(iside,inode) - solution(iside,inode);
    }
  });
}

} // Namespace PHAL
