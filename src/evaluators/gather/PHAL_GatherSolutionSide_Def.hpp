//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "PHAL_GatherSolutionSide.hpp"

#include "Albany_DOFManager.hpp"
#include "Albany_Macros.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_AbstractDiscretization.hpp"

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Shards_CellTopology.hpp"

namespace PHAL {

template<typename EvalT, typename Traits>
GatherSolutionSide<EvalT,Traits>::
GatherSolutionSide(const Teuchos::ParameterList& p,
                   const Teuchos::RCP<Albany::Layouts>& dl)
{
  sideSetName = p.get<std::string>("Side Set Name");

  solRank = p.get<int>("Solution Rank");
  TEUCHOS_TEST_FOR_EXCEPTION (solRank<0 || solRank>2, std::runtime_error,
      "Error! Unsupported solution rank (" + std::to_string(solRank) + "\n");

  if (p.isType<int>("Offset of First DOF")) {
    offset = p.get<int>("Offset of First DOF");
  }

  if (p.isType<std::string>("Solution Name")) {
    const auto& name = p.get<std::string>("Solution Name");
    switch (solRank) {
      case 0:
        val = decltype(val) (name,dl->node_scalar);
        this->addEvaluatedField(val);
        numFields = 1;
        break;
      case 1:
        valVec = decltype(valVec) (name,dl->node_vector);
        this->addEvaluatedField(valVec);
        numFields = dl->node_vector->dimension(2);
        break;
      case 2:
        valTensor = decltype(valTensor) (name,dl->node_tensor);
        numDim = dl->node_tensor->dimension(2);
        numFields = numDim*numDim;
        this->addEvaluatedField(valTensor);
        break;
    }
    enableSolution = true;
  }

  if (p.isType<std::string>("Solution Dot Name")) {
    const auto& name = p.get<std::string>("Solution Dot Name");
    switch (solRank) {
      case 0:
        val_dot = decltype(val_dot) (name,dl->node_scalar);
        this->addEvaluatedField(val_dot);
        numFields = 1;
        break;
      case 1:
        valVec_dot = decltype(valVec_dot) (name,dl->node_vector);
        this->addEvaluatedField(valVec_dot);
        numFields = dl->node_vector->dimension(2);
        break;
      case 2:
        valTensor_dot = decltype(valTensor_dot) (name,dl->node_tensor);
        this->addEvaluatedField(valTensor_dot);
        numDim = dl->node_tensor->dimension(2);
        numFields = numDim*numDim;
        break;
    }
    enableSolutionDot = true;
  }

  if (p.isType<std::string>("Solution DotDot Name")) {
    const auto& name = p.get<std::string>("Solution DotDot Name");
    switch (solRank) {
      case 0:
        val_dotdot = decltype(val_dotdot) (name,dl->node_scalar);
        this->addEvaluatedField(val_dotdot);
        numFields = 1;
        break;
      case 1:
        valVec_dotdot = decltype(valVec_dotdot) (name,dl->node_vector);
        this->addEvaluatedField(valVec_dotdot);
        numFields = dl->node_vector->dimension(2);
        break;
      case 2:
        valTensor_dotdot = decltype(valTensor_dotdot) (name,dl->node_tensor);
        this->addEvaluatedField(valTensor_dotdot);
        numDim = dl->node_tensor->dimension(2);
        numFields = numDim*numDim;
        break;
    }
    enableSolutionDotDot = true;
  }

  TEUCHOS_TEST_FOR_EXCEPTION (
      !enableSolution && !enableSolutionDot && !enableSolutionDotDot,
      std::logic_error,
      "Error! This GatherSolutionSide evaluator is not gathering anything.\n");

  this->setName("Gather Solution Side"+PHX::print<EvalT>() );
}

// **********************************************************************
template<typename EvalT, typename Traits>
void GatherSolutionSide<EvalT,Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& /* fm */)
{
  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields(),false);
}

template<typename EvalT, typename Traits>
void GatherSolutionSide<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // Check for early return
  auto sideSet = workset.sideSetViews->at(sideSetName);
  if (sideSet.size==0) {
    return;
  }

  const auto& x       = workset.x;
  const auto& xdot    = workset.xdot;
  const auto& xdotdot = workset.xdotdot;

  const bool gather_x       = enableSolution       && !x.is_null();
  const bool gather_xdot    = enableSolutionDot    && !xdot.is_null();
  const bool gather_xdotdot = enableSolutionDotDot && !xdotdot.is_null();

  const auto& disc     = workset.disc;
  const auto& dof_mgr  = disc->getDOFManager();
  const auto& elem_lids = disc->getElementLIDs_host(workset.wsIndex);
  const auto& elem_dof_lids = dof_mgr->elem_dof_lids().host();

  Teuchos::ArrayRCP<const ST> x_data, xdot_data, xdotdot_data;
  if (gather_x)
    x_data = Albany::getLocalData(x);
  if (gather_xdot)
    xdot_data = Albany::getLocalData(xdot);
  if (gather_xdotdot)
    xdotdot_data = Albany::getLocalData(xdotdot);

  constexpr auto ALL = Kokkos::ALL();
  for (int iside=0; iside<sideSet.size; ++iside) {
    const int ws_elem_idx = sideSet.ws_elem_idx(iside);
    const int elem_LID = elem_lids(ws_elem_idx);
    const int side_pos = sideSet.side_pos(iside);
    const auto dof_lids = Kokkos::subview(elem_dof_lids,elem_LID,ALL);

    for (int eq=0; eq<numFields; ++eq) {
      const auto& offsets = dof_mgr->getGIDFieldOffsetsSide(offset+eq,side_pos);
      for (unsigned node=0; node<offsets.size(); ++node) {
        const auto lid = dof_lids(offsets[node]);
        get_ref(iside,node,eq) = x_data[lid];
        if (gather_xdot) {
          get_ref_dot(iside,node,eq) = xdot_data[lid];
        }
        if (gather_xdotdot) {
          get_ref_dotdot(iside,node,eq) = xdotdot_data[lid];
        }
      }
    }
  }
}

// **********************************************************************
// Specialization: Jacobian
// **********************************************************************

template<>
inline void GatherSolutionSide<PHAL::AlbanyTraits::Jacobian, PHAL::AlbanyTraits>::
evaluateFields(PHAL::AlbanyTraits::EvalData workset)
{
  // Check for early return
  auto sideSet = workset.sideSetViews->at(sideSetName);
  if (sideSet.size==0) {
    return;
  }

  const auto& x       = workset.x;
  const auto& xdot    = workset.xdot;
  const auto& xdotdot = workset.xdotdot;

  const bool gather_x       = enableSolution       && !x.is_null();
  const bool gather_xdot    = enableSolutionDot    && !xdot.is_null();
  const bool gather_xdotdot = enableSolutionDotDot && !xdotdot.is_null();

  const auto& disc     = workset.disc;
  const auto& elem_lids = disc->getElementLIDs_host(workset.wsIndex);
  const auto& dof_mgr  = disc->getDOFManager();
  const auto& node_dof_mgr = disc->getNodeDOFManager();
  const auto& elem_dof_lids = dof_mgr->elem_dof_lids().host();

  Teuchos::ArrayRCP<const ST> x_data, xdot_data, xdotdot_data;
  if (gather_x)
    x_data = Albany::getLocalData(x);
  if (gather_xdot)
    xdot_data = Albany::getLocalData(xdot);
  if (gather_xdotdot)
    xdotdot_data = Albany::getLocalData(xdotdot);

  const int neq = dof_mgr->getNumFields();
  constexpr auto ALL = Kokkos::ALL();

  for (int iside=0; iside<sideSet.size; ++iside) {
    const int icell = sideSet.ws_elem_idx(iside);
    const int elem_LID = elem_lids(icell);
    const int side_pos = sideSet.side_pos(iside);
    const auto dof_lids = Kokkos::subview(elem_dof_lids,elem_LID,ALL);

    for (int eq=0; eq<numFields; ++eq) {
      // NOTE: node_offsets is to compute the idx of the side node within the cell.
      const auto& offsets = dof_mgr->getGIDFieldOffsetsSide(eq,side_pos);
      const auto& node_offsets = node_dof_mgr->getGIDFieldOffsetsSide(0,side_pos);
      const int numNodes = offsets.size();
      for (int node=0; node<numNodes; ++node) {
        const int start = neq*node_offsets[node] + offset;
        const auto lid = dof_lids(offsets[node]);
        if (gather_x) {
          ref_t val_ref = get_ref(iside,node,eq);
          val_ref = FadType(val_ref.size(),x_data[lid]);
          val_ref.fastAccessDx(start + eq) = workset.j_coeff;
        }
        if (gather_xdot) {
          ref_t val_ref = get_ref_dot(iside,node,eq);
          val_ref = FadType(val_ref.size(),xdot_data[lid]);
          val_ref.fastAccessDx(start + eq) = workset.m_coeff;
        }
        if (gather_xdotdot) {
          ref_t val_ref = get_ref_dotdot(iside,node,eq);
          val_ref = FadType(val_ref.size(),xdotdot_data[lid]);
          val_ref.fastAccessDx(start + eq) = workset.n_coeff;
        }
      }
    }
  }
}

// **********************************************************************
// Specialization: Tangent
// **********************************************************************

template<>
inline void GatherSolutionSide<PHAL::AlbanyTraits::Tangent, PHAL::AlbanyTraits>::
evaluateFields(PHAL::AlbanyTraits::EvalData workset)
{
  // Check for early return
  auto sideSet = workset.sideSetViews->at(sideSetName);
  if (sideSet.size==0) {
    return;
  }

  const auto& x        = workset.x;
  const auto& xdot     = workset.xdot;
  const auto& xdotdot  = workset.xdotdot;
  const auto& Vx       = workset.Vx;
  const auto& Vxdot    = workset.Vxdot;
  const auto& Vxdotdot = workset.Vxdotdot;

  const bool gather_x        = enableSolution       && !x.is_null();
  const bool gather_xdot     = enableSolutionDot    && !xdot.is_null();
  const bool gather_xdotdot  = enableSolutionDotDot && !xdotdot.is_null();

  const bool gather_Vx       = workset.j_coeff!=0 && !Vx.is_null();
  const bool gather_Vxdot    = workset.m_coeff!=0 && !Vxdot.is_null();
  const bool gather_Vxdotdot = workset.n_coeff!=0 && !Vxdotdot.is_null();

  const auto& disc     = workset.disc;
  const auto& elem_lids = disc->getElementLIDs_host(workset.wsIndex);
  const auto& dof_mgr  = disc->getDOFManager();
  const auto& elem_dof_lids = dof_mgr->elem_dof_lids().host();

  Teuchos::ArrayRCP<const ST> x_data, xdot_data, xdotdot_data;
  if (gather_x)
    x_data = Albany::getLocalData(x);
  if (gather_xdot)
    xdot_data = Albany::getLocalData(xdot);
  if (gather_xdotdot)
    xdotdot_data = Albany::getLocalData(xdotdot);

  Teuchos::ArrayRCP<Teuchos::ArrayRCP<const ST>> Vx_data, Vxdot_data, Vxdotdot_data;

  if (gather_Vx)
    Vx_data = Albany::getLocalData(workset.Vx);
  if (gather_Vxdot)
    Vxdot_data = Albany::getLocalData(workset.Vxdot);
  if (gather_Vxdotdot)
    Vxdotdot_data = Albany::getLocalData(workset.Vxdotdot);

  constexpr auto ALL = Kokkos::ALL();
  for (int iside=0; iside<sideSet.size; ++iside) {
    const int icell = sideSet.ws_elem_idx(iside);
    const int elem_LID = elem_lids(icell);
    const int side_pos = sideSet.side_pos(iside);
    const auto dof_lids = Kokkos::subview(elem_dof_lids,elem_LID,ALL);

    for (int eq=0; eq<numFields; ++eq) {
      const auto& offsets = dof_mgr->getGIDFieldOffsetsSide(eq,side_pos);
      const int numNodes = offsets.size();
      for (int node=0; node<numNodes; ++node) {
        const auto lid = dof_lids(offsets[node]);
        if (gather_x) {
          ref_t ref = get_ref(iside,node,eq);
          if (gather_Vx) {
            ref = TanFadType(ref.size(),x_data[lid]);
            for (int k=0; k<workset.num_cols_x; k++){
              ref.fastAccessDx(k) = workset.j_coeff*Vx_data[k][lid];
            }
          } else {
            ref = TanFadType(x_data[lid]);
          }
        }

        if (gather_xdot) {
          ref_t ref = get_ref_dot(iside,node,eq);
          if (gather_Vxdot) {
            ref = TanFadType(ref.size(),xdot_data[lid]);
            for (int k=0; k<workset.num_cols_x; k++){
              ref.fastAccessDx(k) = workset.m_coeff*Vxdot_data[k][lid];
            }
          } else {
            ref = TanFadType(xdot_data[lid]);
          }
        }

        if (gather_xdotdot) {
          ref_t ref = get_ref_dotdot(iside,node,eq);
          if (gather_Vxdotdot) {
            ref = TanFadType(ref.size(),xdotdot_data[lid]);
            for (int k=0; k<workset.num_cols_x; k++) {
              ref.fastAccessDx(k) = workset.n_coeff*Vxdotdot_data[k][lid];
            }
          } else {
            ref = TanFadType(xdotdot_data[lid]);
          }
        }
      }
    }
  }
}

} // namespace PHAL
