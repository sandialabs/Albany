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
gather_fields_offsets (const Teuchos::RCP<const Albany::DOFManager>& dof_mgr,
                       const Teuchos::RCP<const Albany::DOFManager>& node_dof_mgr)
{
  // Do this only once
  if (m_num_side_nodes.size()>0) {
    return;
  }

  const int numSides = dof_mgr->get_topology().getSideCount();

  int maxSideNodes = 0;
  for (int side=0; side<numSides; ++side) {
    maxSideNodes = std::max(maxSideNodes,static_cast<int>(dof_mgr->getGIDFieldOffsetsSide(offset,side).size()));
  }

  m_num_side_nodes.resize("",numSides);
  m_side_dof_offsets.resize("",numSides,maxSideNodes,numFields);
  m_side_node_offsets.resize("",numSides,maxSideNodes);
  for (int side=0; side<numSides; ++side) {
    const auto& node_offsets = node_dof_mgr->getGIDFieldOffsetsSide(0,side);
    const int numSideNodes = node_offsets.size();
    m_num_side_nodes.host()(side) = numSideNodes;
    for (int node=0; node<numSideNodes; ++node) {
      m_side_node_offsets.host()(side,node) = node_offsets[node];
    }
    for (int eq=0; eq<numFields; ++eq) {
      const auto& offsets = dof_mgr->getGIDFieldOffsetsSide(offset+eq,side);
      ALBANY_ASSERT (static_cast<int>(offsets.size())==numSideNodes,
          "Error! Field " << offset+eq << " has " << offsets.size() << " dofs on side " << side
          << ", but the side has " << numSideNodes << " nodes.\n");
      for (int node=0; node<numSideNodes; ++node) {
        m_side_dof_offsets.host()(side,node,eq) = offsets[node];
      }
    }
  }
  m_num_side_nodes.sync_to_dev();
  m_side_dof_offsets.sync_to_dev();
  m_side_node_offsets.sync_to_dev();
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
  gather_fields_offsets(dof_mgr,disc->getNodeDOFManager());

  constexpr auto ALL = Kokkos::ALL();
  const auto elem_lids      = Kokkos::subview(disc->getWsElementLIDs().dev(),workset.wsIndex,ALL);
  const auto elem_dof_lids  = dof_mgr->elem_dof_lids().dev();
  const auto ws_elem_idx    = sideSet.ws_elem_idx.view_device();
  const auto side_pos       = sideSet.side_pos.view_device();
  const auto num_side_nodes = m_num_side_nodes.dev();
  const auto side_offsets   = m_side_dof_offsets.dev();

  Albany::ThyraVDeviceView<const ST> x_data, xdot_data, xdotdot_data;
  if (gather_x)
    x_data = Albany::getDeviceData(x);
  if (gather_xdot)
    xdot_data = Albany::getDeviceData(xdot);
  if (gather_xdotdot)
    xdotdot_data = Albany::getDeviceData(xdotdot);

  Kokkos::parallel_for(this->getName(),RangePolicy(0,sideSet.size),
                       KOKKOS_CLASS_LAMBDA(const int& iside) {
    const int elem_LID = elem_lids(ws_elem_idx(iside));
    const int pos = side_pos(iside);
    const int numNodes = num_side_nodes(pos);

    for (int eq=0; eq<numFields; ++eq) {
      for (int node=0; node<numNodes; ++node) {
        const auto lid = elem_dof_lids(elem_LID,side_offsets(pos,node,eq));
        if (gather_x) {
          get_ref(iside,node,eq) = x_data(lid);
        }
        if (gather_xdot) {
          get_ref_dot(iside,node,eq) = xdot_data(lid);
        }
        if (gather_xdotdot) {
          get_ref_dotdot(iside,node,eq) = xdotdot_data(lid);
        }
      }
    }
  });
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
  const auto& dof_mgr  = disc->getDOFManager();
  gather_fields_offsets(dof_mgr,disc->getNodeDOFManager());

  constexpr auto ALL = Kokkos::ALL();
  const auto elem_lids      = Kokkos::subview(disc->getWsElementLIDs().dev(),workset.wsIndex,ALL);
  const auto elem_dof_lids  = dof_mgr->elem_dof_lids().dev();
  const auto ws_elem_idx    = sideSet.ws_elem_idx.view_device();
  const auto side_pos       = sideSet.side_pos.view_device();
  const auto num_side_nodes = m_num_side_nodes.dev();
  const auto side_offsets   = m_side_dof_offsets.dev();
  const auto node_offsets   = m_side_node_offsets.dev();

  Albany::ThyraVDeviceView<const ST> x_data, xdot_data, xdotdot_data;
  if (gather_x)
    x_data = Albany::getDeviceData(x);
  if (gather_xdot)
    xdot_data = Albany::getDeviceData(xdot);
  if (gather_xdotdot)
    xdotdot_data = Albany::getDeviceData(xdotdot);

  const int neq = dof_mgr->getNumFields();
  const auto j_coeff = workset.j_coeff;
  const auto m_coeff = workset.m_coeff;
  const auto n_coeff = workset.n_coeff;

  Kokkos::parallel_for(this->getName(),RangePolicy(0,sideSet.size),
                       KOKKOS_CLASS_LAMBDA(const int& iside) {
    const int elem_LID = elem_lids(ws_elem_idx(iside));
    const int pos = side_pos(iside);
    const int numNodes = num_side_nodes(pos);

    for (int eq=0; eq<numFields; ++eq) {
      for (int node=0; node<numNodes; ++node) {
        // NOTE: node_offsets is to compute the idx of the side node within the cell.
        const int start = neq*node_offsets(pos,node) + offset;
        const auto lid = elem_dof_lids(elem_LID,side_offsets(pos,node,eq));
        if (gather_x) {
          ref_t val_ref = get_ref(iside,node,eq);
          val_ref = FadType(val_ref.size(),x_data(lid));
          val_ref.fastAccessDx(start + eq) = j_coeff;
        }
        if (gather_xdot) {
          ref_t val_ref = get_ref_dot(iside,node,eq);
          val_ref = FadType(val_ref.size(),xdot_data(lid));
          val_ref.fastAccessDx(start + eq) = m_coeff;
        }
        if (gather_xdotdot) {
          ref_t val_ref = get_ref_dotdot(iside,node,eq);
          val_ref = FadType(val_ref.size(),xdotdot_data(lid));
          val_ref.fastAccessDx(start + eq) = n_coeff;
        }
      }
    }
  });
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
  const auto& dof_mgr  = disc->getDOFManager();
  gather_fields_offsets(dof_mgr,disc->getNodeDOFManager());

  constexpr auto ALL = Kokkos::ALL();
  const auto elem_lids      = Kokkos::subview(disc->getWsElementLIDs().dev(),workset.wsIndex,ALL);
  const auto elem_dof_lids  = dof_mgr->elem_dof_lids().dev();
  const auto ws_elem_idx    = sideSet.ws_elem_idx.view_device();
  const auto side_pos       = sideSet.side_pos.view_device();
  const auto num_side_nodes = m_num_side_nodes.dev();
  const auto side_offsets   = m_side_dof_offsets.dev();

  Albany::ThyraVDeviceView<const ST> x_data, xdot_data, xdotdot_data;
  if (gather_x)
    x_data = Albany::getDeviceData(x);
  if (gather_xdot)
    xdot_data = Albany::getDeviceData(xdot);
  if (gather_xdotdot)
    xdotdot_data = Albany::getDeviceData(xdotdot);

  Albany::ThyraMVDeviceView<const ST> Vx_data, Vxdot_data, Vxdotdot_data;
  if (gather_Vx)
    Vx_data = Albany::getDeviceData(Vx);
  if (gather_Vxdot)
    Vxdot_data = Albany::getDeviceData(Vxdot);
  if (gather_Vxdotdot)
    Vxdotdot_data = Albany::getDeviceData(Vxdotdot);

  const int ncolsx  = workset.num_cols_x;
  const auto j_coeff = workset.j_coeff;
  const auto m_coeff = workset.m_coeff;
  const auto n_coeff = workset.n_coeff;

  Kokkos::parallel_for(this->getName(),RangePolicy(0,sideSet.size),
                       KOKKOS_CLASS_LAMBDA(const int& iside) {
    const int elem_LID = elem_lids(ws_elem_idx(iside));
    const int pos = side_pos(iside);
    const int numNodes = num_side_nodes(pos);

    for (int eq=0; eq<numFields; ++eq) {
      for (int node=0; node<numNodes; ++node) {
        const auto lid = elem_dof_lids(elem_LID,side_offsets(pos,node,eq));
        if (gather_x) {
          ref_t ref = get_ref(iside,node,eq);
          if (gather_Vx) {
            ref = TanFadType(ref.size(),x_data(lid));
            for (int k=0; k<ncolsx; k++){
              ref.fastAccessDx(k) = j_coeff*Vx_data(lid,k);
            }
          } else {
            ref = TanFadType(x_data(lid));
          }
        }

        if (gather_xdot) {
          ref_t ref = get_ref_dot(iside,node,eq);
          if (gather_Vxdot) {
            ref = TanFadType(ref.size(),xdot_data(lid));
            for (int k=0; k<ncolsx; k++){
              ref.fastAccessDx(k) = m_coeff*Vxdot_data(lid,k);
            }
          } else {
            ref = TanFadType(xdot_data(lid));
          }
        }

        if (gather_xdotdot) {
          ref_t ref = get_ref_dotdot(iside,node,eq);
          if (gather_Vxdotdot) {
            ref = TanFadType(ref.size(),xdotdot_data(lid));
            for (int k=0; k<ncolsx; k++) {
              ref.fastAccessDx(k) = n_coeff*Vxdotdot_data(lid,k);
            }
          } else {
            ref = TanFadType(xdotdot_data(lid));
          }
        }
      }
    }
  });
}

} // namespace PHAL
