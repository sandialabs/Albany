//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "PHAL_GatherSolutionSide.hpp"

#include "Albany_Macros.hpp"
#include "Albany_ThyraUtils.hpp"

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Shards_CellTopology.hpp"

namespace PHAL {

template<typename EvalT, typename Traits>
GatherSolutionSide<EvalT,Traits>::
GatherSolutionSide(const Teuchos::ParameterList& p,
                   const Teuchos::RCP<Albany::Layouts>& dl)
 : enableSolution       (false)
 , enableSolutionDot    (false)
 , enableSolutionDotDot (false)
{
  sideSetName = p.get<std::string>("Side Set Name");

  numFields = numFieldsDot = numFieldsDotDot = 0;
  offset = offsetDot = offsetDotDot = 0;

  if (p.isType<Teuchos::ArrayRCP<std::string>>("Solution Names")) {
    const auto& solution_names = p.get< Teuchos::ArrayRCP<std::string> >("Solution Names");

    if (!solution_names.is_null()) {
      numFields = solution_names.size();
      val.resize(numFields);
      for (int eq=0; eq<numFields; ++eq) {
        val[eq] = PHX::MDField<ScalarT,Cell,Side,Node> (solution_names[eq],dl->node_scalar);
        this->addEvaluatedField(val[eq]);
      }
      enableSolution = (numFields>0);

      if (p.isType<int>("Offset of First DOF")) {
        offset = p.get<int>("Offset of First DOF");
      }
    }
  }

  // repeat for xdot if transient is enabled
  if (p.isType<Teuchos::ArrayRCP<std::string>>("Solution Names Dot")) {
    const auto& names_dot = p.get< Teuchos::ArrayRCP<std::string> >("Solution Names Dot");

    if (!names_dot.is_null()) {
      numFieldsDot = names_dot.size();
      val_dot.resize(numFieldsDot);
      for (int eq=0; eq<numFieldsDot; ++eq) {
        val_dot[eq] = PHX::MDField<ScalarT,Cell,Side,Node> (names_dot[eq],dl->node_scalar);
        this->addEvaluatedField(val_dot[eq]);
      }
      enableSolutionDot = (numFieldsDot>0);

      if (p.isType<int>("Offset of First DOF Dot")) {
        offsetDot = p.get<int>("Offset of First DOF Dot");
      }
    }
  }

  // repeat for xdotdot if acceleration is enabled
  if (p.isType<Teuchos::ArrayRCP<std::string>>("Solution Names Dot Dot")) {
    const auto& names_dotdot = p.get< Teuchos::ArrayRCP<std::string> >("Solution Names Dot Dot");

    if (!names_dotdot.is_null()) {
      numFieldsDotDot = names_dotdot.size();
      val_dotdot.resize(numFieldsDotDot);
      for (int eq=0; eq<numFieldsDotDot; ++eq) {
        val_dotdot[eq] = PHX::MDField<ScalarT,Cell,Side,Node> (names_dotdot[eq],dl->node_scalar);
        this->addEvaluatedField(val_dotdot[eq]);
      }
      enableSolutionDotDot = (numFieldsDotDot>0);

      if (p.isType<int>("Offset of First DOF Dot Dot")) {
        offsetDotDot = p.get<int>("Offset of First DOF Dot Dot");
      }
    }
  }

  TEUCHOS_TEST_FOR_EXCEPTION (!enableSolution && !enableSolutionDot && !enableSolutionDotDot,
                              std::logic_error,
                              "Error! This GatherSolutionSide evaluator is not gathering any field.\n");

  // Storing the sideNodeId->cellNodeId map
  auto cellType = p.get<Teuchos::RCP <shards::CellTopology> > ("Cell Type");
  int sideDim = cellType->getDimension()-1;
  int numSides = cellType->getSideCount();

  sideNodes.resize(numSides);
  for (int side=0; side<numSides; ++side) {
    int thisSideNodes = cellType->getNodeCount(sideDim,side);
    sideNodes[side].resize(thisSideNodes);
    for (int node=0; node<thisSideNodes; ++node) {
      sideNodes[side][node] = cellType->getNodeMap(sideDim,side,node);
    }
  }
  
  this->setName("Gather Solution Side"+PHX::print<EvalT>() );
}

// **********************************************************************
template<typename EvalT, typename Traits>
void GatherSolutionSide<EvalT,Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& /* fm */)
{
  if (enableSolution) {
    num_side_nodes = val[0].extent(2);
  } else if (enableSolutionDot) {
    num_side_nodes = val_dot[0].extent(2);
  } else {
    num_side_nodes = val_dotdot[0].extent(2);
  }
  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields(),false);
}

template<typename EvalT, typename Traits>
void GatherSolutionSide<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  if (workset.sideSets->find(sideSetName)==workset.sideSets->end()) {
    return;
  }

  const bool gather_x       = enableSolution       && !workset.x.is_null();
  const bool gather_xdot    = enableSolutionDot    && !workset.xdot.is_null();
  const bool gather_xdotdot = enableSolutionDotDot && !workset.xdotdot.is_null();

  Teuchos::ArrayRCP<const ST> x_data, xdot_data, xdotdot_data;
  x_data = Albany::getLocalData(workset.x);
  if (gather_xdot) {
    xdot_data = Albany::getLocalData(workset.xdot);
  }
  if (gather_xdotdot) {
    xdotdot_data = Albany::getLocalData(workset.xdotdot);
  }

  auto nodeID = Kokkos::create_mirror_view(workset.wsElNodeEqID);
  Kokkos::deep_copy(nodeID,workset.wsElNodeEqID);

  const std::vector<Albany::SideStruct>& sideSet = workset.sideSets->at(sideSetName);
  for (auto const& it_side : sideSet) {
    // Get the local data of side and cell
    const int cell = it_side.elem_LID;
    const int side = it_side.side_local_id;

    for (int node=0; node<num_side_nodes; ++node) {
      const int cell_node = sideNodes[side][node];
      for (int isol=0; isol<numFields; ++isol) {
        if (gather_x) {
          const int node_id = nodeID(cell,cell_node,offset+isol);
          val[isol](cell,side,node) = x_data[node_id];
        }

        if (gather_xdot) {
          const int node_id = nodeID(cell,cell_node,offsetDot+isol);
          val_dot[isol](cell,side,node) = xdot_data[node_id];
        }
        if (gather_xdotdot) {
          const int node_id = nodeID(cell,cell_node,offsetDotDot+isol);
          val_dotdot[isol](cell,side,node) = xdotdot_data[node_id];
        }
      }
    }
  }
}

// **********************************************************************
// Specialization: Jacobian
// **********************************************************************

template<>
void GatherSolutionSide<PHAL::AlbanyTraits::Jacobian, PHAL::AlbanyTraits>::
evaluateFields(PHAL::AlbanyTraits::EvalData workset)
{
  if (workset.sideSets->find(sideSetName)==workset.sideSets->end()) {
    return;
  }

  const bool gather_x       = enableSolution       && !workset.x.is_null();
  const bool gather_xdot    = enableSolutionDot    && !workset.xdot.is_null();
  const bool gather_xdotdot = enableSolutionDotDot && !workset.xdotdot.is_null();

  Teuchos::ArrayRCP<const ST> x_data, xdot_data, xdotdot_data;
  x_data = Albany::getLocalData(workset.x);
  if (gather_xdot) {
    xdot_data = Albany::getLocalData(workset.xdot);
  }
  if (gather_xdotdot) {
    xdotdot_data = Albany::getLocalData(workset.xdotdot);
  }

  auto nodeID = Kokkos::create_mirror_view(workset.wsElNodeEqID);
  Kokkos::deep_copy(nodeID,workset.wsElNodeEqID);
  const int neq = nodeID.extent(2);

  using RefType = typename PHAL::Ref<ScalarT>::type;
  const int fad_size = val[0](0,0,0).size();

  const std::vector<Albany::SideStruct>& sideSet = workset.sideSets->at(sideSetName);
  for (auto const& it_side : sideSet) {
    // Get the local data of side and cell
    const int cell = it_side.elem_LID;
    const int side = it_side.side_local_id;

    for (int node=0; node<num_side_nodes; ++node) {
      const int cell_node = sideNodes[side][node];
      const int start = neq * node + offset;
      for (int isol=0; isol<numFields; ++isol) {
        if (gather_x) {
          const int node_id = nodeID(cell,cell_node,offset+isol);
          RefType val_ref = val[isol](cell,side,node);
          val_ref = FadType(fad_size,x_data[node_id]);
          val_ref.fastAccessDx(start + isol) = workset.j_coeff;
        }
        if (gather_xdot) {
          const int node_id = nodeID(cell,cell_node,offsetDot+isol);
          RefType val_dot_ref = val_dot[isol](cell,side,node);
          val_dot_ref = FadType(fad_size,xdot_data[node_id]);
          val_dot_ref.fastAccessDx(start + isol) = workset.m_coeff;
        }
        if (gather_xdotdot) {
          const int node_id = nodeID(cell,cell_node,offsetDotDot+isol);
          RefType val_dotdot_ref = val_dotdot[isol](cell,side,node);
          val_dotdot_ref = FadType(fad_size,xdotdot_data[node_id]);
          val_dotdot_ref.fastAccessDx(start + isol) = workset.n_coeff;
        }
      }
    }
  }
}

// **********************************************************************
// Specialization: Tangent
// **********************************************************************

template<>
void GatherSolutionSide<PHAL::AlbanyTraits::Tangent, PHAL::AlbanyTraits>::
evaluateFields(PHAL::AlbanyTraits::EvalData workset)
{
  if (workset.sideSets->find(sideSetName)==workset.sideSets->end()) {
    return;
  }

  const bool gather_Vx       = enableSolution       && !workset.Vx.is_null();
  const bool gather_Vxdot    = enableSolutionDot    && !workset.Vxdot.is_null();
  const bool gather_Vxdotdot = enableSolutionDotDot && !workset.Vxdotdot.is_null();

  Teuchos::ArrayRCP<const ST> x_data, xdot_data, xdotdot_data;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<const ST>> Vx_data, Vxdot_data, Vxdotdot_data;

  x_data = Albany::getLocalData(workset.x);
  Vx_data = Albany::getLocalData(workset.Vx);
  if (gather_Vxdot) {
    xdot_data = Albany::getLocalData(workset.xdot);
    Vxdot_data = Albany::getLocalData(workset.Vxdot);
  }
  if (gather_Vxdotdot) {
    xdotdot_data = Albany::getLocalData(workset.xdotdot);
    Vxdotdot_data = Albany::getLocalData(workset.Vxdotdot);
  }

  auto nodeID = Kokkos::create_mirror_view(workset.wsElNodeEqID);
  Kokkos::deep_copy(nodeID,workset.wsElNodeEqID);
  const int neq = nodeID.extent(2);

  using RefType = typename PHAL::Ref<ScalarT>::type;
  const int fad_size = val[0](0,0,0).size();

  const std::vector<Albany::SideStruct>& sideSet = workset.sideSets->at(sideSetName);
  for (auto const& it_side : sideSet) {
    // Get the local data of side and cell
    const int cell = it_side.elem_LID;
    const int side = it_side.side_local_id;

    for (int node=0; node<num_side_nodes; ++node) {
      const int cell_node = sideNodes[side][node];
      const int start = neq * node + offset;
      for (int isol=0; isol<numFields; ++isol) {
        if (gather_Vx) {
          const int node_id = nodeID(cell,cell_node,offset+isol);
          RefType val_ref = val[isol](cell,side,node);
          val_ref = TanFadType(fad_size,x_data[node_id]);
          val_ref.fastAccessDx(start + isol) = workset.j_coeff;
          if (workset.j_coeff!=0.0) {
            for (int k=0; k<workset.num_cols_x; k++){
              val_ref.fastAccessDx(k) = workset.j_coeff*Vx_data[k][node_id];
            }
          }
        }
        if (gather_Vxdot) {
          const int node_id = nodeID(cell,cell_node,offsetDot+isol);
          RefType val_dot_ref = val_dot[isol](cell,side,node);
          val_dot_ref = TanFadType(fad_size,xdot_data[node_id]);
          if (workset.m_coeff != 0.0) {
            for (int k=0; k<workset.num_cols_x; k++) {
              val_dot_ref.fastAccessDx(k) = workset.m_coeff*Vxdot_data[k][node_id];
            }
          }
        }
        if (gather_Vxdotdot) {
          const int node_id = nodeID(cell,cell_node,offsetDotDot+isol);
          RefType val_dotdot_ref = val_dotdot[isol](cell,side,node);
          val_dotdot_ref = TanFadType(fad_size,xdotdot_data[node_id]);
          if (workset.n_coeff != 0.0) {
            for (int k=0; k<workset.num_cols_x; k++) {
              val_dotdot_ref.fastAccessDx(k) = workset.n_coeff*Vxdotdot_data[k][node_id];
            }
          }
        }
      }
    }
  }
}

} // namespace PHAL
