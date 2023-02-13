//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "LandIce_Gather2DField.hpp"

#include "Albany_AbstractDiscretization.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_GlobalLocalIndexer.hpp"

#include "PHAL_AlbanyTraits.hpp"

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_Print.hpp"
#include "Shards_CellTopology.hpp"

// Nobody includes this file other than the cpp file for ETI,
// so it's ok to inject this name in the LandIce namespace.
using PHALTraits = PHAL::AlbanyTraits;

namespace LandIce {

//**********************************************************************

template<typename EvalT, typename Traits>
Gather2DField<EvalT, Traits>::
Gather2DField(const Teuchos::ParameterList& p,
              const Teuchos::RCP<Albany::Layouts>& dl)
 : field2D(p.get<std::string>("2D Field Name"), dl->node_scalar)
{
  this->addEvaluatedField(field2D);

  fieldLevel = p.get<int>("Field Level");
  TEUCHOS_TEST_FOR_EXCEPTION (fieldLevel<0, Teuchos::Exceptions::InvalidParameter,
      "[Gather2DField] Error! Field level must be non-negative.\n");

  extruded   = p.get<bool>("Extruded");
  numNodes = dl->node_scalar->dimension(1);

  this->setName("Gather2DField"+PHX::print<EvalT>());

  if (p.isType<int>("Offset of First DOF")) {
    offset = p.get<int>("Offset of First DOF");
  } else {
    // Hard-coded for StokesFOThickness
    offset = 2;
  }

  if (p.isType<const std::string>("Mesh Part")) {
    meshPart = p.get<const std::string>("Mesh Part");
  } else {
    meshPart = "upperside";
  }

  this->setName("Gather2DField"+PHX::print<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void Gather2DField<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData /* d */,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(field2D,fm);
}

template<typename EvalT, typename Traits>
void Gather2DField<EvalT, Traits>::
check_topology (const shards::CellTopology& cell_topo) {
  // Get node counts
  if (extruded) {
    const auto topo_hexa  = shards::getCellTopologyData<shards::Hexahedron<8>>();
    const auto topo_wedge = shards::getCellTopologyData<shards::Wedge<6>>();
    TEUCHOS_TEST_FOR_EXCEPTION (
        cell_topo.getName()==topo_hexa->name || cell_topo.getName()==topo_wedge->name, std::runtime_error,
        "Error! Extruded mesh capabilities require HEXA or WEDGE element types.\n");
  }
}

//**********************************************************************
template<>
void Gather2DField<PHALTraits::Residual, PHALTraits>::
evaluateFields (typename PHALTraits::EvalData workset)
{
  // Mesh data
  const auto& layers_data = workset.disc->getLayeredMeshNumberingLO();
  const auto& elem_lids     = workset.disc->getElementLIDs_host(workset.wsIndex);

  const auto  x_data  = Albany::getLocalData(workset.x);
  const auto& dof_mgr       = workset.disc->getNewDOFManager();
  const auto& elem_dof_lids = dof_mgr->elem_dof_lids().host();
  const auto& node_dof_mgr  = workset.disc->getNodeNewDOFManager();
  const auto top = layers_data->top_side_pos;
  const auto bot = layers_data->bot_side_pos;

  if (extruded) {
#ifdef ALBANY_DEBUG
    check_topology(dof_mgr->get_topology());
#endif
    const int field_layer = fieldLevel==0 ? 0 : fieldLevel-1;
    const int field_side_pos = field_layer==0 ? bot : top;
    const auto& field_nodes = dof_mgr->getGIDFieldOffsetsSide(offset,field_side_pos);

    for (std::size_t cell=0; cell<workset.numCells; ++cell ) {
      const int elem_LID = elem_lids(cell);
      const int basal_elem_LID = layers_data->getColumnId(elem_LID);
      const int field_elem_LID = layers_data->getId(basal_elem_LID,field_layer);

      const auto f = [&] (const std::vector<int>& nodes) {
        const int num_nodes = nodes.size();
        for (int inode=0; inode<num_nodes; ++inode) {
          const LO ldof = elem_dof_lids(field_elem_LID,field_nodes[inode]);
          field2D(cell,nodes[inode]) = x_data[ldof];
        }
      };

      // Run lambda on both top and bottom nodes in the cell, but make sure
      // we order them in whatever way the nodes on $field_side_pos would be ordered
      f(node_dof_mgr->getGIDFieldOffsetsSide(0,bot,field_side_pos));
      f(node_dof_mgr->getGIDFieldOffsetsSide(0,top,field_side_pos));
    }
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION (workset.sideSets.is_null(), std::logic_error,
        "Side sets defined in input file but not properly specified on the mesh.\n");

    // Check for early return
    if (workset.sideSets->count(meshPart)==0) {
      return;
    }

    for (const auto& side : workset.sideSets->at(meshPart)) {
      const int cell = side.ws_elem_idx;
      const int elem_LID = elem_lids(cell);
      const int side_pos = side.side_pos;
      
      const auto& offsets = dof_mgr->getGIDFieldOffsetsSide(offset,side_pos);
      const auto& nodes   = node_dof_mgr->getGIDFieldOffsetsSide(0,side_pos);
      const int numSideNodes = nodes.size();
      for (int i=0; i<numSideNodes; ++i){
        const LO ldof = elem_dof_lids(elem_LID,offsets[i]);
        field2D(cell,nodes[i]) = x_data[ldof];
      }
    }
  }
}

//**********************************************************************
template<>
void Gather2DField<PHALTraits::Jacobian, PHALTraits>::
evaluateFields (typename PHALTraits::EvalData workset)
{
  // Mesh data
  const auto& layers_data = workset.disc->getLayeredMeshNumberingLO();
  const auto& elem_lids     = workset.disc->getElementLIDs_host(workset.wsIndex);
  const int   bot = layers_data->bot_side_pos;
  const int   top = layers_data->top_side_pos;

  ALBANY_EXPECT (fieldLevel==0 || fieldLevel==layers_data->numLayers,
      "Field level must be 0 or match the number of layers in the mesh.\n"
      "  - field level: " + std::to_string(fieldLevel) + "\n"
      "  - num layers : " + std::to_string(layers_data->numLayers) + "\n");

  const auto  x_data        = Albany::getLocalData(workset.x);
  const auto& dof_mgr       = workset.disc->getNewDOFManager();
  const auto& elem_dof_lids = dof_mgr->elem_dof_lids().host();
  const auto& node_dof_mgr  = workset.disc->getNodeNewDOFManager();

  const int   neq = dof_mgr->getNumFields();

  if (extruded) {
#ifdef ALBANY_DEBUG
    check_topology(dof_mgr->get_topology());
#endif
    const int field_layer = fieldLevel==0 ? 0 : fieldLevel-1;
    const int field_side_pos = field_layer==0 ? bot : top;
    const auto& field_nodes = dof_mgr->getGIDFieldOffsetsSide(offset,field_side_pos);
    for (std::size_t cell=0; cell<workset.numCells; ++cell ) {
      const int elem_LID = elem_lids(cell);
      const int basal_elem_LID = layers_data->getColumnId(elem_LID);
      const int field_elem_LID = layers_data->getId(basal_elem_LID,field_layer);

      const auto f = [&] (const std::vector<int>& nodes) {
        const int num_nodes = nodes.size();
        for (int inode=0; inode<num_nodes; ++inode) {
          LO ldof = elem_dof_lids(field_elem_LID,field_nodes[inode]);
          int firstunk = neq*nodes[inode] + offset;

          ref_t val = field2D(cell,nodes[inode]);
          val = ScalarT(val.size(),x_data[ldof]);
          val.setUpdateValue(!workset.ignore_residual);
          val.fastAccessDx(firstunk) = workset.j_coeff;
        }
      };

      // Run lambda on both top and bottom nodes in the cell, but make sure
      // we order them in whatever way the nodes on $field_side_pos would be ordered
      f(node_dof_mgr->getGIDFieldOffsetsSide(0,bot,field_side_pos));
      f(node_dof_mgr->getGIDFieldOffsetsSide(0,top,field_side_pos));
    }
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION (workset.sideSets.is_null(), std::logic_error,
        "Side sets defined in input file but not properly specified on the mesh.\n");

    // Check for early return
    if (workset.sideSets->count(meshPart)==0) {
      return;
    }

    for (const auto& side : workset.sideSets->at(meshPart)) {
      // Get the data that corresponds to the side
      const int cell = side.ws_elem_idx;
      const int elem_LID = elem_lids(cell);
      const int side_pos = side.side_pos;
      
      // Note: explicitly as for the offsets to be ordered as the dofs on this side pos
      const auto& offsets =      dof_mgr->getGIDFieldOffsetsSide(offset,side_pos);
      const auto& nodes   = node_dof_mgr->getGIDFieldOffsetsSide(0,     side_pos);
      const int numSideNodes = nodes.size();
      for (int i=0; i<numSideNodes; ++i){
        const LO ldof = elem_dof_lids(elem_LID,offsets[i]);
        ref_t val = field2D(cell,nodes[i]);
        val = ScalarT(val.size(), x_data[ldof]);
        val.fastAccessDx(nodes[i]*neq + offset) = workset.j_coeff;
      }
    }
  }
}

//**********************************************************************
template<>
void Gather2DField<PHALTraits::HessianVec, PHALTraits>::
evaluateFields (typename PHALTraits::EvalData workset)
{
  const auto& hws = workset.hessianWorkset;
  const bool g_xx_is_active = !hws.hess_vec_prod_g_xx.is_null();
  const bool g_xp_is_active = !hws.hess_vec_prod_g_xp.is_null();
  const bool g_px_is_active = !hws.hess_vec_prod_g_px.is_null();
  const bool f_xx_is_active = !hws.hess_vec_prod_f_xx.is_null();
  const bool f_xp_is_active = !hws.hess_vec_prod_f_xp.is_null();
  const bool f_px_is_active = !hws.hess_vec_prod_f_px.is_null();

  // is_x_active is true if we compute the Hessian-vector product contributions of either:
  // Hv_g_xx, Hv_g_xp, Hv_f_xx, or Hv_f_xp, i.e. if the first derivative is w.r.t. the solution.
  // If one of those is active, we have to initialize the first level of AD derivatives:
  // .fastAccessDx().val().
  const bool is_x_active = g_xx_is_active || g_xp_is_active || f_xx_is_active || f_xp_is_active;

  // is_x_direction_active is true if we compute the Hessian-vector product contributions of either:
  // Hv_g_xx, Hv_g_px, Hv_f_xx, or Hv_f_px, i.e. if the second derivative is w.r.t. the solution direction.
  // If one of those is active, we have to initialize the second level of AD derivatives:
  // .val().fastAccessDx().
  const bool is_x_direction_active = g_xx_is_active || g_px_is_active || f_xx_is_active || f_px_is_active;

  if (!is_x_active && !is_x_direction_active) {
    return;
  }

  Teuchos::ArrayRCP<const ST> direction_x_data;
  if (is_x_direction_active) {
    auto direction_x = workset.hessianWorkset.direction_x;

    TEUCHOS_TEST_FOR_EXCEPTION(
        direction_x.is_null(),
        Teuchos::Exceptions::InvalidParameter,
        "\nError in Gather2DField<HessianVec, PHALTraits>: "
        "direction_x is not set and hess_vec_prod_g_xx or"
        "hess_vec_prod_g_px is set.\n");
    direction_x_data = Albany::getLocalData(direction_x->col(0).getConst());
  }

  // Mesh data
  const auto& layers_data = workset.disc->getLayeredMeshNumberingLO();
  const int   top = layers_data->top_side_pos;
  const int   bot = layers_data->bot_side_pos;
  const auto& elem_lids     = workset.disc->getElementLIDs_host(workset.wsIndex);

  const auto x_data = Albany::getLocalData(workset.x);
  const auto& dof_mgr       = workset.disc->getNewDOFManager();
  const auto& elem_dof_lids = dof_mgr->elem_dof_lids().host();
  const auto& node_dof_mgr  = workset.disc->getNodeNewDOFManager();

  const int   neq = dof_mgr->getNumFields();

  if (extruded) {
#ifdef ALBANY_DEBUG
    check_topology(dof_mgr->get_topology());
#endif

    const int field_layer = fieldLevel==0 ? 0 : fieldLevel-1;
    const int field_pos = field_layer==0 ? bot : top;
    // Note: grab sol dofs in same order as the side where the field is defined,
    //       to ensure that corresponding dofs are vertically aligned
    const auto& field_nodes = dof_mgr->getGIDFieldOffsetsSide(offset,field_pos);
    for (std::size_t cell=0; cell<workset.numCells; ++cell ) {
      const int elem_LID = elem_lids(cell);
      const int basal_elem_LID = layers_data->getColumnId(elem_LID);
      const int field_elem_LID = layers_data->getId(basal_elem_LID,field_layer);

      const auto f = [&] (const std::vector<int>& nodes) {
        const int num_nodes = nodes.size();
        for (int inode=0; inode<num_nodes; ++inode) {
          const LO ldof = elem_dof_lids(field_elem_LID,field_nodes[inode]);
          ref_t val = field2D(cell,nodes[inode]);
          val = HessianVecFad(val.size(), x_data[ldof]);
          if (is_x_active) {
            int firstunk = neq*nodes[inode] + offset;
            val.fastAccessDx(firstunk).val() = workset.j_coeff;
          }
          // If we differentiate w.r.t. the solution direction, we have to set
          // the second derivative to the related direction value
          if (is_x_direction_active) {
            val.val().fastAccessDx(0) = direction_x_data[ldof];
          }
        }
      };

      // Run lambda on both top and bottom nodes in the cell
      // Note: grab offsets on top/bot ordered in the same way as on side $field_pos
      //       to guarantee corresponding nodes are vertically aligned.
      f(node_dof_mgr->getGIDFieldOffsetsSide(0,bot,field_pos));
      f(node_dof_mgr->getGIDFieldOffsetsSide(0,top,field_pos));
    }
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION (workset.sideSets.is_null(), std::logic_error,
        "Side sets defined in input file but not properly specified on the mesh.\n");

    // Check for early return
    if (workset.sideSets->count(meshPart)==0) {
      return;
    }

    const auto& sideSet = workset.sideSets->at(meshPart);
    for (const auto& side : sideSet) {
      const int cell = side.ws_elem_idx;
      const int elem_LID = elem_lids(cell);
      const int side_pos = side.side_pos;
      
      // Cannot as for getGIDFieldsOffsetsSide at side_pos, since we need top dofs parsed
      // in same order as both dofs when we access the Fad derivatives
      const auto& offsets = dof_mgr->getGIDFieldOffsetsSide(offset,side_pos);
      const auto& nodes   = node_dof_mgr->getGIDFieldOffsetsSide(0,side_pos);
      const int numSideNodes = nodes.size();
      for (int i=0; i<numSideNodes; ++i){
        const LO ldof = elem_dof_lids(elem_LID,offsets[i]);
        ref_t val = field2D(cell,nodes[i]);
        val = HessianVecFad(val.size(), x_data[ldof]);

        // If we differentiate w.r.t. the solution, we have to set the first
        // derivative to workset.j_coeff
        if (is_x_active)
          val.fastAccessDx(neq*node+this->offset).val() = workset.j_coeff;
        // If we differentiate w.r.t. the solution direction, we have to set
        // the second derivative to the related direction value
        if (is_x_direction_active)
          val.val().fastAccessDx(0) = direction_x_data[ldof];
      }
    }
  }
}

} // namespace LandIce
