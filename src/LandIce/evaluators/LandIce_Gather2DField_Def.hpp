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

  // Get node counts
  auto cell_topo = p.get<Teuchos::RCP<const shards::CellTopology> >("Cell Topology");
  sideDim = cell_topo->getDimension()-1;
  if (extruded) {
    const auto topo_hexa  = shards::getCellTopologyData<shards::Hexahedron<8>>();
    const auto topo_wedge = shards::getCellTopologyData<shards::Wedge<6>>();
    TEUCHOS_TEST_FOR_EXCEPTION (
        cell_topo->getName()==topo_hexa->name || cell_topo->getName()==topo_wedge->name, std::runtime_error,
        "Error! Extruded mesh capabilities require HEXA or WEDGE element types.\n");

  }
  // Shards has both Hexa and Wedge with bot and top in the last two side positions
  m_top_side_pos = cell_topo->getSideCount()-1;
  m_bot_side_pos = m_top_side_pos - 1;
  numSideNodes = cell_topo->getNodeCount(sideDim,m_bot_side_pos);
  numNodes = dl->node_scalar->dimension(1);

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

//**********************************************************************
template<typename EvalT, typename Traits>
void Gather2DField<EvalT, Traits>::
evaluateFields (typename Traits::EvalData workset)
{
  // Precompute some data (only once)
  if (m_bot_dofs_offsets.size()==0) {
    const auto& dof_mgr      = workset.disc->getNewDOFManager();
    const auto& node_dof_mgr = workset.disc->getNodeNewDOFManager();

    m_bot_dofs_offsets.resize("",numSideNodes);
    m_top_dofs_offsets.resize("",numSideNodes);
    m_bot_nodes_offsets.resize("",numSideNodes);
    m_top_nodes_offsets.resize("",numSideNodes);

    auto bot_offsets = dof_mgr->getGIDFieldOffsets_subcell (offset,sideDim,m_bot_side_pos);
    auto top_offsets = dof_mgr->getGIDFieldOffsets_subcell (offset,sideDim,m_top_side_pos);
    auto bot_node_offsets = node_dof_mgr->getGIDFieldOffsets_subcell (0,sideDim,m_bot_side_pos);
    auto top_node_offsets = node_dof_mgr->getGIDFieldOffsets_subcell (0,sideDim,m_top_side_pos);
    for (int i=0; i<numSideNodes; ++i) {
      m_bot_dofs_offsets.host()(i) = bot_offsets[i];
      m_top_dofs_offsets.host()(i) = top_offsets[i];
      m_bot_nodes_offsets.host()(i) = bot_offsets[i];
      m_top_nodes_offsets.host()(i) = top_offsets[i];
    }
    m_bot_dofs_offsets.sync_to_dev();
    m_top_dofs_offsets.sync_to_dev();
    m_bot_nodes_offsets.sync_to_dev();
    m_top_nodes_offsets.sync_to_dev();
  }
  if (extruded) {
    TEUCHOS_TEST_FOR_EXCEPTION (workset.disc->getLayeredMeshNumbering().is_null(),
      std::runtime_error, "Error! No layered numbering in the mesh.\n");

    const auto node_layers_data = workset.disc->getLayeredMeshNumbering();
    ALBANY_EXPECT (fieldLevel==0 || fieldLevel==node_layers_data->numLayers,
        "Field level must be 0 or match the number of layers in the mesh.\n"
        "  - field level: " + std::to_string(fieldLevel) + "\n"
        "  - num layers : " + std::to_string(node_layers_data->numLayers) + "\n");

    if (m_cell_layers_data.is_null()) {
      using LMMI = Albany::LayeredMeshNumbering<int>;
      constexpr auto COL = Albany::LayeredMeshOrdering::COLUMN;

      const auto dof_mgr = workset.disc->getNewDOFManager();

      const auto num_elems = dof_mgr->cell_indexer()->getNumLocalElements();
      const auto numLayers = node_layers_data->numLayers;
      const auto ordering  = node_layers_data->ordering;
      const auto stride = ordering==COL ? numLayers : num_elems;

      m_cell_layers_data = Teuchos::rcp(new LMMI(stride,ordering,node_layers_data->layers_ratio));
      m_field_layer = fieldLevel==0 ? 0 : fieldLevel-1;
    }
  }
  evaluateFieldsImpl (workset);
}

//**********************************************************************
template<>
void Gather2DField<PHALTraits::Residual, PHALTraits>::
evaluateFieldsImpl (typename PHALTraits::EvalData workset)
{
  constexpr auto ALL = Kokkos::ALL();

  const auto& x_data  = Albany::getLocalData(workset.x);
  const auto& dof_mgr       = workset.disc->getNewDOFManager();
  const auto& elem_lids     = workset.disc->getElementLIDs_host(workset.wsIndex);
  const auto& elem_dof_lids = dof_mgr->elem_dof_lids().host();

  if (extruded) {
    const auto& bot_offsets = m_bot_dofs_offsets.host();
    const auto& top_offsets = m_top_dofs_offsets.host();
    const auto& bot_nodes = m_bot_nodes_offsets.host();
    const auto& top_nodes = m_top_nodes_offsets.host();

    for (std::size_t cell=0; cell<workset.numCells; ++cell ) {
      const int elem_LID = elem_lids(cell);
      const int basal_elem_LID = m_cell_layers_data->getColumnId(elem_LID);
      const int field_elem_LID = m_cell_layers_data->getId(basal_elem_LID,m_field_layer);

      const auto dof_lids = Kokkos::subview(elem_dof_lids,field_elem_LID,ALL);
      for (int node=0; node<numSideNodes; ++node) {
        field2D(cell,bot_nodes(node)) = x_data[dof_lids(bot_offsets(node))];
        field2D(cell,top_nodes(node)) = x_data[dof_lids(top_offsets(node))];
      }
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
      const int cell = side.elem_LID;

      const int elem_LID = elem_lids(cell);
      const int side_pos = side.side_pos;
      
      const auto dof_lids = Kokkos::subview(elem_dof_lids,elem_LID,ALL);

      const auto& offsets = (side_pos==m_top_side_pos ? m_top_dofs_offsets : m_bot_dofs_offsets).host();
      const auto& nodes   = (side_pos==m_top_side_pos ? m_top_nodes_offsets : m_bot_nodes_offsets).host();
      for (int i=0; i<numSideNodes; ++i){
        field2D(cell,nodes(i)) = x_data[dof_lids(offsets(i))];
      }
    }
  }
}

//**********************************************************************
template<>
void Gather2DField<PHALTraits::Jacobian, PHALTraits>::
evaluateFieldsImpl (typename PHALTraits::EvalData workset)
{
  constexpr auto ALL = Kokkos::ALL();

  const auto& x_data  = Albany::getLocalData(workset.x);
  const auto& dof_mgr       = workset.disc->getNewDOFManager();
  const auto& elem_lids     = workset.disc->getElementLIDs_host(workset.wsIndex);
  const auto& elem_dof_lids = dof_mgr->elem_dof_lids().host();

  const int neq = dof_mgr->getNumFields();

  if (extruded) {
    const auto& bot_offsets = m_bot_dofs_offsets.host();
    const auto& top_offsets = m_top_dofs_offsets.host();
    const auto& bot_nodes = m_bot_nodes_offsets.host();
    const auto& top_nodes = m_top_nodes_offsets.host();

    int firstunk;
    for (std::size_t cell=0; cell<workset.numCells; ++cell ) {
      const int elem_LID = elem_lids(cell);
      const int basal_elem_LID = m_cell_layers_data->getColumnId(elem_LID);
      const int field_elem_LID = m_cell_layers_data->getId(basal_elem_LID,m_field_layer);

      const auto dof_lids = Kokkos::subview(elem_dof_lids,field_elem_LID,ALL);
      for (int node=0; node<numSideNodes; ++node) {
        firstunk = neq*bot_nodes(node) + offset;
        ref_t val_bot = field2D(cell,bot_nodes(node));
        val_bot = ScalarT(val_bot.size(),x_data[dof_lids(bot_offsets(node))]);

        firstunk = neq*top_nodes(node) + offset;
        ref_t val_top = field2D(cell,top_nodes(node));
        val_top = ScalarT(val_top.size(),x_data[dof_lids(top_offsets(node))]);
      }
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
      const int cell = side.elem_LID;

      const int elem_LID = elem_lids(cell);
      const int side_pos = side.side_pos;
      
      const auto dof_lids = Kokkos::subview(elem_dof_lids,elem_LID,ALL);

      const auto& offsets = (side_pos==m_top_side_pos ? m_top_dofs_offsets : m_bot_dofs_offsets).host();
      const auto& nodes   = (side_pos==m_top_side_pos ? m_top_nodes_offsets : m_bot_nodes_offsets).host();
      for (int i=0; i<numSideNodes; ++i){
        ref_t val = field2D(cell,nodes(i));
        val = ScalarT(val.size(), x_data[dof_lids(offsets(i))]);
        val.fastAccessDx(numSideNodes*neq*fieldLevel+neq*i+offset) = workset.j_coeff;
      }
    }
  }
}

//**********************************************************************
template<>
void Gather2DField<PHALTraits::HessianVec, PHALTraits>::
evaluateFieldsImpl (typename PHALTraits::EvalData workset)
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

  constexpr auto ALL = Kokkos::ALL();
  auto x_data = Albany::getLocalData(workset.x);

  const auto& elem_lids     = workset.disc->getElementLIDs_host(workset.wsIndex);
  const auto& dof_mgr       = workset.disc->getNewDOFManager();
  const auto& elem_dof_lids = dof_mgr->elem_dof_lids().host();

  const int   neq = dof_mgr->getNumFields ();
  if (extruded) {
    const auto& bot_offsets = m_bot_dofs_offsets.host();
    const auto& top_offsets = m_top_dofs_offsets.host();
    const auto& bot_nodes = m_bot_nodes_offsets.host();
    const auto& top_nodes = m_top_nodes_offsets.host();

    int firstunk;
    for (std::size_t cell=0; cell<workset.numCells; ++cell ) {
      const int elem_LID = elem_lids(cell);
      const int basal_elem_LID = m_cell_layers_data->getColumnId(elem_LID);
      const int field_elem_LID = m_cell_layers_data->getId(basal_elem_LID,m_field_layer);

      const auto dof_lids = Kokkos::subview(elem_dof_lids,field_elem_LID,ALL);

      for (int node=0; node<numSideNodes; ++node) {
        LO ldof_top = dof_lids(top_offsets(node));
        LO ldof_bot = dof_lids(bot_offsets(node));

        ref_t val_bot = field2D(cell,bot_nodes(node));
        ref_t val_top = field2D(cell,top_nodes(node));

        val_bot = HessianVecFad(val_bot.size(), x_data[ldof_bot]);
        val_top = HessianVecFad(val_top.size(), x_data[ldof_top]);

        // If we differentiate w.r.t. the solution, we have to set the first
        // derivative to workset.j_coeff
        if (is_x_active) {
          firstunk = neq*bot_nodes(node) + offset;
          val_bot.fastAccessDx(firstunk).val() = workset.j_coeff;
          firstunk = neq*top_nodes(node) + offset;
          val_top.fastAccessDx(firstunk).val() = workset.j_coeff;
        }
        // If we differentiate w.r.t. the solution direction, we have to set
        // the second derivative to the related direction value
        if (is_x_direction_active) {
          val_bot.val().fastAccessDx(0) = direction_x_data[ldof_bot];
          val_top.val().fastAccessDx(0) = direction_x_data[ldof_top];
        }
      }
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
      const int cell = side.elem_LID;
      const int elem_LID = elem_lids(cell);
      const int side_pos = side.side_pos;

      const auto dof_lids = Kokkos::subview(elem_dof_lids,elem_LID,ALL);

      // TODO: cache in the class? The getter is relatively cheap though, so I'm ok for now.
      const auto& offsets = (side_pos==m_top_side_pos ? m_top_dofs_offsets : m_bot_dofs_offsets).host();
      const auto& nodes   = (side_pos==m_top_side_pos ? m_top_nodes_offsets : m_bot_nodes_offsets).host();
      for (int i=0; i<numSideNodes; ++i){
        ref_t val = field2D(cell,nodes(i));
        val = HessianVecFad(val.size(), x_data[dof_lids(offsets(i))]);

        // If we differentiate w.r.t. the solution, we have to set the first
        // derivative to workset.j_coeff
        if (is_x_active)
          val.fastAccessDx(numSideNodes*neq*fieldLevel+neq*i+offset).val() = workset.j_coeff;
        // If we differentiate w.r.t. the solution direction, we have to set
        // the second derivative to the related direction value
        if (is_x_direction_active)
          val.val().fastAccessDx(0) = direction_x_data[dof_lids(offsets(i))];
      }
    }
  }
}

} // namespace LandIce
