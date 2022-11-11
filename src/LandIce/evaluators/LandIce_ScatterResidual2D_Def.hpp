//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "LandIce_ScatterResidual2D.hpp"
#include "PHAL_AlbanyTraits.hpp"

#include "Albany_NodalDOFManager.hpp"
#include "Albany_AbstractDiscretization.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_GlobalLocalIndexer.hpp"

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

namespace PHAL {

using Teuchos::arrayView;

template<typename EvalT, typename Traits>
ScatterResidual2D<EvalT,Traits>::
ScatterResidual2D(const Teuchos::ParameterList& p,
                  const Teuchos::RCP<Albany::Layouts>& dl)
 : Base(p,dl)
{
  TEUCHOS_TEST_FOR_EXCEPTION (this->numFields!=1, std::logic_error,
      "Error! ScatterResidual2D only supports scalar fields.\n");

  auto cell_topo  = p.get<Teuchos::RCP<const shards::CellTopology>>("Cell Topology");
  sideDim = cell_topo->getDimension()-1;

  // Ensure we have ONE cell per layer.
  const auto topo_hexa  = shards::getCellTopologyData<shards::Hexahedron<8>>();
  const auto topo_wedge = shards::getCellTopologyData<shards::Wedge<6>>();
  TEUCHOS_TEST_FOR_EXCEPTION (
      cell_topo->getName()==topo_hexa->name || cell_topo->getName()==topo_wedge->name, std::runtime_error,
      "ScatterResidual2D is only for extruded meshes with 1 element per layer.\n");

  // Shards has both Hexa and Wedge with bot and top in the last two side positions
  m_top_side_pos = cell_topo->getSideCount()-1;
  m_bot_side_pos = m_top_side_pos - 1;

  numSideNodes = cell_topo->getNodeCount(sideDim,m_bot_side_pos);

  fieldLevel = p.get<int>("Field Level");
  meshPart   = p.get<std::string>("Mesh Part");
}

template<typename EvalT, typename Traits>
void ScatterResidual2D<EvalT,Traits>::
evaluateFields (typename Traits::EvalData d)
{
  // Do this only once
  if (m_bot_dofs_offsets.size()==0) {
    const auto& dof_mgr = d.disc->getNewDOFManager();

    this->gather_fields_offsets (dof_mgr);

    const int neq = dof_mgr->getNumFields();

    m_bot_dofs_offsets.resize("",numSideNodes,neq);
    m_top_dofs_offsets.resize("",numSideNodes,neq);

    for (int eq=0; eq<neq; ++eq) {
      auto bot_offsets = dof_mgr->getGIDFieldOffsets_subcell (eq,sideDim,m_bot_side_pos);
      auto top_offsets = dof_mgr->getGIDFieldOffsets_subcell (eq,sideDim,m_top_side_pos);
      for (int i=0; i<numSideNodes; ++i) {
        m_bot_dofs_offsets.host()(i,eq) = bot_offsets[i];
        m_top_dofs_offsets.host()(i,eq) = top_offsets[i];
      }
    }

    m_bot_dofs_offsets.sync_to_dev();
    m_top_dofs_offsets.sync_to_dev();
  }

  evaluateFieldsImpl(d);
}

template<typename EvalT, typename Traits>
ScatterResidualWithExtrudedField<EvalT, Traits>::
ScatterResidualWithExtrudedField(const Teuchos::ParameterList& p,
                                 const Teuchos::RCP<Albany::Layouts>& dl)
 : Base(p,dl)
{
  TEUCHOS_TEST_FOR_EXCEPTION (this->numFields!=1, std::logic_error,
      "Error! ScatterResidual2D only supports scalar fields.\n");

  auto cell_topo  = p.get<Teuchos::RCP<const shards::CellTopology>>("Cell Topology");
  sideDim = cell_topo->getDimension()-1;

  // Ensure we have ONE cell per layer.
  const auto topo_hexa  = shards::getCellTopologyData<shards::Hexahedron<8>>();
  const auto topo_wedge = shards::getCellTopologyData<shards::Wedge<6>>();
  TEUCHOS_TEST_FOR_EXCEPTION ( *cell_topo==topo_hexa || *cell_topo==topo_wedge, std::runtime_error,
      "ScatterResidualWithExtrudedField is only for extruded meshes with 1 element per layer.\n");

  // Shards has both Hexa and Wedge with bot and top in the last two side positions
  m_top_side_pos = cell_topo->getSideCount()-1;
  m_bot_side_pos = m_top_side_pos - 1;


  if (p.isType<int>("Offset 2D Field")) {
    offset2DField = p.get<int>("Offset 2D Field");
  } else {
    offset2DField = numFields-1;
  }
  fieldLevel = p.get<int>("Field Level");
}

template<typename EvalT, typename Traits>
void ScatterResidualWithExtrudedField<EvalT,Traits>::
evaluateFields (typename Traits::EvalData workset)
{
  this->gather_fields_offsets (workset.disc->getNewDOFManager());

  // Build a layer numbering for cell LIDs (just once)
  if (m_cell_layers_data.is_null()) {
    const auto& dof_mgr = workset.disc->getNewDOFManager();
    const auto node_layers_data = workset.disc->getLayeredMeshNumbering();

    ALBANY_EXPECT (fieldLevel>=0 && fieldLevel<=node_layers_data->numLayers,
        "Field level out of bounds.\n"
        "  - field level: " + std::to_string(fieldLevel) + "\n"
        "  - num layers : " + std::to_string(node_layers_data->numLayers) + "\n");

    using LMMI = Albany::LayeredMeshNumbering<int>;
    constexpr auto COL = Albany::LayeredMeshOrdering::COLUMN;

    const auto num_elems = dof_mgr->cell_indexer()->getNumLocalElements();
    const auto numLayers = node_layers_data->numLayers;
    const auto ordering  = node_layers_data->ordering;
    const auto stride = ordering==COL ? numLayers : num_elems;
    m_cell_layers_data = Teuchos::rcp(new LMMI(stride,ordering,node_layers_data->layers_ratio));
    m_field_layer = fieldLevel==0 ? 0 : fieldLevel-1;
  }

  // Do this only once
  if (m_bot_dofs_offsets.size()==0) {
    const auto& dof_mgr      = workset.disc->getNewDOFManager();
    const auto& node_dof_mgr = workset.disc->getNodeNewDOFManager();

    m_bot_dofs_offsets.resize("",numSideNodes);
    m_top_dofs_offsets.resize("",numSideNodes);
    m_bot_nodes_offsets.resize("",numSideNodes);
    m_top_nodes_offsets.resize("",numSideNodes);

    auto bot_offsets = dof_mgr->getGIDFieldOffsets_subcell (offset2DField,sideDim,m_bot_side_pos);
    auto top_offsets = dof_mgr->getGIDFieldOffsets_subcell (offset2DField,sideDim,m_top_side_pos);
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

  evaluateFieldsImpl(workset);
}

// **********************************************************************
// Specializations: Jacobian
// **********************************************************************

template<>
void ScatterResidual2D<AlbanyTraits::Jacobian, AlbanyTraits>::
evaluateFieldsImpl(typename AlbanyTraits::EvalData workset)
{
  TEUCHOS_TEST_FOR_EXCEPTION (workset.sideSets.is_null(), std::logic_error,
      "Side sets not properly specified on the mesh.\n");

  // Check for early return
  if (workset.sideSets->count(meshPart)==0) {
    return;
  }

  constexpr auto ALL = Kokkos::ALL();
  constexpr auto COL = Albany::LayeredMeshOrdering::COLUMN;

  const auto elem_lids    = workset.disc->getElementLIDs_host(workset.wsIndex);
  const auto dof_mgr      = workset.disc->getNewDOFManager();
  const auto elem_dof_lids = dof_mgr->elem_dof_lids().host();

  // Build local id layer numbering for cells.
  // IMPORTANT: this works *because* of how we add elements in the ConnManager
  const auto& mesh = workset.disc->getMeshStruct();
  const auto& cell_layers_gid = mesh->layered_mesh_numbering_cells;
  const int numLayers = cell_layers_gid->numLayers;
  const auto ordering = cell_layers_gid->ordering;
  const auto num_elems = dof_mgr->cell_indexer()->getNumLocalElements();
  const auto stride = ordering==COL ? numLayers : num_elems;
  Albany::LayeredMeshNumbering<int> cell_layers_lid(stride,ordering,Teuchos::null);

  const bool scatter_f = Teuchos::nonnull(workset.f);
  auto f_data = scatter_f ? Albany::getNonconstLocalData(workset.f) : Teuchos::null;
  auto Jac = workset.Jac;

  const int  neq = dof_mgr->getNumFields();
  Teuchos::Array<LO> lcols;
  lcols.reserve(neq*numNodes*(numLayers+1));
  double one = 1;
  auto diagonal_value = arrayView(&one,1);
  const auto& sideSet = workset.sideSets->at(meshPart);
  for (const auto& side : sideSet) {
    const int pos  = side.side_pos;
    const int cell = side.elem_LID;

    const int basal_elem_LID = elem_lids(cell);

    // Gather Jac col indices, and set Jac=1 outside of the level where the field is defined
    lcols.resize(neq*numSideNodes*(numLayers+1));
    for (int ilev=0; ilev<=numLayers; ++ilev) {
      // Get correct cell layer and correct dofs offsets
      const int ilayer = ilev==numLayers ? ilev-1 : ilev;
      const auto dofs_offsets = ilev==numLayers ? m_top_dofs_offsets.host() : m_bot_dofs_offsets.host();

      const int elem_LID = cell_layers_lid.getId(basal_elem_LID,ilayer);

      const auto dof_lids = Kokkos::subview(elem_dof_lids,elem_LID,ALL);

      for (int inode=0; inode<numSideNodes; ++inode) {
        for (int eq=0; eq<neq; ++eq) {
          const int lrow = dof_lids(dofs_offsets(inode,eq));

          // add to lcols
          lcols[ilev*neq*numSideNodes*neq*inode + eq] = lrow;
        }

        // Diagonalize Jac outisde of the 2d field location.
        if (ilev!=fieldLevel) {
          const LO lrow = dof_lids(dofs_offsets(inode,this->offset));
          Albany::setLocalRowValues(Jac, lrow, arrayView(&lrow,1), diagonal_value);
        }
      }
    }

    // Cell layer where we'll do the scatter of the 2d residual
    int layer = fieldLevel==numLayers ? numLayers-1 : 0;
    auto offsets_2d_field = fieldLevel==numLayers ? m_top_dofs_offsets.host() : m_bot_dofs_offsets.host();
    const int elem_LID = cell_layers_lid.getId(basal_elem_LID,layer);
    const auto dof_lids = Kokkos::subview(elem_dof_lids,elem_LID,ALL);

    // Recall: we scatter a single scalar residual, so no loop on [0,numFields) here.
    for (int i=0; i<numSideNodes; ++i) {
      const int lrow = dof_lids(offsets_2d_field(i,this->offset));
      auto res = this->get_resid(cell,i,0);
      if (scatter_f) {
        f_data[lrow] += res.val();
      }
      if (res.hasFastAccess()) {
        Albany::addToLocalRowValues(Jac,lrow,lcols(), arrayView(&(res.fastAccessDx(0)),lcols.size()));
      } // has fast access
    }
  }
}

// **********************************************************************
template<>
void ScatterResidualWithExtrudedField<AlbanyTraits::Jacobian, AlbanyTraits>::
evaluateFieldsImpl(typename AlbanyTraits::EvalData workset)
{
  constexpr auto ALL = Kokkos::ALL();
  constexpr auto COL = Albany::LayeredMeshOrdering::COLUMN;

  const auto& elem_lids     = workset.disc->getElementLIDs_host(workset.wsIndex);
  const auto& dof_mgr       = workset.disc->getNewDOFManager();
  const auto& elem_dof_lids = dof_mgr->elem_dof_lids().host();

  const int neq = dof_mgr->getNumFields();
  const int nunk = this->numNodes*(neq-1);

  Teuchos::Array<LO> lcols_nunk, lcols_nodes, index;
  lcols_nunk.resize(nunk), index.resize(nunk), lcols_nodes.resize(this->numNodes);

  const bool scatter_f = Teuchos::nonnull(workset.f);
  auto f_data = scatter_f? Albany::getNonconstLocalData(workset.f) : Teuchos::null;
  auto Jac = workset.Jac;

  const auto& bot_offsets = m_bot_dofs_offsets.host();
  const auto& top_offsets = m_top_dofs_offsets.host();
  const auto& bot_nodes   = m_bot_nodes_offsets.host();
  const auto& top_nodes   = m_top_nodes_offsets.host();

  const auto sol_offsets   = this->m_fields_offsets.host();
  const auto m_elem_offsets_field2d = dof_mgr->getGIDFieldOffsets(offset2DField);
  for (size_t cell=0; cell<workset.numCells; ++cell ) {
    // 1. All contributions except the extruded field
    const int elem_LID = elem_lids(cell);
    const auto dof_lids = Kokkos::subview(elem_dof_lids,elem_LID,ALL);

    // Local Unks: gather LID of all other dofs (not the extruded field)
    for (int node=0, i=0; node<this->numNodes; ++node){
      for (int eq=0; eq<neq; ++eq) {
        if(eq != offset2DField) {
          lcols_nunk[i] = dof_lids(sol_offsets(node,eq));
          index[i++] = sol_offsets(node,eq);
        }
      }
    }

    for (int node=0; node<this->numNodes; ++node) {
      for (int eq=0; eq<this->numFields; ++eq) {
        if(this->offset + eq != offset2DField) {
          auto res = this->get_resid(cell,node,eq);

          // Safety check: FAD derivs should be inited by now
          assert(res.hasFastAccess());

          const int lrow = dof_lids(sol_offsets(node,this->offset + eq));
          if (scatter_f) {
            f_data[lrow] += res.val();
          }

          // Need to do derivs one-by-one, since we have a 2-level indirection
          for (int lunk = 0; lunk < nunk; lunk++) {
            Albany::addToLocalRowValues(Jac,lrow,
                                        arrayView(&lcols_nunk[lunk],1),
                                        arrayView(&(res.fastAccessDx(index[lunk])), 1));
          }
        }
      }
    }

    // 2. Contribution of the extruded field

    // Get the local ID of the cell that contains the level where the field is,
    // and the lids of the 2d field in that cell
    const int basal_elem_LID = m_cell_layers_data->getColumnId(elem_LID);
    const int field_elem_LID = m_cell_layers_data->getId(basal_elem_LID,m_field_layer);
    const auto field_elem_dof_lids = Kokkos::subview(elem_dof_lids,field_elem_LID,ALL);
    for (int node=0; node<numSideNodes; ++node) {
      lcols_nodes[top_nodes[node]] = field_elem_dof_lids(top_offsets[node]);
      lcols_nodes[bot_nodes[node]] = field_elem_dof_lids(bot_offsets[node]);
    }

    for (int node=0; node<this->numNodes; ++node) {
      for (int eq=0; eq<this->numFields; ++eq) {
        if(eq != offset2DField) {
          const LO lrow = dof_lids(sol_offsets(node,eq));
          auto res = this->get_resid(cell,node,eq);

          // Safety check: FAD derivs should be inited by now
          assert(res.hasFastAccess());

          if (scatter_f) {
            f_data[lrow] += res.val();
          }

          // Need to do derive one-by-one, since they are strided
          for (int i=0; i<this->numNodes; ++i) {
            Albany::addToLocalRowValues(Jac,lrow,
                                        arrayView(&lcols_nodes[i],1),
                                        arrayView(&(res.fastAccessDx(sol_offsets(i,offset2DField))), 1));
          }
        }
      }
    }
  }
}

} // namespace PHAL
