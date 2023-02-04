//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "LandIce_ScatterResidual2D.hpp"
#include "PHAL_AlbanyTraits.hpp"

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

  fieldLevel = p.get<int>("Field Level");
  meshPart   = p.get<std::string>("Mesh Part");
}

template<typename EvalT, typename Traits>
ScatterResidualWithExtrudedField<EvalT, Traits>::
ScatterResidualWithExtrudedField(const Teuchos::ParameterList& p,
                                 const Teuchos::RCP<Albany::Layouts>& dl)
 : Base(p,dl)
{
  if (p.isType<int>("Offset 2D Field")) {
    offset2DField = p.get<int>("Offset 2D Field");
  } else {
    offset2DField = numFields-1;
  }
  fieldLevel = p.get<int>("Field Level");
}

// **********************************************************************
// Specializations: Jacobian
// **********************************************************************

template<>
void ScatterResidual2D<AlbanyTraits::Jacobian, AlbanyTraits>::
evaluateFields(typename AlbanyTraits::EvalData workset)
{
  TEUCHOS_TEST_FOR_EXCEPTION (workset.sideSets.is_null(), std::logic_error,
      "Side sets not properly specified on the mesh.\n");

  // Check for early return
  if (workset.sideSets->count(meshPart)==0) {
    return;
  }

  constexpr auto ALL = Kokkos::ALL();

  const auto elem_lids    = workset.disc->getElementLIDs_host(workset.wsIndex);
  const auto dof_mgr      = workset.disc->getNewDOFManager();
  const auto node_dof_mgr = workset.disc->getNodeNewDOFManager();
  const auto elem_dof_lids = dof_mgr->elem_dof_lids().host();

  const auto& mesh = workset.disc->getMeshStruct();
  const auto& layers_data  = mesh->local_cell_layers_data;
  const int   numLayers = layers_data->numLayers;
  const auto  bot = layers_data->bot_side_pos;
  const auto  top = layers_data->top_side_pos;
  const auto  field_pos = fieldLevel==numLayers ? top : bot;

  const auto& topo = dof_mgr->get_topology();
  const int numSideNodes = topo.getNodeCount(topo.getDimension()-1,top);
  const int neq = dof_mgr->getNumFields();

  const bool scatter_f = Teuchos::nonnull(workset.f);
  auto f_data = scatter_f ? Albany::getNonconstLocalData(workset.f) : Teuchos::null;
  auto Jac = workset.Jac;

  Teuchos::Array<LO>lcols(neq*numSideNodes*(numLayers+1));
  double one = 1;
  const auto& sideSet = workset.sideSets->at(meshPart);
  for (const auto& side : sideSet) {
    const int cell = side.ws_elem_idx;
    // Get column ID of the cell, since it cell might be at the top!
    const int side_elem_LID = elem_lids(cell);
    const int basal_elem_LID = layers_data->getColumnId(side_elem_LID);

    // Gather Jac col indices, and set Jac=1 outside of the level where the field is defined
    for (int ilev=0; ilev<=numLayers; ++ilev) {
      // Get correct cell layer and correct dofs offsets
      const int ilayer = ilev==numLayers ? ilev-1 : ilev;
      const int pos = ilev==numLayers ? top : bot;
      const int elem_LID = layers_data->getId(basal_elem_LID,ilayer);
      const auto dof_lids = Kokkos::subview(elem_dof_lids,elem_LID,ALL);

      for (int eq=0; eq<neq; ++eq) {
        // Note: cannot use getGIDFieldOffsetsSide with pos, since top nodes must
        //       be parsed in the same 2D order as the bot nodes
        const auto& dof_offsets = dof_mgr->getGIDFieldOffsetsSide(eq,pos,field_pos);
        for (int inode=0; inode<numSideNodes; ++inode) {
          const int lrow = dof_lids(dof_offsets[inode]);

          // add to lcols
          lcols[ilev*neq*numSideNodes + neq*inode + eq] = lrow;
        }
      }

      // Diagonalize Jac outisde of the 2d field location.
      if (ilev!=fieldLevel) {
        const auto& dof_offsets = dof_mgr->getGIDFieldOffsetsSide(this->offset,pos);
        for (int inode=0; inode<numSideNodes; ++inode) {
          const LO lrow = dof_lids(dof_offsets[inode]);
          Albany::setLocalRowValue(Jac, lrow, lrow, one);
        }
      }
    }

    // Cell layer where we'll do the scatter of the 2d residual
    const int layer = fieldLevel==numLayers ? fieldLevel-1 : fieldLevel;
    // Note: cannot use getGIDFieldOffsetsSide with pos, since top nodes must
    //       be parsed in the same 2D order as the bot nodes
    const auto& offsets_2d_field = dof_mgr->getGIDFieldOffsetsSide(this->offset,field_pos);
    const int elem_LID = layers_data->getId(basal_elem_LID,layer);
    const auto dof_lids = Kokkos::subview(elem_dof_lids,elem_LID,ALL);
    const auto& cell_node_pos = node_dof_mgr->getGIDFieldOffsetsSide(0,field_pos);

    // Recall: we scatter a single scalar residual, so no loop on [0,numFields) here.
    for (int inode=0; inode<numSideNodes; ++inode) {
      const int lrow = dof_lids(offsets_2d_field[inode]);
      const int cellNode = cell_node_pos[inode];
      auto res = this->get_resid(cell,cellNode,0);
      if (scatter_f) {
        f_data[lrow] += res.val();
      }
      if (res.hasFastAccess()) {
        Albany::addToLocalRowValues(Jac,lrow,lcols.size(),lcols.data(),&res.fastAccessDx(0));
      } // has fast access
    }
  }
}

// **********************************************************************
template<>
void ScatterResidualWithExtrudedField<AlbanyTraits::Jacobian, AlbanyTraits>::
evaluateFields(typename AlbanyTraits::EvalData workset)
{
  // This happens only once
  this->gather_fields_offsets (workset.disc->getNewDOFManager());

  constexpr auto ALL = Kokkos::ALL();

  const auto& mesh = workset.disc->getMeshStruct();
  const auto& layers_data  = mesh->local_cell_layers_data;
  const auto  bot = layers_data->bot_side_pos;
  const auto  top = layers_data->top_side_pos;

  // Pick element layer that contains the field level
  const auto field_layer = fieldLevel==layers_data->numLayers
                         ? fieldLevel-1 : fieldLevel;

  const auto& elem_lids     = workset.disc->getElementLIDs_host(workset.wsIndex);
  const auto& dof_mgr       = workset.disc->getNewDOFManager();
  const auto& elem_dof_lids = dof_mgr->elem_dof_lids().host();
  const auto& node_dof_mgr = workset.disc->getNodeNewDOFManager();
  
  const auto& top_nodes = node_dof_mgr->getGIDFieldOffsetsTopSide(0);
  const auto& bot_nodes = node_dof_mgr->getGIDFieldOffsetsBotSide(0);

  const auto& field_offsets_top = dof_mgr->getGIDFieldOffsetsTopSide(offset2DField);
  const auto& field_offsets_bot = dof_mgr->getGIDFieldOffsetsBotSide(offset2DField);
  const auto& field_offsets = fieldLevel==field_layer ? field_offsets_bot : field_offsets_top;

  const int neq = dof_mgr->getNumFields();
  const int nunk = this->numNodes*(neq-1);
  const int numSideNodes = field_offsets.size();

  Teuchos::Array<LO> lcols_nunk, lcols_nodes, index;
  lcols_nunk.resize(nunk), index.resize(nunk), lcols_nodes.resize(this->numNodes);

  const bool scatter_f = Teuchos::nonnull(workset.f);
  auto f_data = scatter_f? Albany::getNonconstLocalData(workset.f) : Teuchos::null;
  auto Jac = workset.Jac;

  const auto sol_offsets   = this->m_fields_offsets.host();
  const auto elem_offsets_field2d = dof_mgr->getGIDFieldOffsets(offset2DField);
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
            Albany::addToLocalRowValue(Jac,lrow,lcols_nunk[lunk],res.fastAccessDx(index[lunk]));
          }
        }
      }
    }

    // 2. Contribution of the extruded field

    // Get the local ID of the cell that contains the level where the field is,
    // and the lids of the 2d field in that cell
    const int basal_elem_LID = layers_data->getColumnId(elem_LID);
    const int field_elem_LID = layers_data->getId(basal_elem_LID,field_layer);
    const auto field_elem_dof_lids = Kokkos::subview(elem_dof_lids,field_elem_LID,ALL);
    for (int node=0; node<numSideNodes; ++node) {
      lcols_nodes[bot_nodes[node]] = field_elem_dof_lids(field_offsets[node]);
      lcols_nodes[top_nodes[node]] = field_elem_dof_lids(field_offsets[node]);
    }

    for (int node2d=0; node2d<numSideNodes; ++node2d) {
      for (int pos : {bot, top}) {
        for (int eq=0; eq<this->numFields; ++eq) {
          // Helper lambda
          auto f = [&](const std::vector<int>& nodes,
                       const std::vector<int>& dof_offsets) {
            const LO lrow = dof_lids(dof_offsets[node2d]);
            auto res = this->get_resid(cell,nodes[node2d],this->offset+eq);

            // Safety check: FAD derivs should be inited by now
            assert(res.hasFastAccess());

            if (scatter_f) {
              f_data[lrow] += res.val();
            }

            // Need to do derive one-by-one, since they are strided
            for (int i=0; i<this->numNodes; ++i) {
              Albany::addToLocalRowValue(Jac,lrow,lcols_nodes[i],
                  res.fastAccessDx(sol_offsets(i,offset2DField)));
            }
          };

          if(eq != offset2DField) {
            f(top_nodes,dof_mgr->getGIDFieldOffsetsTopSide(eq));
            f(bot_nodes,dof_mgr->getGIDFieldOffsetsBotSide(eq));
          }
        }
      }
    }
  }
}

} // namespace PHAL
