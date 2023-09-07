//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_DOF_MANAGER_HPP
#define ALBANY_DOF_MANAGER_HPP

#include "Albany_GlobalLocalIndexer.hpp"
#include "Albany_ThyraTypes.hpp"
#include "Albany_DualView.hpp"

#include "Panzer_DOFManager.hpp"
#include "Albany_ConnManager.hpp"
#include "Teuchos_RCP.hpp"

namespace Albany {


// A DOF manager is defined over a mesh part. This part can be
// one of the element blocks, or it can be any sub-part of it.
// In either case, the dof lids/gids stored in the class will
// have dimensions as if the DOF mgr is defined over all the
// conn_mgr elem blocks. However, if part_name refers to a part
// that is strictly a subset of the blocks, then the lids at
// entries corresponding to dof outside the part will be -1.
// The stored vector spaces and indexers will reflect this,
// having a global/local size corresponding to only the
// dofs on the given part.
class DOFManager : public panzer::DOFManager {
public:
  // Initializes DOF manager
  // If part_name=="", assume defined over all blocks.
  DOFManager (const Teuchos::RCP<ConnManager>& conn_mgr,
              const Teuchos::RCP<const Teuchos_Comm>& comm,
              const std::string& part_name);

  void build ();

  // Returns (elem_LID,idof)->dof_lid numbering
  const DualView<const int**>& elem_dof_lids () const;

  const Teuchos::RCP<const GlobalLocalIndexer>& cell_indexer () const;

  const Teuchos::RCP<const GlobalLocalIndexer>& indexer () const;
  const Teuchos::RCP<const GlobalLocalIndexer>& ov_indexer () const;

  Teuchos::RCP<const Thyra_VectorSpace> cell_vs () const;

  Teuchos::RCP<const Thyra_VectorSpace> vs    () const;
  Teuchos::RCP<const Thyra_VectorSpace> ov_vs () const;

  using panzer::DOFManager::getElementGIDs;
  const std::vector<GO> & getElementGIDs (int ielem) const {
    return elementGIDs_[ielem];
  }
  using panzer::DOFManager::getGIDFieldOffsets;
  const std::vector<int> & getGIDFieldOffsets (int fieldNum) const {
    return this->getGIDFieldOffsets(elem_block_name(),fieldNum);
  }
  using panzer::DOFManager::getGIDFieldOffsetsKokkos;
  PHX::View<const int*> getGIDFieldOffsetsKokkos (int fieldNum) const {
    return this->getGIDFieldOffsetsKokkos(elem_block_name(),fieldNum);
  }
  using panzer::DOFManager::getGIDFieldOffsets_closure;
  const std::vector<int>&
  getGIDFieldOffsetsSubcell (int fieldNum, int subcell_dim, int subcell_pos) const;

  // Special case of the above, for subcell being the top or bottom side
  // NOTE: only for quad/hexa/wedge
  const std::vector<int>&
  getGIDFieldOffsetsSide (int fieldNum, int side) const;

  // If side!=orderedAsInSide, this version returns side offsets ordered
  // in such a way that off[i] on side=$side is directly above/below
  // off[i] on side=$orderAsInSide.
  // This makes sense ONLY IF $side and $orderAsInSide are both in {top,bot}
  const std::vector<int>&
  getGIDFieldOffsetsSide (int fieldNum, int side, int orderAsInSide) const;

  const std::string& part_name () const {
    return m_part_name;
  }

  const std::string& elem_block_name () const {
    return m_conn_mgr->elem_block_name();
  }

  shards::CellTopology get_topology () const {
    return m_conn_mgr->get_topology();
  }

  Teuchos::RCP<const ConnManager> getAlbanyConnManager() const {
    return m_conn_mgr;
  }

  Teuchos::RCP<const Teuchos_Comm> getAlbanyComm () const {
    return m_comm;
  }

private:
  void albanyBuildGlobalUnknowns ();

  // Create vector spaces and indexers, based on GIDs
  void buildVectorSpaces (const std::vector<GO>& owned,
                          const std::vector<GO>& ownedAndGhosted);

  // Restricts valid IDs to the entities belonging to the given part.
  // This method will maintain the dimensions of elem_dof_lids, but will
  //  - change all lids to -1 if the dof is not on the given part
  //  - recompute vector spaces to reflect the lower number of dofs
  //  - recompute indexers to reflect the change in vector spaces
  //  - recompute LIDs to reflect the restricted GIDs set.
  // Called during build phase.
  void restrict (const std::string& part_name);

  Teuchos::RCP<const Teuchos_Comm>          m_comm;

  Teuchos::RCP<const GlobalLocalIndexer>    m_cell_indexer;
  Teuchos::RCP<const GlobalLocalIndexer>    m_indexer;
  Teuchos::RCP<const GlobalLocalIndexer>    m_ov_indexer;
  DualView<const int**>                     m_elem_dof_lids;

  using vec4int = std::vector<std::vector<std::vector<std::vector<int>>>>;
  // m_subcell_closures[ifield][dim][ord] is the vector of the offsets of
  // field $ifield on the $ord-th subcell of dimension $dim. More precisely,
  // it's the closure of all offsets on all entities belonging to that subcell
  vec4int       m_subcell_closures;

  // Shortcut for location of top/bot sides in the cell list of sides.
  bool m_top_bot_well_defined = false;
  int m_top = -1;
  int m_bot = -1;

  int m_topo_dim;

  // Closure for top/bot side dofs, with dof ordering so that offset[i] lies directly
  // above/below of side_offset[i].
  // E.g., if map = m_side_closure_ordered_as_side, then
  //  - map[F][X][X] return the same as m_subcell_closures[F][side_dim][X]
  //  - t = map[F][top][bot], then dof at offset t[i] is directly above
  //    dof at offset m_subcell_closures[F][side_dim][bot]
  //  - b = map[F][bot][top], then dof at offset t[i] is directly below
  //    dof at offset m_subcell_closures[F][side_dim][top]
  // NOTE: this 
  vec4int       m_side_closure_orderd_as_side;

  Teuchos::RCP<const ConnManager>           m_conn_mgr;

  std::string m_part_name;

  bool m_built = false;
};

} // namespace Albany

#endif // ALBANY_DOF_MANAGER_HPP
