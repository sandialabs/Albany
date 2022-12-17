//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_DOF_MANAGER_HPP
#define ALBANY_DOF_MANAGER_HPP

#include "Albany_GlobalLocalIndexer.hpp"
#include "Albany_ThyraTypes.hpp"
#include "Albany_KokkosTypes.hpp"

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
  getGIDFieldOffsets_subcell (int fieldNum, int subcell_dim, int subcell_pos) const;

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

  Teuchos::RCP<const ConnManager>           m_conn_mgr;

  std::string m_part_name;

  bool m_built = false;
};

} // namespace Albany

#endif // ALBANY_DOF_MANAGER_HPP
