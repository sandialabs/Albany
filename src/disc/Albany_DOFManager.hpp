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

class DOFManager : public panzer::DOFManager {
public:
  // Initializes DOF manager
  DOFManager (const Teuchos::RCP<ConnManager>& conn_mgr,
              const Teuchos::RCP<const Teuchos_Comm>& comm);

  void build ();

  // Returns (elem_LID,idof)->dof_lid numbering
  const DualView<const int**>& elem_dof_lids () const;

  // Returns (elem_LID,idof)->dof_lid numbering, but with elements being
  // the cells of the input dof_mgr. If a certain (elem_LID,idof) pair
  // refers to an entity that does not belong to this DOFManager part,
  // we set dof_lid=-1.
  DualView<const int**> elem_dof_lids (const Teuchos::RCP<const DOFManager>& dof_mgr) const;

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
    return this->getGIDFieldOffsets(part_name(),fieldNum);
  }
  using panzer::DOFManager::getGIDFieldOffsetsKokkos;
  PHX::View<const int*> getGIDFieldOffsetsKokkos (int fieldNum) const {
    return this->getGIDFieldOffsetsKokkos(part_name(),fieldNum);
  }
  using panzer::DOFManager::getGIDFieldOffsets_closure;
  const std::vector<int>&
  getGIDFieldOffsets_subcell (int fieldNum, int subcell_dim, int subcell_pos) const;

  const std::string& part_name () const {
    return m_conn_mgr->part_name();
  }

  shards::CellTopology get_topology () const {
    return m_conn_mgr->get_topology();
  }

  Teuchos::RCP<const ConnManager> getAlbanyConnManager() const {
    return m_conn_mgr;
  }

private:
  Teuchos::RCP<const Teuchos_Comm>          m_comm;

  Teuchos::RCP<const GlobalLocalIndexer>    m_cell_indexer;
  Teuchos::RCP<const GlobalLocalIndexer>    m_indexer;
  Teuchos::RCP<const GlobalLocalIndexer>    m_ov_indexer;
  DualView<const int**>                     m_elem_dof_lids;

  Teuchos::RCP<const ConnManager>           m_conn_mgr;

  bool m_built = false;
};

} // namespace Albany

#endif // ALBANY_DOF_MANAGER_HPP
