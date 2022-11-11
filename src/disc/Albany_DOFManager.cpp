#include "Albany_DOFManager.hpp"

#include "Albany_ThyraUtils.hpp"
#include "Albany_CommUtils.hpp"

#include "Panzer_NodalFieldPattern.hpp"
#include "Panzer_ElemFieldPattern.hpp"

namespace Albany {

DOFManager::
DOFManager (const Teuchos::RCP<ConnManager>& conn_mgr,
            const Teuchos::RCP<const Teuchos_Comm>& comm)
 : m_comm (comm)
 , m_conn_mgr (conn_mgr)
{
  // Check non-null pointers
  TEUCHOS_TEST_FOR_EXCEPTION (comm.is_null(), std::runtime_error,
      "Error! Invalid Teucho_Comm pointer.\n");
  TEUCHOS_TEST_FOR_EXCEPTION (conn_mgr.is_null(), std::runtime_error,
      "Error! Invalid ConnManager pointer.\n");

  setConnManager (conn_mgr,getMpiCommFromTeuchosComm(comm));
}

void DOFManager::build ()
{
  TEUCHOS_TEST_FOR_EXCEPTION (m_built, std::runtime_error,
      "Error! DOFManager::build was already called.\n");

  // 1. Let base class build the GIDs
  this->buildGlobalUnknowns ();

  // 2. Create dual view from base class device view
  using dview = DualView<const int**>;
  typename dview::dev_t lids (this->getLIDs());
  m_elem_dof_lids = dview(lids);
  m_elem_dof_lids.sync_to_host();

  // 3. Build vector spaces
  std::vector<panzer::GlobalOrdinal> panzer_gids;
  Teuchos::ArrayView<const GO> gids;

  this->getOwnedIndices(panzer_gids);
  gids = decltype(gids)(reinterpret_cast<const GO*>(panzer_gids.data()),panzer_gids.size());
  auto vs = createVectorSpace (this->getComm(),gids);

  this->getOwnedAndGhostedIndices(panzer_gids);
  gids = decltype(gids)(reinterpret_cast<const GO*>(panzer_gids.data()),panzer_gids.size());
  auto ov_vs = createVectorSpace (this->getComm(),gids);

  auto cell_gids = m_conn_mgr->getElementsInBlock(part_name());
  auto cell_vs = createVectorSpace (this->getComm(), cell_gids);

  // 4. Build indexers
  m_indexer = createGlobalLocalIndexer (vs);
  m_ov_indexer = createGlobalLocalIndexer (ov_vs);
  m_cell_indexer = createGlobalLocalIndexer (cell_vs);

  // Done
  m_built = true;
}

const DualView<const int**>&
DOFManager::elem_dof_lids () const
{
  TEUCHOS_TEST_FOR_EXCEPTION (not m_built, std::runtime_error,
      "Error! DOFManager::build was not yet called.\n");

  return m_elem_dof_lids;
}

DualView<const int**>
DOFManager::elem_dof_lids (const Teuchos::RCP<const DOFManager>& dof_mgr) const
{
  TEUCHOS_TEST_FOR_EXCEPTION (not m_built, std::runtime_error,
      "Error! DOFManager::build was not yet called.\n");

  if (dof_mgr->part_name()==part_name() &&
      dof_mgr->get_topology()==get_topology()) {

    // Same part name, same cell topology, so the lids are the same
    return m_elem_dof_lids;
  }

  const auto dof_lids = dof_mgr->elem_dof_lids().host();
  const int nelems = dof_lids.extent(0);
  const int nldofs = dof_lids.extent(1);
  const auto indexer_in = dof_mgr->ov_indexer();
  const auto indexer = ov_indexer();
  
  DualView<int**> elem_dof_lids ("",nelems,nldofs);
  for (int ielem=0; ielem<nelems; ++ielem) {
    for (int idof=0; idof<nldofs; ++idof) {
      const auto gid = indexer_in->getGlobalElement(dof_lids(ielem,idof));
      const auto lid = indexer->getLocalElement(gid);
      elem_dof_lids.host()(ielem,idof) = lid;
    }
  }
  elem_dof_lids.sync_to_dev();
  return elem_dof_lids;
}

const Teuchos::RCP<const GlobalLocalIndexer>&
DOFManager::cell_indexer () const
{
  TEUCHOS_TEST_FOR_EXCEPTION (not m_built, std::runtime_error,
      "Error! DOFManager::build was not yet called.\n");

  return m_cell_indexer;
}

const Teuchos::RCP<const GlobalLocalIndexer>&
DOFManager::indexer () const
{
  TEUCHOS_TEST_FOR_EXCEPTION (not m_built, std::runtime_error,
      "Error! DOFManager::build was not yet called.\n");

  return m_indexer;
}

const Teuchos::RCP<const GlobalLocalIndexer>&
DOFManager::ov_indexer () const
{
  TEUCHOS_TEST_FOR_EXCEPTION (not m_built, std::runtime_error,
      "Error! DOFManager::build was not yet called.\n");

  return m_ov_indexer;
}

Teuchos::RCP<const Thyra_VectorSpace>
DOFManager::vs () const
{
  TEUCHOS_TEST_FOR_EXCEPTION (not m_built, std::runtime_error,
      "Error! DOFManager::build was not yet called.\n");

  return m_indexer->getVectorSpace();
}

Teuchos::RCP<const Thyra_VectorSpace>
DOFManager::ov_vs () const
{
  TEUCHOS_TEST_FOR_EXCEPTION (not m_built, std::runtime_error,
      "Error! DOFManager::build was not yet called.\n");

  return m_ov_indexer->getVectorSpace();
}

const std::vector<int>&
DOFManager::
getGIDFieldOffsets_subcell (int fieldNum,
                            int subcell_dim,
                            int subcell_pos) const
{
  const auto& indices_pair =
    this->getGIDFieldOffsets_closure(part_name(),fieldNum, subcell_dim, subcell_pos);

  return indices_pair.first;
}

} // namespace Albany
