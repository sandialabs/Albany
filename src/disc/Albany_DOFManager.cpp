#include "Albany_DOFManager.hpp"

#include "Albany_ThyraUtils.hpp"
#include "Albany_CommUtils.hpp"

#include "Panzer_NodalFieldPattern.hpp"
#include "Panzer_ElemFieldPattern.hpp"

namespace Albany {

Teuchos::RCP<const panzer::FieldPattern>
createFieldPattern (const FE_Type fe_type,
                    const shards::CellTopology& cell_topo)
{
  Teuchos::RCP<const panzer::FieldPattern> fp;
  switch (fe_type) {
    case FE_Type::P1:
      fp = Teuchos::rcp(new panzer::NodalFieldPattern(cell_topo));
      break;
    case FE_Type::P0:
      fp = Teuchos::rcp(new panzer::ElemFieldPattern(cell_topo));
      break;
    default:
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error,
          "Error! Unsupported FE_Type.\n");
  }
  return fp;
}

std::string e2str (const FE_Type fe_type) {
  std::string s;
  switch (fe_type) {
    case FE_Type::P1: s = "P1"; break;
    case FE_Type::P0: s = "P0"; break;
    default:
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error,
          "Error! Unsupported FE_Type.\n");
  }
  return s;
}

DOFManager::
DOFManager (const Teuchos::RCP<panzer::ConnManager>& conn_mgr,
            const Teuchos::RCP<const Teuchos_Comm>& comm)
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
  m_elem_lids = dview(lids);
  m_elem_lids.sync_to_host();

  // 3. Build vector spaces
  std::vector<panzer::GlobalOrdinal> panzer_gids;
  Teuchos::ArrayView<const GO> gids;

  this->getOwnedIndices(panzer_gids);
  gids = decltype(gids)(reinterpret_cast<const GO*>(panzer_gids.data()),panzer_gids.size());
  auto vs = createVectorSpace (this->getComm(),gids);

  this->getOwnedAndGhostedIndices(panzer_gids);
  gids = decltype(gids)(reinterpret_cast<const GO*>(panzer_gids.data()),panzer_gids.size());
  auto ov_vs = createVectorSpace (this->getComm(),gids);

  // 4. Build indexers
  m_indexer = createGlobalLocalIndexer (vs);
  m_ov_indexer = createGlobalLocalIndexer (ov_vs);

  // Done
  m_built = true;
}

const DualView<const int**>&
DOFManager::elem_lids () const
{
  TEUCHOS_TEST_FOR_EXCEPTION (not m_built, std::runtime_error,
      "Error! DOFManager::build was not yet called.\n");

  return m_elem_lids;
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

} // namespace Albany
