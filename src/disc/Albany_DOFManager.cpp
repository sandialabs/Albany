#include "Albany_DOFManager.hpp"

#include "Albany_ThyraUtils.hpp"
#include "Albany_CommUtils.hpp"

#include "Panzer_NodalFieldPattern.hpp"
#include "Panzer_ElemFieldPattern.hpp"

namespace Albany {

DOFManager::
DOFManager (const Teuchos::RCP<ConnManager>& conn_mgr,
            const Teuchos::RCP<const Teuchos_Comm>& comm,
            const std::string& part_name)
 : m_comm (comm)
 , m_conn_mgr (conn_mgr)
{
  // Check non-null pointers
  TEUCHOS_TEST_FOR_EXCEPTION (comm.is_null(), std::runtime_error,
      "Error! Invalid Teucho_Comm pointer.\n");
  TEUCHOS_TEST_FOR_EXCEPTION (conn_mgr.is_null(), std::runtime_error,
      "Error! Invalid ConnManager pointer.\n");

  setConnManager (conn_mgr,getMpiCommFromTeuchosComm(comm));

  if (part_name=="") {
    m_part_name = elem_block_name();
  } else {
    m_part_name = part_name;
  }
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

  // 3. Build cell vector space/indexer
  auto cell_gids = m_conn_mgr->getElementsInBlock(elem_block_name());
  auto cell_vs = createVectorSpace (this->getComm(), cell_gids);
  m_cell_indexer = createGlobalLocalIndexer (cell_vs);

  // 4. Build dof vector spaces and indexers
  std::vector<GO> owned, ownedAndGhosted;
  this->getOwnedIndices(owned);
  this->getOwnedAndGhostedIndices(ownedAndGhosted);
  buildVectorSpaces (owned,ownedAndGhosted);

  // 5. Possibly restrict the DOFs list
  restrict (part_name());

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
DOFManager::cell_vs () const
{
  TEUCHOS_TEST_FOR_EXCEPTION (not m_built, std::runtime_error,
      "Error! DOFManager::build was not yet called.\n");

  return m_cell_indexer->getVectorSpace();
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
    this->getGIDFieldOffsets_closure(elem_block_name(),fieldNum, subcell_dim, subcell_pos);

  return indices_pair.first;
}

const std::vector<int>&
DOFManager::
getGIDFieldOffsetsSide (int fieldNum, int side) const
{
  const auto& topo = get_topology();

  return getGIDFieldOffsets_subcell(fieldNum,topo.getDimension()-1,side);
}


const std::vector<int>&
DOFManager::
getGIDFieldOffsetsTopSide (int fieldNum) const
{
  const auto& topo = get_topology();

#ifdef ALBANY_DEBUG
  constexpr auto  hexa  = shards::getCellTopologyData<shards::Hexahedron<8>>();
  constexpr auto  wedge = shards::getCellTopologyData<shards::Wedge<6>>();
  TEUCHOS_TEST_FOR_EXCEPTION (topo!=quad && topo!=hexa && topo!=wedge, std::runtime_error,
      "Error! DOFManager::getGIDFieldOffsetsBotSide only available for Hexa/Wedge topologies.\n");
#endif
  // Shards has both Hexa and Wedge with bot and top in the last two side positions
  return getGIDFieldOffsetsSide(fieldNum,topo.getSideCount()-1);
}

const std::vector<int>&
DOFManager::
getGIDFieldOffsetsBotSide (int fieldNum) const
{
  const auto& topo = get_topology();

#ifdef ALBANY_DEBUG
  constexpr auto  hexa  = shards::getCellTopologyData<shards::Hexahedron<8>>();
  constexpr auto  wedge = shards::getCellTopologyData<shards::Wedge<6>>();
  TEUCHOS_TEST_FOR_EXCEPTION (topo!=quad && topo!=hexa && topo!=wedge, std::runtime_error,
      "Error! DOFManager::getGIDFieldOffsetsBotSide only available for Hexa/Wedge topologies.\n");
#endif
  // Shards has both Hexa and Wedge with bot and top in the last two side positions
  return getGIDFieldOffsetsSide(fieldNum,topo.getSideCount()-2);
}

void DOFManager::
restrict (const std::string& sub_part_name)
{
  if (elem_block_name()==sub_part_name) {
    // Sub part is the whole elem block, so same dofs, and same lids
    return;
  }

  TEUCHOS_TEST_FOR_EXCEPTION (not m_conn_mgr->contains(sub_part_name), std::runtime_error,
      "Error! Input sub-part name not contained in the ConnManager parts.\n");

  // We need to discard dofs *not* on the given part
  DualView<int**> elem_dof_lids("",m_elem_dof_lids.host().extent(0),m_elem_dof_lids.host().extent(1));
  auto& h_elem_dof_lids = elem_dof_lids.host();
  Kokkos::deep_copy(h_elem_dof_lids,-1);
  const auto& topo = get_topology ();
  const int num_elems = m_conn_mgr->getElementsInBlock().size();
  const int sub_dim = m_conn_mgr->part_dim(sub_part_name);

  // Precompute offsets
  std::vector<std::vector<std::vector<std::vector<int>>>> offsets(sub_dim+1);
  std::cout << "precomputing offsets...\n";
  std::cout << "cell topo: " << topo.getName() << "\n";
  std::cout << "sub_part_name: " << sub_part_name << "\n";
  std::cout << "sub_part_dim: " << sub_dim << "\n";
  for (int dim=0; dim<=sub_dim; ++dim) {
    const int count = topo.getSubcellCount(dim);
    offsets[dim].resize(count);
    std::cout << "  subdim " << dim << ", count=" << count << "\n";
    for (int pos=0; pos<count; ++pos) {
      for (int f=0; f<getNumFields(); ++f) {
        offsets[dim][pos].push_back(getGIDFieldOffsets_subcell (f,dim,pos));
      }
    }
  }

  // Loop over elements and their sub cells, and set the correct LID for all
  // dofs on the sub-part
  for (int ielem=0; ielem<num_elems; ++ielem) {
    auto dof_lids = Kokkos::subview(h_elem_dof_lids,ielem,Kokkos::ALL());
    for (int dim=0; dim<=sub_dim; ++dim) {
      const int count = topo.getSubcellCount(dim);
      for (int pos=0; pos<count; ++pos) {
        if (m_conn_mgr->belongs(sub_part_name,ielem,dim,pos)) {
          for (int f=0; f<getNumFields(); ++f) {
            for (auto o : offsets[dim][pos][f]) {
              dof_lids(o) = m_elem_dof_lids.host()(ielem,o);
            }
          }
        }
      }
    }
  }
  elem_dof_lids.sync_to_dev();

  auto lids_h = elem_dof_lids.host();

  std::vector<GO> owned, owned_or_ghosted, ghosted;
  for (int ielem=0; ielem<lids_h.extent_int(0); ++ielem) {
    const auto& gids = getElementGIDs(ielem);
    for (int idof=0; idof<lids_h.extent_int(1); ++idof) {
      if (lids_h(ielem,idof)==-1) {
        continue;
      }

      // Check if this GID is owned or ghosted
      if (m_indexer->isLocallyOwnedElement(gids[idof])) {
        owned.push_back(gids[idof]);
        ghosted.push_back(gids[idof]);
      } else {
        ghosted.push_back(gids[idof]);
      }
    }
  }

  // Make sure ghosted come after owned
  owned_or_ghosted = owned;
  for (auto g : ghosted)
    owned_or_ghosted.push_back(g);

  // Re-build vector spaces
  buildVectorSpaces (owned,owned_or_ghosted);

  // Correct the GIDs/LIDs stored in m_elem_dof_lids, to reflect the smaller maps
  for (int ielem=0; ielem<lids_h.extent_int(0); ++ielem) {
    const auto& gids = getElementGIDs(ielem);
    for (int idof=0; idof<lids_h.extent_int(1); ++idof) {
      if (lids_h(ielem,idof)==-1) {
        // Invalidate the GID
        elementGIDs_[ielem][idof] = -1;
      } else {
        // Correct the lid
        lids_h(ielem,idof) = m_ov_indexer->getLocalElement(gids[idof]);
      }
    }
  }

  // Update lids
  m_elem_dof_lids = elem_dof_lids;
}

void DOFManager::
buildVectorSpaces (const std::vector<GO>& owned,
                   const std::vector<GO>& ownedAndGhosted)
{
  // Sanity check
  TEUCHOS_TEST_FOR_EXCEPTION (owned.size()>ownedAndGhosted.size(), std::logic_error,
      "Error! Owned GID list is larger than owned+ghosted GID list.\n");

  Teuchos::ArrayView<const GO> gids;
  gids = decltype(gids)(owned.data(),owned.size());
  auto vs = createVectorSpace (this->getComm(),gids);

  gids = decltype(gids)(ownedAndGhosted.data(),ownedAndGhosted.size());
  auto ov_vs = createVectorSpace (this->getComm(),gids);

  // 4. Build indexers
  m_indexer = createGlobalLocalIndexer (vs);
  m_ov_indexer = createGlobalLocalIndexer (ov_vs);
}

} // namespace Albany
