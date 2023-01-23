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
  albanyBuildGlobalUnknowns ();

  // 2. Create dual view from base class device view
  using dview = DualView<const int**>;
  typename dview::dev_t lids (this->getLIDs());
  m_elem_dof_lids = dview(lids);
  m_elem_dof_lids.sync_to_host();

  // 3. Build cell vector space/indexer
  auto cell_gids = m_conn_mgr->getElementsInBlock(elem_block_name());
  auto cell_vs = createVectorSpace (this->getComm(), cell_gids);
  m_cell_indexer = createGlobalLocalIndexer (cell_vs);

  // 4. Possibly restrict the DOFs list
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
      "Error! DOFManager::getGIDFieldOffsetsTopSide only available for Hexa/Wedge topologies.\n");
#endif
  // Shards has both Hexa and Wedge with top in the last side position
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
  // Shards has both Hexa and Wedge with bot in the second to last side position
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
  for (int dim=0; dim<=sub_dim; ++dim) {
    const int count = topo.getSubcellCount(dim);
    offsets[dim].resize(count);
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

  owned_.clear();
  ghosted_.clear();
  auto add_if_not_there = [](std::vector<GO>& v, const GO gid) {
    if (std::find(v.begin(),v.end(),gid)==v.end()) {
      v.push_back(gid);
    }
  };

  for (int ielem=0; ielem<lids_h.extent_int(0); ++ielem) {
    const auto& gids = getElementGIDs(ielem);
    for (int idof=0; idof<lids_h.extent_int(1); ++idof) {
      if (lids_h(ielem,idof)==-1) {
        continue;
      }

      // Check if this GID is owned or ghosted
      if (m_indexer->isLocallyOwnedElement(gids[idof])) {
        add_if_not_there(owned_,gids[idof]);
      } else {
        add_if_not_there(ghosted_,gids[idof]);
      }
    }
  }

  // Make sure ghosted come after owned
  // Re-build vector spaces
  buildVectorSpaces (owned_,ghosted_);

  // Correct the GIDs/LIDs stored, to reflect the smaller maps
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
                   const std::vector<GO>& ghosted)
{
#ifdef ALBANY_DEBUG
  // Sanity check
  auto owned_copy = owned;
  auto ghosted_copy = ghosted;
  std::sort(owned_copy.being(),owned_copy.end());
  std::sort(ghosted_copy.being(),ghosted_copy.end());
  auto it_owned = std::unique(owned_copy.begin(),owned_copy.end());
  auto it_ghosted = std::unique(ghosted_copy.begin(),ghosted_copy.end());
  TEUCHOS_TEST_FOR_EXCEPTION (it_owned!=owned_copy.end(),std::runtime_error,
      "[DOFManager::buildVectorSpaces]\n"
      "  Error! Repeated entries in the owned indices vector.\n");
  TEUCHOS_TEST_FOR_EXCEPTION (it_ghosted!=ghosted_copy.end(),std::runtime_error,
      "[DOFManager::buildVectorSpaces]\n"
      "  Error! Repeated entries in the ghosted indices vector.\n");
  for (size_t i=0,j=0; i<owned.size() && j<ghosted.size();) {
    TEUCHOS_TEST_FOR_EXCEPTION (owned_copy[i]==ghosted_copy[j], std::runtime_error,
      "[DOFManager::buildVectorSpaces]\n"
      "  Error! Owned and ghosted indices vector share some entries.\n");
    if (owned_copy[i]<ghosted_copy[j]) {
      ++i;
    } else {
      ++j;
    }
  }
#endif

  Teuchos::Array<GO> tmp;
  // First owned
  tmp = owned;
  auto vs = createVectorSpace (this->getComm(),tmp());

  // Then add ghosted, for the overlap map
  for (auto g : ghosted) {
    tmp.push_back(g);
  }
  auto ov_vs = createVectorSpace (this->getComm(),tmp());

  // 4. Build indexers
  m_indexer = createGlobalLocalIndexer (vs);
  m_ov_indexer = createGlobalLocalIndexer (ov_vs);
}

void DOFManager::
albanyBuildGlobalUnknowns ()
{
  // WARNING! This method is CUSTOM MADE for a single element block case
  // You can generalize to 2+ blocks, but must have same topology
  using namespace panzer;

  // Build aggregate and geometric field patterns
  std::vector<std::pair<FieldType,Teuchos::RCP<const FieldPattern>>> tmp;
  std::vector<std::tuple< int, FieldType, Teuchos::RCP<const FieldPattern> > > faConstruct;
  for (std::size_t i=0; i<fieldPatterns_.size(); ++i) {
    tmp.push_back(std::make_pair(fieldTypes_[i],fieldPatterns_[i]));
    faConstruct.emplace_back(i, fieldTypes_[fieldAIDOrder_[i]], fieldPatterns_[fieldAIDOrder_[i]]);
  }
  ga_fp_ = Teuchos::rcp(new GeometricAggFieldPattern(tmp));
  fa_fps_.push_back(Teuchos::rcp(new FieldAggPattern(faConstruct, ga_fp_)));

  // Build connectivity.
  // IMPORTANT! Do not use the GeometricAggFieldPattern, since you
  // *need* to count the same geo node multiple times if there are
  // multiple fields that need it.
  connMngr_->buildConnectivity(*fa_fps_.back());

  auto add_if_not_there = [](Teuchos::Array<GO>& v, const GO gid) {
    if (std::find(v.begin(),v.end(),gid)==v.end()) {
      v.push_back(gid);
    }
  };
  // Grab GIDs from connectivity
  const int numElems = m_conn_mgr->getElementsInBlock().size();
  Teuchos::Array<GO> ownedOrGhosted;
  elementGIDs_.resize(numElems);
  elementBlockGIDCount_.resize(1);
  for (int ielem=0; ielem<numElems; ++ielem) {
    const int  ndofs = m_conn_mgr->getConnectivitySize(ielem);
    const auto conn  = m_conn_mgr->getConnectivity(ielem);
    elementGIDs_[ielem].resize(ndofs);
    for (int idof=0; idof<ndofs; ++idof) {
      add_if_not_there(ownedOrGhosted,conn[idof]);
      elementGIDs_[ielem][idof] = conn[idof];
    }
    elementBlockGIDCount_[0] += ndofs;
  }

  // Create a unique vector space
  // NOTE: these are NOT the final spaces, since we do not know
  //       if the owned GIDs appear *before* the ghosted GIDs in
  //       the overlapped space. 
  auto ov_vs = createVectorSpace(getComm(),ownedOrGhosted);
  auto vs = createOneToOneVectorSpace(ov_vs);

  // Store owned/ghosted indices vectors
  auto vs_gids = getGlobalElements(vs);
  owned_ = vs_gids.toVector();
  // The arrays need to be sorted, in order to make the
  // calculation of the set difference linear in # ov gids.
  // That's why we need the copy vs_gids to start with.
  std::sort(vs_gids.begin(),vs_gids.end());
  std::sort(ownedOrGhosted.begin(),ownedOrGhosted.end());
  for (int i=0, iov=0; iov<ownedOrGhosted.size(); ++iov) {
    if (i>=vs_gids.size() or vs_gids[i]!=ownedOrGhosted[iov]) {
      // We ran out of owned gids or this gids does not appear
      // in the owned list. Either way, it's a ghosted gid
      ghosted_.push_back(ownedOrGhosted[iov]);
    } else {
      // GID is in the owned list, go to the next
      ++i;
    }
  }
  vs_gids = owned_;
  for (auto g : ghosted_) {
    vs_gids.push_back(g);
  }
  buildVectorSpaces(owned_, ghosted_);

  // Set local ids
  std::vector<std::vector<LO>> elem_lids (numElems);
  for (int ielem=0; ielem<numElems; ++ielem) {
    auto gids = elementGIDs_[ielem];
    elem_lids[ielem].reserve(gids.size());
    for (auto g : gids) {
      elem_lids[ielem].push_back(m_ov_indexer->getLocalElement(g));
    }
  }
  setLocalIds(elem_lids);

  // Set flag that some DOFManager getters check
  buildConnectivityRun_ = true;
}

} // namespace Albany
