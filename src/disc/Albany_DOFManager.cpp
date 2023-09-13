#include "Albany_DOFManager.hpp"

#include "Albany_ThyraUtils.hpp"
#include "Albany_CommUtils.hpp"

#include "Panzer_NodalFieldPattern.hpp"
#include "Panzer_ElemFieldPattern.hpp"

#include <unordered_set>

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

  const auto& topo = get_topology();

  m_topo_dim = topo.getDimension();
  if (topo.getBaseCellTopologyData()==shards::getCellTopologyData<shards::Line<2>>() or
      topo.getBaseCellTopologyData()==shards::getCellTopologyData<shards::Quadrilateral<4>>() or
      topo.getBaseCellTopologyData()==shards::getCellTopologyData<shards::Hexahedron<8>>()) {
    // Shards has both Hexa and Wedge with top side in last position,
    // and bot side in the second to last side position
    m_top_bot_well_defined = true;
    m_top = topo.getSideCount()-1;
    m_bot = topo.getSideCount()-2;
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
getGIDFieldOffsetsSubcell (int fieldNum,
                           int subcell_dim,
                           int subcell_pos) const
{
#ifdef ALBANY_DEBUG
  auto topo = get_topology();
  TEUCHOS_TEST_FOR_EXCEPTION (
      fieldNum<0 || fieldNum>getNumFields(), std::runtime_error,
      "[DOFManager::getGIDFieldOffsetsSubcell] Field index out of bounds.\n"
      "  - fieldNum: " << fieldNum << "\n"
      "  - num fields: " << getNumField() << "\n");
  TEUCHOS_TEST_FOR_EXCEPTION (
      subcell_dim<0 || subcell_dim>=m_topo_dim, std::runtime_error,
      "[DOFManager::getGIDFieldOffsetsSubcell] Subcell dimension out of bounds.\n"
      "  - subcell_dim: " << subcell_dim << "\n"
      "  - cell dimension: " << m_topo_dim << "\n");
  TEUCHOS_TEST_FOR_EXCEPTION (
      subcell_pos<0 || subcell_pos>=topo.getSubcellCount(subcell_dim), std::runtime_error,
      "[DOFManager::getGIDFieldOffsetsSubcell] Subcell dimension out of bounds.\n"
      "  - subcell_dim: " << subcell_dim << "\n"
      "  - cell dimension: " << topo.getSubcellCount(subcell_dimDimension) << "\n");
#endif

  return m_subcell_closures[fieldNum][subcell_dim][subcell_pos];
}

const std::vector<int>&
DOFManager::
getGIDFieldOffsetsSide (int fieldNum, int side) const
{
  return getGIDFieldOffsetsSubcell(fieldNum,m_topo_dim-1,side);
}

const std::vector<int>&
DOFManager::
getGIDFieldOffsetsSide (int fieldNum, int side, const int orderAsInSide) const
{
#ifdef ALBANY_DEBUG
  TEUCHOS_TEST_FOR_EXCEPTION (
      !m_top_bot_well_defined ||
      (side!=m_top && side!=m_bot) || (orderAsInSide!=m_top && orderAsInSide!=m_bot), std::logic_error,
      "Error! This version of getGIDFieldOffsetsSide only works for top/bot sides.\n"
      "  - side: " << side << "\n"
      "  - orderAsInSide: " << orderAsInSide << "\n"
      "  - top: " << m_top << "\n"
      "  - bot: " << m_bot << "\n");
  TEUCHOS_TEST_FOR_EXCEPTION(fieldNum<0 || fieldNum>=m_side_closure_orderd_as_side.size(),std::runtime_error,
      "Field id " << fieldNum << " out of bounds [0, " << m_side_closure_orderd_as_side.size() << ")\n");
  TEUCHOS_TEST_FOR_EXCEPTION(side<0 || side>=m_side_closure_orderd_as_side[fieldNum].size(),std::runtime_error,
      "Side id " << fieldNum  << " out of bounds [0, " << m_side_closure_orderd_as_side[fieldNum].size() << ")\n");
  TEUCHOS_TEST_FOR_EXCEPTION(orderAsInSide<0 || orderAsInSide>=m_side_closure_orderd_as_side[fieldNum][side].size(),std::runtime_error,
      "Order as side id " << orderAsInSide  << " out of bounds [0, " << m_side_closure_orderd_as_side[fieldNum][side].size() << ")\n");
#endif
  return m_side_closure_orderd_as_side[fieldNum][side][orderAsInSide];
}

void DOFManager::
restrict (const std::string& sub_part_name)
{
  if (elem_block_name()==sub_part_name) {
    // Sub part is the whole elem block, so same dofs, and same lids
    return;
  }

  // We need to discard dofs *not* on the given part
  const int num_elems = m_elem_dof_lids.dev().extent(0);
  const int num_elem_dofs = m_elem_dof_lids.dev().extent(1);

  const auto& topo = get_topology ();
  const int sub_dim = m_conn_mgr->part_dim(sub_part_name);

  // Precompute offsets
  std::vector<std::vector<std::vector<std::vector<int>>>> offsets(sub_dim+1);
  for (int dim=0; dim<=sub_dim; ++dim) {
    const int count = topo.getSubcellCount(dim);
    offsets[dim].resize(count);
    for (int pos=0; pos<count; ++pos) {
      for (int f=0; f<getNumFields(); ++f) {
        offsets[dim][pos].push_back(getGIDFieldOffsetsSubcell (f,dim,pos));
      }
    }
  }

  // Loop over elements and their sub cells, and set the correct LID for all
  // dofs on the sub-part
  std::vector<std::vector<LO>> new_elem_dof_lids(num_elems,std::vector<LO>(num_elem_dofs,-1));
  auto old_elem_dof_lids = m_elem_dof_lids.host();
  const auto& conn_part_mask = m_conn_mgr->getConnectivityMask(sub_part_name);
  for (int ielem=0; ielem<num_elems; ++ielem) {
    auto& new_dof_lids = new_elem_dof_lids[ielem];
    const int  ndofs = m_conn_mgr->getConnectivitySize(ielem);
    const int elem_start = m_conn_mgr->getConnectivityStart(ielem);
    for (int i=0; i<ndofs; ++i) {
      if (conn_part_mask[elem_start+i]==1) {
        new_dof_lids[i] = old_elem_dof_lids(ielem,i);
      }
    }
  }

  owned_.clear();
  ghosted_.clear();

  // We take a set as well, since std::find is on avg O(1) for unordered_set, vs O(N) in an array
  auto add_if_not_there = [](std::vector<GO>& v, std::unordered_set<GO>& s, const GO gid) {
    auto it_bool = s.insert(gid);
    if (it_bool.second) {
      v.push_back(gid);
    }
  };

  std::unordered_set<GO> owned_set, ghosted_set;
  for (int ielem=0; ielem<num_elems; ++ielem) {
    const auto& gids = getElementGIDs(ielem);
    const auto& dof_lids = new_elem_dof_lids[ielem];
    for (int idof=0; idof<num_elem_dofs; ++idof) {
      if (dof_lids[idof]==-1) {
        continue;
      }

      // Check if this GID is owned or ghosted
      if (m_indexer->isLocallyOwnedElement(gids[idof])) {
        add_if_not_there(owned_,owned_set,gids[idof]);
      } else {
        add_if_not_there(ghosted_,ghosted_set,gids[idof]);
      }
    }
  }

#ifndef NDEBUG
  // Check that, at least globally, the sub-part has *some* dofs
  int lcount = owned_.size();
  int gcount = 0;
  Teuchos::reduceAll(*m_comm,Teuchos::REDUCE_SUM,1,&lcount,&gcount);
  TEUCHOS_TEST_FOR_EXCEPTION (gcount==0, std::logic_error,
      "Error! Attempt to restrict a DOFManager to an empty sub-part.\n"
      " - dof mgr part name: " + part_name() + "\n"
      " - sub part name    : " + sub_part_name + "\n");
#endif

  // Make sure ghosted come after owned
  // Re-build vector spaces
  buildVectorSpaces (owned_,ghosted_);

  // Correct the GIDs/LIDs stored, to reflect the smaller maps
  // Update lids in base class as well, in case we use some base class method
  for (int ielem=0; ielem<num_elems; ++ielem) {
    const auto& gids = getElementGIDs(ielem);
    auto& dof_lids = new_elem_dof_lids[ielem];
    for (int idof=0; idof<num_elem_dofs; ++idof) {
      if (dof_lids[idof]==-1) {
        // Invalidate the GID as well, so that, if we accidentally use it,
        // it will hopefully trigger bad stuff
        elementGIDs_[ielem][idof] = -1;
      } else {
        // Correct the lid, reflecting the new vector spaces
        dof_lids[idof] = m_ov_indexer->getLocalElement(gids[idof]);
      }
    }
  }

  setLocalIds(new_elem_dof_lids);
  using dview = DualView<const int**>;
  m_elem_dof_lids = dview(this->getLIDs());
  m_elem_dof_lids.sync_to_host();
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

  // We take a set as well, since std::find is on avg O(1) for unordered_set, vs O(N) in an array
  auto add_if_not_there = [](std::vector<GO>& v, std::unordered_set<GO>& s, const GO gid) {
    auto it_bool = s.insert(gid);
    if (it_bool.second) {
      v.push_back(gid);
    }
  };
  // Grab GIDs from connectivity
  const int numElems = m_conn_mgr->getElementsInBlock().size();
  owned_.clear(); ghosted_.clear();
  std::unordered_set<GO> ownedSet;
  std::unordered_set<GO> ghostedSet;
  elementGIDs_.resize(numElems);
  elementBlockGIDCount_.resize(1);
  for (int ielem=0; ielem<numElems; ++ielem) {
    const int  ndofs = m_conn_mgr->getConnectivitySize(ielem);
    const auto conn  = m_conn_mgr->getConnectivity(ielem);
    const auto ownership  = m_conn_mgr->getOwnership(ielem);
    elementGIDs_[ielem].resize(ndofs);
    for (int idof=0; idof<ndofs; ++idof) {
      if(ownership[idof]==Owned)
        add_if_not_there(owned_,ownedSet,conn[idof]);
      else if (ownership[idof]==Ghosted)
        add_if_not_there(ghosted_,ghostedSet,conn[idof]);
      else
        TEUCHOS_TEST_FOR_EXCEPTION (false, std::logic_error,
            "Error! Found a dof with Unset ownership.\n"
            "  - rank: " << m_comm->getRank() << "\n"
            "  - elem block: " << elem_block_name() << "\n"
            "  - ielem: " << ielem << "\n"
            "  - idof: " << idof << "\n");
      elementGIDs_[ielem][idof] = conn[idof];
    }
    elementBlockGIDCount_[0] += ndofs;
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

  // Build subcell offsets in a way that matches shards ordering
  // Do this only if topology is not a Particle
  constexpr auto nodeDim = 0;
  constexpr auto edgeDim = 1;
  constexpr auto faceDim = 2;
  if (get_topology().getDimension()>nodeDim) {
    const auto& topo = get_topology();
    m_subcell_closures.resize(getNumFields());

    // Pre-build also the special closure for the top side, with dof ordered
    // so that top_closure[i] sits on top of bot_closure[i]
    m_side_closure_orderd_as_side.resize(getNumFields());

    // Order in which to read top (resp, bot) offsets to match bot (resp, top)
    std::vector<int> permutation;
    if (std::string(topo.getBaseName())=="Hexahedron_8") {
      permutation = {0, 3, 2, 1};
    } else {
      permutation = {0, 2, 1};
    }

    for (int f=0; f<getNumFields(); ++f) {
      const auto& name = getFieldString(f);
      const auto& fp = getFieldPattern(name);
      const auto& field_offsets = getGIDFieldOffsets(f);

      // Helper lambda, to reduce code duplication. Adds indices from
      // $ord-th subcell of dim $dim to the in/out closure vector.
      auto add_indices = [&] (const int dim, const int ord,
                              std::vector<int>& closure)
      {
        const auto& indices = fp->getSubcellIndices(dim,ord);
        for (auto i : indices) {
          closure.push_back(field_offsets[i]);
        }
      };

      m_subcell_closures[f].resize(m_topo_dim);
      auto& f_closures = m_subcell_closures[f];

      // Nodes: simply, add offsets at each node
      const int nodeCount = topo.getNodeCount();
      f_closures[nodeDim].resize(nodeCount);
      for (int inode=0; inode<nodeCount; ++inode) {
        add_indices(nodeDim,inode,f_closures[nodeDim][inode]);
      }

      // 1D geometries don't have edges/faces
      if (m_topo_dim==edgeDim) continue;

      // Edges: add offsets on nodes first, then edges themselves
      const int edgeCount = topo.getEdgeCount();
      f_closures[edgeDim].resize(edgeCount);
      for (int iedge=0; iedge<edgeCount; ++iedge) {
        // Add edge nodes
        const int edgeNodeCount = topo.getNodeCount(edgeDim,iedge);
        for (int inode=0; inode<edgeNodeCount; ++inode) {
          const int cellNode = topo.getNodeMap(edgeDim,iedge,inode);
          add_indices(nodeDim,cellNode,f_closures[edgeDim][iedge]);
        }

        // Add edges
        add_indices(edgeDim,iedge,f_closures[edgeDim][iedge]);
      }

      // 2D geometries don't have faces
      if (m_topo_dim==faceDim) continue;

      // Faces: add offsets on nodes first, then edges, then faces themselves
      const int faceCount = topo.getFaceCount();
      f_closures[faceDim].resize(faceCount);
      for (int iface=0; iface<faceCount; ++iface) {
        // Add face nodes
        const int faceNodeCount = topo.getNodeCount(faceDim,iface);
        for (int inode=0; inode<faceNodeCount; ++inode) {
          const int cellNode = topo.getNodeMap(faceDim,iface,inode);
          add_indices(nodeDim,cellNode,f_closures[faceDim][iface]);
        }

        // Add face edges
        auto face_topo = shards::CellTopology(topo.getCellTopologyData(faceDim,iface));
        const int faceEdges = face_topo.getSideCount();
        for (int iedge=0; iedge<faceEdges; ++iedge) {
          const int cellEdge = mapCellFaceEdge(topo.getCellTopologyData(),iface,iedge);
          add_indices(edgeDim,cellEdge,f_closures[faceDim][iface]);
        }

        // Add face
        add_indices(faceDim,iface,f_closures[faceDim][iface]);
      }

      // Shards has both Hexa and Wedge with top in the last side position
      if (m_top_bot_well_defined) {
        auto tmp_top = f_closures[faceDim][m_top];
        auto tmp_bot = f_closures[faceDim][m_bot];
        auto& side_as_side = m_side_closure_orderd_as_side[f];
        side_as_side.resize(faceCount);
        for (auto& v : side_as_side) {
          v.resize(faceCount);
        }

        // These involve no permutation
        side_as_side[m_top][m_top] = tmp_top;
        side_as_side[m_bot][m_bot] = tmp_bot;
        side_as_side[m_top][m_bot].reserve(tmp_top.size());
        side_as_side[m_bot][m_top].reserve(tmp_top.size());
        for (auto p : permutation) {
          side_as_side[m_bot][m_top].push_back(tmp_bot[p]);
          side_as_side[m_top][m_bot].push_back(tmp_top[p]);
        }
      }
    }
  }
}

} // namespace Albany
