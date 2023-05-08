//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "STKConnManager.hpp"
#include "Albany_config.h"

#include "Panzer_FieldPattern.hpp"

#include <stk_util/parallel/ParallelReduce.hpp>
#include <stk_mesh/base/GetEntities.hpp>

namespace Albany {

STKConnManager::
STKConnManager(const Teuchos::RCP<const stk::mesh::MetaData>& metaData,
               const Teuchos::RCP<const stk::mesh::BulkData>& bulkData,
               const std::vector<std::string>& elem_block_names)
{
  // Sanity check
  TEUCHOS_TEST_FOR_EXCEPTION (metaData.is_null(), std::runtime_error,
      "Error! Input meta data pointer is null.\n");
  TEUCHOS_TEST_FOR_EXCEPTION (bulkData.is_null(), std::runtime_error,
      "Error! Input bulk data pointer is null.\n");
  TEUCHOS_TEST_FOR_EXCEPTION (elem_block_names.size()==0, std::runtime_error,
      "Error! Input elem_block names vector is empty.\n");

  m_bulkData = bulkData;
  m_metaData = metaData;

  // Add elem_blocks, and check that 1) compatible dimensions and 2) no intersection
  constexpr auto INVALID = stk::topology::rank_t::INVALID_RANK;
  for (const auto& ebn : elem_block_names) {
    auto elem_block = m_metaData->get_part(ebn);
    TEUCHOS_TEST_FOR_EXCEPTION (elem_block==nullptr, std::runtime_error,
        "[STKConnManager] Error! Elem block '" + ebn + "' not found in the mesh.\n");

    TEUCHOS_TEST_FOR_EXCEPTION (m_elem_blocks_topo.rank()!=INVALID &&
                                elem_block->topology()!=m_elem_blocks_topo, std::logic_error,
        "[STKConnManager] Error! Input elem_blocks do not have the same topology.\n"
        "  - current topo  : " + m_elem_blocks_topo.name() + "\n";
        "  - new elem_block name : " + ebn + "\n"
        "  - new elem_block topo : " + elem_block->topology().name() + "\n"
    );

    for (const auto& p : m_elem_blocks) {
      // NOTE: cannot use stk::mesh::intersect, since that function returns true even if the
      //       intersection is on entities of low dimension. We don't want, e.g., to think
      //       that two element blocks intersect only b/c they have some nodes/sides in common.
      //       We are concerned about overlap of the primary entities.
      stk::mesh::Selector selector(*elem_block);
      selector &= *p.second;

      TEUCHOS_TEST_FOR_EXCEPTION (stk::mesh::count_entities(*m_bulkData,elem_block->primary_entity_rank(),selector)>0, std::logic_error,
          "[STKConnManager] Error! Two input elem_blocks intersect.\n"
          "  - first elem_block name : " + p.second->name() + "\n"
          "  - second elem_block name: " + elem_block->name() + "\n");
    }

    m_elem_blocks_topo = elem_block->topology();

    m_elem_blocks[ebn] = elem_block;
  }

  // Init members of base class
  m_elem_blocks_names = elem_block_names;

  buildMaxEntityIds();

  // get element info from STK_Interface
  // object and build a local element mapping.
  buildLocalElementMapping();
}

Teuchos::RCP<panzer::ConnManager>
STKConnManager::noConnectivityClone() const
{
  std::vector<std::string> elem_blocks_names;
  for (const auto& it : m_elem_blocks) {
    elem_blocks_names.push_back(it.first);
  }
  return Teuchos::rcp(new STKConnManager(m_metaData,m_bulkData,elem_blocks_names));
}

void STKConnManager::clearLocalElementMapping()
{
  m_elements.clear();
  m_elementBlocks.clear();
  m_elmtLidToConn.clear();
  m_connSize.clear();
  m_connectivity.clear();
}

std::vector<GO>
STKConnManager::getElementsInBlock (const std::string& /* blockId */) const
{
  std::vector<GO> gids;
  gids.reserve(m_elements.size());
  for (const auto& e : m_elements) {
    gids.push_back (m_bulkData->identifier(e)-1);
  }
  return gids;
}

void STKConnManager::buildLocalElementMapping()
{
  // Start from scratch
  clearLocalElementMapping(); // forget the past

  // Get blocks names
  std::vector<std::string> blockIds;
  getElementBlockIds(blockIds);

  // Loop over element blocks, and gather elements for each block
  for (const auto& blockId : blockIds) {
    // 1. Grab elements on this block
    std::vector<stk::mesh::Entity> blockElmts;
    getMyElements(blockId,blockElmts);

    // 2. Concatenate them into element LID lookup table
    m_elements.insert(m_elements.end(),blockElmts.begin(),blockElmts.end());

    // 3. Assign local ids
    buildLocalElementIDs(blockElmts);

    // 4. Build blockName->elemLIDs map
    auto& blockElems = m_elementBlocks[blockId];
    blockElems.reserve(blockElmts.size());
    for (const auto& elem : blockElmts) {
      blockElems.push_back (elementLocalId(elem));
    }
  }

#ifdef ALBANY_DEBUG
  // this expensive operation checks ordering of local IDs
  auto cmpLids = [&](stk::mesh::Entity a, stk::mesh::Entity b)->bool{
     return elementLocalId(a) < elementLocalId(b);
  };
  auto copy = m_elements;
  std::sort(copy.begin(), copy.end(), cmpLids);
  TEUCHOS_TEST_FOR_EXCEPTION (copy!=m_elements, std::runtime_error,
      "Error! Elements were supposed to be already sorted.\n"
      "       Something is off, please, contact developers.\n");
#endif

  // Pre allocate space for internal connectivity offsets/sizes
  m_elmtLidToConn.resize(m_elements.size(),0);
  m_connSize.resize(m_elements.size(),0);
}

void STKConnManager::
buildOffsetsAndIdCounts(
    const panzer::FieldPattern & fp,
    LO & nodeIdCnt, LO & edgeIdCnt,
    LO & faceIdCnt, LO & cellIdCnt,
    GO & nodeOffset, GO & edgeOffset,
    GO & faceOffset, GO & cellOffset) const
{
  // get the global counts for all the nodes, faces, edges and cells
  GO maxNodeId = getMaxEntityId(stk::topology::NODE_RANK);
  GO maxEdgeId = getMaxEntityId(stk::topology::EDGE_RANK);
  GO maxFaceId = getMaxEntityId(stk::topology::FACE_RANK);

  // compute ID counts for each sub cell type
  int patternDim = fp.getDimension();
  switch(patternDim) {
    case 3:
      faceIdCnt = fp.getSubcellIndices(2,0).size();
      // Intentional fall-through.
    case 2:
      edgeIdCnt = fp.getSubcellIndices(1,0).size();
      // Intentional fall-through.
    case 1:
      nodeIdCnt = fp.getSubcellIndices(0,0).size();
      // Intentional fall-through.
    case 0:
      cellIdCnt = fp.getSubcellIndices(patternDim,0).size();
      break;
    default:
       TEUCHOS_ASSERT(false);
  };

  // compute offsets for each sub cell type
  nodeOffset = 0;
  edgeOffset = nodeOffset+(maxNodeId+1)*nodeIdCnt;
  faceOffset = edgeOffset+(maxEdgeId+1)*edgeIdCnt;
  cellOffset = faceOffset+(maxFaceId+1)*faceIdCnt;

  // sanity check
  TEUCHOS_ASSERT(nodeOffset <= edgeOffset
              && edgeOffset <= faceOffset
              && faceOffset <= cellOffset);
}

LO STKConnManager::
addSubcellConnectivities (const stk::mesh::Entity element,
                          const stk::mesh::EntityRank subcellRank,
                          const LO idCnt,
                          const GO offset)
{
  if(idCnt<=0)
     return 0;

  // loop over all relations of specified type
  LO numIds = 0;
  const int num_rels = m_bulkData->num_connectivity(element, subcellRank);
  stk::mesh::Entity const* relations = m_bulkData->begin(element, subcellRank);
  for(int sc=0; sc<num_rels; ++sc) {
    stk::mesh::Entity subcell = relations[sc];
    const auto subcell_id = m_bulkData->identifier(subcell);

    auto owned = m_bulkData->bucket(subcell).owned() ? Owned : Ghosted;
    // add connectivities: adjust for STK indexing craziness
    for (LO i=0; i<idCnt; ++i) {
      m_connectivity.push_back(offset+idCnt*(subcell_id-1)+i);
      m_ownership.push_back(owned);
    }
    numIds += idCnt;
  }
  return numIds;
}

void STKConnManager::buildConnectivity(const panzer::FieldPattern & fp)
{
#ifdef HAVE_EXTRA_TIMERS
  using Teuchos::TimeMonitor;
  RCP<Teuchos::TimeMonitor> tM = rcp(new TimeMonitor(*TimeMonitor::getNewTimer(std::string("panzer_stk::STKConnManager::buildConnectivity"))));
#endif

  TEUCHOS_TEST_FOR_EXCEPTION (fp.getCellTopology().getDimension()>m_elem_blocks_topo.dimension(), std::logic_error,
      "[STKConnManager] Error! Field pattern incompatible with stored elem_blocks.\n"
      "  - Pattern dim   : " + std::to_string(fp.getCellTopology().getDimension()) + "\n"
      "  - elem_blocks topo dim: " + std::to_string(m_elem_blocks_topo.dimension()) + "\n");

  // Build sub cell ID counts and offsets
  //    ID counts = How many IDs belong on each subcell (number of mesh DOF used)
  //    Offset = What is starting index for subcell ID type?
  //             Global numbering goes like [node ids, edge ids, face ids, cell ids]
  LO nodeIdCnt=0, edgeIdCnt=0, faceIdCnt=0, cellIdCnt=0;
  GO nodeOffset=0, edgeOffset=0, faceOffset=0, cellOffset=0;
  buildOffsetsAndIdCounts(fp, nodeIdCnt,  edgeIdCnt,  faceIdCnt,  cellIdCnt,
                              nodeOffset, edgeOffset, faceOffset, cellOffset);

  // loop over elements and build global connectivity
  const int numElems = m_elements.size();
  constexpr auto NODE_RANK = stk::topology::NODE_RANK;
  constexpr auto EDGE_RANK = stk::topology::EDGE_RANK;
  constexpr auto FACE_RANK = stk::topology::FACE_RANK;
  const auto elem_rank = fp.getCellTopology().getDimension();

  for (int ielem=0; ielem<numElems; ++ielem) {
    GO numIds = 0;
    stk::mesh::Entity element = m_elements[ielem];

    // Current size of m_connectivity is the offset for this element
    m_elmtLidToConn[ielem] = m_connectivity.size();

    // add connecviities for sub cells
    if (elem_rank>NODE_RANK)
      numIds += addSubcellConnectivities(element,NODE_RANK,nodeIdCnt,nodeOffset);
    if (elem_rank>EDGE_RANK)
      numIds += addSubcellConnectivities(element,EDGE_RANK,edgeIdCnt,edgeOffset);
    if (elem_rank>FACE_RANK)
      numIds += addSubcellConnectivities(element,FACE_RANK,faceIdCnt,faceOffset);

    // add connectivity for parent cells
    if(cellIdCnt>0) {
       // add connectivities: adjust for STK indexing craziness
       const auto cell_id = m_bulkData->identifier(element);
       auto owned = m_bulkData->bucket(element).owned() ? Owned : Ghosted;
       for(LO i=0; i<cellIdCnt; ++i) {
          m_connectivity.push_back(cellOffset+cellIdCnt*(cell_id-1)+i);
          m_ownership.push_back(owned);
       }
       numIds += cellIdCnt;
    }

    m_connSize[ielem] = numIds;
  }
}

std::string STKConnManager::getBlockId (LO localElmtId) const
{
   // walk through the element blocks and figure out which this ID belongs to
   stk::mesh::Entity element = m_elements[localElmtId];

   const auto& b = m_bulkData->bucket(element);
   for (const auto& it : m_elem_blocks) {
      if (b.member(*it.second)) {
        return it.first;
      }
   }

   return "";
}

const std::vector<LO>&
STKConnManager::getAssociatedNeighbors(const LO& /* el */) const
{
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error,
      "Error! Albany does not use elements halos in the mesh, so the method\n"
      "       'STKConnManager::getAssociatedNeighbors' should not have been called.\n");

  static std::vector<LO> ret;
  return ret;
}

void STKConnManager::buildMaxEntityIds() {
  // developed to mirror "comm_mesh_counts" in stk_mesh/base/Comm.cpp
  const auto entityRankCount = m_metaData->entity_rank_count();
  
  stk::ParallelMachine mach = m_bulkData->parallel();
  // procRank_ = stk::parallel_machine_rank(mach);
  
  std::vector<stk::mesh::EntityId> local(entityRankCount,0);
  m_maxEntityId.resize(entityRankCount);
  
  // determine maximum ID for this processor for each entity type
  stk::mesh::Selector ownedPart = m_metaData->locally_owned_part();
  for (auto rank=stk::topology::NODE_RANK; rank<entityRankCount; ++rank) {
    std::vector<stk::mesh::Entity> entities;
  
    stk::mesh::get_selected_entities(ownedPart, m_bulkData->buckets(rank), entities);
  
    // determine maximum ID for this processor
    for (const auto& entity : entities) {
      stk::mesh::EntityId id = m_bulkData->identifier(entity);
      local[rank] = std::max(local[rank],id);
    }
  }
  
  // get largest IDs across processors
  stk::all_reduce_max(mach,local.data(),m_maxEntityId.data(),entityRankCount);
}

int STKConnManager::elementLocalId(stk::mesh::Entity elmt) const
{
  const auto gid = m_bulkData->identifier(elmt);
  auto it = m_localIDHash.find(gid);
  TEUCHOS_ASSERT(it!=m_localIDHash.end());
  return it->second;
}

void STKConnManager::
getMyElements(std::vector<stk::mesh::Entity> & elements) const
{
  TEUCHOS_TEST_FOR_EXCEPTION (m_elem_blocks.size()>1, std::runtime_error,
      "Error! More than one element block in this STKConnManager.\n");

  getMyElements(m_elem_blocks.begin()->first,elements);
}

void STKConnManager::
getMyElements (const std::string & blockID,
               std::vector<stk::mesh::Entity> & elements) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(m_elem_blocks.find(blockID)==m_elem_blocks.end(),std::logic_error,
      "[STKConnManager] Could not find element block '" + blockID + "'\n");

  const auto& elem_block = *m_elem_blocks.at(blockID);

  stk::mesh::Selector selector = elem_block;
  selector &= m_metaData->locally_owned_part();

  // NOTE: it could be that rank!=stk::topology::ELEM_RANK. E.g, we could have
  //       have rank=EDGE_RANK
  stk::topology::rank_t rank = m_elem_blocks_topo.rank();
  stk::mesh::get_selected_entities(selector, m_bulkData->buckets(rank), elements);
}

stk::mesh::EntityId STKConnManager::
getMaxEntityId (const stk::mesh::EntityRank entityRank) const
{
  TEUCHOS_TEST_FOR_EXCEPTION (not m_metaData->check_rank(entityRank), std::logic_error,
      "[STKConnManager::getMaxEntityId] Invalid entity rank: " + std::to_string(entityRank) + "\n");

  return m_maxEntityId[entityRank];
}

void STKConnManager::buildLocalElementIDs(const std::vector<stk::mesh::Entity>& elements)
{
  int currentLocalId = m_localIDHash.size();

  for (auto element : elements) {
    m_localIDHash[m_bulkData->identifier(element)] = currentLocalId;
    ++currentLocalId;
  }
}

// Return true if the $subcell_pos-th subcell of dimension $subcell_dim in
// local element $ielem belongs to sub part $sub_part_name
bool STKConnManager::
belongs (const std::string& sub_part_name,
         const LO ielem, const int subcell_dim, const int subcell_pos) const
{
  using rank_t = stk::topology::rank_t;
  auto rank = subcell_dim==0 ? rank_t::NODE_RANK :
             (subcell_dim==1 ? rank_t::EDGE_RANK :
             (subcell_dim==2 ? rank_t::FACE_RANK : rank_t::ELEM_RANK));
  const auto& elem = m_elements[ielem];
  const auto& sub = *(m_bulkData->begin(elem,rank) + subcell_pos);
  const auto& b = m_bulkData->bucket(sub);

  const auto& p = *m_metaData->get_part(sub_part_name);

  return b.member(p);
}

// Queries the dimension of a part
int STKConnManager::
part_dim (const std::string& part_name) const
{
  const auto& p = *m_metaData->get_part(part_name);
  return p.topology().dimension();
}

} // namespace Albany
