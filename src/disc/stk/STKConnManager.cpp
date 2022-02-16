//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "STKConnManager.hpp"

#include <stk_util/parallel/ParallelReduce.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/Comm.hpp>       // for comm_mesh_counts

#include "Teuchos_FancyOStream.hpp"

namespace Albany {

STKConnManager::
STKConnManager(const Teuchos::RCP<const AbstractSTKMeshStruct>& stkMeshStruct)
{
  // Sanity check
  TEUCHOS_TEST_FOR_EXCEPTION (stkMeshStruct.is_null(), std::runtime_error,
      "Error! Input mesh struct pointer is null.\n");

  m_stkMeshStruct = stkMeshStruct;
  m_bulkData      = m_stkMeshStruct->bulkData;
  m_metaData      = m_stkMeshStruct->metaData;

  // buildEntityCounts();
  buildMaxEntityIds();
  buildLocalElementIDs();
}

Teuchos::RCP<panzer::ConnManager>
STKConnManager::noConnectivityClone() const
{
  return Teuchos::rcp(new STKConnManager(m_stkMeshStruct));
}

void STKConnManager::clearLocalElementMapping()
{
  m_elements.clear();
  m_elementBlocks.clear();
  m_elmtLidToConn.clear();
  m_connSize.clear();
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

    // 3. Build block to LID map
    auto& blockElems = m_elementBlocks[blockId];
    blockElems.reserve(blockElmts.size());
    for (const auto& elem : blockElmts) {
      blockElems.push_back (elementLocalId(elem));
    }
  }

  // this expensive operation guarantees ordering of local IDs
  auto cmpLids = [&](stk::mesh::Entity a, stk::mesh::Entity b)->bool{
     return elementLocalId(a) < elementLocalId(b);
  };
  std::sort(m_elements.begin(), m_elements.end(), cmpLids);

  // Pre allocate space for internal connectivity offsets/sizes
  m_elmtLidToConn.clear();
  m_elmtLidToConn.resize(m_elements.size(),0);

  m_connSize.clear();
  m_connSize.resize(m_elements.size(),0);
}

void STKConnManager::
buildOffsetsAndIdCounts(
    const panzer::FieldPattern & fp,
    LocalOrdinal & nodeIdCnt, LocalOrdinal & edgeIdCnt,
    LocalOrdinal & faceIdCnt, LocalOrdinal & cellIdCnt,
    GlobalOrdinal & nodeOffset, GlobalOrdinal & edgeOffset,
    GlobalOrdinal & faceOffset, GlobalOrdinal & cellOffset) const
{
  // get the global counts for all the nodes, faces, edges and cells
  GlobalOrdinal maxNodeId = getMaxEntityId(stk::topology::NODE_RANK);
  GlobalOrdinal maxEdgeId = getMaxEntityId(stk::topology::EDGE_RANK);
  GlobalOrdinal maxFaceId = getMaxEntityId(stk::topology::FACE_RANK);

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
      cellIdCnt = fp.getSubcellIndices(patternDim,0).size();
      break;
    case 0:
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

STKConnManager::LocalOrdinal
STKConnManager::
addSubcellConnectivities (const stk::mesh::Entity element,
                          const stk::mesh::EntityRank subcellRank,
                          const LocalOrdinal idCnt,
                          const GlobalOrdinal offset)
{
  if(idCnt<=0)
     return 0;

  // loop over all relations of specified type
  LocalOrdinal numIds = 0;
  const int num_rels = m_bulkData->num_connectivity(element, subcellRank);
  stk::mesh::Entity const* relations = m_bulkData->begin(element, subcellRank);
  for(int sc=0; sc<num_rels; ++sc) {
    stk::mesh::Entity subcell = relations[sc];
    const auto subcell_id = m_bulkData->identifier(subcell);

    // add connectivities: adjust for STK indexing craziness
    for (LocalOrdinal i=0; i<idCnt; ++i) {
      m_connectivity.push_back(offset+idCnt*(subcell_id-1)+i);
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

  // get element info from STK_Interface
  // object and build a local element mapping.
  buildLocalElementMapping();

  // Build sub cell ID counts and offsets
  //    ID counts = How many IDs belong on each subcell (number of mesh DOF used)
  //    Offset = What is starting index for subcell ID type?
  //             Global numbering goes like [node ids, edge ids, face ids, cell ids]
  LocalOrdinal nodeIdCnt=0, edgeIdCnt=0, faceIdCnt=0, cellIdCnt=0;
  GlobalOrdinal nodeOffset=0, edgeOffset=0, faceOffset=0, cellOffset=0;
  buildOffsetsAndIdCounts(fp, nodeIdCnt,  edgeIdCnt,  faceIdCnt,  cellIdCnt,
                              nodeOffset, edgeOffset, faceOffset, cellOffset);

  // loop over elements and build global connectivity
  const int numElems = m_elements.size();
  constexpr auto NODE_RANK = stk::topology::NODE_RANK;
  constexpr auto EDGE_RANK = stk::topology::EDGE_RANK;
  constexpr auto FACE_RANK = stk::topology::FACE_RANK;
  for (int ielem=0; ielem<numElems; ++ielem) {
     GlobalOrdinal numIds = 0;
     stk::mesh::Entity element = m_elements[ielem];

     // Current size of m_connectivity is the offset for this element
     m_elmtLidToConn[ielem] = m_connectivity.size();

     // add connecviities for sub cells
     numIds += addSubcellConnectivities(element,NODE_RANK,nodeIdCnt,nodeOffset);
     numIds += addSubcellConnectivities(element,EDGE_RANK,edgeIdCnt,edgeOffset);
     numIds += addSubcellConnectivities(element,FACE_RANK,faceIdCnt,faceOffset);

     // add connectivity for parent cells
     if(cellIdCnt>0) {
        // add connectivities: adjust for STK indexing craziness
        const auto cell_id = m_bulkData->identifier(element);
        for(LocalOrdinal i=0; i<cellIdCnt; ++i) {
           m_connectivity.push_back(cellOffset+cellIdCnt*(cell_id-1));
        }
        numIds += cellIdCnt;
     }

     m_connSize[ielem] = numIds;
  }
}

std::string STKConnManager::getBlockId (LocalOrdinal localElmtId) const
{
   // walk through the element blocks and figure out which this ID belongs to
   stk::mesh::Entity element = m_elements[localElmtId];

   return containingBlockId(element);
}

inline std::size_t
getElementIdx(const std::vector<stk::mesh::Entity>& elements,
              stk::mesh::Entity const e)
{
  return static_cast<std::size_t>(
    std::distance(elements.begin(), std::find(elements.begin(), elements.end(), e)));
}

const std::vector<STKConnManager::LocalOrdinal>&
STKConnManager::getAssociatedNeighbors(const LocalOrdinal& /* el */) const
{
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error,
      "Error! Albany does not use elements halos in the mesh, so the method\n"
      "       'STKConnManager::getAssociatedNeighbors' should not have been called.\n");

  static std::vector<LocalOrdinal> ret;
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
  return elementLocalId(m_bulkData->identifier(elmt));
}

int STKConnManager::elementLocalId(stk::mesh::EntityId gid) const
{
  auto it = m_localIDHash.find(gid);
  TEUCHOS_ASSERT(it!=m_localIDHash.end());
  return it->second;
}

void STKConnManager::getMyElements(std::vector<stk::mesh::Entity> & elements) const
{
  // setup local ownership
  stk::mesh::Selector ownedPart = m_metaData->locally_owned_part();

  // grab elements
  constexpr auto ELEM_RANK = stk::topology::ELEMENT_RANK;
  stk::mesh::get_selected_entities(ownedPart, m_bulkData->buckets(ELEM_RANK), elements);
}

void STKConnManager::
getMyElements (const std::string & blockID,
               std::vector<stk::mesh::Entity> & elements) const
{
  stk::mesh::Part * elementBlock = getElementBlockPart(blockID);

  TEUCHOS_TEST_FOR_EXCEPTION(elementBlock==0,std::logic_error,
      "Could not find element block \"" << blockID << "\"");

  // setup local ownership
  stk::mesh::Selector ownedBlock = m_metaData->locally_owned_part() & (*elementBlock);

  // grab elements
  constexpr auto ELEM_RANK = stk::topology::ELEMENT_RANK;
  stk::mesh::get_selected_entities(ownedBlock, m_bulkData->buckets(ELEM_RANK), elements);
}

stk::mesh::Part * STKConnManager::
getElementBlockPart(const std::string & name) const
{
  auto it = m_stkMeshStruct->elementBlockParts_.find(name);
  if(it==m_stkMeshStruct->elementBlockParts_.end()) return 0;
  return it->second;
}

stk::mesh::EntityId STKConnManager::
getMaxEntityId (const stk::mesh::EntityRank entityRank) const
{
  TEUCHOS_TEST_FOR_EXCEPTION (entityRank>=m_maxEntityId.size(),std::logic_error,
      "[STKConnManager::getMaxEntityId] Invalid entity rank: " << entityRank << "\n");

  return m_maxEntityId[entityRank];
}

std::string STKConnManager::containingBlockId(stk::mesh::Entity elmt) const
{
  for(const auto & eb_pair : m_stkMeshStruct->elementBlockParts_) {
    if(m_bulkData->bucket(elmt).member(*(eb_pair.second))) {
      return eb_pair.first;
    }
  }
  return "";
}

void STKConnManager::buildLocalElementIDs()
{
  int currentLocalId = 0;

  // might be better (faster) to do this by buckets
  std::vector<stk::mesh::Entity> elements;
  getMyElements(elements);

  for (auto element : elements) {
    m_localIDHash[m_bulkData->identifier(element)] = currentLocalId;
    ++currentLocalId;
  }
}

} // namespace Albany
