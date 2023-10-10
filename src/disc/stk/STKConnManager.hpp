//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_STK_CONN_MANAGER_HPP
#define ALBANY_STK_CONN_MANAGER_HPP

#include "Albany_ConnManager.hpp"

#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/BulkData.hpp"
#include "Teuchos_RCP.hpp"

#include <vector>
#include <map>

namespace Albany {

class STKConnManager : public ConnManager {
public:
  // Note: the parts requested MUST satisfy
  //  - they all have the same dimension (e.g., no elems and sides)
  //  - they do NOT intersect
  // The ctor verifies that the conditions are met
  STKConnManager(const Teuchos::RCP<const stk::mesh::MetaData>& metaData,
                 const Teuchos::RCP<const stk::mesh::BulkData>& bulkData,
                 const std::vector<std::string>& elem_blocks_names);

  // Shortcut for single part
  STKConnManager(const Teuchos::RCP<const stk::mesh::MetaData>& metaData,
                 const Teuchos::RCP<const stk::mesh::BulkData>& bulkData,
                 const std::string& elem_block_name)
   : STKConnManager (metaData,bulkData,std::vector<std::string>(1,elem_block_name))
  {
    // Nothing to do
  }

  ~STKConnManager() = default;

  // Do not hide other methods
  using ConnManager::getElementsInBlock;
  std::vector<GO>
  getElementsInBlock (const std::string& elem_block_name) const override;

  /** Tell the connection manager to build the connectivity assuming
    * a particular field pattern.
    *
    * \param[in] fp Field pattern to build connectivity for
    */
  void buildConnectivity(const panzer::FieldPattern & fp) override;

  /** Build a clone of this connection manager, without any assumptions
    * about the required connectivity (e.g. <code>buildConnectivity</code>
    * has never been called).
    */
  Teuchos::RCP<panzer::ConnManager> noConnectivityClone() const override;

  /** Get ID connectivity for a particular element
    *
    * \param[in] localElmtId Local element ID
    *
    * \returns Pointer to beginning of indices, with total size
    *          equal to <code>getConnectivitySize(localElmtId)</code>
    */
  const GO * getConnectivity(LO localElmtId) const override {
    TEUCHOS_TEST_FOR_EXCEPTION (m_connectivity.size()==0, std::logic_error,
        "Error! Cannot call getConnectivity before connectivity is built.\n");
    return &m_connectivity[m_elmtLidToConn[localElmtId]];
  }
  
  /** Get vector of bools associated to connectivity for a particular element, indicating whether the entity is owned by this rank
    *
    * \param[in] localElmtId Local element ID
    *
    * \returns vector of bools, with total size
    *          equal to <code>getConnectivitySize(localElmtId)</code>
    */
  const Ownership* getOwnership(LO localElmtId) const override {
    TEUCHOS_TEST_FOR_EXCEPTION (m_ownership.size()==0, std::logic_error,
        "Error! Cannot call getOwnership before connectivity is built.\n");
    return &m_ownership[m_elmtLidToConn[localElmtId]];
  }

  std::vector<int> getConnectivityMask (const std::string& sub_part_name) const override;

  // Where does element ielem connectivity start in the connectivity array?
  int getConnectivityStart(const LO localElmtId) const override {
    return m_elmtLidToConn[localElmtId];
  }

  /** How many mesh IDs are associated with this element?
    *
    * \param[in] localElmtId Local element ID
    *
    * \returns Number of mesh IDs that are associated with this element.
    */
  LO getConnectivitySize(LO localElmtId) const override {
    return m_connSize[localElmtId];
  }

  /** Get the block ID for a particular element.
    *
    * \param[in] localElmtId Local element ID
    */
  std::string getBlockId(LO localElmtId) const override;

  /** How many element blocks in this mesh?
    */
  std::size_t numElementBlocks() const override {
    return m_elem_blocks.size();
  }

  /** Get block IDs from STK mesh object
    */
  void getElementBlockIds(std::vector<std::string> & elementBlockIds) const override {
    elementBlockIds.resize(0);
    elementBlockIds.reserve(m_elem_blocks_names.size());
    for (const auto& it : m_elem_blocks) {
      elementBlockIds.push_back(it.first);
    }
  }

  /** What are the cellTopologies linked to element blocks in this connection manager?
   */
  void getElementBlockTopologies(std::vector<shards::CellTopology> & elementBlockTopologies) const override {
    elementBlockTopologies.resize(0);
    elementBlockTopologies.reserve(m_elem_blocks.size());
    for (const auto& it : m_elem_blocks) {
      elementBlockTopologies.push_back(stk::mesh::get_cell_topology(it.second->topology()));
    }
  }

  /** Get the local element IDs for a paricular element
    * block. These are only the owned element ids.
    *
    * \param[in] blockIndex Block Index
    *
    * \returns Vector of local element IDs.
    */
  const std::vector<LO> & getElementBlock(const std::string & elem_block_name) const override {
    TEUCHOS_TEST_FOR_EXCEPTION (m_elementBlocks.find(elem_block_name)==m_elementBlocks.end(), std::runtime_error,
        "[STKConnManager] Error! Block '" + elem_block_name + "' not found in the mesh.\n");
    return m_elementBlocks.at(elem_block_name);
  }

  /** Get the local element IDs for a paricular element
    * block. These element ids are not owned, and the element
    * will live on another processor.
    *
    * \param[in] blockIndex Block Index
    *
    * \returns Vector of local element IDs.
    */
  const std::vector<LO> & getNeighborElementBlock(const std::string & elem_block_name) const override {
    TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error,
        "Error! Albany does not use elements halos, so the method\n"
        "       'STKConnManager::getNeighborElementBlock' should not have been called.\n");
    return m_neighborElementBlocks.at(elem_block_name);
  }

  int getOwnedElementCount() const {
    return m_elements.size();
  }

  /** Get elements, if any, associated with <code>el</code>, excluding
    * <code>el</code> itself.
    */
  const std::vector<LO>& getAssociatedNeighbors(const LO& el) const override;

  /** Return whether getAssociatedNeighbors will return true for at least one
    * input. Default implementation returns false.
    */
  // NOTE: Albany should not use neighbors, so always false.
  bool hasAssociatedNeighbors() const override {
    return false;
  }

  // Queries the dimension of a part
  int part_dim (const std::string& part_name) const override;

protected:

  std::string containingBlockId(stk::mesh::Entity elmt) const;

  int elementLocalId (const stk::mesh::Entity elmt) const;
  stk::mesh::EntityId getMaxEntityId (const stk::mesh::EntityRank entityRank) const;

  void getMyElements(std::vector<stk::mesh::Entity> & elements) const;

  void getMyElements(const std::string & blockID,
                     std::vector<stk::mesh::Entity> & elements) const;

  void clearLocalElementMapping();
  void buildLocalElementMapping();
  void buildOffsetsAndIdCounts(
      const panzer::FieldPattern & fp,
      LO & nodeIdCnt, LO & edgeIdCnt,
      LO & faceIdCnt, LO & cellIdCnt,
      GO & nodeOffset, GO & edgeOffset,
      GO & faceOffset, GO & cellOffset) const;

   LO addSubcellConnectivities(
       const stk::mesh::Entity element,
       const stk::mesh::EntityRank subcellRank,
       const LO idCnt,
       const GO offset);

  // Compute max gid for each entity rank
  void buildMaxEntityIds();

  // Assing local ids to elements
  void buildLocalElementIDs(const std::vector<stk::mesh::Entity>& elements);

  std::vector<stk::mesh::Entity>  m_elements;

  // element block information
  // NOTE: Albany should *never* use neighbors, since we do not require
  //       an element halo anywhere. Keep the neighbor fcn anyways,
  //       for simplicity in the getter functions
  std::map<std::string,std::vector<LO> > m_elementBlocks;
  std::map<std::string,std::vector<LO> > m_neighborElementBlocks;

  // Map elemLID to offset in m_connectivity
  std::vector<LO> m_elmtLidToConn;

  // For each elemLID, returns size of connectivity
  std::vector<LO> m_connSize;

  // List of GIDs of entities in each element, spliced together
  std::vector<GO> m_connectivity;

  // Whether entities in each element are owned or ghosted by this MPI rank
  std::vector<Ownership> m_ownership;

  // The max gid for each element rank, across the whole mesh.
  std::vector<stk::mesh::EntityId> m_maxEntityId;

  // Stk Mesh Objects
  Teuchos::RCP<const stk::mesh::MetaData>   m_metaData;
  Teuchos::RCP<const stk::mesh::BulkData>   m_bulkData;

  std::map<std::string,stk::mesh::Part*>    m_elem_blocks;
  stk::topology                             m_elem_blocks_topo;

  std::vector<LO> m_idCnt;

  std::unordered_map<stk::mesh::EntityId, int> m_localIDHash;
};

} // namespace Albany

#endif // ALBANY_STK_CONN_MANAGER_HPP
