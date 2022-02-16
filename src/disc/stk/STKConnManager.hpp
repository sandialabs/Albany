//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_STK_CONN_MANAGER_HPP
#define ALBANY_STK_CONN_MANAGER_HPP

#include "Albany_AbstractSTKMeshStruct.hpp"

#include "Panzer_ConnManager.hpp"
#include "Teuchos_RCP.hpp"

#include <vector>

namespace Albany {

class STKConnManager : public panzer::ConnManager {
public:
  using LocalOrdinal  = typename panzer::ConnManager::LocalOrdinal;
  using GlobalOrdinal = typename panzer::ConnManager::GlobalOrdinal;

  typedef double ProcIdData;
  typedef stk::mesh::Field<double> SolutionFieldType;
  typedef stk::mesh::Field<ProcIdData> ProcIdFieldType;
  typedef stk::mesh::Field<double,stk::mesh::Cartesian> VectorFieldType;

  STKConnManager(const Teuchos::RCP<const AbstractSTKMeshStruct>& stkMeshStruct);

  ~STKConnManager() = default;

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
  const GlobalOrdinal * getConnectivity(LocalOrdinal localElmtId) const override {
    return &m_connectivity[m_elmtLidToConn[localElmtId]];
  }

  /** How many mesh IDs are associated with this element?
    *
    * \param[in] localElmtId Local element ID
    *
    * \returns Number of mesh IDs that are associated with this element.
    */
  LocalOrdinal getConnectivitySize(LocalOrdinal localElmtId) const override {
    return m_connSize[localElmtId];
  }

  /** Get the block ID for a particular element.
    *
    * \param[in] localElmtId Local element ID
    */
  std::string getBlockId(LocalOrdinal localElmtId) const override;

  /** How many element blocks in this mesh?
    */
  std::size_t numElementBlocks() const override {
    return m_stkMeshStruct->ebNames_.size();
  }

  /** Get block IDs from STK mesh object
    */
  void getElementBlockIds(std::vector<std::string> & elementBlockIds) const override {
    elementBlockIds = m_stkMeshStruct->ebNames_;
  }

  /** What are the cellTopologies linked to element blocks in this connection manager?
   */
  void getElementBlockTopologies(std::vector<shards::CellTopology> & elementBlockTopologies) const override {
    elementBlockTopologies = m_stkMeshStruct->elementBlockTopologies_;
  }

  /** Get the local element IDs for a paricular element
    * block. These are only the owned element ids.
    *
    * \param[in] blockIndex Block Index
    *
    * \returns Vector of local element IDs.
    */
  const std::vector<LocalOrdinal> & getElementBlock(const std::string & blockId) const override {
    TEUCHOS_TEST_FOR_EXCEPTION (m_elementBlocks.count(blockId)>=1, std::runtime_error,
        "Error! Block '" << blockId << "' not found in the mesh.\n");
    return m_elementBlocks.at(blockId);
  }

  /** Get the local element IDs for a paricular element
    * block. These element ids are not owned, and the element
    * will live on another processor.
    *
    * \param[in] blockIndex Block Index
    *
    * \returns Vector of local element IDs.
    */
  const std::vector<LocalOrdinal> & getNeighborElementBlock(const std::string & blockId) const override {
    TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error,
        "Error! Albany does not use elements halos, so the method\n"
        "       'STKConnManager::getNeighborElementBlock' should not have been called.\n");
    return m_neighborElementBlocks.at(blockId);
  }

  int getOwnedElementCount() const {
    return m_elements.size();
  }

  /** Get elements, if any, associated with <code>el</code>, excluding
    * <code>el</code> itself.
    */
  const std::vector<LocalOrdinal>& getAssociatedNeighbors(const LocalOrdinal& el) const override;

  /** Return whether getAssociatedNeighbors will return true for at least one
    * input. Default implementation returns false.
    */
  // NOTE: Albany should not use neighbors, so always false.
  bool hasAssociatedNeighbors() const override {
    return false;
  }

protected:

  std::string containingBlockId(stk::mesh::Entity elmt) const;

  int elementLocalId (const stk::mesh::Entity elmt) const;
  int elementLocalId (const stk::mesh::EntityId gid) const;
  stk::mesh::EntityId getMaxEntityId (const stk::mesh::EntityRank entityRank) const;

  void getMyElements(std::vector<stk::mesh::Entity> & elements) const;

  void getMyElements(const std::string & blockID,std::vector<stk::mesh::Entity> & elements) const;

  void clearLocalElementMapping();
  void buildLocalElementMapping();
  void buildOffsetsAndIdCounts(
      const panzer::FieldPattern & fp,
      LocalOrdinal & nodeIdCnt, LocalOrdinal & edgeIdCnt,
      LocalOrdinal & faceIdCnt, LocalOrdinal & cellIdCnt,
      GlobalOrdinal & nodeOffset, GlobalOrdinal & edgeOffset,
      GlobalOrdinal & faceOffset, GlobalOrdinal & cellOffset) const;

   LocalOrdinal addSubcellConnectivities(
       const stk::mesh::Entity element,
       const stk::mesh::EntityRank subcellRank,
       const LocalOrdinal idCnt,
       const GlobalOrdinal offset);

  stk::mesh::Part * getElementBlockPart(const std::string & name) const;

  // Compute max gid for each entity rank
  // void buildEntityCounts();
  void buildMaxEntityIds();

  // Assing local ids to elements
  void buildLocalElementIDs();

  std::vector<stk::mesh::Entity>  m_elements;

  // element block information
  // NOTE: Albany should *never* use neighbors, since we do not require
  //       an element halo anywhere. Keep the neighbor fcn anyways,
  //       for simplicity in the getter functions
  std::map<std::string,std::vector<LocalOrdinal> > m_elementBlocks;
  std::map<std::string,std::vector<LocalOrdinal> > m_neighborElementBlocks;
  // std::map<std::string,GlobalOrdinal> blockIdToIndex_;

  // Map elemLID to offset in m_connectivity
  std::vector<LocalOrdinal> m_elmtLidToConn;

  // For each elemLID, returns size of connectivity
  std::vector<LocalOrdinal> m_connSize;

  // List of GIDs of entities in each element, spliced together
  std::vector<GlobalOrdinal> m_connectivity;

  // The max gid for each element rank, across the whole mesh.
  std::vector<stk::mesh::EntityId> m_maxEntityId;

private:

  //! Stk Mesh Objects
  Teuchos::RCP<const stk::mesh::MetaData>   m_metaData;
  Teuchos::RCP<const stk::mesh::BulkData>   m_bulkData;
  Teuchos::RCP<const AbstractSTKMeshStruct> m_stkMeshStruct;

  std::unordered_map<stk::mesh::EntityId, int> m_localIDHash;
};

} // namespace Albany

#endif // ALBANY_STK_CONN_MANAGER_HPP
