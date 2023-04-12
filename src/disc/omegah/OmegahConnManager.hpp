//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_Omegah_CONN_MANAGER_HPP
#define ALBANY_Omegah_CONN_MANAGER_HPP

#include "Albany_ConnManager.hpp"

#include <Omega_h_mesh.hpp>

#include "Teuchos_RCP.hpp"

#include <vector>
#include <map>
#include <numeric> //std::iota

namespace Albany {

class OmegahConnManager : public ConnManager {
private:
  Omega_h::Mesh mesh;
  std::vector<LO> localElmIds;
  std::vector<LO> emptyHaloVec;
  Omega_h::HostRead<Omega_h::GO> m_connectivity;
  void initLocalElmIds() {
    localElmIds.resize(mesh.nelems());
    std::iota(localElmIds.begin(), localElmIds.end(), 0);
  }
  void getDofsPerEnt(const panzer::FieldPattern & fp, LO entIdCnt[4]) const;
  void getConnectivityOffsets(LO fieldDim, const Omega_h::Adj elmToDim[3], const LO dofsPerEnt[4],
                              GO connectivityOffsets[4], GO connectivityGlobalOffsets[4]);
  void appendConnectivity(const Omega_h::Adj& elmToDim, LO dofsPerEnt,
                          GO startIdx, GO globalStartIdx,
                          LO dim, Omega_h::Write<Omega_h::GO>& elmDownAdj_d) const;
public:
  OmegahConnManager(Omega_h::Mesh in_mesh);

  ~OmegahConnManager() = default;

  // Do not hide other methods
  using ConnManager::getElementsInBlock;
  std::vector<GO>
  getElementsInBlock (const std::string& blockId) const override;

  /** Tell the connection manager to build the connectivity assuming
    * a particular field pattern.
    *
    * \param[in] fp Field pattern to build connectivity for
    */
  void buildConnectivity(const panzer::FieldPattern & fp) override;

  void writeConnectivity();

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
  const GO * getConnectivity(LO localElmtId) const override { //FIXME
    return NULL;
  }

  /** How many mesh IDs are associated with this element?
    *
    * \param[in] localElmtId Local element ID
    *
    * \returns Number of mesh IDs that are associated with this element.
    */
  LO getConnectivitySize(LO localElmtId) const override { //FIXME
    return 42;
  }

  /** Get the block ID for a particular element.
    * Albany only uses one block; return a static string.
    *
    * \param[in] localElmtId Local element ID
    */
  std::string getBlockId(LO localElmtId) const override {
    std::stringstream ss;
    ss << "Error! requested local elemnt id (" << localElmtId << ") is larger than the number of elements on this process (" << mesh.nelems() << ")\n";
    TEUCHOS_TEST_FOR_EXCEPTION (localElmtId >= mesh.nelems(), std::runtime_error, ss.str());
    return m_elem_blocks_names[0];
  };

  /** How many element blocks in this mesh?
    */
  std::size_t numElementBlocks() const override {
    return 1;
  }

  /** Get block IDs from Omegah mesh object
    */
  void getElementBlockIds(std::vector<std::string> & elementBlockIds) const override {
    elementBlockIds = m_elem_blocks_names;
  }

  /** What are the cellTopologies linked to element blocks in this connection manager?
   */
  void getElementBlockTopologies(std::vector<shards::CellTopology> & elementBlockTopologies) const override {
    TEUCHOS_TEST_FOR_EXCEPTION (m_elem_blocks_names.size() != 1, std::logic_error,
        "Error! The OmegahConnManager currently only supports a single block on each process\n");
    TEUCHOS_TEST_FOR_EXCEPTION ( OMEGA_H_SIMPLEX != mesh.family(), std::logic_error,
        "Error! The OmegahConnManager currently supports 2d and 3d meshes with"
        "       straight sided triangles and tets\n");
    if(mesh.dim()==3) {
      shards::CellTopology tetTopo(shards::getCellTopologyData< shards::Tetrahedron<4> >());
      elementBlockTopologies.push_back(tetTopo);
    } else if(mesh.dim()==2) {
      shards::CellTopology triTopo(shards::getCellTopologyData< shards::Triangle<3> >());
      elementBlockTopologies.push_back(triTopo);
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error,
          "Error! The OmegahConnManager currently does not 1d meshes\n");
    }
  }

  /** Get the local element IDs for a paricular element
    * block. These are only the owned element ids.
    * Albany only uses one block; ignore the blockIndex arg.
    *
    * \param[in] blockIndex Block Index
    *
    * \returns Vector of local element IDs.
    */
  const std::vector<LO> & getElementBlock(const std::string &) const override {
    return localElmIds;
  }

  /** Get the local element IDs for a paricular element
    * block. These element ids are not owned, and the element
    * will live on another processor.
    *
    * \param[in] blockIndex Block Index
    *
    * \returns Vector of local element IDs.
    */
  const std::vector<LO> & getNeighborElementBlock(const std::string & blockId) const override {
    TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error,
        "Error! Albany does not use elements halos, so the method\n"
        "       'OmegahConnManager::getNeighborElementBlock' should not have been called.\n");
    return emptyHaloVec;
  }

  int getOwnedElementCount() const {
    return mesh.nelems();
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

  // Returns whether input part name is topologically contained in the
  // parts where this ConnManager is defined.
  bool contains (const std::string& sub_part_name) const override;

  // Return true if the $subcell_pos-th subcell of dimension $subcell_dim in
  // local element $ielem belongs to sub part $sub_part_name
  bool belongs (const std::string& sub_part_name,
                const LO ielem, const int subcell_dim, const int subcell_pos) const override;

  // Queries the dimension of a part
  int part_dim (const std::string& part_name) const override;
};

} // namespace Albany

#endif // ALBANY_Omegah_CONN_MANAGER_HPP
