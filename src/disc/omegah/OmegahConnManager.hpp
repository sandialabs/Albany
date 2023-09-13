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
#include <array> //std::iota

namespace Albany {

struct OmegahPartFilter {
  const LO dim;
  const std::string name;
};

class OmegahConnManager : public ConnManager {
private:
  Omega_h::Mesh& mesh;
  const OmegahPartFilter partFilter;
  std::vector<LO> localElmIds;
  std::vector<LO> emptyHaloVec;
  std::vector<Ownership> m_ownership; //FIXME
  LO m_dofsPerElm = 0;
  std::array<LO,4> m_dofsPerEnt;
  std::array<Omega_h::GOs,4> m_globalDofNumbering;
  Omega_h::HostRead<Omega_h::GO> m_connectivity;
  LO getPartConnectivitySize() const;
  std::array<Omega_h::GOs,4> createGlobalDofNumbering() const;
  Omega_h::GOs createElementToDofConnectivity(const Omega_h::Adj elmToDim[3],
    const std::array<Omega_h::GOs,4>& globalDofNumbering) const;
  Omega_h::GOs createElementToDofConnectivityMask(const std::string& tagName,
    const Omega_h::Adj elmToDim[3]) const;
public:
  OmegahConnManager(Omega_h::Mesh& in_mesh);
  OmegahConnManager(Omega_h::Mesh& in_mesh, std::string partId, const int partDim);

  ~OmegahConnManager() = default;

  Omega_h::GOs getGlobalDofNumbering(int dim) const {
    assert(dim>=0 && dim<=mesh.dim());
    return m_globalDofNumbering[dim];
  }

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

  /** Get global id of the specified element.
    * Note, this function requires a device-to-host
    * memory transfer if the mesh is on the device.
    * In its current form it should only be used for
    * debugging.
    *
    * \param[in] localElmtId Local element ID
    *
    * \returns global id of mesh element
    */
  GO getElementGlobalId(LO localElmtId) const {
    auto gids = mesh.globals(mesh.dim());
    return gids.get(localElmtId);
  }

  /** Get ID connectivity for a particular element
    *
    * \param[in] localElmtId Local element ID
    *
    * \returns Pointer to beginning of indices, with total size
    *          equal to <code>getConnectivitySize(localElmtId)</code>
    *          For the given element, the connectivity is ordered as:
    *          [vtx dofs][edge dofs][face dofs][element dofs]
    */
  const GO * getConnectivity(LO localElmtId) const override {
    TEUCHOS_TEST_FOR_EXCEPTION (m_connectivity.size()==0, std::logic_error,
        "Error! Cannot call getConnectivity before connectivity is built.\n");
    static_assert(sizeof(Omega_h::GO) == sizeof(GO));
    auto ptr = m_connectivity.data() + (localElmtId*m_dofsPerElm);
    return reinterpret_cast<const GO*>(ptr);
  }

  /** How many mesh IDs are associated with this element?
    *
    * \param[in] localElmtId Local element ID
    *
    * \returns Number of mesh IDs that are associated with this element.
    */
  LO getConnectivitySize(LO /*localElmtId*/) const override {
    return m_dofsPerElm; //how can this not be constant across all elements
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
    switch (mesh.family()) {
      case OMEGA_H_SIMPLEX:
        if(mesh.dim()==3) {
          shards::CellTopology tetTopo(shards::getCellTopologyData< shards::Tetrahedron<4> >());
          elementBlockTopologies.push_back(tetTopo);
        } else if(mesh.dim()==2) {
          shards::CellTopology triTopo(shards::getCellTopologyData< shards::Triangle<3> >());
          elementBlockTopologies.push_back(triTopo);
        }
        break;
      case OMEGA_H_HYPERCUBE:
        if(mesh.dim()==3) {
          shards::CellTopology hexa(shards::getCellTopologyData< shards::Hexahedron<8> >());
          elementBlockTopologies.push_back(hexa);
        } else if(mesh.dim()==2) {
          shards::CellTopology quad(shards::getCellTopologyData< shards::Quadrilateral<4> >());
          elementBlockTopologies.push_back(quad);
        }
        break;
      default:
        TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error,
            "Error! Unrecognized/unsupported omegah mesh family.\n");
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

  int getConnectivityStart (const LO ielem) const;
  std::vector<int> getConnectivityMask (const std::string& sub_part_name) const;

  // Queries the dimension of a part
  int part_dim (const std::string& part_name) const override;

  const Ownership* getOwnership(LO localElmtId) const override {
    TEUCHOS_TEST_FOR_EXCEPTION (m_ownership.size()==0, std::logic_error,
        "Error! Cannot call getOwnership before connectivity is built.\n");
    return m_ownership.data() + (localElmtId*m_dofsPerElm);
  }
};

} // namespace Albany

#endif // ALBANY_Omegah_CONN_MANAGER_HPP
