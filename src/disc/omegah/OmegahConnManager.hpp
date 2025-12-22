//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_Omegah_CONN_MANAGER_HPP
#define ALBANY_Omegah_CONN_MANAGER_HPP

#include "Albany_OmegahGenericMesh.hpp"
#include "Albany_ConnManager.hpp"

#include <Omega_h_mesh.hpp>

#include "Teuchos_RCP.hpp"

#include <vector>
#include <map>
#include <numeric> //std::iota
#include <array> //std::iota

namespace Albany {

class OmegahConnManager : public ConnManager {
private:
  Teuchos::RCP<OmegahGenericMesh> albanyMesh;
  Teuchos::RCP<Omega_h::Mesh> mesh;

  std::vector<LO> localElmIds;
  std::vector<LO> emptyHaloVec;
  std::vector<Ownership> owners;
  LO m_dofsPerElm = 0;
  std::array<LO,4> m_dofsPerEnt;
  std::array<Omega_h::GOs,4> m_globalDofNumbering;
  Omega_h::HostRead<Omega_h::GO> m_connectivity;

  LO getPartConnectivitySize() const;
  std::array<Omega_h::GOs,4> createGlobalDofNumbering() const;
  Omega_h::GOs createElementToDofConnectivity(const Omega_h::Adj elmToDim[3],
    const std::array<Omega_h::GOs,4>& globalDofNumbering) const;
  Omega_h::GOs createElementToDofConnectivityMask(
    Omega_h::Read<Omega_h::I8> maskArray[4],
    const Omega_h::Adj elmToDim[3]) const;
  std::vector<Ownership> buildConnectivityOwnership() const;
public:
  //Passing the connectivity array from Omegah to Albany in
  // getConnectivity(LO localElmtId) requires matching global
  // ordinal types.
  static_assert(sizeof(Omega_h::GO) == sizeof(GO));

  OmegahConnManager(const Teuchos::RCP<OmegahGenericMesh>& albanyMesh);
  OmegahConnManager(const Teuchos::RCP<OmegahGenericMesh>& albanyMesh,
                    const std::string& partId);

  ~OmegahConnManager() = default;

  Omega_h::GOs getGlobalDofNumbering(int dim) const {
    assert(dim>=0 && dim<=mesh->dim());
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
  using ConnManager::buildConnectivity;

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
    auto gids = mesh->globals(mesh->dim());
    return gids.get(localElmtId); //FIXME - returns 0 on both ranks in 1d, two element test
  }

  /** Get ID connectivity for a particular element
    * \details the static assertion at the top of the class
    *          ensures that the global ordinal type in
    *          Omega_h and Albany are the same. This function
    *          will fail (silently?) if they don't match.
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
    ss << "Error! requested local elemnt id (" << localElmtId << ") is larger than the number of elements on this process (" << mesh->nelems() << ")\n";
    TEUCHOS_TEST_FOR_EXCEPTION (localElmtId >= mesh->nelems(), std::runtime_error, ss.str());
    return m_elem_blocks_names[0];
  };

  /** How many element blocks in this mesh?
    */
  std::size_t numElementBlocks() const override {
    return 1;
  }

  /** What are the cellTopologies linked to element blocks in this connection manager?
   */
  void getElementBlockTopologies(std::vector<shards::CellTopology> & elementBlockTopologies) const override {

    TEUCHOS_TEST_FOR_EXCEPTION (m_elem_blocks_names.size() != 1, std::logic_error,
        "Error! The OmegahConnManager currently only supports a single block on each process\n");
    TEUCHOS_TEST_FOR_EXCEPTION ( OMEGA_H_SIMPLEX != mesh->family(), std::logic_error,
        "Error! The OmegahConnManager currently supports 1d, 2d, 3d meshes with"
        "        edges, triangles, and tets.\n");
    switch (mesh->family()) {
      case OMEGA_H_SIMPLEX:
        if(mesh->dim()==3) {
          shards::CellTopology tetTopo(shards::getCellTopologyData< shards::Tetrahedron<4> >());
          elementBlockTopologies.push_back(tetTopo);
        } else if(mesh->dim()==2) {
          shards::CellTopology triTopo(shards::getCellTopologyData< shards::Triangle<3> >());
          elementBlockTopologies.push_back(triTopo);
        } else if(mesh->dim()==1) {
          shards::CellTopology edgeTopo(shards::getCellTopologyData< shards::Line<2> >());
          elementBlockTopologies.push_back(edgeTopo);
        } else {
          TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error,
              "Error! Omega_h simplex mesh dimension must be 1, 2, or 3.\n");
        }
        break;
      case OMEGA_H_HYPERCUBE:
        if(mesh->dim()==3) {
          shards::CellTopology hexa(shards::getCellTopologyData< shards::Hexahedron<8> >());
          elementBlockTopologies.push_back(hexa);
        } else if(mesh->dim()==2) {
          shards::CellTopology quad(shards::getCellTopologyData< shards::Quadrilateral<4> >());
          elementBlockTopologies.push_back(quad);
        } else {
          TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error,
              "Error! Omega_h hypercube mesh dimension must be 2 or 3.\n");
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
  using ConnManager::getElementBlock;
  const std::vector<LO> & getElementBlock(const std::string &) const override {
    return localElmIds;
  }

  int getOwnedElementCount() const;

  int getConnectivityStart (const LO localElmtId) const override;
  std::vector<int> getConnectivityMask (const std::string& sub_part_name) const override;

  // Queries the dimension of a part
  // Do not hide other methods
  using ConnManager::part_dim;
  int part_dim (const std::string& part_name) const override;

  const Ownership* getOwnership(LO localElmtId) const override;
};

} // namespace Albany

#endif // ALBANY_Omegah_CONN_MANAGER_HPP
