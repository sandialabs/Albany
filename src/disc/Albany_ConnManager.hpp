//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_CONN_MANAGER_HPP
#define ALBANY_CONN_MANAGER_HPP

#include "Albany_ScalarOrdinalTypes.hpp"

#include "Panzer_ConnManager.hpp"

#include <vector>

namespace Albany {

enum Ownership {
  Unset = 0,
  Owned = 1,
  Ghosted = 2
};

// The only purpose of this class is to expose one bit of information
// that the panzer base class doesn't, which is a list of global ids
// of the elements in a block.
class ConnManager : public panzer::ConnManager {
public:
  virtual ~ConnManager() = default;

  virtual std::vector<GO>
  getElementsInBlock (const std::string& blockId) const = 0;

  std::vector<GO> getElementsInBlock () const {
    return getElementsInBlock(elem_block_name());
  }

  using panzer::ConnManager::getElementBlock;
  const std::vector<LO> getElementBlock () const {
    return this->getElementBlock(elem_block_name());
  }

  const std::vector<LO> & getNeighborElementBlock(const std::string & blockId) const override
  {
    static std::vector<LO> emptyVec;
    throw std::runtime_error(
        "Error! Albany does not use elements halos, so the method\n"
        "       'OmegahConnManager::getNeighborElementBlock' should not have been called.\n");
    return emptyVec;
  }

  // Albany should not use neighbors, so always false.
  bool hasAssociatedNeighbors() const override { return false; }

  const std::vector<LO>& getAssociatedNeighbors(const LO& /* el */) const override
  {
    static std::vector<LO> ret;
    std::runtime_error(
        "Error! Albany does not use elements halos in the mesh, so the method\n"
        "       'STKConnManager::getAssociatedNeighbors' should not have been called.\n");

    return ret;
  }

  // Where element ielem start in the 1d connectivity array
  virtual int getConnectivityStart (const LO ielem) const = 0;

  // Get a mask vector (1=yes, 0=no) telling if each dof entity is contained in the given mesh part
  virtual std::vector<int> getConnectivityMask (const std::string& sub_part_name) const = 0;

  // Queries the dimension of a part
  virtual int part_dim (const std::string& part_name) const = 0;
  int part_dim () const { return part_dim(elem_block_name()); }
  
  /** Get array of Ownership enums associated to connectivity for a particular element, indicating whether the entity is owned by this rank
    *
    * \param[in] localElmtId Local element ID
    *
    * \returns Pointer to beginning of bools, with total size
    *          equal to <code>getConnectivitySize(localElmtId)</code>
    */
  virtual const Ownership* getOwnership(LO localElmtId) const = 0;

  const std::string& elem_block_name () const {
    TEUCHOS_TEST_FOR_EXCEPTION (m_elem_blocks_names.size()!=1, std::runtime_error,
        "[ConnManager::part_name] Error! Multiple part names.\n");

    return m_elem_blocks_names[0];
  }

  void getElementBlockIds(std::vector<std::string> & elementBlockIds) const override {
    elementBlockIds = m_elem_blocks_names;
  }

  shards::CellTopology get_topology () const {
    std::vector<shards::CellTopology> topologies;
    this->getElementBlockTopologies(topologies);
    TEUCHOS_TEST_FOR_EXCEPTION (topologies.size()!=1, std::runtime_error,
        "[ConnManager::get_topology] Error! Multiple topologies.\n");

    return topologies[0];
  }
protected:
  std::vector<std::string> m_elem_blocks_names;

  bool m_is_connectivity_built = false;
};

} // namespace Albany

#endif // ALBANY_CONN_MANAGER_HPP
