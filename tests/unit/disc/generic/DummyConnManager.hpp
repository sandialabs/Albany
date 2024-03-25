//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_DUMMY_CONN_MANAGER_HPP
#define ALBANY_DUMMY_CONN_MANAGER_HPP

#include "Albany_ConnManager.hpp"
#include <Shards_CellTopology.hpp>

#include <vector>
#include <map>

namespace Albany {

class DummyConnManager : public ConnManager {
public:
  DummyConnManager (const shards::CellTopology& topo);

  ~DummyConnManager() = default;

  using ConnManager::getElementsInBlock;
  std::vector<GO>
  getElementsInBlock (const std::string& elem_block_name) const override;

  void buildConnectivity(const panzer::FieldPattern & fp) override;

  Teuchos::RCP<panzer::ConnManager> noConnectivityClone() const override;

  const GO * getConnectivity(LO localElmtId) const override
  {
    return nullptr;
  }

  const Ownership* getOwnership(LO localElmtId) const override
  {
    return nullptr;
  }

  std::vector<int> getConnectivityMask (const std::string& sub_part_name) const override;

  int getConnectivityStart(const LO /* localElmtId */) const override
  {
    return 0;
  }

  LO getConnectivitySize(LO localElmtId) const override
  {
    return 0;
  }

  std::string getBlockId(LO localElmtId) const override
  {
    return m_elem_blocks_names[0];
  }

  std::size_t numElementBlocks() const override
  {
    return 1;
  }

  void getElementBlockIds(std::vector<std::string> & /* elementBlockIds */) const override
  {
  }

  void getElementBlockTopologies(std::vector<shards::CellTopology> & /* elementBlockTopologies */) const override
  {
  }

  const std::vector<LO> & getElementBlock(const std::string & /* elem_block_name */) const override
  {
    static std::vector<LO> v;
    return v;
  }

  int getOwnedElementCount() const
  {
    return 0;
  }

  // // Queries the dimension of a part
  int part_dim (const std::string& part_name) const override
  {
    return m_topo.getDimension();
  }

protected:

  shards::CellTopology m_topo;
};

} // namespace Albany

#endif // ALBANY_DUMMY_CONN_MANAGER_HPP
