//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_DUMMY_CONN_MANAGER_HPP
#define ALBANY_DUMMY_CONN_MANAGER_HPP

#include "DummyMesh.hpp"
#include "Albany_ConnManager.hpp"

#include <Shards_CellTopology.hpp>

#include <vector>
#include <map>

namespace Albany {

class DummyConnManager : public ConnManager {
public:
  DummyConnManager (const Teuchos::RCP<const DummyMesh>& mesh);

  ~DummyConnManager() = default;

  using ConnManager::getElementsInBlock;
  std::vector<GO>
  getElementsInBlock (const std::string& elem_block_name) const override;

  void buildConnectivity(const panzer::FieldPattern & fp) override;

  Teuchos::RCP<panzer::ConnManager> noConnectivityClone() const override;

  const GO * getConnectivity(LO ielem) const override
  {
    TEUCHOS_TEST_FOR_EXCEPTION (not is_connectivity_built(), std::logic_error,
        "Error! Cannot call getConnectivity before connectivity is build.\n");
    return m_connectivity.data() + ielem*m_num_dofs_per_elem;
  }

  const Ownership* getOwnership(LO ielem) const override
  {
    TEUCHOS_TEST_FOR_EXCEPTION (not is_connectivity_built(), std::logic_error,
        "Error! Cannot call getOwnership before connectivity is build.\n");
    return m_ownership.data() + ielem*m_num_dofs_per_elem;
  }

  std::vector<int> getConnectivityMask (const std::string& /* sub_part_name */) const override
  {
    throw std::runtime_error("DummyConnManager::getConnectivityMask is not implemented");
    return {};
  }

  int getConnectivityStart(const LO ielem) const override
  {
    TEUCHOS_TEST_FOR_EXCEPTION (not is_connectivity_built(), std::logic_error,
        "Error! Cannot call getConnectivityStart before connectivity is build.\n");
    return ielem*m_num_dofs_per_elem;
  }

  LO getConnectivitySize(LO /* localElmtId */) const override
  {
    TEUCHOS_TEST_FOR_EXCEPTION (not is_connectivity_built(), std::logic_error,
        "Error! Cannot call getConnectivitySize before connectivity is build.\n");
    return m_num_dofs_per_elem;
  }

  std::string getBlockId(LO /* localElmtId */) const override
  {
    return m_elem_blocks_names[0];
  }

  std::size_t numElementBlocks() const override
  {
    return 1;
  }

  void getElementBlockIds(std::vector<std::string> & elementBlockIds) const override
  {
    const auto& ms = m_mesh->meshSpecs[0];
    elementBlockIds.resize(1,ms->ebName);
  }

  void getElementBlockTopologies(std::vector<shards::CellTopology> & elementBlockTopologies) const override
  {
    const auto& ms = m_mesh->meshSpecs[0];
    elementBlockTopologies.resize(1,shards::CellTopology(&ms->ctd));
  }

  const std::vector<LO> & getElementBlock(const std::string & /* elem_block_name */) const override
  {
    return m_elem_lids;
  }

  int getOwnedElementCount() const
  {
    return m_mesh->get_num_local_elements();
  }

  // // Queries the dimension of a part
  int part_dim (const std::string& part_name) const override
  {
    const auto& ms = m_mesh->meshSpecs[0];
    TEUCHOS_TEST_FOR_EXCEPTION (part_name!=ms->ebName, std::invalid_argument,
        "Input part name does not match mesh eb name");
    return ms->numDim;
  }

protected:

  std::vector<GO>           m_connectivity;
  std::vector<Ownership>    m_ownership;
  std::vector<LO>           m_elem_lids;

  int   m_num_dofs_per_elem = 0;

  Teuchos::RCP<const DummyMesh> m_mesh;
};

} // namespace Albany

#endif // ALBANY_DUMMY_CONN_MANAGER_HPP
