//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_CONN_MANAGER_HPP
#define ALBANY_CONN_MANAGER_HPP

// #include "Albany_DiscretizationUtils.hpp"
#include "Albany_ScalarOrdinalTypes.hpp"

#include "Panzer_ConnManager.hpp"

#include <vector>

namespace Albany {

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

  // Returns whether input part name is topologically contained in the
  // parts where this ConnManager is defined.
  virtual bool contains (const std::string& sub_part_name) const = 0;

  // Return true if the $subcell_pos-th subcell of dimension $subcell_dim in
  // local element $ielem belongs to sub part $sub_part_name
  virtual bool belongs (const std::string& sub_part_name,
                        const LO ielem, const int subcell_dim, const int subcell_pos) const = 0;

  // Queries the dimension of a part
  virtual int part_dim (const std::string& part_name) const = 0;

  const std::string& elem_block_name () const {
    TEUCHOS_TEST_FOR_EXCEPTION (m_elem_blocks_names.size()!=1, std::runtime_error,
        "[ConnManager::part_name] Error! Multiple part names.\n");

    return m_elem_blocks_names[0];
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
};

} // namespace Albany

#endif // ALBANY_CONN_MANAGER_HPP
