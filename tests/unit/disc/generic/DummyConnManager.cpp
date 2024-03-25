//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "DummyConnManager.hpp"

namespace Albany {

DummyConnManager::
DummyConnManager (const shards::CellTopology& topo)
 : m_topo(topo)
{
  // Init members of base class
  m_elem_blocks_names.resize(1,"eb0");

  // buildMaxEntityIds();

  // get element info from STK_Interface
  // object and build a local element mapping.
  // buildLocalElementMapping();
}

Teuchos::RCP<panzer::ConnManager>
DummyConnManager::noConnectivityClone() const
{
  return Teuchos::rcp(new DummyConnManager(m_topo));
}

std::vector<int>
DummyConnManager::getConnectivityMask (const std::string& sub_part_name) const
{
  return {};
}

std::vector<GO>
DummyConnManager::getElementsInBlock (const std::string& /* blockId */) const
{
  return {};
}

void DummyConnManager::buildConnectivity(const panzer::FieldPattern & /* fp */)
{
}

} // namespace Albany
