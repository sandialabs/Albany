#include "Albany_SerialConnManager1d.hpp"

#include <Panzer_FieldAggPattern.hpp>
#include <Panzer_IntrepidFieldPattern.hpp>
#include <Intrepid2_TensorBasis.hpp>

namespace Albany {

SerialConnManager1d::
SerialConnManager1d (const int numElems)
 : m_num_elems (numElems)
{
  m_elem_lids.resize(numElems);
  m_elem_gids.resize(numElems);
  for (int i=0; i<numElems; ++i) {
    m_elem_lids[i] = m_elem_gids[i] = i;
  }

  m_elem_blocks_names = {"mesh1d"};
}

Teuchos::RCP<panzer::ConnManager>
SerialConnManager1d::noConnectivityClone() const
{
  return Teuchos::rcp(new SerialConnManager1d(m_num_elems));
}

void SerialConnManager1d::
getElementBlockTopologies(std::vector<shards::CellTopology> & elementBlockTopologies) const
{
  elementBlockTopologies.resize(1,shards::CellTopology(shards::getCellTopologyData<shards::Line<2>>()));
}

const std::vector<LO>&
SerialConnManager1d::getElementBlock(const std::string& blockId) const
{
  TEUCHOS_TEST_FOR_EXCEPTION (blockId!=m_elem_blocks_names[0],std::logic_error,
      "[SerialConnManager1d::getElementBlock] Error! Invalid elem block name: " + blockId + ".\n");

  return m_elem_lids;
}

int SerialConnManager1d::
getConnectivitySize (const LO /* ielem */) const
{
  TEUCHOS_TEST_FOR_EXCEPTION (not m_is_connectivity_built, std::logic_error,
      "Error! Cannot call getConnectivitySize before connectivity is build.\n");
  return m_ndofs_per_elem;
}

int SerialConnManager1d::
getConnectivityStart (const LO ielem) const
{
  TEUCHOS_TEST_FOR_EXCEPTION (not m_is_connectivity_built, std::logic_error,
      "Error! Cannot call getConnectivityStart before connectivity is build.\n");
  return ielem*m_ndofs_per_elem;
}

const GO* SerialConnManager1d::
getConnectivity (const LO ielem) const
{
  TEUCHOS_TEST_FOR_EXCEPTION (not m_is_connectivity_built, std::logic_error,
      "Error! Cannot call getConnectivity before connectivity is build.\n");
  return m_connectivity.data() + getConnectivityStart(ielem);
}

std::vector<int>
SerialConnManager1d::
getConnectivityMask (const std::string& /* sub_part_name */) const
{
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error,
      "Error! SerialConnManager1d does not support getConnectivityMask.\n");

  static std::vector<int> ret;
  return ret;
}

int SerialConnManager1d::
part_dim (const std::string& part_name) const
{
  TEUCHOS_TEST_FOR_EXCEPTION (part_name!=m_elem_blocks_names[0], std::runtime_error,
      "Error! Invalid part name '" + part_name + "' for SerialConnManager1d.\n");

  return 1;
}

const Ownership*
SerialConnManager1d::getOwnership(LO localElmtId) const
{
  return m_ownership.data() + getConnectivityStart(localElmtId);
}

void SerialConnManager1d::
buildConnectivity(const panzer::FieldPattern & fp)
{
  TEUCHOS_TEST_FOR_EXCEPTION (fp.getDimension()!=1, std::logic_error,
      "Error! SerialConnManager1d::buildConnectivity called with FieldPattern of dimension " << fp.getDimension() << "\n");

  auto num_node_ids = fp.getSubcellIndices(0,0).size();
  auto num_elem_ids = fp.getSubcellIndices(1,0).size();

  m_ndofs_per_elem = 2*num_node_ids + num_elem_ids;
  for (int ie=0; ie<m_num_elems; ++ie) {
    for (int i=0; i<num_node_ids; ++i) {
      m_connectivity.push_back(m_connectivity.size());
    }
    for (int i=0; i<num_node_ids; ++i) {
      m_connectivity.push_back(m_connectivity.size());
    }
    for (int i=0; i<num_elem_ids; ++i) {
      m_connectivity.push_back(m_connectivity.size());
    }
  }

  m_ownership.resize(m_num_elems*m_ndofs_per_elem,Ownership::Owned);

  m_is_connectivity_built = true;
}

} // namespace Albany
