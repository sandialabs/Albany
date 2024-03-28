//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "DummyConnManager.hpp"

#include <Panzer_FieldPattern.hpp>

#include <numeric>

namespace Albany {

DummyConnManager::
DummyConnManager (const Teuchos::RCP<const DummyMesh>& mesh)
 : m_mesh(mesh)
{
  // Init members of base class
  const auto& ms = m_mesh->meshSpecs[0];
  m_elem_blocks_names.resize(1,ms->ebName);

  int ne = m_mesh->get_num_local_elements();
  m_elem_lids.resize(ne);
  std::iota(m_elem_lids.begin(),m_elem_lids.end(),0);
}

Teuchos::RCP<panzer::ConnManager>
DummyConnManager::noConnectivityClone() const
{
  return Teuchos::rcp(new DummyConnManager(m_mesh));
}

std::vector<GO>
DummyConnManager::getElementsInBlock (const std::string& blockId) const
{
  TEUCHOS_TEST_FOR_EXCEPTION (blockId!=elem_block_name(), std::invalid_argument,
      "Input part name does not match mesh eb name");
  return m_mesh->my_elems();
}

void
DummyConnManager::buildConnectivity(const panzer::FieldPattern& fp)
{
  TEUCHOS_TEST_FOR_EXCEPTION (m_is_connectivity_built, std::logic_error,
      "[DummyConnManager::buildConnectivity] Connectivity was already built.\n");

  const auto& topo = get_topology();
  const int ne = m_mesh->get_num_local_elements();
  printf("building dummy conn mgr, topo=%s\n",topo.getName());

  const int patternDim = fp.getDimension();
  int faceIdCnt, edgeIdCnt, nodeIdCnt, cellIdCnt;
  switch(patternDim) {
    case 3:
      faceIdCnt = fp.getSubcellIndices(2,0).size();
      m_num_dofs_per_elem += faceIdCnt*topo.getFaceCount();
    case 2:
      edgeIdCnt = fp.getSubcellIndices(1,0).size();
      m_num_dofs_per_elem += edgeIdCnt*topo.getEdgeCount();
    case 1:
      nodeIdCnt = fp.getSubcellIndices(0,0).size();
      m_num_dofs_per_elem += nodeIdCnt*topo.getNodeCount();
    case 0:
      cellIdCnt = fp.getSubcellIndices(patternDim,0).size();
      m_num_dofs_per_elem += cellIdCnt;
      break;
    default:
       TEUCHOS_ASSERT(false);
  };

  GO nodeOffset = 0;
  GO edgeOffset = nodeOffset + m_mesh->num_global_nodes();
  GO faceOffset = edgeOffset + m_mesh->num_global_edges();
  GO cellOffset = faceOffset + m_mesh->num_global_faces();
  for (int ie=0; ie<ne; ++ie) {
    const GO gelem = m_mesh->my_elems()[ie];
    std::cout << "ie=" << ie << "\n";
    if (topo.getDimension()>0) {
      // Add node indices
      const auto& nodes = m_mesh->elem2node().at(ie);
      for (int inode=0; inode<topo.getNodeCount(); ++inode) {
        for (int id=0; id<nodeIdCnt; ++id) {
          m_connectivity.push_back(nodeOffset + nodes[inode]*nodeIdCnt + id);
        }
      }
    }

    if (topo.getDimension()>1) {
      // Add edge indices
      const auto& edges = m_mesh->elem2edge().at(ie);
      for (int iedge=0; iedge<topo.getEdgeCount(); ++iedge) {
        for (int id=0; id<edgeIdCnt; ++id) {
          m_connectivity.push_back(edgeOffset + edges[iedge]*edgeIdCnt + id);
        }
      }
    }

    if (topo.getDimension()>2) {
      // Add face indices
      const auto& faces = m_mesh->elem2face().at(ie);
      for (int iface=0; iface<topo.getFaceCount(); ++iface) {
        for (int id=0; id<edgeIdCnt; ++id) {
          m_connectivity.push_back(faceOffset + faces[iface]*edgeIdCnt + id);
        }
      }
    }

    // Add cell indices
    for (int id=0; id<cellIdCnt; ++id) {
      m_connectivity.push_back(cellOffset + gelem*cellIdCnt + id);
    }
  }

  m_ownership.resize(m_connectivity.size(),Ownership::Owned);

  m_is_connectivity_built = true;
}

} // namespace Albany
