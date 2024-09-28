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
  TEUCHOS_TEST_FOR_EXCEPTION (is_connectivity_built(), std::logic_error,
      "[DummyConnManager::buildConnectivity] Connectivity was already built.\n");

  const auto& topo = get_topology();
  const int ne = m_mesh->get_num_local_elements();

  const int patternDim = fp.getDimension();
  GO nodeDofCount = 0;
  GO edgeDofCount = 0;
  GO faceDofCount = 0;
  GO cellDofCount = 0;
  switch(patternDim) {
    case 3:
      for (int iface=0; iface<topo.getFaceCount(); ++iface) {
        m_num_dofs_per_elem += fp.getSubcellIndices(2,iface).size();
        faceDofCount += fp.getSubcellIndices(2,iface).size();
      }
    case 2:
      for (int iedge=0; iedge<topo.getEdgeCount(); ++iedge) {
        m_num_dofs_per_elem += fp.getSubcellIndices(1,iedge).size();
        edgeDofCount += fp.getSubcellIndices(1,iedge).size();
      }
    case 1:
      for (int inode=0; inode<topo.getNodeCount(); ++inode) {
        m_num_dofs_per_elem += fp.getSubcellIndices(0,inode).size();
        nodeDofCount += fp.getSubcellIndices(0,inode).size();
      }
    case 0:
      m_num_dofs_per_elem += fp.getSubcellIndices(patternDim,0).size();
      cellDofCount += fp.getSubcellIndices(patternDim,0).size();
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
    // std::cout << "dummy" << topo.getDimension() << "d, ie=" << ie << "\n";
    int start = m_connectivity.size();
    if (topo.getDimension()>0) {
      // Add node indices
      const auto& nodes = m_mesh->elem2node().at(ie);
      for (int inode=0; inode<topo.getNodeCount(); ++inode) {
        int nodeIdCnt = fp.getSubcellIndices(0,inode).size();
        for (int id=0; id<nodeIdCnt; ++id) {
          m_connectivity.push_back(nodeOffset + nodes[inode]*nodeIdCnt + id);
        }
      }
    }

    if (topo.getDimension()>1) {
      // Add edge indices
      const auto& edges = m_mesh->elem2edge().at(ie);
      for (int iedge=0; iedge<topo.getEdgeCount(); ++iedge) {
        int edgeIdCnt = fp.getSubcellIndices(1,iedge).size();
        for (int id=0; id<edgeIdCnt; ++id) {
          m_connectivity.push_back(edgeOffset + edges[iedge]*edgeIdCnt + id);
        }
      }
    }

    if (topo.getDimension()>2) {
      // Add face indices
      const auto& faces = m_mesh->elem2face().at(ie);
      for (int iface=0; iface<topo.getFaceCount(); ++iface) {
        int faceIdCnt = fp.getSubcellIndices(2,iface).size();
        for (int id=0; id<faceIdCnt; ++id) {
          m_connectivity.push_back(faceOffset + faces[iface]*faceIdCnt + id);
        }
      }
    }

    // Add cell indices
    int cellIdCnt = fp.getSubcellIndices(patternDim,0).size();
    for (int id=0; id<cellIdCnt; ++id) {
      m_connectivity.push_back(cellOffset + gelem*cellIdCnt + id);
    }

    int cnt = m_connectivity.size()-start;
    // for (int ii=0; ii<cnt; ++ii) {
    //   std::cout << " " << m_connectivity[start+ii];
    // }
    // std::cout << "\n";
  }

  m_ownership.resize(m_connectivity.size(),Ownership::Owned);
}

} // namespace Albany
