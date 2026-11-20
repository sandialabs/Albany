//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_MpasBasalSTKMeshStruct.hpp"
#include "Albany_DiscretizationUtils.hpp"

#include <Shards_BasicTopologies.hpp>

#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/Selector.hpp>

#ifdef ALBANY_SEACAS
#include <stk_io/IossBridge.hpp>
#endif

#include <set>

namespace Albany
{

MpasBasalSTKMeshStruct::
MpasBasalSTKMeshStruct (const Teuchos::RCP<Teuchos::ParameterList>& params,
                        const Teuchos::RCP<const Teuchos_Comm>& comm,
                        const std::vector<int>& indexToVertexID,
                        const std::vector<int>& vertexProcIDs,
                        const std::vector<double>& verticesCoords,
                        const std::vector<std::vector<int>>& procsSharingVertices,
                        const std::vector<int>& indexToTriangleID,
                        const std::vector<int>& verticesOnTria,
                        const std::vector<int>& indexToEdgeID,
                        const std::vector<bool>& isBoundaryEdge,
                        const std::vector<int>& trianglesOnEdge,
                        const std::vector<int>& verticesOnEdge,
                        const std::map<std::string,std::vector<int>>& nodeSets,
                        const std::map<std::string,std::vector<int>>& sideSets,
                        const int numParams)
 : GenericSTKMeshStruct(params, 2, numParams)
 , m_indexToVertexID(indexToVertexID)
 , m_vertexProcIDs(vertexProcIDs)
 , m_verticesCoords(verticesCoords)
 , m_procsSharingVertices(procsSharingVertices)
 , m_indexToTriangleID(indexToTriangleID)
 , m_verticesOnTria(verticesOnTria)
 , m_indexToEdgeID(indexToEdgeID)
 , m_isBoundaryEdge(isBoundaryEdge)
 , m_trianglesOnEdge(trianglesOnEdge)
 , m_verticesOnEdge(verticesOnEdge)
 , m_nodeSets(nodeSets)
 , m_sideSets(sideSets)
{
  const int nV = m_indexToVertexID.size();
  const int nT = m_indexToTriangleID.size();
  const int nE = m_indexToEdgeID.size();
  TEUCHOS_TEST_FOR_EXCEPTION (
      static_cast<int>(m_vertexProcIDs.size())!=nV || static_cast<int>(m_verticesCoords.size())!=3*nV ||
      static_cast<int>(m_procsSharingVertices.size())!=nV || static_cast<int>(m_verticesOnTria.size())!=3*nT ||
      static_cast<int>(m_isBoundaryEdge.size())!=nE || static_cast<int>(m_trianglesOnEdge.size())!=2*nE ||
      static_cast<int>(m_verticesOnEdge.size())!=2*nE, std::invalid_argument,
      "[MpasBasalSTKMeshStruct] Error! Inconsistent sizes of the input arrays.\n");
  TEUCHOS_TEST_FOR_EXCEPTION (m_sideSets.count("lateralside")==1, std::invalid_argument,
      "[MpasBasalSTKMeshStruct] Error! The side set 'lateralside' is built by the mesh (all the boundary edges).\n");

  constexpr auto NODE_RANK = stk::topology::NODE_RANK;
  const     auto SIDE_RANK = metaData->side_rank();

  // Element block
  const std::string ebn = "Element Block 0";
  partVec.push_back(&metaData->declare_part_with_topology(ebn, stk::topology::TRI_3_2D));
  shards::CellTopology shards_ctd = stk::mesh::get_cell_topology(stk::topology::TRI_3_2D);
  this->addElementBlockInfo(0, ebn, partVec[0], shards_ctd);
#ifdef ALBANY_SEACAS
  stk::io::put_io_part_attribute(*partVec[0]);
#endif

  // Node sets
  std::vector<std::string> nsNames;
  for (const auto& ns : m_nodeSets) {
    nsNames.push_back(ns.first);
    nsPartVec[ns.first] = &metaData->declare_part(ns.first, NODE_RANK);
#ifdef ALBANY_SEACAS
    stk::io::put_io_part_attribute(*nsPartVec[ns.first]);
#endif
    for (int ib : ns.second) {
      TEUCHOS_TEST_FOR_EXCEPTION (ib<0 || ib>=nV, std::invalid_argument,
          "[MpasBasalSTKMeshStruct] Error! Invalid vertex index " << ib << " in node set '" << ns.first << "'.\n");
    }
  }

  // Side sets
  std::vector<std::string> ssNames (1,"lateralside");
  for (const auto& ss : m_sideSets) {
    ssNames.push_back(ss.first);
    for (int ie : ss.second) {
      TEUCHOS_TEST_FOR_EXCEPTION (ie<0 || ie>=nE || !m_isBoundaryEdge[ie], std::invalid_argument,
          "[MpasBasalSTKMeshStruct] Error! Side set '" << ss.first << "' contains edge index " << ie
          << ", which is not a local boundary edge.\n");
    }
  }
  for (const auto& ssn : ssNames) {
    ssPartVec[ssn] = &metaData->declare_part(ssn, SIDE_RANK);
    stk::mesh::set_topology(*ssPartVec[ssn], stk::topology::LINE_2);
#ifdef ALBANY_SEACAS
    stk::io::put_io_part_attribute(*ssPartVec[ssn]);
#endif
  }

  numDim = 2;
  const int worksetSizeMax = params->get<int>("Workset Size", DEFAULT_WORKSET_SIZE);
  const int worksetSize = computeWorksetSize(worksetSizeMax, nT);

  const CellTopologyData& ctd = *shards_ctd.getCellTopologyData();
  this->meshSpecs[0] = Teuchos::rcp(
      new MeshSpecsStruct(MeshType::Unstructured, ctd, numDim, nsNames, ssNames,
                          worksetSize, ebn, ebNameToIndex));

  // Create a mesh specs object for EACH side set
  this->initializeSideSetMeshSpecs(comm);
}

void MpasBasalSTKMeshStruct::
setBulkData (const Teuchos::RCP<const Teuchos_Comm>& comm)
{
  constexpr auto NODE_RANK = stk::topology::NODE_RANK;
  constexpr auto ELEM_RANK = stk::topology::ELEM_RANK;
  const     auto SIDE_RANK = metaData->side_rank();

  const int nV = m_indexToVertexID.size();
  const int nT = m_indexToTriangleID.size();
  const int nE = m_indexToEdgeID.size();

  metaData->commit();

  bulkData->modification_begin();

  auto proc_rank_field = fieldContainer->getProcRankField();
  auto coordinates_field = fieldContainer->getCoordinatesField();

  // Nodes
  const stk::mesh::PartVector noParts;
  std::vector<stk::mesh::Entity> nodes(nV);
  for (int ib=0; ib<nV; ++ib) {
    nodes[ib] = bulkData->declare_node(m_indexToVertexID[ib], noParts);
    for (int proc : m_procsSharingVertices[ib]) {
      bulkData->add_node_sharing(nodes[ib], proc);
    }
    double* coord = stk::mesh::field_data(*coordinates_field, nodes[ib]);
    coord[0] = m_verticesCoords[3*ib];
    coord[1] = m_verticesCoords[3*ib+1];
  }

  // Node sets
  stk::mesh::PartVector singlePartVec(1);
  for (const auto& ns : m_nodeSets) {
    singlePartVec[0] = nsPartVec[ns.first];
    for (int ib : ns.second) {
      bulkData->change_entity_parts(nodes[ib], singlePartVec);
    }
  }

  // Elements, with nodes ordered as in verticesOnTria
  singlePartVec[0] = partVec[0];
  for (int it=0; it<nT; ++it) {
    stk::mesh::Entity elem = bulkData->declare_element(m_indexToTriangleID[it], singlePartVec);
    for (int j=0; j<3; ++j) {
      bulkData->declare_relation(elem, nodes[m_verticesOnTria[3*it+j]], j);
    }
    if (proc_rank_field) {
      int* p_rank = stk::mesh::field_data(*proc_rank_field, elem);
      if (p_rank)
        p_rank[0] = comm->getRank();
    }
  }

  // Sides: all the boundary edges are in lateralside, and possibly in other side sets
  std::vector<std::set<std::string>> edgeSideSets(nE);
  for (const auto& ss : m_sideSets) {
    for (int ie : ss.second) {
      edgeSideSets[ie].insert(ss.first);
    }
  }
  const auto& ctd = this->meshSpecs[0]->ctd;
  for (int ie=0; ie<nE; ++ie) {
    if (!m_isBoundaryEdge[ie]) continue;

    stk::mesh::PartVector sideParts (1,ssPartVec["lateralside"]);
    for (const auto& ssn : edgeSideSets[ie]) {
      sideParts.push_back(ssPartVec[ssn]);
    }

    // Find the position of the edge in its (only) triangle. Side j of a Triangle<3> joins nodes j and (j+1)%3
    const int it = m_trianglesOnEdge[2*ie];
    TEUCHOS_TEST_FOR_EXCEPTION (it<0 || it>=nT, std::runtime_error,
        "[MpasBasalSTKMeshStruct] Error! The boundary edge " << m_indexToEdgeID[ie] << " has no local triangle.\n");
    const int v0 = m_verticesOnEdge[2*ie], v1 = m_verticesOnEdge[2*ie+1];
    int side_pos = -1;
    for (int j=0; j<3 && side_pos<0; ++j) {
      const int a = m_verticesOnTria[3*it+j], b = m_verticesOnTria[3*it+(j+1)%3];
      if ((a==v0 && b==v1) || (a==v1 && b==v0)) side_pos = j;
    }
    TEUCHOS_TEST_FOR_EXCEPTION (side_pos<0, std::runtime_error,
        "[MpasBasalSTKMeshStruct] Error! The edge " << m_indexToEdgeID[ie] << " is not an edge of its triangle "
        << m_indexToTriangleID[it] << ".\n");

    stk::mesh::Entity elem = bulkData->get_entity(ELEM_RANK, m_indexToTriangleID[it]);
    stk::mesh::Entity side = bulkData->declare_entity(SIDE_RANK, m_indexToEdgeID[ie], sideParts);
    bulkData->declare_relation(elem, side, side_pos);
    stk::mesh::Entity const* rel_elemNodes = bulkData->begin_nodes(elem);
    for (int k=0; k<2; ++k) {
      bulkData->declare_relation(side, rel_elemNodes[ctd.side[side_pos].node[k]], k);
    }
  }

  bulkData->modification_end();

  // Change the ownership of the nodes to reflect the MPAS one
  stk::mesh::EntityProcVec node_to_proc;
  for (int ib=0; ib<nV; ++ib) {
    stk::mesh::Entity node = bulkData->get_entity(NODE_RANK, m_indexToVertexID[ib]);
    const int procID = m_vertexProcIDs[ib];
    if (bulkData->bucket(node).owned() && (procID != bulkData->parallel_rank())) {
      node_to_proc.push_back(std::make_pair(node, procID));
    }
  }
  bulkData->change_entity_owner(node_to_proc);

  this->setDefaultCoordinates3d();

  this->loadRequiredInputFields(comm);

  m_bulk_data_set = true;
}

} // namespace Albany
