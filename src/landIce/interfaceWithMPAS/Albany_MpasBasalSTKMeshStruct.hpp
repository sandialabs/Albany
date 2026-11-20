//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_MPAS_BASAL_STK_MESH_STRUCT_HPP
#define ALBANY_MPAS_BASAL_STK_MESH_STRUCT_HPP

#include "Albany_GenericSTKMeshStruct.hpp"

#include <map>
#include <string>
#include <vector>

namespace Albany {

// The 2D triangulation that MPAS passes to Albany (the FE mesh dual to the MPAS Voronoi mesh), built
// from the distributed arrays of velocity_solver_extrude_3d_grid. It is the basal mesh of the logically
// extruded mesh used by the MPAS interface.
//
//  - Nodes are the local vertices (MPAS cells), with STK ids indexToVertexID, shared with the ranks in
//    procsSharingVertices and owned by the rank in vertexProcIDs (MPAS's ownership, no rebalance).
//  - Elements are the local triangles (MPAS vertices), with STK ids indexToTriangleID, owned by this
//    rank, with nodes in the order of verticesOnTria.
//  - Side sets: "lateralside", with all the boundary edges, plus the side sets given in sideSets
//    (local indices of boundary edges). Sides have STK ids indexToEdgeID.
//  - Node sets: the ones given in nodeSets (local vertex indices).
// The names in nodeSets and sideSets must be the same on all ranks (the lists can be empty).
// The input arrays are copied, since the mesh is built (setBulkData) after the constructor returns.
class MpasBasalSTKMeshStruct : public GenericSTKMeshStruct
{
public:

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
                          const int numParams);

  ~MpasBasalSTKMeshStruct () = default;

  void setBulkData (const Teuchos::RCP<const Teuchos_Comm>& comm) override;

  // The initial guess is passed to the application directly, so there is no restart solution
  bool hasRestartSolution () const override { return false; }
  double restartDataTime () const override { return -1.0; }

private:

  std::vector<int>                   m_indexToVertexID;
  std::vector<int>                   m_vertexProcIDs;
  std::vector<double>                m_verticesCoords;
  std::vector<std::vector<int>>      m_procsSharingVertices;
  std::vector<int>                   m_indexToTriangleID;
  std::vector<int>                   m_verticesOnTria;
  std::vector<int>                   m_indexToEdgeID;
  std::vector<bool>                  m_isBoundaryEdge;
  std::vector<int>                   m_trianglesOnEdge;
  std::vector<int>                   m_verticesOnEdge;
  std::map<std::string,std::vector<int>> m_nodeSets;
  std::map<std::string,std::vector<int>> m_sideSets;
};

} // namespace Albany

#endif // ALBANY_MPAS_BASAL_STK_MESH_STRUCT_HPP
