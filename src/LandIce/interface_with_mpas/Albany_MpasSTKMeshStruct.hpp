//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_MPAS_STK_MESH_STRUCT_HPP
#define ALBANY_MPAS_STK_MESH_STRUCT_HPP

#include "Albany_GenericSTKMeshStruct.hpp"

void procsSharingVertex(const int vertex, std::vector<int>& procIds);

namespace Albany {

class MpasSTKMeshStruct : public GenericSTKMeshStruct
{
public:

  MpasSTKMeshStruct(const Teuchos::RCP<Teuchos::ParameterList>& params,
      const Teuchos::RCP<const Teuchos_Comm>& commT,
      const std::vector<GO>& indexToTriangleID,
      int globalTrianglesStride, int numLayers, const int numParams, int Ordering = 0);


  ~MpasSTKMeshStruct( ) = default;

  void setFieldAndBulkData(
      const Teuchos::RCP<const Teuchos_Comm>& /* comm */,
      const Teuchos::RCP<Teuchos::ParameterList>& /* params */,
      const unsigned int /* neq_ */,
      const Albany::AbstractFieldContainer::FieldContainerRequirements& /* req */,
      const Teuchos::RCP<Albany::StateInfoStruct>& /* sis */,
      const unsigned int /* worksetSize */,
      const std::map<std::string,Teuchos::RCP<Albany::StateInfoStruct> >& /* side_set_sis */ = {},
      const std::map<std::string,AbstractFieldContainer::FieldContainerRequirements>& /* side_set_req */ = {})
  {
    // Do nothing
  }

  //! Flag if solution has a restart values -- used in Init Cond
  bool hasRestartSolution() const {return hasRestartSol; }

  void setHasRestartSolution(bool hasRestartSolution) {hasRestartSol = hasRestartSolution; }

  void setRestartDataTime(double restartT) {restartTime = restartT; }

  //! If restarting, convenience function to return restart data time
  double restartDataTime() const {return restartTime;}

  void constructMesh(
      const Teuchos::RCP<const Teuchos_Comm>& commT,
      const Teuchos::RCP<Teuchos::ParameterList>& params,
      const unsigned int neq_,
      const Albany::AbstractFieldContainer::FieldContainerRequirements& req,
      const Albany::StateManager& stateMgr,
      const std::vector<int>& indexToVertexID,
      const std::vector<int>& vertexProcIDs,
      const std::vector<double>& verticesCoords,
      int nGlobalVertices,
      const std::vector<int>& verticesOnTria,
      const std::vector<std::vector<int>>  procsSharingVertices,
      const std::vector<bool>& isBoundaryEdge,
      const std::vector<int>& trianglesOnEdge,
      const std::vector<int>& verticesOnEdge,
      const std::vector<int>& indexToEdgeID,
      int globalEdgesStride,
      const std::vector<GO>& indexToTriangleID,
      int globalTrianglesStride,
      const std::vector<int>& dirichletNodesIds,
      const std::vector<int>& floating2dLateralEdgesIds,
      const unsigned int worksetSize,
      int numLayers, int Ordering = 0);


  bool getInterleavedOrdering() const {return this->interleavedOrdering;}

  enum elemShapeType {Tetrahedron, Wedge};

private:

  Teuchos::RCP<const Teuchos::ParameterList> getValidDiscretizationParameters() const;
  elemShapeType ElemShape;

  Teuchos::RCP<Teuchos::FancyOStream> out;
  bool periodic;
  int NumEles; //number of elements
  bool hasRestartSol;
  double restartTime;
  Teuchos::RCP<const Thyra_SpmdVectorSpace> elem_vs; //element vector space
  LayeredMeshOrdering Ordering;

  int prismType(int const* prismVertexIds, int& minIndex);

  void tetrasFromPrismStructured( int const* prismVertexGIds, int tetrasIdsOnPrism[][4]);

  void setBdFacesOnPrism (const std::vector<std::vector<std::vector<int> > >& prismStruct, const std::vector<int>& prismFaceIds, std::vector<int>& tetraPos, std::vector<int>& facePos);
};

} // namespace Albany

#endif // ALBANY_MPAS_STK_MESH_STRUCT_HPP
