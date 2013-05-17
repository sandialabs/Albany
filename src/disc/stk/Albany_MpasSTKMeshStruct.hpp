//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_MPAS_STKMESHSTRUCT_HPP
#define ALBANY_MPAS_STKMESHSTRUCT_HPP

#include "Albany_GenericSTKMeshStruct.hpp"


namespace Albany {

  class MpasSTKMeshStruct : public GenericSTKMeshStruct {

    public:

	MpasSTKMeshStruct(const Teuchos::RCP<Teuchos::ParameterList>& params,
	                                               const Teuchos::RCP<const Epetra_Comm>& comm,
	                                               const std::vector<int>& indexToTriangleID, const std::vector<int>& verticesOnTria, int nGlobalTriangles);

	MpasSTKMeshStruct(const Teuchos::RCP<Teuchos::ParameterList>& params,
	                                             const Teuchos::RCP<const Epetra_Comm>& comm,
	                                             const std::vector<int>& indexToTriangleID, const std::vector<int>& verticesOnTria, int nGlobalTriangles, int numLayers, int Ordering = 0);

	MpasSTKMeshStruct(const Teuchos::RCP<Teuchos::ParameterList>& params,
		                                         const Teuchos::RCP<const Epetra_Comm>& comm,
		                                         const std::vector<int>& indexToTriangleID, int nGlobalTriangles, int numLayers, int Ordering = 0);


    ~MpasSTKMeshStruct();

    void setFieldAndBulkData(
                      const Teuchos::RCP<const Epetra_Comm>& comm,
                      const Teuchos::RCP<Teuchos::ParameterList>& params,
                      const unsigned int neq_,
                      const Albany::AbstractFieldContainer::FieldContainerRequirements& req,
                      const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                      const unsigned int worksetSize){};

    //! Flag if solution has a restart values -- used in Init Cond
    bool hasRestartSolution() const {return false; }

    //! If restarting, convenience function to return restart data time
    double restartDataTime() const {return -1.;};


    void constructMesh(
            const Teuchos::RCP<const Epetra_Comm>& comm,
            const Teuchos::RCP<Teuchos::ParameterList>& params,
            const unsigned int neq_,
            const Albany::AbstractFieldContainer::FieldContainerRequirements& req,
            const Teuchos::RCP<Albany::StateInfoStruct>& sis,
            const std::vector<int>& indexToVertexID, const std::vector<double>& verticesCoords, const std::vector<bool>& isVertexBoundary, int nGlobalVertices,
            const std::vector<int>& verticesOnTria,
            const std::vector<bool>& isBoundaryEdge, const std::vector<int>& trianglesOnEdge, const std::vector<int>& trianglesPositionsOnEdge,
            const std::vector<int>& verticesOnEdge, const std::vector<int>& indexToEdgeID, int nGlobalEdges,
            const unsigned int worksetSize);

    void constructMesh(
		   const Teuchos::RCP<const Epetra_Comm>& comm,
		   const Teuchos::RCP<Teuchos::ParameterList>& params,
		   const unsigned int neq_,
		   const Albany::AbstractFieldContainer::FieldContainerRequirements& req,
		   const Teuchos::RCP<Albany::StateInfoStruct>& sis,
		   const std::vector<int>& indexToVertexID, const std::vector<double>& verticesCoords, const std::vector<bool>& isVertexBoundary, int nGlobalVertices,
		   const std::vector<int>& verticesOnTria,
		   const std::vector<bool>& isBoundaryEdge, const std::vector<int>& trianglesOnEdge, const std::vector<int>& trianglesPositionsOnEdge,
		   const std::vector<int>& verticesOnEdge,
		   const std::vector<int>& indexToEdgeID, int nGlobalEdges,
		   const std::vector<int>& indexToTriangleID,
		   const unsigned int worksetSize,
		   int numLayers, int Ordering = 0);

    void constructMesh(
		   const Teuchos::RCP<const Epetra_Comm>& comm,
		   const Teuchos::RCP<Teuchos::ParameterList>& params,
		   const unsigned int neq_,
		   const Albany::AbstractFieldContainer::FieldContainerRequirements& req,
		   const Teuchos::RCP<Albany::StateInfoStruct>& sis,
		   const std::vector<int>& indexToVertexID, const std::vector<int>& indexToMpasVertexID, const std::vector<double>& verticesCoords, const std::vector<bool>& isVertexBoundary, int nGlobalVertices,
		   const std::vector<int>& verticesOnTria,
		   const std::vector<bool>& isBoundaryEdge, const std::vector<int>& trianglesOnEdge, const std::vector<int>& trianglesPositionsOnEdge,
		   const std::vector<int>& verticesOnEdge,
		   const std::vector<int>& indexToEdgeID, int nGlobalEdges,
		   const std::vector<int>& indexToTriangleID,
		   const unsigned int worksetSize,
		   int numLayers, int Ordering = 0);

    inline void tetrasFromPrismStructured (int const* prismVertexLIds, int const* prismVertexGIds, int tetrasIdsOnPrism[][4]);

    private:

    Teuchos::RCP<const Teuchos::ParameterList>
      getValidDiscretizationParameters() const;

    Teuchos::RCP<Teuchos::FancyOStream> out;
    bool periodic;
    int NumEles; //number of elements
    Teuchos::RCP<Epetra_Map> elem_map; //element map 
    /*
    const std::vector<int>& indexToTriangleID;
    const std::vector<int>& verticesOnTria;
    const int nGlobalTriangles;
    const std::vector<int>& indexToVertexID;
    const std::vector<double>& verticesCoords;
    const std::vector<bool>& isVertexBoundary;
    const int nGlobalVertices;
    const std::vector<bool>& isBoundaryEdge;
    const std::vector<int>& trianglesOnEdge;
    const std::vector<int>& trianglesPositionsOnEdge;
    const std::vector<int>& verticesOnEdge;
    const std::vector<int>& indexToEdgeID;
    const int nGlobalEdges;
    const int numDim;
    const int numLayers;
    const int Ordering;
    */
  };



  inline void MpasSTKMeshStruct::tetrasFromPrismStructured (int const* prismVertexMpasIds, int const* prismVertexGIds, int tetrasIdsOnPrism[][4])
  {
      int PrismVerticesMap[6][6] = {{0, 1, 2, 3, 4, 5}, {1, 2, 0, 4, 5, 3}, {2, 0, 1, 5, 3, 4}, {3, 5, 4, 0, 2, 1}, {4, 3, 5, 1, 0, 2}, {5, 4, 3, 2, 1, 0}};

      int tetraOfPrism[2][3][4] = {{{0, 1, 2, 5}, {0, 1, 5, 4}, {0, 4, 5, 3}}, {{0, 1, 2, 4}, {0, 4, 2, 5}, {0, 4, 5, 3}}};

      int minIndex = std::min_element (prismVertexMpasIds, prismVertexMpasIds + 3) - prismVertexMpasIds;

      int v1 (prismVertexMpasIds[PrismVerticesMap[minIndex][1]]);
      int v2 (prismVertexMpasIds[PrismVerticesMap[minIndex][2]]);

      int prismType = v1  > v2;

      for (int iTetra = 0; iTetra < 3; iTetra++)
          for (int iVertex = 0; iVertex < 4; iVertex++)
          {
              tetrasIdsOnPrism[iTetra][iVertex] = prismVertexGIds[tetraOfPrism[prismType][iTetra][iVertex]];
          }

      // return;

      int reorderedPrismLIds[6];

      for (int ii = 0; ii < 6; ii++)
      {
          reorderedPrismLIds[ii] = prismVertexGIds[PrismVerticesMap[minIndex][ii]];
      }

      for (int iTetra = 0; iTetra < 3; iTetra++)
          for (int iVertex = 0; iVertex < 4; iVertex++)
          {
              tetrasIdsOnPrism[iTetra][iVertex] = reorderedPrismLIds[tetraOfPrism[prismType][iTetra][iVertex]];
          }
  }

}
#endif
