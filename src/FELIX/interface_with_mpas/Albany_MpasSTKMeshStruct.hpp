//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_MPAS_STKMESHSTRUCT_HPP
#define ALBANY_MPAS_STKMESHSTRUCT_HPP

#include "Albany_GenericSTKMeshStruct.hpp"



//extern void tetrasFromPrismStructured (long long int const* prismVertexMpasIds, long long int const* prismVertexGIds, long long int tetrasIdsOnPrism[][4]);
extern void tetrasFromPrismStructured (int const* prismVertexMpasIds, int const* prismVertexGIds, int tetrasIdsOnPrism[][4]);
extern void setBdFacesOnPrism (const std::vector<std::vector<std::vector<int> > >& prismStruct, const std::vector<int>& prismFaceIds, std::vector<int>& tetraPos, std::vector<int>& facePos);
extern void procsSharingVertex(const int vertex, std::vector<int>& procIds);

namespace Albany {

  class MpasSTKMeshStruct : public GenericSTKMeshStruct {

    public:

//	MpasSTKMeshStruct(const Teuchos::RCP<Teuchos::ParameterList>& params,
	//                                               const Teuchos::RCP<const Teuchos_Comm>& commT,
	  //                                             const std::vector<GO>& indexToTriangleID, const std::vector<int>& verticesOnTria, int nGlobalTriangles);

	MpasSTKMeshStruct(const Teuchos::RCP<Teuchos::ParameterList>& params,
	                                             const Teuchos::RCP<const Teuchos_Comm>& commT,
	                                             const std::vector<GO>& indexToTriangleID, const std::vector<int>& verticesOnTria, int nGlobalTriangles, int numLayers, int Ordering = 0);

	MpasSTKMeshStruct(const Teuchos::RCP<Teuchos::ParameterList>& params,
		                                         const Teuchos::RCP<const Teuchos_Comm>& commT,
		                                         const std::vector<GO>& indexToTriangleID, int nGlobalTriangles, int numLayers, int Ordering = 0);


    ~MpasSTKMeshStruct();

    void setFieldAndBulkData(
                      const Teuchos::RCP<const Teuchos_Comm>& commT,
                      const Teuchos::RCP<Teuchos::ParameterList>& params,
                      const unsigned int neq_,
                      const Albany::AbstractFieldContainer::FieldContainerRequirements& req,
                      const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                      const unsigned int worksetSize,
                      const std::map<std::string,Teuchos::RCP<Albany::StateInfoStruct> >& side_set_sis = {},
                      const std::map<std::string,AbstractFieldContainer::FieldContainerRequirements>& side_set_req = {}){};

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
		   const Teuchos::RCP<Albany::StateInfoStruct>& sis,
		   const std::vector<int>& indexToVertexID, const std::vector<double>& verticesCoords, const std::vector<bool>& isVertexBoundary, int nGlobalVertices,
		   const std::vector<int>& verticesOnTria,
		   const std::vector<bool>& isBoundaryEdge, const std::vector<int>& trianglesOnEdge, const std::vector<int>& trianglesPositionsOnEdge,
		   const std::vector<int>& verticesOnEdge,
		   const std::vector<int>& indexToEdgeID, int nGlobalEdges,
		   const std::vector<GO>& indexToTriangleID,
		   const std::vector<int>& dirichletNodesIds,
		   const std::vector<int>& floating2dLateralEdgesIds,
		   const unsigned int worksetSize,
		   int numLayers, int Ordering = 0);

    void constructMesh(
		   const Teuchos::RCP<const Teuchos_Comm>& commT,
		   const Teuchos::RCP<Teuchos::ParameterList>& params,
		   const unsigned int neq_,
		   const Albany::AbstractFieldContainer::FieldContainerRequirements& req,
		   const Teuchos::RCP<Albany::StateInfoStruct>& sis,
		   const std::vector<int>& indexToVertexID, const std::vector<int>& indexToMpasVertexID, const std::vector<double>& verticesCoords, const std::vector<bool>& isVertexBoundary, int nGlobalVertices,
		   const std::vector<int>& verticesOnTria,
		   const std::vector<bool>& isBoundaryEdge, const std::vector<int>& trianglesOnEdge, const std::vector<int>& trianglesPositionsOnEdge,
		   const std::vector<int>& verticesOnEdge,
		   const std::vector<int>& indexToEdgeID, int nGlobalEdges,
		   const std::vector<GO>& indexToTriangleID,
		   const std::vector<int>& dirichletNodesIds,
		   const std::vector<int>& floating2dLateralEdgesIds,
		   const unsigned int worksetSize,
		   int numLayers, int Ordering = 0);


    const bool getInterleavedOrdering() const {return this->interleavedOrdering;}

    private:

//    inline void tetrasFromPrismStructured (int const* prismVertexLIds, int const* prismVertexGIds, int tetrasIdsOnPrism[][4]);
 //   inline voidsetBdFacesOnPrism (const std::vector<std::vector<std::vector<int> > >& prismStruct, const std::vector<int>& prismFaceIds, std::vector<int>& tetraPos, std::vector<int>& facePos);

    Teuchos::RCP<const Teuchos::ParameterList>
      getValidDiscretizationParameters() const;

    Teuchos::RCP<Teuchos::FancyOStream> out;
    bool periodic;
    int NumEles; //number of elements
    bool hasRestartSol;
    double restartTime;
    Teuchos::RCP<Tpetra_Map> elem_mapT; //element map

    protected:

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

/*

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




	void MpasSTKMeshStruct::setBdFacesOnPrism (const std::vector<std::vector<std::vector<int> > >& prismStruct, const std::vector<int>& prismFaceIds, std::vector<int>& tetraPos, std::vector<int>& facePos)
	{
		int numTriaFaces = prismFaceIds.size() - 2;
		tetraPos.assign(numTriaFaces,-1);
		facePos.assign(numTriaFaces,-1);


		for (int iTetra (0), k (0); (iTetra < 3 && k < numTriaFaces); iTetra++)
		{
			bool found;
			for (int jFaceLocalId = 0; jFaceLocalId < 4; jFaceLocalId++ )
			{
				found = true;
				for (int ip (0); ip < 3 && found; ip++)
				{
					int localId = prismStruct[iTetra][jFaceLocalId][ip];
					int j = 0;
					found = false;
					while ( (j < prismFaceIds.size()) && !found )
					{
						found = (localId == prismFaceIds[j]);
						j++;
					}
				}
				if (found)
				{
					tetraPos[k] = iTetra;
					facePos[k] = jFaceLocalId;
					k += found;
					break;
				}
			}
		}
	}
*/
}

#endif
