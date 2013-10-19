//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef ALBANY_ASCII_STKMESH3DSTRUCT_HPP
#define ALBANY_ASCII_STKMESH3DSTRUCT_HPP

#include "Albany_GenericSTKMeshStruct.hpp"
#include <string>
#include <iostream>

//#include <Ionit_Initializer.h>

namespace Albany {

  class AsciiSTKMesh3D : public GenericSTKMeshStruct {

    public:

    AsciiSTKMesh3D(
                  const Teuchos::RCP<Teuchos::ParameterList>& params, 
                  const Teuchos::RCP<const Epetra_Comm>& epetra_comm);

    ~AsciiSTKMesh3D();

    void setFieldAndBulkData(
                  const Teuchos::RCP<const Epetra_Comm>& comm,
                  const Teuchos::RCP<Teuchos::ParameterList>& params,
                  const unsigned int neq_,
                  const AbstractFieldContainer::FieldContainerRequirements& req,
                  const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                  const unsigned int worksetSize);

    //! Flag if solution has a restart values -- used in Init Cond
    bool hasRestartSolution() const {return false; }

    //! If restarting, convenience function to return restart data time
    double restartDataTime() const {return -1.0; }

    private:
    //Ioss::Init::Initializer ioInit;

    Teuchos::RCP<Albany::GenericSTKMeshStruct> meshStruct2D;

    inline void tetrasFromPrismStructured (int const* prismVertexMpasIds, int const* prismVertexGIds, int tetrasIdsOnPrism[][4]);
    inline void setBdFacesOnPrism (const std::vector<std::vector<std::vector<int> > >& prismStruct, const std::vector<int>& prismFaceIds, std::vector<int>& tetraPos, std::vector<int>& facePos);
    void read2DFileSerial(std::string &fname, Epetra_Vector& content, const Teuchos::RCP<const Epetra_Comm>& comm);
    void readFileSerial(std::string &fname, std::vector<Epetra_Vector>& contentVec, const Teuchos::RCP<const Epetra_Comm>& comm);


    Teuchos::RCP<const Teuchos::ParameterList>
      getValidDiscretizationParameters() const;

    Teuchos::RCP<Teuchos::FancyOStream> out;
    bool periodic;
    int NumNodes; //number of nodes
    int NumEles; //number of elements
    int NumBdEdges; //number of faces on basal boundary
   // double (*xyz)[3]; //hard-coded for 3D for now
 ///   double* sh;
 //   int (*eles)[4]; //hard-coded for quads for now
 //   int (*be)[2]; //hard-coded for hexes for now (meaning boundary faces are quads)
 //   int (*bf)[4]; //hard-coded for hexes for now (meaning boundary faces are quads)
  };


   void AsciiSTKMesh3D::tetrasFromPrismStructured (int const* prismVertexMpasIds, int const* prismVertexGIds, int tetrasIdsOnPrism[][4])
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




    void AsciiSTKMesh3D::setBdFacesOnPrism (const std::vector<std::vector<std::vector<int> > >& prismStruct, const std::vector<int>& prismFaceIds, std::vector<int>& tetraPos, std::vector<int>& facePos)
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

}
#endif
