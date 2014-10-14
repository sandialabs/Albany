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

  class ExtrudedSTKMeshStruct : public GenericSTKMeshStruct {

    public:

    ExtrudedSTKMeshStruct(
                  const Teuchos::RCP<Teuchos::ParameterList>& params, 
                  const Teuchos::RCP<const Epetra_Comm>& epetra_comm);

    ~ExtrudedSTKMeshStruct();

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

    inline void computeMap();
    inline int prismType(long long int const* prismVertexMpasIds, int& minIndex);
    inline void tetrasFromPrismStructured (long long int const* prismVertexMpasIds, long long int const* prismVertexGIds, long long int tetrasIdsOnPrism[][4]);
    void read2DFileSerial(std::string &fname, Epetra_Vector& content, const Teuchos::RCP<const Epetra_Comm>& comm);
    void readFileSerial(std::string &fname, std::vector<Epetra_Vector>& contentVec, const Teuchos::RCP<const Epetra_Comm>& comm);
    void readFileSerial(std::string &fname, const Epetra_Map& map_serial, const Epetra_Map& map, const Epetra_Import& importOperator, std::vector<Epetra_Vector>& temperatureVec, std::vector<double>& zCoords, const Teuchos::RCP<const Epetra_Comm>& comm);


    Teuchos::RCP<const Teuchos::ParameterList>
      getValidDiscretizationParameters() const;

    Teuchos::RCP<Teuchos::FancyOStream> out;
    bool periodic;
    enum elemShapeType {Tetrahedron, Wedge, Hexahedron};
    elemShapeType ElemShape;
    int NumBaseElemeNodes;
    int NumNodes; //number of nodes
    int NumEles; //number of elements
    int NumBdEdges; //number of faces on basal boundary
  };



  int ExtrudedSTKMeshStruct::prismType(long long int const* prismVertexMpasIds, int& minIndex)
  {
    int PrismVerticesMap[6][6] = {{0, 1, 2, 3, 4, 5}, {1, 2, 0, 4, 5, 3}, {2, 0, 1, 5, 3, 4}, {3, 5, 4, 0, 2, 1}, {4, 3, 5, 1, 0, 2}, {5, 4, 3, 2, 1, 0}};
    minIndex = std::min_element (prismVertexMpasIds, prismVertexMpasIds + 3) - prismVertexMpasIds;

    int v1 (prismVertexMpasIds[PrismVerticesMap[minIndex][1]]);
    int v2 (prismVertexMpasIds[PrismVerticesMap[minIndex][2]]);

    return v1  > v2;
  }

   void ExtrudedSTKMeshStruct::tetrasFromPrismStructured (long long int const* prismVertexMpasIds, long long int const* prismVertexGIds, long long int tetrasIdsOnPrism[][4])
    {
        int PrismVerticesMap[6][6] = {{0, 1, 2, 3, 4, 5}, {1, 2, 0, 4, 5, 3}, {2, 0, 1, 5, 3, 4}, {3, 5, 4, 0, 2, 1}, {4, 3, 5, 1, 0, 2}, {5, 4, 3, 2, 1, 0}};

        int tetraOfPrism[2][3][4] = {{{0, 1, 2, 5}, {0, 1, 5, 4}, {0, 4, 5, 3}}, {{0, 1, 2, 4}, {0, 4, 2, 5}, {0, 4, 5, 3}}};

        int tetraAdjacentToPrismLateralFace[2][3][2] = {{{1, 2}, {0, 1}, {0, 2}}, {{0, 2}, {0, 1}, {1, 2}}};
        int tetraFaceIdOnPrismLateralFace[2][3][2] = {{{0, 0}, {1, 1}, {2, 2}}, {{0, 0}, {1, 1}, {2, 2}}};
        int tetraAdjacentToBottomFace = 0; //does not depend on type;
        int tetraAdjacentToUpperFace = 2; //does not depend on type;
        int tetraFaceIdOnBottomFace = 3; //does not depend on type;
        int tetraFaceIdOnUpperFace = 0; //does not depend on type;

        int minIndex;
        int prismType = this->prismType(prismVertexMpasIds, minIndex);

        long long int reorderedPrismLIds[6];

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


    void ExtrudedSTKMeshStruct::computeMap()
    {
      int PrismVerticesMap[6][6] = {{0, 1, 2, 3, 4, 5}, {1, 2, 0, 4, 5, 3}, {2, 0, 1, 5, 3, 4}, {3, 5, 4, 0, 2, 1}, {4, 3, 5, 1, 0, 2}, {5, 4, 3, 2, 1, 0}};

      int tetraOfPrism[2][3][4] = {{{0, 1, 2, 5}, {0, 1, 5, 4}, {0, 4, 5, 3}}, {{0, 1, 2, 4}, {0, 4, 2, 5}, {0, 4, 5, 3}}};

      int TetraFaces[4][3] = {{0 , 1 , 3}, {1 , 2 , 3}, {0 , 3 , 2}, {0 , 2 , 1}};

      int PrismFaces[5][4] = {{0 , 1 , 4 , 3}, {1 , 2 , 5 , 4}, {0 , 3 , 5 , 2}, {0 , 2 , 1 , -1}, {3 , 4 , 5, -1}};


      for(int pType=0; pType<2; ++pType){
        std::cout<< "pType: " << pType <<std::endl;
        for(int minIndex = 0; minIndex<6; ++minIndex){
          std::cout<< "mIndex: " << minIndex <<std::endl;
          for(int pFace = 0; pFace<5; ++pFace){
            for(int tFace =0; tFace<4; ++tFace){
              for(int iTetra =0; iTetra<3; ++iTetra){
                int count=0;
                for(int in =0; in<3; ++in){
                  int node=PrismVerticesMap[minIndex][tetraOfPrism[pType][iTetra][TetraFaces[tFace][in]]];
                  for(int i=0; i<4; ++i)
                    count += (node == PrismFaces[pFace][i]);
                }
                if(count == 3)
                  std::cout << pFace << " " << tFace << " " << iTetra << std::endl;
                }
              }
            }
          }
        std::cout<<std::endl;
        }
      std::cout<<std::endl;
      }
}
#endif
