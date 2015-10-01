//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#ifndef ALBANY_ASCII_STKMESHSTRUCT_HPP
#define ALBANY_ASCII_STKMESHSTRUCT_HPP

#include "Albany_GenericSTKMeshStruct.hpp"

//#include <Ionit_Initializer.h>

namespace Albany {

  class AsciiSTKMeshStruct : public GenericSTKMeshStruct {

    public:

//Constructor for meshes read from ASCII file
    AsciiSTKMeshStruct(
                  const Teuchos::RCP<Teuchos::ParameterList>& params,
                  const Teuchos::RCP<const Teuchos_Comm>& commT);


    ~AsciiSTKMeshStruct();

    void setFieldAndBulkData(
                  const Teuchos::RCP<const Teuchos_Comm>& commT,
                  const Teuchos::RCP<Teuchos::ParameterList>& params,
                  const unsigned int neq_,
                  const AbstractFieldContainer::FieldContainerRequirements& req,
                  const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                  const unsigned int worksetSize);

    //! Flag if solution has a restart values -- used in Init Cond
    bool hasRestartSolution() const {return false; }

    //! If restarting, convenience function to return restart data time
    double restartDataTime() const {return -1.0; }

    //Is this necessary here?
//    bool getInterleavedOrdering() const {return this->interleavedOrdering;}

    private:
    //Ioss::Init::Initializer ioInit;

    Teuchos::RCP<const Teuchos::ParameterList>
      getValidDiscretizationParameters() const;

    Teuchos::RCP<Teuchos::FancyOStream> out;
    bool periodic;
    bool contigIDs; //boolean specifying if node / element / face IDs are contiguous; only relevant for 1 processor run
    int NumNodes; //number of nodes
    int NumEles; //number of elements
    int NumBasalFaces; //number of faces on basal boundary
    double (*xyz)[3]; //hard-coded for 3D for now
    double* sh;
    double* beta;
    Teuchos::Array<GO> globalElesID; //int array to define element map
    Teuchos::Array<GO> globalNodesID; //int array to define node map
    Teuchos::Array<GO> basalFacesID; //int array to define basal face map
    int (*eles)[8]; //hard-coded for 3D hexes for now
    double *flwa; //double array that gives value of flow factor
    double *temper; //double array that gives value of flow factor
    bool have_sh; // Does surface height data exist?
    bool have_bf; // Does basal face connectivity file exist?
    bool have_flwa; // Does flwa (flow factor) file exist?
    bool have_temp; // Does temperature file exist?
    bool have_beta; // Does beta (basal fraction) file exist?
    int (*bf)[5]; //hard-coded for 3D hexes for now (meaning boundary faces are quads)
    Teuchos::RCP<Tpetra_Map> elem_mapT; //element map
    Teuchos::RCP<Tpetra_Map> node_mapT; //node map
    Teuchos::RCP<Tpetra_Map> basal_face_mapT; //basalface map

    protected:
  };

}
#endif
