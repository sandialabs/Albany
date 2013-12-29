//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FMDB_VTK_HPP
#define FMDB_VTK_HPP

#include "Teuchos_RCP.hpp"
#include "Epetra_Comm.h"
#include "AlbPUMI_FMDBMeshStruct.hpp"

#include "pumi_mesh.h"

namespace AlbPUMI {

class FMDBVtk {

  public:

    FMDBVtk(FMDBMeshStruct& meshStruct, const Teuchos::RCP<const Epetra_Comm>& comm_);

    ~FMDBVtk();

    void writeFile(const double time);
    void setFileName(const std::string& fname){ outputFileName = fname; }

    void debugMeshWrite(const char* filename){

        std::cout << "VTK output format does not currently support debug mesh output" << std::endl;
        std::cout << "because integration point data is not supported." << std::endl;

    }

  private:

    std::ofstream vtu_collection_file;

    pMeshMdl mesh;

    bool doCollection;
    std::string outputFileName;

    int remeshFileIndex;


    //! Epetra communicator
    Teuchos::RCP<const Epetra_Comm> comm;

};

}

#endif

