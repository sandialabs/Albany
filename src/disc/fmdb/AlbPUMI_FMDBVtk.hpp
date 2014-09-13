//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/12/14: no Epetra except Epetra_Comm.
#ifndef FMDB_VTK_HPP
#define FMDB_VTK_HPP

#include "Teuchos_RCP.hpp"
#include "Epetra_Comm.h"
#include "AlbPUMI_FMDBMeshStruct.hpp"

namespace AlbPUMI {

class FMDBVtk {

  public:

    FMDBVtk(FMDBMeshStruct& meshStruct, const Teuchos::RCP<const Epetra_Comm>& comm_);

    ~FMDBVtk();

    void writeFile(const double time);
    void setFileName(const std::string& fname){ outputFileName = fname; }

  private:

    std::ofstream vtu_collection_file;

    apf::Mesh* mesh;

    bool doCollection;
    std::string outputFileName;

    int remeshFileIndex;


    //! Epetra communicator
    Teuchos::RCP<const Epetra_Comm> comm;

};

}

#endif

