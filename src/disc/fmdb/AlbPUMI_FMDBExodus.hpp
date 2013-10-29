//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FMDB_EXODUS_HPP
#define FMDB_EXODUS_HPP

#include "Teuchos_RCP.hpp"
#include "Epetra_Comm.h"

#include "pumi_mesh.h"

namespace AlbPUMI {

class FMDBExodus {

  public:

    FMDBExodus(const std::string& outputFile, pMeshMdl mesh, const Teuchos::RCP<const Epetra_Comm>& comm_);

    ~FMDBExodus();

    void writeFile(const double time);

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

