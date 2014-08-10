//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FMDB_EXODUS_HPP
#define FMDB_EXODUS_HPP

#include "Teuchos_RCP.hpp"
#include "Epetra_Comm.h"
#include "AlbPUMI_FMDBMeshStruct.hpp"

namespace AlbPUMI {

class FMDBExodus {

  public:

    FMDBExodus(FMDBMeshStruct& meshStruct, const Teuchos::RCP<const Epetra_Comm>& comm_);

    ~FMDBExodus();

    void write(const char* filename, const double time);
    void writeFile(const double time);

    void setFileName(const std::string& fname){ outputFileName = fname; }

  private:
    apf::Mesh2* mesh;
    apf::StkModels* sets_p;
    std::string outputFileName;
};

}

#endif

