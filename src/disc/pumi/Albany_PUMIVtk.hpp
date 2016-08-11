//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_PUMIVTK_HPP
#define ALBANY_PUMIVTK_HPP

#include "Albany_PUMIOutput.hpp"

namespace Albany {

class PUMIVtk : public PUMIOutput {

  public:

    PUMIVtk(const Teuchos::RCP<APFMeshStruct>& meshStruct,
            const Teuchos::RCP<const Teuchos_Comm>& commT_);

    ~PUMIVtk();

    void writeFile(const double time);
    void setFileName(const std::string& fname){ outputFileName = fname; }

  private:

    void callAPFWrite(std::string const& path);

    std::fstream vtu_collection_file;

    Teuchos::RCP<APFMeshStruct> mesh_struct;

    bool doCollection;
    std::string outputFileName;

    int remeshFileIndex;

    long ofilepos;

    //! Epetra communicator
    Teuchos::RCP<const Teuchos_Comm> commT;

};

}

#endif

