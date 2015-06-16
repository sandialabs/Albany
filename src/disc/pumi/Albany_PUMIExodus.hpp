//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#ifndef ALBANY_PUMIEXODUS_HPP
#define ALBANY_PUMIEXODUS_HPP

#include "Albany_PUMIOutput.hpp"

namespace Albany {

class PUMIExodus : public PUMIOutput {

  public:

    PUMIExodus(const Teuchos::RCP<APFMeshStruct>& meshStruct,
               const Teuchos::RCP<const Teuchos_Comm>& comm_);

    ~PUMIExodus();

    void write(const char* filename, const double time);
    void writeFile(const double time);

    void setFileName(const std::string& fname){ outputFileName = fname; }

  private:
    Teuchos::RCP<APFMeshStruct> mesh_struct;
    apf::Mesh2* mesh;
    apf::StkModels& sets_p;
    std::string outputFileName;
    Teuchos::RCP<const Teuchos_Comm> comm;
};

}

#endif

