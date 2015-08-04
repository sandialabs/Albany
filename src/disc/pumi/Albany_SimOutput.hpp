//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_SIMOUTPUT_HPP
#define ALBANY_SIMOUTPUT_HPP

#include "Albany_PUMIOutput.hpp"

namespace apf {
class MeshSIM;
}

namespace Albany {

class SimOutput : public PUMIOutput {

  public:

    SimOutput(const Teuchos::RCP<APFMeshStruct>& meshStruct_,
              const Teuchos::RCP<const Teuchos_Comm>& commT_);

    ~SimOutput();

    void writeFile(const double time);
    void setFileName(const std::string& fname);

  private:
    //! Epetra communicator
    Teuchos::RCP<const Teuchos_Comm> commT;
    apf::MeshSIM* mesh;
    std::string filename;
    int index;
};

}

#endif


