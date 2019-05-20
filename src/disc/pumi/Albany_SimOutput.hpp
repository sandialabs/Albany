//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_SIM_OUTPUT_HPP
#define ALBANY_SIM_OUTPUT_HPP

#include "Albany_PUMIOutput.hpp"

namespace apf {
class MeshSIM;
}

namespace Albany {

class SimOutput : public PUMIOutput {
public:

  SimOutput(const Teuchos::RCP<APFMeshStruct>& meshStruct_,
            const Teuchos::RCP<const Teuchos_Comm>& comm_);

  ~SimOutput() = default;

  void writeFile(const double time);
  void setFileName(const std::string& fname);

private:
  //! Communicator
  Teuchos::RCP<const Teuchos_Comm> comm;
  apf::MeshSIM* mesh;
  std::string filename;
  int index;
};

} // namespace Albany

#endif // ALBANY_SIM_OUTPUT_HPP
