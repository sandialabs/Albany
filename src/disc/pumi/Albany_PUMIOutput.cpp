//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_PUMIOutput.hpp"
#include "Albany_PUMIVtk.hpp"
#include "Albany_PUMIExodus.hpp"

Albany::PUMIOutput::~PUMIOutput() {
}

Albany::PUMIOutput* Albany::PUMIOutput::create(
  const Teuchos::RCP<PUMIMeshStruct>& meshStruct,
  const Teuchos::RCP<const Teuchos_Comm>& comm) {
  if (meshStruct->outputFileName.find("exo") != std::string::npos)
    return new PUMIExodus(meshStruct, comm);
  else
    return new PUMIVtk(meshStruct, comm);
}
