//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_PUMIOutput.hpp"
#include "Albany_PUMIVtk.hpp"
#if defined(HAVE_STK)
#include "Albany_PUMIExodus.hpp"
#endif

Albany::PUMIOutput::~PUMIOutput() {
}

Albany::PUMIOutput* Albany::PUMIOutput::create(
  const Teuchos::RCP<PUMIMeshStruct>& meshStruct,
  const Teuchos::RCP<const Teuchos_Comm>& comm) {
#if defined(HAVE_STK)
  if (meshStruct->outputFileName.find("exo") != std::string::npos)
    return new PUMIExodus(meshStruct, comm);
#endif
  if (meshStruct->outputFileName.find("vtk") != std::string::npos)
    return new PUMIVtk(meshStruct, comm);
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
			      "PUMIDiscretization: Output should be .exo or .vtk, have "<<
            meshStruct->outputFileName << '\n');
}
