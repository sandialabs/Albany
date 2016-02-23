//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_DataTypes.hpp"
#include "Albany_Utils.hpp"

#include "Albany_Application.hpp"
#include "Albany_MORFacadeImpl.hpp"
#include "Albany_ObserverImpl.hpp"

#include "MOR_EpetraLocalMapMVMatrixMarketUtils.hpp"
#include "MOR_ReducedSpace.hpp"

#include "Epetra_Comm.h"
#include "Epetra_Vector.h"

#include "Teuchos_RCP.hpp"
#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_Comm.hpp"
#include "Teuchos_FancyOStream.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"

#include <string>
#include <iostream>

// Global variable that denotes this is the Tpetra executable
bool TpetraBuild = false;

int main(int argc, char *argv[]) {
  using Teuchos::RCP;

  // Communicators
  Teuchos::GlobalMPISession mpiSession(&argc, &argv);
  const Albany_MPI_Comm nativeComm = Albany_MPI_COMM_WORLD;
  const RCP<const Teuchos::Comm<int> > teuchosComm = Albany::createTeuchosCommFromMpiComm(nativeComm);
  const RCP<const Epetra_Comm> epetraComm = Albany::createEpetraCommFromMpiComm(nativeComm);

  // Standard output
  const RCP<Teuchos::FancyOStream> out = Teuchos::VerboseObjectBase::getDefaultOStream();

  // Parse command-line argument for input file
  const std::string firstArg = (argc > 1) ? argv[1] : "";
  if (firstArg.empty() || firstArg == "--help") {
    *out << "AlbanyRomPostProcess input-file-path\n";
    return 0;
  }
  const std::string inputFileName = argv[1];

  // Parse XML input file
  const RCP<Teuchos::ParameterList> topLevelParams = Teuchos::createParameterList("Albany Parameters");
  Teuchos::updateParametersFromXmlFileAndBroadcast(inputFileName, topLevelParams.ptr(), *teuchosComm);
  (void) Teuchos::sublist(topLevelParams, "Debug Output");

  const RCP<Albany::Application> application(new Albany::Application(teuchosComm, topLevelParams));

  const RCP<Teuchos::ParameterList> postParams = Teuchos::sublist(topLevelParams, "Post Processing");

  const RCP<Teuchos::ParameterList> spaceParams = Teuchos::sublist(postParams, "Reduced Space");
  const RCP<Albany::MORFacadeImpl> morFacade =
    Teuchos::rcp_dynamic_cast<Albany::MORFacadeImpl>(application->getMorFacade());
  const RCP<const MOR::ReducedSpace> reducedSpace = morFacade->spaceFactory().create(spaceParams);

  const std::string generalizedCoordsFilename =
    postParams->get("Generalized Coordinates Input File Name", "generalized_coordinates.mtx");
  const RCP<const Epetra_MultiVector> reducedSolutions = MOR::readLocalMapMultiVectorFromMatrixMarket(
      generalizedCoordsFilename, reducedSpace->comm(), reducedSpace->basisSize());

  const std::string stampsFilename =
    postParams->get("Generalized Coordinates Stamps Input File Name", "stamps_" + generalizedCoordsFilename);
  const RCP<const Epetra_MultiVector> stamps = MOR::readLocalMapMultiVectorFromMatrixMarket(
      stampsFilename, reducedSpace->comm(), 1);

  Albany::ObserverImpl observer(application);

  Epetra_Vector fullSolution(reducedSpace->basisMap(), /*zeroOut =*/ false);
  const Teuchos::RCP<const Epetra_Vector> fullSolution_dot = Teuchos::null; // Time derivative not handled
  for (int step = 0; step < reducedSolutions->NumVectors(); ++step) {
    reducedSpace->expansion(*(*reducedSolutions)(step), fullSolution);
    const double stamp = (*stamps)[step][0];
    observer.observeSolution(stamp, fullSolution, fullSolution_dot.ptr());
  }
}
