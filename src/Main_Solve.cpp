//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <iostream>
#include <string>


#include "Albany_Utils.hpp"
#include "Albany_SolverFactory.hpp"

#include "Piro_Epetra_PerformSolve.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_TimeMonitor.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Teuchos_StandardCatchMacros.hpp"
#include "Epetra_Map.h"  //Needed for serial, somehow

// Uncomment for run time nan checking
// This is set in the toplevel CMakeLists.txt file
//#define ALBANY_CHECK_FPE

#ifdef ALBANY_CHECK_FPE
#include <math.h>
//#include <Accelerate/Accelerate.h>
#include <xmmintrin.h>
#endif

int main(int argc, char *argv[]) {

  int status=0; // 0 = pass, failures are incremented
  bool success = true;
  Teuchos::GlobalMPISession mpiSession(&argc,&argv);

#ifdef ALBANY_CHECK_FPE
	_mm_setcsr(_MM_MASK_MASK &~
		(_MM_MASK_OVERFLOW | _MM_MASK_INVALID | _MM_MASK_DIV_ZERO) );
#endif

  using Teuchos::RCP;
  using Teuchos::rcp;

  RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());

  // Command-line argument for input file
  std::string xmlfilename;
  if(argc > 1){

    if(!strcmp(argv[1],"--help")){
      printf("albany [inputfile.xml]\n");
      exit(1);
    }
    else
      xmlfilename = argv[1];

  }
  else
    xmlfilename = "input.xml";

  try {
    RCP<Teuchos::Time> totalTime =
      Teuchos::TimeMonitor::getNewTimer("Albany: ***Total Time***");

    RCP<Teuchos::Time> setupTime =
      Teuchos::TimeMonitor::getNewTimer("Albany: Setup Time");
    Teuchos::TimeMonitor totalTimer(*totalTime); //start timer
    Teuchos::TimeMonitor setupTimer(*setupTime); //start timer

    Albany::SolverFactory slvrfctry(xmlfilename, Albany_MPI_COMM_WORLD);
    RCP<Epetra_Comm> appComm = Albany::createEpetraCommFromMpiComm(Albany_MPI_COMM_WORLD);
    RCP<Albany::Application> app;
    RCP<EpetraExt::ModelEvaluator> solver =
      slvrfctry.createAndGetAlbanyApp(app, appComm, appComm);

    setupTimer.~TimeMonitor();

    *out << "Before main solve" << endl;

    Teuchos::ParameterList &solveParams =
      slvrfctry.getAnalysisParameters().sublist("Solve", /*mustAlreadyExist =*/ false);
    // By default, request the sensitivities if not explicitly disabled
    solveParams.get("Compute Sensitivities", true);

    Teuchos::Array<Teuchos::RCP<const Epetra_Vector> > responses;
    Teuchos::Array<Teuchos::Array<Teuchos::RCP<const Epetra_MultiVector> > > sensitivities;

    Piro::Epetra::PerformSolve(*solver, solveParams, responses, sensitivities);

    *out << "After main solve" << endl;

    const int num_p = solver->createOutArgs().Np(); // Number of *vectors* of parameters
    const int num_g = responses.size(); // Number of *vectors* of responses

    *out << "Finished eval of first model: Params, Responses "
      << std::setprecision(12) << endl;

    for (int i=0; i<num_p; i++)
      solver->get_p_init(i)->Print(*out << "\nParameter vector " << i << ":\n");

    for (int i=0; i<num_g-1; i++) {
      const RCP<const Epetra_Vector> g = responses[i];
      bool is_scalar = true;

      if (app != Teuchos::null)
        is_scalar = app->getResponse(i)->isScalarResponse();

      if (is_scalar) {
        g->Print(*out << "\nResponse vector " << i << ":\n");

        if (num_p == 0) {
          // Just calculate regression data
          status += slvrfctry.checkSolveTestResults(i, 0, g.get(), NULL);
        } else {
          for (int j=0; j<num_p; j++) {
            const RCP<const Epetra_MultiVector> dgdp = sensitivities[i][j];
            if (Teuchos::nonnull(dgdp)) {
              dgdp->Print(*out << "\nSensitivities (" << i << "," << j << "):!\n");
            }
            status += slvrfctry.checkSolveTestResults(i, j, g.get(), dgdp.get());
          }
        }
      }
    }

    const RCP<const Epetra_Vector> xfinal = responses.back();
    double mnv; xfinal->MeanValue(&mnv);
    *out << "Main_Solve: MeanValue of final solution " << mnv << endl;
    *out << "\nNumber of Failed Comparisons: " << status << endl;
  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);
  if (!success) status+=10000;

  Teuchos::TimeMonitor::summarize(*out,false,true,false/*zero timers*/);
  return status;
}
