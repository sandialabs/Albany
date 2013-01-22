//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <iostream>
#include <string>


#include "Albany_Utils.hpp"
#include "Albany_SolverFactory.hpp"
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

    EpetraExt::ModelEvaluator::InArgs params_in = solver->createInArgs();
    EpetraExt::ModelEvaluator::OutArgs responses_out = solver->createOutArgs();
    int num_p = params_in.Np();     // Number of *vectors* of parameters
    int num_g = responses_out.Ng(); // Number of *vectors* of responses

    // Set input parameters
    for (int i=0; i<num_p; i++) {
      RCP<Epetra_Vector> p = rcp(new Epetra_Vector(*(solver->get_p_init(i))));
       params_in.set_p(i,p);
    }

    // Set output responses and derivatives
    for (int i=0; i<num_g-1; i++) {
      RCP<const Epetra_Map> g_map = solver->get_g_map(i);
      RCP<Epetra_Vector> g = rcp(new Epetra_Vector(*g_map));
      responses_out.set_g(i,g);

      for (int j=0; j<num_p; j++) {
	RCP<const Epetra_Map> p_map = solver->get_p_map(j);
	if (!responses_out.supports(EpetraExt::ModelEvaluator::OUT_ARG_DgDp,
				    i, j).none()) {
	  *out << "Main: model supports sensitivities, so will request DgDp" << endl;
	  *out << " Num Responses: " << g_map->NumGlobalElements()
	       << ",   Num Parameters: " << p_map->NumGlobalElements() << endl;

	  if (p_map->NumGlobalElements() > 0) {
	    RCP<Epetra_MultiVector> dgdp =
	      rcp(new Epetra_MultiVector(*g_map, p_map->NumGlobalElements()));
	    responses_out.set_DgDp(i,j,dgdp);
	  }
	}
      }
    }
    RCP<Epetra_Vector> xfinal =
      rcp(new Epetra_Vector(*(solver->get_g_map(num_g-1)),true) );
    responses_out.set_g(num_g-1,xfinal);

    setupTimer.~TimeMonitor();
    *out << "Before main solve" << endl;
    solver->evalModel(params_in, responses_out);
    *out << "After main solve" << endl;

    *out << "Finished eval of first model: Params, Responses "
         << std::setprecision(12) << endl;

    for (int i=0; i<num_p; i++)
      params_in.get_p(i)->Print(*out << "\nParameter vector " << i << ":\n");

    for (int i=0; i<num_g-1; i++) {

      RCP<Epetra_Vector> g = responses_out.get_g(i);
      bool is_scalar = true;

      if (app != Teuchos::null)
        is_scalar = app->getResponse(i)->isScalarResponse();

      if (is_scalar) {
        g->Print(*out << "\nResponse vector " << i << ":\n");

        if(num_p == 0){
            // Just calculate regression data
            status += slvrfctry.checkTestResults(i, 0, g.get(), NULL);
        }
        else
          for (int j=0; j<num_p; j++) {

            if (!responses_out.supports(EpetraExt::ModelEvaluator::OUT_ARG_DgDp,
             i, j).none()) {

             RCP<Epetra_MultiVector> dgdp =
               responses_out.get_DgDp(i,j).getMultiVector();

             if (dgdp != Teuchos::null)
               dgdp->Print(*out << "\nSensitivities (" << i << "," << j << "):!\n");

             status += slvrfctry.checkTestResults(i, j, g.get(), dgdp.get());

           }
        }
      }
    }
    double mnv; xfinal->MeanValue(&mnv);
    *out << "Main_Solve: MeanValue of final solution " << mnv << endl;
    *out << "\nNumber of Failed Comparisons: " << status << endl;
  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);
  if (!success) status+=10000;

  Teuchos::TimeMonitor::summarize(*out,false,true,false/*zero timers*/);
  return status;
}

