/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/


#include <iostream>
#include <string>

#include "Albany_SolverFactory.hpp"
#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_TimeMonitor.hpp"
#include "Teuchos_VerboseObject.hpp"

int main(int argc, char *argv[]) {

  int status=0; // 0 = pass, failures are incremented

  // Initialize MPI and timer
  Teuchos::GlobalMPISession mpiSession(&argc,&argv);

#ifdef ALBANY_MPI
  double total_time = -MPI_Wtime();
#endif
  MPI_Comm appComm = MPI_COMM_WORLD;

  using Teuchos::RCP;
  using Teuchos::rcp;

  RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());

  // Command-line argument for input file
  char * xmlfilename=0;
  char defaultfile[10]={"input.xml"};
  if(argc>1){
    if(!strcmp(argv[1],"--help")){
      printf("albany [inputfile.xml]\n");
      exit(1);
    }
    else
      xmlfilename=argv[1];
  }
  else
    xmlfilename=defaultfile;
       
  
  try {

    RCP<Teuchos::Time> totalTime = 
      Teuchos::TimeMonitor::getNewTimer("Albany ***Total Time***");
    RCP<Teuchos::Time> setupTime = 
      Teuchos::TimeMonitor::getNewTimer("Albany Setup Time");
    Teuchos::TimeMonitor totalTimer(*totalTime); //start timer
    Teuchos::TimeMonitor setupTimer(*setupTime); //start timer

    Albany::SolverFactory slvrfctry(xmlfilename, appComm);
    RCP<EpetraExt::ModelEvaluator> App = slvrfctry.create();

    EpetraExt::ModelEvaluator::InArgs params_in = App->createInArgs();
    EpetraExt::ModelEvaluator::OutArgs responses_out = App->createOutArgs();
    int num_p = params_in.Np();     // Number of *vectors* of parameters
    int num_g = responses_out.Ng(); // Number of *vectors* of responses
    RCP<Epetra_Vector> p1;
    RCP<Epetra_Vector> g1;

    if (num_p > 0)
      p1 = rcp(new Epetra_Vector(*(App->get_p_init(0))));
    if (num_g > 1)
      g1 = rcp(new Epetra_Vector(*(App->get_g_map(0))));
    RCP<Epetra_Vector> xfinal =
      rcp(new Epetra_Vector(*(App->get_g_map(num_g-1))));

    // Sensitivity Analysis stuff
    bool supportsSensitivities = false;
    RCP<Epetra_MultiVector> dgdp;

    if (num_p>0 && num_g>1) {
      supportsSensitivities =
        !responses_out.supports(EpetraExt::ModelEvaluator::OUT_ARG_DgDp, 0, 0).none();

      if (supportsSensitivities) {
        *out << "Main: model supports sensitivities, so will request DgDp" << endl;
        //dgdp = rcp(new Epetra_MultiVector(p1->Map(), g1->GlobalLength() ));
        *out << " Num Responses: " << g1->GlobalLength() 
             << ",   Num Parameters: " << p1->GlobalLength() << endl;

        if (p1->GlobalLength() > 0)
          dgdp = rcp(new Epetra_MultiVector(g1->Map(), p1->GlobalLength() ));
        else
          supportsSensitivities = false;
      }
    }

    // Evaluate first model
    if (num_p > 0)  params_in.set_p(0,p1);
    if (num_g > 1)  responses_out.set_g(0,g1);
    responses_out.set_g(num_g-1,xfinal);

    if (supportsSensitivities) responses_out.set_DgDp(0,0,dgdp);
    setupTimer.~TimeMonitor();
    App->evalModel(params_in, responses_out);

    *out << "Finished eval of first model: Params, Responses " 
         << std::setprecision(12) << endl;
    if (num_p>0) p1->Print(*out << "\nParameters!\n");
    if (num_g>1) g1->Print(*out << "\nResponses!\n");
    if (supportsSensitivities)
      dgdp->Print(*out << "\nSensitivities!\n");
    double mnv; xfinal->MeanValue(&mnv);
    *out << "Main_Solve: MeanValue of final solution " << mnv << endl;

    //cout << "Final Solution \n" << *xfinal << endl;

    status += slvrfctry.checkTestResults(g1.get(), dgdp.get());
    *out << "\nNumber of Failed Comparisons: " << status << endl;
  }

  catch (std::exception& e) {
    cout << e.what() << endl;
    status = 10;
  }
  catch (string& s) {
    cout << s << endl;
    status = 20;
  }
  catch (char *s) {
    cout << s << endl;
    status = 30;
  }
  catch (...) {
    cout << "Caught unknown exception!" << endl;
    status = 40;
  }

#ifdef ALBANY_MPI
  MPI_Barrier(MPI_COMM_WORLD);
  total_time +=  MPI_Wtime();
  *out << "\n\nTOTAL TIME     " << total_time << "  " << total_time << endl;
#else
  *out << "\tTOTAL TIME =     -999.0  -999.0" << endl;
#endif

  Teuchos::TimeMonitor::summarize(*out,false,true,false);
  return status;
}
