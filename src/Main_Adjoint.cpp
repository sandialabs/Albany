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

#include "Albany_Utils.hpp"
#include "Albany_SolverFactory.hpp"
#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_TimeMonitor.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Teuchos_StandardCatchMacros.hpp"
#include "Epetra_Map.h"  //Needed for serial, somehow

int main(int argc, char *argv[]) {

  int status=0; // 0 = pass, failures are incremented
  bool success = true;
  Teuchos::GlobalMPISession mpiSession(&argc,&argv);

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
  
  char * xmladjfilename=0;
  char defaultadjfile[18]={"input_adjoint.xml"};
  if(argc>1){
    if(!strcmp(argv[1],"--help")){
      printf("albany [inputadjfile.xml]\n");
      exit(1);
    }
    else
      xmladjfilename=argv[2];
  }
  else
    xmladjfilename=defaultadjfile;



  try {

    RCP<Teuchos::Time> totalTime = 
      Teuchos::TimeMonitor::getNewTimer("Albany: ***Total Time***");
    RCP<Teuchos::Time> setupTime = 
      Teuchos::TimeMonitor::getNewTimer("Albany: Setup Time");
    Teuchos::TimeMonitor totalTimer(*totalTime); //start timer
    Teuchos::TimeMonitor setupTimer(*setupTime); //start timer

    Albany::SolverFactory slvrfctry(xmlfilename, Albany_MPI_COMM_WORLD);
    RCP<Epetra_Comm> appComm = Albany::createEpetraCommFromMpiComm(Albany_MPI_COMM_WORLD);
    RCP<EpetraExt::ModelEvaluator> App = slvrfctry.create(appComm, appComm);

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
      rcp(new Epetra_Vector(*(App->get_g_map(num_g-1)),true) );

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

    status += slvrfctry.checkTestResults(0, 0, g1.get(), dgdp.get());
    *out << "\nNumber of Failed Comparisons: " << status << endl;

    // write out the current mesh to an exodus file

    // promote the mesh to higher order


    // Start adjoint solve

    RCP<Epetra_Vector> xinit =
      rcp(new Epetra_Vector(*(App->get_g_map(num_g-1)),true) );
    
    RCP<Teuchos::Time> totalAdjTime = 
      Teuchos::TimeMonitor::getNewTimer("Albany: ***Total Time***");
    RCP<Teuchos::Time> setupAdjTime = 
      Teuchos::TimeMonitor::getNewTimer("Albany: Setup Time");
    Teuchos::TimeMonitor totalAdjTimer(*totalAdjTime); //start timer
    Teuchos::TimeMonitor setupAdjTimer(*setupAdjTime); //start timer

    Albany::SolverFactory adjslvrfctry(xmladjfilename, Albany_MPI_COMM_WORLD);
    RCP<Epetra_Comm> adjappComm = Albany::createEpetraCommFromMpiComm(Albany_MPI_COMM_WORLD);
    RCP<EpetraExt::ModelEvaluator> AdjApp = adjslvrfctry.create(adjappComm, adjappComm, xinit);

    EpetraExt::ModelEvaluator::InArgs adj_params_in = AdjApp->createInArgs();
    EpetraExt::ModelEvaluator::OutArgs adj_responses_out = AdjApp->createOutArgs();
    int adj_num_p = adj_params_in.Np();     // Number of *vectors* of parameters
    int adj_num_g = adj_responses_out.Ng(); // Number of *vectors* of responses
    RCP<Epetra_Vector> adj_p1;
    RCP<Epetra_Vector> adj_g1;

    if (adj_num_p > 0)
      adj_p1 = rcp(new Epetra_Vector(*(AdjApp->get_p_init(0))));
    if (adj_num_g > 1)
      adj_g1 = rcp(new Epetra_Vector(*(AdjApp->get_g_map(0))));
    RCP<Epetra_Vector> adj_xfinal =
      rcp(new Epetra_Vector(*(AdjApp->get_g_map(adj_num_g-1)),true) );

    // Sensitivity Analysis stuff
    bool adj_supportsSensitivities = false;
    RCP<Epetra_MultiVector> adj_dgdp;

    if (adj_num_p>0 && adj_num_g>1) {
      adj_supportsSensitivities =
        !adj_responses_out.supports(EpetraExt::ModelEvaluator::OUT_ARG_DgDp, 0, 0).none();

      if (adj_supportsSensitivities) {
        *out << "Main: model supports sensitivities, so will request DgDp" << endl;
        //dgdp = rcp(new Epetra_MultiVector(p1->Map(), g1->GlobalLength() ));
        *out << " Num Responses: " << adj_g1->GlobalLength() 
             << ",   Num Parameters: " << adj_p1->GlobalLength() << endl;

        if (adj_p1->GlobalLength() > 0)
          adj_dgdp = rcp(new Epetra_MultiVector(adj_g1->Map(), adj_p1->GlobalLength() ));
        else
          adj_supportsSensitivities = false;
      }
    }


    /* Set the initial guess for the adjoint to be the forward solution
       projected onto the adjoint degrees of freedom.  The adjoint problem
       is linear, so this is only performed so that the derivative of the
       response function (which forms the RHS) is correct.
    */

    // Teuchos::RCP<ENAT::SGNOXSolver> AdjApp =
    //  Teuchos::rcp_dynamic_cast<ENAT::SGNOXSolver>(adjslvrfctry.create(adjapp_comm, adjapp_comm, xfinal));



    // Evaluate adjoint model
    if (adj_num_p > 0)  adj_params_in.set_p(0,adj_p1);
    if (adj_num_g > 1)  adj_responses_out.set_g(0,adj_g1);
    adj_responses_out.set_g(adj_num_g-1,adj_xfinal);

    if (adj_supportsSensitivities) adj_responses_out.set_DgDp(0,0,adj_dgdp);
    setupAdjTimer.~TimeMonitor();
    AdjApp->evalModel(adj_params_in, adj_responses_out);

    *out << "Finished eval of adjoint model: Params, Responses " 
         << std::setprecision(12) << endl;
    if (adj_num_p>0) adj_p1->Print(*out << "\nParameters!\n");
    if (adj_num_g>1) adj_g1->Print(*out << "\nResponses!\n");
    if (adj_supportsSensitivities)
      adj_dgdp->Print(*out << "\nSensitivities!\n");
    double adj_mnv; adj_xfinal->MeanValue(&adj_mnv);
    *out << "Main_Solve: MeanValue of final solution " << adj_mnv << endl;

    //cout << "Final Solution \n" << *xfinal << endl;

    status += adjslvrfctry.checkTestResults(0, 0, adj_g1.get(), adj_dgdp.get());
    *out << "\nNumber of Failed Comparisons: " << status << endl;
  }

  TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);
  if (!success) status+=10000;

  Teuchos::TimeMonitor::summarize(*out,false,true,false/*zero timers*/);
  return status;
}
