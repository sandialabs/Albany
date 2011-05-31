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
  
  //***********************************************************
  // Command-line argument for input file
  //***********************************************************

  char * xmlfilename1=0;
  char defaultfile[10]={"input.xml"};
  if(argc>1){
    if(!strcmp(argv[1],"--help")){
      printf("albany [inputfile.xml]\n");
      exit(1);
    }
    else
      xmlfilename1=argv[1];
  }
  else
    xmlfilename1=defaultfile;

  
  char * xmlfilename2=0;
  char defaultfile2[11]={"input2.xml"};
  if(argc>1){
    if(!strcmp(argv[1],"--help")){
      printf("albany [inputfile2.xml]\n");
      exit(1);
    }
    else
      xmlfilename2=argv[2];
  }
  else
    xmlfilename2=defaultfile2;



  try {

    RCP<Teuchos::Time> totalTime = 
      Teuchos::TimeMonitor::getNewTimer("Albany: ***Total Time***");
    RCP<Teuchos::Time> setupTime = 
      Teuchos::TimeMonitor::getNewTimer("Albany: Setup Time");
    Teuchos::TimeMonitor totalTimer(*totalTime); //start timer
    Teuchos::TimeMonitor setupTimer(*setupTime); //start timer


    //***********************************************************
    // Set up the first model
    //***********************************************************

    Albany::SolverFactory slvrfctry1(xmlfilename1, Albany_MPI_COMM_WORLD);
    RCP<Epetra_Comm> appComm1 = Albany::createEpetraCommFromMpiComm(Albany_MPI_COMM_WORLD);
    RCP<EpetraExt::ModelEvaluator> App1 = slvrfctry1.create(appComm1, appComm1);

    EpetraExt::ModelEvaluator::InArgs params_in1 = App1->createInArgs();
    EpetraExt::ModelEvaluator::OutArgs responses_out1 = App1->createOutArgs();
    int num_p1 = params_in1.Np();     // Number of *vectors* of parameters
    int num_g1 = responses_out1.Ng(); // Number of *vectors* of responses
    RCP<Epetra_Vector> p1;
    RCP<Epetra_Vector> g1;

    if (num_p1 > 0)
      p1 = rcp(new Epetra_Vector(*(App1->get_p_init(0))));
    if (num_g1 > 1)
      g1 = rcp(new Epetra_Vector(*(App1->get_g_map(0))));
    RCP<Epetra_Vector> xfinal1 =
      rcp(new Epetra_Vector(*(App1->get_g_map(num_g1-1)),true) );

    //***********************************************************
    // Set up second model
    //***********************************************************

    Albany::SolverFactory slvrfctry2(xmlfilename2, Albany_MPI_COMM_WORLD);
    RCP<Epetra_Comm> appComm2 = Albany::createEpetraCommFromMpiComm(Albany_MPI_COMM_WORLD);
    RCP<EpetraExt::ModelEvaluator> App2 = slvrfctry2.create(appComm2, appComm2);

    EpetraExt::ModelEvaluator::InArgs params_in2 = App2->createInArgs();
    EpetraExt::ModelEvaluator::OutArgs responses_out2 = App2->createOutArgs();
    int num_p2 = params_in2.Np();     // Number of *vectors* of parameters
    int num_g2 = responses_out2.Ng(); // Number of *vectors* of responses
    RCP<Epetra_Vector> p2;
    RCP<Epetra_Vector> g2;

    if (num_p2 > 0)
      p2 = rcp(new Epetra_Vector(*(App2->get_p_init(0))));
    if (num_g2 > 1)
      g2 = rcp(new Epetra_Vector(*(App2->get_g_map(0))));
    RCP<Epetra_Vector> xfinal2 =
      rcp(new Epetra_Vector(*(App2->get_g_map(num_g2-1)),true) );


    //***********************************************************
    // Set up an iteration solution procedure
    //***********************************************************
 
    double mnv1prev; xfinal1->MeanValue(&mnv1prev);
    double mnv2prev; xfinal2->MeanValue(&mnv2prev);
    double conv_error = 1;
    double conv_tol = 1e-6;
    int max_iter = 25;
    int iter = 0;

    while ((conv_error > conv_tol) && (iter < max_iter)){
    
        iter = iter + 1;

        *p1 = *g2;

        // Evaluate first model
        if (num_p1 > 0)  params_in1.set_p(0,p1);
    	if (num_g1 > 1)  responses_out1.set_g(0,g1);
    	responses_out1.set_g(num_g1-1,xfinal1);

    	setupTimer.~TimeMonitor();
    	App1->evalModel(params_in1, responses_out1);

    	*out << "Finished eval of first model: Params, Responses " 
        	 << std::setprecision(12) << endl;
    	if (num_p1>0) p1->Print(*out << "\nParameters!\n");
    	if (num_g1>1) g1->Print(*out << "\nResponses!\n");
    	double mnv1; xfinal1->MeanValue(&mnv1);
    	*out << "Main_Solve: MeanValue of first solution " << mnv1 << endl;


    	// Evaluate second model
    	if (num_p2 > 0)  params_in2.set_p(0,p2);
    	if (num_g2 > 1)  responses_out2.set_g(0,g2);
    	responses_out2.set_g(num_g2-1,xfinal2);

        *p2 = *g1;

    	setupTimer.~TimeMonitor();
    	App2->evalModel(params_in2, responses_out2);

    	*out << "Finished eval of second model: Params, Responses " 
        	 << std::setprecision(12) << endl;
    	if (num_p2>0) p2->Print(*out << "\nParameters!\n");
    	if (num_g2>1) g2->Print(*out << "\nResponses!\n");
    	double mnv2; xfinal2->MeanValue(&mnv2);
    	*out << "Main_Solve: MeanValue of second solution " << mnv2 << endl;
 
        conv_error = abs(mnv1 - mnv1prev) + abs(mnv2 - mnv2prev);
        mnv1prev = mnv1;
        mnv2prev = mnv2;
    }

    *out << "Number of iterations " << iter << endl;
    *out << "Iterative error " << conv_error << endl;
  }

  TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);
  if (!success) status+=10000;

  Teuchos::TimeMonitor::summarize(*out,false,true,false/*zero timers*/);
  return status;
}
