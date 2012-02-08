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
#include "Piro_Epetra_NECoupledModelEvaluator.hpp"
#include "Piro_Epetra_Factory.hpp"
#include "Epetra_LocalMap.h"
#include "Epetra_Import.h"

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

  std::string xmlfilename1, xmlfilename2, xmlfilename3;
  if (argc != 4 || (argc>1 && !strcmp(argv[1],"--help"))) {
    std::cout << "albany input1.xml input2.xml input_coupled.xml\n";
    std::exit(1);
  }
  xmlfilename1=argv[1];
  xmlfilename2=argv[2];
  xmlfilename3=argv[3];



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
    RCP<Epetra_Comm> appComm1 = 
      Albany::createEpetraCommFromMpiComm(Albany_MPI_COMM_WORLD);
    RCP<Albany::Application> app1;
    RCP<EpetraExt::ModelEvaluator> model1 = 
      slvrfctry1.createAlbanyAppAndModel(app1, appComm1);
    Teuchos::ParameterList& appParams1 = slvrfctry1.getParameters();
    Teuchos::RCP< Teuchos::ParameterList> piroParams1 = 
      Teuchos::rcp(&(appParams1.sublist("Piro")),false);

    //***********************************************************
    // Set up second model
    //***********************************************************

    Albany::SolverFactory slvrfctry2(xmlfilename2, Albany_MPI_COMM_WORLD);
    RCP<Epetra_Comm> appComm2 = 
      Albany::createEpetraCommFromMpiComm(Albany_MPI_COMM_WORLD);
    RCP<Albany::Application> app2;
    RCP<EpetraExt::ModelEvaluator> model2 = 
      slvrfctry2.createAlbanyAppAndModel(app2, appComm2);
    Teuchos::ParameterList& appParams2 = slvrfctry2.getParameters();
    Teuchos::RCP< Teuchos::ParameterList> piroParams2 = 
      Teuchos::rcp(&(appParams2.sublist("Piro")),false);

    
    //***********************************************************
    // Set up coupled model
    //***********************************************************
    Albany::SolverFactory coupled_slvrfctry(xmlfilename3, 
					    Albany_MPI_COMM_WORLD);
    RCP<Epetra_Comm> coupledComm = 
      Albany::createEpetraCommFromMpiComm(Albany_MPI_COMM_WORLD);
    Teuchos::ParameterList& coupledParams = coupled_slvrfctry.getParameters();
    Teuchos::RCP< Teuchos::ParameterList> coupledPiroParams = 
      Teuchos::rcp(&(coupledParams.sublist("Piro")),false);
    RCP<EpetraExt::ModelEvaluator> coupledModel =
      rcp(new Piro::Epetra::NECoupledModelEvaluator(model1, model2,
						    piroParams1, piroParams2,
						    coupledPiroParams, 
						    coupledComm));
    RCP<EpetraExt::ModelEvaluator> coupledSolver =
      Piro::Epetra::Factory::createSolver(coupledPiroParams, coupledModel);
    
    // Solve coupled system
    EpetraExt::ModelEvaluator::InArgs inArgs = coupledSolver->createInArgs();
    EpetraExt::ModelEvaluator::OutArgs outArgs = coupledSolver->createOutArgs();
    for (int i=0; i<inArgs.Np(); i++)
      inArgs.set_p(i, coupledSolver->get_p_init(i));
    for (int i=0; i<outArgs.Ng(); i++) {
      RCP<Epetra_Vector> g = 
	rcp(new Epetra_Vector(*(coupledSolver->get_g_map(i))));
      outArgs.set_g(i, g);
    }
    coupledSolver->evalModel(inArgs, outArgs);
    
    // Print results
    RCP<Epetra_Vector> x_final = outArgs.get_g(outArgs.Ng()-1);
    RCP<Epetra_Vector> x_final_local = x_final;
    if (x_final->Map().DistributedGlobal()) {
      Epetra_LocalMap local_map(x_final->GlobalLength(), 0, 
				x_final->Map().Comm());
      x_final_local = rcp(new Epetra_Vector(local_map));
      Epetra_Import importer(local_map, x_final->Map());
      x_final_local->Import(*x_final, importer, Insert);
    }
    *out << std::endl
	 << "Final value of coupling variables:" << std::endl
	 << *x_final_local << std::endl;

    status += coupled_slvrfctry.checkTestResults(0, 0, x_final_local.get(), 
						 NULL);
    *out << "\nNumber of Failed Comparisons: " << status << endl;
  }

  TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);
  if (!success) status+=10000;

  Teuchos::TimeMonitor::summarize(*out,false,true,false/*zero timers*/);
  return status;
}
