//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <iostream>
#include <string>

#include "Albany_Utils.hpp"
#include "Albany_SolverFactory.hpp"
#include "Albany_NOXObserver.hpp"
#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_TimeMonitor.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Teuchos_StandardCatchMacros.hpp"
#include "Epetra_Map.h"  //Needed for serial, somehow
#include "Piro_Epetra_NECoupledModelEvaluator.hpp"
#include "Piro_Epetra_SolverFactory.hpp"
#include "Epetra_LocalMap.h"
#include "Epetra_Import.h"

#include "Albany_Networks.hpp"

// Global variable that denotes this is the Tpetra executable
bool TpetraBuild = false;

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

  Albany::CmdLineArgs cmd;
  cmd.parse_cmdline(argc, argv, *out);
  std::string xmlfilename_coupled = cmd.xml_filename;

  try {

    RCP<Teuchos::Time> totalTime = 
      Teuchos::TimeMonitor::getNewTimer("Albany: ***Total Time***");
    RCP<Teuchos::Time> setupTime = 
      Teuchos::TimeMonitor::getNewTimer("Albany: Setup Time");
    Teuchos::TimeMonitor totalTimer(*totalTime); //start timer
    Teuchos::TimeMonitor setupTimer(*setupTime); //start timer

    //***********************************************************
    // Set up coupled model
    //***********************************************************

    RCP<Epetra_Comm> coupledComm = 
      Albany::createEpetraCommFromMpiComm(Albany_MPI_COMM_WORLD);
    RCP<const Teuchos_Comm> comm =
      Tpetra::DefaultPlatform::getDefaultPlatform().getComm();
    // Connect vtune for performance profiling
    if (cmd.vtune) {
      Albany::connect_vtune(comm->getRank());
    }
    Albany::SolverFactory coupled_slvrfctry(xmlfilename_coupled, comm);
    Teuchos::ParameterList& coupledParams = coupled_slvrfctry.getParameters();
    Teuchos::ParameterList& coupledSystemParams = 
      coupledParams.sublist("Coupled System");
    Teuchos::RCP< Teuchos::ParameterList> coupledPiroParams = 
      Teuchos::rcp(&(coupledParams.sublist("Piro")),false);
    Teuchos::Array<std::string> model_filenames =
      coupledSystemParams.get<Teuchos::Array<std::string> >("Model XML Files");
    int num_models = model_filenames.size();
    Teuchos::Array< RCP<Albany::Application> > apps(num_models);
    Teuchos::Array< RCP<EpetraExt::ModelEvaluator> > models(num_models);
    Teuchos::Array< RCP<Teuchos::ParameterList> > piroParams(num_models);

    // Set up each model
    for (int m=0; m<num_models; m++) {
      Albany::SolverFactory slvrfctry(model_filenames[m], 
				      comm);
      models[m] = slvrfctry.createAlbanyAppAndModel(apps[m], comm);
      Teuchos::ParameterList& appParams = slvrfctry.getParameters();
      piroParams[m] = Teuchos::rcp(&(appParams.sublist("Piro")),false);
    }
    
    // Setup network model
    std::string network_name = 
      coupledSystemParams.get("Network Model", "Param To Response");
    RCP<Piro::Epetra::AbstractNetworkModel> network_model;
    if (network_name == "Param To Response")
      network_model = rcp(new Piro::Epetra::ParamToResponseNetworkModel);
    else if (network_name == "Reactor Network")
      network_model = rcp(new Albany::ReactorNetworkModel(1));
    else
      TEUCHOS_TEST_FOR_EXCEPTION(
	true, std::logic_error, "Invalid network model name " << network_name);
    RCP<EpetraExt::ModelEvaluator> coupledModel =
      rcp(new Piro::Epetra::NECoupledModelEvaluator(models, piroParams,
						    network_model,
						    coupledPiroParams, 
						    coupledComm));
    Piro::Epetra::SolverFactory piroEpetraFactory;
    RCP<EpetraExt::ModelEvaluator> coupledSolver =
      piroEpetraFactory.createSolver(coupledPiroParams, coupledModel);
    
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

    // "observe solution" -- need to integrate this with the solvers
    for (int m=0; m<num_models; m++) {
      Albany_NOXObserver observer(apps[m]);
      observer.observeSolution(*(outArgs.get_g(m)));
    }
    
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

    status += coupled_slvrfctry.checkSolveTestResults(0, 0, x_final_local.get(), NULL);
    *out << "\nNumber of Failed Comparisons: " << status << std::endl;
  }

  TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);
  if (!success) status+=10000;

  Teuchos::TimeMonitor::summarize(*out,false,true,false/*zero timers*/);
  return status;
}
