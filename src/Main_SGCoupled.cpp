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
#include "Stokhos.hpp"
#include "Stokhos_Epetra.hpp"
#include "Piro_Epetra_StokhosSolver.hpp"
#include "Piro_Epetra_NECoupledModelEvaluator.hpp"

#include "Albany_Networks.hpp"

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

  std::string xmlfilename_coupled;
  if(argc > 1){
    if(!strcmp(argv[1],"--help")){
      std::cout << "albany [inputfile.xml]" << std::endl;
      std::exit(1);
    }
    else
      xmlfilename_coupled = argv[1];
  }
  else
    xmlfilename_coupled = "input.xml";


  try {

    RCP<Teuchos::Time> totalTime = 
      Teuchos::TimeMonitor::getNewTimer("AlbanySG: ***Total Time***");
    RCP<Teuchos::Time> setupTime = 
      Teuchos::TimeMonitor::getNewTimer("AlbanySG: Setup Time");
    Teuchos::TimeMonitor totalTimer(*totalTime); //start timer

    //***********************************************************
    // Set up coupled solver first to setup comm's
    //***********************************************************
    Teuchos::RCP<Epetra_Comm> globalComm = 
      Albany::createEpetraCommFromMpiComm(Albany_MPI_COMM_WORLD);
    Albany::SolverFactory coupled_slvrfctry(xmlfilename_coupled, 
					    Albany_MPI_COMM_WORLD);
    Teuchos::ParameterList& coupledParams = coupled_slvrfctry.getParameters();
    Teuchos::ParameterList& coupledSystemParams = 
      coupledParams.sublist("Coupled System");
    Teuchos::Array<std::string> model_filenames =
      coupledSystemParams.get<Teuchos::Array<std::string> >("Model XML Files");
    int num_models = model_filenames.size();
    Teuchos::Array< RCP<Albany::Application> > apps(num_models);
    Teuchos::Array< RCP<EpetraExt::ModelEvaluator> > models(num_models);
    Teuchos::Array< RCP<Teuchos::ParameterList> > piroParams(num_models);
    Teuchos::RCP< Teuchos::ParameterList> coupledPiroParams = 
      Teuchos::rcp(&(coupledParams.sublist("Piro")),false);
    Teuchos::RCP<Piro::Epetra::StokhosSolver> coupledSolver =
      Teuchos::rcp(new Piro::Epetra::StokhosSolver(coupledPiroParams, 
						   globalComm));
    Teuchos::RCP<const Epetra_Comm> app_comm = coupledSolver->getSpatialComm();

    // Set up each model
    for (int m=0; m<num_models; m++) {
      Albany::SolverFactory slvrfctry(
	model_filenames[m], 
	Albany::getMpiCommFromEpetraComm(*app_comm));
      models[m] = slvrfctry.createAlbanyAppAndModel(apps[m], app_comm);
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
						    globalComm));
    coupledSolver->setup(coupledModel);

    // Solve coupled system
    EpetraExt::ModelEvaluator::InArgs inArgs = coupledSolver->createInArgs();
    EpetraExt::ModelEvaluator::OutArgs outArgs = coupledSolver->createOutArgs();
    for (int i=0; i<inArgs.Np(); i++)
      if (inArgs.supports(EpetraExt::ModelEvaluator::IN_ARG_p_sg, i))
	inArgs.set_p_sg(i, coupledSolver->get_p_sg_init(i));
    for (int i=0; i<outArgs.Ng(); i++) 
      if (outArgs.supports(EpetraExt::ModelEvaluator::OUT_ARG_g_sg, i)) {
	RCP<Stokhos::EpetraVectorOrthogPoly> g_sg = 
	  coupledSolver->create_g_sg(i);
	outArgs.set_g_sg(i, g_sg);
      }
    coupledSolver->evalModel(inArgs, outArgs);

    // Print results
    bool printResponse = 
      coupledSystemParams.get("Print Response Expansion", true);
    int idx = outArgs.Ng()-1;
    Teuchos::RCP<Stokhos::EpetraVectorOrthogPoly> g_sg = 
      outArgs.get_g_sg(idx);
    Teuchos::RCP<Stokhos::SGModelEvaluator> sg_model =
      coupledSolver->get_sg_model();
    Teuchos::RCP<Stokhos::EpetraVectorOrthogPoly> g_sg_local = 
      //sg_model->import_solution_poly(*(g_sg->getBlockVector()));
      g_sg;
    Epetra_Vector g_mean(*(g_sg->coefficientMap()));
    Epetra_Vector g_std_dev(*(g_sg->coefficientMap()));
    g_sg->computeMean(g_mean);
    g_sg->computeStandardDeviation(g_std_dev);
    RCP<Epetra_Vector> g_mean_local = rcp(&g_mean,false);
    RCP<Epetra_Vector> g_std_dev_local = rcp(&g_std_dev,false);
    if (g_mean.Map().DistributedGlobal()) {
      Epetra_LocalMap local_map(g_mean.GlobalLength(), 0, 
				g_mean.Map().Comm());
      g_mean_local = rcp(new Epetra_Vector(local_map));
      g_std_dev_local = rcp(new Epetra_Vector(local_map));
      Epetra_Import importer(local_map, g_mean.Map());
      g_mean_local->Import(g_mean, importer, Insert);
      g_std_dev_local->Import(g_std_dev, importer, Insert);
    }
    out->precision(16);
    *out << std::endl
	 << "Final value of coupling variables:" << std::endl
	 << "Mean:" << std::endl << *g_mean_local << std::endl
	 << "Std. Dev.:" << std::endl << *g_std_dev_local << std::endl;
    if (printResponse)
      *out << "PCE:" << std::endl << *g_sg_local << std::endl;

    status += coupled_slvrfctry.checkTestResults(0, 0, NULL, NULL, NULL, 
						 Teuchos::null, g_sg_local,
						 g_mean_local.get(), 
						 g_std_dev_local.get());
    *out << "\nNumber of Failed Comparisons: " << status << endl;
  }

  TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);
  if (!success) status+=10000;

  Teuchos::TimeMonitor::summarize(*out,false,true,false/*zero timers*/);
  return status;
}
