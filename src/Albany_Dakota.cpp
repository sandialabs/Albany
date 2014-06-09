//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifdef ALBANY_DAKOTA
#include <iostream>
#include "Albany_Dakota.hpp"
#include "Albany_Utils.hpp"

#include "Teuchos_XMLParameterListHelpers.hpp"
#include "TriKota_MPDirectApplicInterface.hpp"
#include "Piro_Epetra_StokhosMPSolver.hpp"

#include "TriKota_Driver.hpp"
#include "TriKota_DirectApplicInterface.hpp"
#include "Albany_SolverFactory.hpp"

// Standard use case for TriKota
//   Dakota is run in library mode with its interface
//   implemented with an EpetraExt::ModelEvaluator
int Albany_Dakota(int argc, char *argv[])
{ // Assumes MPI_Init() already called, and using MPI_COMM_WORLD
  using std::cout;
  using std::endl;
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::FancyOStream;
  using Teuchos::VerboseObjectBase;
  using Teuchos::OSTab;
  using Teuchos::ParameterList;

  const RCP<FancyOStream> out = VerboseObjectBase::getDefaultOStream();

  *out << "\nStarting Albany_Dakota!" << endl;

  // Parse parameters
  std::string xml_filename = "input.xml";
  if(argc>1){
    if(!std::strcmp(argv[1],"--help")){
      *out << "albany [inputfile.xml]\n";
      std::exit(1);
    }
    else
      xml_filename=argv[1];
  }
  RCP<ParameterList> appParams = 
    Teuchos::getParametersFromXmlFile(xml_filename);
  ParameterList& dakotaParams = appParams->sublist("Piro").sublist("Dakota");
  std::string dakota_input_file = 
    dakotaParams.get("Input File", "dakota.in");
  std::string dakota_output_file = 
    dakotaParams.get("Output File", "dakota.out");
  std::string dakota_restart_file = 
    dakotaParams.get("Restart File", "dakota.res");
  std::string dakota_error_file = 
    dakotaParams.get("Error File", "dakota.err");

  std::string dakotaRestartIn;
  std::string dakRestartIn;
  if (dakotaParams.isParameter("Restart File To Read")) {
    dakRestartIn = dakotaParams.get<std::string>("Restart File To Read");
  }
  int dakotaRestartEvals= dakotaParams.get("Restart Evals To Read", 0);

  int p_index = dakotaParams.get("Parameter Vector Index", 0);
  int g_index = dakotaParams.get("Response Vector Index", 0);

  // Construct driver
  TriKota::Driver dakota(dakota_input_file,
			 dakota_output_file,
			 dakota_restart_file,
			 dakota_error_file,
                         dakRestartIn, dakotaRestartEvals );

  
  // Construct a ModelEvaluator for your application with the
  // MPI_Comm chosen by Dakota. This example ModelEvaluator 
  // only takes an input file name and MPI_Comm to construct,
  // and must not be constructed if Dakota assigns MPI_COMM_NULL.

  Albany_MPI_Comm analysis_comm = dakota.getAnalysisComm();
  if (analysis_comm == Albany_MPI_COMM_NULL)
    return 0;
  RCP<Epetra_Comm> appComm = 
      Albany::createEpetraCommFromMpiComm(analysis_comm);
  RCP<Albany::SolverFactory> slvrfctry = 
    rcp(new Albany::SolverFactory(xml_filename, analysis_comm));

  // Construct a concrete Dakota interface with an EpetraExt::ModelEvaluator
  // trikota_interface is freed in the destructor for the Dakota interface class
  RCP<Dakota::DirectApplicInterface> trikota_interface;
  bool use_multi_point = dakotaParams.get("Use Multi-Point", false);
  if (use_multi_point) {
    // Create MP solver
    RCP<ParameterList> mpParams = 
      rcp(&(dakotaParams.sublist("Multi-Point")),false);
    ParameterList& appParams2 = slvrfctry->getParameters();
    RCP<ParameterList> piroParams = 
      rcp(&(appParams2.sublist("Piro")),false);
    RCP<Piro::Epetra::StokhosMPSolver> mp_solver =
      rcp(new Piro::Epetra::StokhosMPSolver(
	    piroParams, mpParams, appComm,
	    mpParams->get("Block Size", 10),
	    mpParams->get("Number of Spatial Processors", -1)));

    // Create application & model evaluator
    Teuchos::RCP<Albany::Application> app;
    Teuchos::RCP<EpetraExt::ModelEvaluator> model = 
      slvrfctry->createAlbanyAppAndModel(app, appComm);

    // Setup rest of solver
    mp_solver->setup(model);
    
    trikota_interface = 
      rcp(new TriKota::MPDirectApplicInterface(dakota.getProblemDescDB(), 
					       mp_solver, p_index, g_index), 
	  false);
  }
  else {
    RCP<EpetraExt::ModelEvaluator> App = slvrfctry->create(appComm, appComm);
    trikota_interface =
      rcp(new TriKota::DirectApplicInterface(dakota.getProblemDescDB(), App, 
					     p_index, g_index), 
	  false);
  } 

  // Run the requested Dakota strategy using this interface
  dakota.run(trikota_interface.get());

  if (dakota.rankZero()) {
    Dakota::RealVector finalValues = 
      dakota.getFinalSolution().continuous_variables();
    *out << "\nAlbany_Dakota: Final Values from Dakota = " 
         << std::setprecision(8) << finalValues << endl;

    return slvrfctry->checkDakotaTestResults(0, &finalValues);
  }
  else return 0;
}
#endif
