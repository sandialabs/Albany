//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_config.h"

#ifdef ALBANY_DAKOTA

#include "Albany_SolverFactory.hpp"
#include "Albany_Utils.hpp"

#include <Teuchos_YamlParameterListCoreHelpers.hpp>

#include "TriKota_Driver.hpp"
#include "TriKota_ThyraDirectApplicInterface.hpp"
#include "Teuchos_TestForException.hpp"

#include <iostream>

// Standard use case for TriKota
int Albany_Dakota(int argc, char *argv[])
{
  // Assumes MPI_Init() already called, and using MPI_COMM_WORLD
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
  Albany::CmdLineArgs cmd;
  cmd.parse_cmdline(argc, argv, *out);

  RCP<ParameterList> appParams    = Teuchos::getParametersFromYamlFile(cmd.yaml_filename);
  ParameterList&     dakotaParams = appParams->sublist("Piro").sublist("Dakota");

  std::string dakota_input_file   = dakotaParams.get("Input File", "dakota.in");
  std::string dakota_output_file  = dakotaParams.get("Output File", "dakota.out");
  std::string dakota_restart_file = dakotaParams.get("Restart File", "dakota.res");
  std::string dakota_error_file   = dakotaParams.get("Error File", "dakota.err");

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
  if (analysis_comm == Albany_MPI_COMM_NULL) {
    return 0;
  }

  RCP<const Teuchos_Comm> appComm = Albany::createTeuchosCommFromMpiComm(analysis_comm);

  RCP<Albany::SolverFactory> slvrfctry = rcp(new Albany::SolverFactory(cmd.yaml_filename, appComm));
  const auto& bt = slvrfctry->getParameters().get("Build Type","Tpetra");
  if (bt=="Tpetra") {
    // Set the static variable that denotes this as a Tpetra run
    static_cast<void>(Albany::build_type(Albany::BuildType::Tpetra));
  } else if (bt=="Epetra") {
    // Set the static variable that denotes this as a Epetra run
    static_cast<void>(Albany::build_type(Albany::BuildType::Epetra));
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidArgument,
                               "Error! Invalid choice (" + bt + ") for 'BuildType'.\n"
                               "       Valid choicses are 'Epetra', 'Tpetra'.\n");
  }

  // Connect vtune for performance profiling
  if (cmd.vtune) {
    Albany::connect_vtune(appComm->getRank());
  }

  // Construct a concrete Dakota interface with an EpetraExt::ModelEvaluator
  // trikota_interface is freed in the destructor for the Dakota interface class
  RCP<Dakota::DirectApplicInterface> trikota_interface;

  RCP<Thyra::ResponseOnlyModelEvaluatorBase<ST> > app = slvrfctry->create(appComm, appComm);
  trikota_interface = rcp(new TriKota::ThyraDirectApplicInterface(dakota.getProblemDescDB(), app, p_index, g_index), false);

  // Run the requested Dakota strategy using this interface
  dakota.run(trikota_interface.get());

  if (dakota.rankZero()) {
    Dakota::RealVector finalValues = dakota.getFinalSolution().continuous_variables();
    *out << "\nAlbany_DakotaT: Final Values from Dakota = "
         << std::setprecision(8) << finalValues << endl;

    return slvrfctry->checkDakotaTestResults(0, &finalValues);
  } else {
    return 0;
  }
}

#endif // ALBANY_DAKOTA
