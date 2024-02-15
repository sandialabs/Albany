//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <iostream>

#include "Albany_RegressionTests.hpp"
#include "Albany_SolverFactory.hpp"
#include "Albany_ObserverImpl.hpp"
#include "Albany_FactoriesHelpers.hpp"
#include "Albany_Utils.hpp"
#include "Albany_CommUtils.hpp"
#include "Albany_StringUtils.hpp"

#include <Piro_PerformAnalysis.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_StackedTimer.hpp>
#include <Teuchos_TimeMonitor.hpp>
#include <Teuchos_VerboseObject.hpp>
#include <Teuchos_StandardCatchMacros.hpp>

int main(int argc, char *argv[]) {

  int failures(0);
  bool success = true;

  Teuchos::GlobalMPISession mpiSession(&argc, &argv, nullptr);

  Kokkos::initialize(argc, argv);

  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());

  // Command-line argument for input file
  Albany::CmdLineArgs cmd("inputAnalysis.yaml");
  cmd.parse_cmdline(argc, argv, *out);

  Albany::PrintHeader(*out);

  bool reportTimers = true;
  const auto stackedTimer = Teuchos::rcp(
      new Teuchos::StackedTimer("Albany Total Time"));
  Teuchos::TimeMonitor::setStackedTimer(stackedTimer);

  try {
    *out << "\nStarting Albany Analysis via Piro!" << std::endl;
    stackedTimer->start("Albany: Setup Time");

    Teuchos::RCP<const Teuchos_Comm> comm = Albany::getDefaultComm();

    // Connect vtune for performance profiling
    if (cmd.vtune) {
      Albany::connect_vtune(comm->getRank());
    }

    Albany::SolverFactory slvrfctry (cmd.yaml_filename, comm);
    Teuchos::ParameterList &debugParams =
        slvrfctry.getParameters()->sublist("Debug Output", true);
    reportTimers = debugParams.get<bool>("Report Timers", true);

    const bool reportMPIInfo = debugParams.get<bool>("Report MPI Info", false);
    if (reportMPIInfo) Albany::PrintMPIInfo(std::cout);

    auto const& bt = slvrfctry.getParameters()->get<std::string>("Build Type","NONE");

    if (bt=="Tpetra") {
      // Set the static variable that denotes this as a Tpetra run
      static_cast<void>(Albany::build_type(Albany::BuildType::Tpetra));
    } else if (bt=="Epetra") {
      // Set the static variable that denotes this as a Epetra run
      static_cast<void>(Albany::build_type(Albany::BuildType::Epetra));
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidArgument,
                                 "Error! Invalid choice (" + bt + ") for 'BuildType'.\n"
                                 "       Valid choices are 'Epetra', 'Tpetra'.\n");
    }

    // Make sure all the pb factories are registered *before* the Application
    // is created (since in the App ctor the pb factories are queried)
    Albany::register_pb_factories();

    // Create app (null initial guess)
    const auto albanyApp = slvrfctry.createApplication(comm);
    //Forward model model evaluator
    const auto albanyModel = slvrfctry.createModel(albanyApp, false);

    //Adjoint model model evaluator 
    
    const bool explicitMatrixTranspose = slvrfctry.getParameters()->sublist("Piro").isParameter("Enable Explicit Matrix Transpose") ? 
                                           slvrfctry.getParameters()->sublist("Piro").get<bool>("Enable Explicit Matrix Transpose") : 
                                           false;

    const auto albanyAdjointModel = explicitMatrixTranspose ? slvrfctry.createModel(albanyApp, true) : Teuchos::null; 
    const auto solver      = slvrfctry.createSolver(comm, albanyModel, albanyAdjointModel, false);

    stackedTimer->stop("Albany: Setup Time");

    Teuchos::RCP< Thyra::VectorBase<double> > p;

    Teuchos::RCP<Albany::ObserverImpl> observer = Teuchos::rcp( new Albany::ObserverImpl(albanyApp));

    // If no analysis section set in input file, default to simple "Solve"
    std::string analysisPackage = slvrfctry.getAnalysisParameters().get("Analysis Package","Solve");
    if(analysisPackage == "HDSA") {
      const auto distParamLib = albanyApp->getDistributedParameterLibrary();
      auto p_opt = distParamLib->get("param_opt")->vector();
      Teuchos::RCP< Thyra::VectorBase<double> > u_opt = distParamLib->get("solution_opt")->vector()->clone_v();
      std::cout << __FILE__ << ": " << __LINE__ << ", norm(u_opt): " << u_opt->norm_2() << std::endl;

      std::vector<Teuchos::RCP< Thyra::VectorBase<double> > > p_samples, u_diff_at_samples; 
      p_samples.push_back(distParamLib->get("param_sample_0")->vector());
      p_samples.push_back(distParamLib->get("param_sample_1")->vector());
      u_diff_at_samples.push_back(distParamLib->get("solution_diff_sample_0")->vector());
      u_diff_at_samples.push_back(distParamLib->get("solution_diff_sample_1")->vector());
      Piro::PerformAnalysis(*solver, slvrfctry.getParameters()->sublist("Piro"), p, observer, u_opt, p_opt, u_diff_at_samples, p_samples);
    } else
    Piro::PerformAnalysis(*solver, slvrfctry.getParameters()->sublist("Piro"), p, observer);

    Albany::RegressionTests regression(slvrfctry.getParameters());
    auto status = regression.checkAnalysisTestResults(0, p);
    failures = status.first;

    *out << "\nNumber of Comparisons Attempted: " << status.second << std::endl;
    *out << "Number of Failed Comparisons: " << failures << std::endl;
  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);
  if (!success) failures+=10000;

  stackedTimer->stopBaseTimer();
  if (reportTimers) {
    Teuchos::StackedTimer::OutputOptions options;
    options.output_fraction = true;
    options.output_minmax = true;
    stackedTimer->report(std::cout, Teuchos::DefaultComm<int>::getComm(), options);
  }

  Kokkos::finalize();

  return failures;
}
