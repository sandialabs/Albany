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

    Albany::SolverFactory slvrfctry (cmd.yaml_filename, comm);
    Teuchos::ParameterList &debugParams =
        slvrfctry.getParameters()->sublist("Debug Output", true);
    reportTimers = debugParams.get<bool>("Report Timers", true);

    const bool reportMPIInfo = debugParams.get<bool>("Report MPI Info", false);
    if (reportMPIInfo) Albany::PrintMPIInfo(std::cout);

    // Make sure all the pb factories are registered *before* the Application
    // is created (since in the App ctor the pb factories are queried)
    Albany::register_pb_factories();

    // Create app (null initial guess)
    const auto albanyApp = slvrfctry.createApplication(comm);
    //Forward model model evaluator
    const auto albanyModel = slvrfctry.createModel(albanyApp, false);

    //Adjoint model model evaluator 
    
    const bool explicitMatrixTranspose = slvrfctry.getParameters()->sublist("Piro").isParameter("Enable Explicit Matrix Transpose") 
                                         && slvrfctry.getParameters()->sublist("Piro").get<bool>("Enable Explicit Matrix Transpose");

    const bool transientAnalysis = slvrfctry.getParameters()->sublist("Piro").isSublist("Analysis")
                                   && slvrfctry.getParameters()->sublist("Piro").sublist("Analysis").isParameter("Transient")
                                   && slvrfctry.getParameters()->sublist("Piro").sublist("Analysis").get<bool>("Transient");

    const auto albanyAdjointModel = explicitMatrixTranspose || transientAnalysis ? slvrfctry.createModel(albanyApp, true) : Teuchos::null; 
    const auto solver      = slvrfctry.createSolver(albanyModel, albanyAdjointModel, false);

    stackedTimer->stop("Albany: Setup Time");

    Teuchos::RCP< Thyra::VectorBase<double> > p;

    Teuchos::RCP<Albany::ObserverImpl> observer = Teuchos::rcp( new Albany::ObserverImpl(albanyApp));

    // If no analysis section set in input file, default to simple "Solve"
    std::string analysisPackage = slvrfctry.getAnalysisParameters().get("Analysis Package","Solve");
    if(analysisPackage == "HDSA") {      
      auto& hdsaPList = slvrfctry.getAnalysisParameters().sublist("HDSA");
      if(hdsaPList.isParameter("Normal Random Generator Seed")) {
        int seed = hdsaPList.get<int>("Normal Random Generator Seed");
        hdsaPList.set("Normal Random Generator Seed", seed + comm->getRank());
      }
      const std::string p_sample_root_name = hdsaPList.get("Parameter Sample Root Name", "param_sample");
      const std::string x_diff_root_name = hdsaPList.get("Solution Difference Root Name", "solution_diff_sample");
      const auto distParamLib = albanyApp->getDistributedParameterLibrary();      
      std::vector<Teuchos::RCP< Thyra::VectorBase<double> > > p_samples, u_diff_at_samples;
      
      //we clone these vectors because we don't want them to be modify somewhere else in the code as parameters. 
      //for the HDSA analysis these are data
      int num_samples=hdsaPList.get("Number Of Data Samples",2);
      for(int i=0; i<num_samples; ++i) {
        p_samples.push_back(distParamLib->get(util::strint(p_sample_root_name, i, '_'))->vector()->clone_v());
        u_diff_at_samples.push_back(distParamLib->get(util::strint(x_diff_root_name, i, '_'))->vector()->clone_v());
      }
      Piro::PerformAnalysis(*solver, slvrfctry.getParameters()->sublist("Piro"), p, observer, u_diff_at_samples, p_samples);
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
