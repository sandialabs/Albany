//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <iostream>

#include "Albany_Utils.hpp"
#include "Albany_CommUtils.hpp"
#include "Albany_SolverFactory.hpp"

#include <Piro_PerformAnalysis.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_StackedTimer.hpp>
#include <Teuchos_TimeMonitor.hpp>
#include <Teuchos_VerboseObject.hpp>
#include <Teuchos_StandardCatchMacros.hpp>

int main(int argc, char *argv[]) {

  int status=0; // 0 = pass, failures are incremented
  bool success = true;

  Teuchos::GlobalMPISession mpiSession(&argc,&argv);

  Kokkos::initialize(argc, argv);

  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());

  // Command-line argument for input file
  Albany::CmdLineArgs cmd("inputAnalysis.yaml");
  cmd.parse_cmdline(argc, argv, *out);

  Albany::PrintHeader(*out);

  const auto stackedTimer = Teuchos::rcp(
      new Teuchos::StackedTimer("Albany Total Time"));
  Teuchos::TimeMonitor::setStackedTimer(stackedTimer);

  try {
    *out << "\nStarting Albany Analysis via Piro!" << std::endl;

    Teuchos::RCP<const Teuchos_Comm> comm = Albany::getDefaultComm();

    // Connect vtune for performance profiling
    if (cmd.vtune) {
      Albany::connect_vtune(comm->getRank());
    }

    Albany::SolverFactory slvrfctry (cmd.yaml_filename, comm);

    const auto& bt = slvrfctry.getParameters().get("Build Type","Tpetra");
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

    Teuchos::RCP<Thyra::ResponseOnlyModelEvaluatorBase<ST> > appThyra = slvrfctry.create(comm, comm);

    Teuchos::RCP< Thyra::VectorBase<double> > p;

    // If no analysis section set in input file, default to simple "Solve"
    std::string analysisPackage = slvrfctry.getAnalysisParameters().get("Analysis Package","Solve");
    status = Piro::PerformAnalysis(*appThyra, slvrfctry.getAnalysisParameters(), p); 

    status = slvrfctry.checkAnalysisTestResults(0, p);

    // Regression comparisons for Dakota runs only valid on Proc 0.
    if (mpiSession.getRank()>0)  status=0;
    else *out << "\nNumber of Failed Comparisons: " << status << std::endl;
  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);
  if (!success) status+=10000;

  stackedTimer->stop("Albany Total Time");
  Teuchos::StackedTimer::OutputOptions options;
  options.output_fraction = true;
  options.output_minmax = true;
  stackedTimer->report(std::cout, Teuchos::DefaultComm<int>::getComm(), options);

  Kokkos::finalize_all();

  return status;
}
