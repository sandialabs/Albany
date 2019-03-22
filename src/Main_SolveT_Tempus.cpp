//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <iostream>
#include <string>

#include "Albany_Utils.hpp"
#include "Albany_SolverFactory.hpp"
#include "Albany_Memory.hpp"
#include "Albany_CombineAndScatterManager.hpp"
#include "Albany_CommUtils.hpp"
#include "Albany_ThyraUtils.hpp"

#include "Piro_PerformSolve.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_StackedTimer.hpp"
#include "Teuchos_TimeMonitor.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Teuchos_StandardCatchMacros.hpp"
#include "Teuchos_FancyOStream.hpp"
#include "Thyra_DefaultProductVector.hpp"
#include "Thyra_DefaultProductVectorSpace.hpp"

#include "Tempus_IntegratorBasic.hpp" 
#include "Piro_ObserverToTempusIntegrationObserverAdapter.hpp"

// Uncomment for run time nan checking
// This is set in the toplevel CMakeLists.txt file
//#define ALBANY_CHECK_FPE

#ifdef ALBANY_CHECK_FPE
#include <math.h>
//#include <Accelerate/Accelerate.h>
#include <xmmintrin.h>
#endif
//#define ALBANY_FLUSH_DENORMALS
#ifdef ALBANY_FLUSH_DENORMALS
#include <xmmintrin.h>
#include <pmmintrin.h>
#endif

#include "Albany_DataTypes.hpp"

#include "Phalanx_config.hpp"

#include "Kokkos_Core.hpp"

#ifdef ALBANY_APF
#include "Albany_APFMeshStruct.hpp"
#endif

int main(int argc, char *argv[]) {

  // Global variable that denotes this is the Tpetra executable
  static_cast<void>(Albany::build_type(Albany::BuildType::Tpetra));

  int status=0; // 0 = pass, failures are incremented
  bool success = true;

  Teuchos::GlobalMPISession mpiSession(&argc,&argv);
  Kokkos::initialize(argc, argv);

#ifdef ALBANY_FLUSH_DENORMALS
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif

#ifdef ALBANY_CHECK_FPE
//	_mm_setcsr(_MM_MASK_MASK &~
//		(_MM_MASK_OVERFLOW | _MM_MASK_INVALID | _MM_MASK_DIV_ZERO) );
    _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() & ~_MM_MASK_INVALID);
#endif

#ifdef ALBANY_APF
  Albany::APFMeshStruct::initialize_libraries(&argc, &argv);
#endif

  using Teuchos::RCP;
  using Teuchos::rcp;

  RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());

  // Command-line argument for input file
  Albany::CmdLineArgs cmd;
  cmd.parse_cmdline(argc, argv, *out);

  const auto stackedTimer = Teuchos::rcp(
      new Teuchos::StackedTimer("Albany Stacked Timer"));
  Teuchos::TimeMonitor::setStackedTimer(stackedTimer);

  try {
    auto totalTimer = Teuchos::rcp(new Teuchos::TimeMonitor(
        *Teuchos::TimeMonitor::getNewTimer("Albany: Total Time")));
    auto setupTimer = Teuchos::rcp(new Teuchos::TimeMonitor(
        *Teuchos::TimeMonitor::getNewTimer("Albany: Setup Time")));

    RCP<const Teuchos_Comm> comm = Albany::getDefaultComm();

    // Connect vtune for performance profiling
    if (cmd.vtune) {
      Albany::connect_vtune(comm->getRank());
    }

    Albany::SolverFactory slvrfctry(cmd.xml_filename, comm);
    RCP<Albany::Application> app;
    const RCP<Thyra::ResponseOnlyModelEvaluatorBase<ST> > solver =
      slvrfctry.createAndGetAlbanyAppT(app, comm, comm);
    
    setupTimer = Teuchos::null;

    Teuchos::ParameterList &appPL = slvrfctry.getParameters();
    // Create debug output object
    Teuchos::ParameterList &debugParams =
      appPL.sublist("Debug Output", true);
    bool writeToMatrixMarketSoln = debugParams.get("Write Solution to MatrixMarket", false);
    bool writeToMatrixMarketDistrSolnMap = debugParams.get("Write Distributed Solution and Map to MatrixMarket", false);
    bool writeToCoutSoln = debugParams.get("Write Solution to Standard Output", false);

    std::string solnMethod = appPL.sublist("Problem").get<std::string>("Solution Method"); 
    if (solnMethod == "Transient Tempus No Piro") { 
      //Start of code to use Tempus to perform time-integration without going through Piro
      Teuchos::RCP<Thyra::ModelEvaluator<ST>> model = slvrfctry.returnModelT();
      Teuchos::RCP<Teuchos::ParameterList> tempusPL = Teuchos::null; 
      if (appPL.sublist("Piro").isSublist("Tempus")) {
        tempusPL = Teuchos::rcp(&(appPL.sublist("Piro").sublist("Tempus")), false); 
      }
      else {
        TEUCHOS_TEST_FOR_EXCEPTION(true,
          Teuchos::Exceptions::InvalidParameter,
          std::endl << "Error!  No Tempus sublist when attempting to run problem with Transient Tempus No Piro " <<
          "Solution Method. " << std::endl);
      }   
      auto integrator = Tempus::integratorBasic<double>(tempusPL, model);
      auto piro_observer = slvrfctry.returnObserverT(); 
      Teuchos::RCP<Tempus::IntegratorObserver<double> > tempus_observer = Teuchos::null;
      if (Teuchos::nonnull(piro_observer)) {
        auto solutionHistory = integrator->getSolutionHistory();
        auto timeStepControl = integrator->getTimeStepControl();
        tempus_observer = Teuchos::rcp(new Piro::ObserverToTempusIntegrationObserverAdapter<double>(solutionHistory, 
                                           timeStepControl, piro_observer));
      }
      if (Teuchos::nonnull(tempus_observer)) {
        integrator->setObserver(tempus_observer); 
        integrator->initialize(); 
      }
      double time = integrator->getTime();
      *out << "\n Final time = " << time << "\n"; 
      Teuchos::RCP<const Thyra_Vector> x = integrator->getX();
      if (writeToCoutSoln == true) {
        Albany::printThyraVector(*out << "\nxfinal = \n", x);
      }
      if (writeToMatrixMarketSoln == true) {
        Teuchos::RCP<const Thyra_VectorSpace> root_vs = Albany::createGatherVectorSpace(Albany::getSpmdVectorSpace(x->range()));
        auto cas_manager = Albany::createCombineAndScatterManager(root_vs,x->range());

        auto x_serial = Thyra::createMember(Teuchos::rcp_implicit_cast<const Thyra_VectorSpace>(root_vs));
        cas_manager->combine(x,x_serial,Albany::CombineMode::INSERT);
        Albany::writeMatrixMarket(x,"xfinal_tempus");
      }
      if (writeToMatrixMarketDistrSolnMap == true) {
        Albany::writeMatrixMarket(x,"xfinal_tempus_distributed");
        Albany::writeMatrixMarket(x->range(),"xfinal_tempus_distributed_map");
      }
      *out << "\n Finish Transient Tempus No Piro time integration!\n";
      //End of code to use Tempus to perform time-integration without going through Piro
    }
    else {
        TEUCHOS_TEST_FOR_EXCEPTION(true, 
            Teuchos::Exceptions::InvalidParameter,
            "\n Error!  AlbanyTempus executable can only be run with 'Transient Tempus No Piro' Solution Method.  " <<
            "You have selected Solution Method = " <<  solnMethod << "\n");
    }
  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);
  if (!success) status+=10000;

  stackedTimer->stop("Albany Stacked Timer");
  Teuchos::StackedTimer::OutputOptions options;
  options.output_fraction = true;
  options.output_minmax = true;
  stackedTimer->report(std::cout, Teuchos::DefaultComm<int>::getComm(), options);

#ifdef ALBANY_APF
  Albany::APFMeshStruct::finalize_libraries();
#endif

  Kokkos::finalize_all();

  return status;
}
