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

#include "Piro_PerformSolve.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_StackedTimer.hpp"
#include "Teuchos_TimeMonitor.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Teuchos_StandardCatchMacros.hpp"
#include "Tpetra_Core.hpp"

#include "Kokkos_Core.hpp"

#ifdef ALBANY_PERIDIGM
#if defined(ALBANY_EPETRA)
#include "PeridigmManager.hpp"
#endif
#endif

// Uncomment for run time nan checking
// This is set in the toplevel CMakeLists.txt file
//
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

#include "Albany_ThyraUtils.hpp"

int main(int argc, char *argv[]) {

  int status=0; // 0 = pass, failures are incremented
  bool success = true;

  // Set the linear algebra type
  Albany::build_type(Albany::BuildType::Epetra);

#ifdef ALBANY_DEBUG
  Teuchos::GlobalMPISession mpiSession(&argc, &argv);
#else // bypass printing process startup info
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL);
#endif

  Kokkos::initialize(argc, argv);

#ifdef ALBANY_FLUSH_DENORMALS
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif

#ifdef ALBANY_CHECK_FPE
   // Catch FPEs. Follow Main_SolveT.cpp's approach to checking for floating
   // point exceptions.
   //_mm_setcsr(_MM_MASK_MASK &~ (_MM_MASK_OVERFLOW | _MM_MASK_INVALID | _MM_MASK_DIV_ZERO) );
   _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() & ~_MM_MASK_INVALID);
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

    RCP<const Teuchos_Comm> comm =
      Tpetra::getDefaultComm();

    // Connect vtune for performance profiling
    if (cmd.vtune) {
      Albany::connect_vtune(comm->getRank());
    }

    Albany::SolverFactory slvrfctry(cmd.xml_filename, comm);
    RCP<const Epetra_Comm> appComm = Albany::createEpetraCommFromTeuchosComm(comm);
    RCP<Albany::Application> app;
    const RCP<Thyra::ModelEvaluator<double> > solver =
      slvrfctry.createThyraSolverAndGetAlbanyApp(app, comm, comm);

    setupTimer = Teuchos::null;

//    PHX::InitializeKokkosDevice();
   
    Teuchos::ParameterList &solveParams =
      slvrfctry.getAnalysisParameters().sublist("Solve", /*mustAlreadyExist =*/ false);
    // By default, request the sensitivities if not explicitly disabled
    solveParams.get("Compute Sensitivities", true);

    Teuchos::Array<Teuchos::RCP<const Thyra::VectorBase<double> > > thyraResponses;
    Teuchos::Array<Teuchos::Array<Teuchos::RCP<const Thyra::MultiVectorBase<double> > > > thyraSensitivities;

       // The PoissonSchrodinger_SchroPo and PoissonSchroMosCap1D tests seg fault as albanyApp is null -
       // For now, do not resize the response vectors. FIXME sort out this issue.
    if(Teuchos::nonnull(app))
      Piro::PerformSolveBase(*solver, solveParams, thyraResponses, thyraSensitivities, app->getAdaptSolMgr()->getSolObserver());
    else
      Piro::PerformSolveBase(*solver, solveParams, thyraResponses, thyraSensitivities);

    const int num_p = solver->Np(); // Number of *vectors* of parameters
    const int num_g = solver->Ng(); // Number of *vectors* of responses

    *out << "Finished eval of first model: Params, Responses "
      << std::setprecision(12) << std::endl;

    Teuchos::ParameterList& parameterParams = slvrfctry.getParameters().sublist("Problem").sublist("Parameters");
    int num_param_vecs = (parameterParams.isType<int>("Number")) ?
        int(parameterParams.get("Number", 0) > 0) :
        parameterParams.get("Number of Parameter Vectors", 0);

    const Thyra::ModelEvaluatorBase::InArgs<double> nominal = solver->getNominalValues();
    double norm2;
    for (int i=0; i<num_p; i++) {
      if(i < num_param_vecs) {
        *out << "\nParameter vector " << i << ":\n";
        Albany::printThyraVector(*out,nominal.get_p(i));
      } else { //distributed parameters, we print only 2-norm
        norm2 = nominal.get_p(i)->norm_2();
        *out << "\nDistributed Parameter " << i << ":  " << norm2 << " (two-norm)\n" << std::endl;
      }
    }

    for (int i=0; i<num_g-1; i++) {
      const RCP<const Thyra_Vector> g = thyraResponses[i];
      bool is_scalar = true;

      if (app != Teuchos::null)
        is_scalar = app->getResponse(i)->isScalarResponse();

      if (is_scalar) {
        *out << "\nResponse vector " << i << ":\n";
        Albany::printThyraVector(*out,g);

        if (num_p == 0) {
          // Just calculate regression data
          status += slvrfctry.checkSolveTestResults(i, 0, g, Teuchos::null);
        } else {
          for (int j=0; j<num_p; j++) {
            const RCP<const Thyra_MultiVector> dgdp = thyraSensitivities[i][j];
            if (Teuchos::nonnull(dgdp)) {
              if(j < num_param_vecs) {
                *out << "\nSensitivities (" << i << "," << j << "): \n";
                Albany::printThyraMultiVector(*out,dgdp);
                status += slvrfctry.checkSolveTestResults(i, j, g, dgdp);
              }
              else {
                auto small_vs = dgdp->domain()->smallVecSpcFcty()->createVecSpc(1);
                auto norms = Thyra::createMembers(small_vs,dgdp->domain()->dim());
                auto norms_vals = Albany::getNonconstLocalData(norms);
                *out << "\nSensitivities (" << i << "," << j  << ") for Distributed Parameters:  (two-norm)\n";
                *out << "    ";
                for(int ir=0; ir<dgdp->domain()->dim(); ++ir) {
                  norm2 = dgdp->col(ir)->norm_2();
                  norms_vals[ir][0] = norm2;
                    *out << "    " << norm2;
                }
                *out << "\n" << std::endl;
                //check response and sensitivities for distributed parameters
                status += slvrfctry.checkSolveTestResults(i, j, g, norms);
              }
            }
          }
        }
      }
    }

    // Create debug output object
    Teuchos::ParameterList &debugParams =
      slvrfctry.getParameters().sublist("Debug Output", true);
    bool writeToMatrixMarketSoln = debugParams.get("Write Solution to MatrixMarket", false);
    bool writeToMatrixMarketDistrSolnMap = debugParams.get("Write Distributed Solution and Map to MatrixMarket", false);
    bool writeToCoutSoln = debugParams.get("Write Solution to Standard Output", false);


    const RCP<const Thyra_Vector> xfinal = thyraResponses.back();
    double mnv = Albany::mean( xfinal );
    *out << "Main_Solve: MeanValue of final solution " << mnv << std::endl;
    *out << "\nNumber of Failed Comparisons: " << status << std::endl;
    if (writeToCoutSoln == true) {
       std::cout << "xfinal: \n";
       Albany::printThyraVector(std::cout,xfinal);
    }

    if (debugParams.get<bool>("Analyze Memory", false)) {
      Albany::printMemoryAnalysis(std::cout, comm);
    }

    if (writeToMatrixMarketSoln == true) { 
      Albany::writeMatrixMarket(xfinal, "xfinal");
    }
    if (writeToMatrixMarketDistrSolnMap == true) {
      Albany::writeMatrixMarket(xfinal, "xfinal");
      Albany::writeMatrixMarket(xfinal->space(), "xfinal_distributed_map");
    }
  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);
  if (!success) status+=10000;
  
  stackedTimer->stop("Albany Stacked Timer");
  Teuchos::StackedTimer::OutputOptions options;
  options.output_fraction = true;
  options.output_minmax = true;
  stackedTimer->report(std::cout, Teuchos::DefaultComm<int>::getComm(), options);

  Kokkos::finalize_all();
 
  return status;
}
