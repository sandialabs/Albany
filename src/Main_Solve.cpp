//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <iostream>
#include <string>

#include "Albany_Memory.hpp"
#include "Albany_SolverFactory.hpp"
#include "Albany_Utils.hpp"
#include "Albany_CommUtils.hpp"
#include "Albany_ThyraUtils.hpp"

#include "Piro_PerformSolve.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Teuchos_FancyOStream.hpp"
#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_StackedTimer.hpp"
#include "Teuchos_StandardCatchMacros.hpp"
#include "Teuchos_TimeMonitor.hpp"
#include "Teuchos_VerboseObject.hpp"

#include "Thyra_DefaultProductVector.hpp"
#include "Thyra_DefaultProductVectorSpace.hpp"
#include "Thyra_VectorStdOps.hpp"
#include "Thyra_MultiVectorStdOps.hpp"

#if defined(ALBANY_CHECK_FPE) || defined(ALBANY_STRONG_FPE_CHECK) || defined(ALBANY_FLUSH_DENORMALS)
#include <xmmintrin.h>
#endif

#if defined(ALBANY_CHECK_FPE) || defined(ALBANY_STRONG_FPE_CHECK)
#include <cmath>
#endif

#if defined(ALBANY_FLUSH_DENORMALS)
#include <pmmintrin.h>
#endif

#include "Albany_DataTypes.hpp"

#include "Phalanx_config.hpp"

#include "Kokkos_Core.hpp"

#if defined(ALBANY_APF)
#include "Albany_APFMeshStruct.hpp"
#endif

int main(int argc, char *argv[])
{
  int status = 0;  // 0 = pass, failures are incremented
  bool success = true;

  Teuchos::GlobalMPISession mpiSession(&argc, &argv);
  Kokkos::initialize(argc, argv);

#if defined(ALBANY_FLUSH_DENORMALS)
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif

#if defined(ALBANY_CHECK_FPE)

  _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK()
      & ~_MM_MASK_INVALID);

#elif defined(ALBANY_STRONG_FPE_CHECK)

  _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK()
      & ~( _MM_MASK_INVALID 
           | _MM_MASK_DIV_ZERO 
           | _MM_MASK_OVERFLOW 
//           | _MM_MASK_UNDERFLOW 
         )
      );

#endif

#if defined(ALBANY_APF)
  Albany::APFMeshStruct::initialize_libraries(&argc, &argv);
#endif

  using Teuchos::RCP;
  using Teuchos::rcp;

  RCP<Teuchos::FancyOStream> out(
      Teuchos::VerboseObjectBase::getDefaultOStream());

  // Command-line argument for input file
  Albany::CmdLineArgs cmd("inputT.yaml");
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
    if (cmd.vtune) { Albany::connect_vtune(comm->getRank()); }

    Albany::SolverFactory slvrfctry(cmd.yaml_filename, comm);

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
                                 "       Valid choicses are 'Epetra', 'Tpetra'.\n");
    }

    RCP<Albany::Application> app;
    const RCP<Thyra::ResponseOnlyModelEvaluatorBase<ST>> solver =
        slvrfctry.createAndGetAlbanyApp(app, comm, comm);

    setupTimer = Teuchos::null;

    std::string solnMethod =
        slvrfctry.getParameters().sublist("Problem").get<std::string>(
            "Solution Method");
    if (solnMethod == "Transient Tempus No Piro") {
      TEUCHOS_TEST_FOR_EXCEPTION(
          true, Teuchos::Exceptions::InvalidParameter,
          std::endl
              << "Error!  Please run AlbanyTempus executable with Solution "
                 "Method = Transient Tempus No Piro.\n");
    }
    Teuchos::ParameterList &solveParams =
        slvrfctry.getAnalysisParameters().sublist(
            "Solve", /*mustAlreadyExist =*/false);

    Teuchos::Array<Teuchos::RCP<const Thyra_Vector>> thyraResponses;
    Teuchos::Array<
        Teuchos::Array<Teuchos::RCP<const Thyra_MultiVector>>>
        thyraSensitivities;
    Piro::PerformSolve(
        *solver, solveParams, thyraResponses, thyraSensitivities);

    // Check if thyraResponses are product vectors or regular vectors
    Teuchos::RCP<const Thyra_ProductVector> r_prod;
    if (thyraResponses.size() > 0) {
      r_prod =
          Teuchos::nonnull(thyraResponses[0])
              ? Teuchos::rcp_dynamic_cast<const Thyra_ProductVector>(
                    thyraResponses[0], false)
              : Teuchos::null;
    }

    const int num_p = solver->Np();  // Number of *vectors* of parameters
    int num_g = solver->Ng();        // Number of *vectors* of responses
    if (r_prod != Teuchos::null && num_g > 0) {
      *out << "WARNING: For Thyra::ProductVectorBase, printing of responses "
              "does not work yet!  "
           << "No responses will be printed even though you requested " << num_g
           << " responses. \n";
      num_g = 1;
    }

    *out << "Finished eval of first model: Params, Responses "
         << std::setprecision(12) << std::endl;

    Teuchos::ParameterList &parameterParams =
        slvrfctry.getParameters().sublist("Problem").sublist("Parameters");
    Teuchos::ParameterList &responseParams =
        slvrfctry.getParameters().sublist("Problem").sublist(
            "Response Functions");

    int num_param_vecs = parameterParams.get("Number of Parameter Vectors", 0);
    bool using_old_parameter_list = false;
    if (parameterParams.isType<int>("Number")) {
      int numParameters = parameterParams.get<int>("Number");
      if (numParameters > 0) {
        num_param_vecs = 1;
        using_old_parameter_list = true;
      }
    }

    int num_response_vecs = responseParams.get("Number of Response Vectors", 0);
    bool using_old_response_list = false;
    if (responseParams.isType<int>("Number")) {
      int numParameters = responseParams.get<int>("Number");
      if (numParameters > 0) {
        num_response_vecs = 1;
        using_old_response_list = true;
      }
    }

    Teuchos::Array<Teuchos::RCP<Teuchos::Array<std::string>>> param_names;
    param_names.resize(num_param_vecs);
    for (int l = 0; l < num_param_vecs; ++l) {
      const Teuchos::ParameterList *pList =
          using_old_parameter_list
              ? &parameterParams
              : &(parameterParams.sublist(
                    Albany::strint("Parameter Vector", l)));

      const int numParameters = pList->get<int>("Number");
      TEUCHOS_TEST_FOR_EXCEPTION(
          numParameters == 0, Teuchos::Exceptions::InvalidParameter,
          std::endl
              << "Error!  In Albany::ModelEvaluator constructor:  "
              << "Parameter vector " << l << " has zero parameters!"
              << std::endl);

      param_names[l] =
          Teuchos::rcp(new Teuchos::Array<std::string>(numParameters));
      for (int k = 0; k < numParameters; ++k) {
        (*param_names[l])[k] =
            pList->get<std::string>(Albany::strint("Parameter", k));
      }
    }

    Teuchos::Array<Teuchos::RCP<Teuchos::Array<std::string>>> response_names;
    response_names.resize(num_response_vecs);
    for (int l = 0; l < num_response_vecs; ++l) {
      const Teuchos::ParameterList *pList =
          using_old_response_list
              ? &responseParams
              : &(responseParams.sublist(Albany::strint("Response Vector", l)));

      bool number_exists = pList->getEntryPtr("Number");

      if (number_exists) {
        const int numParameters = pList->get<int>("Number");
        TEUCHOS_TEST_FOR_EXCEPTION(
            numParameters == 0, Teuchos::Exceptions::InvalidParameter,
            std::endl
                << "Error!  In Albany::ModelEvaluator constructor:  "
                << "Response vector " << l << " has zero parameters!"
                << std::endl);

        response_names[l] =
            Teuchos::rcp(new Teuchos::Array<std::string>(numParameters));
        for (int k = 0; k < numParameters; ++k) {
          (*response_names[l])[k] =
              pList->get<std::string>(Albany::strint("Response", k));
        }
      } else {
        response_names[l] = Teuchos::null;
      }
    }

    const Thyra_InArgs nominal = solver->getNominalValues();

    // Check if parameters are product vectors or regular vectors
    Teuchos::RCP<const Thyra_ProductVector> p_prod;
    if (num_p > 0) {
      p_prod =
          Teuchos::nonnull(nominal.get_p(0))
              ? Teuchos::rcp_dynamic_cast<const Thyra_ProductVector>(
                    nominal.get_p(0), false)
              : Teuchos::null;
      if (p_prod == Teuchos::null) {
        // Thyra vector case (default -- for
        // everything except CoupledSchwarz right now
        for (int i = 0; i < num_p; i++) {
          if(i < num_param_vecs)
            Albany::printThyraVector(*out << "\nParameter vector " << i << ":\n", *param_names[i],nominal.get_p(i));
          else { //distributed parameter
            ST norm2 = nominal.get_p(i)->norm_2();
            *out << "\nDistributed Parameter " << i << ", (two-norm): "  << norm2 << std::endl;
          }
        }
      } else {
        // Thyra product vector case
        for (int i = 0; i < num_p; i++) {
          Teuchos::RCP<const Thyra_ProductVector> pT =
              Teuchos::rcp_dynamic_cast<const Thyra_ProductVector>(
                  nominal.get_p(i), true);
          // IKT: note that we are assuming the parameters are all the same for
          // all the models
          // that are being coupled (in CoupledSchwarz) so we print the
          // parameters from the 0th
          // model only.  LOCA does not populate p for more than 1 model at the
          // moment so we cannot
          // allow for different parameters in different models.
          Albany::printThyraVector(
              *out << "\nParameter vector " << i << ":\n", *param_names[i], pT->getVectorBlock(0));
        }
      }
    }

    for (int i = 0; i < num_g - 1; i++) {
      const RCP<const Thyra_Vector> g = thyraResponses[i];
      if (!app->getResponse(i)->isScalarResponse()) continue;

      if (response_names[i] != Teuchos::null) {
        *out << "\nResponse vector " << i << ": " << *response_names[i]
             << "\n";
      } else {
        *out << "\nResponse vector " << i << ":\n";
      }
      Albany::printThyraVector(*out, g);

      if (num_p == 0)  
          status += slvrfctry.checkSolveTestResults(i, 0, g, Teuchos::null);
      for (int j=0; j<num_p; j++) {
        Teuchos::RCP<const Thyra_MultiVector> dgdp = thyraSensitivities[i][j];
        if (Teuchos::nonnull(dgdp)) {
          if(j < num_param_vecs) {
            Albany::printThyraMultiVector(
                *out << "\nSensitivities (" << i << "," << j << "):\n", dgdp);
                //check response and sensitivities for scalar parameters
                status += slvrfctry.checkSolveTestResults(i, j, g, dgdp);
          }
          else {
            auto small_vs = dgdp->domain()->smallVecSpcFcty()->createVecSpc(1);
            auto norms = Thyra::createMembers(small_vs,dgdp->domain()->dim());
            auto norms_vals = Albany::getNonconstLocalData(norms);
            *out << "\nSensitivities (" << i << "," << j  << ") for Distributed Parameters:  (two-norm)\n";
            *out << "    ";
            for(int ir=0; ir<dgdp->domain()->dim(); ++ir) {
              auto norm2 = dgdp->col(ir)->norm_2();
              norms_vals[ir][0] = norm2;
                *out << "    " << norm2;
            }
            *out << "\n" << std::endl;
            //check response and sensitivities for distributed parameters
            status += slvrfctry.checkSolveTestResults(i, j, g, norms);
          }
        }
        else //check response only, no sensitivities
          status += slvrfctry.checkSolveTestResults(i, 0, g, Teuchos::null);
      }
    }

    // Create debug output object
    if (thyraResponses.size()>0) {
      Teuchos::ParameterList &debugParams =
          slvrfctry.getParameters().sublist("Debug Output", true);
      bool writeToMatrixMarketSoln =
          debugParams.get("Write Solution to MatrixMarket", false);
      bool writeToMatrixMarketDistrSolnMap = debugParams.get(
          "Write Distributed Solution and Map to MatrixMarket", false);
      bool writeToCoutSoln =
          debugParams.get("Write Solution to Standard Output", false);

      const RCP<const Thyra_Vector> xfinal = thyraResponses.back();
      auto mnv = Albany::mean(xfinal);
      *out << "\nMain_Solve: MeanValue of final solution " << mnv << std::endl;
      *out << "\nNumber of Failed Comparisons: " << status << std::endl;
      if (writeToCoutSoln == true) {
        Albany::printThyraVector(*out << "\nxfinal:\n", xfinal);
      }

      if (debugParams.get<bool>("Analyze Memory", false))
        Albany::printMemoryAnalysis(std::cout, comm);

      if (writeToMatrixMarketSoln == true) {
        Albany::writeMatrixMarket(xfinal,"xfinal");
      }
      if (writeToMatrixMarketDistrSolnMap == true) {
        Albany::writeMatrixMarket(xfinal->space(),"xfinal_distributed_map");
      }
    }
  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);
  if (!success) status += 10000;

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
