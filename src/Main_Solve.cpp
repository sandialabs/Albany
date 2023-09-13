//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <iostream>
#include <string>

#include "Albany_RegressionTests.hpp"
#include "Albany_SolverFactory.hpp"
#include "Albany_PiroObserver.hpp"
#include "Albany_Memory.hpp"
#include "Albany_CommUtils.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_Utils.hpp"
#include "Albany_StringUtils.hpp"
#include "Albany_DataTypes.hpp"

#include "Albany_FactoriesHelpers.hpp"

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

#if defined(ALBANY_OMEGAH)
#include "Albany_Omegah.hpp"
#endif

#include "Phalanx_config.hpp"

#include "Kokkos_Core.hpp"

int main(int argc, char *argv[])
{
  int failures(0), comparisons(0);
  bool success = true;

  Teuchos::GlobalMPISession mpiSession(&argc, &argv, nullptr);
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

  using Teuchos::RCP;
  using Teuchos::rcp;

  RCP<Teuchos::FancyOStream> out(
      Teuchos::VerboseObjectBase::getDefaultOStream());

  // Command-line argument for input file
  Albany::CmdLineArgs cmd("input.yaml");
  cmd.parse_cmdline(argc, argv, *out);

  Albany::PrintHeader(*out);

  bool reportTimers = true;
  const auto stackedTimer = Teuchos::rcp(
      new Teuchos::StackedTimer("Albany Total Time"));
  Teuchos::TimeMonitor::setStackedTimer(stackedTimer);
  try {
    stackedTimer->start("Albany: Setup Time");

    RCP<const Teuchos_Comm> comm = Albany::getDefaultComm();

#if defined(ALBANY_OMEGAH)
    Albany::init_omegah_lib(argc,argv,comm);
    fprintf(stderr, "initialized omegah lib\n");
#endif

    // Connect vtune for performance profiling
    if (cmd.vtune) { Albany::connect_vtune(comm->getRank()); }

    Albany::SolverFactory slvrfctry(cmd.yaml_filename, comm);

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

    // Explicit adjoint model is not needed if we are not computing adjoint sensitivities
    const bool explicitAdjointModel = albanyApp->isAdjointSensitivities() && explicitMatrixTranspose;
    const auto albanyAdjointModel = explicitAdjointModel ? slvrfctry.createModel(albanyApp, true) : Teuchos::null; 
    const auto solver      = slvrfctry.createSolver(comm, albanyModel, albanyAdjointModel);

    stackedTimer->stop("Albany: Setup Time");

    std::string solnMethod =
        slvrfctry.getParameters()->sublist("Problem").get<std::string>(
            "Solution Method");
    Teuchos::ParameterList &solveParams =
        slvrfctry.getAnalysisParameters().sublist(
            "Solve", /*mustAlreadyExist =*/false);

    Teuchos::Array<Teuchos::RCP<const Thyra_Vector>> thyraResponses;
    Teuchos::Array<Teuchos::Array<Teuchos::RCP<const Thyra_MultiVector>>> thyraSensitivities;
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

    const Teuchos::ParameterList &parameterParams =
        slvrfctry.getParameters()->sublist("Problem").sublist("Parameters");
    const Teuchos::ParameterList &responseParams =
        slvrfctry.getParameters()->sublist("Problem").sublist(
            "Response Functions");

    int total_num_param_vecs, num_param_vecs, numDistParams;
    Albany::getParameterSizes(parameterParams, total_num_param_vecs, num_param_vecs, numDistParams);

    int num_responses = responseParams.get<int>("Number Of Responses");
    if (responseParams.isType<int>("Number")) {
      int numParameters = responseParams.get<int>("Number");
      num_responses = numParameters;
    }

    Teuchos::Array<Teuchos::RCP<Teuchos::Array<std::string>>> param_names;
    param_names.resize(num_param_vecs);
    for (int l = 0; l < num_param_vecs; ++l) {
      const Teuchos::ParameterList & pList =
          parameterParams.sublist(util::strint("Parameter", l));

      const std::string& parameterType = pList.isParameter("Type") ?
          pList.get<std::string>("Type") : std::string("Scalar");
      if(parameterType == "Scalar") {
        param_names[l] =
            Teuchos::rcp(new Teuchos::Array<std::string>(1));
        (*param_names[l])[0] =
            pList.get<std::string>("Name");
      }
      if(parameterType == "Vector") {
        const int numParameters = pList.get<int>("Dimension");
        TEUCHOS_TEST_FOR_EXCEPTION(
            numParameters == 0,
            Teuchos::Exceptions::InvalidParameter,
            std::endl
                << "Error!  In Albany::ModelEvaluator constructor:  "
                << "Parameter vector "
                << l
                << " has zero parameters!"
                << std::endl);

        param_names[l] =
            Teuchos::rcp(new Teuchos::Array<std::string>(numParameters));
        for (int k = 0; k < numParameters; ++k) {
          (*param_names[l])[k] =
              pList.sublist(util::strint("Scalar", k)).get<std::string>("Name");
        }
      }
    }

    Teuchos::Array<std::string> response_names;
    response_names.resize(num_responses);
    for (int l = 0; l < num_responses; ++l) {
      const Teuchos::ParameterList& pList =
        responseParams.sublist(util::strint("Response", l));

      const std::string& type = pList.isParameter("Type") ?
          pList.get<std::string>("Type") : std::string("Scalar Response");

      if (type=="Sum Of Responses") {
        const int num_sub_responses = pList.get<int>("Number Of Responses");
        TEUCHOS_TEST_FOR_EXCEPTION(
            num_sub_responses == 0, Teuchos::Exceptions::InvalidParameter,
            std::endl
                << "Error!  In Albany::ModelEvaluator constructor:  "
                << "Response vector " << l << " has zero parameters!"
                << std::endl);
        response_names[l] = "Sum Of Responses: ";
        for (int k = 0; k < num_sub_responses; ++k) {
          response_names[l] += pList.sublist(util::strint("Response", k)).get<std::string>("Name");
          if( k != num_sub_responses-1)
            response_names[l] += " + ";
        }
      }
      else
        response_names[l] = pList.get<std::string>("Name");
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
      } else if (num_param_vecs == 0) {
        for (int i = 0; i < num_p; i++) {
          ST norm2 = p_prod->getVectorBlock(i)->norm_2();
          *out << "\nDistributed Parameter " << i << ", (two-norm): "  << norm2 << std::endl;
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

    Albany::RegressionTests regression(slvrfctry.getParameters());

    bool writeToMatrixMarketDgDp = debugParams.get("Write DgDp to MatrixMarket", false);

    //Check/print responses and sensitivities
    for (int i = 0; i < num_g-1; i++) {
      const RCP<const Thyra_Vector> g = thyraResponses[i];
      if (!albanyApp->getResponse(i)->isScalarResponse()) continue;

      *out << "\nResponse " << i << ": " << response_names[i] << "\n";

      Albany::printThyraVector(*out, g);

      //check response
      auto respStatus = regression.checkResponse(i, g);
      failures += respStatus.first;
      comparisons += respStatus.second;

      //check sensitivities
      for (int j = 0; j < num_p; j++) {
        std::pair<int,int> sensStatus(0,0);
        if (thyraSensitivities[i][0].is_null())
          continue;
        Teuchos::RCP<const Thyra_ProductMultiVector> prodvec_thyraSensitivity
          = Teuchos::rcp_dynamic_cast<const Thyra_ProductMultiVector>(thyraSensitivities[i][0]);

        TEUCHOS_TEST_FOR_EXCEPTION (prodvec_thyraSensitivity.is_null() && num_p != 1, 
                                    Teuchos::Exceptions::InvalidParameter,
                                    "Error! thyraSensitivities["<< i <<"][0] is not null and not a Thyra_ProductMultiVector.\n");

        Teuchos::RCP<const Thyra_MultiVector> dgdp = (Teuchos::nonnull(prodvec_thyraSensitivity)) ? prodvec_thyraSensitivity->getMultiVectorBlock(j) : thyraSensitivities[i][0];
        if (Teuchos::nonnull(dgdp)) {
          if (writeToMatrixMarketDgDp) {
            std::string name = "dgdp_" + std::to_string(i) + "_" + std::to_string(j);
            Albany::writeMatrixMarket(dgdp, name);
          }
          if(j < num_param_vecs) {
            Albany::printThyraMultiVector(
                *out << "\nSensitivities (" << i << "," << j << "):\n", dgdp);
            //check response and sensitivities for scalar parameters
            sensStatus = regression.checkSensitivity(i, j, dgdp);
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
            sensStatus = regression.checkSensitivity(i, j, norms);
          }
          failures += sensStatus.first;
          comparisons += sensStatus.second;
        } else { //response only
          std::ostringstream error_msg;
          error_msg << "There are Sensitivity Tests for sensitivity ("
              << i << ", " << j << "), but that sensitivity has not been computed!";
          regression.assertNoSensitivityTests(i,j,error_msg.str());
        }
      }
    }

    // Create debug output object
    if (thyraResponses.size()>0) {
      Teuchos::ParameterList &debugParams =
          slvrfctry.getParameters()->sublist("Debug Output", true);
      bool writeToMatrixMarketDistrSolnMap = debugParams.get(
          "Write Distributed Solution and Map to MatrixMarket", false);

      const RCP<const Thyra_Vector> xfinal = thyraResponses.back();
      auto mnv = Albany::mean(xfinal);
      *out << "\nMain_Solve: MeanValue of final solution " << mnv << std::endl;
      *out << "\nNumber of Comparisons Attempted: " << comparisons << std::endl;
      *out << "Number of Failed Comparisons: " << failures << std::endl;

      if (debugParams.get<bool>("Analyze Memory", false))
        Albany::printMemoryAnalysis(std::cout, comm);

      if (writeToMatrixMarketDistrSolnMap == true) {
        Albany::writeMatrixMarket(xfinal->space(),"xfinal_distributed_map");
        Albany::writeMatrixMarket(xfinal,"xfinal_distributed");
      }
    }
  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);
  if (!success) failures += 10000;

  stackedTimer->stopBaseTimer();
  if (reportTimers) {
    Teuchos::StackedTimer::OutputOptions options;
    options.output_fraction = true;
    options.output_minmax = true;
    stackedTimer->report(std::cout, Teuchos::DefaultComm<int>::getComm(), options);
  }

  Kokkos::finalize();

#if defined(ALBANY_OMEGAH)
    Albany::finalize_omegah_lib();
#endif

  return failures;
}
