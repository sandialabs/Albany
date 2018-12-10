//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
//#define DEBUG

#include "Albany_Application.hpp"
#include "AAdapt_RC_Manager.hpp"
#include "Albany_DiscretizationFactory.hpp"
#include "Albany_ProblemFactory.hpp"
#include "Albany_ResponseFactory.hpp"
#include "Albany_Utils.hpp"
#include "Teuchos_TimeMonitor.hpp"
#include <MatrixMarket_Tpetra.hpp>

#if defined(ALBANY_EPETRA)
#include "EpetraExt_MultiVectorOut.h"
#include "EpetraExt_RowMatrixOut.h"
#include "EpetraExt_VectorOut.h"
#include "Epetra_LocalMap.h"
#include "Petra_Converters.hpp"
#endif

#include "Albany_DataTypes.hpp"
#include <string>

#include "Albany_DummyParameterAccessor.hpp"

#ifdef ALBANY_TEKO
#include "Teko_InverseFactoryOperator.hpp"
#if defined(ALBANY_EPETRA)
#include "Teko_StridedEpetraOperator.hpp"
#endif
#endif

//#if defined(ATO_USES_COGENT)
#ifdef ALBANY_ATO
#if defined(ALBANY_EPETRA)
#include "ATO_XFEM_Preconditioner.hpp"
#endif
#include "ATOT_XFEM_Preconditioner.hpp"
#endif
//#endif

#include "Albany_ScalarResponseFunction.hpp"
#include "PHAL_Utilities.hpp"

#ifdef ALBANY_PERIDIGM
#if defined(ALBANY_EPETRA)
#include "PeridigmManager.hpp"
#endif
#endif

#if defined(ALBANY_LCM)
#include "SolutionSniffer.hpp"
#endif // ALBANY_LCM

#include "Albany_ThyraUtils.hpp"
// TODO: remove this if/when the thyra refactor is 100% complete,
//       and there is no more any Tpetra stuff in this class
#include "Albany_TpetraThyraUtils.hpp"

//#define WRITE_TO_MATRIX_MARKET
//#define DEBUG_OUTPUT
//#define DEBUG_OUTPUT2

using Teuchos::ArrayRCP;
using Teuchos::RCP;
using Teuchos::rcp;
using Teuchos::rcp_dynamic_cast;
using Teuchos::TimeMonitor;
using Teuchos::getFancyOStream;
using Teuchos::rcpFromRef;

int countJac; // counter which counts instances of Jacobian (for debug output)
int countRes; // counter which counts instances of residual (for debug output)
int countScale;
int previous_app;
int current_app;

const Tpetra::global_size_t INVALID =
    Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();

Albany::Application::Application(const RCP<const Teuchos_Comm> &comm_,
                                 const RCP<Teuchos::ParameterList> &params,
                                 const RCP<const Tpetra_Vector> &initial_guess, 
                                 const bool schwarz)
    : commT(comm_), out(Teuchos::VerboseObjectBase::getDefaultOStream()),
      physicsBasedPreconditioner(false), shapeParamsHaveBeenReset(false),
      morphFromInit(true), perturbBetaForDirichlets(0.0), phxGraphVisDetail(0),
      stateGraphVisDetail(0), params_(params), requires_sdbcs_(false),
      requires_orig_dbcs_(false), no_dir_bcs_(false), is_schwarz_{schwarz} {
#if defined(ALBANY_EPETRA)
  comm = Albany::createEpetraCommFromTeuchosComm(comm_);
#endif
  initialSetUp(params);
  createMeshSpecs();
  buildProblem();
  createDiscretization();
  finalSetUp(params, initial_guess);
  prev_times_.resize(1);
#ifdef ALBANY_LCM
  int num_apps = apps_.size();
  if (num_apps == 0) {
    num_apps = 1;
  }
  prev_times_.resize(num_apps);
  for (int i = 0; i < num_apps; ++i) {
    prev_times_[i] = -1.0;
  }
  previous_app = 0;
  current_app = 0;
#endif
}

Albany::Application::Application(const RCP<const Teuchos_Comm> &comm_)
    : commT(comm_), out(Teuchos::VerboseObjectBase::getDefaultOStream()),
      physicsBasedPreconditioner(false), shapeParamsHaveBeenReset(false),
      morphFromInit(true), perturbBetaForDirichlets(0.0), phxGraphVisDetail(0),
      stateGraphVisDetail(0), requires_sdbcs_(false), no_dir_bcs_(false),
      requires_orig_dbcs_(false) {
#if defined(ALBANY_EPETRA)
  comm = Albany::createEpetraCommFromTeuchosComm(comm_);
#endif
#ifdef ALBANY_LCM
  int num_apps = apps_.size();
  if (num_apps == 0)
    num_apps = 1;
  prev_times_.resize(num_apps);
  for (int i = 0; i < num_apps; ++i)
    prev_times_[i] = -1.0;
  previous_app = 0;
  current_app = 0;
#endif
}

namespace {
int calcTangentDerivDimension(
    const Teuchos::RCP<Teuchos::ParameterList> &problemParams) {
  Teuchos::ParameterList &parameterParams =
      problemParams->sublist("Parameters");
  int num_param_vecs = parameterParams.get("Number of Parameter Vectors", 0);
  bool using_old_parameter_list = false;
  if (parameterParams.isType<int>("Number")) {
    int numParameters = parameterParams.get<int>("Number");
    if (numParameters > 0) {
      num_param_vecs = 1;
      using_old_parameter_list = true;
    }
  }
  int np = 0;
  for (int i = 0; i < num_param_vecs; ++i) {
    Teuchos::ParameterList &pList =
        using_old_parameter_list
            ? parameterParams
            : parameterParams.sublist(Albany::strint("Parameter Vector", i));
    np += pList.get<int>("Number");
  }
  return std::max(1, np);
}
} // namespace

void Albany::Application::initialSetUp(
    const RCP<Teuchos::ParameterList> &params) {
  // Create parameter libraries
  paramLib = rcp(new ParamLib);
  distParamLib = rcp(new DistParamLib);

#ifdef ALBANY_DEBUG
#if defined(ALBANY_EPETRA)
  int break_set = (getenv("ALBANY_BREAK") == NULL) ? 0 : 1;
  int env_status = 0;
  int length = 1;
  comm->SumAll(&break_set, &env_status, length);
  if (env_status != 0) {
    *out << "Host and Process Ids for tasks" << std::endl;
    comm->Barrier();
    int nproc = comm->NumProc();
    for (int i = 0; i < nproc; i++) {
      if (i == comm->MyPID()) {
        char buf[80];
        char hostname[80];
        gethostname(hostname, sizeof(hostname));
        sprintf(buf, "Host: %s   PID: %d", hostname, getpid());
        *out << buf << std::endl;
        std::cout.flush();
        sleep(1);
      }
      comm->Barrier();
    }
    if (comm->MyPID() == 0) {
      char go = ' ';
      std::cout << "\n";
      std::cout << "** Client has paused because the environment variable "
                   "ALBANY_BREAK has been set.\n";
      std::cout << "** You may attach a debugger to processes now.\n";
      std::cout << "**\n";
      std::cout << "** Enter a character (not whitespace), then <Return> to "
                   "continue. > ";
      std::cout.flush();
      std::cin >> go;
      std::cout << "\n** Now pausing for 3 seconds.\n";
      std::cout.flush();
    }
    sleep(3);
  }
  comm->Barrier();
#endif
#endif

#if !defined(ALBANY_EPETRA)
  removeEpetraRelatedPLs(params);
#endif

  // Create problem object
  problemParams = Teuchos::sublist(params, "Problem", true);

  Albany::ProblemFactory problemFactory(params, paramLib, commT);
  rc_mgr = AAdapt::rc::Manager::create(Teuchos::rcp(&stateMgr, false),
                                       *problemParams);
  if (Teuchos::nonnull(rc_mgr))
    problemFactory.setReferenceConfigurationManager(rc_mgr);
  problem = problemFactory.create();

  // Validate Problem parameters against list for this specific problem
  problemParams->validateParameters(*(problem->getValidProblemParameters()), 0);

  try {
    tangent_deriv_dim = calcTangentDerivDimension(problemParams);
  } catch (...) {
    tangent_deriv_dim = 1;
  }

  // Pull the number of solution vectors out of the problem and send them to the
  // discretization list, if the user specifies this in the problem
  Teuchos::ParameterList &discParams = params->sublist("Discretization");

  // Set in Albany_AbstractProblem constructor or in siblings
  num_time_deriv = problemParams->get<int>("Number Of Time Derivatives");
  
  // Possibly set in the Discretization list in the input file - this overrides
  // the above if set
  int num_time_deriv_from_input =
      discParams.get<int>("Number Of Time Derivatives", -1);
  if (num_time_deriv_from_input <
      0) // Use the value from the problem by default
    discParams.set<int>("Number Of Time Derivatives", num_time_deriv);
  else
    num_time_deriv = num_time_deriv_from_input;

#ifdef ALBANY_DTK
  if (is_schwarz_ == true) {
    //Write DTK Field to Exodus if Schwarz is used 
    discParams.set<bool>("Output DTK Field to Exodus", true); 
  }
#endif

  TEUCHOS_TEST_FOR_EXCEPTION(
      num_time_deriv > 2, std::logic_error,
      "Input error: number of time derivatives must be <= 2 "
          << "(solution, solution_dot, solution_dotdot)");

  // Save the solution method to be used
  std::string solutionMethod = problemParams->get("Solution Method", "Steady");
  if (solutionMethod == "Steady") {
    solMethod = Steady;
  } else if (solutionMethod == "Continuation") {
    solMethod = Continuation;
    bool const have_piro = params->isSublist("Piro");

    ALBANY_ASSERT(have_piro == true);

    Teuchos::ParameterList &piro_params = params->sublist("Piro");

    bool const have_nox = piro_params.isSublist("NOX");

    if (have_nox) {
      Teuchos::ParameterList nox_params = piro_params.sublist("NOX");
      std::string nonlinear_solver =
          nox_params.get<std::string>("Nonlinear Solver");
    }
  } else if (solutionMethod == "Transient") {
    solMethod = Transient;
  } else if (solutionMethod == "Eigensolve") {
    solMethod = Eigensolve;
  } else if (solutionMethod == "Aeras Hyperviscosity") {
    solMethod = Transient;
  } else if (solutionMethod == "Transient Tempus" ||
             "Transient Tempus No Piro") {
#ifdef ALBANY_TEMPUS
    solMethod = TransientTempus;

    // Add NOX pre-post-operator for debugging.
    bool const have_piro = params->isSublist("Piro");

    ALBANY_ASSERT(have_piro == true);

    Teuchos::ParameterList &piro_params = params->sublist("Piro");

    bool const have_dbcs = problemParams->isSublist("Dirichlet BCs");

    if (have_dbcs == false)
      no_dir_bcs_ = true;

    bool const have_tempus = piro_params.isSublist("Tempus");

    ALBANY_ASSERT(have_tempus == true);

    Teuchos::ParameterList &tempus_params = piro_params.sublist("Tempus");

    bool const have_tempus_stepper = tempus_params.isSublist("Tempus Stepper");

    ALBANY_ASSERT(have_tempus_stepper == true);

    Teuchos::ParameterList &tempus_stepper_params =
        tempus_params.sublist("Tempus Stepper");

    std::string stepper_type =
        tempus_stepper_params.get<std::string>("Stepper Type");

    Teuchos::ParameterList nox_params;

    if ((stepper_type == "Newmark Implicit d-Form") ||
        (stepper_type == "Newmark Implicit a-Form")) {

      bool const have_solver_name =
          tempus_stepper_params.isType<std::string>("Solver Name");

      ALBANY_ASSERT(have_solver_name == true);

      std::string const solver_name =
          tempus_stepper_params.get<std::string>("Solver Name");

      Teuchos::ParameterList &solver_name_params =
          tempus_stepper_params.sublist(solver_name);

      bool const have_nox = solver_name_params.isSublist("NOX");

      ALBANY_ASSERT(have_nox == true);

      nox_params = solver_name_params.sublist("NOX");

      std::string nonlinear_solver =
          nox_params.get<std::string>("Nonlinear Solver");

      // Set flag marking that we are running with Tempus + d-Form Newmark +
      // SDBCs.
      if (stepper_type == "Newmark Implicit d-Form") {
        if (nonlinear_solver != "Line Search Based") {
          TEUCHOS_TEST_FOR_EXCEPTION(
              true, std::logic_error,
              "Newmark Implicit d-Form Stepper Type will not work correctly "
              "with 'Nonlinear Solver' = "
                  << nonlinear_solver
                  << "!  The valid Nonlinear Solver for this scheme is 'Line "
                     "Search Based'.");
        }
      }
      if (stepper_type == "Newmark Implicit a-Form") {
        requires_orig_dbcs_ = true;
      }
    } else if (stepper_type == "Newmark Explicit a-Form") {
      requires_sdbcs_ = true;
    }

#if defined(DEBUG)
    bool const have_solver_opts = nox_params.isSublist("Solver Options");

    ALBANY_ASSERT(have_solver_opts == true);

    Teuchos::ParameterList &solver_opts = nox_params.sublist("Solver Options");

    std::string const ppo_str{"User Defined Pre/Post Operator"};

    bool const have_ppo = solver_opts.isParameter(ppo_str);

    Teuchos::RCP<NOX::Abstract::PrePostOperator> ppo{Teuchos::null};

    if (have_ppo == true) {
      ppo = solver_opts.get<decltype(ppo)>(ppo_str);
    } else {
      ppo = Teuchos::rcp(new LCM::SolutionSniffer);
      solver_opts.set(ppo_str, ppo);
      ALBANY_ASSERT(solver_opts.isParameter(ppo_str) == true);
    }
#endif // DEBUG

#else
    TEUCHOS_TEST_FOR_EXCEPTION(
        true, std::logic_error,
        "Solution Method = "
            << solutionMethod << " is not valid because "
            << "Trilinos was not built with Tempus turned ON.\n");
#endif
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(
        true, std::logic_error,
        "Solution Method must be Steady, Transient, Transient Tempus, "
        "Transient Tempus No Piro, "
            << "Continuation, Eigensolve, or Aeras Hyperviscosity, not : "
            << solutionMethod);
  }

  bool expl = false;
  std::string stepperType;
  if (solMethod == Transient) {
    // Get Piro PL
    Teuchos::RCP<Teuchos::ParameterList> piroParams =
        Teuchos::sublist(params, "Piro", true);
    // Check if there is Rythmos Solver sublist, and get the stepper type
    if (piroParams->isSublist("Rythmos Solver")) {
      Teuchos::RCP<Teuchos::ParameterList> rythmosSolverParams =
          Teuchos::sublist(piroParams, "Rythmos Solver", true);
      if (rythmosSolverParams->isSublist("Rythmos")) {
        Teuchos::RCP<Teuchos::ParameterList> rythmosParams =
            Teuchos::sublist(rythmosSolverParams, "Rythmos", true);
        if (rythmosParams->isSublist("Stepper Settings")) {
          Teuchos::RCP<Teuchos::ParameterList> stepperSettingsParams =
              Teuchos::sublist(rythmosParams, "Stepper Settings", true);
          if (stepperSettingsParams->isSublist("Stepper Selection")) {
            Teuchos::RCP<Teuchos::ParameterList> stepperSelectionParams =
                Teuchos::sublist(stepperSettingsParams, "Stepper Selection",
                                 true);
            stepperType =
                stepperSelectionParams->get("Stepper Type", "Backward Euler");
          }
        }
      }
    }
    // Check if there is Rythmos sublist, and get the stepper type
    else if (piroParams->isSublist("Rythmos")) {
      Teuchos::RCP<Teuchos::ParameterList> rythmosParams =
          Teuchos::sublist(piroParams, "Rythmos", true);
      stepperType = rythmosParams->get("Stepper Type", "Backward Euler");
    }
    // Search for "Explicit" in the stepperType name.  If it's found, set expl
    // to true.
    if (stepperType.find("Explicit") != std::string::npos)
      expl = true;
  } else if (solMethod == TransientTempus) {
    // Get Piro PL
    Teuchos::RCP<Teuchos::ParameterList> piroParams =
        Teuchos::sublist(params, "Piro", true);
    // Check if there is Rythmos Solver sublist, and get the stepper type
    if (piroParams->isSublist("Tempus")) {
      Teuchos::RCP<Teuchos::ParameterList> rythmosSolverParams =
          Teuchos::sublist(piroParams, "Tempus", true);
    }
    // IKT, 10/26/16, FIXME: get whether method is explicit from Tempus
    // parameter list  expl = true;
  }

  determinePiroSolver(params);

  physicsBasedPreconditioner =
      problemParams->get("Use Physics-Based Preconditioner", false);
  if (physicsBasedPreconditioner) {
    precType = problemParams->get("Physics-Based Preconditioner", "Teko");
#ifdef ALBANY_TEKO
    if (precType == "Teko")
      precParams = Teuchos::sublist(problemParams, "Teko", true);
#endif
//#ifdef ATO_USES_COGENT
#ifdef ALBANY_ATO
    if (precType == "XFEM")
      precParams = Teuchos::sublist(problemParams, "XFEM", true);
#endif
    //#endif
  }


  // Create debug output object
  RCP<Teuchos::ParameterList> debugParams =
      Teuchos::sublist(params, "Debug Output", true);
  writeToMatrixMarketJac =
      debugParams->get("Write Jacobian to MatrixMarket", 0);
  computeJacCondNum = debugParams->get("Compute Jacobian Condition Number", 0);
  writeToMatrixMarketRes =
      debugParams->get("Write Residual to MatrixMarket", 0);
  writeToCoutJac = debugParams->get("Write Jacobian to Standard Output", 0);
  writeToCoutRes = debugParams->get("Write Residual to Standard Output", 0);
  derivatives_check_ = debugParams->get<int>("Derivative Check", 0);
  // the above 4 parameters cannot have values < -1
  if (writeToMatrixMarketJac < -1) {
    TEUCHOS_TEST_FOR_EXCEPTION(
        true, Teuchos::Exceptions::InvalidParameter,
        std::endl
            << "Error in Albany::Application constructor:  "
            << "Invalid Parameter Write Jacobian to MatrixMarket.  Acceptable "
               "values are -1, 0, 1, 2, ... "
            << std::endl);
  }
  if (writeToMatrixMarketRes < -1) {
    TEUCHOS_TEST_FOR_EXCEPTION(
        true, Teuchos::Exceptions::InvalidParameter,
        std::endl
            << "Error in Albany::Application constructor:  "
            << "Invalid Parameter Write Residual to MatrixMarket.  Acceptable "
               "values are -1, 0, 1, 2, ... "
            << std::endl);
  }
  if (writeToCoutJac < -1) {
    TEUCHOS_TEST_FOR_EXCEPTION(
        true, Teuchos::Exceptions::InvalidParameter,
        std::endl
            << "Error in Albany::Application constructor:  "
            << "Invalid Parameter Write Jacobian to Standard Output.  "
               "Acceptable values are -1, 0, 1, 2, ... "
            << std::endl);
  }
  if (writeToCoutRes < -1) {
    TEUCHOS_TEST_FOR_EXCEPTION(
        true, Teuchos::Exceptions::InvalidParameter,
        std::endl
            << "Error in Albany::Application constructor:  "
            << "Invalid Parameter Write Residual to Standard Output.  "
               "Acceptable values are -1, 0, 1, 2, ... "
            << std::endl);
  }
  if (computeJacCondNum < -1) {
    TEUCHOS_TEST_FOR_EXCEPTION(
        true, Teuchos::Exceptions::InvalidParameter,
        std::endl
            << "Error in Albany::Application constructor:  "
            << "Invalid Parameter Compute Jacobian Condition Number.  "
               "Acceptable values are -1, 0, 1, 2, ... "
            << std::endl);
  }
  countJac = 0; // initiate counter that counts instances of Jacobian matrix to
                // 0
  countRes = 0; // initiate counter that counts instances of residual vector to
                // 0

  // FIXME: call setScaleBCDofs only on first step rather than at every Newton
  // step.  It's called every step now b/c calling it once did not work for
  // Schwarz problems.
  countScale = 0;
  // Create discretization object
  discFactory = rcp(new Albany::DiscretizationFactory(params, commT, expl));

#if defined(ALBANY_LCM)
  // Check for Schwarz parameters
  bool const has_app_array = params->isParameter("Application Array");

  bool const has_app_index = params->isParameter("Application Index");

  bool const has_app_name_index_map =
      params->isParameter("Application Name Index Map");

  // Only if all these are present set them in the app.
  bool const has_all = has_app_array && has_app_index && has_app_name_index_map;

  if (has_all == true) {
    Teuchos::ArrayRCP<Teuchos::RCP<Albany::Application>> aa =
        params->get<Teuchos::ArrayRCP<Teuchos::RCP<Albany::Application>>>(
            "Application Array");

    int const ai = params->get<int>("Application Index");

    Teuchos::RCP<std::map<std::string, int>> anim =
        params->get<Teuchos::RCP<std::map<std::string, int>>>(
            "Application Name Index Map");

    this->setApplications(aa.create_weak());

    this->setAppIndex(ai);

    this->setAppNameIndexMap(anim);
  }
#endif // ALBANY_LCM

#ifdef ALBANY_PERIDIGM
#if defined(ALBANY_EPETRA)
  LCM::PeridigmManager::initializeSingleton(params);
#endif
#endif
}

void Albany::Application::createMeshSpecs() {
  // Get mesh specification object: worksetSize, cell topology, etc
  meshSpecs = discFactory->createMeshSpecs();
}

void Albany::Application::createMeshSpecs(
    Teuchos::RCP<Albany::AbstractMeshStruct> mesh) {
  // Get mesh specification object: worksetSize, cell topology, etc
  meshSpecs = discFactory->createMeshSpecs(mesh);
}

void Albany::Application::buildProblem() {
#if defined(ALBANY_LCM)
  // This is needed for Schwarz coupling so that when Dirichlet
  // BCs are created we know what application is doing it.
  problem->setApplication(Teuchos::rcp(this, false));
#endif // ALBANY_LCM

  problem->buildProblem(meshSpecs, stateMgr);

  if ((requires_sdbcs_ == true) && (problem->useSDBCs() == false) &&
      (no_dir_bcs_ == false)) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
                               "Error in Albany::Application: you are using a "
                               "solver that requires SDBCs yet you are not "
                               "using SDBCs!\n");
  }

  if ((requires_orig_dbcs_ == true) && (problem->useSDBCs() == true)) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
                               "Error in Albany::Application: you are using a "
                               "solver with SDBCs that does not work correctly "
                               "with them!\n");
  }

  if ((no_dir_bcs_ == true) && (scaleBCdofs == true))
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
                               "Error in Albany::Application: you are attempting "
                               "to set 'Scale DOF BCs = true' for a problem with no  "
                               "Dirichlet BCs!  Scaling will do nothing.  Re-run " 
                               "with 'Scale DOF BCs = false'\n");
  }

  neq = problem->numEquations();
  spatial_dimension = problem->spatialDimension();

  // Construct responses
  // This really needs to happen after the discretization is created for
  // distributed responses, but currently it can't be moved because there
  // are responses that setup states, which has to happen before the
  // discretization is created.  We will delay setup of the distributed
  // responses to deal with this temporarily.
  Teuchos::ParameterList &responseList =
      problemParams->sublist("Response Functions");
  ResponseFactory responseFactory(Teuchos::rcp(this, false), problem, meshSpecs,
                                  Teuchos::rcp(&stateMgr, false));
  responses = responseFactory.createResponseFunctions(responseList);
  observe_responses = responseList.get("Observe Responses", true);
  response_observ_freq = responseList.get("Responses Observation Frequency", 1);
  const Teuchos::Array<unsigned int> defaultDataUnsignedInt;
  relative_responses =
      responseList.get("Relative Responses Markers", defaultDataUnsignedInt);

  // Build state field manager
  if (Teuchos::nonnull(rc_mgr))
    rc_mgr->beginBuildingSfm();
  sfm.resize(meshSpecs.size());
  Teuchos::RCP<PHX::DataLayout> dummy =
      Teuchos::rcp(new PHX::MDALayout<Dummy>(0));
  for (int ps = 0; ps < meshSpecs.size(); ps++) {
    std::string elementBlockName = meshSpecs[ps]->ebName;
    std::vector<std::string> responseIDs_to_require =
        stateMgr.getResidResponseIDsToRequire(elementBlockName);
    sfm[ps] = Teuchos::rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
    Teuchos::Array<Teuchos::RCP<const PHX::FieldTag>> tags =
        problem->buildEvaluators(*sfm[ps], *meshSpecs[ps], stateMgr,
                                 BUILD_STATE_FM, Teuchos::null);
    std::vector<std::string>::const_iterator it;
    for (it = responseIDs_to_require.begin();
         it != responseIDs_to_require.end(); it++) {
      const std::string &responseID = *it;
      PHX::Tag<PHAL::AlbanyTraits::Residual::ScalarT> res_response_tag(
          responseID, dummy);
      sfm[ps]->requireField<PHAL::AlbanyTraits::Residual>(res_response_tag);
    }
  }
  if (Teuchos::nonnull(rc_mgr))
    rc_mgr->endBuildingSfm();
}

void Albany::Application::createDiscretization() {
  // Create the full mesh
  disc = discFactory->createDiscretization(
      neq, problem->getSideSetEquations(), stateMgr.getStateInfoStruct(),
      stateMgr.getSideSetStateInfoStruct(), problem->getFieldRequirements(),
      problem->getSideSetFieldRequirements(), problem->getNullSpace());
  // The following is for Aeras problems.
  explicit_scheme = disc->isExplicitScheme();
}

void Albany::Application::setScaling(
    const Teuchos::RCP<Teuchos::ParameterList> &params) 
{
  // get info from Scaling parameter list (for scaling Jacobian/residual)
  RCP<Teuchos::ParameterList> scalingParams =
      Teuchos::sublist(params, "Scaling", true);
  scale = scalingParams->get<double>("Scale", 0.0);
  scaleBCdofs = scalingParams->get<bool>("Scale BC Dofs", false);
  std::string scaleType = scalingParams->get<std::string>("Type", "Constant");

  if (scale == 0.0) {
    scale = 1.0;
    //If LCM problem with no scale specified and not using SDBCs, set scaleBCdofs = true with diagonal scale type 
    if ((isLCMProblem(params) == true) && (problem->useSDBCs() == false)) {
      scaleBCdofs = true; 
      scaleType = "Diagonal"; 
    }
  }

  if (scaleType == "Constant") {
    scale_type = CONSTANT;
  } else if (scaleType == "Diagonal") {
    scale_type = DIAG;
    scale = 1.0e1;
  } else if (scaleType == "Abs Row Sum") {
    scale_type = ABSROWSUM;
    scale = 1.0e1;
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(
        true, std::logic_error,
        "The scaling Type you selected "
            << scaleType << " is not supported!"
            << "Supported scaling Types are currently: Constant" << std::endl);
  }

  if (scale == 1.0)
    scaleBCdofs = false;

}

bool Albany::Application::isLCMProblem(
    const Teuchos::RCP<Teuchos::ParameterList> &params) const 
{
  //FIXME: fill in this function to check if we have LCM problem 
  return false; 
  /*RCP<Teuchos::ParameterList> problemParams =
      Teuchos::sublist(params, "Problem", true);
  if ((problemParams->get("Name", "Heat 1D") == "Mechanics 2D") ||
      (problemParams->get("Name", "Heat 1D") == "Mechanics 3D") ||
      (problemParams->get("Name", "Heat 1D") == "Elasticity 1D") ||
      (problemParams->get("Name", "Heat 1D") == "Elasticity 2D") ||
      (problemParams->get("Name", "Heat 1D") == "Elasticity 3D") ||
      (problemParams->get("Name", "Heat 1D") == "ThermoElasticity 1D") ||
      (problemParams->get("Name", "Heat 1D") == "ThermoElasticity 2D") ||
      (problemParams->get("Name", "Heat 1D") == "ThermoElasticity 3D") )
  { 
    return true; 
  }
  else 
  {
    return false; 
  }*/
}

void Albany::Application::finalSetUp(
    const Teuchos::RCP<Teuchos::ParameterList> &params,
    const Teuchos::RCP<const Tpetra_Vector> &initial_guess) {

  bool TpetraBuild = Albany::build_type() == Albany::BuildType::Tpetra;

  setScaling(params); 

  /*
   RCP<const Tpetra_Vector> initial_guessT;
   if (Teuchos::nonnull(initial_guess)) {
   initial_guessT = Petra::EpetraVector_To_TpetraVectorConst(*initial_guess,
   commT);
   }
   */

  // Now that space is allocated in STK for state fields, initialize states.
  // If the states have been already allocated, skip this.
  if (!stateMgr.areStateVarsAllocated())
    stateMgr.setupStateArrays(disc);

#if defined(ALBANY_EPETRA)
  if (!TpetraBuild) {
    RCP<Epetra_Vector> initial_guessE;
    if (Teuchos::nonnull(initial_guess)) {
      Petra::TpetraVector_To_EpetraVector(initial_guess, initial_guessE, comm);
    }
    solMgr =
        rcp(new AAdapt::AdaptiveSolutionManager(params, disc, initial_guessE));
  }
#endif

  solMgrT = rcp(new AAdapt::AdaptiveSolutionManagerT(
      params, initial_guess, paramLib, stateMgr,
      // Prevent a circular dependency.
      Teuchos::rcp(rc_mgr.get(), false), commT));
  if (Teuchos::nonnull(rc_mgr))
    rc_mgr->setSolutionManager(solMgrT);

#ifdef ALBANY_PERIDIGM
#if defined(ALBANY_EPETRA)
  if (Teuchos::nonnull(LCM::PeridigmManager::self())) {
    LCM::PeridigmManager::self()->setDirichletFields(disc);
  }
#endif
#endif

  try {
    // dp-todo getNodalParameterSIS() needs to be implemented in PUMI. Until
    // then, catch the exception and continue.
    // Create Distributed parameters and initialize them with data stored in the
    // mesh.
    const Albany::StateInfoStruct &distParamSIS = disc->getNodalParameterSIS();
    for (int is = 0; is < distParamSIS.size(); is++) {
      // Get name of distributed parameter
      const std::string &param_name = distParamSIS[is]->name;

      // Get parameter maps and build parameter vector
      Teuchos::RCP<Tpetra_Vector> dist_paramT, dist_param_lowerboundT, dist_param_upperboundT;
      Teuchos::RCP<const Tpetra_Map> node_mapT, overlap_node_mapT;
      node_mapT = disc->getMapT(
          param_name); // Petra::EpetraMap_To_TpetraMap(node_map, commT);
      overlap_node_mapT = disc->getOverlapMapT(
          param_name); // Petra::EpetraMap_To_TpetraMap(overlap_node_map,
                       // commT);
      dist_paramT = Teuchos::rcp(new Tpetra_Vector(node_mapT));
      dist_param_lowerboundT = Teuchos::rcp(new Tpetra_Vector(node_mapT));
      dist_param_upperboundT = Teuchos::rcp(new Tpetra_Vector(node_mapT));
      std::stringstream lowerbound_name, upperbound_name;
      lowerbound_name << param_name << "_lowerbound";
      upperbound_name << param_name << "_upperbound";

      // Initialize parameter with data stored in the mesh
      disc->getFieldT(*dist_paramT, param_name);
      const auto& nodal_param_states = disc->getNodalParameterSIS();
      bool has_lowerbound(false), has_upperbound(false);
      for (int is = 0; is < nodal_param_states.size(); is++) {
        has_lowerbound = has_lowerbound || (nodal_param_states[is]->name == lowerbound_name.str());
        has_upperbound = has_upperbound || (nodal_param_states[is]->name == upperbound_name.str());
      }
      if(has_lowerbound)
        disc->getFieldT(*dist_param_lowerboundT, lowerbound_name.str() );
      else
        dist_param_lowerboundT->putScalar(std::numeric_limits<ST>::lowest());
      if(has_upperbound)
        disc->getFieldT(*dist_param_upperboundT, upperbound_name.str());
      else
        dist_param_upperboundT->putScalar(std::numeric_limits<ST>::max());

      // JR: for now, initialize to constant value from user input if requested.
      // This needs to be generalized.
      if (params->sublist("Problem").isType<Teuchos::ParameterList>(
              "Topology Parameters")) {
        Teuchos::ParameterList &topoParams =
            params->sublist("Problem").sublist("Topology Parameters");
        if (topoParams.isType<std::string>("Entity Type") &&
            topoParams.isType<double>("Initial Value")) {
          if (topoParams.get<std::string>("Entity Type") ==
                  "Distributed Parameter" &&
              topoParams.get<std::string>("Topology Name") == param_name) {
            double initVal = topoParams.get<double>("Initial Value");
            dist_paramT->putScalar(initVal);
          }
        }
      }

      // Create distributed parameter and set workset_elem_dofs
      Teuchos::RCP<TpetraDistributedParameter> parameter(
          new TpetraDistributedParameter(param_name, dist_paramT, dist_param_lowerboundT, dist_param_upperboundT,
                                         node_mapT, overlap_node_mapT));
      parameter->set_workset_elem_dofs(
          Teuchos::rcpFromRef(disc->getElNodeEqID(param_name)));

      // Add parameter to the distributed parameter library
      distParamLib->add(parameter->name(), parameter);
    }
  } catch (const std::logic_error &) {
  }

  // Now setup response functions (see note above)
  for (int i = 0; i < responses.size(); i++) {
    responses[i]->setup();
  }

  // Set up memory for workset
  fm = problem->getFieldManager();
  TEUCHOS_TEST_FOR_EXCEPTION(fm == Teuchos::null, std::logic_error,
                             "getFieldManager not implemented!!!");
  dfm = problem->getDirichletFieldManager();

  offsets_ = problem->getOffsets();
  nodeSetIDs_ = problem->getNodeSetIDs();

  nfm = problem->getNeumannFieldManager();

  if (commT->getRank() == 0) {
    phxGraphVisDetail =
        problemParams->get("Phalanx Graph Visualization Detail", 0);
    stateGraphVisDetail = phxGraphVisDetail;
  }

  *out << "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n"
       << " Sacado ParameterLibrary has been initialized:\n " << *paramLib
       << "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n"
       << std::endl;

  // Allow Problem to add custom NOX status test
  problem->applyProblemSpecificSolverSettings(params);

  ignore_residual_in_jacobian =
      problemParams->get("Ignore Residual In Jacobian", false);

  perturbBetaForDirichlets = problemParams->get("Perturb Dirichlet", 0.0);

  is_adjoint = problemParams->get("Solve Adjoint", false);

  // For backward compatibility, use any value at the old location of the
  // "Compute Sensitivity" flag as a default value for the new flag location
  // when the latter has been left undefined
  const std::string sensitivityToken = "Compute Sensitivities";
  const Teuchos::Ptr<const bool> oldSensitivityFlag(
      problemParams->getPtr<bool>(sensitivityToken));
  if (Teuchos::nonnull(oldSensitivityFlag)) {
    Teuchos::ParameterList &solveParams =
        params->sublist("Piro").sublist("Analysis").sublist("Solve");
    solveParams.get(sensitivityToken, *oldSensitivityFlag);
  }

  // MPerego: Preforming post registration setup here to make sure that the
  // discretization is already created, so that  derivative dimensions are known.
  // Cannot do post registration right before the evaluate , as done for other
  // field managers.  because memoizer hack is needed by Aeras.
  // TODO, determine when it's best to perform post setup registration and fix
  // memoizer hack if needed.
  for (int i = 0; i < responses.size(); ++i) {
    responses[i]->postRegSetup();
  }

/*
 * Initialize mesh adaptation features
 */

#if defined(ALBANY_EPETRA)
  if (!TpetraBuild && solMgr->hasAdaptation()) {

    solMgr->buildAdaptiveProblem(paramLib, stateMgr, commT);
  }
#endif

#ifdef ALBANY_PERIDIGM
#if defined(ALBANY_EPETRA)
  if (Teuchos::nonnull(LCM::PeridigmManager::self())) {
    LCM::PeridigmManager::self()->initialize(params, disc, commT);
    LCM::PeridigmManager::self()->insertPeridigmNonzerosIntoAlbanyGraph();
  }
#endif
#endif
}

Albany::Application::~Application() {
#ifdef ALBANY_DEBUG
  *out << "Calling destructor for Albany_Application" << std::endl;
#endif
}

RCP<Albany::AbstractDiscretization>
Albany::Application::getDiscretization() const {
  return disc;
}

RCP<Albany::AbstractProblem> Albany::Application::getProblem() const {
  return problem;
}

RCP<const Teuchos_Comm> Albany::Application::getComm() const { return commT; }

#if defined(ALBANY_EPETRA)
RCP<const Epetra_Comm> Albany::Application::getEpetraComm() const {
  return comm;
}

RCP<const Epetra_Map> Albany::Application::getMap() const {
  return disc->getMap();
}
#endif

RCP<const Tpetra_Map> Albany::Application::getMapT() const {
  return disc->getMapT();
}

Teuchos::RCP<const Thyra_VectorSpace> Albany::Application::getVectorSpace() const
{
  return disc->getVectorSpace();
}

#if defined(ALBANY_EPETRA)
RCP<const Epetra_CrsGraph> Albany::Application::getJacobianGraph() const {
  return disc->getJacobianGraph();
}
#endif

RCP<const Tpetra_CrsGraph> Albany::Application::getJacobianGraphT() const {
  return disc->getJacobianGraphT();
}

RCP<Tpetra_Operator> Albany::Application::getPreconditionerT() {
//#if defined(ATO_USES_COGENT)
#ifdef ALBANY_ATO
  if (precType == "XFEM") {
    return rcp(new ATOT::XFEM::Preconditioner(precParams));
  } else
#endif
    //#endif
    return Teuchos::null;
}

#if defined(ALBANY_EPETRA)
RCP<Epetra_Operator> Albany::Application::getPreconditioner() {
#if defined(ALBANY_TEKO)
  if (precType == "Teko") {
    // inverseLib = Teko::InverseLibrary::buildFromStratimikos();
    inverseLib = Teko::InverseLibrary::buildFromParameterList(
        precParams->sublist("Inverse Factory Library"));
    inverseLib->PrintAvailableInverses(*out);

    inverseFac = inverseLib->getInverseFactory(
        precParams->get("Preconditioner Name", "Amesos"));

    // get desired blocking of unknowns
    std::stringstream ss;
    ss << precParams->get<std::string>("Unknown Blocking");

    // figure out the decomposition requested by the string
    unsigned int num = 0, sum = 0;
    while (not ss.eof()) {
      ss >> num;
      TEUCHOS_ASSERT(num > 0);
      sum += num;
      blockDecomp.push_back(num);
    }
    TEUCHOS_ASSERT(neq == sum);

    return rcp(new Teko::Epetra::InverseFactoryOperator(inverseFac));
  } else
#endif
//#if defined(ATO_USES_COGENT)
#ifdef ALBANY_ATO
      if (precType == "XFEM") {
    return rcp(new ATO::XFEM::Preconditioner(precParams));
  } else
#endif
    //#endif
    return Teuchos::null;
}

RCP<const Epetra_Vector> Albany::Application::getInitialSolution() const {
  const Teuchos::RCP<const Tpetra_MultiVector> xMV =
      solMgrT->getInitialSolution();
  Teuchos::RCP<const Tpetra_Vector> xT = xMV->getVector(0);

  const Teuchos::RCP<Epetra_Vector> &initial_x = solMgr->get_initial_x();
  Petra::TpetraVector_To_EpetraVector(xT, *initial_x, comm);
  return initial_x;
}

RCP<const Epetra_Vector> Albany::Application::getInitialSolutionDot() const {
  const Teuchos::RCP<const Tpetra_MultiVector> xMV =
      solMgrT->getInitialSolution();
  if (xMV->getNumVectors() < 2)
    return Teuchos::null;
  Teuchos::RCP<const Tpetra_Vector> xdotT = xMV->getVector(1);

  const Teuchos::RCP<Epetra_Vector> &initial_x_dot = solMgr->get_initial_xdot();
  Petra::TpetraVector_To_EpetraVector(xdotT, *initial_x_dot, comm);
  return initial_x_dot;
}

RCP<const Epetra_Vector> Albany::Application::getInitialSolutionDotDot() const {
  const Teuchos::RCP<const Tpetra_MultiVector> xMV =
      solMgrT->getInitialSolution();
  if (xMV->getNumVectors() < 3)
    return Teuchos::null;
  Teuchos::RCP<const Tpetra_Vector> xdotdotT = xMV->getVector(2);

  const Teuchos::RCP<Epetra_Vector> &initial_x_dotdot =
      solMgr->get_initial_xdotdot();
  Petra::TpetraVector_To_EpetraVector(xdotdotT, *initial_x_dotdot, comm);
  return initial_x_dotdot;
}
#endif

RCP<ParamLib> Albany::Application::getParamLib() const { return paramLib; }

RCP<DistParamLib> Albany::Application::getDistParamLib() const {
  return distParamLib;
}

int Albany::Application::getNumResponses() const { return responses.size(); }

Teuchos::RCP<Albany::AbstractResponseFunction>
Albany::Application::getResponse(int i) const {
  return responses[i];
}

bool Albany::Application::suppliesPreconditioner() const {
  return physicsBasedPreconditioner;
}

namespace {
// amb-nfm I think right now there is some confusion about nfm. Long ago, nfm
// was
// like dfm, just a single field manager. Then it became an array like fm. At
// that time, it may have been true that nfm was indexed just like fm, using
// wsPhysIndex. However, it is clear at present (7 Nov 2014) that nfm is
// definitely not indexed like fm. As an example, compare nfm in
// Albany::MechanicsProblem::constructNeumannEvaluators and fm in
// Albany::MechanicsProblem::buildProblem. For now, I'm going to keep nfm as an
// array, but this this new function is a wrapper around the unclear intended
// behavior.
inline Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>> &deref_nfm(
    Teuchos::ArrayRCP<Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>>> &nfm,
    const Albany::WorksetArray<int>::type &wsPhysIndex, int ws) {
  return nfm.size() == 1 ? // Currently, all problems seem to have one nfm ...
             nfm[0]
                         : // ... hence this is the intended behavior ...
             nfm[wsPhysIndex[ws]]; // ... and this is not, but may one day be
                                   // again.
}

// Convenience routine for setting dfm workset data. Cut down on redundant code.
void dfm_set(PHAL::Workset &workset,
             const Teuchos::RCP<const Thyra_Vector> &x,
             const Teuchos::RCP<const Thyra_Vector> &xd,
             const Teuchos::RCP<const Thyra_Vector> &xdd,
             Teuchos::RCP<AAdapt::rc::Manager> &rc_mgr) {
  workset.x       = Teuchos::nonnull(rc_mgr) ? rc_mgr->add_x(x) : x;
  workset.xdot    = Teuchos::nonnull(rc_mgr) ? rc_mgr->add_x(xd) : xd;
  workset.xdotdot = Teuchos::nonnull(rc_mgr) ? rc_mgr->add_x(xdd) : xdd;
  workset.transientTerms = Teuchos::nonnull(xd);
  workset.accelerationTerms = Teuchos::nonnull(xdd);
}

// For the perturbation xd,
//     f_i(x + xd) = f_i(x) + J_i(x) xd + O(xd' H_i(x) xd),
// where J_i is the i'th row of the Jacobian matrix and H_i is the Hessian of
// f_i at x. We don't have the Hessian, however, so approximate the last term by
// norm(f) O(xd' xd). We use the inf-norm throughout.
//   For check_lvl >= 1, check that f(x + xd) - f(x) is approximately equal to
// J(x) xd by computing
//     reldif(f(x + dx) - f(x), J(x) dx)
//        = norm(f(x + dx) - f(x) - J(x) dx) /
//          max(norm(f(x + dx) - f(x)), norm(J(x) dx)).
// This norm should be on the order of norm(xd).
//   For check_lvl >= 2, output a multivector in matrix market format having
// columns
//     [x, dx, f(x), f(x + dx) - f(x), f(x + dx) - f(x) - J(x) dx].
//   The purpose of this derivative checker is to help find programming errors
// in the Jacobian. Automatic differentiation largely or entirely prevents math
// errors, but other kinds of programming errors (uninitialized memory,
// accidental omission of a FadType, etc.) can cause errors. The common symptom
// of such an error is that the residual is correct, and therefore so is the
// solution, but convergence to the solution is not quadratic.
//   A complementary method to check for errors in the Jacobian is to use
//     Piro -> Jacobian Operator = Matrix-Free,
// which works for Epetra-based problems.
//   Enable this check using the debug block:
//     <ParameterList>
//       <ParameterList name="Debug Output">
//         <Parameter name="Derivative Check" type="int" value="1"/>
void checkDerivatives(Albany::Application &app, const double time,
                      const Teuchos::RCP<const Thyra_Vector>& x,
                      const Teuchos::RCP<const Thyra_Vector>& xdot,
                      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
                      const Teuchos::Array<ParamVec> &p,
                      const Teuchos::RCP<const Thyra_Vector>& fi,
                      const Teuchos::RCP<const Thyra_LinearOp>& jacobian,
                      const int check_lvl) {
  if (check_lvl <= 0) {
    return;
  }

  // Work vectors. x's map is compatible with f's, so don't distinguish among
  // maps in this function.
  Teuchos::RCP<Thyra_Vector> w1 = Thyra::createMember(x->space());
  Teuchos::RCP<Thyra_Vector> w2 = Thyra::createMember(x->space());
  Teuchos::RCP<Thyra_Vector> w3 = Thyra::createMember(x->space());

  Teuchos::RCP<Thyra_MultiVector> mv;
  if (check_lvl > 1) {
    mv = Thyra::createMembers(x->space(), 5);
  }

  // Construct a perturbation.
  const double delta = 1e-7;
  Teuchos::RCP<Thyra_Vector> xd = w1;
  xd->randomize(-Teuchos::ScalarTraits<ST>::rmax(),Teuchos::ScalarTraits<ST>::rmax());
  Teuchos::RCP<Thyra_Vector> xpd = w2;
  {
    const Teuchos::ArrayRCP<const RealType> x_d = Albany::getLocalData(x);
    const Teuchos::ArrayRCP<RealType> xd_d = Albany::getNonconstLocalData(xd);
    const Teuchos::ArrayRCP<RealType> xpd_d = Albany::getNonconstLocalData(xpd);
    for (size_t i = 0; i < x_d.size(); ++i) {
      xd_d[i] = 2 * xd_d[i] - 1;
      const double xdi = xd_d[i];
      if (x_d[i] == 0) {
        // No scalar-level way to get the magnitude of x_i, so just go with
        // something:
        xd_d[i] = xpd_d[i] = delta * xd_d[i];
      } else {
        // Make the perturbation meaningful relative to the magnitude of x_i.
        xpd_d[i] = (1 + delta * xd_d[i]) * x_d[i]; // mult line
        // Sanitize xd_d.
        xd_d[i] = xpd_d[i] - x_d[i];
        if (xd_d[i] == 0) {
          // Underflow in "mult line" occurred because x_d[i] is something like
          // 1e-314. That's a possible sign of uninitialized memory. However,
          // carry on here to get a good perturbation by reverting to the
          // no-magnitude case:
          xd_d[i] = xpd_d[i] = delta * xd_d[i];
        }
      }
    }
  }
  if (Teuchos::nonnull(mv)) {
    Albany::scale_and_update(mv->col(0),0.0,x,1.0);
    Albany::scale_and_update(mv->col(1),0.0,xd,1.0);
  }

  // If necessary, compute f(x).
  Teuchos::RCP<const Thyra_Vector> f;
  if (fi.is_null()) {
    Teuchos::RCP<Thyra_Vector> tmp = Thyra::createMember(x->space());
    app.computeGlobalResidual(time, x, xdot, xdotdot, p, tmp);
    f = tmp;
  } else {
    f = fi;
  }
  if (Teuchos::nonnull(mv)) {
    mv->col(2)->assign(0);
    mv->col(2)->update(1.0,*f);
  }

  // fpd = f(xpd).
  Teuchos::RCP<Thyra_Vector> fpd = w3;
  app.computeGlobalResidual(time, xpd, xdot, xdotdot, p, fpd);

  // fd = fpd - f.
  Teuchos::RCP<Thyra_Vector> fd = fpd;
  Albany::scale_and_update(fpd,1.0,f,-1.0);
  if (Teuchos::nonnull(mv)) {
    Albany::scale_and_update(mv->col(3),0.0,fd,1.0);
  }

  // Jxd = J xd.
  Teuchos::RCP<Thyra_Vector> Jxd = w2;
  jacobian->apply(Thyra::NOTRANS,*xd, Jxd.ptr(),1.0,0.0);

  // Norms.
  const ST fdn = fd->norm_inf();
  const ST Jxdn = Jxd->norm_inf();
  const ST xdn = xd->norm_inf();
  // d = norm(fd - Jxd).
  Teuchos::RCP<Thyra_Vector> d = fd;
  Albany::scale_and_update(d,1.0,Jxd,-1.0);
  if (Teuchos::nonnull(mv)) {
    Albany::scale_and_update(mv->col(4),0.0,d,1.0);
  }
  const double dn = d->norm_inf();
 
  // Assess.
  const double den = std::max(fdn, Jxdn), e = dn / den;
  *Teuchos::VerboseObjectBase::getDefaultOStream()
      << "Albany::Application Check Derivatives level " << check_lvl << ":\n"
      << "   reldif(f(x + dx) - f(x), J(x) dx) = " << e
      << ",\n which should be on the order of " << xdn << "\n";

  if (Teuchos::nonnull(mv)) {
    static int ctr = 0;
    std::stringstream ss;
    ss << "dc" << ctr << ".mm";
    Albany::writeMatrixMarket(mv.getConst(),"dc",ctr);
    ++ctr;
  }
}
} // namespace

void Albany::Application::computeGlobalResidualImpl(
    double const current_time,
    const Teuchos::RCP<const Thyra_Vector> x,
    const Teuchos::RCP<const Thyra_Vector> x_dot,
    const Teuchos::RCP<const Thyra_Vector> x_dotdot,
    Teuchos::Array<ParamVec> const &p,
    const Teuchos::RCP<Thyra_Vector>& f,
    double dt) {
  TEUCHOS_FUNC_TIME_MONITOR("> Albany Fill: Residual");
  postRegSetup("Residual");

  // Load connectivity map and coordinates
  const auto &wsElNodeEqID = disc->getWsElNodeEqID();
  const auto &coords = disc->getCoords();
  const auto &wsEBNames = disc->getWsEBNames();
  const auto &wsPhysIndex = disc->getWsPhysIndex();

  int const numWorksets = wsElNodeEqID.size();

  const Teuchos::RCP<Thyra_Vector> overlapped_f = solMgrT->get_overlapped_f();

  Teuchos::RCP<const CombineAndScatterManager> cas_manager = solMgrT->get_cas_manager();

  // Scatter x and xdot to the overlapped distrbution
  solMgrT->scatterX(x, x_dot, x_dotdot);

  // Scatter distributed parameters
  distParamLib->scatter();

  // Set parameters
  for (int i = 0; i < p.size(); i++) {
    for (unsigned int j = 0; j < p[i].size(); j++) {
      p[i][j].family->setRealValueForAllTypes(p[i][j].baseValue);
    }
  }

#if defined(ALBANY_LCM)
  // Store pointers to solution and time derivatives.
  // Needed for Schwarz coupling.
  if (x != Teuchos::null)
    x_ = Teuchos::rcp(new Tpetra_Vector(*Albany::getConstTpetraVector(x)));
  else
    x_ = Teuchos::null;
  if (x_dot != Teuchos::null)
    xdot_ = Teuchos::rcp(new Tpetra_Vector(*Albany::getConstTpetraVector(x_dot)));
  else
    xdot_ = Teuchos::null;
  if (x_dotdot != Teuchos::null)
    xdotdot_ = Teuchos::rcp(new Tpetra_Vector(*Albany::getConstTpetraVector(x_dotdot)));
  else
    xdotdot_ = Teuchos::null;
#endif // ALBANY_LCM

  // Zero out overlapped residual - Tpetra
  overlapped_f->assign(0.0);
  f->assign(0.0);

#ifdef ALBANY_PERIDIGM
#if defined(ALBANY_EPETRA)
  const Teuchos::RCP<LCM::PeridigmManager> &peridigmManager =
      LCM::PeridigmManager::self();
  if (Teuchos::nonnull(peridigmManager)) {
    peridigmManager->setCurrentTimeAndDisplacement(current_time, x);
    peridigmManager->evaluateInternalForce();
  }
#endif
#endif

  // Set data in Workset struct, and perform fill via field manager
  {
    if (Teuchos::nonnull(rc_mgr)) {
      rc_mgr->init_x_if_not(x->space());
    }

    PHAL::Workset workset;

    double const
    this_time = fixTime(current_time);

    loadBasicWorksetInfo(workset, this_time);

    workset.time_step = dt; 

    workset.f = overlapped_f;

    for (int ws = 0; ws < numWorksets; ws++) {
      loadWorksetBucketInfo<PHAL::AlbanyTraits::Residual>(workset, ws);

#ifdef DEBUG_OUTPUT
      *out << "IKT countRes = " << countRes
           << ", computeGlobalResid workset.x = \n ";
      describe(workset.x.getConst(),*out, Teuchos::VERB_EXTREME);
#endif

      // FillType template argument used to specialize Sacado
#ifdef DEBUG_OUTPUT2
      std::cout << "calling FM evaluate fields in computeGlobalResidualImpl" << std::endl;
#endif
      fm[wsPhysIndex[ws]]->evaluateFields<PHAL::AlbanyTraits::Residual>(
          workset);
      if (nfm != Teuchos::null) {
#ifdef ALBANY_PERIDIGM
        // DJL this is a hack to avoid running a block with sphere elements
        // through a Neumann field manager that was constructed for a non-sphere
        // element topology.  The root cause is that Albany currently supports
        // only a single Neumann field manager.  The history on that is murky.
        // The single field manager is created for a specific element topology,
        // and it fails if applied to worksets with a different element
        // topology. The Peridigm use case is a discretization that contains
        // blocks with sphere elements and blocks with standard FEM solid
        // elements, and we want to apply Neumann BC to the standard solid
        // elements.
        if (workset.sideSets->size() != 0) {
          deref_nfm(nfm, wsPhysIndex, ws)
              ->evaluateFields<PHAL::AlbanyTraits::Residual>(workset);
        }
#else
        deref_nfm(nfm, wsPhysIndex, ws)
            ->evaluateFields<PHAL::AlbanyTraits::Residual>(workset);
#endif
      }
    }
  }

  // Assemble the residual into a non-overlapping vector
  cas_manager->combine(overlapped_f,f,CombineMode::ADD);

  // Allocate scaleVec_
#ifdef ALBANY_MPI
  if (scale != 1.0) {
    if (scaleVec_ == Teuchos::null) {
      scaleVec_ = Thyra::createMember(f->space());
      scaleVec_->assign(0.0);
      setScale();
    } else if (Teuchos::nonnull(f)) {
      if (scaleVec_->space()->dim() != f->space()->dim()) {
        scaleVec_ = Thyra::createMember(f->space());
        scaleVec_->assign(0.0);
        setScale();
      }
    }
  }
#else
  ALBANY_ASSERT(scale == 1.0, "non-unity scale implementation requires MPI!");
#endif

#ifdef WRITE_TO_MATRIX_MARKET
  char nameResUnscaled[100]; // create string for file name
  sprintf(nameResUnscaled, "resUnscaled%i_residual.mm", countScale);
  Tpetra::MatrixMarket::Writer<Tpetra_CrsMatrix>::writeDenseFile(nameResUnscaled, Albany::getConstTpetraVector(f));
#endif

  if (scaleBCdofs == false && scale != 1.0) {
    Thyra::ele_wise_scale(*scaleVec_,f.ptr());
  }

#ifdef WRITE_TO_MATRIX_MARKET
  char nameResScaled[100]; // create string for file name
  sprintf(nameResScaled, "resScaled%i_residual.mm", countScale);
  Tpetra::MatrixMarket::Writer<Tpetra_CrsMatrix>::writeDenseFile(nameResScaled, Albany::getConstTpetraVector(f));
#endif

#if defined(ALBANY_LCM)
  // Push the assembled residual values back into the overlap vector
  cas_manager->scatter(f,overlapped_f,CombineMode::INSERT);
  // Write the residual to the discretization, which will later (optionally)
  // be written to the output file
  disc->setResidualField(overlapped_f);
#endif // ALBANY_LCM

  // Apply Dirichlet conditions using dfm (Dirchelt Field Manager)

  if (dfm != Teuchos::null) {
    PHAL::Workset workset;

    workset.f = f;

    loadWorksetNodesetInfo(workset);

    if (scaleBCdofs == true) {
      setScaleBCDofs(workset);
#ifdef WRITE_TO_MATRIX_MARKET
      char nameScale[100]; // create string for file name
      if (commT->getSize() == 1) {
        sprintf(nameScale, "scale%i.mm", countScale);
        Tpetra::MatrixMarket::Writer<Tpetra_CrsMatrix>::writeDenseFile(nameScale, Albany::getConstTpetraVector(scaleVec_));
      }
      else {
        // LB 7/24/18: is the conversion to serial really needed? Tpetra can handle distributed vectors in the output.
        //             Besides, the serial map has GIDs from 0 to num_global_elements-1. This *may not* be the same
        //             set of GIDs as in the solution map (this may really become an issue when we start tackling
        //             block discretizations)
/*
 *        // create serial map that puts the whole solution on processor 0
 *        int numMyElements = (scaleVec_->getMap()->getComm()->getRank() == 0)
 *                                  ? scaleVec_->getMap()->getGlobalNumElements()
 *                                  : 0;
 *        Teuchos::RCP<const Tpetra_Map> serial_map =
 *              Teuchos::rcp(new const Tpetra_Map(INVALID, numMyElements, 0, commT));
 *
 *        // create importer from parallel map to serial map and populate serial
 *        // solution scale_serial
 *        Teuchos::RCP<Tpetra_Import> importOperator =
 *            Teuchos::rcp(new Tpetra_Import(scaleVec_->getMap(), serial_map));
 *        Teuchos::RCP<Tpetra_Vector> scale_serial =
 *            Teuchos::rcp(new Tpetra_Vector(serial_map));
 *        scale_serial->doImport(*scaleVec_, *importOperator, Tpetra::INSERT);
 */
        sprintf(nameScale, "scaleSerial%i.mm", countScale);
        Tpetra::MatrixMarket::Writer<Tpetra_CrsMatrix>::writeDenseFile(nameScale, Albany::getConstTpetraVector(scaleVec_));
      }
#endif
      countScale++;
    }

    dfm_set(workset, x, x_dot, x_dotdot, rc_mgr);

    double const
    this_time = fixTime(current_time);

    workset.current_time = this_time;

#if defined(ALBANY_LCM)
    // Needed for more specialized Dirichlet BCs (e.g. Schwarz coupling)
    workset.apps_ = apps_;
    workset.current_app_ = Teuchos::rcp(this, false);
#endif // ALBANY_LCM

    workset.distParamLib = distParamLib;
    workset.disc = disc;

    // FillType template argument used to specialize Sacado
#ifdef DEBUG_OUTPUT2
    std::cout << "calling DFM evaluate fields in computeGlobalResidualImplT" << std::endl;
#endif
    dfm->evaluateFields<PHAL::AlbanyTraits::Residual>(workset);
  }

  // scale residual by scaleVec_ if scaleBCdofs is on
  if (scaleBCdofs == true) {
    Thyra::ele_wise_scale(*scaleVec_,f.ptr());
  }
}

void Albany::Application::computeGlobalResidual(
    const double current_time,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& x_dot,
    const Teuchos::RCP<const Thyra_Vector>& x_dotdot,
    const Teuchos::Array<ParamVec> &p,
    const Teuchos::RCP<Thyra_Vector>& f,
    const double dt) 
{
  // Create non-owning RCPs to Tpetra objects
  // to be passed to the implementation
  if (problem->useSDBCs() == false) {
    this->computeGlobalResidualImpl(
        current_time, x, x_dot, x_dotdot,
        p, f, dt);
  } else {
    // Temporary, while we refactor
    this->computeGlobalResidualSDBCsImpl(
        current_time, x, x_dot, x_dotdot,
        p, f, dt);
  }

  // Debut output
  if (writeToMatrixMarketRes !=
      0) {          // If requesting writing to MatrixMarket of residual...
    char name[100]; // create string for file name
    if (writeToMatrixMarketRes ==
        -1) { // write residual to MatrixMarket every time it arises
      sprintf(name, "rhs%i.mm", countRes);
      Tpetra::MatrixMarket::Writer<Tpetra_CrsMatrix>::writeDenseFile(name, Albany::getConstTpetraVector(f));
    } else {
      if (countRes ==
          writeToMatrixMarketRes) { // write residual only at requested count#
        sprintf(name, "rhs%i.mm", countRes);
        Tpetra::MatrixMarket::Writer<Tpetra_CrsMatrix>::writeDenseFile(name, Albany::getConstTpetraVector(f));
      }
    }
  }
  if (writeToCoutRes != 0) {    // If requesting writing of residual to cout...
    if (writeToCoutRes == -1) { // cout residual time it arises
      std::cout << "Global Residual #" << countRes << ": " << std::endl;
      describe(f.getConst(),*out, Teuchos::VERB_EXTREME);
    } else {
      if (countRes == writeToCoutRes) { // cout residual only at requested
                                        // count#
        std::cout << "Global Residual #" << countRes << ": " << std::endl;
        describe(f.getConst(),*out, Teuchos::VERB_EXTREME);
      }
    }
  }
  if (writeToMatrixMarketRes != 0 || writeToCoutRes != 0) {
    countRes++; // increment residual counter
  }
}

void Albany::Application::computeGlobalJacobianImpl(
    const double alpha, const double beta, const double omega,
    const double current_time,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
    const Teuchos::Array<ParamVec> &p,
    const Teuchos::RCP<Thyra_Vector>& f,
    const Teuchos::RCP<Thyra_LinearOp>& jac,
    const double dt) {
  TEUCHOS_FUNC_TIME_MONITOR("> Albany Fill: Jacobian");

  postRegSetup("Jacobian");

  // Load connectivity map and coordinates
  const auto &wsElNodeEqID = disc->getWsElNodeEqID();
  const auto &coords = disc->getCoords();
  const auto &wsEBNames = disc->getWsEBNames();
  const auto &wsPhysIndex = disc->getWsPhysIndex();

  int numWorksets = wsElNodeEqID.size();

  Teuchos::RCP<Thyra_Vector> overlapped_f;
  if (Teuchos::nonnull(f)) {
    overlapped_f = solMgrT->get_overlapped_f();
  }

  Teuchos::RCP<Thyra_LinearOp> overlapped_jac =solMgrT->get_overlapped_jac();
  auto cas_manager = solMgrT->get_cas_manager();

  // Scatter x and xdot to the overlapped distribution
  solMgrT->scatterX(x, xdot, xdotdot);

  // Scatter distributed parameters
  distParamLib->scatter();

  // Set parameters
  for (int i = 0; i < p.size(); i++)
    for (unsigned int j = 0; j < p[i].size(); j++)
      p[i][j].family->setRealValueForAllTypes(p[i][j].baseValue);

  // Zero out overlapped residual
  if (Teuchos::nonnull(f)) {
    overlapped_f->assign(0.0);
    f->assign(0.0);
  }

  // Zero out Jacobian
  resumeFill(jac);
  assign(jac,0.0);

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  if (!isFillActive(overlapped_jac)) {
    resumeFill(overlapped_jac);
  }
#endif
  assign(overlapped_jac,0.0);
#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  if (isFillActive(overlapped_jac)) {
    fillComplete(overlapped_jac);
  }
  if (!isFillActive(overlapped_jac)) {
    resumeFill(overlapped_jac);
  }
#endif

  // Set data in Workset struct, and perform fill via field manager
  {
    PHAL::Workset workset;

    double const
    this_time = fixTime(current_time);

    loadBasicWorksetInfo(workset, this_time);

    workset.time_step = dt; 

#ifdef DEBUG_OUTPUT
    *out << "IKT countJac = " << countJac
         << ", computeGlobalJacobian workset.x = \n";
    describe(workset.x.getConst(),*out, Teuchos::VERB_EXTREME);
#endif

    workset.f   = overlapped_f;
    workset.Jac = overlapped_jac;
    loadWorksetJacobianInfo(workset, alpha, beta, omega);

    // fill Jacobian derivative dimensions:
    for (int ps = 0; ps < fm.size(); ps++) {
      (workset.Jacobian_deriv_dims)
          .push_back(
              PHAL::getDerivativeDimensions<PHAL::AlbanyTraits::Jacobian>(
                  this, ps, explicit_scheme));
    }

    for (int ws = 0; ws < numWorksets; ws++) {
      loadWorksetBucketInfo<PHAL::AlbanyTraits::Jacobian>(workset, ws);
      // FillType template argument used to specialize Sacado
#ifdef DEBUG_OUTPUT2
      std::cout << "calling FM evaluate fields in computeGlobalJacobianImplT" << std::endl;
#endif
      fm[wsPhysIndex[ws]]->evaluateFields<PHAL::AlbanyTraits::Jacobian>(
          workset);
      if (Teuchos::nonnull(nfm))
#ifdef ALBANY_PERIDIGM
        // DJL avoid passing a sphere mesh through a nfm that was
        // created for non-sphere topology.
        if (workset.sideSets->size() != 0) {
          deref_nfm(nfm, wsPhysIndex, ws)
              ->evaluateFields<PHAL::AlbanyTraits::Jacobian>(workset);
        }
#else
        deref_nfm(nfm, wsPhysIndex, ws)
            ->evaluateFields<PHAL::AlbanyTraits::Jacobian>(workset);
#endif
    }
  }

  {
    TEUCHOS_FUNC_TIME_MONITOR("> Albany Fill: Jacobian Export");
    // Allocate and populate scaleVec_
    if (scale != 1.0) {
      if (scaleVec_ == Teuchos::null ||
          scaleVec_->space()->dim() != jac->domain()->dim()) {
        scaleVec_ = Thyra::createMember(jac->range());
        scaleVec_->assign(0.0);
        setScale();
      }
    }

    // Assemble global residual
    if (Teuchos::nonnull(f)) {
      cas_manager->combine(overlapped_f,f,CombineMode::ADD);
    }
    // Assemble global Jacobian
    cas_manager->combine(overlapped_jac,jac,CombineMode::ADD);

#ifdef ALBANY_PERIDIGM
#if defined(ALBANY_EPETRA)
    if (Teuchos::nonnull(LCM::PeridigmManager::self())) {
      LCM::PeridigmManager::self()
          ->copyPeridigmTangentStiffnessMatrixIntoAlbanyJacobian(jac);
    }
#endif
#endif

    // scale Jacobian
    if (scaleBCdofs == false && scale != 1.0) {
      fillComplete(jac);
#ifdef WRITE_TO_MATRIX_MARKET
      char nameJacUnscaled[100]; // create string for file name
      sprintf(nameJacUnscaled, "jacUnscaled%i.mm", countScale);
      Tpetra::MatrixMarket::Writer<Tpetra_CrsMatrix>::writeSparseFile(nameJacUnscaled, getConstTpetraMatrix(jac));
      if (f != Teuchos::null) {
        char nameResUnscaled[100]; // create string for file name
        sprintf(nameResUnscaled, "resUnscaled%i.mm", countScale);
        Tpetra::MatrixMarket::Writer<Tpetra_CrsMatrix>::writeDenseFile(nameResUnscaled, getConstTpetraVector(f));
      }
#endif
      // set the scaling
      setScale(jac);

      // scale Jacobian
      // We MUST be able to cast jac to ScaledLinearOpBase in order to left scale it.
      auto jac_scaled_lop = Teuchos::rcp_dynamic_cast<Thyra::ScaledLinearOpBase<ST>>(jac,true);
      jac_scaled_lop->scaleLeft(*scaleVec_);
      resumeFill(jac);
      // scale residual
      if (Teuchos::nonnull(f)) {
        Thyra::ele_wise_scale(*scaleVec_,f.ptr());
      }
#ifdef WRITE_TO_MATRIX_MARKET
      char nameJacScaled[100]; // create string for file name
      sprintf(nameJacScaled, "jacScaled%i.mm", countScale);
      Tpetra::MatrixMarket::Writer<Tpetra_CrsMatrix>::writeSparseFile(nameJacScaled, getConstTpetraMatrix(jac));
      if (f != Teuchos::null) {
        char nameResScaled[100]; // create string for file name
        sprintf(nameResScaled, "resScaled%i.mm", countScale);
        Tpetra::MatrixMarket::Writer<Tpetra_CrsMatrix>::writeDenseFile(nameResScaled, getConstTpetraVector(f));
      }
      char nameScale[100]; // create string for file name
      sprintf(nameScale, "scale%i.mm", countScale);
      Tpetra::MatrixMarket::Writer<Tpetra_CrsMatrix>::writeDenseFile(nameScale, getConstTpetraVector(scaleVec_));
#endif
      countScale++;
    }
  } // End timer
  // Apply Dirichlet conditions using dfm (Dirchelt Field Manager)
  if (Teuchos::nonnull(dfm)) {
    PHAL::Workset workset;

    workset.f = f;
    workset.Jac = jac;
    workset.m_coeff = alpha;
    workset.n_coeff = omega;
    workset.j_coeff = beta;

    double const
    this_time = fixTime(current_time);

    workset.current_time = this_time;

    if (beta == 0.0 && perturbBetaForDirichlets > 0.0)
      workset.j_coeff = perturbBetaForDirichlets;

    dfm_set(workset, x, xdot, xdotdot, rc_mgr);

    loadWorksetNodesetInfo(workset);

    if (scaleBCdofs == true) {
      setScaleBCDofs(workset, jac);
#ifdef WRITE_TO_MATRIX_MARKET
      char nameScale[100]; // create string for file name
      if (commT->getSize() == 1) {
        sprintf(nameScale, "scale%i.mm", countScale);
      }
      else {
        Tpetra::MatrixMarket::Writer<Tpetra_CrsMatrix>::writeDenseFile(nameScale, getConstTpetraVector(scaleVec_));
        // LB 7/24/18: is the conversion to serial really needed? Tpetra can handle distributed vectors in the output.
        //             Besides, the serial map has GIDs from 0 to num_global_elements-1. This *may not* be the same
        //             set of GIDs as in the solution map (this may really become an issue when we start tackling
        //             block discretizations)
        // create serial map that puts the whole solution on processor 0
/*
 *         int numMyElements = (scaleVec_->getMap()->getComm()->getRank() == 0)
 *                                 ? scaleVec_->getMap()->getGlobalNumElements()
 *                                 : 0;
 *         Teuchos::RCP<const Tpetra_Map> serial_map =
 *             Teuchos::rcp(new const Tpetra_Map(INVALID, numMyElements, 0, commT));
 * 
 *         // create importer from parallel map to serial map and populate serial
 *         // solution scale_serial
 *         Teuchos::RCP<Tpetra_Import> importOperator =
 *             Teuchos::rcp(new Tpetra_Import(scaleVec_->getMap(), serial_map));
 *         Teuchos::RCP<Tpetra_Vector> scale_serial =
 *             Teuchos::rcp(new Tpetra_Vector(serial_map));
 *         scale_serial->doImport(*scaleVec_, *importOperator, Tpetra::INSERT);
 */
        sprintf(nameScale, "scaleSerial%i.mm", countScale);
        Tpetra::MatrixMarket::Writer<Tpetra_CrsMatrix>::writeDenseFile(nameScale, getConstTpetraVector(scaleVec_));
      }
#endif
      countScale++;
    }

    workset.distParamLib = distParamLib;
    workset.disc = disc;

#if defined(ALBANY_LCM)
    // Needed for more specialized Dirichlet BCs (e.g. Schwarz coupling)
    workset.apps_ = apps_;
    workset.current_app_ = Teuchos::rcp(this, false);
#endif // ALBANY_LCM

    // FillType template argument used to specialize Sacado
#ifdef DEBUG_OUTPUT2
    std::cout << "calling DFM evaluate fields in computeGlobalJacobianImplT" << std::endl;
#endif
    dfm->evaluateFields<PHAL::AlbanyTraits::Jacobian>(workset);
  }
  fillComplete(jac);

  // Apply scaling to residual and Jacobian
  if (scaleBCdofs == true) {
    if (Teuchos::nonnull(f)) {
      Thyra::ele_wise_scale(*scaleVec_, f.ptr());
    }
    // We MUST be able to cast jac to ScaledLinearOpBase in order to left scale it.
    auto jac_scaled_lop = Teuchos::rcp_dynamic_cast<Thyra::ScaledLinearOpBase<ST>>(jac,true);
    jac_scaled_lop->scaleLeft(*scaleVec_);
  }

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  if (isFillActive(overlapped_jac)) {
    // Makes getLocalMatrix() valid.
    fillComplete(overlapped_jac);
  }
#endif
  if (derivatives_check_ > 0) {
    checkDerivatives(*this, current_time, x, xdot, xdotdot,
                      p, f, jac, derivatives_check_);
  }
}

void Albany::Application::computeGlobalJacobianSDBCsImpl(
    const double alpha, const double beta, const double omega,
    const double current_time,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
    const Teuchos::Array<ParamVec> &p,
    const Teuchos::RCP<Thyra_Vector>& f,
    const Teuchos::RCP<Thyra_LinearOp>& jac,
    const double dt) {
  TEUCHOS_FUNC_TIME_MONITOR("> Albany Fill: Jacobian");

  postRegSetup("Jacobian");

  if (scale != 1.0) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
                               "'Scaling' sublist not recognized when using SDBCs. \n" <<
                               "To use scaling with SDBCs, specify 'Scaled SDBC' and/or \n" <<
                               "'SDBC Scaling' in 'Dirichlet BCs' sublist."); 
  }

//#define DEBUG_OUTPUT

#ifdef DEBUG_OUTPUT
  *out << "IKT prev_times_ size = " << prev_times_.size() << '\n';
#endif
  int app_no = 0;
#ifdef ALBANY_LCM
  if (app_index_ < 0)
    app_no = 0;
  else
    app_no = app_index_;
#endif

  bool begin_time_step = false;

#ifdef ALBANY_LCM
  current_app = app_index_;
#ifdef DEBUG_OUTPUT
  *out << " IKT current_app, previous_app = " << current_app << ", "
       << previous_app << '\n';
#endif
#endif

  // Load connectivity map and coordinates
  const auto &wsElNodeEqID = disc->getWsElNodeEqID();
  const auto &coords = disc->getCoords();
  const auto &wsEBNames = disc->getWsEBNames();
  const auto &wsPhysIndex = disc->getWsPhysIndex();

  int numWorksets = wsElNodeEqID.size();

  // The combine-and-scatter manager
  auto cas_manager = solMgrT->get_cas_manager();


  Teuchos::RCP<Thyra_Vector> overlapped_f = Teuchos::nonnull(f) ? solMgrT->get_overlapped_f() : Teuchos::null;
  Teuchos::RCP<Thyra_LinearOp> overlapped_jac = solMgrT->get_overlapped_jac();

  // Scatter x and xdot to the overlapped distribution
  solMgrT->scatterX(x, xdot, xdotdot);

  // Scatter distributed parameters
  distParamLib->scatter();

  // Set parameters
  for (int i = 0; i < p.size(); i++) {
    for (unsigned int j = 0; j < p[i].size(); j++) {
      p[i][j].family->setRealValueForAllTypes(p[i][j].baseValue);
  }}

  // Zero out overlapped residual (if present)
  if (Teuchos::nonnull(f)) {
    overlapped_f->assign(0.0);
    f->assign(0.0);
  }

  // Zero out Jacobian
  resumeFill(jac);
  assign(jac,0.0);

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  if (!isFillActive(overlapped_jac)) {
    resumeFill(overlapped_jac);
  }
#endif
  assign(overlapped_jac,0.0);
#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  if (isFillActive(overlapped_jac)) {
    fillComplete(overlapped_jac);
  }
  if (!isFillActive(overlapped_jac)) {
    resumeFill(overlapped_jac);
  }
#endif

  // Set data in Workset struct, and perform fill via field manager
  {
    PHAL::Workset workset;

    double const
    this_time = fixTime(current_time);

    loadBasicWorksetInfo(workset, this_time);

    workset.time_step = dt; 

#ifdef DEBUG_OUTPUT
    *out << "IKT countJac = " << countJac
         << ", computeGlobalJacobian workset.x = \n";
    describe(workset.x.getConst(),*out, Teuchos::VERB_EXTREME);
    *out << "IKT previous_time, this_time = " << prev_times_[app_no] << ", "
         << this_time << "\n";
#endif
    // Check if previous_time is same as current time.  If not, we are at the
    // start  of a new time step, so we set boolean parameter to true.
    if (prev_times_[app_no] != this_time)
      begin_time_step = true;

    workset.f = overlapped_f;
    workset.Jac = overlapped_jac;
    loadWorksetJacobianInfo(workset, alpha, beta, omega);

    // fill Jacobian derivative dimensions:
    for (int ps = 0; ps < fm.size(); ps++) {
      (workset.Jacobian_deriv_dims)
          .push_back(
              PHAL::getDerivativeDimensions<PHAL::AlbanyTraits::Jacobian>(
                  this, ps, explicit_scheme));
    }

    for (int ws = 0; ws < numWorksets; ws++) {
      loadWorksetBucketInfo<PHAL::AlbanyTraits::Jacobian>(workset, ws);
      // FillType template argument used to specialize Sacado
#ifdef DEBUG_OUTPUT2
      std::cout << "calling FM evaluate fields in computeGlobalJacobianSDBCsImplT" << std::endl;
#endif
      fm[wsPhysIndex[ws]]->evaluateFields<PHAL::AlbanyTraits::Jacobian>(
          workset);
      if (Teuchos::nonnull(nfm))
#ifdef ALBANY_PERIDIGM
        // DJL avoid passing a sphere mesh through a nfm that was
        // created for non-sphere topology.
        if (workset.sideSets->size() != 0) {
          deref_nfm(nfm, wsPhysIndex, ws)
              ->evaluateFields<PHAL::AlbanyTraits::Jacobian>(workset);
        }
#else
        deref_nfm(nfm, wsPhysIndex, ws)
            ->evaluateFields<PHAL::AlbanyTraits::Jacobian>(workset);
#endif
    }
    prev_times_[app_no] = this_time;
    if (previous_app != current_app) {
      begin_time_step = true;
    }
  }

  {
    TEUCHOS_FUNC_TIME_MONITOR("> Albany Fill: Jacobian Export");

    // Assemble global residual
    if (Teuchos::nonnull(f)) {
      cas_manager->combine(overlapped_f,f,CombineMode::ADD);
    }

    // Assemble global Jacobian
    cas_manager->combine(overlapped_jac,jac,CombineMode::ADD);

#ifdef ALBANY_PERIDIGM
#if defined(ALBANY_EPETRA)
    if (Teuchos::nonnull(LCM::PeridigmManager::self())) {
      LCM::PeridigmManager::self()
          ->copyPeridigmTangentStiffnessMatrixIntoAlbanyJacobian(getTpetraOperator(jac));
    }
#endif
#endif
  } // End timer

  // Apply Dirichlet conditions using dfm (Dirchelt Field Manager)
  Teuchos::RCP<Thyra_Vector> x_post_SDBCs;
  if (Teuchos::nonnull(dfm)) {
    PHAL::Workset workset;

    workset.f = f;
    workset.Jac = jac;
    workset.m_coeff = alpha;
    workset.n_coeff = omega;
    workset.j_coeff = beta;

    double const
    this_time = fixTime(current_time);

    workset.current_time = this_time;

    if (beta == 0.0 && perturbBetaForDirichlets > 0.0)
      workset.j_coeff = perturbBetaForDirichlets;

    dfm_set(workset, x, xdot, xdotdot, rc_mgr);

    loadWorksetNodesetInfo(workset);

    workset.distParamLib = distParamLib;
    workset.disc = disc;

#if defined(ALBANY_LCM)
    // Needed for more specialized Dirichlet BCs (e.g. Schwarz coupling)
    workset.apps_ = apps_;
    workset.current_app_ = Teuchos::rcp(this, false);
#endif // ALBANY_LCM

    // FillType template argument used to specialize Sacado
#ifdef DEBUG_OUTPUT2
    std::cout << "calling DFM evaluate fields in computeGlobalJacobianSDBCsImplT" << std::endl;
#endif
    dfm->evaluateFields<PHAL::AlbanyTraits::Jacobian>(workset);
    x_post_SDBCs = workset.x->clone_v();
  }
  fillComplete(jac);

#ifdef DEBUG_OUTPUT
  *out << "IKT begin_time_step? " << begin_time_step << "\n";
#endif
  if (begin_time_step == true) {
    // if (countRes == 0) {
    // Zero out overlapped residual - Tpetra
    if (Teuchos::nonnull(f)) {
      overlapped_f->assign(0.0);
      f->assign(0.0);
    }
    PHAL::Workset workset;

    // Zero out Jacobian
    resumeFill(jac);
    assign(jac,0.0);

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
    if (!isFillActive(overlapped_jac)) {
      resumeFill(overlapped_jac);
    }
#endif
    assign(overlapped_jac,0.0);
#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
    if (isFillActive(overlapped_jac)) {
      // Makes getLocalMatrix() valid.
      fillComplete(overlapped_jac);
    }
    if (!isFillActive(overlapped_jac)) {
      resumeFill(overlapped_jac);
    }

#endif

    const double this_time = fixTime(current_time);

    loadBasicWorksetInfoSDBCs(workset, x_post_SDBCs, this_time);

    workset.f = overlapped_f;
    workset.Jac = overlapped_jac;
    loadWorksetJacobianInfo(workset, alpha, beta, omega);

    // fill Jacobian derivative dimensions:
    for (int ps = 0; ps < fm.size(); ps++) {
      (workset.Jacobian_deriv_dims)
          .push_back(
              PHAL::getDerivativeDimensions<PHAL::AlbanyTraits::Jacobian>(
                  this, ps, explicit_scheme));
    }

    for (int ws = 0; ws < numWorksets; ws++) {
      loadWorksetBucketInfo<PHAL::AlbanyTraits::Jacobian>(workset, ws);

#ifdef DEBUG_OUTPUT
      *out << "IKT countRes = " << countRes
           << ", computeGlobalResid workset.x = \n ";
      describe(workset.x.getConst(),*out, Teuchos::VERB_EXTREME);
#endif

      // FillType template argument used to specialize Sacado
#ifdef DEBUG_OUTPUT2
      std::cout << "calling FM evaluate fields AGAIN in computeGlobalJacobianSDBCsImplT" << std::endl;
#endif
      fm[wsPhysIndex[ws]]->evaluateFields<PHAL::AlbanyTraits::Jacobian>(
          workset);
      if (nfm != Teuchos::null) {
#ifdef ALBANY_PERIDIGM
        // DJL avoid passing a sphere mesh through a nfm that was
        // created for non-sphere topology.
        if (workset.sideSets->size() != 0) {
          deref_nfm(nfm, wsPhysIndex, ws)
              ->evaluateFields<PHAL::AlbanyTraits::Jacobian>(workset);
        }
#else
        deref_nfm(nfm, wsPhysIndex, ws)
            ->evaluateFields<PHAL::AlbanyTraits::Jacobian>(workset);
#endif
      }
    }

    // Assemble the residual into a non-overlapping vector
    if (Teuchos::nonnull(f)) {
      cas_manager->combine(overlapped_f,f,CombineMode::ADD);
    }

    // Assemble global Jacobian
    cas_manager->combine(overlapped_jac,jac,CombineMode::ADD);

#ifdef ALBANY_PERIDIGM
#if defined(ALBANY_EPETRA)
    if (Teuchos::nonnull(LCM::PeridigmManager::self())) {
      LCM::PeridigmManager::self()
          ->copyPeridigmTangentStiffnessMatrixIntoAlbanyJacobian(getTpetraOperator(jac));
    }
#endif
#endif
    if (dfm != Teuchos::null) {

      PHAL::Workset workset;

      workset.f = f;
      workset.Jac = jac;
      workset.m_coeff = alpha;
      workset.n_coeff = omega;
      workset.j_coeff = beta;

      const double this_time = fixTime(current_time);

      workset.current_time = this_time;

      if (beta == 0.0 && perturbBetaForDirichlets > 0.0) {
        workset.j_coeff = perturbBetaForDirichlets;
      }

      dfm_set(workset, x_post_SDBCs, xdot, xdotdot, rc_mgr);

      loadWorksetNodesetInfo(workset);

      workset.distParamLib = distParamLib;
      workset.disc = disc;

#if defined(ALBANY_LCM)
      // Needed for more specialized Dirichlet BCs (e.g. Schwarz coupling)
      workset.apps_ = apps_;
      workset.current_app_ = Teuchos::rcp(this, false);
#endif // ALBANY_LCM

      // FillType template argument used to specialize Sacado
      if (MOR_apply_bcs_){
#ifdef DEBUG_OUTPUT2
        std::cout << "calling DFM evaluate fields AGAIN in computeGlobalJacobianSDBCsImplT" << std::endl;
#endif
        dfm->evaluateFields<PHAL::AlbanyTraits::Jacobian>(workset);
      }
    }
    fillComplete(jac);
  } // endif (begin_time_step == true)
  previous_app = current_app;

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  if (isFillActive(overlapped_jac)) {
    // Makes getLocalMatrix() valid.
    fillComplete(overlapped_jac);
  }
#endif
  if (derivatives_check_ > 0) {
    checkDerivatives(*this, current_time, x, xdot, xdotdot,
                      p, f, jac, derivatives_check_);
  }
}

void Albany::Application::computeGlobalJacobian(
    const double alpha, const double beta, const double omega,
    const double current_time,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
    const Teuchos::Array<ParamVec> &p,
    const Teuchos::RCP<Thyra_Vector>& f,
    const Teuchos::RCP<Thyra_LinearOp>& jac,
    const double dt) 
{
  // Create non-owning RCPs to Tpetra objects
  // to be passed to the implementation
  if (problem->useSDBCs() == false) {
    this->computeGlobalJacobianImpl(alpha, beta, omega, current_time, x, xdot, xdotdot, p, f, jac, dt);
  } else {
    this->computeGlobalJacobianSDBCsImpl(alpha, beta, omega, current_time, x, xdot, xdotdot, p, f, jac, dt);
  }
  // Debut output
  if (writeToMatrixMarketJac != 0) {
    // If requesting writing to MatrixMarket of Jacobian...
    if (writeToMatrixMarketJac == -1) {
      // write jacobian to MatrixMarket every time it arises
      writeMatrixMarket(jac.getConst(),"jac",countJac);
    } else if (countJac == writeToMatrixMarketJac) {
      // write jacobian only at requested count#
      writeMatrixMarket(jac.getConst(),"jac",countJac);
    }
  }
  if (writeToCoutJac != 0) {
    // If requesting writing Jacobian to standard output (cout)...
    if (writeToCoutJac == -1) { // cout jacobian every time it arises
      *out << "Global Jacobian #" << countJac << ":\n";
      describe(jac.getConst(),*out, Teuchos::VERB_EXTREME);
    } else if (countJac == writeToCoutJac) {
      // cout jacobian only at requested count#
      *out << "Global Jacobian #" << countJac << ":\n";
      describe(jac.getConst(),*out, Teuchos::VERB_EXTREME);
    }
  }
  if (computeJacCondNum != 0) { // If requesting computation of condition number
#if defined(ALBANY_EPETRA)
    if (computeJacCondNum == -1) {
      // cout jacobian condition # every time it arises
      double condNum = computeConditionNumber(jac);
      *out << "Jacobian #" << countJac << " condition number = " << condNum << "\n";
    } else if (countJac == computeJacCondNum) { 
      // cout jacobian condition # only at requested count#
      double condNum = computeConditionNumber(jac);
      *out << "Jacobian #" << countJac << " condition number = " << condNum << "\n";
    }
#else
    TEUCHOS_TEST_FOR_EXCEPTION(
        true, std::logic_error,
        "Error in Albany::Application: Compute Jacobian Condition Number debug option "
        "currently relies on an Epetra-based routine in AztecOO.  To use this option, please "
        "rebuild Albany with ENABLE_ALBANY_EPETRA_EXE=ON.  You will then be able to have Albany "
        "output the Jacobian condition number when running either the Albany or AlbanyT executable.\n");
#endif
  }
  if (writeToMatrixMarketJac != 0 || writeToCoutJac != 0 || computeJacCondNum != 0) {
    countJac++; // increment Jacobian counter
  }
}

void Albany::Application::computeGlobalPreconditionerT(
    const RCP<Tpetra_CrsMatrix> &jac, const RCP<Tpetra_Operator> &prec) {
//#if defined(ATO_USES_COGENT)
#ifdef ALBANY_ATO
  if (precType == "XFEM") {
    TEUCHOS_FUNC_TIME_MONITOR("> Albany Fill: Precond");

    *out << "Computing WPrec by Cogent" << std::endl;

    RCP<ATOT::XFEM::Preconditioner> cogentPrec =
        rcp_dynamic_cast<ATOT::XFEM::Preconditioner>(prec);

    cogentPrec->BuildPreconditioner(jac, disc, stateMgr);
  }
#endif
  //#endif
}

#if defined(ALBANY_EPETRA)
void Albany::Application::computeGlobalPreconditioner(
    const RCP<Epetra_CrsMatrix> &jac, const RCP<Epetra_Operator> &prec) {
#if defined(ALBANY_TEKO)
  if (precType == "Teko") {
    TEUCHOS_FUNC_TIME_MONITOR("> Albany Fill: Precond");

    *out << "Computing WPrec by Teko" << std::endl;

    RCP<Teko::Epetra::InverseFactoryOperator> blockPrec =
        rcp_dynamic_cast<Teko::Epetra::InverseFactoryOperator>(prec);

    blockPrec->initInverse();

    wrappedJac = buildWrappedOperator(jac, wrappedJac);
    blockPrec->rebuildInverseOperator(wrappedJac);
  }
#endif
//#if defined(ATO_USES_COGENT)
#ifdef ALBANY_ATO
  if (precType == "XFEM") {
    TEUCHOS_FUNC_TIME_MONITOR("> Albany Fill: Precond");

    *out << "Computing WPrec by Cogent" << std::endl;

    RCP<ATO::XFEM::Preconditioner> cogentPrec =
        rcp_dynamic_cast<ATO::XFEM::Preconditioner>(prec);

    cogentPrec->BuildPreconditioner(jac, disc, stateMgr);
  }
#endif
  //#endif
}
#endif

void Albany::Application::computeGlobalTangent(
    const double alpha, const double beta, const double omega,
    const double current_time, bool sum_derivs,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
    const Teuchos::Array<ParamVec> &par, ParamVec *deriv_par,
    const Teuchos::RCP<const Thyra_MultiVector>& Vx,
    const Teuchos::RCP<const Thyra_MultiVector>& Vxdot,
    const Teuchos::RCP<const Thyra_MultiVector>& Vxdotdot,
    const Teuchos::RCP<const Thyra_MultiVector>& Vp,
    const Teuchos::RCP<Thyra_Vector>& f,
    const Teuchos::RCP<Thyra_MultiVector>& JV,
    const Teuchos::RCP<Thyra_MultiVector>& fp)
{
  TEUCHOS_FUNC_TIME_MONITOR("> Albany Fill: Tangent");

  postRegSetup("Tangent");

  // Load connectivity map and coordinates
  const auto &wsElNodeEqID = disc->getWsElNodeEqID();
  const auto &coords = disc->getCoords();
  const auto &wsEBNames = disc->getWsEBNames();
  const auto &wsPhysIndex = disc->getWsPhysIndex();

  int numWorksets = wsElNodeEqID.size();

  Teuchos::RCP<Thyra_Vector> overlapped_f = solMgrT->get_overlapped_f();

  // The combine-and-scatter manager
  auto cas_manager = solMgrT->get_cas_manager();

  // Scatter x and xdot to the overlapped distrbution
  solMgrT->scatterX(x, xdot, xdotdot);

  // Scatter distributed parameters
  distParamLib->scatter();

  // Scatter Vx to the overlapped distribution
  RCP<Thyra_MultiVector> overlapped_Vx;
  if (Teuchos::nonnull(Vx)) {
    overlapped_Vx = Thyra::createMembers(disc->getOverlapVectorSpace(),Vx->domain()->dim());
    overlapped_Vx->assign(0.0);
    cas_manager->scatter(Vx,overlapped_Vx,Albany::CombineMode::INSERT);
  }

  // Scatter Vxdot to the overlapped distribution
  RCP<Thyra_MultiVector> overlapped_Vxdot;
  if (Teuchos::nonnull(Vxdot)) {
    overlapped_Vxdot = Thyra::createMembers(disc->getOverlapVectorSpace(),Vxdot->domain()->dim());
    overlapped_Vxdot->assign(0.0);
    cas_manager->scatter(Vxdot,overlapped_Vxdot,Albany::CombineMode::INSERT);
  }
  RCP<Thyra_MultiVector> overlapped_Vxdotdot;
  if (Teuchos::nonnull(Vxdotdot)) {
    overlapped_Vxdotdot = Thyra::createMembers(disc->getOverlapVectorSpace(),Vxdotdot->domain()->dim());
    overlapped_Vxdotdot->assign(0.0);
    cas_manager->scatter(Vxdotdot,overlapped_Vxdotdot,Albany::CombineMode::INSERT);
  }

  // Set parameters
  for (int i = 0; i < par.size(); i++) {
    for (unsigned int j = 0; j < par[i].size(); j++) {
      par[i][j].family->setRealValueForAllTypes(par[i][j].baseValue);
  }}

  RCP<ParamVec> params = rcp(deriv_par, false);

  // Zero out overlapped residual
  if (Teuchos::nonnull(f)) {
    overlapped_f->assign(0.0);
    f->assign(0.0);
  }

  RCP<Thyra_MultiVector> overlapped_JV;
  if (Teuchos::nonnull(JV)) {
    overlapped_JV = Thyra::createMembers(disc->getOverlapVectorSpace(), JV->domain()->dim());
    overlapped_JV->assign(0.0);
    JV->assign(0.0);
  }

  RCP<Thyra_MultiVector> overlapped_fp;
  if (Teuchos::nonnull(fp)) {
    overlapped_fp = Thyra::createMembers(disc->getOverlapVectorSpace(), fp->domain()->dim());
    overlapped_fp->assign(0.0);
    fp->assign(0.0);
  }

  // Number of x & xdot tangent directions
  int num_cols_x = 0;
  if (Teuchos::nonnull(Vx)) {
    num_cols_x = Vx->domain()->dim();
  } else if (Teuchos::nonnull(Vxdot)) {
    num_cols_x = Vxdot->domain()->dim();
  } else if (Teuchos::nonnull(Vxdotdot)) {
    num_cols_x = Vxdotdot->domain()->dim();
  }

  // Number of parameter tangent directions
  int num_cols_p = 0;
  if (Teuchos::nonnull(params)) {
    if (Teuchos::nonnull(Vp)) {
      num_cols_p = Vp->domain()->dim();
    } else {
      num_cols_p = params->size();
    }
  }

  // Whether x and param tangent components are added or separate
  int param_offset = 0;
  if (!sum_derivs) {
    param_offset = num_cols_x; // offset of parameter derivs in deriv array
  }

  TEUCHOS_TEST_FOR_EXCEPTION(
      sum_derivs && (num_cols_x != 0) && (num_cols_p != 0) &&
          (num_cols_x != num_cols_p),
      std::logic_error,
      "Seed matrices Vx and Vp must have the same number "
          << " of columns when sum_derivs is true and both are "
          << "non-null!" << std::endl);

  // Initialize
  if (Teuchos::nonnull(params)) {
    TanFadType p;
    int num_cols_tot = param_offset + num_cols_p;
    for (unsigned int i = 0; i < params->size(); i++) {
      p = TanFadType(num_cols_tot, (*params)[i].baseValue);
      if (Teuchos::nonnull(Vp)) {
        // ArrayRCP for const view of Vp's vectors
        Teuchos::ArrayRCP<const ST> Vp_constView;
        for (int k = 0; k < num_cols_p; k++) {
          Vp_constView = getLocalData(Vp->col(k));
          p.fastAccessDx(param_offset + k) =
              Vp_constView[i];
        }
      } else
        p.fastAccessDx(param_offset + i) = 1.0;
      (*params)[i].family->setValue<PHAL::AlbanyTraits::Tangent>(p);
    }
  }

  // Set data in Workset struct, and perform fill via field manager
  {
    PHAL::Workset workset;

    double const
    this_time = fixTime(current_time);

    loadBasicWorksetInfo(workset, this_time);

    workset.params = params;
    workset.Vx = overlapped_Vx;
    workset.Vxdot = overlapped_Vxdot;
    workset.Vxdotdot = overlapped_Vxdotdot;
    workset.Vp = Vp;

    workset.f = overlapped_f;
    workset.JV = overlapped_JV;
    workset.fp = overlapped_fp;
    workset.j_coeff = beta;
    workset.m_coeff = alpha;
    workset.n_coeff = omega;

    workset.num_cols_x = num_cols_x;
    workset.num_cols_p = num_cols_p;
    workset.param_offset = param_offset;

    for (int ws = 0; ws < numWorksets; ws++) {
      loadWorksetBucketInfo<PHAL::AlbanyTraits::Tangent>(workset, ws);

      // FillType template argument used to specialize Sacado
#ifdef DEBUG_OUTPUT2
      std::cout << "calling FM evaluate fields in computeGlobalTangent" << std::endl;
#endif
      fm[wsPhysIndex[ws]]->evaluateFields<PHAL::AlbanyTraits::Tangent>(workset);
      if (nfm != Teuchos::null)
        deref_nfm(nfm, wsPhysIndex, ws)
            ->evaluateFields<PHAL::AlbanyTraits::Tangent>(workset);
    }

    // fill Tangent derivative dimensions
    for (int ps = 0; ps < fm.size(); ps++) {
      (workset.Tangent_deriv_dims)
          .push_back(PHAL::getDerivativeDimensions<PHAL::AlbanyTraits::Tangent>(
              this, ps));
    }
  }

  params = Teuchos::null;

  // Assemble global residual
  if (Teuchos::nonnull(f)) {
    cas_manager->combine(overlapped_f,f,CombineMode::ADD);
  }

  // Assemble derivatives
  if (Teuchos::nonnull(JV)) {
    cas_manager->combine(overlapped_JV,JV,CombineMode::ADD);
  }
  if (Teuchos::nonnull(fp)) {
    cas_manager->combine(overlapped_fp,fp,CombineMode::ADD);
  }

  // Apply Dirichlet conditions using dfm (Dirchelt Field Manager)
  if (Teuchos::nonnull(dfm)) {
    PHAL::Workset workset;

    workset.num_cols_x = num_cols_x;
    workset.num_cols_p = num_cols_p;
    workset.param_offset = param_offset;

    workset.f = f;
    workset.fp = fp;
    workset.JV = JV;
    workset.j_coeff = beta;
    workset.n_coeff = omega;
    workset.Vx = Vx;
    dfm_set(workset, x, xdot, xdotdot, rc_mgr);

    loadWorksetNodesetInfo(workset);
    workset.distParamLib = distParamLib;

    double const
    this_time = fixTime(current_time);

    workset.current_time = this_time;

    workset.disc = disc;

#if defined(ALBANY_LCM)
    // Needed for more specialized Dirichlet BCs (e.g. Schwarz coupling)
    workset.apps_ = apps_;
    workset.current_app_ = Teuchos::rcp(this, false);
#endif // ALBANY_LCM

    // FillType template argument used to specialize Sacado
#ifdef DEBUG_OUTPUT2
    std::cout << "calling DFM evaluate fields in computeGlobalTangent" << std::endl;
#endif
    dfm->evaluateFields<PHAL::AlbanyTraits::Tangent>(workset);
  }
}

void Albany::Application::applyGlobalDistParamDerivImpl(
    const double current_time,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
    const Teuchos::Array<ParamVec> &p,
    const std::string &dist_param_name,
    const bool trans,
    const Teuchos::RCP<const Thyra_MultiVector>& V,
    const Teuchos::RCP<Thyra_MultiVector>& fpV)
{
  TEUCHOS_FUNC_TIME_MONITOR("> Albany Fill: Distributed Parameter Derivative");

  postRegSetup("Distributed Parameter Derivative");

  // Load connectivity map and coordinates
  const auto &wsElNodeEqID = disc->getWsElNodeEqID();
  const auto &coords = disc->getCoords();
  const auto &wsEBNames = disc->getWsEBNames();
  const auto &wsPhysIndex = disc->getWsPhysIndex();

  int numWorksets = wsElNodeEqID.size();

  // The combin-and-scatter manager
  auto cas_manager = solMgrT->get_cas_manager();

  // Scatter x and xdot to the overlapped distribution
  solMgrT->scatterX(x, xdot, xdotdot);

  // Scatter distributed parameters
  distParamLib->scatter();

  // Set parameters
  for (int i = 0; i < p.size(); i++) {
    for (unsigned int j = 0; j < p[i].size(); j++) {
      p[i][j].family->setRealValueForAllTypes(p[i][j].baseValue);
  }}

  Teuchos::RCP<Thyra_MultiVector> overlapped_fpV;
  if (trans) {
    auto vs = Thyra::createVectorSpace<ST>(distParamLib->get(dist_param_name)->overlap_map());
    overlapped_fpV = Thyra::createMembers(vs,V->domain()->dim());
  } else {
    overlapped_fpV = Thyra::createMembers(disc->getOverlapVectorSpace(), fpV->domain()->dim());
  }
  overlapped_fpV->assign(0.0);
  fpV->assign(0.0);

  Teuchos::RCP<const Thyra_MultiVector> V_bc = V;

  // For (df/dp)^T*V, we have to evaluate Dirichlet BC's first
  if (trans && dfm != Teuchos::null) {
    Teuchos::RCP<Thyra_MultiVector> V_bc_nonconst = V->clone_mv();
    V_bc = V_bc_nonconst;

    PHAL::Workset workset;

    workset.fpV = fpV;
    workset.Vp_bc = V_bc_nonconst;
    workset.transpose_dist_param_deriv = trans;
    workset.dist_param_deriv_name = dist_param_name;
    workset.disc = disc;

    double const
    this_time = fixTime(current_time);

    workset.current_time = this_time;

    dfm_set(workset, x, xdot, xdotdot, rc_mgr);

    loadWorksetNodesetInfo(workset);

    // FillType template argument used to specialize Sacado
#ifdef DEBUG_OUTPUT2
    std::cout << "calling DFM evaluate fields in applyGlobalDistParamDerivImpl" << std::endl;
#endif
    dfm->evaluateFields<PHAL::AlbanyTraits::DistParamDeriv>(workset);
  }

  // Import V (after BC's applied) to overlapped distribution
  RCP<Thyra_MultiVector> overlapped_V;
  if (trans) {
    overlapped_V = Thyra::createMembers(disc->getOverlapVectorSpace(), V_bc->domain()->dim());
    overlapped_V->assign(0.0);
    cas_manager->scatter(V_bc,overlapped_V,Albany::CombineMode::INSERT);
  } else {
    Teuchos::RCP<const Thyra_VectorSpace> vs = Thyra::tpetraVectorSpace<ST>(distParamLib->get(dist_param_name)->overlap_map());
    overlapped_V = Thyra::createMembers(vs, V_bc->domain()->dim());
    overlapped_V->assign(0.0);
    cas_manager->scatter(V_bc,overlapped_V,Albany::CombineMode::INSERT);
  }

  // Set data in Workset struct, and perform fill via field manager
  {
    PHAL::Workset workset;

    double const
    this_time = fixTime(current_time);

    loadBasicWorksetInfo(workset, this_time);

    workset.dist_param_deriv_name = dist_param_name;
    workset.Vp = overlapped_V;
    workset.fpV = overlapped_fpV;
    workset.transpose_dist_param_deriv = trans;

    for (int ws = 0; ws < numWorksets; ws++) {
      loadWorksetBucketInfo<PHAL::AlbanyTraits::DistParamDeriv>(workset, ws);

      // FillType template argument used to specialize Sacado
#ifdef DEBUG_OUTPUT2
      std::cout << "calling FM evaluate fields in applyGlobalDistParamDerivImpl" << std::endl;
#endif
      fm[wsPhysIndex[ws]]->evaluateFields<PHAL::AlbanyTraits::DistParamDeriv>(
          workset);
      if (nfm != Teuchos::null)
#ifdef ALBANY_PERIDIGM
        // DJL avoid passing a sphere mesh through a nfm that was
        // created for non-sphere topology.
        if (workset.sideSets->size() != 0) {
          deref_nfm(nfm, wsPhysIndex, ws)
              ->evaluateFields<PHAL::AlbanyTraits::DistParamDeriv>(workset);
        }
#else
        deref_nfm(nfm, wsPhysIndex, ws)
            ->evaluateFields<PHAL::AlbanyTraits::DistParamDeriv>(workset);
#endif
    }
  }

  // std::stringstream pg; pg << "neumann_phalanx_graph_ ";
  // nfm[0]->writeGraphvizFile<PHAL::AlbanyTraits::DistParamDeriv>(pg.str(),true,true);

  {
    TEUCHOS_FUNC_TIME_MONITOR(
        "> Albany Fill: Distributed Parameter Derivative Export");
    // Assemble global df/dp*V
    if (trans) {
      // TODO: make DistParamLib Thyra
      Teuchos::RCP<const Thyra_MultiVector> temp = fpV->clone_mv();
      distParamLib->get(dist_param_name)->export_add(*getTpetraMultiVector(fpV),
                                                     *getConstTpetraMultiVector(overlapped_fpV));
      fpV->update(1.0, *temp); // fpV += temp;

      std::stringstream sensitivity_name;
      sensitivity_name << dist_param_name << "_sensitivity";
      if (distParamLib->has(sensitivity_name.str())) {
        auto sens_vec = createThyraVector(distParamLib->get(sensitivity_name.str())->vector());
        sens_vec->update(1.0, *fpV->col(0));
        distParamLib->get(sensitivity_name.str())->scatter();
      }
    } else {
      cas_manager->combine(overlapped_fpV,fpV,CombineMode::ADD);
    }
  } // End timer

  // Apply Dirichlet conditions using dfm (Dirchelt Field Manager)
  if (!trans && dfm != Teuchos::null) {
    PHAL::Workset workset;

    workset.fpV = fpV;
    workset.Vp = V_bc;
    workset.transpose_dist_param_deriv = trans;

    double const
    this_time = fixTime(current_time);

    workset.current_time = this_time;

    dfm_set(workset, x, xdot, xdotdot, rc_mgr);

    loadWorksetNodesetInfo(workset);

    // FillType template argument used to specialize Sacado
#ifdef DEBUG_OUTPUT2
    std::cout << "calling DFM evaluate fields AGAIN in applyGlobalDistParamDerivImpl" << std::endl;
#endif
    dfm->evaluateFields<PHAL::AlbanyTraits::DistParamDeriv>(workset);
  }
}

void Albany::Application::evaluateResponse(
    int response_index, const double current_time,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
    const Teuchos::Array<ParamVec> &p,
    const Teuchos::RCP<Thyra_Vector>& g)
{
  double const this_time = fixTime(current_time);
  responses[response_index]->evaluateResponse(
      this_time, x, xdot, xdotdot, p, g);
}

void Albany::Application::evaluateResponseTangent(
    int response_index, const double alpha, const double beta,
    const double omega, const double current_time, bool sum_derivs,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
    const Teuchos::Array<ParamVec> &p,
    ParamVec *deriv_p,
    const Teuchos::RCP<const Thyra_MultiVector>& Vx,
    const Teuchos::RCP<const Thyra_MultiVector>& Vxdot,
    const Teuchos::RCP<const Thyra_MultiVector>& Vxdotdot,
    const Teuchos::RCP<const Thyra_MultiVector>& Vp,
    const Teuchos::RCP<Thyra_Vector>& g,
    const Teuchos::RCP<Thyra_MultiVector>& gx,
    const Teuchos::RCP<Thyra_MultiVector>& gp)
{
  double const
  this_time = fixTime(current_time);
  responses[response_index]->evaluateTangent(
      alpha, beta, omega, this_time, sum_derivs, x, xdot, xdotdot, p,
      deriv_p, Vx, Vxdot, Vxdotdot, Vp, g, gx, gp);
}

void Albany::Application::evaluateResponseDerivative(
    int response_index, const double current_time,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
    const Teuchos::Array<ParamVec> &p, ParamVec *deriv_p,
    const Teuchos::RCP<Thyra_Vector>& g,
    const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dx,
    const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dxdot,
    const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dxdotdot,
    const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dp)
{
  double const this_time = fixTime(current_time);

  responses[response_index]->evaluateDerivative(
      this_time, x, xdot, xdotdot, p, deriv_p, g, dg_dx, dg_dxdot,
      dg_dxdotdot, dg_dp);
}

void Albany::Application::evaluateResponseDistParamDeriv(
    int response_index, const double current_time,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
    const Teuchos::Array<ParamVec> &param_array,
    const std::string &dist_param_name,
    const Teuchos::RCP<Thyra_MultiVector>& dg_dp)
{
  TEUCHOS_FUNC_TIME_MONITOR(
      "> Albany Fill: Response Distributed Parameter Derivative");
  double const
  this_time = fixTime(current_time);

  responses[response_index]->evaluateDistParamDeriv(this_time, x, xdot, xdotdot, param_array, dist_param_name, dg_dp);

  if (!dg_dp.is_null()) {
    std::stringstream sensitivity_name;
    sensitivity_name << dist_param_name << "_sensitivity";
    // TODO: make distParamLib Thyra
    if (distParamLib->has(sensitivity_name.str())) {
      auto sensitivity_vec = createThyraVector(distParamLib->get(sensitivity_name.str())->vector());
      // sensitivity_vec = dg_dp->col(0).
      // FIXME This is not correct if the part of sensitivity due to the Lagrange multiplier (fpV) is computed first.
      scale_and_update(sensitivity_vec,0.0,dg_dp->col(0),1.0);
      distParamLib->get(sensitivity_name.str())->scatter();
    }
  }
}

#if defined(ALBANY_EPETRA)
void Albany::Application::evaluateStateFieldManager(
    const double current_time, const Epetra_Vector *xdot,
    const Epetra_Vector *xdotdot, const Epetra_Vector &x) {
  // Scatter x and xdot to the overlapped distrbution
  solMgr->scatterX(x, xdot, xdotdot);

  // Create Tpetra copy of x, called xT
  Teuchos::RCP<const Tpetra_Vector> xT =
      Petra::EpetraVector_To_TpetraVectorConst(x, commT);
  // Create Tpetra copy of xdot, called xdotT
  Teuchos::RCP<const Tpetra_Vector> xdotT;
  if (xdot != NULL && num_time_deriv > 0) {
    xdotT = Petra::EpetraVector_To_TpetraVectorConst(*xdot, commT);
  }
  // Create Tpetra copy of xdotdot, called xdotdotT
  Teuchos::RCP<const Tpetra_Vector> xdotdotT;
  if (xdotdot != NULL && num_time_deriv > 1) {
    xdotdotT = Petra::EpetraVector_To_TpetraVectorConst(*xdotdot, commT);
  }

  this->evaluateStateFieldManagerT(current_time, xdotT.ptr(), xdotdotT.ptr(),
                                   *xT);
}
#endif

void Albany::Application::evaluateStateFieldManagerT(
    const double current_time, const Tpetra_MultiVector &xT) {
  int num_vecs = xT.getNumVectors();

  if (num_vecs == 1)
    this->evaluateStateFieldManagerT(current_time, Teuchos::null, Teuchos::null,
                                     *xT.getVector(0));
  else if (num_vecs == 2)
    this->evaluateStateFieldManagerT(current_time, xT.getVector(1).ptr(),
                                     Teuchos::null, *xT.getVector(0));
  else
    this->evaluateStateFieldManagerT(current_time, xT.getVector(1).ptr(),
                                     xT.getVector(2).ptr(), *xT.getVector(0));
}

void Albany::Application::evaluateStateFieldManagerT(
    const double current_time, Teuchos::Ptr<const Tpetra_Vector> xdotT,
    Teuchos::Ptr<const Tpetra_Vector> xdotdotT, const Tpetra_Vector &xT) {
  {
    const std::string eval = "SFM_Jacobian";
    if (setupSet.find(eval) == setupSet.end()) {
      setupSet.insert(eval);
      for (int ps = 0; ps < sfm.size(); ++ps) {
        std::vector<PHX::index_size_type> derivative_dimensions;
        derivative_dimensions.push_back(
            PHAL::getDerivativeDimensions<PHAL::AlbanyTraits::Jacobian>(
                this, ps, explicit_scheme));
        sfm[ps]
            ->setKokkosExtendedDataTypeDimensions<PHAL::AlbanyTraits::Jacobian>(
                derivative_dimensions);
        sfm[ps]->postRegistrationSetup("");
      }
      // visualize state field manager
      if (stateGraphVisDetail > 0) {
        bool detail = false;
        if (stateGraphVisDetail > 1)
          detail = true;
        *out << "Phalanx writing graphviz file for graph of Residual fill "
                "(detail ="
             << stateGraphVisDetail << ")" << std::endl;
        *out << "Process using 'dot -Tpng -O state_phalanx_graph' \n"
             << std::endl;
        for (int ps = 0; ps < sfm.size(); ps++) {
          std::stringstream pg;
          pg << "state_phalanx_graph_" << ps;
          sfm[ps]->writeGraphvizFile<PHAL::AlbanyTraits::Residual>(
              pg.str(), detail, detail);
        }
        stateGraphVisDetail = -1;
      }
    }
  }

  Teuchos::RCP<Thyra_Vector> overlapped_f = solMgrT->get_overlapped_f();

  // Load connectivity map and coordinates
  const auto &wsElNodeEqID = disc->getWsElNodeEqID();
  const auto &coords = disc->getCoords();
  const auto &wsEBNames = disc->getWsEBNames();
  const auto &wsPhysIndex = disc->getWsPhysIndex();

  int numWorksets = wsElNodeEqID.size();

  // Scatter xT and xdotT to the overlapped distrbution
  solMgrT->scatterXT(xT, xdotT.get(), xdotdotT.get());

  // Scatter distributed parameters
  distParamLib->scatter();

  // Set data in Workset struct
  PHAL::Workset workset;
  loadBasicWorksetInfo(workset, current_time);
  workset.f = overlapped_f;

  // Perform fill via field manager
  if (Teuchos::nonnull(rc_mgr))
    rc_mgr->beginEvaluatingSfm();
  for (int ws = 0; ws < numWorksets; ws++) {
    loadWorksetBucketInfo<PHAL::AlbanyTraits::Residual>(workset, ws);
    sfm[wsPhysIndex[ws]]->evaluateFields<PHAL::AlbanyTraits::Residual>(workset);
  }
  if (Teuchos::nonnull(rc_mgr))
    rc_mgr->endEvaluatingSfm();
}

void Albany::Application::registerShapeParameters() {
  int numShParams = shapeParams.size();
  if (shapeParamNames.size() == 0) {
    shapeParamNames.resize(numShParams);
    for (int i = 0; i < numShParams; i++)
      shapeParamNames[i] = Albany::strint("ShapeParam", i);
  }
  Albany::DummyParameterAccessor<PHAL::AlbanyTraits::Jacobian, SPL_Traits> *dJ =
      new Albany::DummyParameterAccessor<PHAL::AlbanyTraits::Jacobian,
                                         SPL_Traits>();
  Albany::DummyParameterAccessor<PHAL::AlbanyTraits::Tangent, SPL_Traits> *dT =
      new Albany::DummyParameterAccessor<PHAL::AlbanyTraits::Tangent,
                                         SPL_Traits>();

  // Register Parameter for Residual fill using "this->getValue" but
  // create dummy ones for other type that will not be used.
  for (int i = 0; i < numShParams; i++) {
    *out << "Registering Shape Param " << shapeParamNames[i] << std::endl;
    new Sacado::ParameterRegistration<PHAL::AlbanyTraits::Residual, SPL_Traits>(
        shapeParamNames[i], this, paramLib);
    new Sacado::ParameterRegistration<PHAL::AlbanyTraits::Jacobian, SPL_Traits>(
        shapeParamNames[i], dJ, paramLib);
    new Sacado::ParameterRegistration<PHAL::AlbanyTraits::Tangent, SPL_Traits>(
        shapeParamNames[i], dT, paramLib);
  }
}

PHAL::AlbanyTraits::Residual::ScalarT &
Albany::Application::getValue(const std::string &name) {
  int index = -1;
  for (unsigned int i = 0; i < shapeParamNames.size(); i++) {
    if (name == shapeParamNames[i])
      index = i;
  }
  TEUCHOS_TEST_FOR_EXCEPTION(index == -1, std::logic_error,
                             "Error in GatherCoordinateVector::getValue, \n"
                                 << "   Unrecognized param name: " << name
                                 << std::endl);

  shapeParamsHaveBeenReset = true;

  return shapeParams[index];
}

void Albany::Application::postRegSetup(std::string eval) {
  if (setupSet.find(eval) != setupSet.end())
    return;

  setupSet.insert(eval);

  if (eval == "Residual") {
    for (int ps = 0; ps < fm.size(); ps++)
      fm[ps]->postRegistrationSetupForType<PHAL::AlbanyTraits::Residual>(eval);
    if (dfm != Teuchos::null)
      dfm->postRegistrationSetupForType<PHAL::AlbanyTraits::Residual>(eval);
    if (nfm != Teuchos::null)
      for (int ps = 0; ps < nfm.size(); ps++)
        nfm[ps]->postRegistrationSetupForType<PHAL::AlbanyTraits::Residual>(
            eval);
  } else if (eval == "Jacobian") {
    for (int ps = 0; ps < fm.size(); ps++) {
      std::vector<PHX::index_size_type> derivative_dimensions;
      derivative_dimensions.push_back(
          PHAL::getDerivativeDimensions<PHAL::AlbanyTraits::Jacobian>(
              this, ps, explicit_scheme));
      fm[ps]->setKokkosExtendedDataTypeDimensions<PHAL::AlbanyTraits::Jacobian>(
          derivative_dimensions);
      fm[ps]->postRegistrationSetupForType<PHAL::AlbanyTraits::Jacobian>(eval);
      if (nfm != Teuchos::null && ps < nfm.size()) {
        nfm[ps]
            ->setKokkosExtendedDataTypeDimensions<PHAL::AlbanyTraits::Jacobian>(
                derivative_dimensions);
        nfm[ps]->postRegistrationSetupForType<PHAL::AlbanyTraits::Jacobian>(
            eval);
      }
    }
    if (dfm != Teuchos::null) {
      // amb Need to look into this. What happens with DBCs in meshes having
      // different element types?
      std::vector<PHX::index_size_type> derivative_dimensions;
      derivative_dimensions.push_back(
          PHAL::getDerivativeDimensions<PHAL::AlbanyTraits::Jacobian>(
              this, 0, explicit_scheme));
      dfm->setKokkosExtendedDataTypeDimensions<PHAL::AlbanyTraits::Jacobian>(
          derivative_dimensions);
      dfm->postRegistrationSetupForType<PHAL::AlbanyTraits::Jacobian>(eval);
    }
  } else if (eval == "Tangent") {
    for (int ps = 0; ps < fm.size(); ps++) {
      std::vector<PHX::index_size_type> derivative_dimensions;
      derivative_dimensions.push_back(
          PHAL::getDerivativeDimensions<PHAL::AlbanyTraits::Tangent>(this, ps));
      fm[ps]->setKokkosExtendedDataTypeDimensions<PHAL::AlbanyTraits::Tangent>(
          derivative_dimensions);
      fm[ps]->postRegistrationSetupForType<PHAL::AlbanyTraits::Tangent>(eval);
      if (nfm != Teuchos::null && ps < nfm.size()) {
        nfm[ps]
            ->setKokkosExtendedDataTypeDimensions<PHAL::AlbanyTraits::Tangent>(
                derivative_dimensions);
        nfm[ps]->postRegistrationSetupForType<PHAL::AlbanyTraits::Tangent>(
            eval);
      }
    }
    if (dfm != Teuchos::null) {
      // amb Need to look into this. What happens with DBCs in meshes having
      // different element types?
      std::vector<PHX::index_size_type> derivative_dimensions;
      derivative_dimensions.push_back(
          PHAL::getDerivativeDimensions<PHAL::AlbanyTraits::Tangent>(this, 0));
      dfm->setKokkosExtendedDataTypeDimensions<PHAL::AlbanyTraits::Tangent>(
          derivative_dimensions);
      dfm->postRegistrationSetupForType<PHAL::AlbanyTraits::Tangent>(eval);
    }
  } else if (eval == "Distributed Parameter Derivative") { //!!!
    for (int ps = 0; ps < fm.size(); ps++) {
      std::vector<PHX::index_size_type> derivative_dimensions;
      derivative_dimensions.push_back(
          PHAL::getDerivativeDimensions<PHAL::AlbanyTraits::DistParamDeriv>(
              this, ps));
      fm[ps]
          ->setKokkosExtendedDataTypeDimensions<
              PHAL::AlbanyTraits::DistParamDeriv>(derivative_dimensions);
      fm[ps]->postRegistrationSetupForType<PHAL::AlbanyTraits::DistParamDeriv>(
          eval);
    }
    if (dfm != Teuchos::null) {
      std::vector<PHX::index_size_type> derivative_dimensions;
      derivative_dimensions.push_back(
          PHAL::getDerivativeDimensions<PHAL::AlbanyTraits::DistParamDeriv>(
              this, 0));
      dfm->setKokkosExtendedDataTypeDimensions<
          PHAL::AlbanyTraits::DistParamDeriv>(derivative_dimensions);
      dfm->postRegistrationSetupForType<PHAL::AlbanyTraits::DistParamDeriv>(
          eval);
    }
    if (nfm != Teuchos::null)
      for (int ps = 0; ps < nfm.size(); ps++) {
        std::vector<PHX::index_size_type> derivative_dimensions;
        derivative_dimensions.push_back(
            PHAL::getDerivativeDimensions<PHAL::AlbanyTraits::DistParamDeriv>(
                this, ps));
        nfm[ps]
            ->setKokkosExtendedDataTypeDimensions<
                PHAL::AlbanyTraits::DistParamDeriv>(derivative_dimensions);
        nfm[ps]
            ->postRegistrationSetupForType<PHAL::AlbanyTraits::DistParamDeriv>(
                eval);
      }
  } else
    TEUCHOS_TEST_FOR_EXCEPTION(
        eval != "Known Evaluation Name", std::logic_error,
        "Error in setup call \n"
            << " Unrecognized name: " << eval << std::endl);

  // Write out Phalanx Graph if requested, on Proc 0, for Resid and Jacobian
  bool alreadyWroteResidPhxGraph = false;
  bool alreadyWroteJacPhxGraph = false;

  if (phxGraphVisDetail > 0) {
    bool detail = false;
    if (phxGraphVisDetail > 1)
      detail = true;

    if ((eval == "Residual") && (alreadyWroteResidPhxGraph == false)) {
      *out << "Phalanx writing graphviz file for graph of Residual fill "
              "(detail ="
           << phxGraphVisDetail << ")" << std::endl;
      *out << "Process using 'dot -Tpng -O phalanx_graph' \n" << std::endl;
      for (int ps = 0; ps < fm.size(); ps++) {
        std::stringstream pg;
        pg << "phalanx_graph_" << ps;
        fm[ps]->writeGraphvizFile<PHAL::AlbanyTraits::Residual>(pg.str(),
                                                                detail, detail);
      }
      alreadyWroteResidPhxGraph = true;
      //      phxGraphVisDetail = -1;
    } else if ((eval == "Jacobian") && (alreadyWroteJacPhxGraph == false)) {
      *out << "Phalanx writing graphviz file for graph of Jacobian fill "
              "(detail ="
           << phxGraphVisDetail << ")" << std::endl;
      *out << "Process using 'dot -Tpng -O phalanx_graph' \n" << std::endl;
      for (int ps = 0; ps < fm.size(); ps++) {
        std::stringstream pg;
        pg << "phalanx_graph_jac_" << ps;
        fm[ps]->writeGraphvizFile<PHAL::AlbanyTraits::Jacobian>(pg.str(),
                                                                detail, detail);
      }
      alreadyWroteJacPhxGraph = true;
    }
    // Stop writing out phalanx graphs only when a Jacobian and a Residual graph
    // has been written out
    if ((alreadyWroteResidPhxGraph == true) &&
        (alreadyWroteJacPhxGraph == true))
      phxGraphVisDetail = -2;
  }
}

#if defined(ALBANY_EPETRA) && defined(ALBANY_TEKO)
RCP<Epetra_Operator>
Albany::Application::buildWrappedOperator(const RCP<Epetra_Operator> &Jac,
                                          const RCP<Epetra_Operator> &wrapInput,
                                          bool reorder) const {
  RCP<Epetra_Operator> wrappedOp = wrapInput;
  // if only one block just use orignal jacobian
  if (blockDecomp.size() == 1)
    return (Jac);

  // initialize jacobian
  if (wrappedOp == Teuchos::null)
    wrappedOp = rcp(new Teko::Epetra::StridedEpetraOperator(blockDecomp, Jac));
  else
    rcp_dynamic_cast<Teko::Epetra::StridedEpetraOperator>(wrappedOp)
        ->RebuildOps();

  // test blocked operator for correctness
  if (precParams->get("Test Blocked Operator", false)) {
    bool result =
        rcp_dynamic_cast<Teko::Epetra::StridedEpetraOperator>(wrappedOp)
            ->testAgainstFullOperator(6, 1e-14);

    *out << "Teko: Tested operator correctness:  "
         << (result ? "passed" : "FAILED!") << std::endl;
  }
  return wrappedOp;
}
#endif

void Albany::Application::determinePiroSolver(
    const Teuchos::RCP<Teuchos::ParameterList> &topLevelParams) {

  const Teuchos::RCP<Teuchos::ParameterList> &localProblemParams =
      Teuchos::sublist(topLevelParams, "Problem", true);

  const Teuchos::RCP<Teuchos::ParameterList> &piroParams =
      Teuchos::sublist(topLevelParams, "Piro");

  // If not explicitly specified, determine which Piro solver to use from the
  // problem parameters
  if (!piroParams->getPtr<std::string>("Solver Type")) {

    const std::string secondOrder =
        localProblemParams->get("Second Order", "No");

    TEUCHOS_TEST_FOR_EXCEPTION(
        secondOrder != "No" && secondOrder != "Velocity Verlet" &&
            secondOrder != "Newmark" && secondOrder != "Trapezoid Rule",
        std::logic_error,
        "Invalid value for Second Order: (No, Velocity Verlet, Newmark, "
        "Trapezoid Rule): "
            << secondOrder << "\n");

    // Populate the Piro parameter list accordingly to inform the Piro solver
    // factory
    std::string piroSolverToken;
    if (solMethod == Steady) {
      piroSolverToken = "NOX";
    } else if (solMethod == Continuation) {
      piroSolverToken = "LOCA";
    } else if (solMethod == Transient) {
      piroSolverToken = (secondOrder == "No") ? "Rythmos" : secondOrder;
    } else if (solMethod == TransientTempus) {
      piroSolverToken = (secondOrder == "No") ? "Tempus" : secondOrder;
    } else {
      // Piro cannot handle the corresponding problem
      piroSolverToken = "Unsupported";
    }

    piroParams->set("Solver Type", piroSolverToken);
  }
}

void Albany::Application::loadBasicWorksetInfo(PHAL::Workset &workset,
                                                double current_time)
{
  auto overlapped_MV = solMgrT->getOverlappedSolution_Thyra();
  auto numVectors = overlapped_MV->domain()->dim();

  workset.x = overlapped_MV->col(0);
  workset.xdot = numVectors > 1 ? overlapped_MV->col(1) : Teuchos::null;
  workset.xdotdot = numVectors > 2 ? overlapped_MV->col(2) : Teuchos::null;

  workset.numEqs = neq;
  workset.current_time = current_time;
  workset.distParamLib = distParamLib;
  workset.disc = disc;
  // workset.delta_time = delta_time;
  workset.transientTerms = Teuchos::nonnull(workset.xdot);
  workset.accelerationTerms = Teuchos::nonnull(workset.xdotdot);
}

void Albany::Application::loadBasicWorksetInfoSDBCs(
    PHAL::Workset &workset,
    const Teuchos::RCP<const Thyra_Vector>& owned_sol,
    const double current_time)
{
  // Scatter owned solution into the overlapped one
  auto overlapped_MV = solMgrT->getOverlappedSolution_Thyra();
  auto overlapped_sol = Thyra::createMember(overlapped_MV->range());
  overlapped_sol->assign(0.0);
  solMgrT->get_cas_manager()->scatter(owned_sol,overlapped_sol,CombineMode::INSERT);

  auto numVectors = overlapped_MV->domain()->dim();
  workset.x = overlapped_sol;
  workset.xdot = numVectors > 1 ? overlapped_MV->col(1) : Teuchos::null;
  workset.xdotdot = numVectors > 2 ? overlapped_MV->col(2) : Teuchos::null;

  workset.numEqs = neq;
  workset.current_time = current_time;
  workset.distParamLib = distParamLib;
  workset.disc = disc;
  // workset.delta_time = delta_time;
  workset.transientTerms = Teuchos::nonnull(workset.xdot);
  workset.accelerationTerms = Teuchos::nonnull(workset.xdotdot);
}

void Albany::Application::loadWorksetJacobianInfo(PHAL::Workset &workset,
                                                  const double alpha,
                                                  const double beta,
                                                  const double omega) {
  workset.m_coeff = alpha;
  workset.n_coeff = omega;
  workset.j_coeff = beta;
  workset.ignore_residual = ignore_residual_in_jacobian;
  workset.is_adjoint = is_adjoint;
}

void Albany::Application::loadWorksetNodesetInfo(PHAL::Workset &workset) {
  workset.nodeSets = Teuchos::rcpFromRef(disc->getNodeSets());
  workset.nodeSetCoords = Teuchos::rcpFromRef(disc->getNodeSetCoords());
}

void Albany::Application::setScale(Teuchos::RCP<const Thyra_LinearOp> jac) 
{
  if (scaleBCdofs == true) {
    if (scaleVec_->norm_2() == 0.0) { 
      scaleVec_->assign(1.0);  
    }
    return; 
  }

  if (scale_type == CONSTANT) { // constant scaling
    scaleVec_->assign(1.0 / scale);
  } else if (scale_type == DIAG) { // diagonal scaling
    if (jac == Teuchos::null) {
      scaleVec_->assign(1.0);
    } else {
      getDiagonalCopy(jac,scaleVec_);
      Thyra::reciprocal(*scaleVec_,scaleVec_.ptr());
    }
  } else if (scale_type == ABSROWSUM) { // absolute value of row sum scaling
    if (jac == Teuchos::null) {
      scaleVec_->assign(1.0);
    } else {
      scaleVec_->assign(0.0);
      // We MUST be able to cast the linear op to RowStatLinearOpBase, in order to get row informations
      auto jac_row_stat = Teuchos::rcp_dynamic_cast<const Thyra::RowStatLinearOpBase<ST>>(jac,true);

      // Compute the inverse of the absolute row sum
      jac_row_stat->getRowStat(Thyra::RowStatLinearOpBaseUtils::ROW_STAT_INV_ROW_SUM,scaleVec_.ptr());
    }
  }
}

void Albany::Application::setScaleBCDofs(PHAL::Workset &workset, Teuchos::RCP<const Thyra_LinearOp> jac) 
{
  //First step: set scaleVec_ to all 1.0s if it is all 0s
  if (scaleVec_->norm_2() == 0) { 
    scaleVec_->assign(1.0);
  }

  //If calling setScaleBCDofs with null Jacobian, don't recompute the scaling
  if (jac == Teuchos::null) {
    return;
  }

  //For diagonal or abs row sum scaling, set the scale equal to the maximum magnitude value 
  //of the diagonal / abs row sum (inf-norm).  This way, scaling adjusts throughout 
  //the simulation based on the Jacobian.  
  Teuchos::RCP<Thyra_Vector> tmp = Thyra::createMember(scaleVec_->space());
  if (scale_type == DIAG) {
    getDiagonalCopy(jac,tmp);
    scale = tmp->norm_inf(); 
  } else if (scale_type == ABSROWSUM) {
    // We MUST be able to cast the linear op to RowStatLinearOpBase, in order to get row informations
    auto jac_row_stat = Teuchos::rcp_dynamic_cast<const Thyra::RowStatLinearOpBase<ST>>(jac,true);

    // Compute the absolute row sum
    jac_row_stat->getRowStat(Thyra::RowStatLinearOpBaseUtils::ROW_STAT_ROW_SUM,tmp.ptr());
    scale = tmp->norm_inf(); 
  }

  if (scale == 0.0) {
    scale = 1.0; 
  }

  // TODO: cast scaleVec_ to SpmdVectorBase, and get the local data.
  //       Right now, the getNonconstLocalData in SpmdVectorBase does not work correctly
  //       for Tpetra, since the Tpetra host view is not marked as modified (see Trilinos issue #3180)
  //       Therefore, for now we ASSUME the underlying vector
  auto scaleVecLocalData = getNonconstLocalData(scaleVec_);
  for (int ns=0; ns<nodeSetIDs_.size(); ns++) {
    std::string key = nodeSetIDs_[ns]; 
    //std::cout << "IKTIKT key = " << key << std::endl; 
    const std::vector<std::vector<int> >& nsNodes = workset.nodeSets->find(key)->second;
    for (unsigned int i = 0; i < nsNodes.size(); i++) {
      //std::cout << "IKTIKT ns, offsets size: " << ns << ", " << offsets_[ns].size() << "\n";
      for (unsigned j = 0; j < offsets_[ns].size(); j++) {
        int lunk = nsNodes[i][offsets_[ns][j]];
        scaleVecLocalData[lunk] = scale;
      }
    }
  } 

  if (problem->getSideSetEquations().size() > 0) {
  
    TEUCHOS_TEST_FOR_EXCEPTION(
      true,
      std::logic_error,
      "Albany::Application::setScaleBCDofs is not yet implemented for"
          << " sideset equations!\n"); 

    // Case where we have sideset equations: loop through sidesets' nodesets
    // Note: the side discretizations' nodesets are indexed progressively:
    //       nodeSet_0,...,nodeSetN,
    //       side0_nodeSet0,...,side0_nodeSetN0,
    //       ...,
    //       sideM_nodeSet0,...,sideM_nodeSetNM
    // Therefore, we simply loop through the sideset discretizations (in order)
    // and in each one we loop through its nodesets
    //
    // IKT, FIXME, 6/30/18: the code below needs to be reimplemented using
    // nodeSetIDs_, as done above, if wishing to use setScaleBCDofs with 
    // nodeset specified via the sideset discretization.
    //
    /*const auto &sdn =
        disc->getMeshStruct()->getMeshSpecs()[0]->sideSetMeshNames;
    for (int isd = 0; isd < sdn.size(); ++isd) {
      const auto &sd = disc->getSideSetDiscretizations().at(sdn[isd]);
      for (auto iterator = sd->getNodeSets().begin();
           iterator != sd->getNodeSets().end(); iterator++) {
        // std::cout << "key: " << iterator->first <<  std::endl;
        const std::vector<std::vector<int>> &nsNodes = iterator->second;
        for (unsigned int i = 0; i < nsNodes.size(); i++) {
          // std::cout << "l, offsets size: " << l << ", " << offsets_[l].size()
          // << std::endl;
          for (unsigned j = 0; j < offsets_[l].size(); j++) {
            int lunk = nsNodes[i][offsets_[l][j]];
            // std::cout << "l, j, i, offsets_: " << l << ", " << j << ", " << i
            // << ", " << offsets_[l][j] << std::endl;  std::cout << "lunk = " <<
            // lunk << std::endl;
            scaleVec_->replaceLocalValue(lunk, scale);
          }
        }
        l++;
      }
    }*/
  }
  /*std::cout << "scaleVec_: " <<std::endl;
  scaleVec_->describe(*out, Teuchos::VERB_EXTREME);*/
}

void Albany::Application::loadWorksetSidesetInfo(PHAL::Workset &workset,
                                                 const int ws) {

  workset.sideSets = Teuchos::rcpFromRef(disc->getSideSets(ws));
}

void Albany::Application::setupBasicWorksetInfo(
    PHAL::Workset &workset, double current_time,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
    const Teuchos::Array<ParamVec> &p)
{
  Teuchos::RCP<const Thyra_MultiVector> overlapped_MV = solMgrT->getOverlappedSolution_Thyra();
  auto numVectors = overlapped_MV->domain()->dim();

  Teuchos::RCP<const Thyra_Vector> overlapped_x = overlapped_MV->col(0);
  Teuchos::RCP<const Thyra_Vector> overlapped_xdot = numVectors > 1 ? overlapped_MV->col(1) : Teuchos::null;
  Teuchos::RCP<const Thyra_Vector> overlapped_xdotdot = numVectors > 2 ? overlapped_MV->col(2) : Teuchos::null;

  // Scatter xT and xdotT to the overlapped distrbution
  solMgrT->scatterX(x, xdot, xdotdot);

  // Scatter distributed parameters
  distParamLib->scatter();

  // Set parameters
  for (int i = 0; i < p.size(); i++) {
    for (unsigned int j = 0; j < p[i].size(); j++) {
      p[i][j].family->setRealValueForAllTypes(p[i][j].baseValue);
  }}

  workset.x = overlapped_x;
  workset.xdot = overlapped_xdot;
  workset.xdotdot = overlapped_xdotdot;
  workset.distParamLib = distParamLib;
  workset.disc = disc;

  const double this_time = fixTime(current_time);

  workset.current_time = this_time;

  workset.transientTerms = Teuchos::nonnull(workset.xdot);
  workset.accelerationTerms = Teuchos::nonnull(workset.xdotdot);

  workset.comm = commT;

  workset.x_cas_manager = solMgrT->get_cas_manager();
}

void Albany::Application::setupTangentWorksetInfo(
    PHAL::Workset &workset, double current_time, bool sum_derivs,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
    const Teuchos::Array<ParamVec> &p,
    ParamVec *deriv_p,
    const Teuchos::RCP<const Thyra_MultiVector>& Vx,
    const Teuchos::RCP<const Thyra_MultiVector>& Vxdot,
    const Teuchos::RCP<const Thyra_MultiVector>& Vxdotdot,
    const Teuchos::RCP<const Thyra_MultiVector>& Vp)
{
  setupBasicWorksetInfo(workset, current_time, x, xdot, xdotdot, p);

  // Scatter Vx dot the overlapped distribution
  RCP<Thyra_MultiVector> overlapped_Vx;
  if (Vx != Teuchos::null) {
    overlapped_Vx = Thyra::createMembers(disc->getOverlapVectorSpace(),Vx->domain()->dim());
    overlapped_Vx->assign(0.0);
    solMgrT->get_cas_manager()->scatter(Vx,overlapped_Vx,Albany::CombineMode::INSERT);
  }

  // Scatter Vx dot the overlapped distribution
  RCP<Thyra_MultiVector> overlapped_Vxdot;
  if (Vxdot != Teuchos::null) {
    overlapped_Vxdot = Thyra::createMembers(disc->getOverlapVectorSpace(),Vxdot->domain()->dim());
    overlapped_Vxdot->assign(0.0);
    solMgrT->get_cas_manager()->scatter(Vxdot,overlapped_Vxdot,Albany::CombineMode::INSERT);
  }
  RCP<Thyra_MultiVector> overlapped_Vxdotdot;
  if (Vxdotdot != Teuchos::null) {
    overlapped_Vxdotdot = Thyra::createMembers(disc->getOverlapVectorSpace(),Vxdotdot->domain()->dim());
    overlapped_Vxdotdot->assign(0.0);
    solMgrT->get_cas_manager()->scatter(Vxdotdot,overlapped_Vxdotdot,Albany::CombineMode::INSERT);
  }

  // RCP<const Epetra_MultiVector > vp = rcp(Vp, false);
  RCP<ParamVec> params = rcp(deriv_p, false);

  // Number of x & xdot tangent directions
  int num_cols_x = 0;
  if (Vx != Teuchos::null) {
    num_cols_x = Vx->domain()->dim();
  } else if (Vxdot != Teuchos::null) {
    num_cols_x = Vxdot->domain()->dim();
  } else if (Vxdotdot != Teuchos::null) {
    num_cols_x = Vxdotdot->domain()->dim();
  }

  // Number of parameter tangent directions
  int num_cols_p = 0;
  if (params != Teuchos::null) {
    if (Vp != Teuchos::null) {
      num_cols_p = Vp->domain()->dim();
    } else {
      num_cols_p = params->size();
    }
  }

  // Whether x and param tangent components are added or separate
  int param_offset = 0;
  if (!sum_derivs)
    param_offset = num_cols_x; // offset of parameter derivs in deriv array

  TEUCHOS_TEST_FOR_EXCEPTION(
      sum_derivs && (num_cols_x != 0) && (num_cols_p != 0) &&
          (num_cols_x != num_cols_p),
      std::logic_error,
      "Seed matrices Vx and Vp must have the same number "
          << " of columns when sum_derivs is true and both are "
          << "non-null!" << std::endl);

  // Initialize
  if (params != Teuchos::null) {
    TanFadType p;
    int num_cols_tot = param_offset + num_cols_p;
    for (unsigned int i = 0; i < params->size(); i++) {
      p = TanFadType(num_cols_tot, (*params)[i].baseValue);
      auto VpT = Albany::getConstTpetraMultiVector(Vp);
      if (VpT != Teuchos::null) {
        Teuchos::ArrayRCP<const ST> VpT_constView;
        for (int k = 0; k < num_cols_p; k++) {
          VpT_constView = VpT->getData(k);
          p.fastAccessDx(param_offset + k) = VpT_constView[i];
        }
      } else
        p.fastAccessDx(param_offset + i) = 1.0;
      (*params)[i].family->setValue<PHAL::AlbanyTraits::Tangent>(p);
    }
  }

  workset.params = params;
  workset.Vx = overlapped_Vx;
  workset.Vxdot = overlapped_Vxdot;
  workset.Vxdotdot = overlapped_Vxdotdot;
  workset.Vp = Vp;
  workset.num_cols_x = num_cols_x;
  workset.num_cols_p = num_cols_p;
  workset.param_offset = param_offset;
}

void Albany::Application::removeEpetraRelatedPLs(
    const Teuchos::RCP<Teuchos::ParameterList> &params) {

  if (params->isSublist("Piro")) {
    Teuchos::ParameterList &piroPL = params->sublist("Piro", true);
    if (piroPL.isSublist("Rythmos")) {
      Teuchos::ParameterList &rytPL = piroPL.sublist("Rythmos", true);
      if (rytPL.isSublist("Stratimikos")) {
        Teuchos::ParameterList &strataPL = rytPL.sublist("Stratimikos", true);
        if (strataPL.isSublist("Linear Solver Types")) {
          Teuchos::ParameterList &lsPL =
              strataPL.sublist("Linear Solver Types", true);
          if (lsPL.isSublist("AztecOO")) {
            lsPL.remove("AztecOO", true);
          }
          if (strataPL.isSublist("Preconditioner Types")) {
            Teuchos::ParameterList &precPL =
                strataPL.sublist("Preconditioner Types", true);
            if (precPL.isSublist("ML")) {
              precPL.remove("ML", true);
            }
          }
        }
      }
    }
    if (piroPL.isSublist("NOX")) {
      Teuchos::ParameterList &noxPL = piroPL.sublist("NOX", true);
      if (noxPL.isSublist("Direction")) {
        Teuchos::ParameterList &dirPL = noxPL.sublist("Direction", true);
        if (dirPL.isSublist("Newton")) {
          Teuchos::ParameterList &newPL = dirPL.sublist("Newton", true);
          if (newPL.isSublist("Stratimikos Linear Solver")) {
            Teuchos::ParameterList &stratPL =
                newPL.sublist("Stratimikos Linear Solver", true);
            if (stratPL.isSublist("Stratimikos")) {
              Teuchos::ParameterList &strataPL =
                  stratPL.sublist("Stratimikos", true);
              if (strataPL.isSublist("AztecOO")) {
                strataPL.remove("AztecOO", true);
              }
              if (strataPL.isSublist("Linear Solver Types")) {
                Teuchos::ParameterList &lsPL =
                    strataPL.sublist("Linear Solver Types", true);
                if (lsPL.isSublist("AztecOO")) {
                  lsPL.remove("AztecOO", true);
                }
              }
            }
          }
        }
      }
    }
  }
}

#if defined(ALBANY_LCM)
void Albany::Application::setCoupledAppBlockNodeset(
    std::string const &app_name, std::string const &block_name,
    std::string const &nodeset_name) {
  // Check for valid application name
  auto it = app_name_index_map_->find(app_name);

  TEUCHOS_TEST_FOR_EXCEPTION(
      it == app_name_index_map_->end(), std::logic_error,
      "Trying to couple to an unknown Application: " << app_name << '\n');

  int const app_index = it->second;

  auto block_nodeset_names = std::make_pair(block_name, nodeset_name);

  auto app_index_block_names = std::make_pair(app_index, block_nodeset_names);

  coupled_app_index_block_nodeset_names_map_.insert(app_index_block_names);
}
#endif

void Albany::Application::computeGlobalResidualSDBCsImpl(
    double const current_time,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
    Teuchos::Array<ParamVec> const &p,
    const Teuchos::RCP<Thyra_Vector>& f,
    double const dt) {
  TEUCHOS_FUNC_TIME_MONITOR("> Albany Fill: Residual");
  postRegSetup("Residual");

  if (scale != 1.0) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
                               "'Scaling' sublist not recognized when using SDBCs. \n" <<
                               "To use scaling with SDBCs, specify 'Scaled SDBC' and/or \n" <<
                               "'SDBC Scaling' in 'Dirichlet BCs' sublist."); 
  }

//#define DEBUG_OUTPUT

#ifdef DEBUG_OUTPUT
  *out << "IKT prev_times_ size = " << prev_times_.size() << '\n';
#endif
  int app_no = 0;
#ifdef ALBANY_LCM
  if (app_index_ < 0)
    app_no = 0;
  else
    app_no = app_index_;
#endif

  bool begin_time_step = false;
#ifdef ALBANY_LCM
  current_app = app_index_;
#ifdef DEBUG_OUTPUT
  *out << " IKT current_app, previous_app = " << current_app << ", "
       << previous_app << '\n';
#endif
#endif

  // Load connectivity map and coordinates
  const auto &wsElNodeEqID = disc->getWsElNodeEqID();
  const auto &coords = disc->getCoords();
  const auto &wsEBNames = disc->getWsEBNames();
  const auto &wsPhysIndex = disc->getWsPhysIndex();

  int const numWorksets = wsElNodeEqID.size();

  const Teuchos::RCP<Thyra_Vector> overlapped_f = solMgrT->get_overlapped_f();
  const auto cas_manager = solMgrT->get_cas_manager();

  // Scatter x and xdot to the overlapped distrbution
  solMgrT->scatterX(x, xdot, xdotdot);

  // Scatter distributed parameters
  distParamLib->scatter();

  // Set parameters
  for (int i = 0; i < p.size(); i++) {
    for (unsigned int j = 0; j < p[i].size(); j++) {
      p[i][j].family->setRealValueForAllTypes(p[i][j].baseValue);
    }
  }

#if defined(ALBANY_LCM)
  // Store pointers to solution and time derivatives.
  // Needed for Schwarz coupling.
  if (x != Teuchos::null) {
    x_ = Teuchos::rcp(new Tpetra_Vector(*Albany::getConstTpetraVector(x)));
  } else {
    x_ = Teuchos::null;
  }
  if (xdot != Teuchos::null) {
    xdot_ = Teuchos::rcp(new Tpetra_Vector(*Albany::getConstTpetraVector(xdot)));
  } else {
    xdot_ = Teuchos::null;
  }
  if (xdotdot != Teuchos::null) {
    xdotdot_ = Teuchos::rcp(new Tpetra_Vector(*Albany::getConstTpetraVector(xdotdot)));
  } else {
    xdotdot_ = Teuchos::null;
  }
#endif // ALBANY_LCM

  // Zero out overlapped residual - Tpetra
  overlapped_f->assign(0.0);
  f->assign(0.0);

#ifdef ALBANY_PERIDIGM
#if defined(ALBANY_EPETRA)
  const Teuchos::RCP<LCM::PeridigmManager> &peridigmManager =
      LCM::PeridigmManager::self();
  if (Teuchos::nonnull(peridigmManager)) {
    peridigmManager->setCurrentTimeAndDisplacement(current_time, Albany::getConstTpetraVector(x));
    peridigmManager->evaluateInternalForce();
  }
#endif
#endif

  // Set data in Workset struct, and perform fill via field manager
  {
    if (Teuchos::nonnull(rc_mgr)) {
      rc_mgr->init_x_if_not(x->space());
    }

    PHAL::Workset workset;

    double const
    this_time = fixTime(current_time);

    loadBasicWorksetInfo(workset, this_time);
    
    workset.time_step = dt; 

#ifdef DEBUG_OUTPUT
    *out << "IKT previous_time, this_time = " << prev_times_[app_no] << ", "
         << this_time << "\n";
#endif
    // Check if previous_time is same as current time.  If not, we are at the
    // start  of a new time step, so we set boolean parameter to true.
    begin_time_step = prev_times_[app_no] != this_time;

    workset.f = overlapped_f;

    for (int ws = 0; ws < numWorksets; ws++) {
      loadWorksetBucketInfo<PHAL::AlbanyTraits::Residual>(workset, ws);

#ifdef DEBUG_OUTPUT
      *out << "IKT countRes = " << countRes
           << ", computeGlobalResid workset.x = \n ";
      describe(workset.x.getConst(),*out,Teuchos::VERB_EXTREME);
#endif

      // FillType template argument used to specialize Sacado
#ifdef DEBUG_OUTPUT2
      std::cout << "calling FM evaluate fields in computeGlobalResidualSDBCsImplT" << std::endl;
#endif
      fm[wsPhysIndex[ws]]->evaluateFields<PHAL::AlbanyTraits::Residual>(
          workset);
      if (nfm != Teuchos::null) {
#ifdef ALBANY_PERIDIGM
        // DJL this is a hack to avoid running a block with sphere elements
        // through a Neumann field manager that was constructed for a non-sphere
        // element topology.  The root cause is that Albany currently supports
        // only a single Neumann field manager.  The history on that is murky.
        // The single field manager is created for a specific element topology,
        // and it fails if applied to worksets with a different element
        // topology. The Peridigm use case is a discretization that contains
        // blocks with sphere elements and blocks with standard FEM solid
        // elements, and we want to apply Neumann BC to the standard solid
        // elements.
        if (workset.sideSets->size() != 0) {
          deref_nfm(nfm, wsPhysIndex, ws)
              ->evaluateFields<PHAL::AlbanyTraits::Residual>(workset);
        }
#else
        deref_nfm(nfm, wsPhysIndex, ws)
            ->evaluateFields<PHAL::AlbanyTraits::Residual>(workset);
#endif
      }
    }
    prev_times_[app_no] = this_time;
    if (previous_app != current_app) {
      begin_time_step = true;
    }
  }

  // Assemble the residual into a non-overlapping vector
  cas_manager->combine(overlapped_f,f,CombineMode::ADD);

#ifdef ALBANY_LCM
  // Push the assembled residual values back into the overlap vector
  cas_manager->scatter(f,overlapped_f,CombineMode::INSERT);
  // Write the residual to the discretization, which will later (optionally)
  // be written to the output file
  disc->setResidualField(overlapped_f);
#endif // ALBANY_LCM

  // Apply Dirichlet conditions using dfm (Dirchelt Field Manager)
  Teuchos::RCP<Thyra_Vector> x_post_SDBCs;
  if (dfm != Teuchos::null) {
    PHAL::Workset workset;

    workset.f = f;

    loadWorksetNodesetInfo(workset);

    dfm_set(workset, x, xdot, xdotdot, rc_mgr);

    double const
    this_time = fixTime(current_time);

    workset.current_time = this_time;

    workset.distParamLib = distParamLib;
    workset.disc = disc;

#if defined(ALBANY_LCM)
    // Needed for more specialized Dirichlet BCs (e.g. Schwarz coupling)
    workset.apps_ = apps_;
    workset.current_app_ = Teuchos::rcp(this, false);
#endif // ALBANY_LCM

    // FillType template argument used to specialize Sacado
#ifdef DEBUG_OUTPUT2
    std::cout << "calling DFM evaluate fields in computeGlobalResidualSDBCsImplT" << std::endl;
#endif
    dfm->evaluateFields<PHAL::AlbanyTraits::Residual>(workset);
    x_post_SDBCs = workset.x->clone_v();
  }

#ifdef DEBUG_OUTPUT
  *out << "IKT begin_time_step? " << begin_time_step << "\n";
#endif

  if (begin_time_step == true) {
    // if (countRes == 0) {
    // Zero out overlapped residual - Tpetra
    overlapped_f->assign(0.0);
    f->assign(0.0);
    PHAL::Workset workset;

    double const
    this_time = fixTime(current_time);

    loadBasicWorksetInfoSDBCs(workset, x_post_SDBCs, this_time);

    workset.f = overlapped_f;

    for (int ws = 0; ws < numWorksets; ws++) {
      loadWorksetBucketInfo<PHAL::AlbanyTraits::Residual>(workset, ws);

#ifdef DEBUG_OUTPUT
      *out << "IKT countRes = " << countRes
           << ", computeGlobalResid workset.x = \n ";
      describe(workset.x.getConst(),*out,Teuchos::VERB_EXTREME);
#endif

      // FillType template argument used to specialize Sacado
#ifdef DEBUG_OUTPUT2
      std::cout << "calling FM evaluate fields AGAIN in computeGlobalResidualSDBCsImplT" << std::endl;
#endif
      fm[wsPhysIndex[ws]]->evaluateFields<PHAL::AlbanyTraits::Residual>(
          workset);
      if (nfm != Teuchos::null) {
#ifdef ALBANY_PERIDIGM
        // DJL this is a hack to avoid running a block with sphere elements
        // through a Neumann field manager that was constructed for a non-sphere
        // element topology.  The root cause is that Albany currently supports
        // only a single Neumann field manager.  The history on that is murky.
        // The single field manager is created for a specific element topology,
        // and it fails if applied to worksets with a different element
        // topology. The Peridigm use case is a discretization that contains
        // blocks with sphere elements and blocks with standard FEM solid
        // elements, and we want to apply Neumann BC to the standard solid
        // elements.
        if (workset.sideSets->size() != 0) {
          deref_nfm(nfm, wsPhysIndex, ws)
              ->evaluateFields<PHAL::AlbanyTraits::Residual>(workset);
        }
#else
        deref_nfm(nfm, wsPhysIndex, ws)
            ->evaluateFields<PHAL::AlbanyTraits::Residual>(workset);
#endif
      }
    }

    // Assemble the residual into a non-overlapping vector
    cas_manager->combine(overlapped_f,f,CombineMode::ADD);

#ifdef ALBANY_LCM
    // Push the assembled residual values back into the overlap vector
    cas_manager->scatter(f,overlapped_f,CombineMode::INSERT);
    // Write the residual to the discretization, which will later (optionally)
    // be written to the output file
    disc->setResidualField(overlapped_f);
#endif // ALBANY_LCM
    if (dfm != Teuchos::null) {

      PHAL::Workset workset;

      workset.f = f;

      loadWorksetNodesetInfo(workset);

      dfm_set(workset, x_post_SDBCs, xdot, xdotdot, rc_mgr);

      double const
      this_time = fixTime(current_time);

      workset.current_time = this_time;

      workset.distParamLib = distParamLib;
      workset.disc = disc;

#if defined(ALBANY_LCM)
      // Needed for more specialized Dirichlet BCs (e.g. Schwarz coupling)
      workset.apps_ = apps_;
      workset.current_app_ = Teuchos::rcp(this, false);
#endif // ALBANY_LCM

      // FillType template argument used to specialize Sacado
      if (MOR_apply_bcs_){
#ifdef DEBUG_OUTPUT2
        std::cout << "calling DFM evaluate fields AGAIN in computeGlobalResidualSDBCsImplT" << std::endl;
#endif
        dfm->evaluateFields<PHAL::AlbanyTraits::Residual>(workset);
      }
    }
  } // endif (begin_time_step == true)
  previous_app = current_app;
}
