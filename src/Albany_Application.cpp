//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_Application.hpp"
#include "Albany_DiscretizationFactory.hpp"
#include "Albany_DistributedParameterLibrary.hpp"
#include "Albany_Macros.hpp"
#include "Albany_ProblemFactory.hpp"
#include "Albany_ResponseFactory.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_Utils.hpp"
#include "Albany_DataTypes.hpp"
#include "Albany_ScalarResponseFunction.hpp"
#include "Albany_StringUtils.hpp"

#include "PHAL_Utilities.hpp"
#include "Albany_KokkosUtils.hpp"
#include "Albany_TpetraThyraUtils.hpp"
#include "Albany_Hessian.hpp"

#include "Thyra_MultiVectorStdOps.hpp"
#include "Thyra_VectorBase.hpp"
#include "Thyra_VectorStdOps.hpp"

#include "Teuchos_TimeMonitor.hpp"
#include "Zoltan2_TpetraCrsColorer.hpp"

#include <stdexcept>
#include <string>

//#define WRITE_TO_MATRIX_MARKET
//#define DEBUG_OUTPUT

using Teuchos::ArrayRCP;
using Teuchos::getFancyOStream;
using Teuchos::RCP;
using Teuchos::rcp;
using Teuchos::rcp_dynamic_cast;
using Teuchos::rcpFromRef;
using Teuchos::TimeMonitor;

int countJac;   // counter which counts instances of Jacobian (for debug output)
int countRes;   // counter which counts instances of residual (for debug output)
int countSoln;  // counter which counts instances of solution (for debug output)
int countScale;

namespace Albany {

Application::Application(
    const RCP<const Teuchos_Comm>&     comm_,
    const RCP<Teuchos::ParameterList>& params,
    const RCP<const Thyra_Vector>&     initial_guess)
    : no_dir_bcs_(false),
      requires_sdbcs_(false),
      requires_orig_dbcs_(false),
      comm(comm_),
      out(Teuchos::VerboseObjectBase::getDefaultOStream()),
      params_(params),
      physicsBasedPreconditioner(false),
      phxGraphVisDetail(0),
      stateGraphVisDetail(0),
      morphFromInit(true),
      perturbBetaForDirichlets(0.0)
{
  initialSetUp(params);
  createMeshSpecs();
  buildProblem();
  createDiscretization();
  finalSetUp(params, initial_guess);
}

Application::Application(const RCP<const Teuchos_Comm>& comm_)
    : no_dir_bcs_(false),
      requires_sdbcs_(false),
      requires_orig_dbcs_(false),
      comm(comm_),
      out(Teuchos::VerboseObjectBase::getDefaultOStream()),
      physicsBasedPreconditioner(false),
      phxGraphVisDetail(0),
      stateGraphVisDetail(0),
      morphFromInit(true),
      perturbBetaForDirichlets(0.0)
{
  // Nothing to be done here
}

void
Application::initialSetUp(const RCP<Teuchos::ParameterList>& params)
{
  // Create parameter libraries
  paramLib     = rcp(new ParamLib);
  distParamLib = rcp(new DistributedParameterLibrary);

#ifdef ALBANY_DEBUG
  int break_set  = (getenv("ALBANY_BREAK") == NULL) ? 0 : 1;
  int env_status = 0;
  int length     = 1;
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

  // Create problem object
  problemParams = Teuchos::sublist(params, "Problem", true);

  const auto& problem_type = problemParams->get<std::string>("Name");
  const auto& pb_factories = FactoriesContainer<ProblemFactory>::instance();
  problem = pb_factories.create(problem_type,comm,params,paramLib);
  TEUCHOS_TEST_FOR_EXCEPTION (problem.is_null(), std::runtime_error,
    "Error! Could not create problem '" + problem_type + "'.\n");

  // Validate Problem parameters against list for this specific problem
  problemParams->validateParameters(*(problem->getValidProblemParameters()), 0);

  try {
    tangent_deriv_dim = calcTangentDerivDimension(problemParams);
  } catch (...) {
    tangent_deriv_dim = 1;
  }


  const bool compute_sens = problemParams->get<bool>("Compute Sensitivities", false);
  std::string sens_method = "None"; 
  int sens_param_index = 0; 
  int resp_fn_index = 0; 
  if (params->isSublist("Piro")) {
    Teuchos::RCP<Teuchos::ParameterList> piroParams = Teuchos::sublist(params, "Piro", true);
    if (compute_sens == true) {
      sens_method = piroParams->get<std::string>("Sensitivity Method", "Forward"); 
    }
    if (piroParams->isSublist("Tempus")) {
      Teuchos::RCP<Teuchos::ParameterList> tempusParams = Teuchos::sublist(piroParams, "Tempus", true);
      if (tempusParams->isSublist("Sensitivities")) {
        Teuchos::RCP<Teuchos::ParameterList> sensParams = Teuchos::sublist(tempusParams, "Sensitivities", true);
        sens_param_index = sensParams->get<int>("Sensitivity Parameter Index", 0); 
	      resp_fn_index = sensParams->get<int>("Response Function Index", 0);
      }
    }
  }
  if (compute_sens == true) { 
    if (sens_method == "Adjoint") {
      adjoint_sens = true; 
    }
    else if (sens_method == "None") {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
        "Error!  'Compute Sensitivities' is true but 'Sensitivity Method' is set to 'None'!\n"); 
    }
  }
  else {
    if (sens_method != "None") {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
        "Error!  'Compute Sensitivities' is false but 'Sensitivity Method' has been set to " 
	<< sens_method << ".\n"); 

    }
  }

  // Pull the number of solution vectors out of the problem and send them to the
  // discretization list, if the user specifies this in the problem
  Teuchos::RCP<Teuchos::ParameterList> discParams = Teuchos::sublist(params, "Discretization", true);

  // Initialize Phalanx postRegistration setup
  phxSetup = Teuchos::rcp(new PHAL::Setup());
  phxSetup->init_problem_params(problemParams);

  // If memoization is active, set workset size to -1 (otherwise memoization won't work)
  if (phxSetup->memoizer_active()) {
    int worksetSize = discParams->get("Workset Size", -1);
    TEUCHOS_TEST_FOR_EXCEPTION(worksetSize != -1, std::logic_error,
        "Input error: Memoization is active but Workset Size is not set to -1!\n" <<
        "             A single workset is needed to active memoization.\n")
    discParams->set("Workset Size", -1);
  }

  // Set in Albany_AbstractProblem constructor or in siblings
  num_time_deriv = problemParams->get<int>("Number Of Time Derivatives");

  // Possibly set in the Discretization list in the input file - this overrides
  // the above if set
  int num_time_deriv_from_input =
      discParams->get<int>("Number Of Time Derivatives", -1);
  if (num_time_deriv_from_input <
      0)  // Use the value from the problem by default
    discParams->set<int>("Number Of Time Derivatives", num_time_deriv);
  else
    num_time_deriv = num_time_deriv_from_input;

  discParams->set<std::string>("Sensitivity Method", sens_method);
  discParams->set<int>("Sensitivity Parameter Index", sens_param_index); 
  discParams->set<int>("Response Function Index", resp_fn_index); 

  TEUCHOS_TEST_FOR_EXCEPTION(
      num_time_deriv > 2,
      std::logic_error,
      "Input error: number of time derivatives must be <= 2 "
          << "(solution, solution_dot, solution_dotdot)");

  // Save the solution method to be used
  std::string solutionMethod = problemParams->get("Solution Method", "Steady");
  if (solutionMethod == "Steady") {
    solMethod = Steady;
  } else if (solutionMethod == "Continuation") {
    solMethod            = Continuation;
    bool const have_piro = params->isSublist("Piro");

    ALBANY_ASSERT(have_piro == true, "Error! Piro sublist not found.");

    Teuchos::ParameterList& piro_params = params->sublist("Piro");

    bool const have_nox = piro_params.isSublist("NOX");

    if (have_nox) {
      Teuchos::ParameterList nox_params = piro_params.sublist("NOX");
      std::string            nonlinear_solver =
          nox_params.get<std::string>("Nonlinear Solver");
    }
  } else if (solutionMethod == "Transient") {
    solMethod = Transient;
  } else if (
      solutionMethod == "Transient") {
    solMethod = Transient;

    // Add NOX pre-post-operator for debugging.
    bool const have_piro = params->isSublist("Piro");

    ALBANY_ASSERT(have_piro == true, "Error! Piro sublist not found.\n");

    Teuchos::ParameterList& piro_params = params->sublist("Piro");

    bool const have_dbcs = problemParams->isSublist("Dirichlet BCs");

    if (have_dbcs == false) no_dir_bcs_ = true;

    bool const have_tempus = piro_params.isSublist("Tempus");

    ALBANY_ASSERT(have_tempus == true, "Error! Tempus sublist not found.\n");

    Teuchos::ParameterList& tempus_params = piro_params.sublist("Tempus");

    bool const have_tempus_stepper = tempus_params.isSublist("Tempus Stepper");

    ALBANY_ASSERT(
        have_tempus_stepper == true,
        "Error! Tempus stepper sublist not found.\n");

    Teuchos::ParameterList& tempus_stepper_params =
        tempus_params.sublist("Tempus Stepper");

    std::string stepper_type =
        tempus_stepper_params.get<std::string>("Stepper Type");

    Teuchos::ParameterList nox_params;

    //The following code checks if we are using an Explicit stepper in Tempus, so as 
    //to do appropriate error checking (e.g., disallow DBCs, which do not work with explicit steppers). 
    //IKT, 8/13/2020: warning - the logic here may not encompass all explicit steppers
    //in Tempus! 
    std::string const expl_str = "Explicit"; 
    std::string const forward_eul = "Forward Euler"; 
    bool is_explicit_scheme = false; 
    std::size_t found = stepper_type.find(expl_str); 
    std::size_t found2 = stepper_type.find(forward_eul); 
    if ((found != std::string::npos) || (found2 != std::string::npos)) {
      is_explicit_scheme = true; 
    }
    if ((stepper_type == "General ERK") || (stepper_type == "RK1")) {
      is_explicit_scheme = true;
    } 
    
    if ((stepper_type == "Newmark Implicit d-Form") ||
        (stepper_type == "Newmark Implicit a-Form")) {
      bool const have_solver_name =
          tempus_stepper_params.isType<std::string>("Solver Name");

      ALBANY_ASSERT(
          have_solver_name == true,
          "Error! Implicit solver sublist not found.\n");

      std::string const solver_name =
          tempus_stepper_params.get<std::string>("Solver Name");

      Teuchos::ParameterList& solver_name_params =
          tempus_stepper_params.sublist(solver_name);

      bool const have_nox = solver_name_params.isSublist("NOX");

      ALBANY_ASSERT(have_nox == true, "Error! Nox sublist not found.\n");

      nox_params = solver_name_params.sublist("NOX");

      std::string nonlinear_solver =
          nox_params.get<std::string>("Nonlinear Solver");

      // Set flag marking that we are running with Tempus + d-Form Newmark +
      // SDBCs.
      if (stepper_type == "Newmark Implicit d-Form") {
        if (nonlinear_solver != "Line Search Based") {
          TEUCHOS_TEST_FOR_EXCEPTION(
              true,
              std::logic_error,
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
    }
    //Explicit steppers require SDBCs
    if (is_explicit_scheme == true) {
      requires_sdbcs_ = true;
    }
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(
        true,
        std::logic_error,
        "Solution Method must be Steady, Transient, Transient, "
            << "Continuation, not : "
            << solutionMethod);
  }

  bool        expl = false;

  determinePiroSolver(params);

  physicsBasedPreconditioner =
      problemParams->get("Use Physics-Based Preconditioner", false);
  if (physicsBasedPreconditioner) {
    precType = problemParams->get("Physics-Based Preconditioner", "Teko");
#ifdef ALBANY_TEKO
    if (precType == "Teko")
      precParams = Teuchos::sublist(problemParams, "Teko", true);
#endif
  }

  //validate Hessian parameters
  if(problemParams->isSublist("Hessian")) {
    Teuchos::RCP<Teuchos::ParameterList> validHessianParams = Teuchos::rcp(new Teuchos::ParameterList("validHessianParams"));
    validHessianParams->set<bool>("Write Hessian MatrixMarket", false);
    validHessianParams->set<bool>("Use AD for Hessian-vector products (default)", true);

    validHessianParams->sublist("Residual").set<bool>("Use AD for Hessian-vector products (default)", false);
    validHessianParams->sublist("Residual").set<std::string>("Enable AD for Hessian-vector product contributions of", "(x,x) (p0,p0) (p0,p1) (p1,p0) (p1,p1)");
    validHessianParams->sublist("Residual").set<std::string>("Disable AD for Hessian-vector product contributions of", "(x,p0) (x,p1) (p0,x) (p1,x)");

    // We want to allow the user to add hessian settings for responses (parameters) even if those responses (parameters) are not used.
    int maxNumResonses(3),maxNumParameters(3);

    //Make sure that the problem does not require more responses (parameters) than anticipated. 
    if (problemParams->isSublist("Response Functions")) 
      maxNumResonses = std::max(maxNumResonses, problemParams->sublist("Response Functions").get<int>("Number Of Responses"));
    if (problemParams->isSublist("Parameters")) 
      maxNumParameters = std::max(maxNumParameters, problemParams->sublist("Parameters").get<int>("Number Of Parameters"));

    for(int response_index = 0;  response_index < maxNumResonses; response_index++) {
      auto& validHessianResponseParams = validHessianParams->sublist(util::strint("Response", response_index));
      validHessianResponseParams.set<bool>("Use AD for Hessian-vector products (default)", false);
      validHessianResponseParams.set<bool>("Reconstruct H_pp", false);
      validHessianResponseParams.set<std::string>("Enable AD for Hessian-vector product contributions of", "(x,x) (p0,p0) (p0,p1) (p1,p0) (p1,p1)");
      validHessianResponseParams.set<std::string>("Disable AD for Hessian-vector product contributions of", "(x,p0) (x,p1) (p0,x) (p1,x)");
      for (int i=0; i<maxNumParameters; ++i) {
        auto& pl = validHessianResponseParams.sublist(util::strint("Parameter", i), false,"");
        pl.set<bool>("Replace H_pp with Identity",false);
        pl.set<bool>("Reconstruct H_pp using Hessian-vector products",true);
        pl.sublist("H_pp Solver",false,"");
      }
    }

    auto hessianParams = problemParams->sublist("Hessian");
    hessianParams.validateParameters(*validHessianParams,2);
  }



  // Create debug output object
  RCP<Teuchos::ParameterList> debugParams =
      Teuchos::sublist(params, "Debug Output", true);
  writeToMatrixMarketJac =
      debugParams->get("Write Jacobian to MatrixMarket", 0);
  writeToMatrixMarketRes =
      debugParams->get("Write Residual to MatrixMarket", 0);
  writeToMatrixMarketSoln =
      debugParams->get("Write Solution to MatrixMarket", 0);
  writeToCoutJac     = debugParams->get("Write Jacobian to Standard Output", 0);
  writeToCoutRes     = debugParams->get("Write Residual to Standard Output", 0);
  writeToCoutSoln    = debugParams->get("Write Solution to Standard Output", 0);
  derivatives_check_ = debugParams->get<int>("Derivative Check", 0);
  // the above 4 parameters cannot have values < -1
  if (writeToMatrixMarketJac < -1) {
    TEUCHOS_TEST_FOR_EXCEPTION(
        true,
        Teuchos::Exceptions::InvalidParameter,
            "\nError in Albany::Application constructor:  "
            << "Invalid Parameter Write Jacobian to MatrixMarket.  Acceptable "
               "values are -1, 0, 1, 2, ...\n ");
  }
  if (writeToMatrixMarketRes < -1) {
    TEUCHOS_TEST_FOR_EXCEPTION(
        true,
        Teuchos::Exceptions::InvalidParameter,
            "\nError in Albany::Application constructor:  "
            << "Invalid Parameter Write Residual to MatrixMarket.  Acceptable "
               "values are -1, 0, 1, 2, ...\n");
  }
  if (writeToMatrixMarketSoln < -1) {
    TEUCHOS_TEST_FOR_EXCEPTION(
        true,
        Teuchos::Exceptions::InvalidParameter,
            "\nError in Albany::Application constructor:  "
         << "Invalid Parameter Write Solution to MatrixMarket.  Acceptable "
            "values are -1, 0, 1, 2, ... \n");
  }
  if (writeToCoutJac < -1) {
    TEUCHOS_TEST_FOR_EXCEPTION(
        true,
        Teuchos::Exceptions::InvalidParameter,
             "\nError in Albany::Application constructor:  "
            << "Invalid Parameter Write Jacobian to Standard Output.  "
               "Acceptable values are -1, 0, 1, 2, ...\n");
  }
  if (writeToCoutRes < -1) {
    TEUCHOS_TEST_FOR_EXCEPTION(
        true,
        Teuchos::Exceptions::InvalidParameter,
             "\nError in Albany::Application constructor:  "
            << "Invalid Parameter Write Residual to Standard Output.  "
               "Acceptable values are -1, 0, 1, 2, ... \n");
  }
  if (writeToCoutSoln < -1) {
    TEUCHOS_TEST_FOR_EXCEPTION(
        true,
        Teuchos::Exceptions::InvalidParameter,
             "\nError in Albany::Application constructor:  "
            << "Invalid Parameter Write Solution to Standard Output.  "
               "Acceptable values are -1, 0, 1, 2, ... \n");
  }

  countJac = 0;  // initiate counter that counts instances of Jacobian matrix to
                 // 0
  countRes = 0;  // initiate counter that counts instances of residual vector to
                 // 0
  countSoln = 0; // initiate counter that counts instances of solution vector to
                 // 0

  // FIXME: call setScaleBCDofs only on first step rather than at every Newton
  // step.  It's called every step now b/c calling it once did not work for
  // Schwarz problems.
  countScale = 0;
  // Create discretization object
  discFactory = rcp(new Albany::DiscretizationFactory(params, comm, expl));
}

void
Application::createMeshSpecs()
{
  // Get mesh specification object: worksetSize, cell topology, etc
  meshSpecs = discFactory->createMeshSpecs();
}

void
Application::createMeshSpecs(Teuchos::RCP<AbstractMeshStruct> mesh)
{
  // Get mesh specification object: worksetSize, cell topology, etc
  meshSpecs = discFactory->createMeshSpecs(mesh);
}

void
Application::buildProblem()
{
  TEUCHOS_FUNC_TIME_MONITOR("Albany_App: buildProblem");
  problem->buildProblem(meshSpecs, stateMgr);

  if ((requires_sdbcs_ == true) && (problem->useSDBCs() == false) &&
      (no_dir_bcs_ == false)) {
    TEUCHOS_TEST_FOR_EXCEPTION(
        true,
        std::logic_error,
        "Error in Albany::Application: you are using a "
        "solver that requires SDBCs yet you are not "
        "using SDBCs!  Explicit time-steppers require SDBCs.\n");
  }

  if ((requires_orig_dbcs_ == true) && (problem->useSDBCs() == true)) {
    TEUCHOS_TEST_FOR_EXCEPTION(
        true,
        std::logic_error,
        "Error in Albany::Application: you are using a "
        "solver with SDBCs that does not work correctly "
        "with them!\n");
  }

  if ((no_dir_bcs_ == true) && (scaleBCdofs == true)) {
    TEUCHOS_TEST_FOR_EXCEPTION(
        true,
        std::logic_error,
        "Error in Albany::Application: you are attempting "
        "to set 'Scale DOF BCs = true' for a problem with no  "
        "Dirichlet BCs!  Scaling will do nothing.  Re-run "
        "with 'Scale DOF BCs = false'\n");
  }

  neq               = problem->numEquations();
  spatial_dimension = problem->spatialDimension();

  // Construct responses
  // This really needs to happen after the discretization is created for
  // distributed responses, but currently it can't be moved because there
  // are responses that setup states, which has to happen before the
  // discretization is created.  We will delay setup of the distributed
  // responses to deal with this temporarily.
  Teuchos::ParameterList& responseList =
      problemParams->sublist("Response Functions");
  ResponseFactory responseFactory(
      Teuchos::rcp(this, false),
      problem,
      meshSpecs,
      Teuchos::rcp(&stateMgr, false));
  responses            = responseFactory.createResponseFunctions(responseList);
  observe_responses    = responseList.get("Observe Responses", true);
  response_observ_freq = responseList.get("Responses Observation Frequency", 1);
  const Teuchos::Array<unsigned int> defaultDataUnsignedInt;

  // Build state field manager
  sfm.resize(meshSpecs.size());
  Teuchos::RCP<PHX::DataLayout> dummy =
      Teuchos::rcp(new PHX::MDALayout<Dummy>(0));
  for (int ps = 0; ps < meshSpecs.size(); ps++) {
    std::string              elementBlockName = meshSpecs[ps]->ebName;
    std::vector<std::string> responseIDs_to_require =
        stateMgr.getResidResponseIDsToRequire(elementBlockName);
    sfm[ps] = Teuchos::rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
    Teuchos::Array<Teuchos::RCP<const PHX::FieldTag>> tags =
        problem->buildEvaluators(
            *sfm[ps], *meshSpecs[ps], stateMgr, BUILD_STATE_FM, Teuchos::null);
    std::vector<std::string>::const_iterator it;
    for (it = responseIDs_to_require.begin();
         it != responseIDs_to_require.end();
         it++) {
      const std::string&                              responseID = *it;
      PHX::Tag<PHAL::AlbanyTraits::Residual::ScalarT> res_response_tag(
          responseID, dummy);
      sfm[ps]->requireField<PHAL::AlbanyTraits::Residual>(res_response_tag);
    }
  }
}

void
Application::createDiscretization()
{
  TEUCHOS_FUNC_TIME_MONITOR("Albany_App: createDiscretization");
  // Create the full mesh
  disc = discFactory->createDiscretization(
      neq,
      problem->getSideSetEquations(),
      stateMgr.getStateInfoStruct(),
      stateMgr.getSideSetStateInfoStruct(),
      problem->getNullSpace());
  // For extruded meshes, we need the number of layers in postRegistrationSetup
  auto layeredMeshNumbering = disc->getLayeredMeshNumberingGO();
  if (!layeredMeshNumbering.is_null()) {
    int numLayers = layeredMeshNumbering->numLayers;
    phxSetup->set_num_layers(numLayers);
  }
}

void
Application::setScaling(const Teuchos::RCP<Teuchos::ParameterList>& params)
{
  // get info from Scaling parameter list (for scaling Jacobian/residual)
  RCP<Teuchos::ParameterList> scalingParams =
      Teuchos::sublist(params, "Scaling", true);
  scale                 = scalingParams->get<double>("Scale", 0.0);
  scaleBCdofs           = scalingParams->get<bool>("Scale BC Dofs", false);
  std::string scaleType = scalingParams->get<std::string>("Type", "Constant");

  if (scale == 0.0) {
    scale = 1.0;
  }

  if (scaleType == "Constant") {
    scale_type = CONSTANT;
  } else if (scaleType == "Diagonal") {
    scale_type = DIAG;
    scale      = 1.0e1;
  } else if (scaleType == "Abs Row Sum") {
    scale_type = ABSROWSUM;
    scale      = 1.0e1;
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(
        true,
        std::logic_error,
        "The scaling Type you selected "
            << scaleType << " is not supported!"
            << "Supported scaling Types are currently: Constant" << std::endl);
  }

  if (scale == 1.0) scaleBCdofs = false;

  if ((scale != 1.0) && (problem->useSDBCs() == true)) {
    TEUCHOS_TEST_FOR_EXCEPTION(
        true,
        std::logic_error,
        "'Scaling' sublist not recognized when using SDBCs.");
  }
}

void
Application::finalSetUp(
    const Teuchos::RCP<Teuchos::ParameterList>& params,
    const Teuchos::RCP<const Thyra_Vector>&     initial_guess)
{
  setScaling(params);

  // Now that space is allocated in STK for state fields, initialize states.
  // If the states have been already allocated, skip this.
  if (!stateMgr.areStateVarsAllocated()) stateMgr.initStateArrays(disc);

  solMgr = rcp(new SolutionManager(
      params,
      initial_guess,
      paramLib,
      disc,
      comm));

  // Create Distributed parameters and initialize them with data stored in the
  // mesh.
  const StateInfoStruct& distParamSIS = disc->getNodalParameterSIS();
  for (const auto& sis : distParamSIS) {
    // Get name of distributed parameter
    const std::string& param_name = sis->name;

    // Get parameter vector spaces and build parameter vector
    // Create distributed parameter and set workset_elem_dofs
    auto p_dof_mgr = disc->getDOFManager(param_name,sis->meshPart);
    auto parameter = Teuchos::rcp(new DistributedParameter(param_name, p_dof_mgr));

    // Get the vector and lower/upper bounds, and fill them with available
    // data
    Teuchos::RCP<Thyra_Vector> dist_param = parameter->vector();
    Teuchos::RCP<Thyra_Vector> dist_param_lowerbound =
        parameter->lower_bounds_vector();
    Teuchos::RCP<Thyra_Vector> dist_param_upperbound =
        parameter->upper_bounds_vector();

    std::string lowerbound_name, upperbound_name;
    lowerbound_name = param_name + "_lowerbound";
    upperbound_name = param_name + "_upperbound";

    // Initialize parameter with data stored in the mesh
    disc->getField(*dist_param, param_name);
    const auto& nodal_param_states = disc->getNodalParameterSIS();
    bool        has_lowerbound(false), has_upperbound(false);
    for (int ist = 0; ist < static_cast<int>(nodal_param_states.size());
         ist++) {
      has_lowerbound |= nodal_param_states[ist]->name == lowerbound_name;
      has_upperbound |= nodal_param_states[ist]->name == upperbound_name;
    }
    if (has_lowerbound) {
      disc->getField(*dist_param_lowerbound, lowerbound_name);
    } else {
      dist_param_lowerbound->assign(std::numeric_limits<ST>::lowest());
    }
    if (has_upperbound) {
      disc->getField(*dist_param_upperbound, upperbound_name);
    } else {
      dist_param_upperbound->assign(std::numeric_limits<ST>::max());
    }
    // JR: for now, initialize to constant value from user input if requested.
    // This needs to be generalized.
    if (params->sublist("Problem").isType<Teuchos::ParameterList>(
            "Topology Parameters")) {
      Teuchos::ParameterList& topoParams =
          params->sublist("Problem").sublist("Topology Parameters");
      if (topoParams.isType<std::string>("Entity Type") &&
          topoParams.isType<double>("Initial Value")) {
        if (topoParams.get<std::string>("Entity Type") ==
                "Distributed Parameter" &&
            topoParams.get<std::string>("Topology Name") == param_name) {
          double initVal = topoParams.get<double>("Initial Value");
          dist_param->assign(initVal);
        }
      }
    }

    // Add parameter to the distributed parameter library
    distParamLib->add(parameter->name(), parameter);
  }

  // Now setup response functions (see note above)
  for (int i = 0; i < responses.size(); i++) { responses[i]->setup(); }

  // Set up memory for workset
  fm = problem->getFieldManager();
  TEUCHOS_TEST_FOR_EXCEPTION(
      fm == Teuchos::null,
      std::logic_error,
      "getFieldManager not implemented!!!");
  dfm = problem->getDirichletFieldManager();

  offsets_    = problem->getOffsets();
  nodeSetIDs_ = problem->getNodeSetIDs();

  nfm = problem->getNeumannFieldManager();

  if (comm->getRank() == 0) {
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

  // For backward compatibility, use any value at the old location of the
  // "Compute Sensitivity" flag as a default value for the new flag location
  // when the latter has been left undefined
  const std::string              sensitivityToken = "Compute Sensitivities";
  const Teuchos::Ptr<const bool> oldSensitivityFlag(
      problemParams->getPtr<bool>(sensitivityToken));
  if (Teuchos::nonnull(oldSensitivityFlag)) {
    Teuchos::ParameterList& solveParams =
        params->sublist("Piro").sublist("Analysis").sublist("Solve");
    solveParams.get(sensitivityToken, *oldSensitivityFlag);
  }

  // MPerego: Preforming post registration setup here to make sure that the
  // discretization is already created, so that  derivative dimensions are
  // known. Cannot do post registration right before the evaluate , as done for
  // other field managers.  because memoizer hack is needed by Aeras.
  // TODO, determine when it's best to perform post setup registration and fix
  // memoizer hack if needed.
  for (int i = 0; i < responses.size(); ++i) { responses[i]->postRegSetup(); }
}

template<typename Traits>
void
Application::setDynamicLayoutSizes(Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>>& in_fm) const
{
  // get number of worksets
  const int numWorksets = disc->getNumWorksets();

  // compute largest workset size over all worksets for each sideset name
  std::map<std::string, unsigned int> maxSideSetSizes;
  for (int i = 0; i < numWorksets; ++i) {
    const LocalSideSetInfoList& sideSetView = disc->getSideSetViews(i);
    for (auto it = sideSetView.begin(); it != sideSetView.end(); ++it) {
      if (maxSideSetSizes.find(it->first) == maxSideSetSizes.end()) {
        maxSideSetSizes[it->first] = 0;
      }
      unsigned int sideSetSize = it->second.size;
      maxSideSetSizes[it->first] = std::max(maxSideSetSizes[it->first], sideSetSize);
    }
  }

  // Iterate over tags and set extents for sideset fields
  const auto& tags = in_fm->getFieldTagsForSizing<Traits>();
  in_fm->buildDagForType<Traits>();
  for (auto& t : tags) {
    auto& t_dl = t->nonConstDataLayout();
    std::vector<PHX::Device::size_type> t_dims;
    t_dl.dimensions(t_dims);

    // Check if dimension[0] is Side
    std::string first_dim_name = t_dl.name(0);
    std::string t_identifier = t_dl.identifier();
    std::string sideSetName = t_identifier.substr(0, t_identifier.find("<"));

    TEUCHOS_TEST_FOR_EXCEPTION(first_dim_name == PHX::print<Side>() && sideSetName.empty(), std::logic_error, "Dynamic sizing error: Identifier is that of a sideset but has no sideset name.\n");

    if (first_dim_name == PHX::print<Side>() && maxSideSetSizes.find(sideSetName) != maxSideSetSizes.end()) {

      t_dims[0] = maxSideSetSizes[sideSetName];
#ifdef ALBANY_DEBUG
      auto g_dim0 = t_tims[0];
      Teuchos::reduceAll(*comm,Teuchos::REDUCE_SUM,t_dims[0],g_dim0);
      ALBANY_ASSERT (g_dim0>0, std::logic_error,
          "Dynamic sizing error: global extent of first rank should not be 0!\n");
#endif

      switch (t_dims.size()) {
        case 1:
          t_dl.setExtents(t_dims[0]);
          break;
        case 2:
          t_dl.setExtents(t_dims[0], t_dims[1]);
          break;
        case 3:
          t_dl.setExtents(t_dims[0], t_dims[1], t_dims[2]);
          break;
        case 4:
          t_dl.setExtents(t_dims[0], t_dims[1], t_dims[2], t_dims[3]);
          break;
        case 5:
          t_dl.setExtents(t_dims[0], t_dims[1], t_dims[2], t_dims[3], t_dims[4]);
          break;
        case 6:
          t_dl.setExtents(t_dims[0], t_dims[1], t_dims[2], t_dims[3], t_dims[4], t_dims[5]);
          break;
        default:
          TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error,
            "Error! Sideset dynamic sizing as encountered a layout with more field tags than expected.\n");
      }
    }
  }
}

RCP<AbstractDiscretization>
Application::getDiscretization() const
{
  return disc;
}

RCP<AbstractProblem>
Application::getProblem() const
{
  return problem;
}

RCP<const Teuchos_Comm>
Application::getComm() const
{
  return comm;
}

Teuchos::RCP<const Thyra_VectorSpace>
Application::getVectorSpace() const
{
  return disc->getVectorSpace();
}

RCP<Thyra_LinearOp>
Application::createJacobianOp() const
{
  return disc->createJacobianOp();
}

RCP<Thyra_LinearOp>
Application::getPreconditioner()
{
  return Teuchos::null;
}

RCP<ParamLib>
Application::getParamLib() const
{
  return paramLib;
}

RCP<DistributedParameterLibrary>
Application::getDistributedParameterLibrary() const
{
  return distParamLib;
}

int
Application::getNumResponses() const
{
  return responses.size();
}

Teuchos::RCP<AbstractResponseFunction>
Application::getResponse(int i) const
{
  return responses[i];
}

bool
Application::suppliesPreconditioner() const
{
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
inline Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>>&
deref_nfm(
    Teuchos::ArrayRCP<Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>>>& nfm,
    const WorksetArray<int>&       wsPhysIndex,
    int                            ws)
{
  return nfm.size() == 1 ?  // Currently, all problems seem to have one nfm ...
             nfm[0] :       // ... hence this is the intended behavior ...
             nfm[wsPhysIndex[ws]];  // ... and this is not, but may one day be
                                    // again.
}

// Convenience routine for setting dfm workset data. Cut down on redundant code.
void
dfm_set(
    PHAL::Workset&                          workset,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xd,
    const Teuchos::RCP<const Thyra_Vector>& xdd)
{
  workset.x              = x;
  workset.xdot           = xd;
  workset.xdotdot        = xdd;
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
void
checkDerivatives(
    Application&                              app,
    const double                              time,
    const Teuchos::RCP<const Thyra_Vector>&   x,
    const Teuchos::RCP<const Thyra_Vector>&   xdot,
    const Teuchos::RCP<const Thyra_Vector>&   xdotdot,
    const Teuchos::Array<ParamVec>&           p,
    const Teuchos::RCP<const Thyra_Vector>&   fi,
    const Teuchos::RCP<const Thyra_LinearOp>& jacobian,
    const int                                 check_lvl)
{
  if (check_lvl <= 0) { return; }

  // Work vectors. x's map is compatible with f's, so don't distinguish among
  // maps in this function.
  Teuchos::RCP<Thyra_Vector> w1 = Thyra::createMember(x->space());
  Teuchos::RCP<Thyra_Vector> w2 = Thyra::createMember(x->space());
  Teuchos::RCP<Thyra_Vector> w3 = Thyra::createMember(x->space());

  Teuchos::RCP<Thyra_MultiVector> mv;
  if (check_lvl > 1) { mv = Thyra::createMembers(x->space(), 5); }

  // Construct a perturbation.
  const double               delta = 1e-7;
  Teuchos::RCP<Thyra_Vector> xd    = w1;
  xd->randomize(
      -Teuchos::ScalarTraits<ST>::rmax(), Teuchos::ScalarTraits<ST>::rmax());
  Teuchos::RCP<Thyra_Vector> xpd = w2;
  {
    const Teuchos::ArrayRCP<const RealType> x_d   = getLocalData(x);
    const Teuchos::ArrayRCP<RealType>       xd_d  = getNonconstLocalData(xd);
    const Teuchos::ArrayRCP<RealType>       xpd_d = getNonconstLocalData(xpd);
    for (int i = 0; i < x_d.size(); ++i) {
      xd_d[i] = 2 * xd_d[i] - 1;
      if (x_d[i] == 0) {
        // No scalar-level way to get the magnitude of x_i, so just go with
        // something:
        xd_d[i] = xpd_d[i] = delta * xd_d[i];
      } else {
        // Make the perturbation meaningful relative to the magnitude of x_i.
        xpd_d[i] = (1 + delta * xd_d[i]) * x_d[i];  // mult line
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
    scale_and_update(mv->col(0), 0.0, x, 1.0);
    scale_and_update(mv->col(1), 0.0, xd, 1.0);
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
    mv->col(2)->update(1.0, *f);
  }

  // fpd = f(xpd).
  Teuchos::RCP<Thyra_Vector> fpd = w3;
  app.computeGlobalResidual(time, xpd, xdot, xdotdot, p, fpd);

  // fd = fpd - f.
  Teuchos::RCP<Thyra_Vector> fd = fpd;
  scale_and_update(fpd, 1.0, f, -1.0);
  if (Teuchos::nonnull(mv)) { scale_and_update(mv->col(3), 0.0, fd, 1.0); }

  // Jxd = J xd.
  Teuchos::RCP<Thyra_Vector> Jxd = w2;
  jacobian->apply(Thyra::NOTRANS, *xd, Jxd.ptr(), 1.0, 0.0);

  // Norms.
  const ST fdn  = fd->norm_inf();
  const ST Jxdn = Jxd->norm_inf();
  const ST xdn  = xd->norm_inf();
  // d = norm(fd - Jxd).
  Teuchos::RCP<Thyra_Vector> d = fd;
  scale_and_update(d, 1.0, Jxd, -1.0);
  if (Teuchos::nonnull(mv)) { scale_and_update(mv->col(4), 0.0, d, 1.0); }
  const double dn = d->norm_inf();

  // Assess.
  const double den = std::max(fdn, Jxdn), e = dn / den;
  *Teuchos::VerboseObjectBase::getDefaultOStream()
      << "Albany::Application Check Derivatives level " << check_lvl << ":\n"
      << "   reldif(f(x + dx) - f(x), J(x) dx) = " << e
      << ",\n which should be on the order of " << xdn << "\n";

  if (Teuchos::nonnull(mv)) {
    static int        ctr = 0;
    std::stringstream ss;
    ss << "dc" << ctr << ".mm";
    writeMatrixMarket(mv.getConst(), "dc", ctr);
    ++ctr;
  }
}
}  // namespace

PHAL::Workset
Application::set_dfm_workset(
    double const                            current_time,
    const Teuchos::RCP<const Thyra_Vector>  x,
    const Teuchos::RCP<const Thyra_Vector>  x_dot,
    const Teuchos::RCP<const Thyra_Vector>  x_dotdot,
    const Teuchos::RCP<Thyra_Vector>&       f,
    const Teuchos::RCP<const Thyra_Vector>& x_post_SDBCs)
{
  PHAL::Workset workset;

  workset.f = f;

  loadWorksetNodesetInfo(workset);

  if (scaleBCdofs == true) {
    setScaleBCDofs(workset);
#ifdef WRITE_TO_MATRIX_MARKET
    writeMatrixMarket(scaleVec_, scale, countScale);
#endif
    countScale++;
  }

  if (x_post_SDBCs == Teuchos::null)
    dfm_set(workset, x, x_dot, x_dotdot);
  else
    dfm_set(workset, x_post_SDBCs, x_dot, x_dotdot);

  double const this_time = fixTime(current_time);

  workset.current_time = this_time;

  workset.distParamLib = distParamLib;
  workset.disc         = disc;

  return workset;
}

template <>
void
Application::postRegSetup<PHAL::AlbanyTraits::Residual>()
{
  using EvalT = PHAL::AlbanyTraits::Residual;

  std::string evalName = PHAL::evalName<EvalT>("FM",0);
  if (phxSetup->contain_eval(evalName)) return;

  for (int ps = 0; ps < fm.size(); ps++) {
    evalName = PHAL::evalName<EvalT>("FM",ps);
    phxSetup->insert_eval(evalName);

    setDynamicLayoutSizes<EvalT>(fm[ps]);

    fm[ps]->postRegistrationSetupForType<EvalT>(*phxSetup);

    // Update phalanx saved/unsaved fields based on field dependencies
    phxSetup->check_fields(fm[ps]->getFieldTagsForSizing<EvalT>());
    phxSetup->update_fields();

    writePhalanxGraph<EvalT>(fm[ps],evalName,phxGraphVisDetail);
  }
  if (dfm != Teuchos::null) {
    evalName = PHAL::evalName<EvalT>("DFM",0);
    phxSetup->insert_eval(evalName);

    setDynamicLayoutSizes<EvalT>(dfm);

    dfm->postRegistrationSetupForType<EvalT>(*phxSetup);

    // Update phalanx saved/unsaved fields based on field dependencies
    phxSetup->check_fields(dfm->getFieldTagsForSizing<EvalT>());
    phxSetup->update_fields();

    writePhalanxGraph<EvalT>(dfm,evalName,phxGraphVisDetail);
  }
  if (nfm != Teuchos::null)
    for (int ps = 0; ps < nfm.size(); ps++) {
      evalName = PHAL::evalName<EvalT>("NFM",ps);
      phxSetup->insert_eval(evalName);

      setDynamicLayoutSizes<EvalT>(nfm[ps]);

      nfm[ps]->postRegistrationSetupForType<EvalT>(*phxSetup);

      // Update phalanx saved/unsaved fields based on field dependencies
      phxSetup->check_fields(nfm[ps]->getFieldTagsForSizing<EvalT>());
      phxSetup->update_fields();

      writePhalanxGraph<EvalT>(nfm[ps],evalName,phxGraphVisDetail);
    }
}

template <>
void
Application::postRegSetup<PHAL::AlbanyTraits::Jacobian>()
{
  postRegSetupDImpl<PHAL::AlbanyTraits::Jacobian>();
}

template <>
void
Application::postRegSetup<PHAL::AlbanyTraits::Tangent>()
{
  postRegSetupDImpl<PHAL::AlbanyTraits::Tangent>();
}

template <>
void
Application::postRegSetup<PHAL::AlbanyTraits::DistParamDeriv>()
{
  postRegSetupDImpl<PHAL::AlbanyTraits::DistParamDeriv>();
}

template <>
void
Application::postRegSetup<PHAL::AlbanyTraits::HessianVec>()
{
  postRegSetupDImpl<PHAL::AlbanyTraits::HessianVec>();
}

template <typename EvalT>
void
Application::postRegSetupDImpl()
{
  std::string evalName = PHAL::evalName<EvalT>("FM",0);
  if (phxSetup->contain_eval(evalName)) return;

  for (int ps = 0; ps < fm.size(); ps++) {
    evalName = PHAL::evalName<EvalT>("FM",ps);
    phxSetup->insert_eval(evalName);

    std::vector<PHX::index_size_type> derivative_dimensions;
    derivative_dimensions.push_back(
        PHAL::getDerivativeDimensions<EvalT>(this, ps));
    fm[ps]->setKokkosExtendedDataTypeDimensions<EvalT>(derivative_dimensions);
    setDynamicLayoutSizes<EvalT>(fm[ps]);
    fm[ps]->postRegistrationSetupForType<EvalT>(*phxSetup);

    // Update phalanx saved/unsaved fields based on field dependencies
    phxSetup->check_fields(fm[ps]->getFieldTagsForSizing<EvalT>());
    phxSetup->update_fields();

    writePhalanxGraph<EvalT>(fm[ps],evalName,phxGraphVisDetail);

    if (nfm != Teuchos::null && ps < nfm.size()) {
      evalName = PHAL::evalName<EvalT>("NFM",ps);
      phxSetup->insert_eval(evalName);

      nfm[ps]->setKokkosExtendedDataTypeDimensions<EvalT>(derivative_dimensions);
      setDynamicLayoutSizes<EvalT>(nfm[ps]);
      nfm[ps]->postRegistrationSetupForType<EvalT>(*phxSetup);

      // Update phalanx saved/unsaved fields based on field dependencies
      phxSetup->check_fields(nfm[ps]->getFieldTagsForSizing<EvalT>());
      phxSetup->update_fields();

      writePhalanxGraph<EvalT>(nfm[ps],evalName,phxGraphVisDetail);
    }
  }
  if (dfm != Teuchos::null) {
    evalName = PHAL::evalName<EvalT>("DFM",0);
    phxSetup->insert_eval(evalName);

    // amb Need to look into this. What happens with DBCs in meshes having
    // different element types?
    std::vector<PHX::index_size_type> derivative_dimensions;
    derivative_dimensions.push_back(
        PHAL::getDerivativeDimensions<EvalT>(this, 0));
    dfm->setKokkosExtendedDataTypeDimensions<EvalT>(derivative_dimensions);
    setDynamicLayoutSizes<EvalT>(dfm);
    dfm->postRegistrationSetupForType<EvalT>(*phxSetup);

    // Update phalanx saved/unsaved fields based on field dependencies
    phxSetup->check_fields(dfm->getFieldTagsForSizing<EvalT>());
    phxSetup->update_fields();

    writePhalanxGraph<EvalT>(dfm,evalName,phxGraphVisDetail);
  }

  // postRegSetup is where most of the memory is allocated. Let's check device
  // memory consumption and print a warning if it's near the device limit
  if (KU::IsNearDeviceMemoryLimit()) {
    *out << "WARNING: Running low on device memory. Performance degradation "
         << "may occur due to host<->device data migration. Consider using "
         << "more devices." << std::endl;
  }
}

template <typename EvalT>
void
Application::writePhalanxGraph(
    Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>> fieldMgr,
    const std::string& evalName, const int& phxGraphVizDetail)
{
  if (phxGraphVisDetail > 0) {
    const bool detail = (phxGraphVizDetail > 1) ? true : false;
    *out << "Phalanx writing graphviz file for graph of " << evalName << " (detail = "
        << phxGraphVisDetail << ")" << std::endl;
    const std::string graphName = "phalanxGraph" + evalName;
    *out << "Process using 'dot -Tpng -O " << graphName << std::endl;
    fieldMgr->writeGraphvizFile<EvalT>(graphName, detail, detail);

    // Print phalanx setup info
    phxSetup->print(*out);
  }
}


void
Application::computeGlobalResidualImpl(
    double const                           current_time,
    const Teuchos::RCP<const Thyra_Vector> x,
    const Teuchos::RCP<const Thyra_Vector> x_dot,
    const Teuchos::RCP<const Thyra_Vector> x_dotdot,
    Teuchos::Array<ParamVec> const&        /* p */,
    const Teuchos::RCP<Thyra_Vector>&      f,
    double                                 dt)
{
  //#define DEBUG_OUTPUT

  TEUCHOS_FUNC_TIME_MONITOR("Albany Fill: Residual");
  using EvalT = PHAL::AlbanyTraits::Residual;
  postRegSetup<EvalT>();

  // Load connectivity map and coordinates
  const auto& wsPhysIndex  = disc->getWsPhysIndex();

  const int numWorksets = disc->getNumWorksets();

  const Teuchos::RCP<Thyra_Vector> overlapped_f = solMgr->get_overlapped_f();

  Teuchos::RCP<const CombineAndScatterManager> cas_manager =
      solMgr->get_cas_manager();

  // Scatter x and xdot to the overlapped distribution
  solMgr->scatterX(*x, x_dot.ptr(), x_dotdot.ptr());

  // Scatter distributed parameters
  distParamLib->scatter();

  // Zero out overlapped residual
  overlapped_f->assign(0.0);
  f->assign(0.0);

  // Set data in Workset struct, and perform fill via field manager
  {
    TEUCHOS_FUNC_TIME_MONITOR("Albany Residual Fill: Evaluate");

    PHAL::Workset workset;

    double const this_time = fixTime(current_time);

    loadBasicWorksetInfo(workset, this_time);

    Teuchos::RCP<Thyra_Vector> x_post_SDBCs;
    if ((dfm != Teuchos::null) && (problem->useSDBCs() == true)) {
#ifdef DEBUG_OUTPUT
      *out << "IKT before preEvaluate countRes = " << countRes
           << ", computeGlobalResid workset.x = \n ";
      describe(workset.x.getConst(), *out, Teuchos::VERB_EXTREME);
#endif
      workset = set_dfm_workset(current_time, x, x_dot, x_dotdot, f);

      // FillType template argument used to specialize Sacado
      dfm->preEvaluate<EvalT>(workset);
      x_post_SDBCs = workset.x->clone_v();
#ifdef DEBUG_OUTPUT
      *out << "IKT after preEvaluate countRes = " << countRes
           << ", computeGlobalResid workset.x = \n ";
      describe(workset.x.getConst(), *out, Teuchos::VERB_EXTREME);
#endif
      loadBasicWorksetInfoSDBCs(workset, x_post_SDBCs, this_time);
    }

    workset.time_step = dt;

    workset.f = overlapped_f;

    for (int ws = 0; ws < numWorksets; ws++) {
      const std::string evalName = PHAL::evalName<EvalT>("FM", wsPhysIndex[ws]);
      loadWorksetBucketInfo(workset, ws, evalName);

      // FillType template argument used to specialize Sacado
      fm[wsPhysIndex[ws]]->evaluateFields<EvalT>(workset);
#ifdef DEBUG_OUTPUT
      *out << "IKT after fm evaluateFields countRes = " << countRes
           << ", computeGlobalResid workset.x = \n ";
      describe(workset.x.getConst(), *out, Teuchos::VERB_EXTREME);
#endif

      if (nfm != Teuchos::null) {
        deref_nfm(nfm, wsPhysIndex, ws)
            ->evaluateFields<EvalT>(workset);
      }
    }
  }

  // Assemble the residual into a non-overlapping vector
  {
    TEUCHOS_FUNC_TIME_MONITOR("Albany Residual Fill: Export");
    cas_manager->combine(overlapped_f, f, CombineMode::ADD);
  }

  // Allocate scaleVec_
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

#ifdef WRITE_TO_MATRIX_MARKET
  char nameResUnscaled[100];  // create string for file name
  sprintf(nameResUnscaled, "resUnscaled%i_residual", countScale);
  writeMatrixMarket(f, nameResUnscaled);
#endif

  if (scaleBCdofs == false && scale != 1.0) {
    Thyra::ele_wise_scale<ST>(*scaleVec_, f.ptr());
  }

#ifdef WRITE_TO_MATRIX_MARKET
  char nameResScaled[100];  // create string for file name
  sprintf(nameResScaled, "resScaled%i_residual", countScale);
  writeMatrixMarket(f, nameResScaled);
#endif

  // Apply Dirichlet conditions using dfm (Dirchelt Field Manager)

  if (dfm != Teuchos::null) {
    PHAL::Workset workset =
        set_dfm_workset(current_time, x, x_dot, x_dotdot, f);

    // FillType template argument used to specialize Sacado
    dfm->evaluateFields<EvalT>(workset);
  }

  // scale residual by scaleVec_ if scaleBCdofs is on
  if (scaleBCdofs == true) { Thyra::ele_wise_scale<ST>(*scaleVec_, f.ptr()); }
}

void
Application::computeGlobalResidual(
    const double                            current_time,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& x_dot,
    const Teuchos::RCP<const Thyra_Vector>& x_dotdot,
    const Teuchos::Array<ParamVec>&         p,
    const Teuchos::RCP<Thyra_Vector>&       f,
    const double                            dt)
{
  this->computeGlobalResidualImpl(current_time, x, x_dot, x_dotdot, p, f, dt);

  // Debug output
  if (writeToMatrixMarketRes != 0) {  // If requesting writing to MatrixMarket of residual...
    if (writeToMatrixMarketRes == -1) {  // write residual to MatrixMarket every time it arises
      writeMatrixMarket(f, "rhs", countRes);
    }
    else {
      if (countRes == writeToMatrixMarketRes) {  // write residual only at requested count#
        writeMatrixMarket(f, "rhs", countRes);
      }
    }
  }
  if (writeToMatrixMarketSoln != 0) {  // If requesting writing to MatrixMarket of solution...
    if (writeToMatrixMarketSoln == -1) {  // write solution to MatrixMarket every time it arises
      writeMatrixMarket(x, "sol", countSoln);
    }
    else {
      if (countRes == writeToMatrixMarketSoln) {  // write residual only at requested count#
        writeMatrixMarket(x, "sol", countSoln);
      }
    }
  }
  if (writeToCoutRes != 0) {     // If requesting writing of residual to cout...
    if (writeToCoutRes == -1) {  // cout residual time it arises
      std::cout << "Global Residual #" << countRes << ": " << std::endl;
      describe(f.getConst(), *out, Teuchos::VERB_EXTREME);
    } else {
      if (countRes == writeToCoutRes) {  // cout residual only at requested
                                         // count#
        std::cout << "Global Residual #" << countRes << ": " << std::endl;
        describe(f.getConst(), *out, Teuchos::VERB_EXTREME);
      }
    }
  }
  if (writeToCoutSoln != 0) {    // If requesting writing of solution to cout...
    if (writeToCoutSoln == -1) {  // cout solution time it arises
      std::cout << "Global Solution #" << countSoln << ": " << std::endl;
      describe(x.getConst(), *out, Teuchos::VERB_EXTREME);
    } else {
      if (countSoln == writeToCoutSoln) {  // cout solution only at requested
                                           // count #
        std::cout << "Global Solution #" << countSoln << ": " << std::endl;
        describe(x.getConst(), *out, Teuchos::VERB_EXTREME);
      }
    }
  }
  if (writeToMatrixMarketRes != 0 || writeToCoutRes != 0) {
    countRes++;  // increment residual counter
  }
  if (writeToMatrixMarketSoln != 0 || writeToCoutSoln != 0) {
    countSoln++;  // increment solution counter
  }
}

void
Application::computeGlobalJacobianImpl(
    const double                            alpha,
    const double                            beta,
    const double                            omega,
    const double                            current_time,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
    const Teuchos::Array<ParamVec>&         p,
    const Teuchos::RCP<Thyra_Vector>&       f,
    const Teuchos::RCP<Thyra_LinearOp>&     jac,
    const double                            dt)
{
  TEUCHOS_FUNC_TIME_MONITOR("Albany Fill: Jacobian");

  TEUCHOS_TEST_FOR_EXCEPTION (jac.is_null(), std::logic_error,
    "Error! When calling 'computeGlobalJacobianImpl', the Jacobian pointer must be valid.\n");

  using EvalT = PHAL::AlbanyTraits::Jacobian;
  postRegSetup<EvalT>();

  // Load connectivity map and coordinates
  const auto& wsPhysIndex  = disc->getWsPhysIndex();

  const int numWorksets = disc->getNumWorksets();

  Teuchos::RCP<Thyra_Vector> overlapped_f;
  if (Teuchos::nonnull(f)) { overlapped_f = solMgr->get_overlapped_f(); }

  auto cas_manager = solMgr->get_cas_manager();

  // Scatter x and xdot to the overlapped distribution
  solMgr->scatterX(*x, xdot.ptr(), xdotdot.ptr());

  // Scatter distributed parameters
  distParamLib->scatter();

  // Zero out overlapped residual
  if (Teuchos::nonnull(f)) {
    overlapped_f->assign(0.0);
    f->assign(0.0);
  }

  // Zero out Jacobian
  beginFEAssembly(jac);
  assign(jac, 0.0);

  // Set data in Workset struct, and perform fill via field manager
  {
    TEUCHOS_FUNC_TIME_MONITOR("Albany Jacobian Fill: Evaluate");
    PHAL::Workset workset;

    double const this_time = fixTime(current_time);

    loadBasicWorksetInfo(workset, this_time);

    workset.time_step = dt;

#ifdef DEBUG_OUTPUT
    *out << "IKT countJac = " << countJac
         << ", computeGlobalJacobian workset.x = \n";
    describe(workset.x.getConst(), *out, Teuchos::VERB_EXTREME);
#endif

    workset.f   = overlapped_f;
    workset.Jac = jac;
    loadWorksetJacobianInfo(workset, alpha, beta, omega);

    if (!workset.f.is_null()) {
      workset.f_kokkos = getNonconstDeviceData(workset.f);
    }
    
    workset.Jac_kokkos = getNonconstDeviceData(workset.Jac);

    for (int ws = 0; ws < numWorksets; ws++) {
      const std::string evalName = PHAL::evalName<EvalT>("FM", wsPhysIndex[ws]);
      loadWorksetBucketInfo(workset, ws, evalName);

      // FillType template argument used to specialize Sacado
      fm[wsPhysIndex[ws]]->evaluateFields<EvalT>(workset);
      if (Teuchos::nonnull(nfm))
        deref_nfm(nfm, wsPhysIndex, ws)
            ->evaluateFields<EvalT>(workset);
    }
  }

  // This will also assemble global jacobian (i.e., do import/export)
  {
    TEUCHOS_FUNC_TIME_MONITOR("Albany Jacobian Fill: Export");
    endFEAssembly(jac);
  }

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
    cas_manager->combine(overlapped_f, f, CombineMode::ADD);
  }

  // scale Jacobian
  if (scaleBCdofs == false && scale != 1.0) {
#ifdef WRITE_TO_MATRIX_MARKET
    writeMatrixMarket(jac, "jacUnscaled", countScale);
    if (f != Teuchos::null) {
      writeMatrixMarket(f, "resUnscaled", countScale);
    }
#endif
    // set the scaling
    setScale(jac);

    // scale Jacobian
    // We MUST be able to cast jac to ScaledLinearOpBase in order to left
    // scale it.
    auto jac_scaled_lop =
        Teuchos::rcp_dynamic_cast<Thyra::ScaledLinearOpBase<ST>>(jac, true);
    jac_scaled_lop->scaleLeft(*scaleVec_);
    // scale residual
    /*IKTif (Teuchos::nonnull(f)) {
      Thyra::ele_wise_scale<ST>(*scaleVec_,f.ptr());
    }*/
#ifdef WRITE_TO_MATRIX_MARKET
    writeMatrixMarket(jac, "jacScaled", countScale);
    if (f != Teuchos::null) { writeMatrixMarket(f, "resScaled", countScale); }
    writeMatrixMarket(scaeleVec_, "scale", countScale);
#endif
    countScale++;
  }

  // Apply Dirichlet conditions using dfm (Dirchelt Field Manager)
  if (Teuchos::nonnull(dfm)) {
    // Re-open the jacobian
    beginModify(jac);

    PHAL::Workset workset;

    workset.f       = f;
    workset.Jac     = jac;
    workset.m_coeff = alpha;
    workset.n_coeff = omega;
    workset.j_coeff = beta;

    double const this_time = fixTime(current_time);

    workset.current_time = this_time;

    if (beta == 0.0 && perturbBetaForDirichlets > 0.0)
      workset.j_coeff = perturbBetaForDirichlets;

    dfm_set(workset, x, xdot, xdotdot);
    workset.disc = disc;
    workset.distParamLib = distParamLib;

    loadWorksetNodesetInfo(workset);

    if(problem->useSDBCs() == true)
      dfm->preEvaluate<EvalT>(workset);

    if (scaleBCdofs == true) {
      setScaleBCDofs(workset, jac);
#ifdef WRITE_TO_MATRIX_MARKET
      writeMatrixMarket(scaleVec_, scale, countScale);
#endif
      countScale++;
    }

    workset.distParamLib = distParamLib;
    workset.disc         = disc;

    // FillType template argument used to specialize Sacado
    dfm->evaluateFields<EvalT>(workset);

    // Close the jacobian
    endModify(jac);
  }

  // Apply scaling to residual and Jacobian
  if (scaleBCdofs == true) {
    if (Teuchos::nonnull(f)) { Thyra::ele_wise_scale<ST>(*scaleVec_, f.ptr()); }
    // We MUST be able to cast jac to ScaledLinearOpBase in order to left scale
    // it.
    auto jac_scaled_lop =
        Teuchos::rcp_dynamic_cast<Thyra::ScaledLinearOpBase<ST>>(jac, true);
    jac_scaled_lop->scaleLeft(*scaleVec_);
  }

  if (derivatives_check_ > 0) {
    checkDerivatives(
        *this, current_time, x, xdot, xdotdot, p, f, jac, derivatives_check_);
  }
}

void
Application::computeGlobalJacobian(
    const double                            alpha,
    const double                            beta,
    const double                            omega,
    const double                            current_time,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
    const Teuchos::Array<ParamVec>&         p,
    const Teuchos::RCP<Thyra_Vector>&       f,
    const Teuchos::RCP<Thyra_LinearOp>&     jac,
    const double                            dt)
{
  this->computeGlobalJacobianImpl(
      alpha, beta, omega, current_time, x, xdot, xdotdot, p, f, jac, dt);
  // Debut output
  if (writeToMatrixMarketJac != 0) {
    // If requesting writing to MatrixMarket of Jacobian...
    if (writeToMatrixMarketJac == -1) {
      // write jacobian to MatrixMarket every time it arises
      writeMatrixMarket(jac.getConst(), "jac", countJac);
    } else if (countJac == writeToMatrixMarketJac) {
      // write jacobian only at requested count#
      writeMatrixMarket(jac.getConst(), "jac", countJac);
    }
  }
  if (writeToCoutJac != 0) {
    // If requesting writing Jacobian to standard output (cout)...
    if (writeToCoutJac == -1) {  // cout jacobian every time it arises
      *out << "Global Jacobian #" << countJac << ":\n";
      describe(jac.getConst(), *out, Teuchos::VERB_EXTREME);
    } else if (countJac == writeToCoutJac) {
      // cout jacobian only at requested count#
      *out << "Global Jacobian #" << countJac << ":\n";
      describe(jac.getConst(), *out, Teuchos::VERB_EXTREME);
    }
  }

  if (writeToMatrixMarketJac != 0 || writeToCoutJac != 0) {
    countJac++;  // increment Jacobian counter
  }
  // Debut output - residual
  if (f != Teuchos::null) {
    if (writeToMatrixMarketRes != 0) {  // If requesting writing to MatrixMarket of residual...
      if (writeToMatrixMarketRes == -1) {  // write residual to MatrixMarket every time it arises
        writeMatrixMarket(f, "rhs", countRes);
      }
      else {
        if (countRes == writeToMatrixMarketRes) {  // write residual only at requested count#
          writeMatrixMarket(f, "rhs", countRes);
        }
      }
    }
    if (writeToCoutRes != 0) {     // If requesting writing of residual to cout...
      if (writeToCoutRes == -1) {  // cout residual time it arises
        std::cout << "Global Residual #" << countRes << ": " << std::endl;
        describe(f.getConst(), *out, Teuchos::VERB_EXTREME);
      }
      else {
        if (countRes == writeToCoutRes) {  // cout residual only at requested
                                         // count#
          std::cout << "Global Residual #" << countRes << ": " << std::endl;
          describe(f.getConst(), *out, Teuchos::VERB_EXTREME);
        }
      }
    }

    if (writeToMatrixMarketRes != 0 || writeToCoutRes != 0) {
      countRes++;  // increment residual counter
    }
  }
}

void
Application::computeGlobalTangent(
    const double                                 alpha,
    const double                                 beta,
    const double                                 omega,
    const double                                 current_time,
    bool                                         sum_derivs,
    const Teuchos::RCP<const Thyra_Vector>&      x,
    const Teuchos::RCP<const Thyra_Vector>&      xdot,
    const Teuchos::RCP<const Thyra_Vector>&      xdotdot,
    Teuchos::Array<ParamVec>&                    par,
    const int                                    parameter_index,
    const Teuchos::RCP<const Thyra_MultiVector>& Vx,
    const Teuchos::RCP<const Thyra_MultiVector>& Vxdot,
    const Teuchos::RCP<const Thyra_MultiVector>& Vxdotdot,
    const Teuchos::RCP<const Thyra_MultiVector>& Vp,
    const Teuchos::RCP<Thyra_Vector>&            f,
    const Teuchos::RCP<Thyra_MultiVector>&       JV,
    const Teuchos::RCP<Thyra_MultiVector>&       fp)
{
  TEUCHOS_FUNC_TIME_MONITOR("Albany Fill: Tangent");
  using EvalT = PHAL::AlbanyTraits::Tangent;
  postRegSetup<EvalT>();

  // Load connectivity map and coordinates
  const auto& wsPhysIndex  = disc->getWsPhysIndex();

  const int numWorksets = disc->getNumWorksets();

  Teuchos::RCP<Thyra_Vector> overlapped_f = solMgr->get_overlapped_f();

  // The combine-and-scatter manager
  auto cas_manager = solMgr->get_cas_manager();

  // Scatter x and xdot to the overlapped distribution
  solMgr->scatterX(*x, xdot.ptr(), xdotdot.ptr());

  // Scatter distributed parameters
  distParamLib->scatter();

  // Scatter Vx to the overlapped distribution
  RCP<Thyra_MultiVector> overlapped_Vx;
  if (Teuchos::nonnull(Vx)) {
    overlapped_Vx = Thyra::createMembers(
        cas_manager->getOverlappedVectorSpace(), Vx->domain()->dim());
    overlapped_Vx->assign(0.0);
    cas_manager->scatter(Vx, overlapped_Vx, CombineMode::INSERT);
  }

  // Scatter Vxdot to the overlapped distribution
  RCP<Thyra_MultiVector> overlapped_Vxdot;
  if (Teuchos::nonnull(Vxdot)) {
    overlapped_Vxdot = Thyra::createMembers(
        cas_manager->getOverlappedVectorSpace(), Vxdot->domain()->dim());
    overlapped_Vxdot->assign(0.0);
    cas_manager->scatter(Vxdot, overlapped_Vxdot, CombineMode::INSERT);
  }
  RCP<Thyra_MultiVector> overlapped_Vxdotdot;
  if (Teuchos::nonnull(Vxdotdot)) {
    overlapped_Vxdotdot = Thyra::createMembers(
        cas_manager->getOverlappedVectorSpace(), Vxdotdot->domain()->dim());
    overlapped_Vxdotdot->assign(0.0);
    cas_manager->scatter(Vxdotdot, overlapped_Vxdotdot, CombineMode::INSERT);
  }

  // Set parameters
  // We have to reset the parameters here to be sure to zero out the
  // previously used derivatives.
  for (int i = 0; i < par.size(); i++) {
    for (unsigned int j = 0; j < par[i].size(); j++) {
      par[i][j].family->setRealValueForAllTypes(par[i][j].baseValue);
    }
  }

  RCP<ParamVec> params;
  if (parameter_index < par.size()){
    params = Teuchos::rcp(new ParamVec(par[parameter_index]));
  }

  // Zero out overlapped residual
  if (Teuchos::nonnull(f)) {
    overlapped_f->assign(0.0);
    f->assign(0.0);
  }

  RCP<Thyra_MultiVector> overlapped_JV;
  if (Teuchos::nonnull(JV)) {
    overlapped_JV = Thyra::createMembers(
        cas_manager->getOverlappedVectorSpace(), JV->domain()->dim());
    overlapped_JV->assign(0.0);
    JV->assign(0.0);
  }

  RCP<Thyra_MultiVector> overlapped_fp;
  if (Teuchos::nonnull(fp)) {
    overlapped_fp = Thyra::createMembers(
        cas_manager->getOverlappedVectorSpace(), fp->domain()->dim());
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
    param_offset = num_cols_x;  // offset of parameter derivs in deriv array
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
    int        num_cols_tot = param_offset + num_cols_p;

    TEUCHOS_TEST_FOR_EXCEPTION(
      num_cols_tot >  tangent_deriv_dim,
      std::logic_error,
      "Error in Albany::Application::computeGlobalTangent "
       << "The number of derivative columnes cannot exceed the derivative dimension" 
       << std::endl);
    for (unsigned int i = 0; i < params->size(); i++) {
      p = TanFadType(tangent_deriv_dim, (*params)[i].baseValue);
      if (Teuchos::nonnull(Vp)) {
        // ArrayRCP for const view of Vp's vectors
        Teuchos::ArrayRCP<const ST> Vp_constView;
        for (int k = 0; k < num_cols_p; k++) {
          Vp_constView                     = getLocalData(Vp->col(k));
          p.fastAccessDx(param_offset + k) = Vp_constView[i];
        }
      } else
        p.fastAccessDx(param_offset + i) = 1.0;
      (*params)[i].family->setValue<EvalT>(p);
    }
  }

  // Set data in Workset struct, and perform fill via field manager
  {
    TEUCHOS_FUNC_TIME_MONITOR("Albany Tangent Fill: Evaluate");
    PHAL::Workset workset;

    double const this_time = fixTime(current_time);

    loadBasicWorksetInfo(workset, this_time);

    workset.params   = params;
    workset.Vx       = overlapped_Vx;
    workset.Vxdot    = overlapped_Vxdot;
    workset.Vxdotdot = overlapped_Vxdotdot;
    workset.Vp       = Vp;

    workset.f       = overlapped_f;
    workset.JV      = overlapped_JV;
    workset.fp      = overlapped_fp;
    workset.j_coeff = beta;
    workset.m_coeff = alpha;
    workset.n_coeff = omega;

    workset.num_cols_x   = num_cols_x;
    workset.num_cols_p   = num_cols_p;
    workset.param_offset = param_offset;

    for (int ws = 0; ws < numWorksets; ws++) {
      const std::string evalName = PHAL::evalName<EvalT>("FM", wsPhysIndex[ws]);
      loadWorksetBucketInfo(workset, ws, evalName);

      // FillType template argument used to specialize Sacado
      fm[wsPhysIndex[ws]]->evaluateFields<EvalT>(workset);
      if (nfm != Teuchos::null)
        deref_nfm(nfm, wsPhysIndex, ws)
            ->evaluateFields<EvalT>(workset);
    }
  }

  params = Teuchos::null;

  {
    TEUCHOS_FUNC_TIME_MONITOR("Albany Tangent Fill: Export");
    // Assemble global residual
    if (Teuchos::nonnull(f)) {
      cas_manager->combine(overlapped_f, f, CombineMode::ADD);
    }

    // Assemble derivatives
    if (Teuchos::nonnull(JV)) {
      cas_manager->combine(overlapped_JV, JV, CombineMode::ADD);
    }
    if (Teuchos::nonnull(fp)) {
      cas_manager->combine(overlapped_fp, fp, CombineMode::ADD);
    }
  }

  // Apply Dirichlet conditions using dfm (Dirchelt Field Manager)
  if (Teuchos::nonnull(dfm)) {
    PHAL::Workset workset;

    workset.num_cols_x   = num_cols_x;
    workset.num_cols_p   = num_cols_p;
    workset.param_offset = param_offset;

    workset.f       = f;
    workset.fp      = fp;
    workset.JV      = JV;
    workset.j_coeff = beta;
    workset.n_coeff = omega;
    workset.Vx      = Vx;
    dfm_set(workset, x, xdot, xdotdot);

    loadWorksetNodesetInfo(workset);
    workset.distParamLib = distParamLib;

    double const this_time = fixTime(current_time);

    workset.current_time = this_time;

    workset.disc = disc;

    // FillType template argument used to specialize Sacado
    dfm->evaluateFields<EvalT>(workset);
  }
}

void
Application::applyGlobalDistParamDerivImpl(
    const double                                 current_time,
    const Teuchos::RCP<const Thyra_Vector>&      x,
    const Teuchos::RCP<const Thyra_Vector>&      xdot,
    const Teuchos::RCP<const Thyra_Vector>&      xdotdot,
    const Teuchos::Array<ParamVec>&              /* p */,
    const std::string&                           dist_param_name,
    const bool                                   trans,
    const Teuchos::RCP<const Thyra_MultiVector>& V,
    const Teuchos::RCP<Thyra_MultiVector>&       fpV)
{
  TEUCHOS_FUNC_TIME_MONITOR("Albany Fill: Distributed Parameter Derivative");
  using EvalT = PHAL::AlbanyTraits::DistParamDeriv;
  postRegSetup<EvalT>();

  // Load connectivity map and coordinates
  const auto& wsPhysIndex  = disc->getWsPhysIndex();

  const int numWorksets = disc->getNumWorksets();

  // The combin-and-scatter manager
  auto cas_manager = solMgr->get_cas_manager();

  // Scatter x and xdot to the overlapped distribution
  solMgr->scatterX(*x, xdot.ptr(), xdotdot.ptr());

  // Scatter distributed parameters
  distParamLib->scatter();

  Teuchos::RCP<Thyra_MultiVector> overlapped_fpV;
  if (trans) {
    const auto& vs = distParamLib->get(dist_param_name)->overlap_vector_space();
    overlapped_fpV = Thyra::createMembers(vs, V->domain()->dim());
  } else {
    overlapped_fpV = Thyra::createMembers(
        cas_manager->getOverlappedVectorSpace(), fpV->domain()->dim());
  }
  overlapped_fpV->assign(0.0);
  fpV->assign(0.0);

  Teuchos::RCP<Thyra_MultiVector> V_bc;

  // For (df/dp)^T*V, we have to evaluate Dirichlet BC's first
  if (trans && dfm != Teuchos::null) {
    V_bc = V->clone_mv();

    PHAL::Workset workset;

    workset.fpV                        = fpV;
    workset.Vp                         = V;
    workset.Vp_bc                      = V_bc;
    workset.transpose_dist_param_deriv = trans;
    workset.dist_param_deriv_name      = dist_param_name;
    workset.disc                       = disc;

    double const this_time = fixTime(current_time);

    workset.current_time = this_time;

    dfm_set(workset, x, xdot, xdotdot);

    loadWorksetNodesetInfo(workset);

    // FillType template argument used to specialize Sacado
    dfm->preEvaluate<EvalT>(workset);
    dfm->evaluateFields<EvalT>(workset);
  }

  // Import V (after BC's applied) to overlapped distribution
  RCP<Thyra_MultiVector> overlapped_V;
  if (trans) {
    overlapped_V = Thyra::createMembers(
        cas_manager->getOverlappedVectorSpace(), V->domain()->dim());
    overlapped_V->assign(0.0);
    if (dfm != Teuchos::null)
      cas_manager->scatter(V_bc, overlapped_V, CombineMode::INSERT);
    else
      cas_manager->scatter(V, overlapped_V, CombineMode::INSERT);
  } else {
    const auto& vs = distParamLib->get(dist_param_name)->overlap_vector_space();
    overlapped_V   = Thyra::createMembers(vs, V->domain()->dim());
    overlapped_V->assign(0.0);
    distParamLib->get(dist_param_name)->get_cas_manager()
        ->scatter(V, overlapped_V, CombineMode::INSERT);
  }

  // Set data in Workset struct, and perform fill via field manager
  {
    TEUCHOS_FUNC_TIME_MONITOR("Albany Distributed Parameter Derivative Fill: Evaluate");

    PHAL::Workset workset;

    double const this_time = fixTime(current_time);

    loadBasicWorksetInfo(workset, this_time);

    workset.dist_param_deriv_name      = dist_param_name;
    workset.Vp                         = overlapped_V;
    workset.fpV                        = overlapped_fpV;
    workset.transpose_dist_param_deriv = trans;

    for (int ws = 0; ws < numWorksets; ws++) {
      const std::string evalName = PHAL::evalName<EvalT>("FM", wsPhysIndex[ws]);
      loadWorksetBucketInfo(workset, ws, evalName);

      // FillType template argument used to specialize Sacado
      fm[wsPhysIndex[ws]]->evaluateFields<EvalT>(workset);
      if (nfm != Teuchos::null)
        deref_nfm(nfm, wsPhysIndex, ws)
            ->evaluateFields<EvalT>(workset);
    }
  }

  {
    TEUCHOS_FUNC_TIME_MONITOR(
        "Albany Distributed Parameter Derivative Fill: Export");
    // Assemble global df/dp*V
    if (trans) {
      Teuchos::RCP<const Thyra_MultiVector> temp = fpV->clone_mv();

      distParamLib->get(dist_param_name)
          ->get_cas_manager()
          ->combine(overlapped_fpV, fpV, CombineMode::ADD);

      fpV->update(1.0, *temp);  // fpV += temp;

      std::stringstream sensitivity_name;
      sensitivity_name << dist_param_name << "_sensitivity";
      if (distParamLib->has(sensitivity_name.str())) {
        auto sens_vec = distParamLib->get(sensitivity_name.str())->vector();
        sens_vec->update(1.0, *fpV->col(0));
        distParamLib->get(sensitivity_name.str())->scatter();
      }
    } else {
      cas_manager->combine(overlapped_fpV, fpV, CombineMode::ADD);
    }
  }  // End timer

  // Apply Dirichlet conditions using dfm (Dirchelt Field Manager)
  if (!trans && dfm != Teuchos::null) {
    PHAL::Workset workset;

    workset.dist_param_deriv_name      = dist_param_name;
    workset.fpV                        = fpV;
    workset.Vp                         = V;
    workset.transpose_dist_param_deriv = trans;
    workset.disc                       = disc;

    double const this_time = fixTime(current_time);

    workset.current_time = this_time;

    dfm_set(workset, x, xdot, xdotdot);

    loadWorksetNodesetInfo(workset);

    // FillType template argument used to specialize Sacado
    dfm->evaluateFields<EvalT>(workset);
  }
}

void
Application::evaluateResponse(
    int                                     response_index,
    const double                            current_time,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
    const Teuchos::Array<ParamVec>&         p,
    const Teuchos::RCP<Thyra_Vector>&       g)
{
  TEUCHOS_FUNC_TIME_MONITOR(
      "Albany Fill: Response");
  double const this_time = fixTime(current_time);
  responses[response_index]->evaluateResponse(
      this_time, x, xdot, xdotdot, p, g);
}

void
Application::evaluateResponseTangent(
    int                                          response_index,
    int                                          parameter_index,
    const double                                 alpha,
    const double                                 beta,
    const double                                 omega,
    const double                                 current_time,
    bool                                         sum_derivs,
    const Teuchos::RCP<const Thyra_Vector>&      x,
    const Teuchos::RCP<const Thyra_Vector>&      xdot,
    const Teuchos::RCP<const Thyra_Vector>&      xdotdot,
    Teuchos::Array<ParamVec>&                    p,
    const Teuchos::RCP<const Thyra_MultiVector>& Vx,
    const Teuchos::RCP<const Thyra_MultiVector>& Vxdot,
    const Teuchos::RCP<const Thyra_MultiVector>& Vxdotdot,
    const Teuchos::RCP<const Thyra_MultiVector>& Vp,
    const Teuchos::RCP<Thyra_Vector>&            g,
    const Teuchos::RCP<Thyra_MultiVector>&       gx,
    const Teuchos::RCP<Thyra_MultiVector>&       gp)
{
  TEUCHOS_FUNC_TIME_MONITOR(
      "Albany Fill: Response Tangent");
  double const this_time = fixTime(current_time);
  responses[response_index]->evaluateTangent(
      alpha,
      beta,
      omega,
      this_time,
      sum_derivs,
      x,
      xdot,
      xdotdot,
      p,
      parameter_index,
      Vx,
      Vxdot,
      Vxdotdot,
      Vp,
      g,
      gx,
      gp);
}

void
Application::evaluateResponseDerivative(
    int                                              response_index,
    const double                                     current_time,
    const Teuchos::RCP<const Thyra_Vector>&          x,
    const Teuchos::RCP<const Thyra_Vector>&          xdot,
    const Teuchos::RCP<const Thyra_Vector>&          xdotdot,
    const Teuchos::Array<ParamVec>&                  p,
    const int                                        parameter_index,
    const Teuchos::RCP<Thyra_Vector>&                g,
    const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dx,
    const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dxdot,
    const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dxdotdot,
    const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dp)
{
  TEUCHOS_FUNC_TIME_MONITOR(
      "Albany Fill: Response Derivative");
  double const this_time = fixTime(current_time);

  responses[response_index]->evaluateDerivative(
      this_time,
      x,
      xdot,
      xdotdot,
      p,
      parameter_index,
      g,
      dg_dx,
      dg_dxdot,
      dg_dxdotdot,
      dg_dp);
}

void
Application::evaluateResponseDistParamDeriv(
    int                                     response_index,
    const double                            current_time,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
    const Teuchos::Array<ParamVec>&         param_array,
    const std::string&                      dist_param_name,
    const Teuchos::RCP<Thyra_MultiVector>&  dg_dp)
{
  TEUCHOS_FUNC_TIME_MONITOR(
      "Albany Fill: Response Distributed Parameter Derivative");
  double const this_time = fixTime(current_time);

  responses[response_index]->evaluateDistParamDeriv(
      this_time, x, xdot, xdotdot, param_array, dist_param_name, dg_dp);

  if (!dg_dp.is_null()) {
    std::stringstream sensitivity_name;
    sensitivity_name << dist_param_name << "_sensitivity";
    // TODO: make distParamLib Thyra
    if (distParamLib->has(sensitivity_name.str())) {
      auto sensitivity_vec =
          distParamLib->get(sensitivity_name.str())->vector();
      // FIXME This is not correct if the part of sensitivity due to the
      // Lagrange multiplier (fpV) is computed first.
      scale_and_update(sensitivity_vec, 0.0, dg_dp->col(0), 1.0);
      distParamLib->get(sensitivity_name.str())->scatter();
    }
  }
}

void
Application::evaluateResponse_HessVecProd_xx(
    int                                     response_index,
    const double                            current_time,
    const Teuchos::RCP<const Thyra_MultiVector>& v,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
    const Teuchos::Array<ParamVec>&         param_array,
    const Teuchos::RCP<Thyra_MultiVector>&  Hv_g_xx)
{
  TEUCHOS_FUNC_TIME_MONITOR(
      "Albany Fill: Response Distributed Parameter Hessian Vector Product");
  double const this_time = fixTime(current_time);

  using EvalT = PHAL::AlbanyTraits::HessianVec;
  postRegSetup<EvalT>();

  responses[response_index]->evaluate_HessVecProd_xx(
      this_time, v, x, xdot, xdotdot, param_array, Hv_g_xx);

  if (!Hv_g_xx.is_null()) {
    std::stringstream hessianvectorproduct_name;
    hessianvectorproduct_name << "Hv_g_xx";
  }
}

void
Application::evaluateResponse_HessVecProd_xp(
    int                                     response_index,
    const double                            current_time,
    const Teuchos::RCP<const Thyra_MultiVector>& v,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
    const Teuchos::Array<ParamVec>&         param_array,
    const std::string&                      param_direction_name,
    const Teuchos::RCP<Thyra_MultiVector>&  Hv_g_xp)
{
  TEUCHOS_FUNC_TIME_MONITOR(
      "Albany Fill: Response Parameter Hessian Vector Product");
  double const this_time = fixTime(current_time);

  using EvalT = PHAL::AlbanyTraits::HessianVec;
  postRegSetup<EvalT>();

  responses[response_index]->evaluate_HessVecProd_xp(
      this_time, v, x, xdot, xdotdot, param_array, param_direction_name, Hv_g_xp);

  if (!Hv_g_xp.is_null()) {
    std::stringstream hessianvectorproduct_name;
    hessianvectorproduct_name << param_direction_name << "_Hv_g_xp";
  }
}

void
Application::evaluateResponse_HessVecProd_px(
    int                                     response_index,
    const double                            current_time,
    const Teuchos::RCP<const Thyra_MultiVector>& v,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
    const Teuchos::Array<ParamVec>&         param_array,
    const std::string&                      param_name,
    const Teuchos::RCP<Thyra_MultiVector>&  Hv_g_px)
{
  TEUCHOS_FUNC_TIME_MONITOR(
      "Albany Fill: Response Parameter Hessian Vector Product");
  double const this_time = fixTime(current_time);

  using EvalT = PHAL::AlbanyTraits::HessianVec;
  postRegSetup<EvalT>();

  responses[response_index]->evaluate_HessVecProd_px(
      this_time, v, x, xdot, xdotdot, param_array, param_name, Hv_g_px);

  if (!Hv_g_px.is_null()) {
    std::stringstream hessianvectorproduct_name;
    hessianvectorproduct_name << param_name << "_Hv_g_px";
  }
}

void
Application::evaluateResponse_HessVecProd_pp(
    int                                     response_index,
    const double                            current_time,
    const Teuchos::RCP<const Thyra_MultiVector>& v,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
    const Teuchos::Array<ParamVec>&         param_array,
    const std::string&                      param_name,
    const std::string&                      param_direction_name,
    const Teuchos::RCP<Thyra_MultiVector>&  Hv_g_pp)
{
  TEUCHOS_FUNC_TIME_MONITOR(
      "Albany Fill: Response Parameter Hessian Vector Product");
  double const this_time = fixTime(current_time);

  using EvalT = PHAL::AlbanyTraits::HessianVec;
  postRegSetup<EvalT>();

  responses[response_index]->evaluate_HessVecProd_pp(
      this_time, v, x, xdot, xdotdot, param_array, param_name, param_direction_name, Hv_g_pp);

  if (!Hv_g_pp.is_null()) {
    std::stringstream hessianvectorproduct_name;
    hessianvectorproduct_name << param_name << "_" << param_direction_name << "_Hv_g_pp";
  }
}

void
Application::evaluateResponseHessian_pp(
    int                                     response_index,
    int                                     parameter_index,
    const double                            current_time,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
    const Teuchos::Array<ParamVec>&         param_array,
    const std::string&                      param_name,
    const Teuchos::RCP<Thyra_LinearOp>&     H)
{
  int l1;
  bool l1_is_distributed;
  Albany::getParameterVectorID(l1, l1_is_distributed, param_name);

  // Get the crs Hessian:
  auto Ht = Albany::getTpetraMatrix(H);

  auto hessianResponseParams = problemParams->sublist("Hessian").sublist(util::strint("Response", response_index));
  bool replace_by_I = hessianResponseParams.sublist(util::strint("Parameter", parameter_index)).get("Replace H_pp with Identity", false);

  if (replace_by_I) {
    auto range= H->range();
    int numElements = getLocalSubdim(range);

    beginModify(H);
    Teuchos::Array<ST> val(1,1.0);
    Teuchos::Array<LO> col(1);
    for (int i=0; i<numElements; ++i) {
      col[0] = i;
      setLocalRowValues(H,i,col,val);
    }
    endModify(H);
  } else if (l1_is_distributed) {
    Teuchos::ParameterList coloring_params;
    std::string matrixType = "Hessian";
    coloring_params.set("matrixType", matrixType);
    coloring_params.set("symmetric", true);

    // Create a colorer
    Zoltan2::TpetraCrsColorer<Tpetra_CrsMatrix> colorer(Ht);

    colorer.computeColoring(coloring_params);

    // Compute seed matrix V -- this matrix is a dense
    // matrix of 0/1 indicating the compression via coloring

    const int numColors = colorer.getNumColors();

    // Compute the seed matrix
    RCP<Tpetra_MultiVector> V = rcp(new Tpetra_MultiVector(Ht->getDomainMap(), numColors));
    colorer.computeSeedMatrix(*V);

    // Apply the Hessian to all the directions
    RCP<Tpetra_MultiVector> HV = rcp(new Tpetra_MultiVector(Ht->getDomainMap(), numColors));

    for (int i = 0; i < numColors; ++i) {
      RCP<const Thyra_MultiVector> v_i = Albany::createConstThyraMultiVector(V->getVector(i));
      RCP<Thyra_MultiVector> Hv_i = Albany::createThyraMultiVector(HV->getVectorNonConst(i));
      evaluateResponse_HessVecProd_pp(
        response_index,
        current_time,
        v_i,
        x,
        xdot,
        xdotdot,
        param_array,
        param_name,
        param_name,
        Hv_i);
    }

    // Reconstruct the Hessian matrix based on the Hessian-vector products
    colorer.reconstructMatrix(*HV, *Ht);

  } else {
    beginFEAssembly(H);
    int numColors = Ht->getDomainMap()->getLocalNumElements();
    RCP<Tpetra_Vector> v = rcp(new Tpetra_Vector(Ht->getDomainMap()));
    RCP<Tpetra_Vector> Hv = rcp(new Tpetra_Vector(Ht->getDomainMap()));

    ST values[1];
    GO cols[1];

    for (int i = 0; i < numColors; ++i) {
      v->replaceGlobalValue (i, 1.);
      if (i>0)
        v->replaceGlobalValue (i-1, 0.);

      RCP<const Thyra_MultiVector> v_i = Albany::createConstThyraMultiVector(v);
      RCP<Thyra_MultiVector> Hv_i = Albany::createThyraMultiVector(Hv);
      evaluateResponse_HessVecProd_pp(
        response_index,
        current_time,
        v_i,
        x,
        xdot,
        xdotdot,
        param_array,
        param_name,
        param_name,
        Hv_i);
      auto Hv_data = Hv->getData();
      for (int j = 0; j < numColors; ++j) {
        values[0] = Hv_data[j];
        cols[0] = i;
        Ht->replaceGlobalValues(j, 1, values, cols);
      }
    }
    endFEAssembly(H);
  }

#ifdef WRITE_TO_MATRIX_MARKET
  writeMatrixMarket(Ht.getConst(), "H", parameter_index);
#endif
}

void
Application::evaluateResidual_HessVecProd_xx(
    const double                            current_time,
    const Teuchos::RCP<const Thyra_MultiVector>& v,
    const Teuchos::RCP<const Thyra_Vector>& z,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
    const Teuchos::Array<ParamVec>&         param_array,
    const Teuchos::RCP<Thyra_MultiVector>&  Hv_f_xx)
{
  TEUCHOS_FUNC_TIME_MONITOR(
      "Albany Fill: Residual Distributed Parameter Hessian Vector Product");

  using EvalT = PHAL::AlbanyTraits::HessianVec;
  postRegSetup<EvalT>();

  // Set data in Workset struct
  PHAL::Workset workset;
  setupBasicWorksetInfo(workset, current_time, x, xdot, xdotdot, param_array);

  if(!v.is_null()) {
    workset.hessianWorkset.direction_x = Thyra::createMembers(workset.x_cas_manager->getOverlappedVectorSpace(),v->domain()->dim());
    workset.x_cas_manager->scatter(v->clone_mv(), workset.hessianWorkset.direction_x, Albany::CombineMode::INSERT);
  }

  if(!z.is_null()) {
    workset.hessianWorkset.f_multiplier = z->clone_v();
  }

  if(!Hv_f_xx.is_null()) {
    workset.j_coeff = 1.0;
    workset.hessianWorkset.hess_vec_prod_f_xx = Hv_f_xx;
    workset.hessianWorkset.overlapped_hess_vec_prod_f_xx = Thyra::createMembers(workset.x_cas_manager->getOverlappedVectorSpace(),Hv_f_xx->domain()->dim());
    workset.hessianWorkset.overlapped_hess_vec_prod_f_xx->assign(0.0);

    const auto& wsPhysIndex  = disc->getWsPhysIndex();

    if (dfm != Teuchos::null) {
      loadWorksetNodesetInfo(workset);

      dfm->preEvaluate<EvalT>(workset);
    }

    if(!z.is_null()) {
      workset.hessianWorkset.overlapped_f_multiplier = Thyra::createMember(workset.x_cas_manager->getOverlappedVectorSpace());
      workset.x_cas_manager->scatter(workset.hessianWorkset.f_multiplier, workset.hessianWorkset.overlapped_f_multiplier, Albany::CombineMode::INSERT);
    }

    const int numWorksets = disc->getNumWorksets();

    for (int ws = 0; ws < numWorksets; ws++) {
      const std::string evalName = PHAL::evalName<EvalT>("FM", wsPhysIndex[ws]);
      loadWorksetBucketInfo(workset, ws, evalName);

      // FillType template argument used to specialize Sacado
      fm[wsPhysIndex[ws]]->evaluateFields<EvalT>(workset);
    }

    if (dfm != Teuchos::null) {
      dfm->evaluateFields<EvalT>(workset);
    }

    workset.x_cas_manager->combine(workset.hessianWorkset.overlapped_hess_vec_prod_f_xx, workset.hessianWorkset.hess_vec_prod_f_xx, Albany::CombineMode::ADD);

    std::stringstream hessianvectorproduct_name;
    hessianvectorproduct_name << "Hv_f_xx";
  }
}

void
Application::evaluateResidual_HessVecProd_xp(
    const double                            current_time,
    const Teuchos::RCP<const Thyra_MultiVector>& v,
    const Teuchos::RCP<const Thyra_Vector>& z,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
    const Teuchos::Array<ParamVec>&         param_array,
    const std::string&                      param_direction_name,
    const Teuchos::RCP<Thyra_MultiVector>&  Hv_f_xp)
{
  TEUCHOS_FUNC_TIME_MONITOR(
      "Albany Fill: Response Distributed Parameter Hessian Vector Product");

  using EvalT = PHAL::AlbanyTraits::HessianVec;
  postRegSetup<EvalT>();

  // First, the function checks whether the parameter associated to param_direction_name
  // is a distributed parameter (l2_is_distributed==true) or a parameter vector
  // (l2_is_distributed==false).
  int l2;
  bool l2_is_distributed;
  Albany::getParameterVectorID(l2, l2_is_distributed, param_direction_name);

  // Set data in Workset struct
  PHAL::Workset workset;
  setupBasicWorksetInfo(workset, current_time, x, xdot, xdotdot, param_array);

  // If the parameter associated to param_direction_name is a parameter vector, 
  // the initialization of the second derivatives must be performed now:
  if (!l2_is_distributed) {
    ParamVec params_l2 = param_array[l2];
    unsigned int num_cols_p_l2 = params_l2.size();

    Teuchos::ArrayRCP<const ST> v_constView;
    if(!v.is_null()) {
      v_constView = Albany::getLocalData(v->col(0));
    }

    HessianVecFad p_val;
    for (unsigned int i = 0; i < num_cols_p_l2; i++) {
      p_val = params_l2[i].family->getValue<EvalT>();
      p_val.val().fastAccessDx(0) = v_constView[i];
      params_l2[i].family->setValue<EvalT>(p_val);
    }
  }

  // If the parameter associated to param_direction_name is a distributed parameter,
  // the direction vectors should be scattered to have overlapped directions:
  if(l2_is_distributed && !v.is_null()) {
    workset.hessianWorkset.p_direction_cas_manager = workset.distParamLib->get(param_direction_name)->get_cas_manager();
    workset.hessianWorkset.direction_p = Thyra::createMembers(workset.hessianWorkset.p_direction_cas_manager->getOverlappedVectorSpace(),v->domain()->dim());
    workset.hessianWorkset.p_direction_cas_manager->scatter(v->clone_mv(), workset.hessianWorkset.direction_p, Albany::CombineMode::INSERT);
  }

  if(!z.is_null()) {
    workset.hessianWorkset.f_multiplier = z->clone_v();
  }

  if(!Hv_f_xp.is_null()) {
    workset.j_coeff = 1.0;
    workset.hessianWorkset.dist_param_deriv_direction_name = param_direction_name;
    workset.hessianWorkset.hess_vec_prod_f_xp = Hv_f_xp;
    workset.hessianWorkset.overlapped_hess_vec_prod_f_xp = Thyra::createMembers(workset.x_cas_manager->getOverlappedVectorSpace(),Hv_f_xp->domain()->dim());
    workset.hessianWorkset.overlapped_hess_vec_prod_f_xp->assign(0.0);

    const auto& wsPhysIndex  = disc->getWsPhysIndex();

    if (dfm != Teuchos::null) {
      loadWorksetNodesetInfo(workset);

      dfm->preEvaluate<EvalT>(workset);
    }

    if(!z.is_null()) {
      workset.hessianWorkset.overlapped_f_multiplier = Thyra::createMember(workset.x_cas_manager->getOverlappedVectorSpace());
      workset.x_cas_manager->scatter(workset.hessianWorkset.f_multiplier, workset.hessianWorkset.overlapped_f_multiplier, Albany::CombineMode::INSERT);
    }

    const int numWorksets = disc->getNumWorksets();

    for (int ws = 0; ws < numWorksets; ws++) {
      const std::string evalName = PHAL::evalName<EvalT>("FM", wsPhysIndex[ws]);
      loadWorksetBucketInfo(workset, ws, evalName);

      // FillType template argument used to specialize Sacado
      fm[wsPhysIndex[ws]]->evaluateFields<EvalT>(workset);
    }

    if (dfm != Teuchos::null) {
      dfm->evaluateFields<EvalT>(workset);
    }

    workset.x_cas_manager->combine(workset.hessianWorkset.overlapped_hess_vec_prod_f_xp, workset.hessianWorkset.hess_vec_prod_f_xp, Albany::CombineMode::ADD);

    std::stringstream hessianvectorproduct_name;
    hessianvectorproduct_name << "Hv_f_xp";
  }
}

void
Application::evaluateResidual_HessVecProd_px(
    const double                            current_time,
    const Teuchos::RCP<const Thyra_MultiVector>& v,
    const Teuchos::RCP<const Thyra_Vector>& z,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
    const Teuchos::Array<ParamVec>&         param_array,
    const std::string&                      param_name,
    const Teuchos::RCP<Thyra_MultiVector>&  Hv_f_px)
{
  TEUCHOS_FUNC_TIME_MONITOR(
      "Albany Fill: Response Distributed Parameter Hessian Vector Product");

  using EvalT = PHAL::AlbanyTraits::HessianVec;
  postRegSetup<EvalT>();

  // First, the function checks whether the parameter associated to param_name
  // is a distributed parameter (l1_is_distributed==true) or a parameter vector
  // (l1_is_distributed==false).
  int l1;
  bool l1_is_distributed;
  Albany::getParameterVectorID(l1, l1_is_distributed, param_name);

  // Set data in Workset struct
  PHAL::Workset workset;
  setupBasicWorksetInfo(workset, current_time, x, xdot, xdotdot, param_array);

  // If the parameter associated to param_name is a parameter vector, 
  // the initialization of the first derivatives must be performed now:
  if (!l1_is_distributed) {
    ParamVec params_l1 = param_array[l1];
    int num_cols_p_l1 = params_l1.size();

    // Assumes that there is only one element block
    int deriv_size = PHAL::getDerivativeDimensions<EvalT>(this, 0);
    TEUCHOS_TEST_FOR_EXCEPTION(
        num_cols_p_l1 > deriv_size,
        std::runtime_error,
            "\nError in Albany::Application::evaluateResidual_HessVecProd_px  "
            << "Number of parameters columns cannot be larger than the derivative size\n ");

    HessianVecFad p_val;
    for (int i = 0; i < num_cols_p_l1; i++) {
      p_val = HessianVecFad(deriv_size, params_l1[i].baseValue);
      p_val.fastAccessDx(i).val() = 1.0;
      params_l1[i].family->setValue<EvalT>(p_val);
    }
  }

  if(!v.is_null()) {
    workset.hessianWorkset.direction_x = Thyra::createMembers(workset.x_cas_manager->getOverlappedVectorSpace(),v->domain()->dim());
    workset.x_cas_manager->scatter(v->clone_mv(), workset.hessianWorkset.direction_x, Albany::CombineMode::INSERT);
  }

  if(!z.is_null()) {
    workset.hessianWorkset.f_multiplier = z->clone_v();
  }

  if(!Hv_f_px.is_null()) {
    workset.j_coeff = 1.0;
    workset.dist_param_deriv_name = param_name;
    if (l1_is_distributed) {
      workset.p_cas_manager = workset.distParamLib->get(param_name)->get_cas_manager();
      workset.hessianWorkset.overlapped_hess_vec_prod_f_px = Thyra::createMembers(workset.p_cas_manager->getOverlappedVectorSpace(),Hv_f_px->domain()->dim());
    }
    else {
      auto overlapped = Hv_f_px->col(0)->space();

      int n_local_params = 0;
      int n_total_params = overlapped->dim();

      if (workset.comm->getRank()==0)
        n_local_params = n_total_params;
      std::vector<GO> my_gids;
      for (int i=0; i<n_local_params; ++i)
        my_gids.push_back(i);
      Teuchos::ArrayView<GO> gids(my_gids);

      auto owned = Albany::createVectorSpace(workset.comm, gids, n_total_params);
      workset.p_cas_manager = createCombineAndScatterManager(owned, overlapped);
      workset.hessianWorkset.overlapped_hess_vec_prod_f_px = 
        Thyra::createMembers(workset.p_cas_manager->getOverlappedVectorSpace(), Hv_f_px->domain()->dim());
    }
    workset.hessianWorkset.hess_vec_prod_f_px = Hv_f_px;
    workset.hessianWorkset.overlapped_hess_vec_prod_f_px->assign(0.0);

    const auto& wsPhysIndex  = disc->getWsPhysIndex();

    if (dfm != Teuchos::null) {
      loadWorksetNodesetInfo(workset);

      dfm->preEvaluate<EvalT>(workset);
    }

    if(!z.is_null()) {
      workset.hessianWorkset.overlapped_f_multiplier = Thyra::createMember(workset.x_cas_manager->getOverlappedVectorSpace());
      workset.x_cas_manager->scatter(workset.hessianWorkset.f_multiplier, workset.hessianWorkset.overlapped_f_multiplier, Albany::CombineMode::INSERT);
    }

    const int numWorksets = disc->getNumWorksets();

    for (int ws = 0; ws < numWorksets; ws++) {
      const std::string evalName = PHAL::evalName<EvalT>("FM", wsPhysIndex[ws]);
      loadWorksetBucketInfo(workset, ws, evalName);

      // FillType template argument used to specialize Sacado
      fm[wsPhysIndex[ws]]->evaluateFields<EvalT>(workset);
    }

    if (dfm != Teuchos::null) {
      dfm->evaluateFields<EvalT>(workset);
    }

    if (l1_is_distributed)
      workset.p_cas_manager->combine(workset.hessianWorkset.overlapped_hess_vec_prod_f_px, workset.hessianWorkset.hess_vec_prod_f_px, Albany::CombineMode::ADD);
    else {
      auto tmp = Thyra::createMembers(workset.p_cas_manager->getOwnedVectorSpace(), workset.hessianWorkset.overlapped_hess_vec_prod_f_px->domain()->dim());
      workset.p_cas_manager->combine(workset.hessianWorkset.overlapped_hess_vec_prod_f_px, tmp, Albany::CombineMode::ADD);
      workset.p_cas_manager->scatter(tmp, workset.hessianWorkset.hess_vec_prod_f_px, Albany::CombineMode::INSERT);
    }

    std::stringstream hessianvectorproduct_name;
    hessianvectorproduct_name << "Hv_f_px";
  }
}

void
Application::evaluateResidual_HessVecProd_pp(
    const double                            current_time,
    const Teuchos::RCP<const Thyra_MultiVector>& v,
    const Teuchos::RCP<const Thyra_Vector>& z,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
    const Teuchos::Array<ParamVec>&         param_array,
    const std::string&                      param_name,
    const std::string&                      param_direction_name,
    const Teuchos::RCP<Thyra_MultiVector>&  Hv_f_pp)
{
  TEUCHOS_FUNC_TIME_MONITOR(
      "Albany Fill: Response Distributed Parameter Hessian Vector Product");

  using EvalT = PHAL::AlbanyTraits::HessianVec;
  postRegSetup<EvalT>();

  // First, the function checks whether the parameter associated to param_name
  // is a distributed parameter (l1_is_distributed==true) or a parameter vector
  // (l1_is_distributed==false).
  int l1;
  bool l1_is_distributed;
  Albany::getParameterVectorID(l1, l1_is_distributed, param_name);

  // Then the function checks whether the parameter associated to param_direction_name
  // is a distributed parameter (l2_is_distributed==true) or a parameter vector
  // (l2_is_distributed==false).
  int l2;
  bool l2_is_distributed;
  Albany::getParameterVectorID(l2, l2_is_distributed, param_direction_name);

  // Set data in Workset struct
  PHAL::Workset workset;
  setupBasicWorksetInfo(workset, current_time, x, xdot, xdotdot, param_array);

  // If the parameter associated to param_name is a parameter vector, 
  // the initialization of the first derivatives must be performed now:
  if (!l1_is_distributed) {
    ParamVec params_l1 = param_array[l1];
    int num_cols_p_l1 = params_l1.size();

    // Assumes that there is only one element block
    int deriv_size = PHAL::getDerivativeDimensions<EvalT>(this, 0);
    TEUCHOS_TEST_FOR_EXCEPTION(
        num_cols_p_l1 > deriv_size,
        std::runtime_error,
            "\nError in Albany::Application::evaluateResidual_HessVecProd_pp  "
            << "Number of parameters columns cannot be larger than the derivative size\n ");

    HessianVecFad p_val;
    for (int i = 0; i < num_cols_p_l1; i++) {
      p_val = HessianVecFad(deriv_size, params_l1[i].baseValue);
      p_val.fastAccessDx(i).val() = 1.0;
      params_l1[i].family->setValue<EvalT>(p_val);
    }
  }

  // If the parameter associated to param_direction_name is a parameter vector, 
  // the initialization of the second derivatives must be performed now:
  if (!l2_is_distributed) {
    ParamVec params_l2 = param_array[l2];
    int num_cols_p_l2 = params_l2.size();

    Teuchos::ArrayRCP<const ST> v_constView;
    if(!v.is_null()) {
      v_constView = Albany::getLocalData(v->col(0));
    }

    HessianVecFad p_val;
    for (int i = 0; i < num_cols_p_l2; i++) {
      p_val = params_l2[i].family->getValue<EvalT>();
      p_val.val().fastAccessDx(0) = v_constView[i];
      params_l2[i].family->setValue<EvalT>(p_val);
    }
  }

  // If the parameter associated to param_direction_name is a distributed parameter,
  // the direction vectors should be scattered to have overlapped directions:
  if(l2_is_distributed && !v.is_null()) {
    workset.hessianWorkset.p_direction_cas_manager = workset.distParamLib->get(param_direction_name)->get_cas_manager();
    workset.hessianWorkset.direction_p = Thyra::createMembers(workset.hessianWorkset.p_direction_cas_manager->getOverlappedVectorSpace(),v->domain()->dim());
    workset.hessianWorkset.p_direction_cas_manager->scatter(v->clone_mv(), workset.hessianWorkset.direction_p, Albany::CombineMode::INSERT);
  }

  if(!z.is_null()) {
    workset.hessianWorkset.f_multiplier = z->clone_v();
  }

  if(!Hv_f_pp.is_null()) {
    workset.dist_param_deriv_name = param_name;
    workset.hessianWorkset.dist_param_deriv_direction_name = param_direction_name;
    if (l1_is_distributed) {
      workset.p_cas_manager = workset.distParamLib->get(param_name)->get_cas_manager();
      workset.hessianWorkset.overlapped_hess_vec_prod_f_pp = Thyra::createMembers(workset.p_cas_manager->getOverlappedVectorSpace(),Hv_f_pp->domain()->dim());
    }
    else {
      auto overlapped = Hv_f_pp->col(0)->space();

      int n_local_params = 0;
      int n_total_params = overlapped->dim();

      if (workset.comm->getRank()==0)
        n_local_params = n_total_params;
      std::vector<GO> my_gids;
      for (int i=0; i<n_local_params; ++i)
        my_gids.push_back(i);
      Teuchos::ArrayView<GO> gids(my_gids);

      auto owned = Albany::createVectorSpace(workset.comm, gids, n_total_params);
      workset.p_cas_manager = createCombineAndScatterManager(owned, overlapped);
      workset.hessianWorkset.overlapped_hess_vec_prod_f_pp = 
        Thyra::createMembers(workset.p_cas_manager->getOverlappedVectorSpace(), Hv_f_pp->domain()->dim());
    }

    workset.hessianWorkset.hess_vec_prod_f_pp = Hv_f_pp;
    workset.hessianWorkset.overlapped_hess_vec_prod_f_pp->assign(0.0);

    const auto& wsPhysIndex  = disc->getWsPhysIndex();

    if (dfm != Teuchos::null) {
      loadWorksetNodesetInfo(workset);

      dfm->preEvaluate<EvalT>(workset);
    }

    if(!z.is_null()) {
      workset.hessianWorkset.overlapped_f_multiplier = Thyra::createMember(workset.x_cas_manager->getOverlappedVectorSpace());
      workset.x_cas_manager->scatter(workset.hessianWorkset.f_multiplier, workset.hessianWorkset.overlapped_f_multiplier, Albany::CombineMode::INSERT);
    }

    const int numWorksets = disc->getNumWorksets();

    for (int ws = 0; ws < numWorksets; ws++) {
      const std::string evalName = PHAL::evalName<EvalT>("FM", wsPhysIndex[ws]);
      loadWorksetBucketInfo(workset, ws, evalName);

      // FillType template argument used to specialize Sacado
      fm[wsPhysIndex[ws]]->evaluateFields<EvalT>(workset);
    }

    if (dfm != Teuchos::null) {
      dfm->evaluateFields<EvalT>(workset);
    }

    if (l1_is_distributed)
      workset.p_cas_manager->combine(workset.hessianWorkset.overlapped_hess_vec_prod_f_pp, workset.hessianWorkset.hess_vec_prod_f_pp, Albany::CombineMode::ADD);
    else {
      auto tmp = Thyra::createMembers(workset.p_cas_manager->getOwnedVectorSpace(), workset.hessianWorkset.overlapped_hess_vec_prod_f_pp->domain()->dim());
      workset.p_cas_manager->combine(workset.hessianWorkset.overlapped_hess_vec_prod_f_pp, tmp, Albany::CombineMode::ADD);
      workset.p_cas_manager->scatter(tmp, workset.hessianWorkset.hess_vec_prod_f_pp, Albany::CombineMode::INSERT);
    }

    std::stringstream hessianvectorproduct_name;
    hessianvectorproduct_name << "Hv_f_pp";
  }
}

void
Application::evaluateStateFieldManager(
    const double             current_time,
    const Thyra_MultiVector& x,
    Teuchos::Ptr<const Thyra_MultiVector> dxdp)
{
  int num_vecs = x.domain()->dim();

  if (num_vecs == 1) {
    this->evaluateStateFieldManager(
        current_time, *x.col(0), Teuchos::null, Teuchos::null, dxdp);
  } else if (num_vecs == 2) {
    this->evaluateStateFieldManager(
        current_time, *x.col(0), x.col(1).ptr(), Teuchos::null, dxdp);
  } else {
    this->evaluateStateFieldManager(
        current_time, *x.col(0), x.col(1).ptr(), x.col(2).ptr(), dxdp);
  }
}

void
Application::evaluateStateFieldManager(
    const double                     current_time,
    const Thyra_Vector&              x,
    Teuchos::Ptr<const Thyra_Vector> xdot,
    Teuchos::Ptr<const Thyra_Vector> xdotdot,
    Teuchos::Ptr<const Thyra_MultiVector> dxdp )
{
  TEUCHOS_FUNC_TIME_MONITOR("Albany Fill: State Residual");
  {
    std::string evalName = PHAL::evalName<PHAL::AlbanyTraits::Residual>("SFM",0);
    if (!phxSetup->contain_eval(evalName)) {
      for (int ps = 0; ps < sfm.size(); ++ps) {
        evalName = PHAL::evalName<PHAL::AlbanyTraits::Residual>("SFM",ps);
        phxSetup->insert_eval(evalName);

        std::vector<PHX::index_size_type> derivative_dimensions;
        derivative_dimensions.push_back(
            PHAL::getDerivativeDimensions<PHAL::AlbanyTraits::Jacobian>(
                this, ps));
        sfm[ps]
            ->setKokkosExtendedDataTypeDimensions<PHAL::AlbanyTraits::Jacobian>(
                derivative_dimensions);
        setDynamicLayoutSizes<PHAL::AlbanyTraits::Residual>(sfm[ps]);
        sfm[ps]->postRegistrationSetup(*phxSetup);

        // Update phalanx saved/unsaved fields based on field dependencies
        phxSetup->check_fields(
            sfm[ps]->getFieldTagsForSizing<PHAL::AlbanyTraits::Residual>());
        phxSetup->update_fields();

        writePhalanxGraph<PHAL::AlbanyTraits::Residual>(sfm[ps], evalName,
            stateGraphVisDetail);
      }
    }
  }

  Teuchos::RCP<Thyra_Vector> overlapped_f = solMgr->get_overlapped_f();

  // Load connectivity map and coordinates
  const auto& wsPhysIndex  = disc->getWsPhysIndex();

  const int numWorksets = disc->getNumWorksets();

  // Scatter to the overlapped distribution
  solMgr->scatterX(x, xdot, xdotdot, dxdp);

  // Scatter distributed parameters
  distParamLib->scatter();

  // Set data in Workset struct
  PHAL::Workset workset;
  loadBasicWorksetInfo(workset, current_time);
  workset.f = overlapped_f;

  // Perform fill via field manager
  for (int ws = 0; ws < numWorksets; ws++) {
    const std::string evalName = PHAL::evalName<PHAL::AlbanyTraits::Residual>(
        "SFM", wsPhysIndex[ws]);
    loadWorksetBucketInfo(workset, ws, evalName);
    sfm[wsPhysIndex[ws]]->evaluateFields<PHAL::AlbanyTraits::Residual>(workset);
  }
}

void
Application::determinePiroSolver(
    const Teuchos::RCP<Teuchos::ParameterList>& topLevelParams)
{
  const Teuchos::RCP<Teuchos::ParameterList>& localProblemParams =
      Teuchos::sublist(topLevelParams, "Problem", true);

  const Teuchos::RCP<Teuchos::ParameterList>& piroParams =
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
      piroSolverToken = (secondOrder == "No") ? "Tempus" : secondOrder;
    } else {
      // Piro cannot handle the corresponding problem
      piroSolverToken = "Unsupported";
    }

    piroParams->set("Solver Type", piroSolverToken);
  }
}

void
Application::loadWorksetBucketInfo(PHAL::Workset& workset, const int& ws,
    const std::string& evalName)
{
  auto const& coords             = disc->getCoords();
  auto const& wsEBNames          = disc->getWsEBNames();

  workset.numCells             = disc->getWorksetsSizes()[ws];
  workset.wsCoords             = coords[ws];
  workset.EBName               = wsEBNames[ws];
  workset.wsIndex              = ws;


  workset.savedMDFields = phxSetup->get_saved_fields(evalName);

  // Sidesets are integrated within the Cells
  loadWorksetSidesetInfo(workset, ws);

  workset.stateArrayPtr = &disc->getStateArrays(StateStruct::ElemState)[ws];
}

void
Application::loadBasicWorksetInfo(PHAL::Workset& workset, double current_time)
{
  auto overlapped_MV = solMgr->getOverlappedSolution();
  auto numVectors    = overlapped_MV->domain()->dim();

  workset.x       = overlapped_MV->col(0);
  workset.xdot    = numVectors > 1 ? overlapped_MV->col(1) : Teuchos::null;
  workset.xdotdot = numVectors > 2 ? overlapped_MV->col(2) : Teuchos::null;

  workset.numEqs       = neq;
  workset.current_time = current_time;
  workset.distParamLib = distParamLib;
  workset.disc         = disc;
  // workset.delta_time = delta_time;
  workset.transientTerms    = Teuchos::nonnull(workset.xdot);
  workset.accelerationTerms = Teuchos::nonnull(workset.xdotdot);
}

void
Application::loadBasicWorksetInfoSDBCs(
    PHAL::Workset&                          workset,
    const Teuchos::RCP<const Thyra_Vector>& owned_sol,
    const double                            current_time)
{
  // Scatter owned solution into the overlapped one
  auto overlapped_MV  = solMgr->getOverlappedSolution();
  auto overlapped_sol = Thyra::createMember(overlapped_MV->range());
  overlapped_sol->assign(0.0);
  solMgr->get_cas_manager()->scatter(
      owned_sol, overlapped_sol, CombineMode::INSERT);

  auto numVectors = overlapped_MV->domain()->dim();
  workset.x       = overlapped_sol;
  workset.xdot    = numVectors > 1 ? overlapped_MV->col(1) : Teuchos::null;
  workset.xdotdot = numVectors > 2 ? overlapped_MV->col(2) : Teuchos::null;

  workset.numEqs       = neq;
  workset.current_time = current_time;
  workset.distParamLib = distParamLib;
  workset.disc         = disc;
  // workset.delta_time = delta_time;
  workset.transientTerms    = Teuchos::nonnull(workset.xdot);
  workset.accelerationTerms = Teuchos::nonnull(workset.xdotdot);
}

void
Application::loadWorksetJacobianInfo(
    PHAL::Workset& workset,
    const double   alpha,
    const double   beta,
    const double   omega)
{
  workset.m_coeff         = alpha;
  workset.n_coeff         = omega;
  workset.j_coeff         = beta;
  workset.ignore_residual = ignore_residual_in_jacobian;
}

void
Application::loadWorksetNodesetInfo(PHAL::Workset& workset)
{
  workset.nodeSets      = Teuchos::rcpFromRef(disc->getNodeSets());
  workset.nodeSetCoords = Teuchos::rcpFromRef(disc->getNodeSetCoords());
}

void
Application::setScale(Teuchos::RCP<const Thyra_LinearOp> jac)
{
  if (scaleBCdofs == true) {
    if (scaleVec_->norm_2() == 0.0) { scaleVec_->assign(1.0); }
    return;
  }

  if (scale_type == CONSTANT) {  // constant scaling
    scaleVec_->assign(1.0 / scale);
  } else if (scale_type == DIAG) {  // diagonal scaling
    if (jac == Teuchos::null) {
      scaleVec_->assign(1.0);
    } else {
      getDiagonalCopy(jac, scaleVec_);
      Thyra::reciprocal<ST>(*scaleVec_, scaleVec_.ptr());
    }
  } else if (scale_type == ABSROWSUM) {  // absolute value of row sum scaling
    if (jac == Teuchos::null) {
      scaleVec_->assign(1.0);
    } else {
      scaleVec_->assign(0.0);
      // We MUST be able to cast the linear op to RowStatLinearOpBase, in order
      // to get row information
      auto jac_row_stat =
          Teuchos::rcp_dynamic_cast<const Thyra::RowStatLinearOpBase<ST>>(
              jac, true);

      // Compute the inverse of the absolute row sum
      jac_row_stat->getRowStat(
          Thyra::RowStatLinearOpBaseUtils::ROW_STAT_INV_ROW_SUM,
          scaleVec_.ptr());
    }
  }
}

void
Application::setScaleBCDofs(
    PHAL::Workset&                     workset,
    Teuchos::RCP<const Thyra_LinearOp> jac)
{
  // First step: set scaleVec_ to all 1.0s if it is all 0s
  if (scaleVec_->norm_2() == 0) { scaleVec_->assign(1.0); }

  // If calling setScaleBCDofs with null Jacobian, don't recompute the scaling
  if (jac == Teuchos::null) { return; }

  // For diagonal or abs row sum scaling, set the scale equal to the maximum
  // magnitude value of the diagonal / abs row sum (inf-norm).  This way, scaling
  // adjusts throughout the simulation based on the Jacobian.
  Teuchos::RCP<Thyra_Vector> tmp = Thyra::createMember(scaleVec_->space());
  if (scale_type == DIAG) {
    getDiagonalCopy(jac, tmp);
    scale = tmp->norm_inf();
  } else if (scale_type == ABSROWSUM) {
    // We MUST be able to cast the linear op to RowStatLinearOpBase, in order to
    // get row information
    auto jac_row_stat =
        Teuchos::rcp_dynamic_cast<const Thyra::RowStatLinearOpBase<ST>>(
            jac, true);

    // Compute the absolute row sum
    jac_row_stat->getRowStat(
        Thyra::RowStatLinearOpBaseUtils::ROW_STAT_ROW_SUM, tmp.ptr());
    scale = tmp->norm_inf();
  }

  if (scale == 0.0) { scale = 1.0; }

  auto scaleVecLocalData = getNonconstLocalData(scaleVec_);
  for (size_t ns = 0; ns < nodeSetIDs_.size(); ns++) {
    std::string key = nodeSetIDs_[ns];
    const auto& ns_offsets = offsets_[ns];
    // std::cout << "IKTIKT key = " << key << std::endl;
    const auto& ns_node_elem_pos = workset.nodeSets->at(key);
    const auto& sol_dof_mgr   = disc->getDOFManager();
    const auto& sol_elem_dof_lids = sol_dof_mgr->elem_dof_lids().host();

    std::vector<std::vector<int>> sol_offsets(ns_offsets.size());
    for (unsigned j=0; j<ns_offsets.size(); ++j) {
      sol_offsets[j] = sol_dof_mgr->getGIDFieldOffsets(j);
    }
    for (const auto& ep : ns_node_elem_pos) {
      const int ielem = ep.first;
      const int pos   = ep.second;
      // std::cout << "IKTIKT ns, offsets size: " << ns << ", " <<
      // offsets_[ns].size() << "\n";
      for (unsigned j=0; j<ns_offsets.size(); ++j) {
        const int x_lid = sol_elem_dof_lids(ielem,sol_offsets[ns_offsets[j]][pos]);
        scaleVecLocalData[x_lid] = scale;
      }
    }
  }

  if (problem->getSideSetEquations().size() > 0) {
    TEUCHOS_TEST_FOR_EXCEPTION(
        true,
        std::logic_error,
        "Application::setScaleBCDofs is not yet implemented for"
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
        disc->getMeshStruct()->meshSpecs[0]->sideSetMeshNames;
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
            // << ", " << offsets_[l][j] << std::endl;  std::cout << "lunk = "
    <<
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

void
Application::loadWorksetSidesetInfo(PHAL::Workset& workset, const int ws)
{
  workset.sideSets = Teuchos::rcpFromRef(disc->getSideSets(ws));
  workset.sideSetViews = Teuchos::rcpFromRef(disc->getSideSetViews(ws));
  workset.localDOFViews = Teuchos::rcpFromRef(disc->getLocalDOFViews(ws));
}

void
Application::setupBasicWorksetInfo(
    PHAL::Workset&                          workset,
    double                                  current_time,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
    const Teuchos::Array<ParamVec>&         p)
{
  Teuchos::RCP<const Thyra_MultiVector> overlapped_MV =
      solMgr->getOverlappedSolution();
  auto numVectors = overlapped_MV->domain()->dim();

  Teuchos::RCP<const Thyra_Vector> overlapped_x = overlapped_MV->col(0);
  Teuchos::RCP<const Thyra_Vector> overlapped_xdot =
      numVectors > 1 ? overlapped_MV->col(1) : Teuchos::null;
  Teuchos::RCP<const Thyra_Vector> overlapped_xdotdot =
      numVectors > 2 ? overlapped_MV->col(2) : Teuchos::null;

  // Scatter xT and xdotT to the overlapped distribution
  solMgr->scatterX(*x, xdot.ptr(), xdotdot.ptr());

  // Scatter distributed parameters
  distParamLib->scatter();

  // Set parameters
  // We have to reset the parameters here to be sure to zero out the
  // previously used derivatives.
  for (int i = 0; i < p.size(); i++) {
    for (unsigned int j = 0; j < p[i].size(); j++) {
      p[i][j].family->setRealValueForAllTypes(p[i][j].baseValue);
    }
  }

  workset.x            = overlapped_x;
  workset.xdot         = overlapped_xdot;
  workset.xdotdot      = overlapped_xdotdot;
  workset.distParamLib = distParamLib;
  workset.disc         = disc;

  const double this_time = fixTime(current_time);

  workset.current_time = this_time;

  workset.transientTerms    = Teuchos::nonnull(workset.xdot);
  workset.accelerationTerms = Teuchos::nonnull(workset.xdotdot);

  workset.comm = comm;

  workset.x_cas_manager = solMgr->get_cas_manager();
}

int
Application::calcTangentDerivDimension(
    const Teuchos::RCP<Teuchos::ParameterList>& problemParams)
{
  int numScalarParameters(0);
  Albany::CalculateNumberParams(problemParams, &numScalarParameters);
  return numScalarParameters;
}

void
Application::setupTangentWorksetInfo(
    PHAL::Workset&                               workset,
    double                                       current_time,
    bool                                         sum_derivs,
    const Teuchos::RCP<const Thyra_Vector>&      x,
    const Teuchos::RCP<const Thyra_Vector>&      xdot,
    const Teuchos::RCP<const Thyra_Vector>&      xdotdot,
    Teuchos::Array<ParamVec>&                    p,
    const int                                    parameter_index,
    const Teuchos::RCP<const Thyra_MultiVector>& Vx,
    const Teuchos::RCP<const Thyra_MultiVector>& Vxdot,
    const Teuchos::RCP<const Thyra_MultiVector>& Vxdotdot,
    const Teuchos::RCP<const Thyra_MultiVector>& Vp)
{
  setupBasicWorksetInfo(workset, current_time, x, xdot, xdotdot, p);

  auto dof_mgr = disc->getDOFManager();

  // Scatter Vx dot the overlapped distribution
  RCP<Thyra_MultiVector> overlapped_Vx;
  if (Vx != Teuchos::null) {
    overlapped_Vx = Thyra::createMembers(
        dof_mgr->ov_vs(), Vx->domain()->dim());
    overlapped_Vx->assign(0.0);
    solMgr->get_cas_manager()->scatter(Vx, overlapped_Vx, CombineMode::INSERT);
  }

  // Scatter Vx dot the overlapped distribution
  RCP<Thyra_MultiVector> overlapped_Vxdot;
  if (Vxdot != Teuchos::null) {
    overlapped_Vxdot = Thyra::createMembers(
        dof_mgr->ov_vs(), Vxdot->domain()->dim());
    overlapped_Vxdot->assign(0.0);
    solMgr->get_cas_manager()->scatter(
        Vxdot, overlapped_Vxdot, CombineMode::INSERT);
  }
  RCP<Thyra_MultiVector> overlapped_Vxdotdot;
  if (Vxdotdot != Teuchos::null) {
    overlapped_Vxdotdot = Thyra::createMembers(
        dof_mgr->ov_vs(), Vxdotdot->domain()->dim());
    overlapped_Vxdotdot->assign(0.0);
    solMgr->get_cas_manager()->scatter(
        Vxdotdot, overlapped_Vxdotdot, CombineMode::INSERT);
  }

  RCP<ParamVec> params;
  if (parameter_index < p.size()){
    params = Teuchos::rcp(new ParamVec(p[parameter_index]));
  }

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
    param_offset = num_cols_x;  // offset of parameter derivs in deriv array

  TEUCHOS_TEST_FOR_EXCEPTION(
      sum_derivs && (num_cols_x != 0) && (num_cols_p != 0) &&
          (num_cols_x != num_cols_p),
      std::logic_error,
      "Seed matrices Vx and Vp must have the same number "
          << " of columns when sum_derivs is true and both are "
          << "non-null!" << std::endl);

  // Initialize
  if (params != Teuchos::null) {
    TanFadType p_val;
    int        num_cols_tot = param_offset + num_cols_p;
    for (unsigned int i = 0; i < params->size(); i++) {
      p_val = TanFadType(num_cols_tot, (*params)[i].baseValue);
      if (Vp != Teuchos::null) {
        auto Vp_constView = getLocalData(Vp);
        for (int k = 0; k < num_cols_p; k++) {
          p_val.fastAccessDx(param_offset + k) = Vp_constView[k][i];
        }
      } else
        p_val.fastAccessDx(param_offset + i) = 1.0;
      (*params)[i].family->setValue<PHAL::AlbanyTraits::Tangent>(p_val);
    }
  }

  workset.params       = params;
  workset.Vx           = overlapped_Vx;
  workset.Vxdot        = overlapped_Vxdot;
  workset.Vxdotdot     = overlapped_Vxdotdot;
  workset.Vp           = Vp;
  workset.num_cols_x   = num_cols_x;
  workset.num_cols_p   = num_cols_p;
  workset.param_offset = param_offset;
}

}  // namespace Albany
