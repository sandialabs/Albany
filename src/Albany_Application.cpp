//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
//#define DEBUG

#include "Albany_Application.hpp"
#include "Albany_DiscretizationFactory.hpp"
#include "Albany_DistributedParameterLibrary.hpp"
#include "Albany_Macros.hpp"
#include "Albany_ProblemFactory.hpp"
#include "Albany_ResponseFactory.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_Utils.hpp"
#include "Thyra_MultiVectorStdOps.hpp"
#include "Thyra_VectorBase.hpp"
#include "Thyra_VectorStdOps.hpp"

#include "Teuchos_TimeMonitor.hpp"

#include <string>
#include "Albany_DataTypes.hpp"

#include "Albany_DummyParameterAccessor.hpp"

#ifdef ALBANY_TEKO
#include "Teko_InverseFactoryOperator.hpp"
#endif

#include "Albany_ScalarResponseFunction.hpp"
#include "PHAL_Utilities.hpp"

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
      shapeParamsHaveBeenReset(false),
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
      shapeParamsHaveBeenReset(false),
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

#if !defined(ALBANY_EPETRA)
  removeEpetraRelatedPLs(params);
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
  // Initialize Phalanx postRegistration setup
  phxSetup = Teuchos::rcp(new PHAL::Setup());
  phxSetup->init_problem_params(problemParams);

  // Pull the number of solution vectors out of the problem and send them to the
  // discretization list, if the user specifies this in the problem
  Teuchos::ParameterList& discParams = params->sublist("Discretization");

  // Set in Albany_AbstractProblem constructor or in siblings
  num_time_deriv = problemParams->get<int>("Number Of Time Derivatives");

  // Possibly set in the Discretization list in the input file - this overrides
  // the above if set
  int num_time_deriv_from_input =
      discParams.get<int>("Number Of Time Derivatives", -1);
  if (num_time_deriv_from_input <
      0)  // Use the value from the problem by default
    discParams.set<int>("Number Of Time Derivatives", num_time_deriv);
  else
    num_time_deriv = num_time_deriv_from_input;

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
  } else if (solutionMethod == "Eigensolve") {
    solMethod = Eigensolve;
  } else if (
      solutionMethod == "Transient") {
#ifdef ALBANY_TEMPUS
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
    } else if (stepper_type == "Newmark Explicit a-Form") {
      requires_sdbcs_ = true;
    }
#else
    TEUCHOS_TEST_FOR_EXCEPTION(
        true,
        std::logic_error,
        "Solution Method = "
            << solutionMethod << " is not valid because "
            << "Trilinos was not built with Tempus turned ON.\n");
#endif
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(
        true,
        std::logic_error,
        "Solution Method must be Steady, Transient, Transient, "
            << "Continuation, Eigensolve, not : "
            << solutionMethod);
  }

  bool        expl = false;
  std::string stepperType;
  if (solMethod == Transient) {
    // Get Piro PL
    Teuchos::RCP<Teuchos::ParameterList> piroParams =
        Teuchos::sublist(params, "Piro", true);
    // Check if there is Tempus Solver sublist, and get the stepper type
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
  }

  // Create debug output object
  RCP<Teuchos::ParameterList> debugParams =
      Teuchos::sublist(params, "Debug Output", true);
  writeToMatrixMarketJac =
      debugParams->get("Write Jacobian to MatrixMarket", 0);
  computeJacCondNum = debugParams->get("Compute Jacobian Condition Number", 0);
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
  if (computeJacCondNum < -1) {
    TEUCHOS_TEST_FOR_EXCEPTION(
        true,
        Teuchos::Exceptions::InvalidParameter,
             "\nError in Albany::Application constructor:  "
            << "Invalid Parameter Compute Jacobian Condition Number.  "
               "Acceptable values are -1, 0, 1, 2, ...\n"); 
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
  problem->buildProblem(meshSpecs, stateMgr);

  if ((requires_sdbcs_ == true) && (problem->useSDBCs() == false) &&
      (no_dir_bcs_ == false)) {
    TEUCHOS_TEST_FOR_EXCEPTION(
        true,
        std::logic_error,
        "Error in Albany::Application: you are using a "
        "solver that requires SDBCs yet you are not "
        "using SDBCs!\n");
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
  relative_responses =
      responseList.get("Relative Responses Markers", defaultDataUnsignedInt);

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
  // Create the full mesh
  disc = discFactory->createDiscretization(
      neq,
      problem->getSideSetEquations(),
      stateMgr.getStateInfoStruct(),
      stateMgr.getSideSetStateInfoStruct(),
      problem->getFieldRequirements(),
      problem->getSideSetFieldRequirements(),
      problem->getNullSpace()); 
  // The following is for Aeras problems.
  explicit_scheme = disc->isExplicitScheme();
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
  if (!stateMgr.areStateVarsAllocated()) stateMgr.setupStateArrays(disc);

  solMgr = rcp(new AAdapt::AdaptiveSolutionManager(
      params,
      initial_guess,
      paramLib,
      disc,
      comm)); 

  try {
    // Create Distributed parameters and initialize them with data stored in the
    // mesh.
    const StateInfoStruct& distParamSIS = disc->getNodalParameterSIS();
    for (size_t is = 0; is < distParamSIS.size(); is++) {
      // Get name of distributed parameter
      const std::string& param_name = distParamSIS[is]->name;

      // Get parameter vector spaces and build parameter vector
      // Create distributed parameter and set workset_elem_dofs
      Teuchos::RCP<DistributedParameter> parameter(new DistributedParameter(
          param_name,
          disc->getVectorSpace(param_name),
          disc->getOverlapVectorSpace(param_name)));
      parameter->set_workset_elem_dofs(
          Teuchos::rcpFromRef(disc->getElNodeEqID(param_name)));

      // Get the vector and lower/upper bounds, and fill them with available
      // data
      Teuchos::RCP<Thyra_Vector> dist_param = parameter->vector();
      Teuchos::RCP<Thyra_Vector> dist_param_lowerbound =
          parameter->lower_bounds_vector();
      Teuchos::RCP<Thyra_Vector> dist_param_upperbound =
          parameter->upper_bounds_vector();

      std::stringstream lowerbound_name, upperbound_name;
      lowerbound_name << param_name << "_lowerbound";
      upperbound_name << param_name << "_upperbound";

      // Initialize parameter with data stored in the mesh
      disc->getField(*dist_param, param_name);
      const auto& nodal_param_states = disc->getNodalParameterSIS();
      bool        has_lowerbound(false), has_upperbound(false);
      for (int ist = 0; ist < static_cast<int>(nodal_param_states.size());
           ist++) {
        has_lowerbound = has_lowerbound || (nodal_param_states[ist]->name ==
                                            lowerbound_name.str());
        has_upperbound = has_upperbound || (nodal_param_states[ist]->name ==
                                            upperbound_name.str());
      }
      if (has_lowerbound) {
        disc->getField(*dist_param_lowerbound, lowerbound_name.str());
      } else {
        dist_param_lowerbound->assign(std::numeric_limits<ST>::lowest());
      }
      if (has_upperbound) {
        disc->getField(*dist_param_upperbound, upperbound_name.str());
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
  } catch (const std::logic_error&) {
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

  is_adjoint = problemParams->get("Solve Adjoint", false);

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
    const WorksetArray<int>::type& wsPhysIndex,
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

    fm[ps]->postRegistrationSetupForType<EvalT>(*phxSetup);

    // Update phalanx saved/unsaved fields based on field dependencies
    phxSetup->check_fields(fm[ps]->getFieldTagsForSizing<EvalT>());
    phxSetup->update_fields();

    writePhalanxGraph<EvalT>(fm[ps],evalName,phxGraphVisDetail);
  }
  if (dfm != Teuchos::null) {
    evalName = PHAL::evalName<EvalT>("DFM",0);
    phxSetup->insert_eval(evalName);

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
        PHAL::getDerivativeDimensions<EvalT>(this, ps, explicit_scheme));
    fm[ps]->setKokkosExtendedDataTypeDimensions<EvalT>(derivative_dimensions);
    fm[ps]->postRegistrationSetupForType<EvalT>(*phxSetup);

    // Update phalanx saved/unsaved fields based on field dependencies
    phxSetup->check_fields(fm[ps]->getFieldTagsForSizing<EvalT>());
    phxSetup->update_fields();

    writePhalanxGraph<EvalT>(fm[ps],evalName,phxGraphVisDetail);

    if (nfm != Teuchos::null && ps < nfm.size()) {
      evalName = PHAL::evalName<EvalT>("NFM",ps);
      phxSetup->insert_eval(evalName);

      nfm[ps]->setKokkosExtendedDataTypeDimensions<EvalT>(derivative_dimensions);
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
        PHAL::getDerivativeDimensions<EvalT>(this, 0, explicit_scheme));
    dfm->setKokkosExtendedDataTypeDimensions<EvalT>(derivative_dimensions);
    dfm->postRegistrationSetupForType<EvalT>(*phxSetup);

    // Update phalanx saved/unsaved fields based on field dependencies
    phxSetup->check_fields(dfm->getFieldTagsForSizing<EvalT>());
    phxSetup->update_fields();

    writePhalanxGraph<EvalT>(dfm,evalName,phxGraphVisDetail);
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
    Teuchos::Array<ParamVec> const&        p,
    const Teuchos::RCP<Thyra_Vector>&      f,
    double                                 dt)
{
  //#define DEBUG_OUTPUT

  TEUCHOS_FUNC_TIME_MONITOR("Albany Fill: Residual");
  using EvalT = PHAL::AlbanyTraits::Residual;
  postRegSetup<EvalT>();

  // Load connectivity map and coordinates
  const auto& wsElNodeEqID = disc->getWsElNodeEqID();
  const auto& wsPhysIndex  = disc->getWsPhysIndex();

  int const numWorksets = wsElNodeEqID.size();

  const Teuchos::RCP<Thyra_Vector> overlapped_f = solMgr->get_overlapped_f();

  Teuchos::RCP<const CombineAndScatterManager> cas_manager =
      solMgr->get_cas_manager();

  // Scatter x and xdot to the overlapped distrbution
  solMgr->scatterX(*x, x_dot.ptr(), x_dotdot.ptr());

  // Scatter distributed parameters
  distParamLib->scatter();

  // Set parameters
  for (int i = 0; i < p.size(); i++) {
    for (unsigned int j = 0; j < p[i].size(); j++) {
      p[i][j].family->setRealValueForAllTypes(p[i][j].baseValue);
    }
  }

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
      loadWorksetBucketInfo<EvalT>(workset, ws, evalName);

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
  using EvalT = PHAL::AlbanyTraits::Jacobian;
  postRegSetup<EvalT>();

  // Load connectivity map and coordinates
  const auto& wsElNodeEqID = disc->getWsElNodeEqID();
  const auto& wsPhysIndex  = disc->getWsPhysIndex();

  int numWorksets = wsElNodeEqID.size();

  Teuchos::RCP<Thyra_Vector> overlapped_f;
  if (Teuchos::nonnull(f)) { overlapped_f = solMgr->get_overlapped_f(); }

  Teuchos::RCP<Thyra_LinearOp> overlapped_jac = solMgr->get_overlapped_jac();
  auto                         cas_manager    = solMgr->get_cas_manager();

  // Scatter x and xdot to the overlapped distribution
  solMgr->scatterX(*x, xdot.ptr(), xdotdot.ptr());

  // Scatter distributed parameters
  distParamLib->scatter();

  // Set parameters
  for (int i = 0; i < p.size(); i++) {
    for (unsigned int j = 0; j < p[i].size(); j++) {
      p[i][j].family->setRealValueForAllTypes(p[i][j].baseValue);
    }
  }

  // Zero out overlapped residual
  if (Teuchos::nonnull(f)) {
    overlapped_f->assign(0.0);
    f->assign(0.0);
  }

  // Zero out Jacobian
  resumeFill(jac);
  assign(jac, 0.0);

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  if (!isFillActive(overlapped_jac)) { resumeFill(overlapped_jac); }
#endif
  assign(overlapped_jac, 0.0);
#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  if (isFillActive(overlapped_jac)) { fillComplete(overlapped_jac); }
  if (!isFillActive(overlapped_jac)) { resumeFill(overlapped_jac); }
#endif

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
    workset.Jac = overlapped_jac;
    loadWorksetJacobianInfo(workset, alpha, beta, omega);

    // fill Jacobian derivative dimensions:
    for (int ps = 0; ps < fm.size(); ps++) {
      (workset.Jacobian_deriv_dims)
          .push_back(
              PHAL::getDerivativeDimensions<EvalT>(
                  this, ps, explicit_scheme));
    }

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
    if (!workset.f.is_null()) {
      workset.f_kokkos = getNonconstDeviceData(workset.f);
    }
    if (!workset.Jac.is_null()) {
      workset.Jac_kokkos = getNonconstDeviceData(workset.Jac);
    }
#endif
    for (int ws = 0; ws < numWorksets; ws++) {
      const std::string evalName = PHAL::evalName<EvalT>("FM", wsPhysIndex[ws]);
      loadWorksetBucketInfo<EvalT>(workset, ws, evalName);

      // FillType template argument used to specialize Sacado
      fm[wsPhysIndex[ws]]->evaluateFields<EvalT>(workset);
      if (Teuchos::nonnull(nfm))
        deref_nfm(nfm, wsPhysIndex, ws)
            ->evaluateFields<EvalT>(workset);
    }
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

  {
    TEUCHOS_FUNC_TIME_MONITOR("Albany Jacobian Fill: Export");
    // Assemble global residual
    if (Teuchos::nonnull(f)) {
      cas_manager->combine(overlapped_f, f, CombineMode::ADD);
    }
    // Assemble global Jacobian
    cas_manager->combine(overlapped_jac, jac, CombineMode::ADD);
  }

  // scale Jacobian
  if (scaleBCdofs == false && scale != 1.0) {
    fillComplete(jac);
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
    resumeFill(jac);
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

    if(problem->useSDBCs() == true)
      dfm->preEvaluate<EvalT>(workset);

    loadWorksetNodesetInfo(workset);

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
  }
  fillComplete(jac);

  // Apply scaling to residual and Jacobian
  if (scaleBCdofs == true) {
    if (Teuchos::nonnull(f)) { Thyra::ele_wise_scale<ST>(*scaleVec_, f.ptr()); }
    // We MUST be able to cast jac to ScaledLinearOpBase in order to left scale
    // it.
    auto jac_scaled_lop =
        Teuchos::rcp_dynamic_cast<Thyra::ScaledLinearOpBase<ST>>(jac, true);
    jac_scaled_lop->scaleLeft(*scaleVec_);
  }

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  if (isFillActive(overlapped_jac)) {
    // Makes getLocalMatrix() valid.
    fillComplete(overlapped_jac);
  }
#endif
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
  if (computeJacCondNum !=
      0) {  // If requesting computation of condition number
#if defined(ALBANY_EPETRA)
    if (computeJacCondNum == -1) {
      // cout jacobian condition # every time it arises
      double condNum = computeConditionNumber(jac);
      *out << "Jacobian #" << countJac << " condition number = " << condNum
           << "\n";
    } else if (countJac == computeJacCondNum) {
      // cout jacobian condition # only at requested count#
      double condNum = computeConditionNumber(jac);
      *out << "Jacobian #" << countJac << " condition number = " << condNum
           << "\n";
    }
#else
    TEUCHOS_TEST_FOR_EXCEPTION(
        true,
        std::logic_error,
        "Error in Albany::Application: Compute Jacobian Condition Number debug "
        "option "
        "currently relies on an Epetra-based routine in AztecOO.\n To use this "
        "option, please "
        "rebuild Albany with ENABLE_ALBANY_EPETRA=ON.\nYou will then be able "
        "to have Albany "
        "output the Jacobian condition number when running either the Tpetra "
        "or Epetra stack.\n"
        "Notice that ENABLE_ALBANY_EPETRA is ON by default, so you must have "
        "disabled it explicitly.\n");
#endif
  }
  if (writeToMatrixMarketJac != 0 || writeToCoutJac != 0 ||
      computeJacCondNum != 0) {
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
    const Teuchos::Array<ParamVec>&              par,
    ParamVec*                                    deriv_par,
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
  const auto& wsElNodeEqID = disc->getWsElNodeEqID();
  const auto& wsPhysIndex  = disc->getWsPhysIndex();

  int numWorksets = wsElNodeEqID.size();

  Teuchos::RCP<Thyra_Vector> overlapped_f = solMgr->get_overlapped_f();

  // The combine-and-scatter manager
  auto cas_manager = solMgr->get_cas_manager();

  // Scatter x and xdot to the overlapped distrbution
  solMgr->scatterX(*x, xdot.ptr(), xdotdot.ptr()); 

  // Scatter distributed parameters
  distParamLib->scatter();

  // Scatter Vx to the overlapped distribution
  RCP<Thyra_MultiVector> overlapped_Vx;
  if (Teuchos::nonnull(Vx)) {
    overlapped_Vx = Thyra::createMembers(
        disc->getOverlapVectorSpace(), Vx->domain()->dim());
    overlapped_Vx->assign(0.0);
    cas_manager->scatter(Vx, overlapped_Vx, CombineMode::INSERT);
  }

  // Scatter Vxdot to the overlapped distribution
  RCP<Thyra_MultiVector> overlapped_Vxdot;
  if (Teuchos::nonnull(Vxdot)) {
    overlapped_Vxdot = Thyra::createMembers(
        disc->getOverlapVectorSpace(), Vxdot->domain()->dim());
    overlapped_Vxdot->assign(0.0);
    cas_manager->scatter(Vxdot, overlapped_Vxdot, CombineMode::INSERT);
  }
  RCP<Thyra_MultiVector> overlapped_Vxdotdot;
  if (Teuchos::nonnull(Vxdotdot)) {
    overlapped_Vxdotdot = Thyra::createMembers(
        disc->getOverlapVectorSpace(), Vxdotdot->domain()->dim());
    overlapped_Vxdotdot->assign(0.0);
    cas_manager->scatter(Vxdotdot, overlapped_Vxdotdot, CombineMode::INSERT);
  }

  // Set parameters
  for (int i = 0; i < par.size(); i++) {
    for (unsigned int j = 0; j < par[i].size(); j++) {
      par[i][j].family->setRealValueForAllTypes(par[i][j].baseValue);
    }
  }

  RCP<ParamVec> params = rcp(deriv_par, false);

  // Zero out overlapped residual
  if (Teuchos::nonnull(f)) {
    overlapped_f->assign(0.0);
    f->assign(0.0);
  }

  RCP<Thyra_MultiVector> overlapped_JV;
  if (Teuchos::nonnull(JV)) {
    overlapped_JV = Thyra::createMembers(
        disc->getOverlapVectorSpace(), JV->domain()->dim());
    overlapped_JV->assign(0.0);
    JV->assign(0.0);
  }

  RCP<Thyra_MultiVector> overlapped_fp;
  if (Teuchos::nonnull(fp)) {
    overlapped_fp = Thyra::createMembers(
        disc->getOverlapVectorSpace(), fp->domain()->dim());
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
    for (unsigned int i = 0; i < params->size(); i++) {
      p = TanFadType(num_cols_tot, (*params)[i].baseValue);
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
      loadWorksetBucketInfo<EvalT>(workset, ws, evalName);

      // FillType template argument used to specialize Sacado
      fm[wsPhysIndex[ws]]->evaluateFields<EvalT>(workset);
      if (nfm != Teuchos::null)
        deref_nfm(nfm, wsPhysIndex, ws)
            ->evaluateFields<EvalT>(workset);
    }

    // fill Tangent derivative dimensions
    for (int ps = 0; ps < fm.size(); ps++) {
      (workset.Tangent_deriv_dims)
          .push_back(PHAL::getDerivativeDimensions<EvalT>(
              this, ps));
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
    const Teuchos::Array<ParamVec>&              p,
    const std::string&                           dist_param_name,
    const bool                                   trans,
    const Teuchos::RCP<const Thyra_MultiVector>& V,
    const Teuchos::RCP<Thyra_MultiVector>&       fpV)
{
  TEUCHOS_FUNC_TIME_MONITOR("Albany Fill: Distributed Parameter Derivative");
  using EvalT = PHAL::AlbanyTraits::DistParamDeriv;
  postRegSetup<EvalT>();

  // Load connectivity map and coordinates
  const auto& wsElNodeEqID = disc->getWsElNodeEqID();
  const auto& wsPhysIndex  = disc->getWsPhysIndex();

  int numWorksets = wsElNodeEqID.size();

  // The combin-and-scatter manager
  auto cas_manager = solMgr->get_cas_manager();

  // Scatter x and xdot to the overlapped distribution
  solMgr->scatterX(*x, xdot.ptr(), xdotdot.ptr());

  // Scatter distributed parameters
  distParamLib->scatter();

  // Set parameters
  for (int i = 0; i < p.size(); i++) {
    for (unsigned int j = 0; j < p[i].size(); j++) {
      p[i][j].family->setRealValueForAllTypes(p[i][j].baseValue);
    }
  }

  Teuchos::RCP<Thyra_MultiVector> overlapped_fpV;
  if (trans) {
    const auto& vs = distParamLib->get(dist_param_name)->overlap_vector_space();
    overlapped_fpV = Thyra::createMembers(vs, V->domain()->dim());
  } else {
    overlapped_fpV = Thyra::createMembers(
        disc->getOverlapVectorSpace(), fpV->domain()->dim());
  }
  overlapped_fpV->assign(0.0);
  fpV->assign(0.0);

  Teuchos::RCP<const Thyra_MultiVector> V_bc = V;

  // For (df/dp)^T*V, we have to evaluate Dirichlet BC's first
  if (trans && dfm != Teuchos::null) {
    Teuchos::RCP<Thyra_MultiVector> V_bc_nonconst = V->clone_mv();
    V_bc                                          = V_bc_nonconst;

    PHAL::Workset workset;

    workset.fpV                        = fpV;
    workset.Vp_bc                      = V_bc_nonconst;
    workset.transpose_dist_param_deriv = trans;
    workset.dist_param_deriv_name      = dist_param_name;
    workset.disc                       = disc;

    double const this_time = fixTime(current_time);

    workset.current_time = this_time;

    dfm_set(workset, x, xdot, xdotdot);

    loadWorksetNodesetInfo(workset);

    // FillType template argument used to specialize Sacado
    dfm->evaluateFields<EvalT>(workset);
  }

  // Import V (after BC's applied) to overlapped distribution
  RCP<Thyra_MultiVector> overlapped_V;
  if (trans) {
    overlapped_V = Thyra::createMembers(
        disc->getOverlapVectorSpace(), V_bc->domain()->dim());
    overlapped_V->assign(0.0);
    cas_manager->scatter(V_bc, overlapped_V, CombineMode::INSERT);
  } else {
    const auto& vs = distParamLib->get(dist_param_name)->overlap_vector_space();
    overlapped_V   = Thyra::createMembers(vs, V_bc->domain()->dim());
    overlapped_V->assign(0.0);
    distParamLib->get(dist_param_name)->get_cas_manager()
        ->scatter(V_bc, overlapped_V, CombineMode::INSERT);
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
      loadWorksetBucketInfo<EvalT>(workset, ws, evalName);

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
    workset.Vp                         = V_bc;
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
    const double                                 alpha,
    const double                                 beta,
    const double                                 omega,
    const double                                 current_time,
    bool                                         sum_derivs,
    const Teuchos::RCP<const Thyra_Vector>&      x,
    const Teuchos::RCP<const Thyra_Vector>&      xdot,
    const Teuchos::RCP<const Thyra_Vector>&      xdotdot,
    const Teuchos::Array<ParamVec>&              p,
    ParamVec*                                    deriv_p,
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
      deriv_p,
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
    ParamVec*                                        deriv_p,
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
      deriv_p,
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
                this, ps, explicit_scheme));
        sfm[ps]
            ->setKokkosExtendedDataTypeDimensions<PHAL::AlbanyTraits::Jacobian>(
                derivative_dimensions);
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
  const auto& wsElNodeEqID = disc->getWsElNodeEqID();
  const auto& wsPhysIndex  = disc->getWsPhysIndex();

  int numWorksets = wsElNodeEqID.size();

  // Scatter to the overlapped distrbution
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
    loadWorksetBucketInfo<PHAL::AlbanyTraits::Residual>(workset, ws, evalName);
    sfm[wsPhysIndex[ws]]->evaluateFields<PHAL::AlbanyTraits::Residual>(workset);
  }
}

void
Application::registerShapeParameters()
{
  int numShParams = shapeParams.size();
  if (shapeParamNames.size() == 0) {
    shapeParamNames.resize(numShParams);
    for (int i = 0; i < numShParams; i++)
      shapeParamNames[i] = strint("ShapeParam", i);
  }
  DummyParameterAccessor<PHAL::AlbanyTraits::Jacobian, SPL_Traits>* dJ =
      new DummyParameterAccessor<PHAL::AlbanyTraits::Jacobian, SPL_Traits>();
  DummyParameterAccessor<PHAL::AlbanyTraits::Tangent, SPL_Traits>* dT =
      new DummyParameterAccessor<PHAL::AlbanyTraits::Tangent, SPL_Traits>();

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

PHAL::AlbanyTraits::Residual::ScalarT&
Application::getValue(const std::string& name)
{
  int index = -1;
  for (unsigned int i = 0; i < shapeParamNames.size(); i++) {
    if (name == shapeParamNames[i]) index = i;
  }
  TEUCHOS_TEST_FOR_EXCEPTION(
      index == -1,
      std::logic_error,
      "Error in GatherCoordinateVector::getValue, \n"
          << "   Unrecognized param name: " << name << std::endl);

  shapeParamsHaveBeenReset = true;

  return shapeParams[index];
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
  workset.is_adjoint      = is_adjoint;
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
      // to get row informations
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
    // get row informations
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
    // std::cout << "IKTIKT key = " << key << std::endl;
    const std::vector<std::vector<int>>& nsNodes =
        workset.nodeSets->find(key)->second;
    for (unsigned int i = 0; i < nsNodes.size(); i++) {
      // std::cout << "IKTIKT ns, offsets size: " << ns << ", " <<
      // offsets_[ns].size() << "\n";
      for (unsigned j = 0; j < offsets_[ns].size(); j++) {
        int lunk                = nsNodes[i][offsets_[ns][j]];
        scaleVecLocalData[lunk] = scale;
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

  // Scatter xT and xdotT to the overlapped distrbution
  solMgr->scatterX(*x, xdot.ptr(), xdotdot.ptr());

  // Scatter distributed parameters
  distParamLib->scatter();

  // Set parameters
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
  int np = Albany::CalculateNumberParams(problemParams);
  return std::max(1, np);
}

void
Application::setupTangentWorksetInfo(
    PHAL::Workset&                               workset,
    double                                       current_time,
    bool                                         sum_derivs,
    const Teuchos::RCP<const Thyra_Vector>&      x,
    const Teuchos::RCP<const Thyra_Vector>&      xdot,
    const Teuchos::RCP<const Thyra_Vector>&      xdotdot,
    const Teuchos::Array<ParamVec>&              p,
    ParamVec*                                    deriv_p,
    const Teuchos::RCP<const Thyra_MultiVector>& Vx,
    const Teuchos::RCP<const Thyra_MultiVector>& Vxdot,
    const Teuchos::RCP<const Thyra_MultiVector>& Vxdotdot,
    const Teuchos::RCP<const Thyra_MultiVector>& Vp)
{
  setupBasicWorksetInfo(workset, current_time, x, xdot, xdotdot, p);

  // Scatter Vx dot the overlapped distribution
  RCP<Thyra_MultiVector> overlapped_Vx;
  if (Vx != Teuchos::null) {
    overlapped_Vx = Thyra::createMembers(
        disc->getOverlapVectorSpace(), Vx->domain()->dim());
    overlapped_Vx->assign(0.0);
    solMgr->get_cas_manager()->scatter(Vx, overlapped_Vx, CombineMode::INSERT);
  }

  // Scatter Vx dot the overlapped distribution
  RCP<Thyra_MultiVector> overlapped_Vxdot;
  if (Vxdot != Teuchos::null) {
    overlapped_Vxdot = Thyra::createMembers(
        disc->getOverlapVectorSpace(), Vxdot->domain()->dim());
    overlapped_Vxdot->assign(0.0);
    solMgr->get_cas_manager()->scatter(
        Vxdot, overlapped_Vxdot, CombineMode::INSERT);
  }
  RCP<Thyra_MultiVector> overlapped_Vxdotdot;
  if (Vxdotdot != Teuchos::null) {
    overlapped_Vxdotdot = Thyra::createMembers(
        disc->getOverlapVectorSpace(), Vxdotdot->domain()->dim());
    overlapped_Vxdotdot->assign(0.0);
    solMgr->get_cas_manager()->scatter(
        Vxdotdot, overlapped_Vxdotdot, CombineMode::INSERT);
  }

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

void
Application::removeEpetraRelatedPLs(
    const Teuchos::RCP<Teuchos::ParameterList>& params)
{
  if (params->isSublist("Piro")) {
    Teuchos::ParameterList& piroPL = params->sublist("Piro", true);
    if (piroPL.isSublist("NOX")) {
      Teuchos::ParameterList& noxPL = piroPL.sublist("NOX", true);
      if (noxPL.isSublist("Direction")) {
        Teuchos::ParameterList& dirPL = noxPL.sublist("Direction", true);
        if (dirPL.isSublist("Newton")) {
          Teuchos::ParameterList& newPL = dirPL.sublist("Newton", true);
          if (newPL.isSublist("Stratimikos Linear Solver")) {
            Teuchos::ParameterList& stratPL =
                newPL.sublist("Stratimikos Linear Solver", true);
            if (stratPL.isSublist("Stratimikos")) {
              Teuchos::ParameterList& strataPL =
                  stratPL.sublist("Stratimikos", true);
              if (strataPL.isSublist("AztecOO")) {
                strataPL.remove("AztecOO", true);
              }
              if (strataPL.isSublist("Linear Solver Types")) {
                Teuchos::ParameterList& lsPL =
                    strataPL.sublist("Linear Solver Types", true);
                if (lsPL.isSublist("AztecOO")) { lsPL.remove("AztecOO", true); }
              }
            }
          }
        }
      }
    }
  }
}

}  // namespace Albany
