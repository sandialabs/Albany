//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_Application.hpp"
#include "Albany_Utils.hpp"
#include "AAdapt_AdaptationFactory.hpp"
#include "AAdapt_RC_Manager.hpp"
#include "Albany_ProblemFactory.hpp"
#include "Albany_DiscretizationFactory.hpp"
#include "Albany_ResponseFactory.hpp"
#include "Stokhos_OrthogPolyBasis.hpp"
#include "Teuchos_TimeMonitor.hpp"

#if defined(ALBANY_EPETRA)
#include "Epetra_LocalMap.h"
#include "EpetraExt_MultiVectorOut.h"
#include "EpetraExt_RowMatrixOut.h"
#include "EpetraExt_VectorOut.h"
#include "Petra_Converters.hpp"
#endif

#include<string>
#include "Albany_DataTypes.hpp"

#include "Albany_DummyParameterAccessor.hpp"
#ifdef ALBANY_CUTR
  #include "CUTR_CubitMeshMover.hpp"
  #include "STKMeshData.hpp"
#endif

#ifdef ALBANY_TEKO
#include "Teko_InverseFactoryOperator.hpp"
#if defined(ALBANY_EPETRA)
#include "Teko_StridedEpetraOperator.hpp"
#endif
#endif

#include "Albany_ScalarResponseFunction.hpp"
#include "PHAL_Utilities.hpp"

#ifdef ALBANY_PERIDIGM
#if defined(ALBANY_EPETRA)
#include "PeridigmManager.hpp"
#endif
#endif

#if defined (ALBANY_GOAL)
#include "GOAL_BCManager.hpp"
#endif

//eb-hack
#include "Adapt_NodalDataVector.hpp"

using Teuchos::ArrayRCP;
using Teuchos::RCP;
using Teuchos::rcp;
using Teuchos::rcp_dynamic_cast;
using Teuchos::TimeMonitor;
using Teuchos::getFancyOStream;
using Teuchos::rcpFromRef;

int countJac; //counter which counts instances of Jacobian (for debug output)
int countRes; //counter which counts instances of residual (for debug output)

extern bool TpetraBuild;

Albany::Application::
Application(const RCP<const Teuchos_Comm>& comm_,
	    const RCP<Teuchos::ParameterList>& params,
	    const RCP<const Tpetra_Vector>& initial_guess) :
  commT(comm_),
  out(Teuchos::VerboseObjectBase::getDefaultOStream()),
  physicsBasedPreconditioner(false),
  shapeParamsHaveBeenReset(false),
  morphFromInit(true), perturbBetaForDirichlets(0.0),
  phxGraphVisDetail(0),
  stateGraphVisDetail(0)
{
#if defined(ALBANY_EPETRA)
  comm = Albany::createEpetraCommFromTeuchosComm(comm_); 
#endif
  initialSetUp(params);
  createMeshSpecs();
  buildProblem();
  createDiscretization();
  finalSetUp(params,initial_guess);
}


Albany::Application::
Application(const RCP<const Teuchos_Comm>& comm_) :
    commT(comm_),
    out(Teuchos::VerboseObjectBase::getDefaultOStream()),
    physicsBasedPreconditioner(false),
    shapeParamsHaveBeenReset(false),
    morphFromInit(true), perturbBetaForDirichlets(0.0),
    phxGraphVisDetail(0),
    stateGraphVisDetail(0)
{
#if defined(ALBANY_EPETRA)
  comm = Albany::createEpetraCommFromTeuchosComm(comm_); 
#endif
};

namespace {
int calcTangentDerivDimension (
  const Teuchos::RCP<Teuchos::ParameterList>& problemParams)
{
  Teuchos::ParameterList& parameterParams =
    problemParams->sublist("Parameters");
  int num_param_vecs =
    parameterParams.get("Number of Parameter Vectors", 0);
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
    Teuchos::ParameterList& pList = using_old_parameter_list ?
      parameterParams :
      parameterParams.sublist(Albany::strint("Parameter Vector", i));
    np += pList.get<int>("Number");
  }
  return std::max(1, np);
}
} // namespace

void Albany::Application::initialSetUp(const RCP<Teuchos::ParameterList>& params) {
  // Create parameter libraries
  paramLib = rcp(new ParamLib);
  distParamLib = rcp(new DistParamLib);

#ifdef ALBANY_DEBUG
#if defined(ALBANY_EPETRA)
  int break_set = (getenv("ALBANY_BREAK") == NULL)?0:1;
  int env_status = 0;
  int length = 1;
  comm->SumAll(&break_set, &env_status, length);
  if(env_status != 0){
    *out << "Host and Process Ids for tasks" << std::endl;
    comm->Barrier();
    int nproc = comm->NumProc();
    for(int i = 0; i < nproc; i++) {
      if(i == comm->MyPID()) {
        char buf[80];
        char hostname[80]; gethostname(hostname, sizeof(hostname));
        sprintf(buf, "Host: %s   PID: %d", hostname, getpid());
        *out << buf << std::endl;
        std::cout.flush();
        sleep(1);
      }
      comm->Barrier();
    }
    if(comm->MyPID() == 0) {
      char go = ' ';
      std::cout << "\n";
      std::cout << "** Client has paused because the environment variable ALBANY_BREAK has been set.\n";
      std::cout << "** You may attach a debugger to processes now.\n";
      std::cout << "**\n";
      std::cout << "** Enter a character (not whitespace), then <Return> to continue. > "; std::cout.flush();
      std::cin >> go;
      std::cout << "\n** Now pausing for 3 seconds.\n"; std::cout.flush();
    }
    sleep(3);
  }
  comm->Barrier();
#endif
#endif

  // Create problem object
  problemParams = Teuchos::sublist(params, "Problem", true);
  Albany::ProblemFactory problemFactory(problemParams, paramLib, commT);
  rc_mgr = AAdapt::rc::Manager::create(
    Teuchos::rcp(&stateMgr, false), *problemParams);
  if (Teuchos::nonnull(rc_mgr))
    problemFactory.setReferenceConfigurationManager(rc_mgr);
  problem = problemFactory.create();

#if defined(ALBANY_GOAL)
  bcMgr = GOAL::BCManager::create(*problemParams);
#endif

  // Validate Problem parameters against list for this specific problem
  problemParams->validateParameters(*(problem->getValidProblemParameters()),0);

  try {
    tangent_deriv_dim = calcTangentDerivDimension(problemParams);
  } catch (...) {
    tangent_deriv_dim = 1;
  }

  // Save the solution method to be used
  std::string solutionMethod = problemParams->get("Solution Method", "Steady");
  if(solutionMethod == "Steady")
    solMethod = Steady;
  else if(solutionMethod == "Continuation")
    solMethod = Continuation;
  else if(solutionMethod == "Transient")
    solMethod = Transient;
  else if(solutionMethod == "Eigensolve")
    solMethod = Eigensolve;
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true,
            std::logic_error, "Solution Method must be Steady, Transient, "
            << "Continuation, or Eigensolve not : " << solutionMethod);

  // Register shape parameters for manipulation by continuation/optimization
  if (problemParams->get("Enable Cubit Shape Parameters",false)) {
#ifdef ALBANY_CUTR
    TEUCHOS_FUNC_TIME_MONITOR("Albany-Cubit MeshMover");
    meshMover = rcp(new CUTR::CubitMeshMover
          (problemParams->get<std::string>("Cubit Base Filename")));

    meshMover->getShapeParams(shapeParamNames, shapeParams);
    *out << "SSS : Registering " << shapeParams.size() << " Shape Parameters" << std::endl;

    registerShapeParameters();

#else
  TEUCHOS_TEST_FOR_EXCEPTION(problemParams->get("Enable Cubit Shape Parameters",false), std::logic_error,
                             "Cubit requested but not Compiled in!");
#endif
  }

  determinePiroSolver(params);

  physicsBasedPreconditioner = problemParams->get("Use Physics-Based Preconditioner",false);
#ifdef ALBANY_TEKO
  if (physicsBasedPreconditioner)
    tekoParams = Teuchos::sublist(problemParams, "Teko", true);
#endif

  // Create debug output object
  RCP<Teuchos::ParameterList> debugParams =
    Teuchos::sublist(params, "Debug Output", true);
  writeToMatrixMarketJac = debugParams->get("Write Jacobian to MatrixMarket", 0);
  writeToMatrixMarketRes = debugParams->get("Write Residual to MatrixMarket", 0);
  writeToCoutJac = debugParams->get("Write Jacobian to Standard Output", 0);
  writeToCoutRes = debugParams->get("Write Residual to Standard Output", 0);
  derivatives_check_ = debugParams->get<int>("Derivative Check", 0);
  //the above 4 parameters cannot have values < -1
  if (writeToMatrixMarketJac < -1)  {TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                                  std::endl << "Error in Albany::Application constructor:  " <<
                                  "Invalid Parameter Write Jacobian to MatrixMarket.  Acceptable values are -1, 0, 1, 2, ... " << std::endl);}
  if (writeToMatrixMarketRes < -1)  {TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                                  std::endl << "Error in Albany::Application constructor:  " <<
                                  "Invalid Parameter Write Residual to MatrixMarket.  Acceptable values are -1, 0, 1, 2, ... " << std::endl);}
  if (writeToCoutJac < -1)  {TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                                  std::endl << "Error in Albany::Application constructor:  " <<
                                  "Invalid Parameter Write Jacobian to Standard Output.  Acceptable values are -1, 0, 1, 2, ... " << std::endl);}
  if (writeToCoutRes < -1)  {TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                                  std::endl << "Error in Albany::Application constructor:  " <<
                                  "Invalid Parameter Write Residual to Standard Output.  Acceptable values are -1, 0, 1, 2, ... " << std::endl);}
  if (writeToMatrixMarketJac != 0 || writeToCoutJac != 0 )
     countJac = 0; //initiate counter that counts instances of Jacobian matrix to 0
  if (writeToMatrixMarketRes != 0 || writeToCoutRes != 0)
     countRes = 0; //initiate counter that counts instances of Jacobian matrix to 0

  // Create discretization object
  discFactory = rcp(new Albany::DiscretizationFactory(params, commT));

#ifdef ALBANY_CUTR
  discFactory->setMeshMover(meshMover);
#endif

#if defined(ALBANY_LCM)
  // Check for Schwarz parameters
  bool const
  has_app_array = params->isParameter("Application Array");

  bool const
  has_app_index = params->isParameter("Application Index");

  bool const
  has_app_name_index_map = params->isParameter("Application Name Index Map");

  // Only if all these are present set them in the app.
  bool const
  has_all = has_app_array && has_app_index && has_app_name_index_map;

  if (has_all == true) {
    Teuchos::ArrayRCP<Teuchos::RCP<Albany::Application>>
    aa = params->get<Teuchos::ArrayRCP<Teuchos::RCP<Albany::Application>>>
    ("Application Array");

    int const
    ai = params->get<int>("Application Index");

    Teuchos::RCP<std::map<std::string, int>>
    anim = params->get<Teuchos::RCP<std::map<std::string, int>>>
    ("Application Name Index Map");

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

void Albany::Application::createMeshSpecs(Teuchos::RCP<Albany::AbstractMeshStruct> mesh) {
  // Get mesh specification object: worksetSize, cell topology, etc
  meshSpecs = discFactory->createMeshSpecs(mesh);
}


void Albany::Application::buildProblem()   {
#if defined(ALBANY_LCM)
  // This is needed for Schwarz coupling so that when Dirichlet
  // BCs are created we know what application is doing it.
  problem->setApplication(Teuchos::rcp(this, false));
#endif //ALBANY_LCM

  problem->buildProblem(meshSpecs, stateMgr);

  neq = problem->numEquations();
  spatial_dimension = problem->spatialDimension();

  // Construct responses
  // This really needs to happen after the discretization is created for
  // distributed responses, but currently it can't be moved because there
  // are responses that setup states, which has to happen before the
  // discretization is created.  We will delay setup of the distributed
  // responses to deal with this temporarily.
  Teuchos::ParameterList& responseList =
    problemParams->sublist("Response Functions");
  ResponseFactory responseFactory(Teuchos::rcp(this,false), problem, meshSpecs,
                                  Teuchos::rcp(&stateMgr,false));
  responses = responseFactory.createResponseFunctions(responseList);

  // Build state field manager
  if (Teuchos::nonnull(rc_mgr)) rc_mgr->beginBuildingSfm();
  sfm.resize(meshSpecs.size());
  Teuchos::RCP<PHX::DataLayout> dummy =
    Teuchos::rcp(new PHX::MDALayout<Dummy>(0));
  for (int ps=0; ps<meshSpecs.size(); ps++) {
    std::string elementBlockName = meshSpecs[ps]->ebName;
    std::vector<std::string>responseIDs_to_require =
      stateMgr.getResidResponseIDsToRequire(elementBlockName);
    sfm[ps] = Teuchos::rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
    Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> > tags =
      problem->buildEvaluators(*sfm[ps], *meshSpecs[ps], stateMgr,
                               BUILD_STATE_FM, Teuchos::null);
    std::vector<std::string>::const_iterator it;
    for (it = responseIDs_to_require.begin();
         it != responseIDs_to_require.end();
         it++) {
      const std::string& responseID = *it;
      PHX::Tag<PHAL::AlbanyTraits::Residual::ScalarT> res_response_tag(
        responseID, dummy);
      sfm[ps]->requireField<PHAL::AlbanyTraits::Residual>(res_response_tag);
    }
  }
  if (Teuchos::nonnull(rc_mgr)) rc_mgr->endBuildingSfm();
}

void Albany::Application::createDiscretization() {
  // Create the full mesh
  disc = discFactory->createDiscretization(neq, stateMgr.getStateInfoStruct(),
                                          problem->getFieldRequirements(),
                                          problem->getNullSpace());
}

void Albany::Application::finalSetUp(const Teuchos::RCP<Teuchos::ParameterList>& params,
    const Teuchos::RCP<const Tpetra_Vector>& initial_guess) {



/*
  RCP<const Tpetra_Vector> initial_guessT;
  if (Teuchos::nonnull(initial_guess)) {
    initial_guessT = Petra::EpetraVector_To_TpetraVectorConst(*initial_guess, commT);
  }
*/

  // Now that space is allocated in STK for state fields, initialize states.
  // If the states have been already allocated, skip this.
  if(!stateMgr.areStateVarsAllocated())
    stateMgr.setStateArrays(disc);

#if defined(ALBANY_EPETRA)
  if(!TpetraBuild){
    RCP<Epetra_Vector> initial_guessE;
    if (Teuchos::nonnull(initial_guess)) {
      Petra::TpetraVector_To_EpetraVector(initial_guess, initial_guessE, comm);
    }
    solMgr = rcp(new AAdapt::AdaptiveSolutionManager(params, disc, initial_guessE));
  }
#endif

  solMgrT = rcp(new AAdapt::AdaptiveSolutionManagerT(
      params, initial_guess, paramLib, stateMgr,
      // Prevent a circular dependency.
      Teuchos::rcp(rc_mgr.get(), false),
      commT));
  if (Teuchos::nonnull(rc_mgr)) rc_mgr->setSolutionManager(solMgrT);

#ifdef ALBANY_PERIDIGM
#if defined(ALBANY_EPETRA)
  if (Teuchos::nonnull(LCM::PeridigmManager::self()))
    LCM::PeridigmManager::self()->setDirichletFields(disc);
#endif
#endif


#if defined(ALBANY_EPETRA)
  try {
    //dp-todo getNodalParameterSIS() needs to be implemented in PUMI. Until
    // then, catch the exception and continue.
    // Create Distributed parameters and initialize them with data stored in the mesh.
    const Albany::StateInfoStruct& distParamSIS = disc->getNodalParameterSIS();
    for(int is=0; is<distParamSIS.size(); is++) {
      // Get name of distributed parameter
      const std::string& param_name = distParamSIS[is]->name;

      // Get parameter maps and build parameter vector
      Teuchos::RCP<Tpetra_Vector> dist_paramT;
      Teuchos::RCP<const Tpetra_Map> node_mapT, overlap_node_mapT;
      { //dp-convert
        const Teuchos::RCP<const Epetra_Map> node_map = disc->getMap(param_name);
        const Teuchos::RCP<const Epetra_Map> overlap_node_map = disc->getOverlapMap(param_name);
        Epetra_Vector dist_param(*node_map);
        // Initialize parameter with data stored in the mesh
        disc->getField(dist_param, param_name);

        // JR: for now, initialize to constant value from user input if requested.  This needs to be generalized.
        if(params->sublist("Problem").isType<Teuchos::ParameterList>("Topology Parameters")){
          Teuchos::ParameterList& topoParams = params->sublist("Problem").sublist("Topology Parameters");
          if(topoParams.isType<std::string>("Entity Type") && topoParams.isType<double>("Initial Value")){
            if(topoParams.get<std::string>("Entity Type") == "Distributed Parameter" &&
               topoParams.get<std::string>("Topology Name") == param_name ){
              double initVal = topoParams.get<double>("Initial Value");
              dist_param.PutScalar(initVal);
            }
          }
        }

        dist_paramT = Petra::EpetraVector_To_TpetraVectorNonConst(dist_param, commT);
        node_mapT = Petra::EpetraMap_To_TpetraMap(node_map, commT);
        overlap_node_mapT = Petra::EpetraMap_To_TpetraMap(overlap_node_map, commT);
      }

      // Create distributed parameter and set workset_elem_dofs
      Teuchos::RCP<TpetraDistributedParameter> parameter(
        new TpetraDistributedParameter(param_name, dist_paramT, node_mapT, overlap_node_mapT));
      parameter->set_workset_elem_dofs(Teuchos::rcpFromRef(disc->getElNodeEqID(param_name)));

      // Add parameter to the distributed parameter library
      distParamLib->add(parameter->name(), parameter);
    }
  } catch (const std::logic_error&) {}
#endif

  // Now setup response functions (see note above)
  if(!TpetraBuild){
#if defined(ALBANY_EPETRA)
    for (int i=0; i<responses.size(); i++)
      responses[i]->setup();
#endif
  }
  else {
    for (int i=0; i<responses.size(); i++)
      responses[i]->setupT();
  }

  // Set up memory for workset
  fm = problem->getFieldManager();
  TEUCHOS_TEST_FOR_EXCEPTION(fm==Teuchos::null, std::logic_error,
                             "getFieldManager not implemented!!!");
  dfm = problem->getDirichletFieldManager();
  nfm = problem->getNeumannFieldManager();

  if (commT->getRank()==0) {
    phxGraphVisDetail= problemParams->get("Phalanx Graph Visualization Detail", 0);
    stateGraphVisDetail= phxGraphVisDetail;
  }

  *out << "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n"
       << " Sacado ParameterLibrary has been initialized:\n "
       << *paramLib
       << "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n"
       << std::endl;

  ignore_residual_in_jacobian =
    problemParams->get("Ignore Residual In Jacobian", false);

  perturbBetaForDirichlets = problemParams->get("Perturb Dirichlet",0.0);

  is_adjoint =
    problemParams->get("Solve Adjoint", false);

  // For backward compatibility, use any value at the old location of the "Compute Sensitivity" flag
  // as a default value for the new flag location when the latter has been left undefined
  const std::string sensitivityToken = "Compute Sensitivities";
  const Teuchos::Ptr<const bool> oldSensitivityFlag(problemParams->getPtr<bool>(sensitivityToken));
  if (Teuchos::nonnull(oldSensitivityFlag)) {
    Teuchos::ParameterList &solveParams = params->sublist("Piro").sublist("Analysis").sublist("Solve");
    solveParams.get(sensitivityToken, *oldSensitivityFlag);
  }

#ifdef ALBANY_MOR
#if defined(ALBANY_EPETRA)
  if(disc->supportsMOR())
    morFacade = createMORFacade(disc, problemParams);
#endif
#endif

/*
 * Initialize mesh adaptation features
 */

#if defined(ALBANY_EPETRA)
  if(!TpetraBuild &&  solMgr->hasAdaptation()){

    solMgr->buildAdaptiveProblem(paramLib, stateMgr, commT);

  }
#endif

#ifdef ALBANY_PERIDIGM
#if defined(ALBANY_EPETRA)
  if (Teuchos::nonnull(LCM::PeridigmManager::self()))
    LCM::PeridigmManager::self()->initialize(params, disc, commT);
#endif
#endif
}

Albany::Application::
~Application()
{
#ifdef ALBANY_DEBUG
  *out << "Calling destructor for Albany_Application" << std::endl;
#endif
}

RCP<Albany::AbstractDiscretization>
Albany::Application::
getDiscretization() const
{
  return disc;
}

RCP<Albany::AbstractProblem>
Albany::Application::
getProblem() const
{
  return problem;
}

RCP<const Teuchos_Comm>
Albany::Application::
getComm() const
{
  return commT;
}

#if defined(ALBANY_EPETRA)
RCP<const Epetra_Map>
Albany::Application::
getMap() const
{
  return disc->getMap();
}
#endif

RCP<const Tpetra_Map>
Albany::Application::
getMapT() const
{
  return disc->getMapT();
}


#if defined(ALBANY_EPETRA)
RCP<const Epetra_CrsGraph>
Albany::Application::
getJacobianGraph() const
{
  return disc->getJacobianGraph();
}
#endif

RCP<const Tpetra_CrsGraph>
Albany::Application::
getJacobianGraphT() const
{
  return disc->getJacobianGraphT();
}

#if defined(ALBANY_EPETRA)
RCP<Epetra_Operator>
Albany::Application::
getPreconditioner()
{
#if defined(ALBANY_TEKO)
   //inverseLib = Teko::InverseLibrary::buildFromStratimikos();
   inverseLib = Teko::InverseLibrary::buildFromParameterList(tekoParams->sublist("Inverse Factory Library"));
   inverseLib->PrintAvailableInverses(*out);

   inverseFac = inverseLib->getInverseFactory(tekoParams->get("Preconditioner Name","Amesos"));

   // get desired blocking of unknowns
   std::stringstream ss;
   ss << tekoParams->get<std::string>("Unknown Blocking");

   // figure out the decomposition requested by the string
   unsigned int num=0,sum=0;
   while(not ss.eof()) {
      ss >> num;
      TEUCHOS_ASSERT(num>0);
      sum += num;
      blockDecomp.push_back(num);
   }
   TEUCHOS_ASSERT(neq==sum);

   return rcp(new Teko::Epetra::InverseFactoryOperator(inverseFac));
#else
   return Teuchos::null; 
#endif
}

RCP<const Epetra_Vector>
Albany::Application::
getInitialSolution() const
{
  const Teuchos::RCP<Epetra_Vector>& initial_x = solMgr->get_initial_x();
  Petra::TpetraVector_To_EpetraVector(this->getInitialSolutionT(), *initial_x, comm);
  return initial_x;
}
#endif

RCP<const Tpetra_Vector>
Albany::Application::
getInitialSolutionT() const
{
  return solMgrT->getInitialSolutionT();
}

#if defined(ALBANY_EPETRA)
RCP<const Epetra_Vector>
Albany::Application::
getInitialSolutionDot() const
{
  const Teuchos::RCP<Epetra_Vector>& initial_x_dot = solMgr->get_initial_xdot();
  Petra::TpetraVector_To_EpetraVector(this->getInitialSolutionDotT(), *initial_x_dot, comm);
  return initial_x_dot;
}
#endif

RCP<const Tpetra_Vector>
Albany::Application::
getInitialSolutionDotT() const
{
  return solMgrT->getInitialSolutionDotT();
}

RCP<ParamLib>
Albany::Application::
getParamLib() const
{
  return paramLib;
}

RCP<DistParamLib>
Albany::Application::
getDistParamLib() const
{
  return distParamLib;
}

int
Albany::Application::
getNumResponses() const {
  return responses.size();
}

Teuchos::RCP<Albany::AbstractResponseFunction>
Albany::Application::
getResponse(int i) const
{
  return responses[i];
}


bool
Albany::Application::
suppliesPreconditioner() const
{
  return physicsBasedPreconditioner;
}

RCP<Stokhos::OrthogPolyExpansion<int,double> >
Albany::Application::
getStochasticExpansion()
{
  return sg_expansion;
}

#ifdef ALBANY_SG
void
Albany::Application::
init_sg(const RCP<const Stokhos::OrthogPolyBasis<int,double> >& basis,
        const RCP<const Stokhos::Quadrature<int,double> >& quad,
        const RCP<Stokhos::OrthogPolyExpansion<int,double> >& expansion,
        const RCP<const EpetraExt::MultiComm>& multiComm)
{

  // Setup stohastic Galerkin
  sg_basis = basis;
  sg_quad = quad;
  sg_expansion = expansion;
  product_comm = multiComm;

  if (sg_overlapped_x == Teuchos::null) {
    sg_overlap_map =
      rcp(new Epetra_LocalMap(sg_basis->size(), 0,
                              product_comm->TimeDomainComm()));
    sg_overlapped_x =
      rcp(new Stokhos::EpetraVectorOrthogPoly(
            sg_basis, sg_overlap_map, disc->getOverlapMap(), product_comm));
    sg_overlapped_xdot =
      rcp(new Stokhos::EpetraVectorOrthogPoly(
            sg_basis, sg_overlap_map, disc->getOverlapMap(), product_comm));
    sg_overlapped_xdotdot =
      rcp(new Stokhos::EpetraVectorOrthogPoly(
            sg_basis, sg_overlap_map, disc->getOverlapMap(), product_comm));
    sg_overlapped_f =
      rcp(new Stokhos::EpetraVectorOrthogPoly(
            sg_basis, sg_overlap_map, disc->getOverlapMap(), product_comm));
    // Delay creation of sg_overlapped_jac until needed
  }

  // Initialize responses
  for (int i=0; i<responses.size(); i++)
    responses[i]->init_sg(basis, quad, expansion, multiComm);
}
#endif

namespace {
//amb-nfm I think right now there is some confusion about nfm. Long ago, nfm was
// like dfm, just a single field manager. Then it became an array like fm. At
// that time, it may have been true that nfm was indexed just like fm, using
// wsPhysIndex. However, it is clear at present (7 Nov 2014) that nfm is
// definitely not indexed like fm. As an example, compare nfm in
// Albany::MechanicsProblem::constructNeumannEvaluators and fm in
// Albany::MechanicsProblem::buildProblem. For now, I'm going to keep nfm as an
// array, but this this new function is a wrapper around the unclear intended
// behavior.
inline Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits> >&
deref_nfm (
  Teuchos::ArrayRCP<Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits> > >& nfm,
  const Albany::WorksetArray<int>::type& wsPhysIndex, int ws)
{
  return
    nfm.size() == 1 ?     // Currently, all problems seem to have one nfm ...
    nfm[0] :              // ... hence this is the intended behavior ...
    nfm[wsPhysIndex[ws]]; // ... and this is not, but may one day be again.
}

// Convenience routine for setting dfm workset data. Cut down on redundant code.
void dfm_set (
  PHAL::Workset& workset,
  const Teuchos::RCP<const Tpetra_Vector>& x,
  const Teuchos::RCP<const Tpetra_Vector>& xd,
  const Teuchos::RCP<const Tpetra_Vector>& xdd,
  Teuchos::RCP<AAdapt::rc::Manager>& rc_mgr)
{
  workset.xT = Teuchos::nonnull(rc_mgr) ? rc_mgr->add_x(x) : x;
  workset.transientTerms = ! Teuchos::nonnull(xd);
  workset.accelerationTerms = ! Teuchos::nonnull(xdd);
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
// accidentaly omission of a FadType, etc.) can cause errors. The common symptom
// of such an error is that the residual is correct, and therefore so is the
// solution, but convergence to the solution is not quadratic.
//   A complementary method to check for errors in the Jacobian is to use
//     Piro -> Jacobian Operator = Matrix-Free,
// which works for Epetra-based problems.
//   Enable this check using the debug block:
//     <ParameterList>
//       <ParameterList name="Debug Output">
//         <Parameter name="Derivative Check" type="int" value="1"/>
void checkDerivatives (Albany::Application& app, const double time,
                       const Teuchos::RCP<const Tpetra_Vector>& xdot,
                       const Teuchos::RCP<const Tpetra_Vector>& xdotdot,
                       const Teuchos::RCP<const Tpetra_Vector>& x,
                       const Teuchos::Array<ParamVec>& p,
                       const Teuchos::RCP<const Tpetra_Vector>& fi,
                       const Teuchos::RCP<const Tpetra_CrsMatrix>& jacobian,
                       const int check_lvl) {
  if (check_lvl <= 0) return;

  // Work vectors. x's map is compatible with f's, so don't distinguish among
  // maps in this function.
  Tpetra_Vector w1(x->getMap()), w2(x->getMap()), w3(x->getMap());

  Teuchos::RCP<Tpetra_MultiVector> mv;
  if (check_lvl > 1)
    mv = Teuchos::rcp(new Tpetra_MultiVector(x->getMap(), 5));

  // Construct a perturbation.
  const double delta = 1e-7;
  Tpetra_Vector& xd = w1;
  xd.randomize();
  Tpetra_Vector& xpd = w2;
  {
    const Teuchos::ArrayRCP<const RealType> x_d = x->getData();
    const Teuchos::ArrayRCP<RealType>
      xd_d = xd.getDataNonConst(), xpd_d = xpd.getDataNonConst();
    for (size_t i = 0; i < x_d.size(); ++i) {
      xd_d[i] = 2*xd_d[i] - 1;
      const double xdi = xd_d[i];
      if (x_d[i] == 0) {
        // No scalar-level way to get the magnitude of x_i, so just go with
        // something:
        xd_d[i] = xpd_d[i] = delta*xd_d[i];
      } else {
        // Make the perturbation meaningful relative to the magnitude of x_i.
        xpd_d[i] = (1 + delta*xd_d[i])*x_d[i]; // mult line
        // Sanitize xd_d.
        xd_d[i] = xpd_d[i] - x_d[i];
        if (xd_d[i] == 0) {
          // Underflow in "mult line" occurred because x_d[i] is something like
          // 1e-314. That's a possible sign of uninitialized memory. However,
          // carry on here to get a good perturbation by reverting to the
          // no-magnitude case:
          xd_d[i] = xpd_d[i] = delta*xd_d[i];
        }
      }
    }
  }
  if (Teuchos::nonnull(mv)) {
    mv->getVectorNonConst(0)->update(1, *x, 0);
    mv->getVectorNonConst(1)->update(1, xd, 0);
  }

  // If necessary, compute f(x).
  Teuchos::RCP<const Tpetra_Vector> f;
  if (fi.is_null()) {
    Teuchos::RCP<Tpetra_Vector>
      w = Teuchos::rcp(new Tpetra_Vector(x->getMap()));
    app.computeGlobalResidualT(time, xdot.get(), xdotdot.get(), *x, p, *w);
    f = w;
  } else {
    f = fi;
  }
  if (Teuchos::nonnull(mv)) mv->getVectorNonConst(2)->update(1, *f, 0);

  // fpd = f(xpd).
  Tpetra_Vector& fpd = w3;
  app.computeGlobalResidualT(time, xdot.get(), xdotdot.get(), xpd, p, fpd);

  // fd = fpd - f.
  Tpetra_Vector& fd = fpd;
  fd.update(-1, *f, 1);
  if (Teuchos::nonnull(mv)) mv->getVectorNonConst(3)->update(1, fd, 0);

  // Jxd = J xd.
  Tpetra_Vector& Jxd = w2;
  jacobian->apply(xd, Jxd);

  // Norms.
  const double fdn = fd.normInf(), Jxdn = Jxd.normInf(), xdn = xd.normInf();
  // d = norm(fd - Jxd).
  Tpetra_Vector& d = fd;
  d.update(-1, Jxd, 1);
  if (Teuchos::nonnull(mv)) mv->getVectorNonConst(4)->update(1, d, 0);
  const double dn = d.normInf();

  // Assess.
  const double
    den = std::max(fdn, Jxdn),
    e = dn / den;
  *Teuchos::VerboseObjectBase::getDefaultOStream()
    << "Albany::Application Check Derivatives level " << check_lvl << ":\n"
    << "   reldif(f(x + dx) - f(x), J(x) dx) = " << e
    << ",\n which should be on the order of " << xdn << "\n";

  if (Teuchos::nonnull(mv)) {
    static int ctr = 0;
    std::stringstream ss;
    ss << "dc" << ctr << ".mm";
    Tpetra_MatrixMarket_Writer::writeDenseFile(ss.str(), mv);
    ++ctr;
  }
}
} // namespace

void
Albany::Application::
computeGlobalResidualImplT(
    const double current_time,
    const Teuchos::RCP<const Tpetra_Vector>& xdotT,
    const Teuchos::RCP<const Tpetra_Vector>& xdotdotT,
    const Teuchos::RCP<const Tpetra_Vector>& xT,
    const Teuchos::Array<ParamVec>& p,
    const Teuchos::RCP<Tpetra_Vector>& fT)
{
  TEUCHOS_FUNC_TIME_MONITOR("> Albany Fill: Residual");
  postRegSetup("Residual");

  // Load connectivity map and coordinates
  const WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > > >::type&
        wsElNodeEqID = disc->getWsElNodeEqID();
  const WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >::type&
        coords = disc->getCoords();
  const WorksetArray<std::string>::type& wsEBNames = disc->getWsEBNames();
  const WorksetArray<int>::type& wsPhysIndex = disc->getWsPhysIndex();

  int numWorksets = wsElNodeEqID.size();

  Teuchos::RCP<Tpetra_Vector> overlapped_fT = solMgrT->get_overlapped_fT();
  Teuchos::RCP<Tpetra_Export> exporterT = solMgrT->get_exporterT();

  // Scatter x and xdot to the overlapped distrbution
  solMgrT->scatterXT(*xT, xdotT.get(), xdotdotT.get());
  
  //Scatter distributed parameters
  distParamLib->scatter();

  // Set parameters
  for (int i=0; i<p.size(); i++)
    for (unsigned int j=0; j<p[i].size(); j++)
      p[i][j].family->setRealValueForAllTypes(p[i][j].baseValue);

  // Mesh motion needs to occur here on the global mesh befor
  // it is potentially carved into worksets.
#ifdef ALBANY_CUTR
  static int first=true;
  if (shapeParamsHaveBeenReset) {
    TEUCHOS_FUNC_TIME_MONITOR("Albany-Cubit MeshMover");

*out << " Calling moveMesh with params: " << std::setprecision(8);
 for (unsigned int i=0; i<shapeParams.size(); i++) *out << shapeParams[i] << "  ";
*out << std::endl;
    meshMover->moveMesh(shapeParams, morphFromInit);
    coords = disc->getCoords();
    shapeParamsHaveBeenReset = false;
  }
#endif

  // Zero out overlapped residual - Tpetra
  overlapped_fT->putScalar(0.0);
  fT->putScalar(0.0);

#ifdef ALBANY_PERIDIGM 
#if defined(ALBANY_EPETRA)
  const Teuchos::RCP<LCM::PeridigmManager>&
    peridigmManager = LCM::PeridigmManager::self();
  if (Teuchos::nonnull(peridigmManager)) {
    peridigmManager->setCurrentTimeAndDisplacement(current_time, xT);
    peridigmManager->evaluateInternalForce();
  }
#endif
#endif

  // Set data in Workset struct, and perform fill via field manager
  {
    if (Teuchos::nonnull(rc_mgr)) rc_mgr->init_x_if_not(xT->getMap());

    PHAL::Workset workset;

    if (!paramLib->isParameter("Time"))
      loadBasicWorksetInfoT( workset, current_time );
    else
      loadBasicWorksetInfoT( workset,
                             paramLib->getRealValue<PHAL::AlbanyTraits::Residual>("Time") );
    workset.fT = overlapped_fT;

    for (int ws=0; ws < numWorksets; ws++) {
      loadWorksetBucketInfo<PHAL::AlbanyTraits::Residual>(workset, ws);

      // FillType template argument used to specialize Sacado
      fm[wsPhysIndex[ws]]->evaluateFields<PHAL::AlbanyTraits::Residual>(workset);
      if (nfm!=Teuchos::null)
         deref_nfm(nfm, wsPhysIndex, ws)->evaluateFields<PHAL::AlbanyTraits::Residual>(workset);
    }
  // workset.wsElNodeEqID_kokkos =Kokkos:: View<int****, PHX::Device ("wsElNodeEqID_kokkos",workset. wsElNodeEqID.size(), workset. wsElNodeEqID[0].size(), workset. wsElNodeEqID[0][0].size());
  }

  fT->doExport(*overlapped_fT, *exporterT, Tpetra::ADD);

  disc->setResidualFieldT(*fT);

  // Apply Dirichlet conditions using dfm (Dirchelt Field Manager)
  if (dfm!=Teuchos::null) {
    PHAL::Workset workset;

    workset.fT = fT;
    loadWorksetNodesetInfo(workset);
    dfm_set(workset, xT, xdotT, xdotdotT, rc_mgr);
    if ( paramLib->isParameter("Time") )
      workset.current_time = paramLib->getRealValue<PHAL::AlbanyTraits::Residual>("Time");
    else
      workset.current_time = current_time;
    workset.distParamLib = distParamLib;
    workset.disc = disc;

#if defined(ALBANY_LCM)
    // Needed for more specialized Dirichlet BCs (e.g. Schwarz coupling)
    workset.apps_ = apps_;
    workset.current_app_ = Teuchos::rcp(this, false);
#endif

    // FillType template argument used to specialize Sacado
    dfm->evaluateFields<PHAL::AlbanyTraits::Residual>(workset);
  }
}

#if defined(ALBANY_EPETRA)
void
Albany::Application::
computeGlobalResidual(const double current_time,
		      const Epetra_Vector* xdot,
		      const Epetra_Vector* xdotdot,
		      const Epetra_Vector& x,
		      const Teuchos::Array<ParamVec>& p,
		      Epetra_Vector& f)
{
  // Scatter x and xdot to the overlapped distribution
  solMgr->scatterX(x, xdot, xdotdot);

  // Create Tpetra copies of Epetra arguments
  // Names of Tpetra entitied are identified by the suffix T
  const Teuchos::RCP<const Tpetra_Vector> xT =
    Petra::EpetraVector_To_TpetraVectorConst(x, commT);

  Teuchos::RCP<const Tpetra_Vector> xdotT;
  if (xdot != NULL) {
     xdotT = Petra::EpetraVector_To_TpetraVectorConst(*xdot, commT);
  }

  Teuchos::RCP<const Tpetra_Vector> xdotdotT;
  if (xdotdot != NULL) {
     xdotdotT = Petra::EpetraVector_To_TpetraVectorConst(*xdotdot, commT);
  }

  const Teuchos::RCP<Tpetra_Vector> fT =
    Petra::EpetraVector_To_TpetraVectorNonConst(f, commT);

  this->computeGlobalResidualImplT(current_time, xdotT, xdotdotT, xT, p, fT);

  // Convert output back from Tpetra to Epetra
  Petra::TpetraVector_To_EpetraVector(fT, f, comm);
  //cout << "Global Resid f\n" << f << std::endl;
  //std::cout << "Global Soln x\n" << x << std::endl;

  //Debut output
  if (writeToMatrixMarketRes != 0) { //If requesting writing to MatrixMarket of residual...
    char name[100];  //create string for file name
    if (writeToMatrixMarketRes == -1) { //write residual to MatrixMarket every time it arises
       sprintf(name, "rhs%i.mm", countRes);
       EpetraExt::MultiVectorToMatrixMarketFile(name, f);
    }
    else {
      if (countRes == writeToMatrixMarketRes) { //write residual only at requested count#
        sprintf(name, "rhs%i.mm", countRes);
        EpetraExt::MultiVectorToMatrixMarketFile(name, f);
      }
    }
  }
  if (writeToCoutRes != 0) { //If requesting writing of residual to cout...
    if (writeToCoutRes == -1) { //cout residual time it arises
       std::cout << "Global Residual #" << countRes << ": " << std::endl;
       std::cout << f << std::endl;
    }
    else {
      if (countRes == writeToCoutRes) { //cout residual only at requested count#
        std::cout << "Global Residual #" << countRes << ": " << std::endl;
        std::cout << f << std::endl;
      }
    }
  }
  if (writeToMatrixMarketRes != 0 || writeToCoutRes != 0)
    countRes++;  //increment residual counter
}
#endif

void
Albany::Application::
computeGlobalResidualT(
    const double current_time,
    const Tpetra_Vector* xdotT,
    const Tpetra_Vector* xdotdotT,
    const Tpetra_Vector& xT,
    const Teuchos::Array<ParamVec>& p,
    Tpetra_Vector& fT)
{
  // Create non-owning RCPs to Tpetra objects
  // to be passed to the implementation
  this->computeGlobalResidualImplT(
      current_time,
      Teuchos::rcp(xdotT, false),
      Teuchos::rcp(xdotdotT, false),
      Teuchos::rcpFromRef(xT),
      p,
      Teuchos::rcpFromRef(fT));

  //Debut output
  if (writeToMatrixMarketRes != 0) { //If requesting writing to MatrixMarket of residual...
    char name[100];  //create string for file name
    if (writeToMatrixMarketRes == -1) { //write residual to MatrixMarket every time it arises
      sprintf(name, "rhs%i.mm", countRes);
      Tpetra_MatrixMarket_Writer::writeDenseFile(name, Teuchos::rcpFromRef(fT));
    }
    else {
      if (countRes == writeToMatrixMarketRes) { //write residual only at requested count#
        sprintf(name, "rhs%i.mm", countRes);
        Tpetra_MatrixMarket_Writer::writeDenseFile(
            name,
            Teuchos::rcpFromRef(fT));
      }
    }
  }
  if (writeToCoutRes != 0) { //If requesting writing of residual to cout...
    if (writeToCoutRes == -1) { //cout residual time it arises
      std::cout << "Global Residual #" << countRes << ": " << std::endl;
      fT.describe(*out, Teuchos::VERB_EXTREME);
    }
    else {
      if (countRes == writeToCoutRes) { //cout residual only at requested count#
        std::cout << "Global Residual #" << countRes << ": " << std::endl;
        fT.describe(*out, Teuchos::VERB_EXTREME);
      }
    }
  }
  if (writeToMatrixMarketRes != 0 || writeToCoutRes != 0) {
    countRes++;  //increment residual counter
  }
}

void
Albany::Application::
computeGlobalJacobianImplT(const double alpha,
		           const double beta,
		           const double omega,
                           const double current_time,
                           const Teuchos::RCP<const Tpetra_Vector>& xdotT,
                           const Teuchos::RCP<const Tpetra_Vector>& xdotdotT,
                           const Teuchos::RCP<const Tpetra_Vector>& xT,
                           const Teuchos::Array<ParamVec>& p,
                           const Teuchos::RCP<Tpetra_Vector>& fT,
                           const Teuchos::RCP<Tpetra_CrsMatrix>& jacT)
{
  TEUCHOS_FUNC_TIME_MONITOR("> Albany Fill: Jacobian");

  postRegSetup("Jacobian");

  // Load connectivity map and coordinates
  const WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > > >::type&
        wsElNodeEqID = disc->getWsElNodeEqID();
  const WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >::type&
        coords = disc->getCoords();
  const WorksetArray<std::string>::type& wsEBNames = disc->getWsEBNames();
  const WorksetArray<int>::type& wsPhysIndex = disc->getWsPhysIndex();

  int numWorksets = wsElNodeEqID.size();

  Teuchos::RCP<Tpetra_Vector>& overlapped_fT = solMgrT->get_overlapped_fT();
  Teuchos::RCP<Tpetra_CrsMatrix>& overlapped_jacT = solMgrT->get_overlapped_jacT();
  Teuchos::RCP<Tpetra_Export>& exporterT = solMgrT->get_exporterT();

  // Scatter x and xdot to the overlapped distribution
  solMgrT->scatterXT(*xT, xdotT.get(), xdotdotT.get());

  // Scatter distributed parameters
  distParamLib->scatter();

  // Set parameters
  for (int i=0; i<p.size(); i++)
    for (unsigned int j=0; j<p[i].size(); j++)
      p[i][j].family->setRealValueForAllTypes(p[i][j].baseValue);

#ifdef ALBANY_CUTR
  if (shapeParamsHaveBeenReset) {
    TEUCHOS_FUNC_TIME_MONITOR("Albany-Cubit MeshMover");

*out << " Calling moveMesh with params: " << std::setprecision(8);
 for (unsigned int i=0; i<shapeParams.size(); i++) *out << shapeParams[i] << "  ";
*out << std::endl;
    meshMover->moveMesh(shapeParams, morphFromInit);
    coords = disc->getCoords();
    shapeParamsHaveBeenReset = false;
  }
#endif

  // Zero out overlapped residual
  if (Teuchos::nonnull(fT)) {
    overlapped_fT->putScalar(0.0);
    fT->putScalar(0.0);
  }

  // Zero out Jacobian
  overlapped_jacT->setAllToScalar(0.0);
  jacT->resumeFill();
  jacT->setAllToScalar(0.0);

  // Set data in Workset struct, and perform fill via field manager
  {
    PHAL::Workset workset;
    if (!paramLib->isParameter("Time")) {
      loadBasicWorksetInfoT( workset, current_time );
    }
    else {
      loadBasicWorksetInfoT( workset,
			    paramLib->getRealValue<PHAL::AlbanyTraits::Residual>("Time") );
    }

    workset.fT        = overlapped_fT;
    workset.JacT      = overlapped_jacT;
    loadWorksetJacobianInfo(workset, alpha, beta, omega);

    for (int ws=0; ws < numWorksets; ws++) {
      loadWorksetBucketInfo<PHAL::AlbanyTraits::Jacobian>(workset, ws);

      // FillType template argument used to specialize Sacado
      fm[wsPhysIndex[ws]]->evaluateFields<PHAL::AlbanyTraits::Jacobian>(workset);
      if (Teuchos::nonnull(nfm))
        deref_nfm(nfm, wsPhysIndex, ws)->evaluateFields<PHAL::AlbanyTraits::Jacobian>(workset);
    }
  }

  { TEUCHOS_FUNC_TIME_MONITOR("> Albany Fill: Jacobian Export");
  // Assemble global residual
  if (Teuchos::nonnull(fT)) {
    fT->doExport(*overlapped_fT, *exporterT, Tpetra::ADD);
  }

  // Assemble global Jacobian
  jacT->doExport(*overlapped_jacT, *exporterT, Tpetra::ADD);
  } // End timer

  // Apply Dirichlet conditions using dfm (Dirchelt Field Manager)
  if (Teuchos::nonnull(dfm)) {
    PHAL::Workset workset;

    workset.fT = fT;
    workset.JacT = jacT;
    workset.m_coeff = alpha;
    workset.n_coeff = omega;
    workset.j_coeff = beta;

    if ( paramLib->isParameter("Time") )
      workset.current_time = paramLib->getRealValue<PHAL::AlbanyTraits::Residual>("Time");
    else
      workset.current_time = current_time;

    if (beta==0.0 && perturbBetaForDirichlets>0.0) workset.j_coeff = perturbBetaForDirichlets;

    dfm_set(workset, xT, xdotT, xdotdotT, rc_mgr);

    loadWorksetNodesetInfo(workset);
    workset.distParamLib = distParamLib;

    workset.disc = disc;

#if defined(ALBANY_LCM)
    // Needed for more specialized Dirichlet BCs (e.g. Schwarz coupling)
    workset.apps_ = apps_;
    workset.current_app_ = Teuchos::rcp(this, false);
#endif

    // FillType template argument used to specialize Sacado
    dfm->evaluateFields<PHAL::AlbanyTraits::Jacobian>(workset);
  }

  jacT->fillComplete();

  if (derivatives_check_ > 0)
    checkDerivatives(*this, current_time, xdotT, xdotdotT, xT, p, fT, jacT,
                     derivatives_check_);
}

#if defined(ALBANY_EPETRA)
void
Albany::Application::
computeGlobalJacobian(const double alpha,
		      const double beta,
		      const double omega,
		      const double current_time,
		      const Epetra_Vector* xdot,
		      const Epetra_Vector* xdotdot,
		      const Epetra_Vector& x,
		      const Teuchos::Array<ParamVec>& p,
		      Epetra_Vector* f,
		      Epetra_CrsMatrix& jac)
{
  // Scatter x and xdot to the overlapped distribution
  solMgr->scatterX(x, xdot, xdotdot);

  // Create Tpetra copies of Epetra arguments
  // Names of Tpetra entitied are identified by the suffix T
  const Teuchos::RCP<const Tpetra_Vector> xT =
    Petra::EpetraVector_To_TpetraVectorConst(x, commT);

  Teuchos::RCP<const Tpetra_Vector> xdotT;
  if (xdot != NULL) {
    xdotT = Petra::EpetraVector_To_TpetraVectorConst(*xdot, commT);
   }

  Teuchos::RCP<const Tpetra_Vector> xdotdotT;
  if (xdotdot != NULL) {
    xdotdotT = Petra::EpetraVector_To_TpetraVectorConst(*xdotdot, commT);
   }

  Teuchos::RCP<Tpetra_Vector> fT;
  if (f != NULL) {
    fT = Petra::EpetraVector_To_TpetraVectorNonConst(*f, commT);
  }

  const Teuchos::RCP<Tpetra_CrsMatrix> jacT =
    Petra::EpetraCrsMatrix_To_TpetraCrsMatrix(jac, commT);

  this->computeGlobalJacobianImplT(alpha, beta, omega, current_time, xdotT, xdotdotT, xT, p, fT, jacT);

  // Convert output back from Tpetra to Epetra
  if (f != NULL) {
    Petra::TpetraVector_To_EpetraVector(fT, *f, comm);
  }

  Petra::TpetraCrsMatrix_To_EpetraCrsMatrix(jacT, jac, comm);
  jac.FillComplete(true);
  //std::cout << "f " << *f << std::endl;;
  //std::cout << "J " << jac << std::endl;;

 //Debut output
  if (writeToMatrixMarketJac != 0) { //If requesting writing to MatrixMarket of Jacobian...
    char name[100];  //create string for file name
    if (writeToMatrixMarketJac == -1) { //write jacobian to MatrixMarket every time it arises
       sprintf(name, "jac%i.mm", countJac);
       EpetraExt::RowMatrixToMatrixMarketFile(name, jac);
    }
    else {
      if (countJac == writeToMatrixMarketJac) { //write jacobian only at requested count#
        sprintf(name, "jac%i.mm", countJac);
        EpetraExt::RowMatrixToMatrixMarketFile(name, jac);
      }
    }
  }
  if (writeToCoutJac != 0) { //If requesting writing Jacobian to standard output (cout)...
    if (writeToCoutJac == -1) { //cout jacobian every time it arises
       std::cout << "Global Jacobian #" << countJac << ": " << std::endl;
       std::cout << jac << std::endl;
    }
    else {
      if (countJac == writeToCoutJac) { //cout jacobian only at requested count#
       std::cout << "Global Jacobian #" << countJac << ": " << std::endl;
       std::cout << jac << std::endl;
      }
    }
  }
  if (writeToMatrixMarketJac != 0 || writeToCoutJac != 0)
    countJac++; //increment Jacobian counter
}
#endif

void
Albany::Application::
computeGlobalJacobianT(
    const double alpha,
    const double beta,
    const double omega,
    const double current_time,
    const Tpetra_Vector* xdotT,
    const Tpetra_Vector* xdotdotT,
    const Tpetra_Vector& xT,
    const Teuchos::Array<ParamVec>& p,
    Tpetra_Vector* fT,
    Tpetra_CrsMatrix& jacT)
{
  // Create non-owning RCPs to Tpetra objects
  // to be passed to the implementation
  this->computeGlobalJacobianImplT(
      alpha,
      beta,
      omega,
      current_time,
      Teuchos::rcp(xdotT, false),
      Teuchos::rcp(xdotdotT, false),
      Teuchos::rcpFromRef(xT),
      p,
      Teuchos::rcp(fT, false),
      Teuchos::rcpFromRef(jacT));
  //Debut output
  if (writeToMatrixMarketJac != 0) { //If requesting writing to MatrixMarket of Jacobian...
    char name[100];  //create string for file name
    if (writeToMatrixMarketJac == -1) { //write jacobian to MatrixMarket every time it arises
      sprintf(name, "jac%i.mm", countJac);
      Tpetra_MatrixMarket_Writer::writeSparseFile(
          name,
          Teuchos::rcpFromRef(jacT));
    }
    else {
      if (countJac == writeToMatrixMarketJac) { //write jacobian only at requested count#
        sprintf(name, "jac%i.mm", countJac);
        Tpetra_MatrixMarket_Writer::writeSparseFile(
            name,
            Teuchos::rcpFromRef(jacT));
      }
    }
  }
  Teuchos::RCP<Teuchos::FancyOStream> out = fancyOStream(rcpFromRef(std::cout));
  if (writeToCoutJac != 0) { //If requesting writing Jacobian to standard output (cout)...
    if (writeToCoutJac == -1) { //cout jacobian every time it arises
      std::cout << "Global Jacobian #" << countJac << ": " << std::endl;
      jacT.describe(*out, Teuchos::VERB_HIGH);
    }
    else {
      if (countJac == writeToCoutJac) { //cout jacobian only at requested count#
        std::cout << "Global Jacobian #" << countJac << ": " << std::endl;
        jacT.describe(*out, Teuchos::VERB_HIGH);
      }
    }
  }
  if (writeToMatrixMarketJac != 0 || writeToCoutJac != 0) {
    countJac++; //increment Jacobian counter
  }
}

#if defined(ALBANY_EPETRA)
void
Albany::Application::
computeGlobalPreconditioner(const RCP<Epetra_CrsMatrix>& jac,
                            const RCP<Epetra_Operator>& prec)
{
#if defined(ALBANY_TEKO)
  TEUCHOS_FUNC_TIME_MONITOR("> Albany Fill: Precond");

  *out << "Computing WPrec by Teko" << std::endl;

  RCP<Teko::Epetra::InverseFactoryOperator> blockPrec
    = rcp_dynamic_cast<Teko::Epetra::InverseFactoryOperator>(prec);

  blockPrec->initInverse();

  wrappedJac = buildWrappedOperator(jac, wrappedJac);
  blockPrec->rebuildInverseOperator(wrappedJac);
#endif
}
#endif

void
Albany::Application::
computeGlobalTangentImplT(
    const double alpha,
    const double beta,
    const double omega,
    const double current_time,
    bool sum_derivs,
    const Teuchos::RCP<const Tpetra_Vector>& xdotT,
    const Teuchos::RCP<const Tpetra_Vector>& xdotdotT,
    const Teuchos::RCP<const Tpetra_Vector>& xT,
    const Teuchos::Array<ParamVec>& par,
    ParamVec* deriv_par,
    const Teuchos::RCP<const Tpetra_MultiVector>& VxT,
    const Teuchos::RCP<const Tpetra_MultiVector>& VxdotT,
    const Teuchos::RCP<const Tpetra_MultiVector>& VxdotdotT,
    const Teuchos::RCP<const Tpetra_MultiVector>& VpT,
    const Teuchos::RCP<Tpetra_Vector>& fT,
    const Teuchos::RCP<Tpetra_MultiVector>& JVT,
    const Teuchos::RCP<Tpetra_MultiVector>& fpT)
{
  TEUCHOS_FUNC_TIME_MONITOR("> Albany Fill: Tangent");

  postRegSetup("Tangent");

  // Load connectivity map and coordinates
  const WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > > >::type&
        wsElNodeEqID = disc->getWsElNodeEqID();
  const WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >::type&
        coords = disc->getCoords();
  const WorksetArray<std::string>::type& wsEBNames = disc->getWsEBNames();
  const WorksetArray<int>::type& wsPhysIndex = disc->getWsPhysIndex();

  int numWorksets = wsElNodeEqID.size();

  Teuchos::RCP<Tpetra_Vector>& overlapped_fT = solMgrT->get_overlapped_fT();
  Teuchos::RCP<Tpetra_Import>& importerT = solMgrT->get_importerT();
  Teuchos::RCP<Tpetra_Export>& exporterT = solMgrT->get_exporterT();

  // Scatter x and xdot to the overlapped distrbution
  solMgrT->scatterXT(*xT, xdotT.get(), xdotdotT.get());

  // Scatter distributed parameters
  distParamLib->scatter();

  // Scatter Vx to the overlapped distribution
  RCP<Tpetra_MultiVector> overlapped_VxT;
  if (Teuchos::nonnull(VxT)) {
    overlapped_VxT =
      rcp(new Tpetra_MultiVector(disc->getOverlapMapT(), VxT->getNumVectors()));
    overlapped_VxT->doImport(*VxT, *importerT, Tpetra::INSERT);
  }

  // Scatter Vxdot to the overlapped distribution
  RCP<Tpetra_MultiVector> overlapped_VxdotT;
  if (Teuchos::nonnull(VxdotT)) {
    overlapped_VxdotT =
      rcp(new Tpetra_MultiVector(disc->getOverlapMapT(), VxdotT->getNumVectors()));
    overlapped_VxdotT->doImport(*VxdotT, *importerT, Tpetra::INSERT);
  }
  RCP<Tpetra_MultiVector> overlapped_VxdotdotT;
  if (Teuchos::nonnull(VxdotdotT)) {
    overlapped_VxdotdotT = rcp(new Tpetra_MultiVector(disc->getOverlapMapT(), VxdotdotT->getNumVectors()));
    overlapped_VxdotdotT->doImport(*VxdotdotT, *importerT, Tpetra::INSERT);
  }

  // Set parameters
  for (int i=0; i<par.size(); i++)
    for (unsigned int j=0; j<par[i].size(); j++)
      par[i][j].family->setRealValueForAllTypes(par[i][j].baseValue);

  RCP<ParamVec> params = rcp(deriv_par, false);

  // Zero out overlapped residual
  if (Teuchos::nonnull(fT)) {
    overlapped_fT->putScalar(0.0);
    fT->putScalar(0.0);
  }

  RCP<Tpetra_MultiVector> overlapped_JVT;
  if (Teuchos::nonnull(JVT)) {
    overlapped_JVT =
      rcp(new Tpetra_MultiVector(disc->getOverlapMapT(), JVT->getNumVectors()));
    overlapped_JVT->putScalar(0.0);
    JVT->putScalar(0.0);
  }

  RCP<Tpetra_MultiVector> overlapped_fpT;
  if (Teuchos::nonnull(fpT)) {
    overlapped_fpT =
      rcp(new Tpetra_MultiVector(disc->getOverlapMapT(), fpT->getNumVectors()));
    overlapped_fpT->putScalar(0.0);
    fpT->putScalar(0.0);
  }

  // Number of x & xdot tangent directions
  int num_cols_x = 0;
  if (Teuchos::nonnull(VxT))
    num_cols_x = VxT->getNumVectors();
  else if (Teuchos::nonnull(VxdotT))
    num_cols_x = VxdotT->getNumVectors();
  else if (Teuchos::nonnull(VxdotdotT))
    num_cols_x = VxdotdotT->getNumVectors();

  // Number of parameter tangent directions
  int num_cols_p = 0;
  if (Teuchos::nonnull(params)) {
    if (Teuchos::nonnull(VpT)) {
      num_cols_p = VpT->getNumVectors();
    }
    else
      num_cols_p = params->size();
  }

  // Whether x and param tangent components are added or separate
  int param_offset = 0;
  if (!sum_derivs)
    param_offset = num_cols_x;  // offset of parameter derivs in deriv array

  TEUCHOS_TEST_FOR_EXCEPTION(sum_derivs &&
                             (num_cols_x != 0) &&
                             (num_cols_p != 0) &&
                             (num_cols_x != num_cols_p),
                             std::logic_error,
                             "Seed matrices Vx and Vp must have the same number " <<
                             " of columns when sum_derivs is true and both are "
                             << "non-null!" << std::endl);

  // Initialize
  if (Teuchos::nonnull(params)) {
    TanFadType p;
    int num_cols_tot = param_offset + num_cols_p;
    for (unsigned int i=0; i<params->size(); i++) {
      p = TanFadType(num_cols_tot, (*params)[i].baseValue);
      if (Teuchos::nonnull(VpT)) {
        //ArrayRCP for const view of Vp's vectors
        Teuchos::ArrayRCP<const ST> VpT_constView;
        for (int k=0; k<num_cols_p; k++) {
          VpT_constView = VpT->getData(k);
          p.fastAccessDx(param_offset+k) = VpT_constView[i];  //CHANGE TO TPETRA!
         }
      }
      else
        p.fastAccessDx(param_offset+i) = 1.0;
      (*params)[i].family->setValue<PHAL::AlbanyTraits::Tangent>(p);
    }
  }

  // Begin shape optimization logic
  ArrayRCP<ArrayRCP<double> > coord_derivs;
  // ws, sp, cell, node, dim
  ArrayRCP<ArrayRCP<ArrayRCP<ArrayRCP<ArrayRCP<double> > > > > ws_coord_derivs;
  ws_coord_derivs.resize(coords.size());
  std::vector<int> coord_deriv_indices;
#ifdef ALBANY_CUTR
  if (shapeParamsHaveBeenReset) {
    TEUCHOS_FUNC_TIME_MONITOR("Albany-Cubit MeshMover");

     int num_sp = 0;
     std::vector<int> shape_param_indices;

     // Find any shape params from param list
     for (unsigned int i=0; i<params->size(); i++) {
       for (unsigned int j=0; j<shapeParamNames.size(); j++) {
         if ((*params)[i].family->getName() == shapeParamNames[j]) {
           num_sp++;
           coord_deriv_indices.resize(num_sp);
           shape_param_indices.resize(num_sp);
           coord_deriv_indices[num_sp-1] = i;
           shape_param_indices[num_sp-1] = j;
         }
       }
     }

    TEUCHOS_TEST_FOR_EXCEPTION( Teuchos::nonnull(VpT), std::logic_error,
				"Derivatives with respect to a vector of shape\n " <<
				"parameters has not been implemented. Need to write\n" <<
				"directional derivative perturbation through meshMover!" <<
				std::endl);

     // Compute FD derivs of coordinate vector w.r.t. shape params
     double eps = 1.0e-4;
     double pert;
     coord_derivs.resize(num_sp);
     for (int ws=0; ws<coords.size(); ws++)  ws_coord_derivs[ws].resize(num_sp);
     for (int i=0; i<num_sp; i++) {
*out << "XXX perturbing parameter " << coord_deriv_indices[i]
     << " which is shapeParam # " << shape_param_indices[i]
     << " with name " <<  shapeParamNames[shape_param_indices[i]]
     << " which should equal " << (*params)[coord_deriv_indices[i]].family->getName() << std::endl;

     pert = (fabs(shapeParams[shape_param_indices[i]]) + 1.0e-2) * eps;

       shapeParams[shape_param_indices[i]] += pert;
*out << " Calling moveMesh with params: " << std::setprecision(8);
for (unsigned int ii=0; ii<shapeParams.size(); ii++) *out << shapeParams[ii] << "  ";
*out << std::endl;
       meshMover->moveMesh(shapeParams, morphFromInit);
       for (int ws=0; ws<coords.size(); ws++) {  //worset
         ws_coord_derivs[ws][i].resize(coords[ws].size());
         for (int e=0; e<coords[ws].size(); e++) { //cell
           ws_coord_derivs[ws][i][e].resize(coords[ws][e].size());
           for (int j=0; j<coords[ws][e].size(); j++) { //node
             ws_coord_derivs[ws][i][e][j].resize(disc->getNumDim());
             for (int d=0; d<disc->getNumDim(); d++)  //node
                ws_coord_derivs[ws][i][e][j][d] = coords[ws][e][j][d];
       } } } }

       shapeParams[shape_param_indices[i]] -= pert;
     }
*out << " Calling moveMesh with params: " << std::setprecision(8);
for (unsigned int i=0; i<shapeParams.size(); i++) *out << shapeParams[i] << "  ";
*out << std::endl;
     meshMover->moveMesh(shapeParams, morphFromInit);
     coords = disc->getCoords();

     for (int i=0; i<num_sp; i++) {
       for (int ws=0; ws<coords.size(); ws++)  //worset
         for (int e=0; e<coords[ws].size(); e++)  //cell
           for (int j=0; j<coords[ws][i].size(); j++)  //node
             for (int d=0; d<disc->getNumDim; d++)  //node
                ws_coord_derivs[ws][i][e][j][d] = (ws_coord_derivs[ws][i][e][j][d] - coords[ws][e][j][d]) / pert;
       }
     }
     shapeParamsHaveBeenReset = false;
  }
  // End shape optimization logic
#endif

  // Set data in Workset struct, and perform fill via field manager
  {
    PHAL::Workset workset;
    if (!paramLib->isParameter("Time")) {
      loadBasicWorksetInfoT( workset, current_time );
    }
    else {
      loadBasicWorksetInfoT( workset,
			    paramLib->getRealValue<PHAL::AlbanyTraits::Residual>("Time") );
    }

    workset.params = params;
    workset.VxT = overlapped_VxT;
    workset.VxdotT = overlapped_VxdotT;
    workset.VxdotdotT = overlapped_VxdotdotT;
    workset.VpT = VpT;

    workset.fT            = overlapped_fT;
    workset.JVT           = overlapped_JVT;
    workset.fpT           = overlapped_fpT;
    workset.j_coeff      = beta;
    workset.m_coeff      = alpha;
    workset.n_coeff      = omega;

    workset.num_cols_x = num_cols_x;
    workset.num_cols_p = num_cols_p;
    workset.param_offset = param_offset;

    workset.coord_deriv_indices = &coord_deriv_indices;

    for (int ws=0; ws < numWorksets; ws++) {
      loadWorksetBucketInfo<PHAL::AlbanyTraits::Tangent>(workset, ws);
      workset.ws_coord_derivs = ws_coord_derivs[ws];

      // FillType template argument used to specialize Sacado
      fm[wsPhysIndex[ws]]->evaluateFields<PHAL::AlbanyTraits::Tangent>(workset);
      if (nfm!=Teuchos::null)
        deref_nfm(nfm, wsPhysIndex, ws)->evaluateFields<PHAL::AlbanyTraits::Tangent>(workset);
    }
  }

  params = Teuchos::null;

  // Assemble global residual
  if (Teuchos::nonnull(fT)) {
    fT->doExport(*overlapped_fT, *exporterT, Tpetra::ADD);
  }

  // Assemble derivatives
  if (Teuchos::nonnull(JVT)) {
    JVT->doExport(*overlapped_JVT, *exporterT, Tpetra::ADD);
  }
  if (Teuchos::nonnull(fpT)) {
    fpT->doExport(*overlapped_fpT, *exporterT, Tpetra::ADD);
  }

  // Apply Dirichlet conditions using dfm (Dirchelt Field Manager)
  if (Teuchos::nonnull(dfm)) {
    PHAL::Workset workset;

    workset.num_cols_x = num_cols_x;
    workset.num_cols_p = num_cols_p;
    workset.param_offset = param_offset;

    workset.fT = fT;
    workset.fpT = fpT;
    workset.JVT = JVT;
    workset.j_coeff = beta;
    workset.n_coeff = omega;
    workset.VxT = VxT;
    dfm_set(workset, xT, xdotT, xdotdotT, rc_mgr);

    loadWorksetNodesetInfo(workset);
    workset.distParamLib = distParamLib;

    if ( paramLib->isParameter("Time") )
      workset.current_time = paramLib->getRealValue<PHAL::AlbanyTraits::Residual>("Time");
    else
      workset.current_time = current_time;

    workset.disc = disc;

#if defined(ALBANY_LCM)
    // Needed for more specialized Dirichlet BCs (e.g. Schwarz coupling)
    workset.apps_ = apps_;
    workset.current_app_ = Teuchos::rcp(this, false);
#endif

    // FillType template argument used to specialize Sacado
    dfm->evaluateFields<PHAL::AlbanyTraits::Tangent>(workset);
  }
}

#if defined(ALBANY_EPETRA)
void
Albany::Application::
computeGlobalTangent(const double alpha,
		     const double beta,
		     const double omega,
		     const double current_time,
		     bool sum_derivs,
		     const Epetra_Vector* xdot,
		     const Epetra_Vector* xdotdot,
		     const Epetra_Vector& x,
		     const Teuchos::Array<ParamVec>& par,
		     ParamVec* deriv_par,
		     const Epetra_MultiVector* Vx,
		     const Epetra_MultiVector* Vxdot,
		     const Epetra_MultiVector* Vxdotdot,
		     const Epetra_MultiVector* Vp,
		     Epetra_Vector* f,
		     Epetra_MultiVector* JV,
		     Epetra_MultiVector* fp)
{
  // Scatter x and xdot to the overlapped distribution
  solMgr->scatterX(x, xdot, xdotdot);

  // Create Tpetra copies of Epetra arguments
  // Names of Tpetra entitied are identified by the suffix T
  Teuchos::RCP<const Tpetra_Vector> xT =
    Petra::EpetraVector_To_TpetraVectorConst(x, commT);

  Teuchos::RCP<const Tpetra_Vector> xdotT;
  if (xdot != NULL) {
    xdotT = Petra::EpetraVector_To_TpetraVectorConst(*xdot, commT);
  }
  Teuchos::RCP<const Tpetra_Vector> xdotdotT;
  if (xdotdot != NULL) {
    xdotdotT = Petra::EpetraVector_To_TpetraVectorConst(*xdotdot, commT);
  }

  Teuchos::RCP<const Tpetra_MultiVector> VxT;
  if (Vx != NULL) {
    VxT = Petra::EpetraMultiVector_To_TpetraMultiVector(*Vx, commT);
  }

  RCP<const Tpetra_MultiVector> VxdotT;
  if (Vxdot != NULL)
    VxdotT = Petra::EpetraMultiVector_To_TpetraMultiVector(*Vxdot, commT);

  RCP<const Tpetra_MultiVector> VxdotdotT;
  if (Vxdotdot != NULL)
    VxdotdotT = Petra::EpetraMultiVector_To_TpetraMultiVector(*Vxdotdot, commT);

  RCP<const Tpetra_MultiVector> VpT;
  if (Vp != NULL)
    VpT = Petra::EpetraMultiVector_To_TpetraMultiVector(*Vp, commT);

  Teuchos::RCP<Tpetra_Vector> fT;
  if (f != NULL)
    fT = Petra::EpetraVector_To_TpetraVectorNonConst(*f, commT);

  Teuchos::RCP<Tpetra_MultiVector> JVT;
  if (JV != NULL)
    JVT = Petra::EpetraMultiVector_To_TpetraMultiVector(*JV, commT);

  RCP<Tpetra_MultiVector> fpT;
  if (fp != NULL)
    fpT = Petra::EpetraMultiVector_To_TpetraMultiVector(*fp, commT);

  this->computeGlobalTangentImplT(
      alpha, beta, omega, current_time, sum_derivs,
      xdotT, xdotdotT, xT,
      par, deriv_par,
      VxT, VxdotT, VxdotdotT, VpT,
      fT, JVT, fpT);

  // Convert output back from Tpetra to Epetra

  if (f != NULL) {
    Petra::TpetraVector_To_EpetraVector(fT, *f, comm);
  }
  if (JV != NULL) {
    Petra::TpetraMultiVector_To_EpetraMultiVector(JVT, *JV, comm);
  }
  if (fp != NULL) {
    Petra::TpetraMultiVector_To_EpetraMultiVector(fpT, *fp, comm);
  }

}
#endif


void
Albany::Application::
computeGlobalTangentT(const double alpha,
		     const double beta,
		     const double omega,
		     const double current_time,
		     bool sum_derivs,
		     const Tpetra_Vector* xdotT,
		     const Tpetra_Vector* xdotdotT,
		     const Tpetra_Vector& xT,
		     const Teuchos::Array<ParamVec>& par,
		     ParamVec* deriv_par,
		     const Tpetra_MultiVector* VxT,
		     const Tpetra_MultiVector* VxdotT,
		     const Tpetra_MultiVector* VxdotdotT,
		     const Tpetra_MultiVector* VpT,
		     Tpetra_Vector* fT,
		     Tpetra_MultiVector* JVT,
		     Tpetra_MultiVector* fpT)
{
  // Create non-owning RCPs to Tpetra objects
  // to be passed to the implementation
  this->computeGlobalTangentImplT(
      alpha, beta, omega, current_time, sum_derivs,
      Teuchos::rcp(xdotT, false), Teuchos::rcp(xdotdotT, false), Teuchos::rcpFromRef(xT),
      par, deriv_par,
      Teuchos::rcp(VxT, false), Teuchos::rcp(VxdotT, false), Teuchos::rcp(VxdotdotT, false), Teuchos::rcp(VpT, false),
      Teuchos::rcp(fT, false), Teuchos::rcp(JVT, false), Teuchos::rcp(fpT, false));
}

void Albany::Application::
applyGlobalDistParamDerivImplT(const double current_time,
                               const Teuchos::RCP<const Tpetra_Vector> &xdotT,
                               const Teuchos::RCP<const Tpetra_Vector> &xdotdotT,
                               const Teuchos::RCP<const Tpetra_Vector> &xT,
                               const Teuchos::Array<ParamVec>& p,
                               const std::string& dist_param_name,
                               const bool trans,
                               const Teuchos::RCP<const Tpetra_MultiVector>& VT,
                               const Teuchos::RCP<Tpetra_MultiVector>& fpVT)
{
  TEUCHOS_FUNC_TIME_MONITOR("> Albany Fill: Distributed Parameter Derivative");

  postRegSetup("Distributed Parameter Derivative");

  // Load connectivity map and coordinates
  const WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > > >::type&
        wsElNodeEqID = disc->getWsElNodeEqID();
  const WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >::type&
        coords = disc->getCoords();
  const WorksetArray<std::string>::type& wsEBNames = disc->getWsEBNames();
  const WorksetArray<int>::type& wsPhysIndex = disc->getWsPhysIndex();

  int numWorksets = wsElNodeEqID.size();

  Teuchos::RCP<Tpetra_Export> exporterT = solMgrT->get_exporterT();

  // Scatter x and xdot to the overlapped distribution
  solMgrT->scatterXT(*xT, xdotT.get(), xdotdotT.get());

  //Scatter distributed parameters
  distParamLib->scatter();

  // Set parameters
  for (int i=0; i<p.size(); i++)
    for (unsigned int j=0; j<p[i].size(); j++)
      p[i][j].family->setRealValueForAllTypes(p[i][j].baseValue);

#ifdef ALBANY_CUTR
  if (shapeParamsHaveBeenReset) {
    TEUCHOS_FUNC_TIME_MONITOR("Albany-Cubit MeshMover");

*out << " Calling moveMesh with params: " << std::setprecision(8);
 for (unsigned int i=0; i<shapeParams.size(); i++) *out << shapeParams[i] << "  ";
*out << std::endl;
    meshMover->moveMesh(shapeParams, morphFromInit);
    coords = disc->getCoords();
    shapeParamsHaveBeenReset = false;
  }
#endif

  RCP<Tpetra_MultiVector> overlapped_fpVT;
  if (trans) {
    overlapped_fpVT = rcp(new Tpetra_MultiVector(distParamLib->get(dist_param_name)->overlap_map(),
                                                 VT->getNumVectors()));
  }
  else {
    overlapped_fpVT = rcp(new Tpetra_MultiVector(disc->getOverlapMapT(), fpVT->getNumVectors()));
  }
  overlapped_fpVT->putScalar(0.0);
  fpVT->putScalar(0.0);

  RCP<const Tpetra_MultiVector> V_bcT = VT;

  // For (df/dp)^T*V, we have to evaluate Dirichlet BC's first
  if (trans && dfm!=Teuchos::null) {
    RCP<Tpetra_MultiVector> V_bc_ncT = rcp(new Tpetra_MultiVector(*VT));
    V_bcT = V_bc_ncT;

    PHAL::Workset workset;

    workset.fpVT = fpVT;
    workset.Vp_bcT = V_bc_ncT;
    workset.transpose_dist_param_deriv = trans;
    workset.dist_param_deriv_name = dist_param_name;
    workset.disc = disc;

    if ( paramLib->isParameter("Time") )
      workset.current_time =
        paramLib->getRealValue<PHAL::AlbanyTraits::Residual>("Time");
    else
      workset.current_time = current_time;

    dfm_set(workset, xT, xdotT, xdotdotT, rc_mgr);

    loadWorksetNodesetInfo(workset);

    // FillType template argument used to specialize Sacado
    dfm->evaluateFields<PHAL::AlbanyTraits::DistParamDeriv>(workset);
  }

  // Import V (after BC's applied) to overlapped distribution
  RCP<Tpetra_MultiVector> overlapped_VT;
  if (trans) {
    Teuchos::RCP<Tpetra_Import>& importer = solMgrT->get_importerT();
    overlapped_VT = rcp(
      new Tpetra_MultiVector(disc->getOverlapMapT(), VT->getNumVectors()));
    overlapped_VT->doImport(*V_bcT, *importer, Tpetra::INSERT);
  }
  else {
    overlapped_VT = rcp(
      new Tpetra_MultiVector(distParamLib->get(dist_param_name)->overlap_map(),
                             VT->getNumVectors()));
    distParamLib->get(dist_param_name)->import(*overlapped_VT, *V_bcT);
  }

  // Set data in Workset struct, and perform fill via field manager
  {
    PHAL::Workset workset;
    if (!paramLib->isParameter("Time"))
//      loadBasicWorksetInfo( workset, overlapped_x, overlapped_xdot, current_time );
      loadBasicWorksetInfoT( workset, current_time );
    else
//      loadBasicWorksetInfo( workset, overlapped_x, overlapped_xdot,
      loadBasicWorksetInfoT( workset,
                            paramLib->getRealValue<PHAL::AlbanyTraits::Residual>("Time") );

    workset.dist_param_deriv_name = dist_param_name;
    workset.VpT = overlapped_VT;
    workset.fpVT = overlapped_fpVT;
    workset.transpose_dist_param_deriv = trans;

    for (int ws=0; ws < numWorksets; ws++) {
      loadWorksetBucketInfo<PHAL::AlbanyTraits::DistParamDeriv>(workset, ws);

      // FillType template argument used to specialize Sacado
      fm[wsPhysIndex[ws]]->evaluateFields<PHAL::AlbanyTraits::DistParamDeriv>(workset);
      if (nfm!=Teuchos::null)
        deref_nfm(nfm, wsPhysIndex, ws)->evaluateFields<PHAL::AlbanyTraits::DistParamDeriv>(workset);
    }
  }

  // std::stringstream pg; pg << "neumann_phalanx_graph_ ";
  // nfm[0]->writeGraphvizFile<PHAL::AlbanyTraits::DistParamDeriv>(pg.str(),true,true);

  { TEUCHOS_FUNC_TIME_MONITOR("> Albany Fill: Distributed Parameter Derivative Export");
  // Assemble global df/dp*V
  if (trans) {
    Tpetra_MultiVector temp(*fpVT,Teuchos::Copy);
    distParamLib->get(dist_param_name)->export_add(*fpVT, *overlapped_fpVT);
    fpVT->update(1.0, temp, 1.0); //fpTV += temp;
  }
  else {
    fpVT->doExport(*overlapped_fpVT, *exporterT, Tpetra::ADD);
  }
  } // End timer

  // Apply Dirichlet conditions using dfm (Dirchelt Field Manager)
  if (!trans && dfm!=Teuchos::null) {
    PHAL::Workset workset;

    workset.fpVT = fpVT;
    workset.VpT = V_bcT;
    workset.transpose_dist_param_deriv = trans;

    if ( paramLib->isParameter("Time") )
      workset.current_time = paramLib->getRealValue<PHAL::AlbanyTraits::Residual>("Time");
    else
      workset.current_time = current_time;

    dfm_set(workset, xT, xdotT, xdotdotT, rc_mgr);

    loadWorksetNodesetInfo(workset);

    // FillType template argument used to specialize Sacado
    dfm->evaluateFields<PHAL::AlbanyTraits::DistParamDeriv>(workset);
  }

}
    
void
Albany::Application::
evaluateResponseT(int response_index,
                 const double current_time,
                 const Tpetra_Vector* xdotT,
                 const Tpetra_Vector* xdotdotT,
                 const Tpetra_Vector& xT,
                 const Teuchos::Array<ParamVec>& p,
                 Tpetra_Vector& gT)
{
  //eb-hack Initialize the vectors here so that we can accumulate the nodal
  // state data state in ProjectIPtoNodalField.
  try {
    Teuchos::RCP<Adapt::NodalDataBase>
      ndb = stateMgr.getStateInfoStruct()->getNodalDataBase();
    if (!ndb.is_null()) {
      ndb->getNodalDataVector()->initializeVectors(0);
      ndb->getNodalDataVector()->initEvaluateCalls(meshSpecs.size());
    }
  } catch (...) { /* No nodal data vector. */ }

  double t = current_time;
  if ( paramLib->isParameter("Time") )
    t = paramLib->getRealValue<PHAL::AlbanyTraits::Residual>("Time");

  responses[response_index]->evaluateResponseT(t, xdotT, xdotdotT, xT, p, gT);
}


void
Albany::Application::
evaluateResponseTangentT(int response_index,
			const double alpha,
			const double beta,
			const double omega,
			const double current_time,
			bool sum_derivs,
			const Tpetra_Vector* xdotT,
			const Tpetra_Vector* xdotdotT,
			const Tpetra_Vector& xT,
			const Teuchos::Array<ParamVec>& p,
			ParamVec* deriv_p,
			const Tpetra_MultiVector* VxdotT,
			const Tpetra_MultiVector* VxdotdotT,
			const Tpetra_MultiVector* VxT,
			const Tpetra_MultiVector* VpT,
			Tpetra_Vector* gT,
			Tpetra_MultiVector* gxT,
			Tpetra_MultiVector* gpT)
{
  double t = current_time;
  if ( paramLib->isParameter("Time") )
    t = paramLib->getRealValue<PHAL::AlbanyTraits::Residual>("Time");

  responses[response_index]->evaluateTangentT(
    alpha, beta, omega, t, sum_derivs, xdotT, xdotdotT, xT, p, deriv_p, VxdotT, VxdotdotT, VxT, VpT, gT, gxT, gpT);
}

#if defined(ALBANY_EPETRA)
void
Albany::Application::
evaluateResponseDerivative(
  int response_index,
  const double current_time,
  const Epetra_Vector* xdot,
  const Epetra_Vector* xdotdot,
  const Epetra_Vector& x,
  const Teuchos::Array<ParamVec>& p,
  ParamVec* deriv_p,
  Epetra_Vector* g,
  const EpetraExt::ModelEvaluator::Derivative& dg_dx,
  const EpetraExt::ModelEvaluator::Derivative& dg_dxdot,
  const EpetraExt::ModelEvaluator::Derivative& dg_dxdotdot,
  const EpetraExt::ModelEvaluator::Derivative& dg_dp)
{
  TEUCHOS_FUNC_TIME_MONITOR("> Albany Fill: Response Gradient");
  double t = current_time;
  if ( paramLib->isParameter("Time") )
    t = paramLib->getRealValue<PHAL::AlbanyTraits::Residual>("Time");

  responses[response_index]->evaluateDerivative(
    t, xdot, xdotdot, x, p, deriv_p, g, dg_dx, dg_dxdot, dg_dxdotdot, dg_dp);
}
#endif 

void
Albany::Application::
evaluateResponseDerivativeT(
  int response_index,
  const double current_time,
  const Tpetra_Vector* xdotT,
  const Tpetra_Vector* xdotdotT,
  const Tpetra_Vector& xT,
  const Teuchos::Array<ParamVec>& p,
  ParamVec* deriv_p,
  Tpetra_Vector* gT,
  const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dxT,
  const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dxdotT,
  const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dxdotdotT,
  const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dpT)
{
  double t = current_time;
  if ( paramLib->isParameter("Time") )
    t = paramLib->getRealValue<PHAL::AlbanyTraits::Residual>("Time");

  responses[response_index]->evaluateDerivativeT(
    t, xdotT, xdotdotT, xT, p, deriv_p, gT, dg_dxT, dg_dxdotT, dg_dxdotdotT, dg_dpT);
}

#if defined(ALBANY_EPETRA)
void
Albany::Application::
evaluateResponseDistParamDeriv(
    int response_index,
    const double current_time,
    const Epetra_Vector* xdot,
    const Epetra_Vector* xdotdot,
    const Epetra_Vector& x,
    const Teuchos::Array<ParamVec>& param_array,
    const std::string& dist_param_name,
    Epetra_MultiVector* dg_dp) {
  TEUCHOS_FUNC_TIME_MONITOR("> Albany Fill: Response Distributed Parameter Derivative");
  double t = current_time;
  if ( paramLib->isParameter("Time") )
    t = paramLib->getRealValue<PHAL::AlbanyTraits::Residual>("Time");

  responses[response_index]->evaluateDistParamDeriv(t, xdot, xdotdot, x, param_array, dist_param_name, dg_dp);
}
#endif

#ifdef ALBANY_SG
void
Albany::Application::
computeGlobalSGResidual(
  const double current_time,
  const Stokhos::EpetraVectorOrthogPoly* sg_xdot,
  const Stokhos::EpetraVectorOrthogPoly* sg_xdotdot,
  const Stokhos::EpetraVectorOrthogPoly& sg_x,
  const Teuchos::Array<ParamVec>& p,
  const Teuchos::Array<int>& sg_p_index,
  const Teuchos::Array< Teuchos::Array<SGType> >& sg_p_vals,
  Stokhos::EpetraVectorOrthogPoly& sg_f)
{
  TEUCHOS_FUNC_TIME_MONITOR("> Albany Fill: SGResidual");

  postRegSetup("SGResidual");

  // Load connectivity map and coordinates
  const WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > > >::type&
        wsElNodeEqID = disc->getWsElNodeEqID();
  const WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >::type&
        coords = disc->getCoords();
  const WorksetArray<std::string>::type& wsEBNames = disc->getWsEBNames();
  const WorksetArray<int>::type& wsPhysIndex = disc->getWsPhysIndex();

  int numWorksets = wsElNodeEqID.size();

  Teuchos::RCP<Epetra_Import>& importer = solMgr->get_importer();
  Teuchos::RCP<Epetra_Export>& exporter = solMgr->get_exporter();

  if (sg_overlapped_x == Teuchos::null ||
      sg_overlapped_x->size() != sg_x.size()) {
    sg_overlap_map =
      rcp(new Epetra_LocalMap(sg_basis->size(), 0,
                              product_comm->TimeDomainComm()));
    sg_overlapped_x =
      rcp(new Stokhos::EpetraVectorOrthogPoly(
            sg_basis, sg_overlap_map, disc->getOverlapMap(), product_comm));
    sg_overlapped_xdot =
        rcp(new Stokhos::EpetraVectorOrthogPoly(
              sg_basis, sg_overlap_map, disc->getOverlapMap(), product_comm));
    sg_overlapped_xdotdot =
        rcp(new Stokhos::EpetraVectorOrthogPoly(
              sg_basis, sg_overlap_map, disc->getOverlapMap(), product_comm));
    sg_overlapped_f =
      rcp(new Stokhos::EpetraVectorOrthogPoly(
            sg_basis, sg_overlap_map, disc->getOverlapMap(), product_comm));
  }

  for (int i=0; i<sg_x.size(); i++) {

    // Scatter x and xdot to the overlapped distrbution
    (*sg_overlapped_x)[i].Import(sg_x[i], *importer, Insert);
    if (sg_xdot != NULL) (*sg_overlapped_xdot)[i].Import((*sg_xdot)[i], *importer, Insert);
    if (sg_xdotdot != NULL) (*sg_overlapped_xdotdot)[i].Import((*sg_xdotdot)[i], *importer, Insert);

    // Zero out overlapped residual
    (*sg_overlapped_f)[i].PutScalar(0.0);
    sg_f[i].PutScalar(0.0);

  }

  // Set parameters
  for (int i=0; i<p.size(); i++)
    for (unsigned int j=0; j<p[i].size(); j++)
      p[i][j].family->setRealValueForAllTypes(p[i][j].baseValue);

  // put current_time (from Rythmos) if this is a transient problem, then compute dt
  //  if (sg_xdot != NULL) timeMgr.setTime(current_time);

#ifdef ALBANY_CUTR
  if (shapeParamsHaveBeenReset) {
    TEUCHOS_FUNC_TIME_MONITOR("Albany-Cubit MeshMover");
*out << " Calling moveMesh with params: " << std::setprecision(8);
for (unsigned int i=0; i<shapeParams.size(); i++) *out << shapeParams[i] << "  ";
*out << std::endl;
    meshMover->moveMesh(shapeParams, morphFromInit);
    coords = disc->getCoords();
    shapeParamsHaveBeenReset = false;
  }
#endif

  // Set SG parameters
  for (int i=0; i<sg_p_index.size(); i++) {
    int ii = sg_p_index[i];
    for (unsigned int j=0; j<p[ii].size(); j++)
      p[ii][j].family->setValue<PHAL::AlbanyTraits::SGResidual>(sg_p_vals[ii][j]);
  }

  // Set data in Workset struct, and perform fill via field manager
  {
    PHAL::Workset workset;

    workset.sg_expansion = sg_expansion;
    workset.sg_x         = sg_overlapped_x;
    workset.sg_xdot      = sg_overlapped_xdot;
    workset.sg_xdotdot      = sg_overlapped_xdotdot;
    workset.sg_f         = sg_overlapped_f;

    workset.current_time = current_time;
    //workset.delta_time = timeMgr.getDeltaTime();
    if (sg_xdot != NULL) workset.transientTerms = true;
    if (sg_xdotdot != NULL) workset.accelerationTerms = true;

    for (int ws=0; ws < numWorksets; ws++) {
      loadWorksetBucketInfo<PHAL::AlbanyTraits::SGResidual>(workset, ws);

      // FillType template argument used to specialize Sacado
      fm[wsPhysIndex[ws]]->evaluateFields<PHAL::AlbanyTraits::SGResidual>(workset);
      if (nfm!=Teuchos::null)
        deref_nfm(nfm, wsPhysIndex, ws)->evaluateFields<PHAL::AlbanyTraits::SGResidual>(workset);
    }
  }

  // Assemble global residual
  for (int i=0; i<sg_f.size(); i++) {
    sg_f[i].Export((*sg_overlapped_f)[i], *exporter, Add);
  }

  // Apply Dirichlet conditions using dfm (Dirchelt Field Manager)
  if (dfm!=Teuchos::null) {
    PHAL::Workset workset;

    workset.sg_f = Teuchos::rcpFromRef(sg_f);
    loadWorksetNodesetInfo(workset);
    workset.distParamLib = distParamLib;
    workset.sg_x = Teuchos::rcpFromRef(sg_x);
    if (sg_xdot != NULL) workset.transientTerms = true;
    if (sg_xdotdot != NULL) workset.accelerationTerms = true;

    if ( paramLib->isParameter("Time") )
      workset.current_time = paramLib->getRealValue<PHAL::AlbanyTraits::Residual>("Time");
    else
      workset.current_time = current_time;

    workset.disc = disc;

#if defined(ALBANY_LCM)
    // Needed for more specialized Dirichlet BCs (e.g. Schwarz coupling)
    workset.apps_ = apps_;
    workset.current_app_ = Teuchos::rcp(this, false);
#endif

    // FillType template argument used to specialize Sacado
    dfm->evaluateFields<PHAL::AlbanyTraits::SGResidual>(workset);

  }

}

void
Albany::Application::
computeGlobalSGJacobian(
  const double alpha,
  const double beta,
  const double omega,
  const double current_time,
  const Stokhos::EpetraVectorOrthogPoly* sg_xdot,
  const Stokhos::EpetraVectorOrthogPoly* sg_xdotdot,
  const Stokhos::EpetraVectorOrthogPoly& sg_x,
  const Teuchos::Array<ParamVec>& p,
  const Teuchos::Array<int>& sg_p_index,
  const Teuchos::Array< Teuchos::Array<SGType> >& sg_p_vals,
  Stokhos::EpetraVectorOrthogPoly* sg_f,
  Stokhos::VectorOrthogPoly<Epetra_CrsMatrix>& sg_jac)
{
  TEUCHOS_FUNC_TIME_MONITOR("> Albany Fill: SGJacobian");

  postRegSetup("SGJacobian");

  // Load connectivity map and coordinates
  const WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > > >::type&
        wsElNodeEqID = disc->getWsElNodeEqID();
  const WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >::type&
        coords = disc->getCoords();
  const WorksetArray<std::string>::type& wsEBNames = disc->getWsEBNames();
  const WorksetArray<int>::type& wsPhysIndex = disc->getWsPhysIndex();

  int numWorksets = wsElNodeEqID.size();

  Teuchos::RCP<Epetra_CrsMatrix>& overlapped_jac = solMgr->get_overlapped_jac();
  Teuchos::RCP<Epetra_Import>& importer = solMgr->get_importer();
  Teuchos::RCP<Epetra_Export>& exporter = solMgr->get_exporter();

  if (sg_overlapped_x == Teuchos::null ||
      sg_overlapped_x->size() != sg_x.size()) {
    sg_overlap_map =
      rcp(new Epetra_LocalMap(sg_basis->size(), 0,
                              product_comm->TimeDomainComm()));
    sg_overlapped_x =
      rcp(new Stokhos::EpetraVectorOrthogPoly(
            sg_basis, sg_overlap_map, disc->getOverlapMap(), product_comm));
    sg_overlapped_xdot =
        rcp(new Stokhos::EpetraVectorOrthogPoly(
              sg_basis, sg_overlap_map, disc->getOverlapMap(), product_comm));
    sg_overlapped_xdotdot =
        rcp(new Stokhos::EpetraVectorOrthogPoly(
              sg_basis, sg_overlap_map, disc->getOverlapMap(), product_comm));
    sg_overlapped_f =
      rcp(new Stokhos::EpetraVectorOrthogPoly(
            sg_basis, sg_overlap_map, disc->getOverlapMap(), product_comm));
  }

  for (int i=0; i<sg_x.size(); i++) {

    // Scatter x and xdot to the overlapped distrbution
    (*sg_overlapped_x)[i].Import(sg_x[i], *importer, Insert);
    if (sg_xdot != NULL) (*sg_overlapped_xdot)[i].Import((*sg_xdot)[i], *importer, Insert);
    if (sg_xdotdot != NULL) (*sg_overlapped_xdotdot)[i].Import((*sg_xdotdot)[i], *importer, Insert);

    // Zero out overlapped residual
    if (sg_f != NULL) {
      (*sg_overlapped_f)[i].PutScalar(0.0);
      (*sg_f)[i].PutScalar(0.0);
    }

  }

  // Create, resize and initialize overlapped Jacobians
  if (sg_overlapped_jac == Teuchos::null ||
      sg_overlapped_jac->size() != sg_jac.size()) {
    RCP<const Stokhos::OrthogPolyBasis<int,double> > sg_basis =
      sg_expansion->getBasis();
    RCP<Epetra_LocalMap> sg_overlap_jac_map =
      rcp(new Epetra_LocalMap(sg_jac.size(), 0,
                              sg_overlap_map->Comm()));
    sg_overlapped_jac =
      rcp(new Stokhos::VectorOrthogPoly<Epetra_CrsMatrix>(
                     sg_basis, sg_overlap_jac_map, *overlapped_jac));
  }
  for (int i=0; i<sg_overlapped_jac->size(); i++)
    (*sg_overlapped_jac)[i].PutScalar(0.0);

  // Zero out overlapped Jacobian
  for (int i=0; i<sg_jac.size(); i++)
    (*sg_overlapped_jac)[i].PutScalar(0.0);

  // Set parameters
  for (int i=0; i<p.size(); i++)
    for (unsigned int j=0; j<p[i].size(); j++)
      p[i][j].family->setRealValueForAllTypes(p[i][j].baseValue);

  // put current_time (from Rythmos) if this is a transient problem, then compute dt
  //  if (sg_xdot != NULL) timeMgr.setTime(current_time);

#ifdef ALBANY_CUTR
  if (shapeParamsHaveBeenReset) {
    TEUCHOS_FUNC_TIME_MONITOR("Albany-Cubit MeshMover");
*out << " Calling moveMesh with params: " << std::setprecision(8);
for (unsigned int i=0; i<shapeParams.size(); i++) *out << shapeParams[i] << "  ";
*out << std::endl;
    meshMover->moveMesh(shapeParams, morphFromInit);
    coords = disc->getCoords();
    shapeParamsHaveBeenReset = false;
  }
#endif

  // Set SG parameters
  for (int i=0; i<sg_p_index.size(); i++) {
    int ii = sg_p_index[i];
    for (unsigned int j=0; j<p[ii].size(); j++)
      p[ii][j].family->setValue<PHAL::AlbanyTraits::SGJacobian>(sg_p_vals[ii][j]);
  }

  RCP< Stokhos::EpetraVectorOrthogPoly > sg_overlapped_ff;
  if (sg_f != NULL)
    sg_overlapped_ff = sg_overlapped_f;

  // Set data in Workset struct, and perform fill via field manager
  {
    PHAL::Workset workset;

    workset.sg_expansion = sg_expansion;
    workset.sg_x         = sg_overlapped_x;
    workset.sg_xdot      = sg_overlapped_xdot;
    workset.sg_xdotdot      = sg_overlapped_xdotdot;
    workset.sg_f         = sg_overlapped_ff;

    workset.sg_Jac       = sg_overlapped_jac;
    loadWorksetJacobianInfo(workset, alpha, beta, omega);
    workset.current_time = current_time;
    //workset.delta_time = timeMgr.getDeltaTime();
    if (sg_xdot != NULL) workset.transientTerms = true;
    if (sg_xdotdot != NULL) workset.accelerationTerms = true;

    for (int ws=0; ws < numWorksets; ws++) {
      loadWorksetBucketInfo<PHAL::AlbanyTraits::SGJacobian>(workset, ws);

      // FillType template argument used to specialize Sacado
      fm[wsPhysIndex[ws]]->evaluateFields<PHAL::AlbanyTraits::SGJacobian>(workset);
      if (nfm!=Teuchos::null)
        deref_nfm(nfm, wsPhysIndex, ws)->evaluateFields<PHAL::AlbanyTraits::SGJacobian>(workset);
    }
  }

  // Assemble global residual
  if (sg_f != NULL)
    for (int i=0; i<sg_f->size(); i++)
      (*sg_f)[i].Export((*sg_overlapped_f)[i], *exporter, Add);

  // Assemble block Jacobians
  RCP<Epetra_CrsMatrix> jac;
  for (int i=0; i<sg_jac.size(); i++) {
    jac = sg_jac.getCoeffPtr(i);
    jac->PutScalar(0.0);
    jac->Export((*sg_overlapped_jac)[i], *exporter, Add);
    jac->FillComplete(true);
  }

  // Apply Dirichlet conditions using dfm (Dirchelt Field Manager)
  if (dfm!=Teuchos::null) {
    PHAL::Workset workset;

    workset.sg_f = rcp(sg_f,false);
    workset.sg_Jac = Teuchos::rcpFromRef(sg_jac);
    workset.j_coeff = beta;
    workset.n_coeff = omega;
    workset.sg_x = Teuchos::rcpFromRef(sg_x);
    if (sg_xdot != NULL) workset.transientTerms = true;
    if (sg_xdotdot != NULL) workset.accelerationTerms = true;

    loadWorksetNodesetInfo(workset);
    workset.distParamLib = distParamLib;

    workset.disc = disc;

#if defined(ALBANY_LCM)
    // Needed for more specialized Dirichlet BCs (e.g. Schwarz coupling)
    workset.apps_ = apps_;
    workset.current_app_ = Teuchos::rcp(this, false);
#endif

    // FillType template argument used to specialize Sacado
    dfm->evaluateFields<PHAL::AlbanyTraits::SGJacobian>(workset);
  }

}

void
Albany::Application::
computeGlobalSGTangent(
  const double alpha,
  const double beta,
  const double omega,
  const double current_time,
  bool sum_derivs,
  const Stokhos::EpetraVectorOrthogPoly* sg_xdot,
  const Stokhos::EpetraVectorOrthogPoly* sg_xdotdot,
  const Stokhos::EpetraVectorOrthogPoly& sg_x,
  const Teuchos::Array<ParamVec>& par,
  const Teuchos::Array<int>& sg_p_index,
  const Teuchos::Array< Teuchos::Array<SGType> >& sg_p_vals,
  ParamVec* deriv_par,
  const Epetra_MultiVector* Vx,
  const Epetra_MultiVector* Vxdot,
  const Epetra_MultiVector* Vxdotdot,
  const Epetra_MultiVector* Vp,
  Stokhos::EpetraVectorOrthogPoly* sg_f,
  Stokhos::EpetraMultiVectorOrthogPoly* sg_JVx,
  Stokhos::EpetraMultiVectorOrthogPoly* sg_fVp)
{
  TEUCHOS_FUNC_TIME_MONITOR("> Albany Fill: SGTangent");

  postRegSetup("SGTangent");

  // Load connectivity map and coordinates
  const WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > > >::type&
        wsElNodeEqID = disc->getWsElNodeEqID();
  const WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >::type&
        coords = disc->getCoords();
  const WorksetArray<std::string>::type& wsEBNames = disc->getWsEBNames();
  const WorksetArray<int>::type& wsPhysIndex = disc->getWsPhysIndex();

  int numWorksets = wsElNodeEqID.size();

  Teuchos::RCP<Epetra_Import>& importer = solMgr->get_importer();
  Teuchos::RCP<Epetra_Export>& exporter = solMgr->get_exporter();

  if (sg_overlapped_x == Teuchos::null ||
      sg_overlapped_x->size() != sg_x.size()) {
    sg_overlap_map =
      rcp(new Epetra_LocalMap(sg_basis->size(), 0,
                              product_comm->TimeDomainComm()));
    sg_overlapped_x =
      rcp(new Stokhos::EpetraVectorOrthogPoly(
            sg_basis, sg_overlap_map, disc->getOverlapMap(), product_comm));
    sg_overlapped_xdot =
        rcp(new Stokhos::EpetraVectorOrthogPoly(
              sg_basis, sg_overlap_map, disc->getOverlapMap(), product_comm));
    sg_overlapped_xdotdot =
        rcp(new Stokhos::EpetraVectorOrthogPoly(
              sg_basis, sg_overlap_map, disc->getOverlapMap(), product_comm));
    sg_overlapped_f =
      rcp(new Stokhos::EpetraVectorOrthogPoly(
            sg_basis, sg_overlap_map, disc->getOverlapMap(), product_comm));
  }

  for (int i=0; i<sg_x.size(); i++) {

    // Scatter x and xdot to the overlapped distrbution
    (*sg_overlapped_x)[i].Import(sg_x[i], *importer, Insert);
    if (sg_xdot != NULL) (*sg_overlapped_xdot)[i].Import((*sg_xdot)[i], *importer, Insert);
    if (sg_xdotdot != NULL) (*sg_overlapped_xdotdot)[i].Import((*sg_xdotdot)[i], *importer, Insert);

    // Zero out overlapped residual
    if (sg_f != NULL) {
      (*sg_overlapped_f)[i].PutScalar(0.0);
      (*sg_f)[i].PutScalar(0.0);
    }

  }

  // Scatter Vx to the overlapped distribution
  RCP<Epetra_MultiVector> overlapped_Vx;
  if (Vx != NULL) {
    overlapped_Vx =
      rcp(new Epetra_MultiVector(*(disc->getOverlapMap()),
                                 Vx->NumVectors()));
    overlapped_Vx->Import(*Vx, *importer, Insert);
  }

  // Scatter Vx dot to the overlapped distribution
  RCP<Epetra_MultiVector> overlapped_Vxdot;
  if (Vxdot != NULL) {
    overlapped_Vxdot = rcp(new Epetra_MultiVector(*(disc->getOverlapMap()), Vxdot->NumVectors()));
    overlapped_Vxdot->Import(*Vxdot, *importer, Insert);
  }
  RCP<Epetra_MultiVector> overlapped_Vxdotdot;
  if (Vxdotdot != NULL) {
    overlapped_Vxdotdot = rcp(new Epetra_MultiVector(*(disc->getOverlapMap()), Vxdotdot->NumVectors()));
    overlapped_Vxdotdot->Import(*Vxdotdot, *importer, Insert);
  }

  // Set parameters
  for (int i=0; i<par.size(); i++)
    for (unsigned int j=0; j<par[i].size(); j++)
      par[i][j].family->setRealValueForAllTypes(par[i][j].baseValue);

  // Set SG parameters
  for (int i=0; i<sg_p_index.size(); i++) {
    int ii = sg_p_index[i];
    for (unsigned int j=0; j<par[ii].size(); j++)
        par[ii][j].family->setValue<PHAL::AlbanyTraits::SGTangent>(sg_p_vals[ii][j]);
  }

  // put current_time (from Rythmos) if this is a transient problem, then compute dt
  //  if (sg_xdot != NULL) timeMgr.setTime(current_time);

  RCP<const Epetra_MultiVector > vp = rcp(Vp, false);
  RCP<ParamVec> params = rcp(deriv_par, false);

  RCP<Stokhos::EpetraVectorOrthogPoly> sg_overlapped_ff;
  if (sg_f != NULL)
    sg_overlapped_ff = sg_overlapped_f;

  Teuchos::RCP< Stokhos::EpetraMultiVectorOrthogPoly > sg_overlapped_JVx;
  if (sg_JVx != NULL) {
    sg_overlapped_JVx =
      Teuchos::rcp(new Stokhos::EpetraMultiVectorOrthogPoly(
                     sg_basis, sg_overlap_map, disc->getOverlapMap(),
                     sg_x.productComm(),
                     (*sg_JVx)[0].NumVectors()));
    sg_JVx->init(0.0);
  }

  Teuchos::RCP<Stokhos::EpetraMultiVectorOrthogPoly > sg_overlapped_fVp;
  if (sg_fVp != NULL) {
    sg_overlapped_fVp =
      Teuchos::rcp(new Stokhos::EpetraMultiVectorOrthogPoly(
                     sg_basis, sg_overlap_map, disc->getOverlapMap(),
                     sg_x.productComm(),
                     (*sg_fVp)[0].NumVectors()));
    sg_fVp->init(0.0);
  }

  // Number of x & xdot tangent directions
  int num_cols_x = 0;
  if (Vx != NULL)
    num_cols_x = Vx->NumVectors();
  else if (Vxdot != NULL)
    num_cols_x = Vxdot->NumVectors();
  else if (Vxdotdot != NULL)
    num_cols_x = Vxdotdot->NumVectors();

  // Number of parameter tangent directions
  int num_cols_p = 0;
  if (params != Teuchos::null) {
    if (Vp != NULL)
      num_cols_p = Vp->NumVectors();
    else
      num_cols_p = params->size();
  }

  // Whether x and param tangent components are added or separate
  int param_offset = 0;
  if (!sum_derivs)
    param_offset = num_cols_x;  // offset of parameter derivs in deriv array

  TEUCHOS_TEST_FOR_EXCEPTION(sum_derivs &&
                     (num_cols_x != 0) &&
                     (num_cols_p != 0) &&
                     (num_cols_x != num_cols_p),
                     std::logic_error,
                     "Seed matrices Vx and Vp must have the same number " <<
                     " of columns when sum_derivs is true and both are "
                     << "non-null!" << std::endl);

  // Initialize
  if (params != Teuchos::null) {
    SGFadType p;
    int num_cols_tot = param_offset + num_cols_p;
    for (unsigned int i=0; i<params->size(); i++) {
      // Get the base value set above
      SGType base_val =
        (*params)[i].family->getValue<PHAL::AlbanyTraits::SGTangent>().val();
      p = SGFadType(num_cols_tot, base_val);
      if (Vp != NULL)
        for (int k=0; k<num_cols_p; k++)
          p.fastAccessDx(param_offset+k) = (*Vp)[k][i];
      else
        p.fastAccessDx(param_offset+i) = 1.0;
      (*params)[i].family->setValue<PHAL::AlbanyTraits::SGTangent>(p);
    }
  }

  // Set data in Workset struct, and perform fill via field manager
  {
    PHAL::Workset workset;

    workset.params = params;
    workset.sg_expansion = sg_expansion;
    workset.sg_x         = sg_overlapped_x;
    workset.sg_xdot      = sg_overlapped_xdot;
    workset.sg_xdotdot      = sg_overlapped_xdotdot;
    workset.Vx = overlapped_Vx;
    workset.Vxdot = overlapped_Vxdot;
    workset.Vxdotdot = overlapped_Vxdotdot;
    workset.Vp = vp;

    workset.sg_f         = sg_overlapped_ff;
    workset.sg_JV        = sg_overlapped_JVx;
    workset.sg_fp        = sg_overlapped_fVp;
    workset.j_coeff      = beta;
    workset.m_coeff      = alpha;
    workset.n_coeff      = omega;

    workset.num_cols_x = num_cols_x;
    workset.num_cols_p = num_cols_p;
    workset.param_offset = param_offset;

    workset.current_time = current_time; //timeMgr.getCurrentTime();
    //    workset.delta_time = timeMgr.getDeltaTime();
    if (sg_xdot != NULL) workset.transientTerms = true;
    if (sg_xdotdot != NULL) workset.accelerationTerms = true;

    for (int ws=0; ws < numWorksets; ws++) {
      loadWorksetBucketInfo<PHAL::AlbanyTraits::SGTangent>(workset, ws);

      // FillType template argument used to specialize Sacado
      fm[wsPhysIndex[ws]]->evaluateFields<PHAL::AlbanyTraits::SGTangent>(workset);
      if (nfm!=Teuchos::null)
        deref_nfm(nfm, wsPhysIndex, ws)->evaluateFields<PHAL::AlbanyTraits::SGTangent>(workset);
    }
  }

  vp = Teuchos::null;
  params = Teuchos::null;

  // Assemble global residual
  if (sg_f != NULL)
    for (int i=0; i<sg_f->size(); i++)
      (*sg_f)[i].Export((*sg_overlapped_f)[i], *exporter, Add);

  // Assemble derivatives
  if (sg_JVx != NULL)
    for (int i=0; i<sg_JVx->size(); i++)
      (*sg_JVx)[i].Export((*sg_overlapped_JVx)[i], *exporter, Add);
  if (sg_fVp != NULL) {
    for (int i=0; i<sg_fVp->size(); i++)
      (*sg_fVp)[i].Export((*sg_overlapped_fVp)[i], *exporter, Add);
  }

  // Apply Dirichlet conditions using dfm (Dirchelt Field Manager)
  if (dfm!=Teuchos::null) {
    PHAL::Workset workset;

    workset.num_cols_x = num_cols_x;
    workset.num_cols_p = num_cols_p;
    workset.param_offset = param_offset;

    workset.sg_f = rcp(sg_f,false);
    workset.sg_fp = rcp(sg_fVp,false);
    workset.sg_JV = rcp(sg_JVx,false);
    workset.j_coeff = beta;
    workset.n_coeff = omega;
    workset.sg_x = Teuchos::rcpFromRef(sg_x);
    workset.Vx = rcp(Vx,false);
    if (sg_xdot != NULL) workset.transientTerms = true;
    if (sg_xdotdot != NULL) workset.accelerationTerms = true;

    loadWorksetNodesetInfo(workset);
    workset.distParamLib = distParamLib;

    workset.disc = disc;

#if defined(ALBANY_LCM)
    // Needed for more specialized Dirichlet BCs (e.g. Schwarz coupling)
    workset.apps_ = apps_;
    workset.current_app_ = Teuchos::rcp(this, false);
#endif

    // FillType template argument used to specialize Sacado
    dfm->evaluateFields<PHAL::AlbanyTraits::SGTangent>(workset);
  }

}

void
Albany::Application::
evaluateSGResponse(
  int response_index,
  const double curr_time,
  const Stokhos::EpetraVectorOrthogPoly* sg_xdot,
  const Stokhos::EpetraVectorOrthogPoly* sg_xdotdot,
  const Stokhos::EpetraVectorOrthogPoly& sg_x,
  const Teuchos::Array<ParamVec>& p,
  const Teuchos::Array<int>& sg_p_index,
  const Teuchos::Array< Teuchos::Array<SGType> >& sg_p_vals,
  Stokhos::EpetraVectorOrthogPoly& sg_g)
{
  TEUCHOS_FUNC_TIME_MONITOR("> Albany Fill: SGResponses");

  responses[response_index]->evaluateSGResponse(
    curr_time, sg_xdot, sg_xdotdot, sg_x, p, sg_p_index, sg_p_vals, sg_g);
}

void
Albany::Application::
evaluateSGResponseTangent(
  int response_index,
  const double alpha,
  const double beta,
  const double omega,
  const double current_time,
  bool sum_derivs,
  const Stokhos::EpetraVectorOrthogPoly* sg_xdot,
  const Stokhos::EpetraVectorOrthogPoly* sg_xdotdot,
  const Stokhos::EpetraVectorOrthogPoly& sg_x,
  const Teuchos::Array<ParamVec>& p,
  const Teuchos::Array<int>& sg_p_index,
  const Teuchos::Array< Teuchos::Array<SGType> >& sg_p_vals,
  ParamVec* deriv_p,
  const Epetra_MultiVector* Vx,
  const Epetra_MultiVector* Vxdot,
  const Epetra_MultiVector* Vxdotdot,
  const Epetra_MultiVector* Vp,
  Stokhos::EpetraVectorOrthogPoly* sg_g,
  Stokhos::EpetraMultiVectorOrthogPoly* sg_JV,
  Stokhos::EpetraMultiVectorOrthogPoly* sg_gp)
{
  TEUCHOS_FUNC_TIME_MONITOR("> Albany Fill: SGResponse Tangent");

  responses[response_index]->evaluateSGTangent(
    alpha, beta, omega, current_time, sum_derivs, sg_xdot, sg_xdotdot, sg_x, p, sg_p_index,
    sg_p_vals, deriv_p, Vx, Vxdot, Vxdotdot, Vp, sg_g, sg_JV, sg_gp);
}

void
Albany::Application::
evaluateSGResponseDerivative(
  int response_index,
  const double current_time,
  const Stokhos::EpetraVectorOrthogPoly* sg_xdot,
  const Stokhos::EpetraVectorOrthogPoly* sg_xdotdot,
  const Stokhos::EpetraVectorOrthogPoly& sg_x,
  const Teuchos::Array<ParamVec>& p,
  const Teuchos::Array<int>& sg_p_index,
  const Teuchos::Array< Teuchos::Array<SGType> >& sg_p_vals,
  ParamVec* deriv_p,
  Stokhos::EpetraVectorOrthogPoly* sg_g,
  const EpetraExt::ModelEvaluator::SGDerivative& sg_dg_dx,
  const EpetraExt::ModelEvaluator::SGDerivative& sg_dg_dxdot,
  const EpetraExt::ModelEvaluator::SGDerivative& sg_dg_dxdotdot,
  const EpetraExt::ModelEvaluator::SGDerivative& sg_dg_dp)
{
  TEUCHOS_FUNC_TIME_MONITOR("> Albany Fill: SGResponse Gradient");

  responses[response_index]->evaluateSGDerivative(
    current_time, sg_xdot, sg_xdotdot, sg_x, p, sg_p_index, sg_p_vals, deriv_p,
    sg_g, sg_dg_dx, sg_dg_dxdot, sg_dg_dxdotdot, sg_dg_dp);
}
#endif 
#ifdef ALBANY_ENSEMBLE 
void
Albany::Application::
computeGlobalMPResidual(
  const double current_time,
  const Stokhos::ProductEpetraVector* mp_xdot,
  const Stokhos::ProductEpetraVector* mp_xdotdot,
  const Stokhos::ProductEpetraVector& mp_x,
  const Teuchos::Array<ParamVec>& p,
  const Teuchos::Array<int>& mp_p_index,
  const Teuchos::Array< Teuchos::Array<MPType> >& mp_p_vals,
  Stokhos::ProductEpetraVector& mp_f)
{
  TEUCHOS_FUNC_TIME_MONITOR("> Albany Fill: MPResidual");

  postRegSetup("MPResidual");

  // Load connectivity map and coordinates
  const WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > > >::type&
        wsElNodeEqID = disc->getWsElNodeEqID();
  const WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >::type&
        coords = disc->getCoords();
  const WorksetArray<std::string>::type& wsEBNames = disc->getWsEBNames();
  const WorksetArray<int>::type& wsPhysIndex = disc->getWsPhysIndex();

  int numWorksets = wsElNodeEqID.size();

  Teuchos::RCP<Epetra_Import>& importer = solMgr->get_importer();
  Teuchos::RCP<Epetra_Export>& exporter = solMgr->get_exporter();

  // Create overlapped multi-point Epetra objects
  if (mp_overlapped_x == Teuchos::null ||
      mp_overlapped_x->size() != mp_x.size()) {
    mp_overlapped_x =
      rcp(new Stokhos::ProductEpetraVector(
            mp_x.map(), disc->getOverlapMap(), mp_x.productComm()));

    if (mp_xdot != NULL)
      mp_overlapped_xdot =
        rcp(new Stokhos::ProductEpetraVector(
              mp_xdot->map(), disc->getOverlapMap(), mp_x.productComm()));

    if (mp_xdotdot != NULL)
      mp_overlapped_xdotdot =
        rcp(new Stokhos::ProductEpetraVector(
              mp_xdotdot->map(), disc->getOverlapMap(), mp_x.productComm()));

    if (mp_xdotdot != NULL)
      mp_overlapped_xdotdot =
	rcp(new Stokhos::ProductEpetraVector(
	      mp_xdotdot->map(), disc->getOverlapMap(), mp_x.productComm()));

  }

  if (mp_overlapped_f == Teuchos::null ||
      mp_overlapped_f->size() != mp_f.size()) {
    mp_overlapped_f =
      rcp(new Stokhos::ProductEpetraVector(
            mp_f.map(), disc->getOverlapMap(), mp_x.productComm()));
  }

  for (int i=0; i<mp_x.size(); i++) {

    // Scatter x and xdot to the overlapped distrbution
    (*mp_overlapped_x)[i].Import(mp_x[i], *importer, Insert);
    if (mp_xdot != NULL) (*mp_overlapped_xdot)[i].Import((*mp_xdot)[i], *importer, Insert);
    if (mp_xdotdot != NULL) (*mp_overlapped_xdotdot)[i].Import((*mp_xdotdot)[i], *importer, Insert);

    // Zero out overlapped residual
    (*mp_overlapped_f)[i].PutScalar(0.0);
    mp_f[i].PutScalar(0.0);

  }

  // Set parameters
  for (int i=0; i<p.size(); i++)
    for (unsigned int j=0; j<p[i].size(); j++)
      p[i][j].family->setRealValueForAllTypes(p[i][j].baseValue);

  // put current_time (from Rythmos) if this is a transient problem, then compute dt
  //  if (mp_xdot != NULL) timeMgr.setTime(current_time);

#ifdef ALBANY_CUTR
  if (shapeParamsHaveBeenReset) {
    TEUCHOS_FUNC_TIME_MONITOR("Albany-Cubit MeshMover");
*out << " Calling moveMesh with params: " << std::setprecision(8);
for (unsigned int i=0; i<shapeParams.size(); i++) *out << shapeParams[i] << "  ";
*out << std::endl;
    meshMover->moveMesh(shapeParams, morphFromInit);
    coords = disc->getCoords();
    shapeParamsHaveBeenReset = false;
  }
#endif

  // Set MP parameters
  for (int i=0; i<mp_p_index.size(); i++) {
    int ii = mp_p_index[i];
    for (unsigned int j=0; j<p[ii].size(); j++)
      p[ii][j].family->setValue<PHAL::AlbanyTraits::MPResidual>(mp_p_vals[ii][j]);
  }

  // Set data in Workset struct, and perform fill via field manager
  {
    PHAL::Workset workset;

    workset.mp_x         = mp_overlapped_x;
    workset.mp_xdot      = mp_overlapped_xdot;
    workset.mp_xdotdot      = mp_overlapped_xdotdot;
    workset.mp_f         = mp_overlapped_f;

    workset.current_time = current_time; //timeMgr.getCurrentTime();
    //    workset.delta_time = timeMgr.getDeltaTime();
    if (mp_xdot != NULL) workset.transientTerms = true;
    if (mp_xdotdot != NULL) workset.accelerationTerms = true;

    for (int ws=0; ws < numWorksets; ws++) {
      loadWorksetBucketInfo<PHAL::AlbanyTraits::MPResidual>(workset, ws);

      // FillType template argument used to specialize Sacado
      fm[wsPhysIndex[ws]]->evaluateFields<PHAL::AlbanyTraits::MPResidual>(workset);
      if (nfm!=Teuchos::null)
        deref_nfm(nfm, wsPhysIndex, ws)->evaluateFields<PHAL::AlbanyTraits::MPResidual>(workset);
    }
  }

  // Assemble global residual
  for (int i=0; i<mp_f.size(); i++) {
    mp_f[i].Export((*mp_overlapped_f)[i], *exporter, Add);
  }

  // Apply Dirichlet conditions using dfm (Dirchelt Field Manager)
  if (dfm!=Teuchos::null) {
    PHAL::Workset workset;

    workset.mp_f = Teuchos::rcpFromRef(mp_f);
    loadWorksetNodesetInfo(workset);
    workset.distParamLib = distParamLib;
    workset.mp_x = Teuchos::rcpFromRef(mp_x);
    if (mp_xdot != NULL) workset.transientTerms = true;
    if (mp_xdotdot != NULL) workset.accelerationTerms = true;

    workset.disc = disc;

#if defined(ALBANY_LCM)
    // Needed for more specialized Dirichlet BCs (e.g. Schwarz coupling)
    workset.apps_ = apps_;
    workset.current_app_ = Teuchos::rcp(this, false);
#endif

    // FillType template argument used to specialize Sacado
    dfm->evaluateFields<PHAL::AlbanyTraits::MPResidual>(workset);

  }
}

void
Albany::Application::
computeGlobalMPJacobian(
  const double alpha,
  const double beta,
  const double omega,
  const double current_time,
  const Stokhos::ProductEpetraVector* mp_xdot,
  const Stokhos::ProductEpetraVector* mp_xdotdot,
  const Stokhos::ProductEpetraVector& mp_x,
  const Teuchos::Array<ParamVec>& p,
  const Teuchos::Array<int>& mp_p_index,
  const Teuchos::Array< Teuchos::Array<MPType> >& mp_p_vals,
  Stokhos::ProductEpetraVector* mp_f,
  Stokhos::ProductContainer<Epetra_CrsMatrix>& mp_jac)
{
  TEUCHOS_FUNC_TIME_MONITOR("> Albany Fill: MPJacobian");

  postRegSetup("MPJacobian");

  // Load connectivity map and coordinates
  const WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > > >::type&
        wsElNodeEqID = disc->getWsElNodeEqID();
  const WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >::type&
        coords = disc->getCoords();
  const WorksetArray<std::string>::type& wsEBNames = disc->getWsEBNames();
  const WorksetArray<int>::type& wsPhysIndex = disc->getWsPhysIndex();

  int numWorksets = wsElNodeEqID.size();

  Teuchos::RCP<Epetra_CrsMatrix>& overlapped_jac = solMgr->get_overlapped_jac();
  Teuchos::RCP<Epetra_Import>& importer = solMgr->get_importer();
  Teuchos::RCP<Epetra_Export>& exporter = solMgr->get_exporter();

  // Create overlapped multi-point Epetra objects
  if (mp_overlapped_x == Teuchos::null ||
      mp_overlapped_x->size() != mp_x.size()) {
    mp_overlapped_x =
      rcp(new Stokhos::ProductEpetraVector(
            mp_x.map(), disc->getOverlapMap(), mp_x.productComm()));

    if (mp_xdot != NULL)
      mp_overlapped_xdot =
        rcp(new Stokhos::ProductEpetraVector(
              mp_xdot->map(), disc->getOverlapMap(), mp_x.productComm()));

    if (mp_xdotdot != NULL)
      mp_overlapped_xdotdot =
        rcp(new Stokhos::ProductEpetraVector(
              mp_xdotdot->map(), disc->getOverlapMap(), mp_x.productComm()));

    if (mp_xdotdot != NULL)
      mp_overlapped_xdotdot =
	rcp(new Stokhos::ProductEpetraVector(
	      mp_xdotdot->map(), disc->getOverlapMap(), mp_x.productComm()));

  }

  if (mp_f != NULL && (mp_overlapped_f == Teuchos::null ||
                       mp_overlapped_f->size() != mp_f->size()))
    mp_overlapped_f =
      rcp(new Stokhos::ProductEpetraVector(
            mp_f->map(), disc->getOverlapMap(), mp_x.productComm()));

  if (mp_overlapped_jac == Teuchos::null ||
      mp_overlapped_jac->size() != mp_jac.size())
    mp_overlapped_jac =
      rcp(new Stokhos::ProductContainer<Epetra_CrsMatrix>(
            mp_jac.map(), *overlapped_jac));

  for (int i=0; i<mp_x.size(); i++) {

    // Scatter x and xdot to the overlapped distrbution
    (*mp_overlapped_x)[i].Import(mp_x[i], *importer, Insert);
    if (mp_xdot != NULL) (*mp_overlapped_xdot)[i].Import((*mp_xdot)[i], *importer, Insert);
    if (mp_xdotdot != NULL) (*mp_overlapped_xdotdot)[i].Import((*mp_xdotdot)[i], *importer, Insert);

    // Zero out overlapped residual
    if (mp_f != NULL) {
      (*mp_overlapped_f)[i].PutScalar(0.0);
      (*mp_f)[i].PutScalar(0.0);
    }

    mp_jac[i].PutScalar(0.0);
    (*mp_overlapped_jac)[i].PutScalar(0.0);

  }

  // Set parameters
  for (int i=0; i<p.size(); i++)
    for (unsigned int j=0; j<p[i].size(); j++)
      p[i][j].family->setRealValueForAllTypes(p[i][j].baseValue);

  // put current_time (from Rythmos) if this is a transient problem, then compute dt
  //  if (mp_xdot != NULL) timeMgr.setTime(current_time);

#ifdef ALBANY_CUTR
  if (shapeParamsHaveBeenReset) {
    TEUCHOS_FUNC_TIME_MONITOR("Albany-Cubit MeshMover");
*out << " Calling moveMesh with params: " << std::setprecision(8);
for (unsigned int i=0; i<shapeParams.size(); i++) *out << shapeParams[i] << "  ";
*out << std::endl;
    meshMover->moveMesh(shapeParams, morphFromInit);
    coords = disc->getCoords();
    shapeParamsHaveBeenReset = false;
  }
#endif

  // Set MP parameters
  for (int i=0; i<mp_p_index.size(); i++) {
    int ii = mp_p_index[i];
    for (unsigned int j=0; j<p[ii].size(); j++)
      p[ii][j].family->setValue<PHAL::AlbanyTraits::MPJacobian>(mp_p_vals[ii][j]);
  }

  RCP< Stokhos::ProductEpetraVector > mp_overlapped_ff;
  if (mp_f != NULL)
    mp_overlapped_ff = mp_overlapped_f;

  // Set data in Workset struct, and perform fill via field manager
  {
    PHAL::Workset workset;

    workset.mp_x         = mp_overlapped_x;
    workset.mp_xdot      = mp_overlapped_xdot;
    workset.mp_xdotdot      = mp_overlapped_xdotdot;
    workset.mp_f         = mp_overlapped_ff;

    workset.mp_Jac       = mp_overlapped_jac;
    loadWorksetJacobianInfo(workset, alpha, beta, omega);
    workset.current_time = current_time; //timeMgr.getCurrentTime();
    //    workset.delta_time = timeMgr.getDeltaTime();
    if (mp_xdot != NULL) workset.transientTerms = true;
    if (mp_xdotdot != NULL) workset.accelerationTerms = true;

    for (int ws=0; ws < numWorksets; ws++) {
      loadWorksetBucketInfo<PHAL::AlbanyTraits::MPJacobian>(workset, ws);

      // FillType template argument used to specialize Sacado
      fm[wsPhysIndex[ws]]->evaluateFields<PHAL::AlbanyTraits::MPJacobian>(workset);
      if (nfm!=Teuchos::null)
        deref_nfm(nfm, wsPhysIndex, ws)->evaluateFields<PHAL::AlbanyTraits::MPJacobian>(workset);
    }
  }

  // Assemble global residual
  if (mp_f != NULL)
    for (int i=0; i<mp_f->size(); i++)
      (*mp_f)[i].Export((*mp_overlapped_f)[i], *exporter, Add);

  // Assemble block Jacobians
  RCP<Epetra_CrsMatrix> jac;
  for (int i=0; i<mp_jac.size(); i++) {
    jac = mp_jac.getCoeffPtr(i);
    jac->PutScalar(0.0);
    jac->Export((*mp_overlapped_jac)[i], *exporter, Add);
    jac->FillComplete(true);
  }

  // Apply Dirichlet conditions using dfm (Dirchelt Field Manager)
  if (dfm!=Teuchos::null) {
    PHAL::Workset workset;

    workset.mp_f = rcp(mp_f,false);
    workset.mp_Jac = Teuchos::rcpFromRef(mp_jac);
    workset.j_coeff = beta;
    workset.n_coeff = omega;
    workset.mp_x = Teuchos::rcpFromRef(mp_x);;
    if (mp_xdot != NULL) workset.transientTerms = true;
    if (mp_xdotdot != NULL) workset.accelerationTerms = true;

    loadWorksetNodesetInfo(workset);
    workset.distParamLib = distParamLib;

    workset.disc = disc;

#if defined(ALBANY_LCM)
    // Needed for more specialized Dirichlet BCs (e.g. Schwarz coupling)
    workset.apps_ = apps_;
    workset.current_app_ = Teuchos::rcp(this, false);
#endif

    // FillType template argument used to specialize Sacado
    dfm->evaluateFields<PHAL::AlbanyTraits::MPJacobian>(workset);
  }
}

void
Albany::Application::
computeGlobalMPTangent(
  const double alpha,
  const double beta,
  const double omega,
  const double current_time,
  bool sum_derivs,
  const Stokhos::ProductEpetraVector* mp_xdot,
  const Stokhos::ProductEpetraVector* mp_xdotdot,
  const Stokhos::ProductEpetraVector& mp_x,
  const Teuchos::Array<ParamVec>& par,
  const Teuchos::Array<int>& mp_p_index,
  const Teuchos::Array< Teuchos::Array<MPType> >& mp_p_vals,
  ParamVec* deriv_par,
  const Epetra_MultiVector* Vx,
  const Epetra_MultiVector* Vxdot,
  const Epetra_MultiVector* Vxdotdot,
  const Epetra_MultiVector* Vp,
  Stokhos::ProductEpetraVector* mp_f,
  Stokhos::ProductEpetraMultiVector* mp_JVx,
  Stokhos::ProductEpetraMultiVector* mp_fVp)
{
  TEUCHOS_FUNC_TIME_MONITOR("> Albany Fill: MPTangent");

  postRegSetup("MPTangent");

  // Load connectivity map and coordinates
  const WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > > >::type&
        wsElNodeEqID = disc->getWsElNodeEqID();
  const WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >::type&
        coords = disc->getCoords();
  const WorksetArray<std::string>::type& wsEBNames = disc->getWsEBNames();
  const WorksetArray<int>::type& wsPhysIndex = disc->getWsPhysIndex();

  int numWorksets = wsElNodeEqID.size();

  Teuchos::RCP<Epetra_Import>& importer = solMgr->get_importer();
  Teuchos::RCP<Epetra_Export>& exporter = solMgr->get_exporter();

  // Create overlapped multi-point Epetra objects
  if (mp_overlapped_x == Teuchos::null ||
      mp_overlapped_x->size() != mp_x.size()) {
    mp_overlapped_x =
      rcp(new Stokhos::ProductEpetraVector(
            mp_x.map(), disc->getOverlapMap(), mp_x.productComm()));

    if (mp_xdot != NULL)
      mp_overlapped_xdot =
        rcp(new Stokhos::ProductEpetraVector(
              mp_xdot->map(), disc->getOverlapMap(), mp_x.productComm()));

    if (mp_xdotdot != NULL)
      mp_overlapped_xdotdot =
        rcp(new Stokhos::ProductEpetraVector(
              mp_xdotdot->map(), disc->getOverlapMap(), mp_x.productComm()));

    if (mp_xdotdot != NULL)
      mp_overlapped_xdotdot =
	rcp(new Stokhos::ProductEpetraVector(
	      mp_xdotdot->map(), disc->getOverlapMap(), mp_x.productComm()));

  }

  if (mp_f != NULL && (mp_overlapped_f == Teuchos::null ||
                       mp_overlapped_f->size() != mp_f->size()))
    mp_overlapped_f =
      rcp(new Stokhos::ProductEpetraVector(
            mp_f->map(), disc->getOverlapMap(), mp_x.productComm()));

  for (int i=0; i<mp_x.size(); i++) {

    // Scatter x and xdot to the overlapped distrbution
    (*mp_overlapped_x)[i].Import(mp_x[i], *importer, Insert);
    if (mp_xdot != NULL) (*mp_overlapped_xdot)[i].Import((*mp_xdot)[i], *importer, Insert);
    if (mp_xdotdot != NULL) (*mp_overlapped_xdotdot)[i].Import((*mp_xdotdot)[i], *importer, Insert);

    // Zero out overlapped residual
    if (mp_f != NULL) {
      (*mp_overlapped_f)[i].PutScalar(0.0);
      (*mp_f)[i].PutScalar(0.0);
    }

  }

  // Scatter Vx to the overlapped distribution
  RCP<Epetra_MultiVector> overlapped_Vx;
  if (Vx != NULL) {
    overlapped_Vx =
      rcp(new Epetra_MultiVector(*(disc->getOverlapMap()), Vx->NumVectors()));
    overlapped_Vx->Import(*Vx, *importer, Insert);
  }

  // Scatter Vx dot to the overlapped distribution
  RCP<Epetra_MultiVector> overlapped_Vxdot;
  if (Vxdot != NULL) {
    overlapped_Vxdot = rcp(new Epetra_MultiVector(*(disc->getOverlapMap()), Vxdot->NumVectors()));
    overlapped_Vxdot->Import(*Vxdot, *importer, Insert);
  }
  RCP<Epetra_MultiVector> overlapped_Vxdotdot;
  if (Vxdotdot != NULL) {
    overlapped_Vxdotdot = rcp(new Epetra_MultiVector(*(disc->getOverlapMap()), Vxdotdot->NumVectors()));
    overlapped_Vxdotdot->Import(*Vxdotdot, *importer, Insert);
  }

  // Set parameters
  for (int i=0; i<par.size(); i++)
    for (unsigned int j=0; j<par[i].size(); j++)
      par[i][j].family->setRealValueForAllTypes(par[i][j].baseValue);

  // Set MP parameters
  for (int i=0; i<mp_p_index.size(); i++) {
    int ii = mp_p_index[i];
    for (unsigned int j=0; j<par[ii].size(); j++)
        par[ii][j].family->setValue<PHAL::AlbanyTraits::MPTangent>(mp_p_vals[ii][j]);
  }

  // put current_time (from Rythmos) if this is a transient problem, then compute dt
  //  if (mp_xdot != NULL) timeMgr.setTime(current_time);

  RCP<const Epetra_MultiVector > vp = rcp(Vp, false);
  RCP<ParamVec> params = rcp(deriv_par, false);

  RCP< Stokhos::ProductEpetraVector > mp_overlapped_ff;
  if (mp_f != NULL)
    mp_overlapped_ff = mp_overlapped_f;

  Teuchos::RCP< Stokhos::ProductEpetraMultiVector > mp_overlapped_JVx;
  if (mp_JVx != NULL) {
    mp_overlapped_JVx =
      Teuchos::rcp(new Stokhos::ProductEpetraMultiVector(
                     mp_JVx->map(), disc->getOverlapMap(), mp_x.productComm(),
                     mp_JVx->numVectors()));
    mp_JVx->init(0.0);
  }

  Teuchos::RCP<Stokhos::ProductEpetraMultiVector > mp_overlapped_fVp;
  if (mp_fVp != NULL) {
    mp_overlapped_fVp =
      Teuchos::rcp(new Stokhos::ProductEpetraMultiVector(
                     mp_fVp->map(), disc->getOverlapMap(), mp_x.productComm(),
                     mp_fVp->numVectors()));
    mp_fVp->init(0.0);
  }

  // Number of x & xdot tangent directions
  int num_cols_x = 0;
  if (Vx != NULL)
    num_cols_x = Vx->NumVectors();
  else if (Vxdot != NULL)
    num_cols_x = Vxdot->NumVectors();
  else if (Vxdotdot != NULL)
    num_cols_x = Vxdotdot->NumVectors();

  // Number of parameter tangent directions
  int num_cols_p = 0;
  if (params != Teuchos::null) {
    if (Vp != NULL)
      num_cols_p = Vp->NumVectors();
    else
      num_cols_p = params->size();
  }

  // Whether x and param tangent components are added or separate
  int param_offset = 0;
  if (!sum_derivs)
    param_offset = num_cols_x;  // offset of parameter derivs in deriv array

  TEUCHOS_TEST_FOR_EXCEPTION(sum_derivs &&
                             (num_cols_x != 0) &&
                             (num_cols_p != 0) &&
                             (num_cols_x != num_cols_p),
                             std::logic_error,
                             "Seed matrices Vx and Vp must have the same number " <<
                             " of columns when sum_derivs is true and both are "
                             << "non-null!" << std::endl);

  // Initialize
  if (params != Teuchos::null) {
    MPFadType p;
    int num_cols_tot = param_offset + num_cols_p;
    for (unsigned int i=0; i<params->size(); i++) {
      // Get the base value set above
      MPType base_val =
        (*params)[i].family->getValue<PHAL::AlbanyTraits::MPTangent>().val();
      p = MPFadType(num_cols_tot, base_val);
      if (Vp != NULL)
        for (int k=0; k<num_cols_p; k++)
          p.fastAccessDx(param_offset+k) = (*Vp)[k][i];
      else
        p.fastAccessDx(param_offset+i) = 1.0;
      (*params)[i].family->setValue<PHAL::AlbanyTraits::MPTangent>(p);
    }
  }

  // Set data in Workset struct, and perform fill via field manager
  {
    PHAL::Workset workset;

    workset.params = params;
    workset.mp_x         = mp_overlapped_x;
    workset.mp_xdot      = mp_overlapped_xdot;
    workset.mp_xdotdot      = mp_overlapped_xdotdot;
    workset.Vx = overlapped_Vx;
    workset.Vxdot = overlapped_Vxdot;
    workset.Vxdotdot = overlapped_Vxdotdot;
    workset.Vp = vp;

    workset.mp_f         = mp_overlapped_ff;
    workset.mp_JV        = mp_overlapped_JVx;
    workset.mp_fp        = mp_overlapped_fVp;
    workset.j_coeff      = beta;
    workset.m_coeff      = alpha;
    workset.n_coeff      = omega;

    workset.num_cols_x = num_cols_x;
    workset.num_cols_p = num_cols_p;
    workset.param_offset = param_offset;

    workset.current_time = current_time; //timeMgr.getCurrentTime();
    //    workset.delta_time = timeMgr.getDeltaTime();
    if (mp_xdot != NULL) workset.transientTerms = true;
    if (mp_xdotdot != NULL) workset.accelerationTerms = true;

    for (int ws=0; ws < numWorksets; ws++) {
      loadWorksetBucketInfo<PHAL::AlbanyTraits::MPTangent>(workset, ws);

      // FillType template argument used to specialize Sacado
      fm[wsPhysIndex[ws]]->evaluateFields<PHAL::AlbanyTraits::MPTangent>(workset);
      if (nfm!=Teuchos::null)
        deref_nfm(nfm, wsPhysIndex, ws)->evaluateFields<PHAL::AlbanyTraits::MPTangent>(workset);
    }
  }

  vp = Teuchos::null;
  params = Teuchos::null;

  // Assemble global residual
  if (mp_f != NULL)
    for (int i=0; i<mp_f->size(); i++)
      (*mp_f)[i].Export((*mp_overlapped_f)[i], *exporter, Add);

  // Assemble derivatives
  if (mp_JVx != NULL)
    for (int i=0; i<mp_JVx->size(); i++)
      (*mp_JVx)[i].Export((*mp_overlapped_JVx)[i], *exporter, Add);
  if (mp_fVp != NULL)
    for (int i=0; i<mp_fVp->size(); i++)
      (*mp_fVp)[i].Export((*mp_overlapped_fVp)[i], *exporter, Add);

  // Apply Dirichlet conditions using dfm (Dirchelt Field Manager)
  if (dfm!=Teuchos::null) {
    PHAL::Workset workset;

    workset.num_cols_x = num_cols_x;
    workset.num_cols_p = num_cols_p;
    workset.param_offset = param_offset;

    workset.mp_f = rcp(mp_f,false);
    workset.mp_fp = rcp(mp_fVp,false);
    workset.mp_JV = rcp(mp_JVx,false);
    workset.j_coeff = beta;
    workset.n_coeff = omega;
    workset.mp_x = Teuchos::rcpFromRef(mp_x);
    workset.Vx = rcp(Vx,false);
    if (mp_xdot != NULL) workset.transientTerms = true;
    if (mp_xdotdot != NULL) workset.accelerationTerms = true;

    loadWorksetNodesetInfo(workset);
    workset.distParamLib = distParamLib;

    workset.disc = disc;

    // FillType template argument used to specialize Sacado
#if defined(ALBANY_LCM)
    // Needed for more specialized Dirichlet BCs (e.g. Schwarz coupling)
    workset.apps_ = apps_;
    workset.current_app_ = Teuchos::rcp(this, false);
#endif

    dfm->evaluateFields<PHAL::AlbanyTraits::MPTangent>(workset);
  }

}

void
Albany::Application::
evaluateMPResponse(
  int response_index,
  const double curr_time,
  const Stokhos::ProductEpetraVector* mp_xdot,
  const Stokhos::ProductEpetraVector* mp_xdotdot,
  const Stokhos::ProductEpetraVector& mp_x,
  const Teuchos::Array<ParamVec>& p,
  const Teuchos::Array<int>& mp_p_index,
  const Teuchos::Array< Teuchos::Array<MPType> >& mp_p_vals,
  Stokhos::ProductEpetraVector& mp_g)
{
  TEUCHOS_FUNC_TIME_MONITOR("> Albany Fill: MPResponses");

  responses[response_index]->evaluateMPResponse(
    curr_time, mp_xdot, mp_xdotdot, mp_x, p, mp_p_index, mp_p_vals, mp_g);
}

void
Albany::Application::
evaluateMPResponseTangent(
  int response_index,
  const double alpha,
  const double beta,
  const double omega,
  const double current_time,
  bool sum_derivs,
  const Stokhos::ProductEpetraVector* mp_xdot,
  const Stokhos::ProductEpetraVector* mp_xdotdot,
  const Stokhos::ProductEpetraVector& mp_x,
  const Teuchos::Array<ParamVec>& p,
  const Teuchos::Array<int>& mp_p_index,
  const Teuchos::Array< Teuchos::Array<MPType> >& mp_p_vals,
  ParamVec* deriv_p,
  const Epetra_MultiVector* Vx,
  const Epetra_MultiVector* Vxdot,
  const Epetra_MultiVector* Vxdotdot,
  const Epetra_MultiVector* Vp,
  Stokhos::ProductEpetraVector* mp_g,
  Stokhos::ProductEpetraMultiVector* mp_JV,
  Stokhos::ProductEpetraMultiVector* mp_gp)
{
  TEUCHOS_FUNC_TIME_MONITOR("> Albany Fill: MPResponse Tangents");

  responses[response_index]->evaluateMPTangent(
    alpha, beta, omega, current_time, sum_derivs, mp_xdot, mp_xdotdot, mp_x, p, mp_p_index,
    mp_p_vals, deriv_p, Vx, Vxdot, Vxdotdot, Vp, mp_g, mp_JV, mp_gp);
}

void
Albany::Application::
evaluateMPResponseDerivative(
  int response_index,
  const double current_time,
  const Stokhos::ProductEpetraVector* mp_xdot,
  const Stokhos::ProductEpetraVector* mp_xdotdot,
  const Stokhos::ProductEpetraVector& mp_x,
  const Teuchos::Array<ParamVec>& p,
  const Teuchos::Array<int>& mp_p_index,
  const Teuchos::Array< Teuchos::Array<MPType> >& mp_p_vals,
  ParamVec* deriv_p,
  Stokhos::ProductEpetraVector* mp_g,
  const EpetraExt::ModelEvaluator::MPDerivative& mp_dg_dx,
  const EpetraExt::ModelEvaluator::MPDerivative& mp_dg_dxdot,
  const EpetraExt::ModelEvaluator::MPDerivative& mp_dg_dxdotdot,
  const EpetraExt::ModelEvaluator::MPDerivative& mp_dg_dp)
{
  TEUCHOS_FUNC_TIME_MONITOR("> Albany Fill: MPResponse Gradient");

  responses[response_index]->evaluateMPDerivative(
    current_time, mp_xdot, mp_xdotdot, mp_x, p, mp_p_index, mp_p_vals, deriv_p,
    mp_g, mp_dg_dx, mp_dg_dxdot, mp_dg_dxdotdot, mp_dg_dp);
}
#endif

#if defined(ALBANY_EPETRA)
void
Albany::Application::
evaluateStateFieldManager(const double current_time,
                          const Epetra_Vector* xdot,
                          const Epetra_Vector* xdotdot,
                          const Epetra_Vector& x)
{
  // Scatter x and xdot to the overlapped distrbution
  solMgr->scatterX(x, xdot, xdotdot);

  //Create Tpetra copy of x, called xT
  Teuchos::RCP<const Tpetra_Vector> xT = Petra::EpetraVector_To_TpetraVectorConst(x, commT);
  //Create Tpetra copy of xdot, called xdotT
  Teuchos::RCP<const Tpetra_Vector> xdotT;
  if (xdot != NULL) {
     xdotT = Petra::EpetraVector_To_TpetraVectorConst(*xdot, commT);
  }
  //Create Tpetra copy of xdotdot, called xdotdotT
  Teuchos::RCP<const Tpetra_Vector> xdotdotT;
  if (xdotdot != NULL) {
     xdotdotT = Petra::EpetraVector_To_TpetraVectorConst(*xdotdot, commT);
  }

  this->evaluateStateFieldManagerT(current_time, xdotT.ptr(), xdotdotT.ptr(), *xT);
}
#endif

void
Albany::Application::
evaluateStateFieldManagerT(
    const double current_time,
    Teuchos::Ptr<const Tpetra_Vector> xdotT,
    Teuchos::Ptr<const Tpetra_Vector> xdotdotT,
    const Tpetra_Vector& xT)
{
  {
    const std::string eval = "SFM_Jacobian";
    if (setupSet.find(eval) == setupSet.end()) {
      setupSet.insert(eval);
      for (int ps = 0; ps < sfm.size(); ++ps) {
        std::vector<PHX::index_size_type> derivative_dimensions;
        derivative_dimensions.push_back(
          PHAL::getDerivativeDimensions<PHAL::AlbanyTraits::Jacobian>(
            this, ps));
        sfm[ps]->setKokkosExtendedDataTypeDimensions
          <PHAL::AlbanyTraits::Jacobian>(derivative_dimensions);
        sfm[ps]->postRegistrationSetup("");
      }
      // visualize state field manager
      if (stateGraphVisDetail > 0) {
        bool detail = false; if (stateGraphVisDetail > 1) detail=true;
        *out << "Phalanx writing graphviz file for graph of Residual fill "
             "(detail =" << stateGraphVisDetail << ")" << std::endl;
        *out << "Process using 'dot -Tpng -O state_phalanx_graph' \n"
             << std::endl;
        for (int ps=0; ps < sfm.size(); ps++) {
          std::stringstream pg; pg << "state_phalanx_graph_" << ps;
          sfm[ps]->writeGraphvizFile<PHAL::AlbanyTraits::Residual>(
            pg.str(),detail,detail);
        }
        stateGraphVisDetail = -1;
      }
    }
  }

  Teuchos::RCP<Tpetra_Vector>& overlapped_fT = solMgrT->get_overlapped_fT();

  // Load connectivity map and coordinates
  const WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > > >::type&
        wsElNodeEqID = disc->getWsElNodeEqID();
  const WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >::type&
        coords = disc->getCoords();
  const WorksetArray<std::string>::type& wsEBNames = disc->getWsEBNames();
  const WorksetArray<int>::type& wsPhysIndex = disc->getWsPhysIndex();

  int numWorksets = wsElNodeEqID.size();

  // Scatter xT and xdotT to the overlapped distrbution
  solMgrT->scatterXT(xT, xdotT.get(), xdotdotT.get());

  //Scatter distributed parameters
  distParamLib->scatter();

  // Set data in Workset struct
  PHAL::Workset workset;
  loadBasicWorksetInfoT( workset, current_time );
  workset.fT = overlapped_fT;

  // Perform fill via field manager
  if (Teuchos::nonnull(rc_mgr)) rc_mgr->beginEvaluatingSfm();
  for (int ws=0; ws < numWorksets; ws++) {
    loadWorksetBucketInfo<PHAL::AlbanyTraits::Residual>(workset, ws);
    sfm[wsPhysIndex[ws]]->evaluateFields<PHAL::AlbanyTraits::Residual>(workset);
  }
  if (Teuchos::nonnull(rc_mgr)) rc_mgr->endEvaluatingSfm();
}

void Albany::Application::registerShapeParameters()
{
  int numShParams = shapeParams.size();
  if (shapeParamNames.size() == 0) {
    shapeParamNames.resize(numShParams);
    for (int i=0; i<numShParams; i++)
       shapeParamNames[i] = Albany::strint("ShapeParam",i);
  }
  Albany::DummyParameterAccessor<PHAL::AlbanyTraits::Jacobian, SPL_Traits> * dJ =
   new Albany::DummyParameterAccessor<PHAL::AlbanyTraits::Jacobian, SPL_Traits>();
  Albany::DummyParameterAccessor<PHAL::AlbanyTraits::Tangent, SPL_Traits> * dT =
   new Albany::DummyParameterAccessor<PHAL::AlbanyTraits::Tangent, SPL_Traits>();
#ifdef ALBANY_SG
  Albany::DummyParameterAccessor<PHAL::AlbanyTraits::SGResidual, SPL_Traits> * dSGR =
   new Albany::DummyParameterAccessor<PHAL::AlbanyTraits::SGResidual, SPL_Traits>();
  Albany::DummyParameterAccessor<PHAL::AlbanyTraits::SGJacobian, SPL_Traits> * dSGJ =
   new Albany::DummyParameterAccessor<PHAL::AlbanyTraits::SGJacobian, SPL_Traits>();
#endif 
#ifdef ALBANY_ENSEMBLE 
  Albany::DummyParameterAccessor<PHAL::AlbanyTraits::MPResidual, SPL_Traits> * dMPR =
   new Albany::DummyParameterAccessor<PHAL::AlbanyTraits::MPResidual, SPL_Traits>();
  Albany::DummyParameterAccessor<PHAL::AlbanyTraits::MPJacobian, SPL_Traits> * dMPJ =
   new Albany::DummyParameterAccessor<PHAL::AlbanyTraits::MPJacobian, SPL_Traits>();
#endif

  // Register Parameter for Residual fill using "this->getValue" but
  // create dummy ones for other type that will not be used.
  for (int i=0; i<numShParams; i++) {
    *out << "Registering Shape Param " << shapeParamNames[i] << std::endl;
    new Sacado::ParameterRegistration<PHAL::AlbanyTraits::Residual, SPL_Traits>
      (shapeParamNames[i], this, paramLib);
    new Sacado::ParameterRegistration<PHAL::AlbanyTraits::Jacobian, SPL_Traits>
      (shapeParamNames[i], dJ, paramLib);
    new Sacado::ParameterRegistration<PHAL::AlbanyTraits::Tangent, SPL_Traits>
      (shapeParamNames[i], dT, paramLib);
#ifdef ALBANY_SG
    new Sacado::ParameterRegistration<PHAL::AlbanyTraits::SGResidual, SPL_Traits>
      (shapeParamNames[i], dSGR, paramLib);
    new Sacado::ParameterRegistration<PHAL::AlbanyTraits::SGJacobian, SPL_Traits>
      (shapeParamNames[i], dSGJ, paramLib);
#endif 
#ifdef ALBANY_ENSEMBLE 
    new Sacado::ParameterRegistration<PHAL::AlbanyTraits::MPResidual, SPL_Traits>
      (shapeParamNames[i], dMPR, paramLib);
    new Sacado::ParameterRegistration<PHAL::AlbanyTraits::MPJacobian, SPL_Traits>
      (shapeParamNames[i], dMPJ, paramLib);
#endif
  }
}

PHAL::AlbanyTraits::Residual::ScalarT&
Albany::Application::getValue(const std::string& name)
{
  int index=-1;
  for (unsigned int i=0; i<shapeParamNames.size(); i++) {
    if (name == shapeParamNames[i]) index = i;
  }
  TEUCHOS_TEST_FOR_EXCEPTION(index==-1,  std::logic_error,
                             "Error in GatherCoordinateVector::getValue, \n" <<
                             "   Unrecognized param name: " << name << std::endl);

  shapeParamsHaveBeenReset = true;

  return shapeParams[index];
}


void Albany::Application::
postRegSetup(std::string eval)
{
  if (setupSet.find(eval) != setupSet.end())  return;

  setupSet.insert(eval);

  if (eval=="Residual") {
    for (int ps=0; ps < fm.size(); ps++)
      fm[ps]->postRegistrationSetupForType<PHAL::AlbanyTraits::Residual>(eval);
    if (dfm!=Teuchos::null)
      dfm->postRegistrationSetupForType<PHAL::AlbanyTraits::Residual>(eval);
    if (nfm!=Teuchos::null)
      for (int ps=0; ps < nfm.size(); ps++)
        nfm[ps]->postRegistrationSetupForType<PHAL::AlbanyTraits::Residual>(eval);
  }
  else if (eval=="Jacobian") {
    for (int ps=0; ps < fm.size(); ps++){
      std::vector<PHX::index_size_type> derivative_dimensions;
      derivative_dimensions.push_back(
        PHAL::getDerivativeDimensions<PHAL::AlbanyTraits::Jacobian>(this, ps));
      fm[ps]->setKokkosExtendedDataTypeDimensions<PHAL::AlbanyTraits::Jacobian>(derivative_dimensions);
      fm[ps]->postRegistrationSetupForType<PHAL::AlbanyTraits::Jacobian>(eval);
      if (nfm!=Teuchos::null && ps < nfm.size()) {
        nfm[ps]->setKokkosExtendedDataTypeDimensions<PHAL::AlbanyTraits::Jacobian>(derivative_dimensions);
        nfm[ps]->postRegistrationSetupForType<PHAL::AlbanyTraits::Jacobian>(eval);
      }
    }
    if (dfm!=Teuchos::null){
      //amb Need to look into this. What happens with DBCs in meshes having
      // different element types?
      std::vector<PHX::index_size_type> derivative_dimensions;
      derivative_dimensions.push_back(
        PHAL::getDerivativeDimensions<PHAL::AlbanyTraits::Jacobian>(this, 0));
      dfm->setKokkosExtendedDataTypeDimensions<PHAL::AlbanyTraits::Jacobian>(derivative_dimensions);
      dfm->postRegistrationSetupForType<PHAL::AlbanyTraits::Jacobian>(eval);
    }
  }
  else if (eval=="Tangent") {
    for (int ps=0; ps < fm.size(); ps++){
      std::vector<PHX::index_size_type> derivative_dimensions;
      derivative_dimensions.push_back(
        PHAL::getDerivativeDimensions<PHAL::AlbanyTraits::Tangent>(this, ps));
      fm[ps]->setKokkosExtendedDataTypeDimensions<PHAL::AlbanyTraits::Tangent>(derivative_dimensions);
      fm[ps]->postRegistrationSetupForType<PHAL::AlbanyTraits::Tangent>(eval);
      if (nfm!=Teuchos::null && ps < nfm.size()) {
        nfm[ps]->setKokkosExtendedDataTypeDimensions<PHAL::AlbanyTraits::Tangent>(derivative_dimensions);
        nfm[ps]->postRegistrationSetupForType<PHAL::AlbanyTraits::Tangent>(eval);
      }
    }
    if (dfm!=Teuchos::null){
      //amb Need to look into this. What happens with DBCs in meshes having
      // different element types?
      std::vector<PHX::index_size_type> derivative_dimensions;
      derivative_dimensions.push_back(
        PHAL::getDerivativeDimensions<PHAL::AlbanyTraits::Tangent>(this, 0));
      dfm->setKokkosExtendedDataTypeDimensions<PHAL::AlbanyTraits::Tangent>(derivative_dimensions);
      dfm->postRegistrationSetupForType<PHAL::AlbanyTraits::Tangent>(eval);
      }
  }
  else if (eval=="Distributed Parameter Derivative") { //!!!
    for (int ps=0; ps < fm.size(); ps++) {
      std::vector<PHX::index_size_type> derivative_dimensions;
      derivative_dimensions.push_back(
        PHAL::getDerivativeDimensions<PHAL::AlbanyTraits::DistParamDeriv>(this, ps));
      fm[ps]->setKokkosExtendedDataTypeDimensions<PHAL::AlbanyTraits::DistParamDeriv>(derivative_dimensions);
      fm[ps]->postRegistrationSetupForType<PHAL::AlbanyTraits::DistParamDeriv>(eval);
    }
    if (dfm!=Teuchos::null) {
      std::vector<PHX::index_size_type> derivative_dimensions;
      derivative_dimensions.push_back(
        PHAL::getDerivativeDimensions<PHAL::AlbanyTraits::DistParamDeriv>(this, 0));
      dfm->setKokkosExtendedDataTypeDimensions<PHAL::AlbanyTraits::DistParamDeriv>(derivative_dimensions);
      dfm->postRegistrationSetupForType<PHAL::AlbanyTraits::DistParamDeriv>(eval);
    }
    if (nfm!=Teuchos::null)
      for (int ps=0; ps < nfm.size(); ps++) {
        std::vector<PHX::index_size_type> derivative_dimensions;
        derivative_dimensions.push_back(
          PHAL::getDerivativeDimensions<PHAL::AlbanyTraits::DistParamDeriv>(this, ps));
        nfm[ps]->setKokkosExtendedDataTypeDimensions<PHAL::AlbanyTraits::DistParamDeriv>(derivative_dimensions);
        nfm[ps]->postRegistrationSetupForType<PHAL::AlbanyTraits::DistParamDeriv>(eval);
      }
  }
#ifdef ALBANY_SG
  else if (eval=="SGResidual") {
    for (int ps=0; ps < fm.size(); ps++)
      fm[ps]->postRegistrationSetupForType<PHAL::AlbanyTraits::SGResidual>(eval);
    if (dfm!=Teuchos::null)
      dfm->postRegistrationSetupForType<PHAL::AlbanyTraits::SGResidual>(eval);
    if (nfm!=Teuchos::null)
      for (int ps=0; ps < nfm.size(); ps++)
        nfm[ps]->postRegistrationSetupForType<PHAL::AlbanyTraits::SGResidual>(eval);
  }
  else if (eval=="SGJacobian") {
    for (int ps=0; ps < fm.size(); ps++){
      std::vector<PHX::index_size_type> derivative_dimensions;
      // Deriv dimension for SGJacobian is retrieved through Jacobian eval type
      derivative_dimensions.push_back(
        PHAL::getDerivativeDimensions<PHAL::AlbanyTraits::Jacobian>(this, ps));
      fm[ps]->setKokkosExtendedDataTypeDimensions<PHAL::AlbanyTraits::SGJacobian>(derivative_dimensions);
      fm[ps]->postRegistrationSetupForType<PHAL::AlbanyTraits::SGJacobian>(eval);
      if (nfm!=Teuchos::null && ps < nfm.size()) {
        nfm[ps]->setKokkosExtendedDataTypeDimensions<PHAL::AlbanyTraits::SGJacobian>(derivative_dimensions);
        nfm[ps]->postRegistrationSetupForType<PHAL::AlbanyTraits::SGJacobian>(eval);
      }
    }
    if (dfm!=Teuchos::null){
      //amb Need to look into this. What happens with DBCs in meshes having
      // different element types?
      std::vector<PHX::index_size_type> derivative_dimensions;
      derivative_dimensions.push_back(
        PHAL::getDerivativeDimensions<PHAL::AlbanyTraits::Jacobian>(this, 0));
      dfm->setKokkosExtendedDataTypeDimensions<PHAL::AlbanyTraits::SGJacobian>(derivative_dimensions);
      dfm->postRegistrationSetupForType<PHAL::AlbanyTraits::SGJacobian>(eval);
    }
  }
  else if (eval=="SGTangent") {
    for (int ps=0; ps < fm.size(); ps++){
      std::vector<PHX::index_size_type> derivative_dimensions;
      // Deriv dimension for SGTangent is retrieved through Tangent eval type
      derivative_dimensions.push_back(
        PHAL::getDerivativeDimensions<PHAL::AlbanyTraits::Tangent>(this, ps));
      fm[ps]->setKokkosExtendedDataTypeDimensions<PHAL::AlbanyTraits::SGTangent>(derivative_dimensions);
      fm[ps]->postRegistrationSetupForType<PHAL::AlbanyTraits::SGTangent>(eval);
      if (nfm!=Teuchos::null && ps < nfm.size()) {
        nfm[ps]->setKokkosExtendedDataTypeDimensions<PHAL::AlbanyTraits::SGTangent>(derivative_dimensions);
        nfm[ps]->postRegistrationSetupForType<PHAL::AlbanyTraits::SGTangent>(eval);
      }
    }
    if (dfm!=Teuchos::null){
      //amb Need to look into this. What happens with DBCs in meshes having
      // different element types?
      std::vector<PHX::index_size_type> derivative_dimensions;
      derivative_dimensions.push_back(
        PHAL::getDerivativeDimensions<PHAL::AlbanyTraits::Tangent>(this, 0));
      dfm->setKokkosExtendedDataTypeDimensions<PHAL::AlbanyTraits::SGTangent>(derivative_dimensions);
      dfm->postRegistrationSetupForType<PHAL::AlbanyTraits::SGTangent>(eval);
      }
  }
#endif 
#ifdef ALBANY_ENSEMBLE 
  else if (eval=="MPResidual") {
    for (int ps=0; ps < fm.size(); ps++)
      fm[ps]->postRegistrationSetupForType<PHAL::AlbanyTraits::MPResidual>(eval);
    if (dfm!=Teuchos::null)
      dfm->postRegistrationSetupForType<PHAL::AlbanyTraits::MPResidual>(eval);
    if (nfm!=Teuchos::null)
      for (int ps=0; ps < nfm.size(); ps++)
        nfm[ps]->postRegistrationSetupForType<PHAL::AlbanyTraits::MPResidual>(eval);
  }
  else if (eval=="MPJacobian") {
    for (int ps=0; ps < fm.size(); ps++){
      std::vector<PHX::index_size_type> derivative_dimensions;
      // Deriv dimension for MPJacobian is retrieved through Jacobian eval type
      derivative_dimensions.push_back(
        PHAL::getDerivativeDimensions<PHAL::AlbanyTraits::Jacobian>(this, ps));
      fm[ps]->setKokkosExtendedDataTypeDimensions<PHAL::AlbanyTraits::MPJacobian>(derivative_dimensions);
      fm[ps]->postRegistrationSetupForType<PHAL::AlbanyTraits::MPJacobian>(eval);
      if (nfm!=Teuchos::null && ps < nfm.size()) {
        nfm[ps]->setKokkosExtendedDataTypeDimensions<PHAL::AlbanyTraits::MPJacobian>(derivative_dimensions);
        nfm[ps]->postRegistrationSetupForType<PHAL::AlbanyTraits::MPJacobian>(eval);
      }
    }
    if (dfm!=Teuchos::null){
      //amb Need to look into this. What happens with DBCs in meshes having
      // different element types?
      std::vector<PHX::index_size_type> derivative_dimensions;
      derivative_dimensions.push_back(
        PHAL::getDerivativeDimensions<PHAL::AlbanyTraits::Jacobian>(this, 0));
      dfm->setKokkosExtendedDataTypeDimensions<PHAL::AlbanyTraits::MPJacobian>(derivative_dimensions);
      dfm->postRegistrationSetupForType<PHAL::AlbanyTraits::MPJacobian>(eval);
    }
  }
  else if (eval=="MPTangent") {
    for (int ps=0; ps < fm.size(); ps++){
      std::vector<PHX::index_size_type> derivative_dimensions;
      // Deriv dimension for MPTangent is retrieved through Tangent eval type
      derivative_dimensions.push_back(
        PHAL::getDerivativeDimensions<PHAL::AlbanyTraits::Tangent>(this, ps));
      fm[ps]->setKokkosExtendedDataTypeDimensions<PHAL::AlbanyTraits::MPTangent>(derivative_dimensions);
      fm[ps]->postRegistrationSetupForType<PHAL::AlbanyTraits::MPTangent>(eval);
      if (nfm!=Teuchos::null && ps < nfm.size()) {
        nfm[ps]->setKokkosExtendedDataTypeDimensions<PHAL::AlbanyTraits::MPTangent>(derivative_dimensions);
        nfm[ps]->postRegistrationSetupForType<PHAL::AlbanyTraits::MPTangent>(eval);
      }
    }
    if (dfm!=Teuchos::null){
      //amb Need to look into this. What happens with DBCs in meshes having
      // different element types?
      std::vector<PHX::index_size_type> derivative_dimensions;
      derivative_dimensions.push_back(
        PHAL::getDerivativeDimensions<PHAL::AlbanyTraits::Tangent>(this, 0));
      dfm->setKokkosExtendedDataTypeDimensions<PHAL::AlbanyTraits::MPTangent>(derivative_dimensions);
      dfm->postRegistrationSetupForType<PHAL::AlbanyTraits::MPTangent>(eval);
      }
  }
#endif
  else
    TEUCHOS_TEST_FOR_EXCEPTION(eval!="Known Evaluation Name",  std::logic_error,
                               "Error in setup call \n" << " Unrecognized name: " << eval << std::endl);


  // Write out Phalanx Graph if requested, on Proc 0, for Resid or Jacobian
  if (phxGraphVisDetail>0) {
    bool detail = false; if (phxGraphVisDetail > 1) detail=true;

    if (eval=="Residual") {
      *out << "Phalanx writing graphviz file for graph of Residual fill (detail ="
           << phxGraphVisDetail << ")"<<std::endl;
      *out << "Process using 'dot -Tpng -O phalanx_graph' \n" << std::endl;
      for (int ps=0; ps < fm.size(); ps++) {
        std::stringstream pg; pg << "phalanx_graph_" << ps;
        fm[ps]->writeGraphvizFile<PHAL::AlbanyTraits::Residual>(pg.str(),detail,detail);
      }
//      phxGraphVisDetail = -1;
    }
    else if (eval=="Jacobian") {
      *out << "Phalanx writing graphviz file for graph of Jacobian fill (detail ="
           << phxGraphVisDetail << ")"<<std::endl;
      *out << "Process using 'dot -Tpng -O phalanx_graph' \n" << std::endl;
      for (int ps=0; ps < fm.size(); ps++) {
        std::stringstream pg; pg << "phalanx_graph_jac_" << ps;
        fm[ps]->writeGraphvizFile<PHAL::AlbanyTraits::Jacobian>(pg.str(),detail,detail);
      }
      phxGraphVisDetail = -2;
    }
  }
}

#if defined(ALBANY_EPETRA) && defined(ALBANY_TEKO)
RCP<Epetra_Operator>
Albany::Application::buildWrappedOperator(const RCP<Epetra_Operator>& Jac,
                                          const RCP<Epetra_Operator>& wrapInput,
                                          bool reorder) const
{
  RCP<Epetra_Operator> wrappedOp = wrapInput;
  // if only one block just use orignal jacobian
  if(blockDecomp.size()==1) return (Jac);

  // initialize jacobian
  if(wrappedOp==Teuchos::null)
     wrappedOp = rcp(new Teko::Epetra::StridedEpetraOperator(blockDecomp,Jac));
  else
     rcp_dynamic_cast<Teko::Epetra::StridedEpetraOperator>(wrappedOp)->RebuildOps();

  // test blocked operator for correctness
  if(tekoParams->get("Test Blocked Operator",false)) {
     bool result
        = rcp_dynamic_cast<Teko::Epetra::StridedEpetraOperator>(wrappedOp)->testAgainstFullOperator(6,1e-14);

     *out << "Teko: Tested operator correctness:  " << (result ? "passed" : "FAILED!") << std::endl;
  }
  return wrappedOp;
}
#endif 

void
Albany::Application::determinePiroSolver(const Teuchos::RCP<Teuchos::ParameterList>& topLevelParams){

  const Teuchos::RCP<Teuchos::ParameterList>& localProblemParams =
    Teuchos::sublist(topLevelParams, "Problem", true);

  const Teuchos::RCP<Teuchos::ParameterList>& piroParams = Teuchos::sublist(topLevelParams, "Piro");

  // If not explicitly specified, determine which Piro solver to use from the problem parameters
  if (!piroParams->getPtr<std::string>("Solver Type")) {

    const std::string secondOrder = localProblemParams->get("Second Order", "No");

    TEUCHOS_TEST_FOR_EXCEPTION(
        secondOrder != "No" &&
        secondOrder != "Velocity Verlet" &&
        secondOrder != "Newmark" &&
        secondOrder != "Trapezoid Rule",
        std::logic_error,
        "Invalid value for Second Order: (No, Velocity Verlet, Newmark, Trapezoid Rule): " <<
        secondOrder <<
        "\n");

    // Populate the Piro parameter list accordingly to inform the Piro solver factory
    std::string piroSolverToken;
    if (solMethod == Steady) {
      piroSolverToken = "NOX";
    } else if (solMethod == Continuation) {
      piroSolverToken = "LOCA";
    } else if (solMethod == Transient) {
      piroSolverToken = (secondOrder == "No") ? "Rythmos" : secondOrder;
    } else {
      // Piro cannot handle the corresponding problem
      piroSolverToken = "Unsupported";
    }

    piroParams->set("Solver Type", piroSolverToken);

  }

}

#if defined(ALBANY_EPETRA)
void Albany::Application::loadBasicWorksetInfo(
       PHAL::Workset& workset,
       double current_time)
{
    workset.numEqs = neq;
    workset.x        = solMgr->get_overlapped_x();
    workset.xdot     = solMgr->get_overlapped_xdot();
    workset.xdotdot     = solMgr->get_overlapped_xdotdot();
    workset.current_time = current_time;
    workset.distParamLib = distParamLib;
    //workset.delta_time = delta_time;
    if (workset.xdot != Teuchos::null) workset.transientTerms = true;
    if (workset.xdotdot != Teuchos::null) workset.accelerationTerms = true;
}
#endif


void Albany::Application::loadBasicWorksetInfoT(
       PHAL::Workset& workset,
       double current_time)
{
    workset.numEqs = neq;
    workset.xT        = solMgrT->get_overlapped_xT();
    workset.xdotT     = solMgrT->get_overlapped_xdotT();
    workset.xdotdotT     = solMgrT->get_overlapped_xdotdotT();
    workset.current_time = current_time;
    workset.distParamLib = distParamLib;
    //workset.delta_time = delta_time;
    if (workset.xdotT != Teuchos::null) workset.transientTerms = true;
    if (workset.xdotdotT != Teuchos::null) workset.accelerationTerms = true;
}

void Albany::Application::loadWorksetJacobianInfo(PHAL::Workset& workset,
                                 const double& alpha, const double& beta, const double& omega)
{
    workset.m_coeff      = alpha;
    workset.n_coeff      = omega;
    workset.j_coeff      = beta;
    workset.ignore_residual = ignore_residual_in_jacobian;
    workset.is_adjoint   = is_adjoint;
}

void Albany::Application::loadWorksetNodesetInfo(PHAL::Workset& workset)
{
    workset.nodeSets = Teuchos::rcpFromRef(disc->getNodeSets());
    workset.nodeSetCoords = Teuchos::rcpFromRef(disc->getNodeSetCoords());

}

void Albany::Application::loadWorksetSidesetInfo(PHAL::Workset& workset, const int ws)
{

    workset.sideSets = Teuchos::rcpFromRef(disc->getSideSets(ws));

}

#if defined(ALBANY_EPETRA)
void Albany::Application::setupBasicWorksetInfo(
  PHAL::Workset& workset,
  double current_time,
  const Epetra_Vector* xdot,
  const Epetra_Vector* xdotdot,
  const Epetra_Vector* x,
  const Teuchos::Array<ParamVec>& p
  )
{

  Teuchos::RCP<Epetra_Vector>& overlapped_x = solMgr->get_overlapped_x();
  Teuchos::RCP<Epetra_Vector>& overlapped_xdot = solMgr->get_overlapped_xdot();
  Teuchos::RCP<Epetra_Vector>& overlapped_xdotdot = solMgr->get_overlapped_xdotdot();
  Teuchos::RCP<Epetra_Import>& importer = solMgr->get_importer();

  // Scatter x and xdot to the overlapped distrbution
  solMgr->scatterX(*x, xdot, xdotdot);

  //Scatter distributed parameters
  distParamLib->scatter();

  // Set parameters
  for (int i=0; i<p.size(); i++)
    for (unsigned int j=0; j<p[i].size(); j++)
      p[i][j].family->setRealValueForAllTypes(p[i][j].baseValue);

  workset.x = overlapped_x;
  workset.xdot = overlapped_xdot;
  workset.xdotdot = overlapped_xdotdot;
  workset.distParamLib = distParamLib;

  if (!paramLib->isParameter("Time"))
    workset.current_time = current_time;
  else
    workset.current_time =
      paramLib->getRealValue<PHAL::AlbanyTraits::Residual>("Time");
  if (overlapped_xdot != Teuchos::null) workset.transientTerms = true;
  if (overlapped_xdotdot != Teuchos::null) workset.accelerationTerms = true;

  // Create Teuchos::Comm from Epetra_Comm
  const Epetra_Comm& comm = x->Map().Comm();
  workset.comm = Albany::createTeuchosCommFromMpiComm(
                  Albany::getMpiCommFromEpetraComm(comm));

  workset.x_importer = importer;
}
#endif

void Albany::Application::setupBasicWorksetInfoT(
  PHAL::Workset& workset,
  double current_time,
  Teuchos::RCP<const Tpetra_Vector> xdotT,
  Teuchos::RCP<const Tpetra_Vector> xdotdotT,
  Teuchos::RCP<const Tpetra_Vector> xT,
  const Teuchos::Array<ParamVec>& p
  )
{
  Teuchos::RCP<Tpetra_Vector>& overlapped_xT = solMgrT->get_overlapped_xT();
  Teuchos::RCP<Tpetra_Vector>& overlapped_xdotT = solMgrT->get_overlapped_xdotT();
  Teuchos::RCP<Tpetra_Vector>& overlapped_xdotdotT = solMgrT->get_overlapped_xdotdotT();
  Teuchos::RCP<Tpetra_Import>& importerT = solMgrT->get_importerT();

  // Scatter xT and xdotT to the overlapped distrbution
  solMgrT->scatterXT(*xT, xdotT.get(), xdotdotT.get());

  //Scatter distributed parameters
  distParamLib->scatter();

  // Set parameters
  for (int i=0; i<p.size(); i++)
    for (unsigned int j=0; j<p[i].size(); j++)
      p[i][j].family->setRealValueForAllTypes(p[i][j].baseValue);

  workset.xT = overlapped_xT;
  workset.xdotT = overlapped_xdotT;
  workset.xdotdotT = overlapped_xdotdotT;
  workset.distParamLib = distParamLib;
  if (!paramLib->isParameter("Time"))
    workset.current_time = current_time;
  else
    workset.current_time =
      paramLib->getRealValue<PHAL::AlbanyTraits::Residual>("Time");
  if (overlapped_xdotT != Teuchos::null) workset.transientTerms = true;
  if (overlapped_xdotdotT != Teuchos::null) workset.accelerationTerms = true;

  workset.comm = commT;

  workset.x_importerT = importerT;

}


#ifdef ALBANY_SG
void Albany::Application::setupBasicWorksetInfo(
  PHAL::Workset& workset,
  double current_time,
  const Stokhos::EpetraVectorOrthogPoly* sg_xdot,
  const Stokhos::EpetraVectorOrthogPoly* sg_xdotdot,
  const Stokhos::EpetraVectorOrthogPoly* sg_x,
  const Teuchos::Array<ParamVec>& p,
  const Teuchos::Array<int>& sg_p_index,
  const Teuchos::Array< Teuchos::Array<SGType> >& sg_p_vals)
{
  Teuchos::RCP<Epetra_Import>& importer = solMgr->get_importer();

  // Scatter x and xdot to the overlapped distrbution
  for (int i=0; i<sg_x->size(); i++) {
    (*sg_overlapped_x)[i].Import((*sg_x)[i], *importer, Insert);
    if (sg_xdot != NULL) (*sg_overlapped_xdot)[i].Import((*sg_xdot)[i], *importer, Insert);
    if (sg_xdotdot != NULL) (*sg_overlapped_xdotdot)[i].Import((*sg_xdotdot)[i], *importer, Insert);
  }

  // Set parameters
  for (int i=0; i<p.size(); i++)
    for (unsigned int j=0; j<p[i].size(); j++)
      p[i][j].family->setRealValueForAllTypes(p[i][j].baseValue);

  // Set SG parameters
  for (int i=0; i<sg_p_index.size(); i++) {
    int ii = sg_p_index[i];
    for (unsigned int j=0; j<p[ii].size(); j++) {
      p[ii][j].family->setValue<PHAL::AlbanyTraits::SGResidual>(sg_p_vals[ii][j]);
      p[ii][j].family->setValue<PHAL::AlbanyTraits::SGTangent>(sg_p_vals[ii][j]);
      p[ii][j].family->setValue<PHAL::AlbanyTraits::SGJacobian>(sg_p_vals[ii][j]);
    }
  }

  workset.sg_expansion = sg_expansion;
  workset.sg_x         = sg_overlapped_x;
  workset.sg_xdot      = sg_overlapped_xdot;
  workset.sg_xdotdot      = sg_overlapped_xdotdot;
  if (!paramLib->isParameter("Time"))
    workset.current_time = current_time;
  else
    workset.current_time =
      paramLib->getRealValue<PHAL::AlbanyTraits::Residual>("Time");
  if (sg_xdot != NULL) workset.transientTerms = true;
  if (sg_xdotdot != NULL) workset.accelerationTerms = true;

  // Create Teuchos::Comm from Epetra_Comm
  const Epetra_Comm& comm = sg_x->coefficientMap()->Comm();
  workset.comm = Albany::createTeuchosCommFromMpiComm(
                  Albany::getMpiCommFromEpetraComm(comm));

  workset.x_importer = importer;
}

#endif 
#ifdef ALBANY_ENSEMBLE 
void Albany::Application::setupBasicWorksetInfo(
  PHAL::Workset& workset,
  double current_time,
  const Stokhos::ProductEpetraVector* mp_xdot,
  const Stokhos::ProductEpetraVector* mp_xdotdot,
  const Stokhos::ProductEpetraVector* mp_x,
  const Teuchos::Array<ParamVec>& p,
  const Teuchos::Array<int>& mp_p_index,
  const Teuchos::Array< Teuchos::Array<MPType> >& mp_p_vals)
{
  Teuchos::RCP<Epetra_Import>& importer = solMgr->get_importer();

  // Scatter x and xdot to the overlapped distrbution
  for (int i=0; i<mp_x->size(); i++) {
    (*mp_overlapped_x)[i].Import((*mp_x)[i], *importer, Insert);
    if (mp_xdot != NULL) (*mp_overlapped_xdot)[i].Import((*mp_xdot)[i], *importer, Insert);
    if (mp_xdotdot != NULL) (*mp_overlapped_xdotdot)[i].Import((*mp_xdotdot)[i], *importer, Insert);
  }

  // Set parameters
  for (int i=0; i<p.size(); i++)
    for (unsigned int j=0; j<p[i].size(); j++)
      p[i][j].family->setRealValueForAllTypes(p[i][j].baseValue);

  // Set MP parameters
  for (int i=0; i<mp_p_index.size(); i++) {
    int ii = mp_p_index[i];
    for (unsigned int j=0; j<p[ii].size(); j++) {
      p[ii][j].family->setValue<PHAL::AlbanyTraits::MPResidual>(mp_p_vals[ii][j]);
      p[ii][j].family->setValue<PHAL::AlbanyTraits::MPTangent>(mp_p_vals[ii][j]);
      p[ii][j].family->setValue<PHAL::AlbanyTraits::MPJacobian>(mp_p_vals[ii][j]);
    }
  }

  workset.mp_x         = mp_overlapped_x;
  workset.mp_xdot      = mp_overlapped_xdot;
  workset.mp_xdotdot      = mp_overlapped_xdotdot;
  if (!paramLib->isParameter("Time"))
    workset.current_time = current_time;
  else
    workset.current_time =
      paramLib->getRealValue<PHAL::AlbanyTraits::Residual>("Time");
  if (mp_xdot != NULL) workset.transientTerms = true;
  if (mp_xdotdot != NULL) workset.accelerationTerms = true;

  // Create Teuchos::Comm from Epetra_Comm
  const Epetra_Comm& comm = mp_x->coefficientMap()->Comm();
  workset.comm = Albany::createTeuchosCommFromMpiComm(
                  Albany::getMpiCommFromEpetraComm(comm));

  workset.x_importer = importer;
}
#endif

#if defined(ALBANY_EPETRA)
void Albany::Application::setupTangentWorksetInfo(
  PHAL::Workset& workset,
  double current_time,
  bool sum_derivs,
  const Epetra_Vector* xdot,
  const Epetra_Vector* xdotdot,
  const Epetra_Vector* x,
  const Teuchos::Array<ParamVec>& p,
  ParamVec* deriv_p,
  const Epetra_MultiVector* Vxdot,
  const Epetra_MultiVector* Vxdotdot,
  const Epetra_MultiVector* Vx,
  const Epetra_MultiVector* Vp)
{
  setupBasicWorksetInfo(workset, current_time, xdot, xdotdot, x, p);

  Teuchos::RCP<Epetra_Import>& importer = solMgr->get_importer();

  // Scatter Vx dot the overlapped distribution
  RCP<Epetra_MultiVector> overlapped_Vx;
  if (Vx != NULL) {
    overlapped_Vx =
      rcp(new Epetra_MultiVector(*(disc->getOverlapMap()),
                                          Vx->NumVectors()));
    overlapped_Vx->Import(*Vx, *importer, Insert);
  }

  // Scatter Vx dot the overlapped distribution
  RCP<Epetra_MultiVector> overlapped_Vxdot;
  if (Vxdot != NULL) {
    overlapped_Vxdot = rcp(new Epetra_MultiVector(*(disc->getOverlapMap()), Vxdot->NumVectors()));
    overlapped_Vxdot->Import(*Vxdot, *importer, Insert);
  }
  RCP<Epetra_MultiVector> overlapped_Vxdotdot;
  if (Vxdotdot != NULL) {
    overlapped_Vxdotdot = rcp(new Epetra_MultiVector(*(disc->getOverlapMap()), Vxdotdot->NumVectors()));
    overlapped_Vxdotdot->Import(*Vxdotdot, *importer, Insert);
  }

  RCP<const Epetra_MultiVector > vp = rcp(Vp, false);
  RCP<ParamVec> params = rcp(deriv_p, false);

  // Number of x & xdot tangent directions
  int num_cols_x = 0;
  if (Vx != NULL)
    num_cols_x = Vx->NumVectors();
  else if (Vxdot != NULL)
    num_cols_x = Vxdot->NumVectors();
  else if (Vxdotdot != NULL)
    num_cols_x = Vxdotdot->NumVectors();

  // Number of parameter tangent directions
  int num_cols_p = 0;
  if (params != Teuchos::null) {
    if (Vp != NULL)
      num_cols_p = Vp->NumVectors();
    else
      num_cols_p = params->size();
  }

  // Whether x and param tangent components are added or separate
  int param_offset = 0;
  if (!sum_derivs)
    param_offset = num_cols_x;  // offset of parameter derivs in deriv array

  TEUCHOS_TEST_FOR_EXCEPTION(
    sum_derivs &&
    (num_cols_x != 0) &&
    (num_cols_p != 0) &&
    (num_cols_x != num_cols_p),
    std::logic_error,
    "Seed matrices Vx and Vp must have the same number " <<
    " of columns when sum_derivs is true and both are "
    << "non-null!" << std::endl);

  // Initialize
  if (params != Teuchos::null) {
    TanFadType p;
    int num_cols_tot = param_offset + num_cols_p;
    for (unsigned int i=0; i<params->size(); i++) {
      p = TanFadType(num_cols_tot, (*params)[i].baseValue);
      if (Vp != NULL)
        for (int k=0; k<num_cols_p; k++)
          p.fastAccessDx(param_offset+k) = (*Vp)[k][i];
      else
        p.fastAccessDx(param_offset+i) = 1.0;
      (*params)[i].family->setValue<PHAL::AlbanyTraits::Tangent>(p);
    }
  }

  workset.params = params;
  workset.Vx = overlapped_Vx;
  workset.Vxdot = overlapped_Vxdot;
  workset.Vxdotdot = overlapped_Vxdotdot;
  workset.Vp = vp;
  workset.num_cols_x = num_cols_x;
  workset.num_cols_p = num_cols_p;
  workset.param_offset = param_offset;
}
#endif

void Albany::Application::setupTangentWorksetInfoT(
  PHAL::Workset& workset,
  double current_time,
  bool sum_derivs,
  Teuchos::RCP<const Tpetra_Vector> xdotT,
  Teuchos::RCP<const Tpetra_Vector> xdotdotT,
  Teuchos::RCP<const Tpetra_Vector> xT,
  const Teuchos::Array<ParamVec>& p,
  ParamVec* deriv_p,
  Teuchos::RCP<const Tpetra_MultiVector> VxdotT,
  Teuchos::RCP<const Tpetra_MultiVector> VxdotdotT,
  Teuchos::RCP<const Tpetra_MultiVector> VxT,
  Teuchos::RCP<const Tpetra_MultiVector> VpT)
{
  setupBasicWorksetInfoT(workset, current_time, xdotT, xdotdotT, xT, p);

  Teuchos::RCP<Tpetra_Import>& importerT = solMgrT->get_importerT();

  // Scatter Vx dot the overlapped distribution
  RCP<Tpetra_MultiVector> overlapped_VxT;
  if (VxT != Teuchos::null) {
    overlapped_VxT =
      rcp(new Tpetra_MultiVector(disc->getOverlapMapT(),
					  VxT->getNumVectors()));
    overlapped_VxT->doImport(*VxT, *importerT, Tpetra::INSERT);
  }

  // Scatter Vx dot the overlapped distribution
  RCP<Tpetra_MultiVector> overlapped_VxdotT;
  if (VxdotT != Teuchos::null) {
    overlapped_VxdotT =
      rcp(new Tpetra_MultiVector(disc->getOverlapMapT(),
				 VxdotT->getNumVectors()));
    overlapped_VxdotT->doImport(*VxdotT, *importerT, Tpetra::INSERT);
  }
  RCP<Tpetra_MultiVector> overlapped_VxdotdotT;
  if (VxdotdotT != Teuchos::null) {
    overlapped_VxdotdotT = rcp(new Tpetra_MultiVector(disc->getOverlapMapT(), VxdotdotT->getNumVectors()));
    overlapped_VxdotdotT->doImport(*VxdotdotT, *importerT, Tpetra::INSERT);
  }

  //RCP<const Epetra_MultiVector > vp = rcp(Vp, false);
  RCP<ParamVec> params = rcp(deriv_p, false);

  // Number of x & xdot tangent directions
  int num_cols_x = 0;
  if (VxT != Teuchos::null)
    num_cols_x = VxT->getNumVectors();
  else if (VxdotT != Teuchos::null)
    num_cols_x = VxdotT->getNumVectors();
  else if (VxdotdotT != Teuchos::null)
    num_cols_x = VxdotdotT->getNumVectors();

  // Number of parameter tangent directions
  int num_cols_p = 0;
  if (params != Teuchos::null) {
    if (VpT != Teuchos::null)
      num_cols_p = VpT->getNumVectors();
    else
      num_cols_p = params->size();
  }

  // Whether x and param tangent components are added or separate
  int param_offset = 0;
  if (!sum_derivs)
    param_offset = num_cols_x;  // offset of parameter derivs in deriv array

  TEUCHOS_TEST_FOR_EXCEPTION(
    sum_derivs &&
    (num_cols_x != 0) &&
    (num_cols_p != 0) &&
    (num_cols_x != num_cols_p),
    std::logic_error,
    "Seed matrices Vx and Vp must have the same number " <<
    " of columns when sum_derivs is true and both are "
    << "non-null!" << std::endl);

  // Initialize
  if (params != Teuchos::null) {
    TanFadType p;
    int num_cols_tot = param_offset + num_cols_p;
    for (unsigned int i=0; i<params->size(); i++) {
      p = TanFadType(num_cols_tot, (*params)[i].baseValue);
      if (VpT != Teuchos::null) {
        Teuchos::ArrayRCP<const ST> VpT_constView;
        for (int k=0; k<num_cols_p; k++) {
          VpT_constView = VpT->getData(k);
          p.fastAccessDx(param_offset+k) = VpT_constView[i];
        }
      }
      else
        p.fastAccessDx(param_offset+i) = 1.0;
      (*params)[i].family->setValue<PHAL::AlbanyTraits::Tangent>(p);
    }
  }

  workset.params = params;
  workset.VxT = overlapped_VxT;
  workset.VxdotT = overlapped_VxdotT;
  workset.VxdotdotT = overlapped_VxdotdotT;
  workset.VpT = VpT;
  workset.num_cols_x = num_cols_x;
  workset.num_cols_p = num_cols_p;
  workset.param_offset = param_offset;
}


#ifdef ALBANY_SG
void Albany::Application::setupTangentWorksetInfo(
  PHAL::Workset& workset,
  double current_time,
  bool sum_derivs,
  const Stokhos::EpetraVectorOrthogPoly* sg_xdot,
  const Stokhos::EpetraVectorOrthogPoly* sg_xdotdot,
  const Stokhos::EpetraVectorOrthogPoly* sg_x,
  const Teuchos::Array<ParamVec>& p,
  ParamVec* deriv_p,
  const Teuchos::Array<int>& sg_p_index,
  const Teuchos::Array< Teuchos::Array<SGType> >& sg_p_vals,
  const Epetra_MultiVector* Vxdot,
  const Epetra_MultiVector* Vxdotdot,
  const Epetra_MultiVector* Vx,
  const Epetra_MultiVector* Vp)
{
  setupBasicWorksetInfo(workset, current_time, sg_xdot, sg_xdotdot, sg_x, p,
                        sg_p_index, sg_p_vals);

  Teuchos::RCP<Epetra_Import>& importer = solMgr->get_importer();

  // Scatter Vx dot the overlapped distribution
  RCP<Epetra_MultiVector> overlapped_Vx;
  if (Vx != NULL) {
    overlapped_Vx =
      rcp(new Epetra_MultiVector(*(disc->getOverlapMap()),
                                          Vx->NumVectors()));
    overlapped_Vx->Import(*Vx, *importer, Insert);
  }

  // Scatter Vx dot the overlapped distribution
  RCP<Epetra_MultiVector> overlapped_Vxdot;
  if (Vxdot != NULL) {
    overlapped_Vxdot = rcp(new Epetra_MultiVector(*(disc->getOverlapMap()), Vxdot->NumVectors()));
    overlapped_Vxdot->Import(*Vxdot, *importer, Insert);
  }
  RCP<Epetra_MultiVector> overlapped_Vxdotdot;
  if (Vxdotdot != NULL) {
    overlapped_Vxdotdot = rcp(new Epetra_MultiVector(*(disc->getOverlapMap()), Vxdotdot->NumVectors()));
    overlapped_Vxdotdot->Import(*Vxdotdot, *importer, Insert);
  }

  RCP<const Epetra_MultiVector > vp = rcp(Vp, false);
  RCP<ParamVec> params = rcp(deriv_p, false);

  // Number of x & xdot tangent directions
  int num_cols_x = 0;
  if (Vx != NULL)
    num_cols_x = Vx->NumVectors();
  else if (Vxdot != NULL)
    num_cols_x = Vxdot->NumVectors();
  else if (Vxdotdot != NULL)
    num_cols_x = Vxdotdot->NumVectors();

  // Number of parameter tangent directions
  int num_cols_p = 0;
  if (params != Teuchos::null) {
    if (Vp != NULL)
      num_cols_p = Vp->NumVectors();
    else
      num_cols_p = params->size();
  }

  // Whether x and param tangent components are added or separate
  int param_offset = 0;
  if (!sum_derivs)
    param_offset = num_cols_x;  // offset of parameter derivs in deriv array

  TEUCHOS_TEST_FOR_EXCEPTION(
    sum_derivs &&
    (num_cols_x != 0) &&
    (num_cols_p != 0) &&
    (num_cols_x != num_cols_p),
    std::logic_error,
    "Seed matrices Vx and Vp must have the same number " <<
    " of columns when sum_derivs is true and both are "
    << "non-null!" << std::endl);

  // Initialize
  if (params != Teuchos::null) {
    SGFadType p;
    int num_cols_tot = param_offset + num_cols_p;
    for (unsigned int i=0; i<params->size(); i++) {
      // Get the base value set above
      SGType base_val =
        (*params)[i].family->getValue<PHAL::AlbanyTraits::SGTangent>().val();
      p = SGFadType(num_cols_tot, base_val);
      if (Vp != NULL)
        for (int k=0; k<num_cols_p; k++)
          p.fastAccessDx(param_offset+k) = (*Vp)[k][i];
      else
        p.fastAccessDx(param_offset+i) = 1.0;
      (*params)[i].family->setValue<PHAL::AlbanyTraits::SGTangent>(p);
    }
  }

  workset.params = params;
  workset.Vx = overlapped_Vx;
  workset.Vxdot = overlapped_Vxdot;
  workset.Vxdotdot = overlapped_Vxdotdot;
  workset.Vp = vp;
  workset.num_cols_x = num_cols_x;
  workset.num_cols_p = num_cols_p;
  workset.param_offset = param_offset;
}

#endif 
#ifdef ALBANY_ENSEMBLE 

void Albany::Application::setupTangentWorksetInfo(
  PHAL::Workset& workset,
  double current_time,
  bool sum_derivs,
  const Stokhos::ProductEpetraVector* mp_xdot,
  const Stokhos::ProductEpetraVector* mp_xdotdot,
  const Stokhos::ProductEpetraVector* mp_x,
  const Teuchos::Array<ParamVec>& p,
  ParamVec* deriv_p,
  const Teuchos::Array<int>& mp_p_index,
  const Teuchos::Array< Teuchos::Array<MPType> >& mp_p_vals,
  const Epetra_MultiVector* Vxdot,
  const Epetra_MultiVector* Vxdotdot,
  const Epetra_MultiVector* Vx,
  const Epetra_MultiVector* Vp)
{
  setupBasicWorksetInfo(workset, current_time, mp_xdot, mp_xdotdot, mp_x, p,
                        mp_p_index, mp_p_vals);

  Teuchos::RCP<Epetra_Import>& importer = solMgr->get_importer();

  // Scatter Vx dot the overlapped distribution
  RCP<Epetra_MultiVector> overlapped_Vx;
  if (Vx != NULL) {
    overlapped_Vx =
      rcp(new Epetra_MultiVector(*(disc->getOverlapMap()),
                                          Vx->NumVectors()));
    overlapped_Vx->Import(*Vx, *importer, Insert);
  }

  // Scatter Vx dot the overlapped distribution
  RCP<Epetra_MultiVector> overlapped_Vxdot;
  if (Vxdot != NULL) {
    overlapped_Vxdot = rcp(new Epetra_MultiVector(*(disc->getOverlapMap()), Vxdot->NumVectors()));
    overlapped_Vxdot->Import(*Vxdot, *importer, Insert);
  }
  RCP<Epetra_MultiVector> overlapped_Vxdotdot;
  if (Vxdotdot != NULL) {
    overlapped_Vxdotdot = rcp(new Epetra_MultiVector(*(disc->getOverlapMap()), Vxdotdot->NumVectors()));
    overlapped_Vxdotdot->Import(*Vxdotdot, *importer, Insert);
  }

  RCP<const Epetra_MultiVector > vp = rcp(Vp, false);
  RCP<ParamVec> params = rcp(deriv_p, false);

  // Number of x & xdot tangent directions
  int num_cols_x = 0;
  if (Vx != NULL)
    num_cols_x = Vx->NumVectors();
  else if (Vxdot != NULL)
    num_cols_x = Vxdot->NumVectors();
  else if (Vxdotdot != NULL)
    num_cols_x = Vxdotdot->NumVectors();

  // Number of parameter tangent directions
  int num_cols_p = 0;
  if (params != Teuchos::null) {
    if (Vp != NULL)
      num_cols_p = Vp->NumVectors();
    else
      num_cols_p = params->size();
  }

  // Whether x and param tangent components are added or separate
  int param_offset = 0;
  if (!sum_derivs)
    param_offset = num_cols_x;  // offset of parameter derivs in deriv array

  TEUCHOS_TEST_FOR_EXCEPTION(
    sum_derivs &&
    (num_cols_x != 0) &&
    (num_cols_p != 0) &&
    (num_cols_x != num_cols_p),
    std::logic_error,
    "Seed matrices Vx and Vp must have the same number " <<
    " of columns when sum_derivs is true and both are "
    << "non-null!" << std::endl);

  // Initialize
  if (params != Teuchos::null) {
    MPFadType p;
    int num_cols_tot = param_offset + num_cols_p;
    for (unsigned int i=0; i<params->size(); i++) {
      // Get the base value set above
      MPType base_val =
        (*params)[i].family->getValue<PHAL::AlbanyTraits::MPTangent>().val();
      p = MPFadType(num_cols_tot, base_val);
      if (Vp != NULL)
        for (int k=0; k<num_cols_p; k++)
          p.fastAccessDx(param_offset+k) = (*Vp)[k][i];
      else
        p.fastAccessDx(param_offset+i) = 1.0;
      (*params)[i].family->setValue<PHAL::AlbanyTraits::MPTangent>(p);
    }
  }

  workset.params = params;
  workset.Vx = overlapped_Vx;
  workset.Vxdot = overlapped_Vxdot;
  workset.Vxdotdot = overlapped_Vxdotdot;
  workset.Vp = vp;
  workset.num_cols_x = num_cols_x;
  workset.num_cols_p = num_cols_p;
  workset.param_offset = param_offset;
}
#endif

#ifdef ALBANY_MOR
#if defined(ALBANY_EPETRA)
Teuchos::RCP<Albany::MORFacade> Albany::Application::getMorFacade()
{
  return morFacade;
}
#endif
#endif

#if defined(ALBANY_LCM)
void
Albany::
Application::
setCoupledAppBlockNodeset(
    std::string const & app_name,
    std::string const & block_name,
    std::string const & nodeset_name)
{
  // Check for valid application name
  auto
  it = app_name_index_map_->find(app_name);

  TEUCHOS_TEST_FOR_EXCEPTION(
      it == app_name_index_map_->end(),
      std::logic_error,
      "Trying to couple to an unknown Application: " <<
      app_name << '\n');

  int const
  app_index = it->second;

  auto
  block_nodeset_names = std::make_pair(block_name, nodeset_name);

  auto
  app_index_block_names = std::make_pair(app_index, block_nodeset_names);

  coupled_app_index_block_nodeset_names_map_.insert(app_index_block_names);
}

#endif // ALBANY_LCM
