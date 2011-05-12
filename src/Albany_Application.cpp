/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/

#include "Albany_Application.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ProblemFactory.hpp"
#include "Albany_DiscretizationFactory.hpp"
#include "Albany_InitialCondition.hpp"
#include "Epetra_LocalMap.h"
#include "Stokhos_OrthogPolyBasis.hpp"
#include "Teuchos_TimeMonitor.hpp"

#include<string>
#include "PHAL_Workset.hpp"
#include "Albany_DataTypes.hpp"

#include "Albany_DummyParameterAccessor.hpp"
#ifdef ALBANY_CUTR
  #include "CUTR_CubitMeshMover.hpp"
  #include "STKMeshData.hpp"
#endif

#include "Teko_InverseFactoryOperator.hpp"
#include "Teko_StridedEpetraOperator.hpp"

#ifdef ALBANY_IOSS
  #include "Albany_STKDiscretization.hpp"
#endif

Albany::Application::Application(
		   const Teuchos::RCP<const Epetra_Comm>& comm,
		   const Teuchos::RCP<Teuchos::ParameterList>& params,
		   const Teuchos::RCP<const Epetra_Vector>& initial_guess) :
  out(Teuchos::VerboseObjectBase::getDefaultOStream()),
  physicsBasedPreconditioner(false),
  shapeParamsHaveBeenReset(false),
  setupCalledResidual(false), setupCalledJacobian(false), setupCalledTangent(false),
  setupCalledSGResidual(false), setupCalledSGJacobian(false),
  setupCalledMPResidual(false), setupCalledMPJacobian(false),
  morphFromInit(true), perturbBetaForDirichlets(0.0)
  //, stateMgr(Albany::StateManager())
{
  timers.push_back(Teuchos::TimeMonitor::getNewTimer("> Albany Fill: Residual"));
  timers.push_back(Teuchos::TimeMonitor::getNewTimer("> Albany Fill: Jacobian"));
  timers.push_back(Teuchos::TimeMonitor::getNewTimer("> Albany Fill: Precond"));
  timers.push_back(Teuchos::TimeMonitor::getNewTimer("> Albany Fill: Tangent"));
  timers.push_back(Teuchos::TimeMonitor::getNewTimer("> Albany Fill: SGResidual"));
  timers.push_back(Teuchos::TimeMonitor::getNewTimer("> Albany Fill: SGJacobian"));
  timers.push_back(Teuchos::TimeMonitor::getNewTimer("> Albany Fill: MPResidual"));
  timers.push_back(Teuchos::TimeMonitor::getNewTimer("> Albany Fill: MPJacobian"));
  timers.push_back(Teuchos::TimeMonitor::getNewTimer("Albany-Cubit MeshMover"));

  // Create parameter library
  paramLib = Teuchos::rcp(new ParamLib);

  // Create problem object
  Teuchos::RCP<Teuchos::ParameterList> problemParams = 
    Teuchos::sublist(params, "Problem", true);
  Albany::ProblemFactory problemFactory(problemParams, paramLib);
  Teuchos::RCP<Albany::AbstractProblem> problem = problemFactory.create();

  // Validate Problem parameters against list for this specific problem
  problemParams->validateParameters(*(problem->getValidProblemParameters()),0);

  // Get number of equations
  neq = problem->numEquations();

  // Register shape parameters for manipulation by continuation/optimization
  if (problemParams->get("Enable Cubit Shape Parameters",false)) {
#ifdef ALBANY_CUTR
    Teuchos::TimeMonitor Timer(*timers[8]); //start timer
    meshMover = Teuchos::rcp(new CUTR::CubitMeshMover
          (problemParams->get<std::string>("Cubit Base Filename")));

    meshMover->getShapeParams(shapeParamNames, shapeParams);
    *out << "SSS : Registering " << shapeParams.size() << " Shape Parameters" << endl;

    registerShapeParameters();

#else
  TEST_FOR_EXCEPTION(problemParams->get("Enable Cubit Shape Parameters",false), std::logic_error,
                     "Cubit requested but not Compiled in!");
#endif
  }

  physicsBasedPreconditioner = problemParams->get("Use Physics-Based Preconditioner",false);
  if (physicsBasedPreconditioner) 
    tekoParams = Teuchos::sublist(problemParams, "Teko", true);

  // Create discretization object
  Teuchos::RCP<Teuchos::ParameterList> discParams = 
    Teuchos::rcpFromRef(params->sublist("Discretization"));
  Albany::DiscretizationFactory discFactory(discParams);
#ifdef ALBANY_CUTR
  discFactory.setMeshMover(meshMover);
#endif
  worksetSize = problemParams->get("Workset Size",100);
  if (worksetSize < 1) { cout << "FIX ME " << endl; worksetSize = 1000;}

  disc = discFactory.create(neq, problem->numStates(), worksetSize, comm);

  // Load connectivity map and coordinates 
  wsElNodeID = disc->getWsElNodeID();
  coords = disc->getCoords();
  coordinates = disc->getCoordinates();
  wsEBNames = disc->getWsEBNames();
  int numDim = disc->getNumDim();
  numWorksets = wsElNodeID.size();


  // Create Epetra objects
  importer = Teuchos::rcp(new Epetra_Import(*(disc->getOverlapMap()), 
                                            *(disc->getMap())));
  exporter = Teuchos::rcp(new Epetra_Export(*(disc->getOverlapMap()), 
                                            *(disc->getMap())));
  overlapped_x = Teuchos::rcp(new Epetra_Vector(*(disc->getOverlapMap())));
  overlapped_xdot = 
      Teuchos::rcp(new Epetra_Vector(*(disc->getOverlapMap())));
  overlapped_f = Teuchos::rcp(new Epetra_Vector(*(disc->getOverlapMap())));
  overlapped_jac = 
    Teuchos::rcp(new Epetra_CrsMatrix(Copy, 
                                      *(disc->getOverlapJacobianGraph())));

  // Initialize solution vector and time deriv
  initial_x = disc->getSolutionField();
  initial_x_dot = Teuchos::rcp(new Epetra_Vector(*(disc->getMap())));

  if (initial_guess != Teuchos::null) *initial_x = *initial_guess;
  else {
    overlapped_x->Import(*initial_x, *importer, Insert);
    Albany::InitialConditions(overlapped_x, wsElNodeID, coordinates, neq, numDim,
                              problemParams->sublist("Initial Condition"));
    Albany::InitialConditions(overlapped_xdot,  wsElNodeID, coordinates, neq, numDim,
                              problemParams->sublist("Initial Condition Dot"));
    initial_x->Export(*overlapped_x, *exporter, Insert);
    initial_x_dot->Export(*overlapped_xdot, *exporter, Insert);
  }

  problem->buildProblem(worksetSize, stateMgr, *disc, responses);

  stateMgr.allocateStateVariables(numWorksets);
  stateMgr.initializeStateVariables(numWorksets);

  // Create response map
  unsigned int total_num_responses = 0;
  for (unsigned int i=0; i<responses.size(); i++)
    total_num_responses += responses[i]->numResponses();
  if (total_num_responses > 0)
    response_map = Teuchos::rcp(new Epetra_LocalMap(total_num_responses, 0,
                                                    *comm));
  // Set up memory for workset

  fm = problem->getFieldManager();
  TEST_FOR_EXCEPTION(fm==Teuchos::null, std::logic_error,
                     "getFieldManager not implemented!!!");
  dfm = problem->getDirichletFieldManager();

  phxGraphVisDetail = problemParams->get("Phalanx Graph Visualization Detail", 0);

  *out << "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n"
       << " Sacado ParameterLibrary has been initialized:\n " 
       << *paramLib 
       << "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n"
       << endl;

  ignore_residual_in_jacobian = 
    problemParams->get("Ignore Residual In Jacobian", false);

  perturbBetaForDirichlets = problemParams->get("Perturb Dirichlet",0.0);
}

Albany::Application::~Application()
{
}

Teuchos::RCP<Albany::AbstractDiscretization>
Albany::Application::getDiscretization() const
{
  return disc;
}

Teuchos::RCP<const Epetra_Map>
Albany::Application::getMap() const
{
  return disc->getMap();
}

Teuchos::RCP<const Epetra_CrsGraph>
Albany::Application::getJacobianGraph() const
{
  return disc->getJacobianGraph();
}

Teuchos::RCP<Epetra_Operator>
Albany::Application::getPreconditioner()
{
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

   return Teuchos::rcp(new Teko::Epetra::InverseFactoryOperator(inverseFac));
}

Teuchos::RCP<const Epetra_Vector>
Albany::Application::getInitialSolution() const
{
  return initial_x;
}

Teuchos::RCP<const Epetra_Vector>
Albany::Application::getInitialSolutionDot() const
{
  return initial_x_dot;
}

Teuchos::RCP<ParamLib> 
Albany::Application::getParamLib()
{
  return paramLib;
}

Teuchos::RCP<const Epetra_Map>
Albany::Application::getResponseMap() const
{
  return response_map;
}

bool
Albany::Application::suppliesPreconditioner() const 
{
  return physicsBasedPreconditioner;
}

Teuchos::RCP<Stokhos::OrthogPolyExpansion<int,double> >
Albany::Application::getStochasticExpansion()
{
  return sg_expansion;
}

void
Albany::Application::init_sg(
  const Teuchos::RCP<Stokhos::OrthogPolyExpansion<int,double> >& expansion,
  const Teuchos::RCP<const EpetraExt::MultiComm>& multiComm)
{

  // Setup stohastic Galerkin
  sg_expansion = expansion;
  Teuchos::RCP<const Stokhos::OrthogPolyBasis<int,double> > sg_basis =
    sg_expansion->getBasis();
  
  if (sg_overlapped_x == Teuchos::null) {
    sg_overlap_map =
      Teuchos::rcp(new Epetra_LocalMap(sg_basis->size(), 0, 
				       multiComm->TimeDomainComm()));
    sg_overlapped_x = 
      Teuchos::rcp(new Stokhos::VectorOrthogPoly<Epetra_Vector>(
		     sg_basis, sg_overlap_map, *overlapped_x));
    sg_overlapped_xdot = 
	Teuchos::rcp(new Stokhos::VectorOrthogPoly<Epetra_Vector>(
		       sg_basis, sg_overlap_map, *overlapped_xdot));
    sg_overlapped_f = 
      Teuchos::rcp(new Stokhos::VectorOrthogPoly<Epetra_Vector>(
		     sg_basis, sg_overlap_map, *overlapped_f));
    // Delay creation of sg_overlapped_jac until needed
  }
}

void
Albany::Application::computeGlobalResidual(
			   const double current_time,
			   const Epetra_Vector* xdot,
			   const Epetra_Vector& x,
			   const Teuchos::Array< Teuchos::RCP<ParamVec> >& p,
			   Epetra_Vector& f)
{
  if (!setupCalledResidual) {
    setupCalledResidual=true;
    fm->postRegistrationSetupForType<PHAL::AlbanyTraits::Residual>(*disc);
    if (dfm!=Teuchos::null)
      dfm->postRegistrationSetupForType<PHAL::AlbanyTraits::Residual>(*disc);
    writeGraphVisFile();
  }

  Teuchos::TimeMonitor Timer(*timers[0]); //start timer

  // Scatter x to the overlapped distrbution
  overlapped_x->Import(x, *importer, Insert);

  // Scatter xdot to the overlapped distribution
  if (xdot != NULL)
    overlapped_xdot->Import(*xdot, *importer, Insert);

  // Set parameters
  for (Teuchos::Array< Teuchos::RCP<ParamVec> >::size_type i=0; i<p.size(); i++) {
    if (p[i] != Teuchos::null)
      for (unsigned int j=0; j<p[i]->size(); j++)
	(*(p[i]))[j].family->setRealValueForAllTypes((*(p[i]))[j].baseValue);
  }

  // Mesh motion needs to occur here on the global mesh befor
  // it is potentially carved into worksets.
#ifdef ALBANY_CUTR
  static int first=true;
  if (shapeParamsHaveBeenReset) {
    Teuchos::TimeMonitor cubitTimer(*timers[8]); //start timer

*out << " Calling moveMesh with params: " << std::setprecision(8);
 for (unsigned int i=0; i<shapeParams.size(); i++) *out << shapeParams[i] << "  ";
*out << endl;
    meshMover->moveMesh(shapeParams, morphFromInit);
    coordinates = disc->getCoordinates();
    shapeParamsHaveBeenReset = false;
  }
#endif

  // Zero out overlapped residual
  overlapped_f->PutScalar(0.0);
  f.PutScalar(0.0);

  // Set data in Workset struct, and perform fill via field manager
  { 
    PHAL::Workset workset(coordinates);

    workset.x        = overlapped_x;
    workset.xdot     = overlapped_xdot;
    workset.f        = overlapped_f;
    workset.current_time = current_time;

    if (xdot != NULL) workset.transientTerms = true;

    workset.worksetSize = worksetSize;
    for (int ws=0; ws < numWorksets; ws++) {
      workset.numCells = wsElNodeID[ws].size();
      workset.wsElNodeID = wsElNodeID[ws];
      workset.wsCoords = coords[ws];
      workset.EBName = wsEBNames[ws];

      workset.oldState = stateMgr.getOldStateVariables(ws);
      workset.newState = stateMgr.getNewStateVariables(ws);

      // FillType template argument used to specialize Sacado
      fm->evaluateFields<PHAL::AlbanyTraits::Residual>(workset);
    }
  }

  f.Export(*overlapped_f, *exporter, Add);

#ifdef ALBANY_IOSS
  Albany::STKDiscretization* stkDisc =
    dynamic_cast<Albany::STKDiscretization*>(disc.get());
  stkDisc->setResidualField(f);
#endif

  // Apply Dirichlet conditions using dfm (Dirchelt Field Manager)
  if (dfm!=Teuchos::null) { 
    PHAL::Workset workset(coordinates);

    workset.f = Teuchos::rcpFromRef(f);
    workset.nodeSets = Teuchos::rcpFromRef(disc->getNodeSets());
    workset.nodeSetCoords = Teuchos::rcpFromRef (disc->getNodeSetCoords());
    workset.x = Teuchos::rcpFromRef(x);;
    if (xdot != NULL) workset.transientTerms = true;

    // FillType template argument used to specialize Sacado
    dfm->evaluateFields<PHAL::AlbanyTraits::Residual>(workset);
  } 
  //cout << f << endl;
}

void
Albany::Application::computeGlobalJacobian(
			     const double alpha, 
			     const double beta,
			     const double current_time,
			     const Epetra_Vector* xdot,
			     const Epetra_Vector& x,
			     const Teuchos::Array< Teuchos::RCP<ParamVec> >& p,
			     Epetra_Vector* f,
			     Epetra_CrsMatrix& jac)
{
  if (!setupCalledJacobian) {
    setupCalledJacobian=true;
    fm->postRegistrationSetupForType<PHAL::AlbanyTraits::Jacobian>(*disc);
    if (dfm!=Teuchos::null)
      dfm->postRegistrationSetupForType<PHAL::AlbanyTraits::Jacobian>(*disc);
    writeGraphVisFile();
  }

  Teuchos::TimeMonitor Timer(*timers[1]); //start timer

  // Scatter x to the overlapped distrbution
  overlapped_x->Import(x, *importer, Insert);

  // Scatter xdot to the overlapped distribution
  if (xdot != NULL)
    overlapped_xdot->Import(*xdot, *importer, Insert);

  // Set parameters
  for (Teuchos::Array< Teuchos::RCP<ParamVec> >::size_type i=0; i<p.size(); i++) {
    if (p[i] != Teuchos::null)
      for (unsigned int j=0; j<p[i]->size(); j++)
	(*(p[i]))[j].family->setRealValueForAllTypes((*(p[i]))[j].baseValue);
  }
#ifdef ALBANY_CUTR
  if (shapeParamsHaveBeenReset) {
    Teuchos::TimeMonitor Timer(*timers[8]); //start timer

*out << " Calling moveMesh with params: " << std::setprecision(8);
 for (unsigned int i=0; i<shapeParams.size(); i++) *out << shapeParams[i] << "  ";
*out << endl;
    meshMover->moveMesh(shapeParams, morphFromInit);
    coordinates = disc->getCoordinates();
    shapeParamsHaveBeenReset = false;
  }
#endif

  // Zero out overlapped residual
  Teuchos::RCP<Epetra_Vector> overlapped_ff;
  if (f != NULL) {
    overlapped_ff = overlapped_f;
    overlapped_ff->PutScalar(0.0);
    f->PutScalar(0.0);
  }

  // Zero out Jacobian
  overlapped_jac->PutScalar(0.0);
  jac.PutScalar(0.0);

  // Set data in Workset struct, and perform fill via field manager
  {
    PHAL::Workset workset(coordinates);

    workset.x        = overlapped_x;
    workset.xdot     = overlapped_xdot;
    workset.f        = overlapped_f;

    workset.Jac          = overlapped_jac;
    workset.j_coeff      = beta;
    workset.m_coeff      = alpha;
    workset.current_time = current_time;
    workset.ignore_residual = ignore_residual_in_jacobian;
    if (xdot != NULL) workset.transientTerms = true;

    workset.worksetSize = worksetSize;
    for (int ws=0; ws < numWorksets; ws++) {
      workset.numCells = wsElNodeID[ws].size();
      workset.wsElNodeID = wsElNodeID[ws];
      workset.wsCoords = coords[ws];
      workset.EBName = wsEBNames[ws];

      workset.oldState = stateMgr.getOldStateVariables(ws);
      workset.newState = stateMgr.getNewStateVariables(ws);

      // FillType template argument used to specialize Sacado
      fm->evaluateFields<PHAL::AlbanyTraits::Jacobian>(workset);
    }
  } 

  // Assemble global residual
  if (f != NULL)
    f->Export(*overlapped_f, *exporter, Add);

  // Assemble global Jacobian
  jac.Export(*overlapped_jac, *exporter, Add);

  // Apply Dirichlet conditions using dfm (Dirchelt Field Manager)
  if (dfm!=Teuchos::null) {
    PHAL::Workset workset(coordinates);

    workset.f = Teuchos::rcp(f,false);
    workset.Jac = Teuchos::rcpFromRef(jac);
    workset.m_coeff = alpha;
    workset.j_coeff = beta;

    if (beta==0.0 && perturbBetaForDirichlets>0.0) workset.j_coeff = perturbBetaForDirichlets;

    workset.x = Teuchos::rcpFromRef(x);;
    if (xdot != NULL) workset.transientTerms = true;

    workset.nodeSets = Teuchos::rcpFromRef (disc->getNodeSets());
    workset.nodeSetCoords = Teuchos::rcpFromRef (disc->getNodeSetCoords());

    // FillType template argument used to specialize Sacado
    dfm->evaluateFields<PHAL::AlbanyTraits::Jacobian>(workset);
  }

  jac.FillComplete(true);
  //cout << "f " << *f << endl;;
  //cout << "J " << jac << endl;;
}

void
Albany::Application::computeGlobalPreconditioner(
			     const Teuchos::RCP<Epetra_CrsMatrix>& jac,
			     const Teuchos::RCP<Epetra_Operator>& prec)
{
  Teuchos::TimeMonitor Timer(*timers[2]); //start timer

  *out << "Computing WPrec by Teko" << endl;

  Teuchos::RCP<Teko::Epetra::InverseFactoryOperator> blockPrec
    = Teuchos::rcp_dynamic_cast<Teko::Epetra::InverseFactoryOperator>(prec);

  blockPrec->initInverse();

  wrappedJac = buildWrappedOperator(jac, wrappedJac);
  blockPrec->rebuildInverseOperator(wrappedJac);
}

void
Albany::Application::computeGlobalTangent(
			   const double alpha, 
			   const double beta,
			   const double current_time,
			   bool sum_derivs,
			   const Epetra_Vector* xdot,
			   const Epetra_Vector& x,
			   const Teuchos::Array< Teuchos::RCP<ParamVec> >& par,
			   ParamVec* deriv_par,
			   const Epetra_MultiVector* Vx,
			   const Epetra_MultiVector* Vxdot,
			   const Epetra_MultiVector* Vp,
			   Epetra_Vector* f,
			   Epetra_MultiVector* JV,
			   Epetra_MultiVector* fp)
{
  if (!setupCalledTangent) {
    setupCalledTangent=true;
    fm->postRegistrationSetupForType<PHAL::AlbanyTraits::Tangent>(*disc);
    if (dfm!=Teuchos::null)
      dfm->postRegistrationSetupForType<PHAL::AlbanyTraits::Tangent>(*disc);
    writeGraphVisFile();
  }

  Teuchos::TimeMonitor Timer(*timers[3]); //start timer

  // Scatter x to the overlapped distrbution
  overlapped_x->Import(x, *importer, Insert);

  // Scatter xdot to the overlapped distribution
  if (xdot != NULL)
    overlapped_xdot->Import(*xdot, *importer, Insert);

  // Scatter Vx dot the overlapped distribution
  Teuchos::RCP<Epetra_MultiVector> overlapped_Vx;
  if (Vx != NULL) {
    overlapped_Vx = 
      Teuchos::rcp(new Epetra_MultiVector(*(disc->getOverlapMap()), 
					  Vx->NumVectors()));
    overlapped_Vx->Import(*Vx, *importer, Insert);
  }

  // Scatter Vx dot the overlapped distribution
  Teuchos::RCP<Epetra_MultiVector> overlapped_Vxdot;
  if (Vxdot != NULL) {
    overlapped_Vxdot = 
      Teuchos::rcp(new Epetra_MultiVector(*(disc->getOverlapMap()), 
					  Vxdot->NumVectors()));
    overlapped_Vxdot->Import(*Vxdot, *importer, Insert);
  }

  // Set parameters
  for (Teuchos::Array< Teuchos::RCP<ParamVec> >::size_type i=0; i<par.size(); i++) {
    if (par[i] != Teuchos::null)
      for (unsigned int j=0; j<par[i]->size(); j++)
	(*(par[i]))[j].family->setRealValueForAllTypes((*(par[i]))[j].baseValue);
  }

  Teuchos::RCP<const Epetra_MultiVector > vp = Teuchos::rcp(Vp, false);
  Teuchos::RCP<ParamVec> params = Teuchos::rcp(deriv_par, false);

  // Zero out overlapped residual
  Teuchos::RCP<Epetra_Vector> overlapped_ff;
  if (f != NULL) {
    overlapped_ff = overlapped_f;
    overlapped_ff->PutScalar(0.0);
    f->PutScalar(0.0);
  }

  Teuchos::RCP<Epetra_MultiVector> overlapped_JV;
  if (JV != NULL) {
    overlapped_JV = 
      Teuchos::rcp(new Epetra_MultiVector(*(disc->getOverlapMap()), 
					  JV->NumVectors()));
    overlapped_JV->PutScalar(0.0);
    JV->PutScalar(0.0);
  }
  
  Teuchos::RCP<Epetra_MultiVector> overlapped_fp;
  if (fp != NULL) {
    overlapped_fp = 
      Teuchos::rcp(new Epetra_MultiVector(*(disc->getOverlapMap()), 
					  fp->NumVectors()));
    overlapped_fp->PutScalar(0.0);
    fp->PutScalar(0.0);
  }

  // Number of x & xdot tangent directions
  int num_cols_x = 0;
  if (Vx != NULL)
    num_cols_x = Vx->NumVectors();
  else if (Vxdot != NULL)
    num_cols_x = Vxdot->NumVectors();

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

  TEST_FOR_EXCEPTION(sum_derivs && 
		     (num_cols_x != 0) && 
		     (num_cols_p != 0) && 
                     (num_cols_x != num_cols_p),
                     std::logic_error,
                     "Seed matrices Vx and Vp must have the same number " << 
                     " of columns when sum_derivs is true and both are "
                     << "non-null!" << std::endl);

  // Initialize 
  if (params != Teuchos::null) {
    FadType p;
    int num_cols_tot = param_offset + num_cols_p;
    for (unsigned int i=0; i<params->size(); i++) {
      p = FadType(num_cols_tot, (*params)[i].baseValue);
      if (Vp != NULL) 
        for (int k=0; k<num_cols_p; k++)
          p.fastAccessDx(param_offset+k) = (*Vp)[k][i];
      else
        p.fastAccessDx(param_offset+i) = 1.0;
      (*params)[i].family->setValue<PHAL::AlbanyTraits::Tangent>(p);
    }
  }

  // Begin shape optimization logic
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> > coord_derivs;
  // ws, sp, cell, node, dim
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> > > > > ws_coord_derivs;
  ws_coord_derivs.resize(coords.size());
  std::vector<int> coord_deriv_indices;
#ifdef ALBANY_CUTR
  if (shapeParamsHaveBeenReset) {
    Teuchos::TimeMonitor Timer(*timers[8]); //start timer

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

    TEST_FOR_EXCEPTION( Vp != NULL, std::logic_error,
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
     << " which should equal " << (*params)[coord_deriv_indices[i]].family->getName() << endl;

     pert = (fabs(shapeParams[shape_param_indices[i]]) + 1.0e-2) * eps;

       coord_derivs[i].resize(coordinates.size());
       shapeParams[shape_param_indices[i]] += pert;
*out << " Calling moveMesh with params: " << std::setprecision(8);
for (unsigned int ii=0; ii<shapeParams.size(); ii++) *out << shapeParams[ii] << "  ";
*out << endl;
       meshMover->moveMesh(shapeParams, morphFromInit);
       coordinates = disc->getCoordinates();
       for (int j=0; j<coordinates.size(); j++)  coord_derivs[i][j] = coordinates[j];

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
*out << endl;
     meshMover->moveMesh(shapeParams, morphFromInit);
     coordinates = disc->getCoordinates();
     for (int i=0; i<num_sp; i++) {
       for (int ws=0; ws<coords.size(); ws++)  //worset
         for (int e=0; e<coords[ws].size(); e++)  //cell
           for (int j=0; j<coords[ws][i].size(); j++)  //node
             for (int d=0; d<disc->getNumDim; d++)  //node
                ws_coord_derivs[ws][i][e][j][d] = (ws_coord_derivs[ws][i][e][j][d] - coords[ws][e][j][d]) / pert;
       for (int j=0; j<coordinates.size(); j++) {
          coord_derivs[i][j] = (coord_derivs[i][j] - coordinates[j]) / pert;
       }
     }
     shapeParamsHaveBeenReset = false;
  }
  // End shape optimization logic
#endif

  // Set data in Workset struct, and perform fill via field manager
  {
    PHAL::Workset workset(coordinates);

    workset.x = overlapped_x;
    workset.xdot = overlapped_xdot;
    workset.params = params;
    workset.Vx = overlapped_Vx;
    workset.Vxdot = overlapped_Vxdot;
    workset.Vp = vp;

    workset.f            = overlapped_f;
    workset.JV           = overlapped_JV;
    workset.fp           = overlapped_fp;
    workset.j_coeff      = beta;
    workset.m_coeff      = alpha;
    workset.current_time = current_time;
    if (xdot != NULL) workset.transientTerms = true;

    workset.num_cols_x = num_cols_x;
    workset.num_cols_p = num_cols_p;
    workset.param_offset = param_offset;

    workset.coord_derivs = coord_derivs;
    workset.coord_deriv_indices = &coord_deriv_indices;

    workset.worksetSize = worksetSize;
    for (int ws=0; ws < numWorksets; ws++) {
      workset.numCells = wsElNodeID[ws].size();
      workset.wsElNodeID = wsElNodeID[ws];
      workset.wsCoords = coords[ws];
      workset.EBName = wsEBNames[ws];
      workset.ws_coord_derivs = ws_coord_derivs[ws];

      workset.oldState = stateMgr.getOldStateVariables(ws);
      workset.newState = stateMgr.getNewStateVariables(ws);

      // FillType template argument used to specialize Sacado
      fm->evaluateFields<PHAL::AlbanyTraits::Tangent>(workset);
    }
  }

  vp = Teuchos::null;
  params = Teuchos::null;

  // Assemble global residual
  if (f != NULL)
    f->Export(*overlapped_f, *exporter, Add);

  // Assemble derivatives
  if (JV != NULL)
    JV->Export(*overlapped_JV, *exporter, Add);
  if (fp != NULL)
    fp->Export(*overlapped_fp, *exporter, Add);

  // Apply Dirichlet conditions using dfm (Dirchelt Field Manager)
  if (dfm!=Teuchos::null) {
    PHAL::Workset workset(coordinates);

    workset.num_cols_x = num_cols_x;
    workset.num_cols_p = num_cols_p;
    workset.param_offset = param_offset;

    workset.f = Teuchos::rcp(f,false);
    workset.fp = Teuchos::rcp(fp,false);
    workset.JV = Teuchos::rcp(JV,false);
    workset.j_coeff = beta;
    workset.x = Teuchos::rcpFromRef(x);
    workset.Vx = Teuchos::rcp(Vx,false);
    if (xdot != NULL) workset.transientTerms = true;

    workset.nodeSets = Teuchos::rcpFromRef(disc->getNodeSets());
    workset.nodeSetCoords = Teuchos::rcpFromRef (disc->getNodeSetCoords());

    // FillType template argument used to specialize Sacado
    dfm->evaluateFields<PHAL::AlbanyTraits::Tangent>(workset);
  }

//*out << "fp " << *fp << endl;

}

void
Albany::Application::
evaluateResponses(const Epetra_Vector* xdot,
                  const Epetra_Vector& x,
                  const Teuchos::Array< Teuchos::RCP<ParamVec> >& p,
                  Epetra_Vector& g)
{
  const Epetra_Comm& comm = x.Map().Comm();
  unsigned int offset = 0;
  for (unsigned int i=0; i<responses.size(); i++) {

    // Create Epetra_Map for response function
    unsigned int num_responses = responses[i]->numResponses();
    Epetra_LocalMap local_response_map(num_responses, 0, comm);

    // Create Epetra_Vector for response function
    Epetra_Vector local_g(local_response_map);

    // Evaluate response function
    responses[i]->evaluateResponses(xdot, x, p, local_g);

    // Copy result into combined result
    for (unsigned int j=0; j<num_responses; j++)
      g[offset+j] = local_g[j];

    // Increment offset in combined result
    offset += num_responses;
  }
}

void
Albany::Application::
evaluateResponseTangents(
	     const Epetra_Vector* xdot,
	     const Epetra_Vector& x,
	     const Teuchos::Array< Teuchos::RCP<ParamVec> >& p,
	     const Teuchos::Array< Teuchos::RCP<ParamVec> >& deriv_p,
	     const Teuchos::Array< Teuchos::RCP<Epetra_MultiVector> >& dxdot_dp,
	     const Teuchos::Array< Teuchos::RCP<Epetra_MultiVector> >& dx_dp,
	     Epetra_Vector* g,
	     const Teuchos::Array< Teuchos::RCP<Epetra_MultiVector> >& gt)
{
  const Epetra_Comm& comm = x.Map().Comm();
  unsigned int offset = 0;
  Teuchos::Array< Teuchos::RCP<Epetra_MultiVector> > local_gt(gt.size());
  for (unsigned int i=0; i<responses.size(); i++) {

    // Create Epetra_Map for response function
    unsigned int num_responses = responses[i]->numResponses();
    Epetra_LocalMap local_response_map(num_responses, 0, comm);

    // Create Epetra_Vectors for response function
    Teuchos::RCP<Epetra_Vector> local_g;
    if (g != NULL)
      local_g = Teuchos::rcp(new Epetra_Vector(local_response_map));
    for (Teuchos::Array< Teuchos::RCP<Epetra_MultiVector> >::size_type j=0; j<gt.size(); j++)
      if (gt[j] != Teuchos::null)
	local_gt[j] = Teuchos::rcp(new Epetra_MultiVector(local_response_map, 
							  gt[j]->NumVectors()));

    // Evaluate response function
    responses[i]->evaluateTangents(xdot, x, p, deriv_p, dxdot_dp, dx_dp, 
				   local_g.get(), local_gt);

    // Copy results into combined result
    for (unsigned int j=0; j<num_responses; j++) {
      if (g != NULL)
        (*g)[offset+j] = (*local_g)[j];
      for (Teuchos::Array< Teuchos::RCP<Epetra_MultiVector> >::size_type l=0; l<gt.size(); l++)
	if (gt[l] != Teuchos::null)
	  for (int k=0; k<gt[l]->NumVectors(); k++)
	    (*gt[l])[k][offset+j] = (*local_gt[l])[k][j];
    }

    // Increment offset in combined result
    offset += num_responses;
  }
}

void
Albany::Application::
evaluateResponseGradients(
		const Epetra_Vector* xdot,
		const Epetra_Vector& x,
		const Teuchos::Array< Teuchos::RCP<ParamVec> >& p,
		const Teuchos::Array< Teuchos::RCP<ParamVec> >& deriv_p,
		Epetra_Vector* g,
		Epetra_MultiVector* dg_dx,
		Epetra_MultiVector* dg_dxdot,
		const Teuchos::Array< Teuchos::RCP<Epetra_MultiVector> >& dg_dp)
{
  const Epetra_Comm& comm = x.Map().Comm();
  unsigned int offset = 0;
  Teuchos::Array< Teuchos::RCP<Epetra_MultiVector> > local_dgdp(dg_dp.size());
  for (unsigned int i=0; i<responses.size(); i++) {

    // Create Epetra_Map for response function
    unsigned int num_responses = responses[i]->numResponses();
    Epetra_LocalMap local_response_map(num_responses, 0, comm);

    // Create Epetra_Vectors for response function
    Teuchos::RCP<Epetra_Vector> local_g;
    if (g != NULL)
      local_g = Teuchos::rcp(new Epetra_Vector(local_response_map));
    Teuchos::RCP<Epetra_MultiVector> local_dgdx;
    if (dg_dx != NULL)
      local_dgdx = Teuchos::rcp(new Epetra_MultiVector(dg_dx->Map(), 
                                                       num_responses));
    Teuchos::RCP<Epetra_MultiVector> local_dgdxdot;
    if (dg_dxdot != NULL)
      local_dgdxdot = Teuchos::rcp(new Epetra_MultiVector(dg_dxdot->Map(), 
                                                          num_responses));
    for (Teuchos::Array< Teuchos::RCP<Epetra_MultiVector> >::size_type j=0; j<dg_dp.size(); j++)
      if (dg_dp[j] != Teuchos::null)
	local_dgdp[j] = Teuchos::rcp(new Epetra_MultiVector(local_response_map, 
							    dg_dp[j]->NumVectors()));

    // Evaluate response function
    responses[i]->evaluateGradients(xdot, x, p, deriv_p, local_g.get(), 
                                    local_dgdx.get(), local_dgdxdot.get(), 
                                    local_dgdp);

    // Copy results into combined result
    for (unsigned int j=0; j<num_responses; j++) {
      if (g != NULL)
        (*g)[offset+j] = (*local_g)[j];
      if (dg_dx != NULL)
        (*dg_dx)(offset+j)->Update(1.0, *((*local_dgdx)(j)), 0.0);
      if (dg_dxdot != NULL)
        (*dg_dxdot)(offset+j)->Update(1.0, *((*local_dgdxdot)(j)), 0.0);
      for (Teuchos::Array< Teuchos::RCP<Epetra_MultiVector> >::size_type l=0; l<dg_dp.size(); l++)
	if (dg_dp[l] != Teuchos::null)
	  for (int k=0; k<dg_dp[l]->NumVectors(); k++)
	    (*dg_dp[l])[k][offset+j] = (*local_dgdp[l])[k][j];
    }

    // Increment offset in combined result
    offset += num_responses;
  }
}

void
Albany::Application::computeGlobalSGResidual(
			const double current_time,
			const Stokhos::VectorOrthogPoly<Epetra_Vector>* sg_xdot,
			const Stokhos::VectorOrthogPoly<Epetra_Vector>& sg_x,
			const ParamVec* p,
			const ParamVec* sg_p,
			const Teuchos::Array<SGType>* sg_p_vals,
			Stokhos::VectorOrthogPoly<Epetra_Vector>& sg_f)
{
  if (!setupCalledSGResidual) {
    setupCalledSGResidual=true;
    fm->postRegistrationSetupForType<PHAL::AlbanyTraits::SGResidual>(*disc);
    if (dfm!=Teuchos::null)
      dfm->postRegistrationSetupForType<PHAL::AlbanyTraits::SGResidual>(*disc);
    writeGraphVisFile();
  }

  Teuchos::TimeMonitor Timer(*timers[4]); //start timer
  //TEUCHOS_FUNC_TIME_MONITOR("Albany::Application::computeGlobalSGResidual");

  for (int i=0; i<sg_x.size(); i++) {

    // Scatter x to the overlapped distrbution
    (*sg_overlapped_x)[i].Import(sg_x[i], *importer, Insert);

    // Scatter xdot to the overlapped distribution
    if (sg_xdot != NULL)
      (*sg_overlapped_xdot)[i].Import((*sg_xdot)[i], *importer, Insert);

    // Zero out overlapped residual
    (*sg_overlapped_f)[i].PutScalar(0.0);
    sg_f[i].PutScalar(0.0);

  }

  // Set real parameters
  if (p != NULL) {
    for (unsigned int i=0; i<p->size(); ++i) {
      (*p)[i].family->setRealValueForAllTypes((*p)[i].baseValue);
    }
  }
#ifdef ALBANY_CUTR
  if (shapeParamsHaveBeenReset) {
    Teuchos::TimeMonitor Timer(*timers[8]); //start timer
*out << " Calling moveMesh with params: " << std::setprecision(8);
for (unsigned int i=0; i<shapeParams.size(); i++) *out << shapeParams[i] << "  ";
*out << endl;
    meshMover->moveMesh(shapeParams, morphFromInit);
    coordinates = disc->getCoordinates();
    shapeParamsHaveBeenReset = false;
  }
#endif

  // Set SG parameters
  if (sg_p != NULL && sg_p_vals != NULL) {
    for (unsigned int i=0; i<sg_p->size(); ++i) {
      (*sg_p)[i].family->setValue<PHAL::AlbanyTraits::SGResidual>((*sg_p_vals)[i]);
    }
  }

  // Set data in Workset struct, and perform fill via field manager
  {  
    PHAL::Workset workset(coordinates);

    workset.sg_expansion = sg_expansion;
    workset.sg_x         = sg_overlapped_x;
    workset.sg_xdot      = sg_overlapped_xdot;
    workset.sg_f         = sg_overlapped_f;

    workset.current_time = current_time;
    if (sg_xdot != NULL) workset.transientTerms = true;

    workset.worksetSize = worksetSize;
    for (int ws=0; ws < numWorksets; ws++) {
      workset.numCells = wsElNodeID[ws].size();
      workset.wsElNodeID = wsElNodeID[ws];
      workset.wsCoords = coords[ws];
      workset.EBName = wsEBNames[ws];

      workset.oldState = stateMgr.getOldStateVariables(ws);
      workset.newState = stateMgr.getNewStateVariables(ws);

      // FillType template argument used to specialize Sacado
      fm->evaluateFields<PHAL::AlbanyTraits::SGResidual>(workset);
    }
  } 

  // Assemble global residual
  for (int i=0; i<sg_f.size(); i++) {
    sg_f[i].Export((*sg_overlapped_f)[i], *exporter, Add);
  }

  // Apply Dirichlet conditions using dfm (Dirchelt Field Manager)
  if (dfm!=Teuchos::null) { 
    PHAL::Workset workset(coordinates);

    workset.sg_f = Teuchos::rcpFromRef(sg_f);
    workset.nodeSets = Teuchos::rcpFromRef(disc->getNodeSets());
    workset.nodeSetCoords = Teuchos::rcpFromRef (disc->getNodeSetCoords());
    workset.sg_x = Teuchos::rcpFromRef(sg_x);
    if (sg_xdot != NULL) workset.transientTerms = true;

    // FillType template argument used to specialize Sacado
    dfm->evaluateFields<PHAL::AlbanyTraits::SGResidual>(workset);

  }
}

void
Albany::Application::computeGlobalSGJacobian(
			double alpha, double beta,
			const double current_time,
			const Stokhos::VectorOrthogPoly<Epetra_Vector>* sg_xdot,
			const Stokhos::VectorOrthogPoly<Epetra_Vector>& sg_x,
			const ParamVec* p,
			const ParamVec* sg_p,
			const Teuchos::Array<SGType>* sg_p_vals,
			Stokhos::VectorOrthogPoly<Epetra_Vector>* sg_f,
			Stokhos::VectorOrthogPoly<Epetra_CrsMatrix>& sg_jac)
{
  if (!setupCalledSGJacobian) {
    setupCalledSGJacobian=true;
    fm->postRegistrationSetupForType<PHAL::AlbanyTraits::SGJacobian>(*disc);
    if (dfm!=Teuchos::null)
      dfm->postRegistrationSetupForType<PHAL::AlbanyTraits::SGJacobian>(*disc);
    writeGraphVisFile();
  }

  Teuchos::TimeMonitor Timer(*timers[5]); //start timer
  //TEUCHOS_FUNC_TIME_MONITOR("Albany::Application::computeGlobalSGJacobian");

  for (int i=0; i<sg_x.size(); i++) {

    // Scatter x to the overlapped distrbution
    (*sg_overlapped_x)[i].Import(sg_x[i], *importer, Insert);

    // Scatter xdot to the overlapped distribution
    if (sg_xdot != NULL)
      (*sg_overlapped_xdot)[i].Import((*sg_xdot)[i], *importer, Insert);

    // Zero out overlapped residual
    if (sg_f != NULL) {
      (*sg_overlapped_f)[i].PutScalar(0.0);
      (*sg_f)[i].PutScalar(0.0);
    }

  }

  // Create, resize and initialize overlapped Jacobians
  if (sg_overlapped_jac == Teuchos::null || 
      sg_overlapped_jac->size() != sg_jac.size()) {
    Teuchos::RCP<const Stokhos::OrthogPolyBasis<int,double> > sg_basis =
      sg_expansion->getBasis();
    Teuchos::RCP<Epetra_LocalMap> sg_overlap_jac_map = 
      Teuchos::rcp(new Epetra_LocalMap(sg_jac.size(), 0, 
				       sg_overlap_map->Comm()));
    sg_overlapped_jac = 
      Teuchos::rcp(new Stokhos::VectorOrthogPoly<Epetra_CrsMatrix>(
		     sg_basis, sg_overlap_jac_map, *overlapped_jac));
  }
  for (int i=0; i<sg_overlapped_jac->size(); i++)
    (*sg_overlapped_jac)[i].PutScalar(0.0);

  // Zero out overlapped Jacobian
  for (int i=0; i<sg_jac.size(); i++)
    (*sg_overlapped_jac)[i].PutScalar(0.0);

  // Set real parameters
  if (p != NULL) {
    for (unsigned int i=0; i<p->size(); ++i) {
      (*p)[i].family->setRealValueForAllTypes((*p)[i].baseValue);
    }
  }
#ifdef ALBANY_CUTR
  if (shapeParamsHaveBeenReset) {
    Teuchos::TimeMonitor Timer(*timers[8]); //start timer
*out << " Calling moveMesh with params: " << std::setprecision(8);
for (unsigned int i=0; i<shapeParams.size(); i++) *out << shapeParams[i] << "  ";
*out << endl;
    meshMover->moveMesh(shapeParams, morphFromInit);
    coordinates = disc->getCoordinates();
    shapeParamsHaveBeenReset = false;
  }
#endif

  // Set SG parameters
  if (sg_p != NULL && sg_p_vals != NULL) {
    for (unsigned int i=0; i<sg_p->size(); ++i) {
      (*sg_p)[i].family->setValue<PHAL::AlbanyTraits::SGJacobian>((*sg_p_vals)[i]);
    }
  }

  Teuchos::RCP< Stokhos::VectorOrthogPoly<Epetra_Vector> > sg_overlapped_ff;
  if (sg_f != NULL)
    sg_overlapped_ff = sg_overlapped_f;

  // Set data in Workset struct, and perform fill via field manager
  {
    PHAL::Workset workset(coordinates);

    workset.sg_expansion = sg_expansion;
    workset.sg_x         = sg_overlapped_x;
    workset.sg_xdot      = sg_overlapped_xdot;
    workset.sg_f         = sg_overlapped_ff;

    workset.sg_Jac       = sg_overlapped_jac;
    workset.j_coeff      = beta;
    workset.m_coeff      = alpha;
    workset.current_time = current_time;
    workset.ignore_residual = ignore_residual_in_jacobian;
    if (sg_xdot != NULL) workset.transientTerms = true;

    workset.worksetSize = worksetSize;
    for (int ws=0; ws < numWorksets; ws++) {
      workset.numCells = wsElNodeID[ws].size();
      workset.wsElNodeID = wsElNodeID[ws];
      workset.wsCoords = coords[ws];
      workset.EBName = wsEBNames[ws];

      workset.oldState = stateMgr.getOldStateVariables(ws);
      workset.newState = stateMgr.getNewStateVariables(ws);

      // FillType template argument used to specialize Sacado
      fm->evaluateFields<PHAL::AlbanyTraits::SGJacobian>(workset);
    }
  } 
  
  // Assemble global residual
  if (sg_f != NULL)
    for (int i=0; i<sg_f->size(); i++)
      (*sg_f)[i].Export((*sg_overlapped_f)[i], *exporter, Add);
    
  // Assemble block Jacobians
  Teuchos::RCP<Epetra_CrsMatrix> jac;
  for (int i=0; i<sg_jac.size(); i++) {
    jac = sg_jac.getCoeffPtr(i);
    jac->PutScalar(0.0);
    jac->Export((*sg_overlapped_jac)[i], *exporter, Add);
    jac->FillComplete(true);
  }

  // Apply Dirichlet conditions using dfm (Dirchelt Field Manager)
  if (dfm!=Teuchos::null) {
    PHAL::Workset workset(coordinates);

    workset.sg_f = Teuchos::rcp(sg_f,false);
    workset.sg_Jac = Teuchos::rcpFromRef(sg_jac);
    workset.j_coeff = beta;
    workset.sg_x = Teuchos::rcpFromRef(sg_x);;
    if (sg_xdot != NULL) workset.transientTerms = true;

    workset.nodeSets = Teuchos::rcpFromRef (disc->getNodeSets());
    workset.nodeSetCoords = Teuchos::rcpFromRef (disc->getNodeSetCoords());

    // FillType template argument used to specialize Sacado
    dfm->evaluateFields<PHAL::AlbanyTraits::SGJacobian>(workset);
  } 
}

void
Albany::Application::
evaluateSGResponses(const Stokhos::VectorOrthogPoly<Epetra_Vector>* sg_xdot,
		    const Stokhos::VectorOrthogPoly<Epetra_Vector>& sg_x,
		    const ParamVec* p,
		    const ParamVec* sg_p,
		    const Teuchos::Array<SGType>* sg_p_vals,
		    Stokhos::VectorOrthogPoly<Epetra_Vector>& sg_g)
{
  TEUCHOS_FUNC_TIME_MONITOR("Albany::Application::evaluateSGResponses");

  const Epetra_Comm& comm = sg_x[0].Map().Comm();
  unsigned int offset = 0;
  Teuchos::RCP<const Stokhos::OrthogPolyBasis<int,double> > basis = 
    sg_x.basis();
  for (unsigned int i=0; i<responses.size(); i++) {

    // Create Epetra_Map for response function
    unsigned int num_responses = responses[i]->numResponses();
    Epetra_LocalMap local_response_map(num_responses, 0, comm);

    // Create Epetra_Vector for response function
    Stokhos::VectorOrthogPoly<Epetra_Vector> local_sg_g(
      basis, sg_overlap_map, local_response_map);
    
    // Evaluate response function
    responses[i]->evaluateSGResponses(sg_xdot, sg_x, p, sg_p, sg_p_vals,
				      local_sg_g);

    // Copy result into combined result
    for (int k=0; k<sg_g.size(); k++)
      for (unsigned int j=0; j<num_responses; j++)
	sg_g[k][offset+j] = local_sg_g[k][j];

    // Increment offset in combined result
    offset += num_responses;
  }
}

void
Albany::Application::computeGlobalMPResidual(
			const double current_time,
			const Stokhos::ProductContainer<Epetra_Vector>* mp_xdot,
			const Stokhos::ProductContainer<Epetra_Vector>& mp_x,
			const ParamVec* p,
			const ParamVec* mp_p,
			const Teuchos::Array<MPType>* mp_p_vals,
			Stokhos::ProductContainer<Epetra_Vector>& mp_f)
{
  if (!setupCalledMPResidual) {
    setupCalledMPResidual=true;
    fm->postRegistrationSetupForType<PHAL::AlbanyTraits::MPResidual>(*disc);
    if (dfm!=Teuchos::null)
      dfm->postRegistrationSetupForType<PHAL::AlbanyTraits::MPResidual>(*disc);
    writeGraphVisFile();
  }

  Teuchos::TimeMonitor Timer(*timers[6]); //start timer
  //TEUCHOS_FUNC_TIME_MONITOR("Albany::Application::computeGlobalMPResidual");

  // Create overlapped multi-point Epetra objects
  if (mp_overlapped_x == Teuchos::null || 
      mp_overlapped_x->size() != mp_x.size()) {
    mp_overlapped_x = 
      Teuchos::rcp(new Stokhos::ProductContainer<Epetra_Vector>(
		     mp_x.map(), *overlapped_x));

    if (mp_xdot != NULL)
      mp_overlapped_xdot = 
	Teuchos::rcp(new Stokhos::ProductContainer<Epetra_Vector>(
		       mp_xdot->map(), *overlapped_xdot));

  }

  if (mp_overlapped_f == Teuchos::null || 
      mp_overlapped_f->size() != mp_f.size()) {
    mp_overlapped_f = 
      Teuchos::rcp(new Stokhos::ProductContainer<Epetra_Vector>(
		     mp_f.map(), *overlapped_f));
  }

  for (int i=0; i<mp_x.size(); i++) {

    // Scatter x to the overlapped distrbution
    (*mp_overlapped_x)[i].Import(mp_x[i], *importer, Insert);

    // Scatter xdot to the overlapped distribution
    if (mp_xdot != NULL)
      (*mp_overlapped_xdot)[i].Import((*mp_xdot)[i], *importer, Insert);

    // Zero out overlapped residual
    (*mp_overlapped_f)[i].PutScalar(0.0);
    mp_f[i].PutScalar(0.0);

  }

  // Set real parameters
  if (p != NULL) {
    for (unsigned int i=0; i<p->size(); ++i) {
      (*p)[i].family->setRealValueForAllTypes((*p)[i].baseValue);
    }
  }
#ifdef ALBANY_CUTR
  if (shapeParamsHaveBeenReset) {
    Teuchos::TimeMonitor Timer(*timers[8]); //start timer
*out << " Calling moveMesh with params: " << std::setprecision(8);
for (unsigned int i=0; i<shapeParams.size(); i++) *out << shapeParams[i] << "  ";
*out << endl;
    meshMover->moveMesh(shapeParams, morphFromInit);
    coordinates = disc->getCoordinates();
    shapeParamsHaveBeenReset = false;
  }
#endif

  // Set MP parameters
  if (mp_p != NULL && mp_p_vals != NULL) {
    for (unsigned int i=0; i<mp_p->size(); ++i) {
      (*mp_p)[i].family->setValue<PHAL::AlbanyTraits::MPResidual>((*mp_p_vals)[i]);
    }
  }

  // Set data in Workset struct, and perform fill via field manager
  {  
    PHAL::Workset workset(coordinates);

    workset.mp_x         = mp_overlapped_x;
    workset.mp_xdot      = mp_overlapped_xdot;
    workset.mp_f         = mp_overlapped_f;

    workset.current_time = current_time;
    if (mp_xdot != NULL) workset.transientTerms = true;

    workset.worksetSize = worksetSize;
    for (int ws=0; ws < numWorksets; ws++) {
      workset.numCells = wsElNodeID[ws].size();
      workset.wsElNodeID = wsElNodeID[ws];
      workset.wsCoords = coords[ws];
      workset.EBName = wsEBNames[ws];

      workset.oldState = stateMgr.getOldStateVariables(ws);
      workset.newState = stateMgr.getNewStateVariables(ws);

      // FillType template argument used to specialize Sacado
      fm->evaluateFields<PHAL::AlbanyTraits::MPResidual>(workset);
    }
  } 

  // Assemble global residual
  for (int i=0; i<mp_f.size(); i++) {
    mp_f[i].Export((*mp_overlapped_f)[i], *exporter, Add);
  }

  // Apply Dirichlet conditions using dfm (Dirchelt Field Manager)
  if (dfm!=Teuchos::null) { 
    PHAL::Workset workset(coordinates);

    workset.mp_f = Teuchos::rcpFromRef(mp_f);
    workset.nodeSets = Teuchos::rcpFromRef(disc->getNodeSets());
    workset.nodeSetCoords = Teuchos::rcpFromRef (disc->getNodeSetCoords());
    workset.mp_x = Teuchos::rcpFromRef(mp_x);
    if (mp_xdot != NULL) workset.transientTerms = true;

    // FillType template argument used to specialize Sacado
    dfm->evaluateFields<PHAL::AlbanyTraits::MPResidual>(workset);

  }
}

void
Albany::Application::computeGlobalMPJacobian(
			double alpha, double beta,
			const double current_time,
			const Stokhos::ProductContainer<Epetra_Vector>* mp_xdot,
			const Stokhos::ProductContainer<Epetra_Vector>& mp_x,
			const ParamVec* p,
			const ParamVec* mp_p,
			const Teuchos::Array<MPType>* mp_p_vals,
			Stokhos::ProductContainer<Epetra_Vector>* mp_f,
			Stokhos::ProductContainer<Epetra_CrsMatrix>& mp_jac)
{
  if (!setupCalledMPJacobian) {
    setupCalledMPJacobian=true;
    fm->postRegistrationSetupForType<PHAL::AlbanyTraits::MPJacobian>(*disc);
    if (dfm!=Teuchos::null)
      dfm->postRegistrationSetupForType<PHAL::AlbanyTraits::MPJacobian>(*disc);
    writeGraphVisFile();
  }

  Teuchos::TimeMonitor Timer(*timers[7]); //start timer
  //TEUCHOS_FUNC_TIME_MONITOR("Albany::Application::computeGlobalMPJacobian");

  // Create overlapped multi-point Epetra objects
  if (mp_overlapped_x == Teuchos::null || 
      mp_overlapped_x->size() != mp_x.size()) {
    mp_overlapped_x = 
      Teuchos::rcp(new Stokhos::ProductContainer<Epetra_Vector>(
		     mp_x.map(), *overlapped_x));

    if (mp_xdot != NULL)
      mp_overlapped_xdot = 
	Teuchos::rcp(new Stokhos::ProductContainer<Epetra_Vector>(
		       mp_xdot->map(), *overlapped_xdot));

  }

  if (mp_f != NULL && (mp_overlapped_f == Teuchos::null || 
		       mp_overlapped_f->size() != mp_f->size()))
    mp_overlapped_f = 
      Teuchos::rcp(new Stokhos::ProductContainer<Epetra_Vector>(
		     mp_f->map(), *overlapped_f));

  if (mp_overlapped_jac == Teuchos::null || 
      mp_overlapped_jac->size() != mp_jac.size())
    mp_overlapped_jac = 
      Teuchos::rcp(new Stokhos::ProductContainer<Epetra_CrsMatrix>(
		     mp_jac.map(), *overlapped_jac));

  for (int i=0; i<mp_x.size(); i++) {

    // Scatter x to the overlapped distrbution
    (*mp_overlapped_x)[i].Import(mp_x[i], *importer, Insert);

    // Scatter xdot to the overlapped distribution
    if (mp_xdot != NULL)
      (*mp_overlapped_xdot)[i].Import((*mp_xdot)[i], *importer, Insert);

    // Zero out overlapped residual
    if (mp_f != NULL) {
      (*mp_overlapped_f)[i].PutScalar(0.0);
      (*mp_f)[i].PutScalar(0.0);
    }

    mp_jac[i].PutScalar(0.0);
    (*mp_overlapped_jac)[i].PutScalar(0.0);

  }

  // Set real parameters
  if (p != NULL) {
    for (unsigned int i=0; i<p->size(); ++i) {
      (*p)[i].family->setRealValueForAllTypes((*p)[i].baseValue);
    }
  }
#ifdef ALBANY_CUTR
  if (shapeParamsHaveBeenReset) {
    Teuchos::TimeMonitor Timer(*timers[8]); //start timer
*out << " Calling moveMesh with params: " << std::setprecision(8);
for (unsigned int i=0; i<shapeParams.size(); i++) *out << shapeParams[i] << "  ";
*out << endl;
    meshMover->moveMesh(shapeParams, morphFromInit);
    coordinates = disc->getCoordinates();
    shapeParamsHaveBeenReset = false;
  }
#endif

  // Set MP parameters
  if (mp_p != NULL && mp_p_vals != NULL) {
    for (unsigned int i=0; i<mp_p->size(); ++i) {
      (*mp_p)[i].family->setValue<PHAL::AlbanyTraits::MPJacobian>((*mp_p_vals)[i]);
    }
  }

  Teuchos::RCP< Stokhos::ProductContainer<Epetra_Vector> > mp_overlapped_ff;
  if (mp_f != NULL)
    mp_overlapped_ff = mp_overlapped_f;

  // Set data in Workset struct, and perform fill via field manager
  {
    PHAL::Workset workset(coordinates);

    workset.mp_x         = mp_overlapped_x;
    workset.mp_xdot      = mp_overlapped_xdot;
    workset.mp_f         = mp_overlapped_ff;

    workset.mp_Jac       = mp_overlapped_jac;
    workset.j_coeff      = beta;
    workset.m_coeff      = alpha;
    workset.current_time = current_time;
    workset.ignore_residual = ignore_residual_in_jacobian;
    if (mp_xdot != NULL) workset.transientTerms = true;

    workset.worksetSize = worksetSize;
    for (int ws=0; ws < numWorksets; ws++) {
      workset.numCells = wsElNodeID[ws].size();
      workset.wsElNodeID = wsElNodeID[ws];
      workset.wsCoords = coords[ws];
      workset.EBName = wsEBNames[ws];

      workset.oldState = stateMgr.getOldStateVariables(ws);
      workset.newState = stateMgr.getNewStateVariables(ws);

      // FillType template argument used to specialize Sacado
      fm->evaluateFields<PHAL::AlbanyTraits::MPJacobian>(workset);
    }
  } 
  
  // Assemble global residual
  if (mp_f != NULL)
    for (int i=0; i<mp_f->size(); i++)
      (*mp_f)[i].Export((*mp_overlapped_f)[i], *exporter, Add);
    
  // Assemble block Jacobians
  Teuchos::RCP<Epetra_CrsMatrix> jac;
  for (int i=0; i<mp_jac.size(); i++) {
    jac = mp_jac.getCoeffPtr(i);
    jac->PutScalar(0.0);
    jac->Export((*mp_overlapped_jac)[i], *exporter, Add);
    jac->FillComplete(true);
  }

  // Apply Dirichlet conditions using dfm (Dirchelt Field Manager)
  if (dfm!=Teuchos::null) {
    PHAL::Workset workset(coordinates);

    workset.mp_f = Teuchos::rcp(mp_f,false);
    workset.mp_Jac = Teuchos::rcpFromRef(mp_jac);
    workset.j_coeff = beta;
    workset.mp_x = Teuchos::rcpFromRef(mp_x);;
    if (mp_xdot != NULL) workset.transientTerms = true;

    workset.nodeSets = Teuchos::rcpFromRef (disc->getNodeSets());
    workset.nodeSetCoords = Teuchos::rcpFromRef (disc->getNodeSetCoords());

    // FillType template argument used to specialize Sacado
    dfm->evaluateFields<PHAL::AlbanyTraits::MPJacobian>(workset);
  } 
}

void
Albany::Application::
evaluateMPResponses(const Stokhos::ProductContainer<Epetra_Vector>* mp_xdot,
		    const Stokhos::ProductContainer<Epetra_Vector>& mp_x,
		    ParamVec* p,
		    ParamVec* mp_p,
		    const Teuchos::Array<MPType>* mp_p_vals,
		    Stokhos::ProductContainer<Epetra_Vector>& mp_g)
{
  TEUCHOS_FUNC_TIME_MONITOR("Albany::Application::evaluateMPResponses");
  
  Teuchos::Array< Teuchos::RCP<ParamVec> > p_array(2);
  p_array[0] = Teuchos::rcp(p, false);
  p_array[1] = Teuchos::rcp(mp_p, false);
  const Epetra_Vector* xdot = NULL;
  for (int i=0; i<mp_x.size(); i++) {
    if (mp_p != NULL && mp_p_vals != NULL)
      for (unsigned int j=0; j<mp_p->size(); ++j)
	(*mp_p)[j].baseValue = (*mp_p_vals)[j].coeff(i);

    if (mp_xdot != NULL)
      xdot = mp_xdot->getCoeffPtr(i).get();
    
    // Evaluate response function
    evaluateResponses(xdot, mp_x[i], p_array, mp_g[i]);
  }
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
  Albany::DummyParameterAccessor<PHAL::AlbanyTraits::SGResidual, SPL_Traits> * dSGR =
   new Albany::DummyParameterAccessor<PHAL::AlbanyTraits::SGResidual, SPL_Traits>();
  Albany::DummyParameterAccessor<PHAL::AlbanyTraits::SGJacobian, SPL_Traits> * dSGJ =
   new Albany::DummyParameterAccessor<PHAL::AlbanyTraits::SGJacobian, SPL_Traits>();
  Albany::DummyParameterAccessor<PHAL::AlbanyTraits::MPResidual, SPL_Traits> * dMPR =
   new Albany::DummyParameterAccessor<PHAL::AlbanyTraits::MPResidual, SPL_Traits>();
  Albany::DummyParameterAccessor<PHAL::AlbanyTraits::MPJacobian, SPL_Traits> * dMPJ =
   new Albany::DummyParameterAccessor<PHAL::AlbanyTraits::MPJacobian, SPL_Traits>();

  // Register Parameter for Residual fill using "this->getValue" but
  // create dummy ones for other type that will not be used.
  for (int i=0; i<numShParams; i++) {
    *out << "Registering Shape Param " << shapeParamNames[i] << endl;
    new Sacado::ParameterRegistration<PHAL::AlbanyTraits::Residual, SPL_Traits>
      (shapeParamNames[i], this, paramLib);
    new Sacado::ParameterRegistration<PHAL::AlbanyTraits::Jacobian, SPL_Traits>
      (shapeParamNames[i], dJ, paramLib);
    new Sacado::ParameterRegistration<PHAL::AlbanyTraits::Tangent, SPL_Traits>
      (shapeParamNames[i], dT, paramLib);
    new Sacado::ParameterRegistration<PHAL::AlbanyTraits::SGResidual, SPL_Traits>
      (shapeParamNames[i], dSGR, paramLib);
    new Sacado::ParameterRegistration<PHAL::AlbanyTraits::SGJacobian, SPL_Traits>
      (shapeParamNames[i], dSGJ, paramLib);
    new Sacado::ParameterRegistration<PHAL::AlbanyTraits::MPResidual, SPL_Traits>
      (shapeParamNames[i], dMPR, paramLib);
    new Sacado::ParameterRegistration<PHAL::AlbanyTraits::MPJacobian, SPL_Traits>
      (shapeParamNames[i], dMPJ, paramLib);
  }
}

PHAL::AlbanyTraits::Residual::ScalarT&
Albany::Application::getValue(const std::string& name)
{
  int index=-1;
  for (unsigned int i=0; i<shapeParamNames.size(); i++) {
    if (name == shapeParamNames[i]) index = i;
  }
  TEST_FOR_EXCEPTION(index==-1,  std::logic_error,
        "Error in GatherCoordinateVector::getValue, \n" <<
        "   Unrecognized param name: " << name << endl);

  shapeParamsHaveBeenReset = true;

  return shapeParams[index];
}

void Albany::Application::writeGraphVisFile() const
{
  // Write just once for first evaluation type called
  if (getMap()->Comm().MyPID()==0 && phxGraphVisDetail > 0) {
     bool detail = false;
     if (phxGraphVisDetail > 1) detail=true;
     *out << "Phalanx writing graphviz file for graph of residual fill (detail ="
          << phxGraphVisDetail << ")"<<endl;
     *out << "Process using 'dot -Tpng -O phalanx_graph' " << endl;
     if (setupCalledResidual) 
       fm->writeGraphvizFile<PHAL::AlbanyTraits::Residual>("phalanx_graph",detail,detail);
     else if (setupCalledJacobian)
       fm->writeGraphvizFile<PHAL::AlbanyTraits::Jacobian>("phalanx_graph",detail,detail);
     else if (setupCalledTangent) 
       fm->writeGraphvizFile<PHAL::AlbanyTraits::Tangent>("phalanx_graph",detail,detail);
     else if (setupCalledSGResidual) 
       fm->writeGraphvizFile<PHAL::AlbanyTraits::SGResidual>("phalanx_graph",detail,detail);
     else if (setupCalledSGJacobian)
       fm->writeGraphvizFile<PHAL::AlbanyTraits::SGJacobian>("phalanx_graph",detail,detail);
     else if (setupCalledMPResidual) 
       fm->writeGraphvizFile<PHAL::AlbanyTraits::MPResidual>("phalanx_graph",detail,detail);
     else if (setupCalledMPJacobian)
       fm->writeGraphvizFile<PHAL::AlbanyTraits::MPJacobian>("phalanx_graph",detail,detail);
     *out << "(If failed to find 1_42, try removing <trilinos-install>/include/boost/)" << endl;

     phxGraphVisDetail = -1;
   }
}

Teuchos::RCP<Epetra_Operator> 
Albany::Application::buildWrappedOperator(const Teuchos::RCP<Epetra_Operator>& Jac,
                                          const Teuchos::RCP<Epetra_Operator>& wrapInput,
                                          bool reorder) const
{
  Teuchos::RCP<Epetra_Operator> wrappedOp = wrapInput;
  // if only one block just use orignal jacobian
  if(blockDecomp.size()==1) return (Jac);

  // initialize jacobian
  if(wrappedOp==Teuchos::null)
     wrappedOp = Teuchos::rcp(new Teko::Epetra::StridedEpetraOperator(blockDecomp,Jac));
  else 
     Teuchos::rcp_dynamic_cast<Teko::Epetra::StridedEpetraOperator>(wrappedOp)->RebuildOps();

  // test blocked operator for correctness
  if(tekoParams->get("Test Blocked Operator",false)) {
     bool result
        = Teuchos::rcp_dynamic_cast<Teko::Epetra::StridedEpetraOperator>(wrappedOp)->testAgainstFullOperator(6,1e-14);

     *out << "Teko: Tested operator correctness:  " << (result ? "passed" : "FAILED!") << std::endl;
  }
  return wrappedOp;
}
