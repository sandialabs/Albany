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
#include "Albany_ResponseFactory.hpp"
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

#ifdef ALBANY_SEACAS
  #include "Albany_STKDiscretization.hpp"
#endif

#include "Albany_ScalarResponseFunction.hpp"

using Teuchos::ArrayRCP;
using Teuchos::RCP;
using Teuchos::rcp;
using Teuchos::rcp_dynamic_cast;
using Teuchos::TimeMonitor;

Albany::Application::
Application(const RCP<const Epetra_Comm>& comm_,
	    const RCP<Teuchos::ParameterList>& params,
	    const RCP<const Epetra_Vector>& initial_guess) :
  comm(comm_),
  out(Teuchos::VerboseObjectBase::getDefaultOStream()),
  physicsBasedPreconditioner(false),
  shapeParamsHaveBeenReset(false),
  morphFromInit(true), perturbBetaForDirichlets(0.0),
  phxGraphVisDetail()
{
  defineTimers();

  // Create parameter library
  paramLib = rcp(new ParamLib);

  // Create problem object
  RCP<Teuchos::ParameterList> problemParams = 
    Teuchos::sublist(params, "Problem", true);
  Albany::ProblemFactory problemFactory(problemParams, paramLib, comm);
  problem = problemFactory.create();

  // Validate Problem parameters against list for this specific problem
  problemParams->validateParameters(*(problem->getValidProblemParameters()),0);

  // Save the solution method to be used
  string solutionMethod = problemParams->get("Solution Method", "Steady");
  if(solutionMethod == "Steady") 
    solMethod = Steady;
  else if(solutionMethod == "Continuation") 
    solMethod = Continuation;
  else if(solutionMethod == "Transient") 
    solMethod = Transient;
  else if(solutionMethod == "Multi-Problem") 
    solMethod = MultiProblem;
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true, 
            std::logic_error, "Solution Method must be Steady, Transient, "
            << "Continuation, or Multi-Problem not : " << solutionMethod);


  // Register shape parameters for manipulation by continuation/optimization
  if (problemParams->get("Enable Cubit Shape Parameters",false)) {
#ifdef ALBANY_CUTR
    TimeMonitor Timer(*timers[10]); //start timer
    meshMover = rcp(new CUTR::CubitMeshMover
          (problemParams->get<std::string>("Cubit Base Filename")));

    meshMover->getShapeParams(shapeParamNames, shapeParams);
    *out << "SSS : Registering " << shapeParams.size() << " Shape Parameters" << endl;

    registerShapeParameters();

#else
  TEUCHOS_TEST_FOR_EXCEPTION(problemParams->get("Enable Cubit Shape Parameters",false), std::logic_error,
			     "Cubit requested but not Compiled in!");
#endif
  }

//  adapter = rcp(new Adaptation(problemParams));

  physicsBasedPreconditioner = problemParams->get("Use Physics-Based Preconditioner",false);
  if (physicsBasedPreconditioner) 
    tekoParams = Teuchos::sublist(problemParams, "Teko", true);

  // Create discretization object
  RCP<Teuchos::ParameterList> discParams = 
    Teuchos::sublist(params, "Discretization", true);
  Albany::DiscretizationFactory discFactory(discParams, comm);
#ifdef ALBANY_CUTR
  discFactory.setMeshMover(meshMover);
#endif

//  discFactory.setAdapter(adapter);

  // Get mesh specification object: worksetSize, cell topology, etc
  ArrayRCP<RCP<Albany::MeshSpecsStruct> > meshSpecs = 
    discFactory.createMeshSpecs();

  problem->buildProblem(meshSpecs, stateMgr);

  // Construct responses
  // This really needs to happen after the discretization is created for
  // distributed responses, but currently it can't be moved because there
  // are responses that setup states, which has to happen before the 
  // discreatization is created.  We will delay setup of the distributed
  // responses to deal with this temporarily.
  Teuchos::ParameterList& responseList = 
    problemParams->sublist("Response Functions");
  ResponseFactory responseFactory(Teuchos::rcp(this,false), problem, meshSpecs, 
				  Teuchos::rcp(&stateMgr,false));
  responses = responseFactory.createResponseFunctions(responseList);

  // Build state field manager
  sfm.resize(meshSpecs.size());
  Teuchos::RCP<PHX::DataLayout> dummy =
    Teuchos::rcp(new PHX::MDALayout<Dummy>(0));
  std::vector<string>responseIDs_to_require = 
    stateMgr.getResidResponseIDsToRequire();
  for (int ps=0; ps<meshSpecs.size(); ps++) {
    sfm[ps] = Teuchos::rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
    Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> > tags = 
      problem->buildEvaluators(*sfm[ps], *meshSpecs[ps], stateMgr, 
			       BUILD_STATE_FM, Teuchos::null);
    std::vector<string>::const_iterator it;
    for (it = responseIDs_to_require.begin(); 
	 it != responseIDs_to_require.end(); 
	 it++) {
      const string& responseID = *it;
      PHX::Tag<PHAL::AlbanyTraits::Residual::ScalarT> res_response_tag(
	responseID, dummy);
      sfm[ps]->requireField<PHAL::AlbanyTraits::Residual>(res_response_tag);
    }
    sfm[ps]->postRegistrationSetup("");
  }
  

  // Create the full mesh
  neq = problem->numEquations();
  disc = discFactory.createDiscretization(neq, stateMgr.getStateInfoStruct());

  // Load connectivity map and coordinates 
  wsElNodeEqID = disc->getWsElNodeEqID();
  coords = disc->getCoords();
  wsEBNames = disc->getWsEBNames();
  wsPhysIndex = disc->getWsPhysIndex();
  int numDim = meshSpecs[0]->numDim;
  numWorksets = wsElNodeEqID.size();

  // Create Epetra objects
  importer = rcp(new Epetra_Import(*(disc->getOverlapMap()), *(disc->getMap())));
  exporter = rcp(new Epetra_Export(*(disc->getOverlapMap()), *(disc->getMap())));
  overlapped_x = rcp(new Epetra_Vector(*(disc->getOverlapMap())));
  overlapped_xdot = rcp(new Epetra_Vector(*(disc->getOverlapMap())));
  overlapped_f = rcp(new Epetra_Vector(*(disc->getOverlapMap())));
  overlapped_jac = rcp(new Epetra_CrsMatrix(Copy, *(disc->getOverlapJacobianGraph())));
  tmp_ovlp_sol = rcp(new Epetra_Vector(*(disc->getOverlapMap())));

  // Initialize solution vector and time deriv
  initial_x = disc->getSolutionField();
  initial_x_dot = rcp(new Epetra_Vector(*(disc->getMap())));

  if (initial_guess != Teuchos::null) *initial_x = *initial_guess;
  else {
    overlapped_x->Import(*initial_x, *importer, Insert);
    Albany::InitialConditions(overlapped_x, wsElNodeEqID, wsEBNames, coords, neq, numDim,
                              problemParams->sublist("Initial Condition"),
                              disc->hasRestartSolution());
    Albany::InitialConditions(overlapped_xdot,  wsElNodeEqID, wsEBNames, coords, neq, numDim,
                              problemParams->sublist("Initial Condition Dot"));
    initial_x->Export(*overlapped_x, *exporter, Insert);
    initial_x_dot->Export(*overlapped_xdot, *exporter, Insert);
  }

  // Now that space is allocated in STK for state fields, initialize states
  stateMgr.setStateArrays(disc);

  // Now setup response functions (see note above)
  for (int i=0; i<responses.size(); i++)
    responses[i]->setup();

  // Set up memory for workset

  fm = problem->getFieldManager();
  TEUCHOS_TEST_FOR_EXCEPTION(fm==Teuchos::null, std::logic_error,
			     "getFieldManager not implemented!!!");
  dfm = problem->getDirichletFieldManager();
  nfm = problem->getNeumannFieldManager();

  if (comm->MyPID()==0) {
    phxGraphVisDetail= problemParams->get("Phalanx Graph Visualization Detail", 0);
    stateGraphVisDetail= phxGraphVisDetail;
  }

  *out << "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n"
       << " Sacado ParameterLibrary has been initialized:\n " 
       << *paramLib 
       << "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n"
       << endl;

  ignore_residual_in_jacobian = 
    problemParams->get("Ignore Residual In Jacobian", false);

  perturbBetaForDirichlets = problemParams->get("Perturb Dirichlet",0.0);

  is_adjoint = 
    problemParams->get("Solve Adjoint", false);
}

Albany::Application::
~Application()
{
}

//the following function sets the problem required for computing rigid body modes for elasticity
//added by IK, Feb. 2012
void
Albany::Application::getRBMInfo(int& numPDEs, int& numElasticityDim, int& numScalar, int& nullSpaceDim)
{
  problem->getRBMInfoForML(numPDEs, numElasticityDim, numScalar, nullSpaceDim);
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

RCP<const Epetra_Comm>
Albany::Application::
getComm() const
{
  return comm;
}

RCP<const Epetra_Map>
Albany::Application::
getMap() const
{
  return disc->getMap();
}

RCP<const Epetra_CrsGraph>
Albany::Application::
getJacobianGraph() const
{
  return disc->getJacobianGraph();
}

RCP<Epetra_Operator>
Albany::Application::
getPreconditioner()
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

   return rcp(new Teko::Epetra::InverseFactoryOperator(inverseFac));
}

RCP<const Epetra_Vector>
Albany::Application::
getInitialSolution() const
{
  return initial_x;
}

RCP<const Epetra_Vector>
Albany::Application::
getInitialSolutionDot() const
{
  return initial_x_dot;
}

RCP<ParamLib> 
Albany::Application::
getParamLib()
{
  return paramLib;
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
    sg_overlapped_f = 
      rcp(new Stokhos::EpetraVectorOrthogPoly(
	    sg_basis, sg_overlap_map, disc->getOverlapMap(), product_comm));
    // Delay creation of sg_overlapped_jac until needed
  }

  // Initialize responses
  for (int i=0; i<responses.size(); i++)
    responses[i]->init_sg(basis, quad, expansion, multiComm);
}

void
Albany::Application::
computeGlobalResidual(const double current_time,
		      const Epetra_Vector* xdot,
		      const Epetra_Vector& x,
		      const Teuchos::Array<ParamVec>& p,
		      Epetra_Vector& f)
{
  postRegSetup("Residual");

  TimeMonitor Timer(*timers[0]); //start timer

  // Scatter x and xdot to the overlapped distrbution
  overlapped_x->Import(x, *importer, Insert);
  if (xdot != NULL) overlapped_xdot->Import(*xdot, *importer, Insert);

  // Set parameters
  for (int i=0; i<p.size(); i++)
    for (unsigned int j=0; j<p[i].size(); j++)
      p[i][j].family->setRealValueForAllTypes(p[i][j].baseValue);

  // Mesh motion needs to occur here on the global mesh befor
  // it is potentially carved into worksets.
#ifdef ALBANY_CUTR
  static int first=true;
  if (shapeParamsHaveBeenReset) {
    TimeMonitor cubitTimer(*timers[10]); //start timer

*out << " Calling moveMesh with params: " << std::setprecision(8);
 for (unsigned int i=0; i<shapeParams.size(); i++) *out << shapeParams[i] << "  ";
*out << endl;
    meshMover->moveMesh(shapeParams, morphFromInit);
    coords = disc->getCoords();
    shapeParamsHaveBeenReset = false;
  }
#endif

//  adapter->adaptit();

  // Zero out overlapped residual
  overlapped_f->PutScalar(0.0);
  f.PutScalar(0.0);

  // Set data in Workset struct, and perform fill via field manager
  { 
    PHAL::Workset workset;

    if (!paramLib->isParameter("Time"))
      loadBasicWorksetInfo( workset, overlapped_x, overlapped_xdot, current_time );
    else 
      loadBasicWorksetInfo( workset, overlapped_x, overlapped_xdot,
			    paramLib->getRealValue<PHAL::AlbanyTraits::Residual>("Time") );
    
    workset.f        = overlapped_f;


    for (int ws=0; ws < numWorksets; ws++) {
      loadWorksetBucketInfo<PHAL::AlbanyTraits::Residual>(workset, ws);

      // FillType template argument used to specialize Sacado
      fm[wsPhysIndex[ws]]->evaluateFields<PHAL::AlbanyTraits::Residual>(workset);
      if (nfm!=Teuchos::null)
         nfm[wsPhysIndex[ws]]->evaluateFields<PHAL::AlbanyTraits::Residual>(workset);
    }
  }

  f.Export(*overlapped_f, *exporter, Add);

#ifdef ALBANY_SEACAS
  Albany::STKDiscretization* stkDisc =
    dynamic_cast<Albany::STKDiscretization*>(disc.get());
  stkDisc->setResidualField(f);
#endif

  // Apply Dirichlet conditions using dfm (Dirchelt Field Manager)
  if (dfm!=Teuchos::null) { 
    PHAL::Workset workset;

    workset.f = Teuchos::rcpFromRef(f);
    loadWorksetNodesetInfo(workset);
    workset.x = Teuchos::rcpFromRef(x);
    if ( paramLib->isParameter("Time") )
      workset.current_time = paramLib->getRealValue<PHAL::AlbanyTraits::Residual>("Time");
    else
      workset.current_time = current_time;
    if (xdot != NULL) workset.transientTerms = true;

    // FillType template argument used to specialize Sacado
    dfm->evaluateFields<PHAL::AlbanyTraits::Residual>(workset);
  } 
  //cout << "Global Resid f\n" << f << endl;
}

void
Albany::Application::
computeGlobalJacobian(const double alpha, 
		      const double beta,
		      const double current_time,
		      const Epetra_Vector* xdot,
		      const Epetra_Vector& x,
		      const Teuchos::Array<ParamVec>& p,
		      Epetra_Vector* f,
		      Epetra_CrsMatrix& jac)
{
  postRegSetup("Jacobian");

  TimeMonitor Timer(*timers[1]); //start timer

  // Scatter x and xdot to the overlapped distrbution
  overlapped_x->Import(x, *importer, Insert);
  if (xdot != NULL) overlapped_xdot->Import(*xdot, *importer, Insert);

  // Set parameters
  for (int i=0; i<p.size(); i++)
    for (unsigned int j=0; j<p[i].size(); j++)
      p[i][j].family->setRealValueForAllTypes(p[i][j].baseValue);

#ifdef ALBANY_CUTR
  if (shapeParamsHaveBeenReset) {
    TimeMonitor Timer(*timers[10]); //start timer

*out << " Calling moveMesh with params: " << std::setprecision(8);
 for (unsigned int i=0; i<shapeParams.size(); i++) *out << shapeParams[i] << "  ";
*out << endl;
    meshMover->moveMesh(shapeParams, morphFromInit);
    coords = disc->getCoords();
    shapeParamsHaveBeenReset = false;
  }
#endif

//  adapter->adaptit();

  // Zero out overlapped residual
  if (f != NULL) {
    overlapped_f->PutScalar(0.0);
    f->PutScalar(0.0);
  }

  // Zero out Jacobian
  overlapped_jac->PutScalar(0.0);
  jac.PutScalar(0.0);

  // Set data in Workset struct, and perform fill via field manager
  {
    PHAL::Workset workset;
    if (!paramLib->isParameter("Time"))
      loadBasicWorksetInfo( workset, overlapped_x, overlapped_xdot, current_time );
    else 
      loadBasicWorksetInfo( workset, overlapped_x, overlapped_xdot,
			    paramLib->getRealValue<PHAL::AlbanyTraits::Residual>("Time") );

    workset.f        = overlapped_f;
    workset.Jac      = overlapped_jac;
    loadWorksetJacobianInfo(workset, alpha, beta);

    for (int ws=0; ws < numWorksets; ws++) {
      loadWorksetBucketInfo<PHAL::AlbanyTraits::Jacobian>(workset, ws);

      // FillType template argument used to specialize Sacado
      fm[wsPhysIndex[ws]]->evaluateFields<PHAL::AlbanyTraits::Jacobian>(workset);
      if (nfm!=Teuchos::null)
        nfm[wsPhysIndex[ws]]->evaluateFields<PHAL::AlbanyTraits::Jacobian>(workset);
    }
  } 

  // Assemble global residual
  if (f != NULL)
    f->Export(*overlapped_f, *exporter, Add);

  // Assemble global Jacobian
  jac.Export(*overlapped_jac, *exporter, Add);

  // Apply Dirichlet conditions using dfm (Dirchelt Field Manager)
  if (dfm!=Teuchos::null) {
    PHAL::Workset workset;

    workset.f = rcp(f,false);
    workset.Jac = Teuchos::rcpFromRef(jac);
    workset.m_coeff = alpha;
    workset.j_coeff = beta;

    if ( paramLib->isParameter("Time") )
      workset.current_time = paramLib->getRealValue<PHAL::AlbanyTraits::Residual>("Time");
    else
      workset.current_time = current_time;

    if (beta==0.0 && perturbBetaForDirichlets>0.0) workset.j_coeff = perturbBetaForDirichlets;

    workset.x = Teuchos::rcpFromRef(x);;
    if (xdot != NULL) workset.transientTerms = true;

    loadWorksetNodesetInfo(workset);

    // FillType template argument used to specialize Sacado
    dfm->evaluateFields<PHAL::AlbanyTraits::Jacobian>(workset);
  }

  jac.FillComplete(true);
  //cout << "f " << *f << endl;;
  //cout << "J " << jac << endl;;
}

void
Albany::Application::
computeGlobalPreconditioner(const RCP<Epetra_CrsMatrix>& jac,
			    const RCP<Epetra_Operator>& prec)
{
  TimeMonitor Timer(*timers[2]); //start timer

  *out << "Computing WPrec by Teko" << endl;

  RCP<Teko::Epetra::InverseFactoryOperator> blockPrec
    = rcp_dynamic_cast<Teko::Epetra::InverseFactoryOperator>(prec);

  blockPrec->initInverse();

  wrappedJac = buildWrappedOperator(jac, wrappedJac);
  blockPrec->rebuildInverseOperator(wrappedJac);
}

void
Albany::Application::
computeGlobalTangent(const double alpha, 
		     const double beta,
		     const double current_time,
		     bool sum_derivs,
		     const Epetra_Vector* xdot,
		     const Epetra_Vector& x,
		     const Teuchos::Array<ParamVec>& par,
		     ParamVec* deriv_par,
		     const Epetra_MultiVector* Vx,
		     const Epetra_MultiVector* Vxdot,
		     const Epetra_MultiVector* Vp,
		     Epetra_Vector* f,
		     Epetra_MultiVector* JV,
		     Epetra_MultiVector* fp)
{
  postRegSetup("Tangent");

  TimeMonitor Timer(*timers[3]); //start timer

  // Scatter x and xdot to the overlapped distrbution
  overlapped_x->Import(x, *importer, Insert);
  if (xdot != NULL) overlapped_xdot->Import(*xdot, *importer, Insert);

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
    overlapped_Vxdot = 
      rcp(new Epetra_MultiVector(*(disc->getOverlapMap()), 
					  Vxdot->NumVectors()));
    overlapped_Vxdot->Import(*Vxdot, *importer, Insert);
  }

  // Set parameters
  for (int i=0; i<par.size(); i++)
    for (unsigned int j=0; j<par[i].size(); j++)
      par[i][j].family->setRealValueForAllTypes(par[i][j].baseValue);

  RCP<const Epetra_MultiVector > vp = rcp(Vp, false);
  RCP<ParamVec> params = rcp(deriv_par, false);

  // Zero out overlapped residual
  RCP<Epetra_Vector> overlapped_ff;
  if (f != NULL) {
    overlapped_ff = overlapped_f;
    overlapped_ff->PutScalar(0.0);
    f->PutScalar(0.0);
  }

  RCP<Epetra_MultiVector> overlapped_JV;
  if (JV != NULL) {
    overlapped_JV = 
      rcp(new Epetra_MultiVector(*(disc->getOverlapMap()), 
					  JV->NumVectors()));
    overlapped_JV->PutScalar(0.0);
    JV->PutScalar(0.0);
  }
  
  RCP<Epetra_MultiVector> overlapped_fp;
  if (fp != NULL) {
    overlapped_fp = 
      rcp(new Epetra_MultiVector(*(disc->getOverlapMap()), 
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
  ArrayRCP<ArrayRCP<double> > coord_derivs;
  // ws, sp, cell, node, dim
  ArrayRCP<ArrayRCP<ArrayRCP<ArrayRCP<ArrayRCP<double> > > > > ws_coord_derivs;
  ws_coord_derivs.resize(coords.size());
  std::vector<int> coord_deriv_indices;
#ifdef ALBANY_CUTR
  if (shapeParamsHaveBeenReset) {
    TimeMonitor Timer(*timers[10]); //start timer

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

    TEUCHOS_TEST_FOR_EXCEPTION( Vp != NULL, std::logic_error,
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

       shapeParams[shape_param_indices[i]] += pert;
*out << " Calling moveMesh with params: " << std::setprecision(8);
for (unsigned int ii=0; ii<shapeParams.size(); ii++) *out << shapeParams[ii] << "  ";
*out << endl;
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
*out << endl;
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

//  adapter->adaptit();

  // Set data in Workset struct, and perform fill via field manager
  {
    PHAL::Workset workset;
    if (!paramLib->isParameter("Time"))
      loadBasicWorksetInfo( workset, overlapped_x, overlapped_xdot, current_time );
    else 
      loadBasicWorksetInfo( workset, overlapped_x, overlapped_xdot,
			    paramLib->getRealValue<PHAL::AlbanyTraits::Residual>("Time") );

    workset.params = params;
    workset.Vx = overlapped_Vx;
    workset.Vxdot = overlapped_Vxdot;
    workset.Vp = vp;

    workset.f            = overlapped_f;
    workset.JV           = overlapped_JV;
    workset.fp           = overlapped_fp;
    workset.j_coeff      = beta;
    workset.m_coeff      = alpha;

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
        nfm[wsPhysIndex[ws]]->evaluateFields<PHAL::AlbanyTraits::Tangent>(workset);
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
    PHAL::Workset workset;

    workset.num_cols_x = num_cols_x;
    workset.num_cols_p = num_cols_p;
    workset.param_offset = param_offset;

    workset.f = rcp(f,false);
    workset.fp = rcp(fp,false);
    workset.JV = rcp(JV,false);
    workset.j_coeff = beta;
    workset.x = Teuchos::rcpFromRef(x);
    workset.Vx = rcp(Vx,false);
    if (xdot != NULL) workset.transientTerms = true;

    loadWorksetNodesetInfo(workset);

    if ( paramLib->isParameter("Time") )
      workset.current_time = paramLib->getRealValue<PHAL::AlbanyTraits::Residual>("Time");
    else
      workset.current_time = current_time;

    // FillType template argument used to specialize Sacado
    dfm->evaluateFields<PHAL::AlbanyTraits::Tangent>(workset);
  }

//*out << "fp " << *fp << endl;

}

void
Albany::Application::
evaluateResponse(int response_index,
		 const double current_time,
		 const Epetra_Vector* xdot,
		 const Epetra_Vector& x,
		 const Teuchos::Array<ParamVec>& p,
		 Epetra_Vector& g)
{  
  double t = current_time;
  if ( paramLib->isParameter("Time") ) 
    t = paramLib->getRealValue<PHAL::AlbanyTraits::Residual>("Time");

  responses[response_index]->evaluateResponse(t, xdot, x, p, g);
}

void
Albany::Application::
evaluateResponseTangent(int response_index,
			const double alpha, 
			const double beta,
			const double current_time,
			bool sum_derivs,
			const Epetra_Vector* xdot,
			const Epetra_Vector& x,
			const Teuchos::Array<ParamVec>& p,
			ParamVec* deriv_p,
			const Epetra_MultiVector* Vxdot,
			const Epetra_MultiVector* Vx,
			const Epetra_MultiVector* Vp,
			Epetra_Vector* g,
			Epetra_MultiVector* gx,
			Epetra_MultiVector* gp)
{
  double t = current_time;
  if ( paramLib->isParameter("Time") ) 
    t = paramLib->getRealValue<PHAL::AlbanyTraits::Residual>("Time");

  responses[response_index]->evaluateTangent(
    alpha, beta, t, sum_derivs, xdot, x, p, deriv_p, Vxdot, Vx, Vp, g, gx, gp);
}

void
Albany::Application::
evaluateResponseDerivative(
  int response_index,
  const double current_time,
  const Epetra_Vector* xdot,
  const Epetra_Vector& x,
  const Teuchos::Array<ParamVec>& p,
  ParamVec* deriv_p,
  Epetra_Vector* g,
  const EpetraExt::ModelEvaluator::Derivative& dg_dx,
  const EpetraExt::ModelEvaluator::Derivative& dg_dxdot,
  const EpetraExt::ModelEvaluator::Derivative& dg_dp)
{
  double t = current_time;
  if ( paramLib->isParameter("Time") ) 
    t = paramLib->getRealValue<PHAL::AlbanyTraits::Residual>("Time");

  responses[response_index]->evaluateDerivative(
    t, xdot, x, p, deriv_p, g, dg_dx, dg_dxdot, dg_dp);
}

void
Albany::Application::
computeGlobalSGResidual(
  const double current_time,
  const Stokhos::EpetraVectorOrthogPoly* sg_xdot,
  const Stokhos::EpetraVectorOrthogPoly& sg_x,
  const Teuchos::Array<ParamVec>& p,
  const Teuchos::Array<int>& sg_p_index,
  const Teuchos::Array< Teuchos::Array<SGType> >& sg_p_vals,
  Stokhos::EpetraVectorOrthogPoly& sg_f)
{
  postRegSetup("SGResidual");

  //std::cout << sg_x << std::endl;

  TimeMonitor Timer(*timers[4]); //start timer

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
    sg_overlapped_f = 
      rcp(new Stokhos::EpetraVectorOrthogPoly(
	    sg_basis, sg_overlap_map, disc->getOverlapMap(), product_comm));
  }

  for (int i=0; i<sg_x.size(); i++) {

    // Scatter x and xdot to the overlapped distrbution
    (*sg_overlapped_x)[i].Import(sg_x[i], *importer, Insert);
    if (sg_xdot != NULL) (*sg_overlapped_xdot)[i].Import((*sg_xdot)[i], *importer, Insert);

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
    TimeMonitor Timer(*timers[10]); //start timer
*out << " Calling moveMesh with params: " << std::setprecision(8);
for (unsigned int i=0; i<shapeParams.size(); i++) *out << shapeParams[i] << "  ";
*out << endl;
    meshMover->moveMesh(shapeParams, morphFromInit);
    coords = disc->getCoords();
    shapeParamsHaveBeenReset = false;
  }
#endif
//  adapter->adaptit();

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
    workset.sg_f         = sg_overlapped_f;

    workset.current_time = current_time;
    //workset.delta_time = timeMgr.getDeltaTime();
    if (sg_xdot != NULL) workset.transientTerms = true;

    for (int ws=0; ws < numWorksets; ws++) {
      loadWorksetBucketInfo<PHAL::AlbanyTraits::SGResidual>(workset, ws);

      // FillType template argument used to specialize Sacado
      fm[wsPhysIndex[ws]]->evaluateFields<PHAL::AlbanyTraits::SGResidual>(workset);
      if (nfm!=Teuchos::null)
        nfm[wsPhysIndex[ws]]->evaluateFields<PHAL::AlbanyTraits::SGResidual>(workset);
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
    workset.sg_x = Teuchos::rcpFromRef(sg_x);
    if (sg_xdot != NULL) workset.transientTerms = true;

    if ( paramLib->isParameter("Time") )
      workset.current_time = paramLib->getRealValue<PHAL::AlbanyTraits::Residual>("Time");
    else
      workset.current_time = current_time;

    // FillType template argument used to specialize Sacado
    dfm->evaluateFields<PHAL::AlbanyTraits::SGResidual>(workset);

  }

  //std::cout << sg_f << std::endl;
}

void
Albany::Application::
computeGlobalSGJacobian(
  const double alpha, 
  const double beta,
  const double current_time,
  const Stokhos::EpetraVectorOrthogPoly* sg_xdot,
  const Stokhos::EpetraVectorOrthogPoly& sg_x,
  const Teuchos::Array<ParamVec>& p,
  const Teuchos::Array<int>& sg_p_index,
  const Teuchos::Array< Teuchos::Array<SGType> >& sg_p_vals,
  Stokhos::EpetraVectorOrthogPoly* sg_f,
  Stokhos::VectorOrthogPoly<Epetra_CrsMatrix>& sg_jac)
{
  postRegSetup("SGJacobian");

  TimeMonitor Timer(*timers[5]); //start timer

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
    sg_overlapped_f = 
      rcp(new Stokhos::EpetraVectorOrthogPoly(
	    sg_basis, sg_overlap_map, disc->getOverlapMap(), product_comm));
  }

  for (int i=0; i<sg_x.size(); i++) {

    // Scatter x and xdot to the overlapped distrbution
    (*sg_overlapped_x)[i].Import(sg_x[i], *importer, Insert);
    if (sg_xdot != NULL) (*sg_overlapped_xdot)[i].Import((*sg_xdot)[i], *importer, Insert);

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
    TimeMonitor Timer(*timers[10]); //start timer
*out << " Calling moveMesh with params: " << std::setprecision(8);
for (unsigned int i=0; i<shapeParams.size(); i++) *out << shapeParams[i] << "  ";
*out << endl;
    meshMover->moveMesh(shapeParams, morphFromInit);
    coords = disc->getCoords();
    shapeParamsHaveBeenReset = false;
  }
#endif
//  adapter->adaptit();

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
    workset.sg_f         = sg_overlapped_ff;

    workset.sg_Jac       = sg_overlapped_jac;
    loadWorksetJacobianInfo(workset, alpha, beta);
    workset.current_time = current_time;
    //workset.delta_time = timeMgr.getDeltaTime();
    if (sg_xdot != NULL) workset.transientTerms = true;

    for (int ws=0; ws < numWorksets; ws++) {
      loadWorksetBucketInfo<PHAL::AlbanyTraits::SGJacobian>(workset, ws);

      // FillType template argument used to specialize Sacado
      fm[wsPhysIndex[ws]]->evaluateFields<PHAL::AlbanyTraits::SGJacobian>(workset);
      if (nfm!=Teuchos::null)
        nfm[wsPhysIndex[ws]]->evaluateFields<PHAL::AlbanyTraits::SGJacobian>(workset);
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
    workset.sg_x = Teuchos::rcpFromRef(sg_x);;
    if (sg_xdot != NULL) workset.transientTerms = true;

    loadWorksetNodesetInfo(workset);

    // FillType template argument used to specialize Sacado
    dfm->evaluateFields<PHAL::AlbanyTraits::SGJacobian>(workset);
  } 

  //std::cout << sg_jac << std::endl;
}

void
Albany::Application::
computeGlobalSGTangent(
  const double alpha, 
  const double beta, 
  const double current_time,
  bool sum_derivs,
  const Stokhos::EpetraVectorOrthogPoly* sg_xdot,
  const Stokhos::EpetraVectorOrthogPoly& sg_x,
  const Teuchos::Array<ParamVec>& par,
  const Teuchos::Array<int>& sg_p_index,
  const Teuchos::Array< Teuchos::Array<SGType> >& sg_p_vals,
  ParamVec* deriv_par,
  const Epetra_MultiVector* Vx,
  const Epetra_MultiVector* Vxdot,
  const Epetra_MultiVector* Vp,
  Stokhos::EpetraVectorOrthogPoly* sg_f,
  Stokhos::EpetraMultiVectorOrthogPoly* sg_JVx,
  Stokhos::EpetraMultiVectorOrthogPoly* sg_fVp)
{
  postRegSetup("SGTangent");

  TimeMonitor Timer(*timers[6]); //start timer

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
    sg_overlapped_f = 
      rcp(new Stokhos::EpetraVectorOrthogPoly(
	    sg_basis, sg_overlap_map, disc->getOverlapMap(), product_comm));
  }

  for (int i=0; i<sg_x.size(); i++) {

    // Scatter x and xdot to the overlapped distrbution
    (*sg_overlapped_x)[i].Import(sg_x[i], *importer, Insert);
    if (sg_xdot != NULL) (*sg_overlapped_xdot)[i].Import((*sg_xdot)[i], *importer, Insert);

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
    overlapped_Vxdot = 
      rcp(new Epetra_MultiVector(*(disc->getOverlapMap()), 
				 Vxdot->NumVectors()));
    overlapped_Vxdot->Import(*Vxdot, *importer, Insert);
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
    workset.Vx = overlapped_Vx;
    workset.Vxdot = overlapped_Vxdot;
    workset.Vp = vp;

    workset.sg_f         = sg_overlapped_ff;
    workset.sg_JV        = sg_overlapped_JVx;
    workset.sg_fp        = sg_overlapped_fVp;
    workset.j_coeff      = beta;
    workset.m_coeff      = alpha;

    workset.num_cols_x = num_cols_x;
    workset.num_cols_p = num_cols_p;
    workset.param_offset = param_offset;

    workset.current_time = current_time; //timeMgr.getCurrentTime();
    //    workset.delta_time = timeMgr.getDeltaTime();
    if (sg_xdot != NULL) workset.transientTerms = true;

    for (int ws=0; ws < numWorksets; ws++) {
      loadWorksetBucketInfo<PHAL::AlbanyTraits::SGTangent>(workset, ws);

      // FillType template argument used to specialize Sacado
      fm[wsPhysIndex[ws]]->evaluateFields<PHAL::AlbanyTraits::SGTangent>(workset);
      if (nfm!=Teuchos::null)
        nfm[wsPhysIndex[ws]]->evaluateFields<PHAL::AlbanyTraits::SGTangent>(workset);
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
    workset.sg_x = Teuchos::rcpFromRef(sg_x);
    workset.Vx = rcp(Vx,false);
    if (sg_xdot != NULL) workset.transientTerms = true;

    loadWorksetNodesetInfo(workset);

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
  const Stokhos::EpetraVectorOrthogPoly& sg_x,
  const Teuchos::Array<ParamVec>& p,
  const Teuchos::Array<int>& sg_p_index,
  const Teuchos::Array< Teuchos::Array<SGType> >& sg_p_vals,
  Stokhos::EpetraVectorOrthogPoly& sg_g)
{
  TEUCHOS_FUNC_TIME_MONITOR("Albany::Application::evaluateSGResponses");

  responses[response_index]->evaluateSGResponse(
    curr_time, sg_xdot, sg_x, p, sg_p_index, sg_p_vals, sg_g);
}

void
Albany::Application::
evaluateSGResponseTangent(
  int response_index,
  const double alpha, 
  const double beta, 
  const double current_time,
  bool sum_derivs,
  const Stokhos::EpetraVectorOrthogPoly* sg_xdot,
  const Stokhos::EpetraVectorOrthogPoly& sg_x,
  const Teuchos::Array<ParamVec>& p,
  const Teuchos::Array<int>& sg_p_index,
  const Teuchos::Array< Teuchos::Array<SGType> >& sg_p_vals,
  ParamVec* deriv_p,
  const Epetra_MultiVector* Vx,
  const Epetra_MultiVector* Vxdot,
  const Epetra_MultiVector* Vp,
  Stokhos::EpetraVectorOrthogPoly* sg_g,
  Stokhos::EpetraMultiVectorOrthogPoly* sg_JV,
  Stokhos::EpetraMultiVectorOrthogPoly* sg_gp)
{
  TEUCHOS_FUNC_TIME_MONITOR("Albany::Application::evaluateSGResponses");

  responses[response_index]->evaluateSGTangent(
    alpha, beta, current_time, sum_derivs, sg_xdot, sg_x, p, sg_p_index, 
    sg_p_vals, deriv_p, Vx, Vxdot, Vp, sg_g, sg_JV, sg_gp);
}

void
Albany::Application::
evaluateSGResponseDerivative(
  int response_index,
  const double current_time,
  const Stokhos::EpetraVectorOrthogPoly* sg_xdot,
  const Stokhos::EpetraVectorOrthogPoly& sg_x,
  const Teuchos::Array<ParamVec>& p,
  const Teuchos::Array<int>& sg_p_index,
  const Teuchos::Array< Teuchos::Array<SGType> >& sg_p_vals,
  ParamVec* deriv_p,
  Stokhos::EpetraVectorOrthogPoly* sg_g,
  const EpetraExt::ModelEvaluator::SGDerivative& sg_dg_dx,
  const EpetraExt::ModelEvaluator::SGDerivative& sg_dg_dxdot,
  const EpetraExt::ModelEvaluator::SGDerivative& sg_dg_dp)
{
  TEUCHOS_FUNC_TIME_MONITOR("Albany::Application::evaluateSGResponses");

  responses[response_index]->evaluateSGDerivative(
    current_time, sg_xdot, sg_x, p, sg_p_index, sg_p_vals, deriv_p,
    sg_g, sg_dg_dx, sg_dg_dxdot, sg_dg_dp);
}

void
Albany::Application::
computeGlobalMPResidual(
  const double current_time,
  const Stokhos::ProductEpetraVector* mp_xdot,
  const Stokhos::ProductEpetraVector& mp_x,
  const Teuchos::Array<ParamVec>& p,
  const Teuchos::Array<int>& mp_p_index,
  const Teuchos::Array< Teuchos::Array<MPType> >& mp_p_vals,
  Stokhos::ProductEpetraVector& mp_f)
{
  postRegSetup("MPResidual");

  TimeMonitor Timer(*timers[7]); //start timer

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
    TimeMonitor Timer(*timers[10]); //start timer
*out << " Calling moveMesh with params: " << std::setprecision(8);
for (unsigned int i=0; i<shapeParams.size(); i++) *out << shapeParams[i] << "  ";
*out << endl;
    meshMover->moveMesh(shapeParams, morphFromInit);
    coords = disc->getCoords();
    shapeParamsHaveBeenReset = false;
  }
#endif
//  adapter->adaptit();

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
    workset.mp_f         = mp_overlapped_f;

    workset.current_time = current_time; //timeMgr.getCurrentTime();
    //    workset.delta_time = timeMgr.getDeltaTime();
    if (mp_xdot != NULL) workset.transientTerms = true;

    for (int ws=0; ws < numWorksets; ws++) {
      loadWorksetBucketInfo<PHAL::AlbanyTraits::MPResidual>(workset, ws);

      // FillType template argument used to specialize Sacado
      fm[wsPhysIndex[ws]]->evaluateFields<PHAL::AlbanyTraits::MPResidual>(workset);
      if (nfm!=Teuchos::null)
        nfm[wsPhysIndex[ws]]->evaluateFields<PHAL::AlbanyTraits::MPResidual>(workset);
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
    workset.mp_x = Teuchos::rcpFromRef(mp_x);
    if (mp_xdot != NULL) workset.transientTerms = true;

    // FillType template argument used to specialize Sacado
    dfm->evaluateFields<PHAL::AlbanyTraits::MPResidual>(workset);

  }
}

void
Albany::Application::
computeGlobalMPJacobian(
  const double alpha, 
  const double beta,
  const double current_time,
  const Stokhos::ProductEpetraVector* mp_xdot,
  const Stokhos::ProductEpetraVector& mp_x,
  const Teuchos::Array<ParamVec>& p,
  const Teuchos::Array<int>& mp_p_index,
  const Teuchos::Array< Teuchos::Array<MPType> >& mp_p_vals,
  Stokhos::ProductEpetraVector* mp_f,
  Stokhos::ProductContainer<Epetra_CrsMatrix>& mp_jac)
{
  postRegSetup("MPJacobian");

  TimeMonitor Timer(*timers[8]); //start timer

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
    TimeMonitor Timer(*timers[10]); //start timer
*out << " Calling moveMesh with params: " << std::setprecision(8);
for (unsigned int i=0; i<shapeParams.size(); i++) *out << shapeParams[i] << "  ";
*out << endl;
    meshMover->moveMesh(shapeParams, morphFromInit);
    coords = disc->getCoords();
    shapeParamsHaveBeenReset = false;
  }
#endif
//  adapter->adaptit();

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
    workset.mp_f         = mp_overlapped_ff;

    workset.mp_Jac       = mp_overlapped_jac;
    loadWorksetJacobianInfo(workset, alpha, beta);
    workset.current_time = current_time; //timeMgr.getCurrentTime();
    //    workset.delta_time = timeMgr.getDeltaTime();
    if (mp_xdot != NULL) workset.transientTerms = true;

    for (int ws=0; ws < numWorksets; ws++) {
      loadWorksetBucketInfo<PHAL::AlbanyTraits::MPJacobian>(workset, ws);

      // FillType template argument used to specialize Sacado
      fm[wsPhysIndex[ws]]->evaluateFields<PHAL::AlbanyTraits::MPJacobian>(workset);
      if (nfm!=Teuchos::null)
        nfm[wsPhysIndex[ws]]->evaluateFields<PHAL::AlbanyTraits::MPJacobian>(workset);
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
    workset.mp_x = Teuchos::rcpFromRef(mp_x);;
    if (mp_xdot != NULL) workset.transientTerms = true;

    loadWorksetNodesetInfo(workset);

    // FillType template argument used to specialize Sacado
    dfm->evaluateFields<PHAL::AlbanyTraits::MPJacobian>(workset);
  } 
}

void
Albany::Application::
computeGlobalMPTangent(
  const double alpha, 
  const double beta, 
  const double current_time,
  bool sum_derivs,
  const Stokhos::ProductEpetraVector* mp_xdot,
  const Stokhos::ProductEpetraVector& mp_x,
  const Teuchos::Array<ParamVec>& par,
  const Teuchos::Array<int>& mp_p_index,
  const Teuchos::Array< Teuchos::Array<MPType> >& mp_p_vals,
  ParamVec* deriv_par,
  const Epetra_MultiVector* Vx,
  const Epetra_MultiVector* Vxdot,
  const Epetra_MultiVector* Vp,
  Stokhos::ProductEpetraVector* mp_f,
  Stokhos::ProductEpetraMultiVector* mp_JVx,
  Stokhos::ProductEpetraMultiVector* mp_fVp)
{
  postRegSetup("MPTangent");

  TimeMonitor Timer(*timers[9]); //start timer

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
    overlapped_Vxdot = 
      rcp(new Epetra_MultiVector(*(disc->getOverlapMap()), 
				 Vxdot->NumVectors()));
    overlapped_Vxdot->Import(*Vxdot, *importer, Insert);
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
    workset.Vx = overlapped_Vx;
    workset.Vxdot = overlapped_Vxdot;
    workset.Vp = vp;

    workset.mp_f         = mp_overlapped_ff;
    workset.mp_JV        = mp_overlapped_JVx;
    workset.mp_fp        = mp_overlapped_fVp;
    workset.j_coeff      = beta;
    workset.m_coeff      = alpha;

    workset.num_cols_x = num_cols_x;
    workset.num_cols_p = num_cols_p;
    workset.param_offset = param_offset;

    workset.current_time = current_time; //timeMgr.getCurrentTime();
    //    workset.delta_time = timeMgr.getDeltaTime();
    if (mp_xdot != NULL) workset.transientTerms = true;

    for (int ws=0; ws < numWorksets; ws++) {
      loadWorksetBucketInfo<PHAL::AlbanyTraits::MPTangent>(workset, ws);

      // FillType template argument used to specialize Sacado
      fm[wsPhysIndex[ws]]->evaluateFields<PHAL::AlbanyTraits::MPTangent>(workset);
      if (nfm!=Teuchos::null)
        nfm[wsPhysIndex[ws]]->evaluateFields<PHAL::AlbanyTraits::MPTangent>(workset);
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
    workset.mp_x = Teuchos::rcpFromRef(mp_x);
    workset.Vx = rcp(Vx,false);
    if (mp_xdot != NULL) workset.transientTerms = true;

    loadWorksetNodesetInfo(workset);

    // FillType template argument used to specialize Sacado
    dfm->evaluateFields<PHAL::AlbanyTraits::MPTangent>(workset);
  }

}

void
Albany::Application::
evaluateMPResponse(
  int response_index,
  const double curr_time,
  const Stokhos::ProductEpetraVector* mp_xdot,
  const Stokhos::ProductEpetraVector& mp_x,
  const Teuchos::Array<ParamVec>& p,
  const Teuchos::Array<int>& mp_p_index,
  const Teuchos::Array< Teuchos::Array<MPType> >& mp_p_vals,
  Stokhos::ProductEpetraVector& mp_g)
{
  TEUCHOS_FUNC_TIME_MONITOR("Albany::Application::evaluateMPResponses");
  
  responses[response_index]->evaluateMPResponse(
    curr_time, mp_xdot, mp_x, p, mp_p_index, mp_p_vals, mp_g);
}

void
Albany::Application::
evaluateMPResponseTangent(
  int response_index,
  const double alpha, 
  const double beta, 
  const double current_time,
  bool sum_derivs,
  const Stokhos::ProductEpetraVector* mp_xdot,
  const Stokhos::ProductEpetraVector& mp_x,
  const Teuchos::Array<ParamVec>& p,
  const Teuchos::Array<int>& mp_p_index,
  const Teuchos::Array< Teuchos::Array<MPType> >& mp_p_vals,
  ParamVec* deriv_p,
  const Epetra_MultiVector* Vx,
  const Epetra_MultiVector* Vxdot,
  const Epetra_MultiVector* Vp,
  Stokhos::ProductEpetraVector* mp_g,
  Stokhos::ProductEpetraMultiVector* mp_JV,
  Stokhos::ProductEpetraMultiVector* mp_gp)
{
  TEUCHOS_FUNC_TIME_MONITOR("Albany::Application::evaluateMPResponseTangent");
  
  responses[response_index]->evaluateMPTangent(
    alpha, beta, current_time, sum_derivs, mp_xdot, mp_x, p, mp_p_index, 
    mp_p_vals, deriv_p, Vx, Vxdot, Vp, mp_g, mp_JV, mp_gp);
}

void
Albany::Application::
evaluateMPResponseDerivative(
  int response_index,
  const double current_time,
  const Stokhos::ProductEpetraVector* mp_xdot,
  const Stokhos::ProductEpetraVector& mp_x,
  const Teuchos::Array<ParamVec>& p,
  const Teuchos::Array<int>& mp_p_index,
  const Teuchos::Array< Teuchos::Array<MPType> >& mp_p_vals,
  ParamVec* deriv_p,
  Stokhos::ProductEpetraVector* mp_g,
  const EpetraExt::ModelEvaluator::MPDerivative& mp_dg_dx,
  const EpetraExt::ModelEvaluator::MPDerivative& mp_dg_dxdot,
  const EpetraExt::ModelEvaluator::MPDerivative& mp_dg_dp)
{
  TEUCHOS_FUNC_TIME_MONITOR("Albany::Application::evaluateMPResponseGradient");
  
  responses[response_index]->evaluateMPDerivative(
    current_time, mp_xdot, mp_x, p, mp_p_index, mp_p_vals, deriv_p,
    mp_g, mp_dg_dx, mp_dg_dxdot, mp_dg_dp);
}

void
Albany::Application::
evaluateStateFieldManager(const double current_time,
			  const Epetra_Vector* xdot,
			  const Epetra_Vector& x)
{
  // visualize state field manager
  if (stateGraphVisDetail>0) {
    bool detail = false; if (stateGraphVisDetail > 1) detail=true;
    *out << "Phalanx writing graphviz file for graph of Residual fill (detail ="
	 << stateGraphVisDetail << ")"<<endl;
    *out << "Process using 'dot -Tpng -O state_phalanx_graph' \n" << endl;
    for (int ps=0; ps < sfm.size(); ps++) {
      std::stringstream pg; pg << "state_phalanx_graph_" << ps;
      sfm[ps]->writeGraphvizFile<PHAL::AlbanyTraits::Residual>(pg.str(),detail,detail);
    }
    stateGraphVisDetail = -1;
  }

  // Scatter x and xdot to the overlapped distrbution
  overlapped_x->Import(x, *importer, Insert);
  if (xdot != NULL) overlapped_xdot->Import(*xdot, *importer, Insert);

  // Set data in Workset struct
  PHAL::Workset workset;
  loadBasicWorksetInfo( workset, overlapped_x, overlapped_xdot, current_time );
  workset.f = overlapped_f;

  // Perform fill via field manager
  for (int ws=0; ws < numWorksets; ws++) {
    loadWorksetBucketInfo<PHAL::AlbanyTraits::Residual>(workset, ws);
    sfm[wsPhysIndex[ws]]->evaluateFields<PHAL::AlbanyTraits::Residual>(workset);
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
  TEUCHOS_TEST_FOR_EXCEPTION(index==-1,  std::logic_error,
			     "Error in GatherCoordinateVector::getValue, \n" <<
			     "   Unrecognized param name: " << name << endl);

  shapeParamsHaveBeenReset = true;

  return shapeParams[index];
}


void Albany::Application::postRegSetup(std::string eval)
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
    for (int ps=0; ps < fm.size(); ps++) 
      fm[ps]->postRegistrationSetupForType<PHAL::AlbanyTraits::Jacobian>(eval);
    if (dfm!=Teuchos::null)
      dfm->postRegistrationSetupForType<PHAL::AlbanyTraits::Jacobian>(eval);
    if (nfm!=Teuchos::null)
      for (int ps=0; ps < nfm.size(); ps++) 
        nfm[ps]->postRegistrationSetupForType<PHAL::AlbanyTraits::Jacobian>(eval);
  }
  else if (eval=="Tangent") {
    for (int ps=0; ps < fm.size(); ps++) 
      fm[ps]->postRegistrationSetupForType<PHAL::AlbanyTraits::Tangent>(eval);
    if (dfm!=Teuchos::null)
      dfm->postRegistrationSetupForType<PHAL::AlbanyTraits::Tangent>(eval);
    if (nfm!=Teuchos::null)
      for (int ps=0; ps < nfm.size(); ps++) 
        nfm[ps]->postRegistrationSetupForType<PHAL::AlbanyTraits::Tangent>(eval);
  }
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
    for (int ps=0; ps < fm.size(); ps++) 
      fm[ps]->postRegistrationSetupForType<PHAL::AlbanyTraits::SGJacobian>(eval);
    if (dfm!=Teuchos::null)
      dfm->postRegistrationSetupForType<PHAL::AlbanyTraits::SGJacobian>(eval);
    if (nfm!=Teuchos::null)
      for (int ps=0; ps < nfm.size(); ps++) 
        nfm[ps]->postRegistrationSetupForType<PHAL::AlbanyTraits::SGJacobian>(eval);
  }
  else if (eval=="SGTangent") {
    for (int ps=0; ps < fm.size(); ps++) 
      fm[ps]->postRegistrationSetupForType<PHAL::AlbanyTraits::SGTangent>(eval);
    if (dfm!=Teuchos::null)
      dfm->postRegistrationSetupForType<PHAL::AlbanyTraits::SGTangent>(eval);
    if (nfm!=Teuchos::null)
      for (int ps=0; ps < nfm.size(); ps++) 
        nfm[ps]->postRegistrationSetupForType<PHAL::AlbanyTraits::SGTangent>(eval);
  }
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
    for (int ps=0; ps < fm.size(); ps++) 
      fm[ps]->postRegistrationSetupForType<PHAL::AlbanyTraits::MPJacobian>(eval);
    if (dfm!=Teuchos::null)
      dfm->postRegistrationSetupForType<PHAL::AlbanyTraits::MPJacobian>(eval);
    if (nfm!=Teuchos::null)
      for (int ps=0; ps < nfm.size(); ps++) 
        nfm[ps]->postRegistrationSetupForType<PHAL::AlbanyTraits::MPJacobian>(eval);
  }
  else if (eval=="MPTangent") {
    for (int ps=0; ps < fm.size(); ps++) 
      fm[ps]->postRegistrationSetupForType<PHAL::AlbanyTraits::MPTangent>(eval);
    if (dfm!=Teuchos::null)
      dfm->postRegistrationSetupForType<PHAL::AlbanyTraits::MPTangent>(eval);
    if (nfm!=Teuchos::null)
      for (int ps=0; ps < nfm.size(); ps++) 
        nfm[ps]->postRegistrationSetupForType<PHAL::AlbanyTraits::MPTangent>(eval);
  }
  else 
    TEUCHOS_TEST_FOR_EXCEPTION(eval!="Known Evaluation Name",  std::logic_error,
			       "Error in setup call \n" << " Unrecognized name: " << eval << endl);


  // Write out Phalanx Graph if requested, on Proc 0, for Resid or Jacobian
  if (phxGraphVisDetail>0) {
    bool detail = false; if (phxGraphVisDetail > 1) detail=true;

    if (eval=="Residual") {
      *out << "Phalanx writing graphviz file for graph of Residual fill (detail ="
           << phxGraphVisDetail << ")"<<endl;
      *out << "Process using 'dot -Tpng -O phalanx_graph' \n" << endl;
      for (int ps=0; ps < fm.size(); ps++) {
        std::stringstream pg; pg << "phalanx_graph_" << ps;
        fm[ps]->writeGraphvizFile<PHAL::AlbanyTraits::Residual>(pg.str(),detail,detail);
      }
      phxGraphVisDetail = -1;
    }
    else if (eval=="Jacobian") {
      *out << "Phalanx writing graphviz file for graph of Jacobian fill (detail ="
           << phxGraphVisDetail << ")"<<endl;
      *out << "Process using 'dot -Tpng -O phalanx_graph' \n" << endl;
      for (int ps=0; ps < fm.size(); ps++) {
        std::stringstream pg; pg << "phalanx_graph_jac_" << ps;
        fm[ps]->writeGraphvizFile<PHAL::AlbanyTraits::Jacobian>(pg.str(),detail,detail);
      }
      phxGraphVisDetail = -2;
    }
  }
}

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

void Albany::Application::defineTimers()
{
  timers.push_back(TimeMonitor::getNewTimer("> Albany Fill: Residual"));
  timers.push_back(TimeMonitor::getNewTimer("> Albany Fill: Jacobian"));
  timers.push_back(TimeMonitor::getNewTimer("> Albany Fill: Precond"));
  timers.push_back(TimeMonitor::getNewTimer("> Albany Fill: Tangent"));
  timers.push_back(TimeMonitor::getNewTimer("> Albany Fill: SGResidual"));
  timers.push_back(TimeMonitor::getNewTimer("> Albany Fill: SGJacobian"));
  timers.push_back(TimeMonitor::getNewTimer("> Albany Fill: SGTangent"));
  timers.push_back(TimeMonitor::getNewTimer("> Albany Fill: MPResidual"));
  timers.push_back(TimeMonitor::getNewTimer("> Albany Fill: MPJacobian"));
  timers.push_back(TimeMonitor::getNewTimer("> Albany Fill: MPTangent"));
  timers.push_back(TimeMonitor::getNewTimer("Albany-Cubit MeshMover"));
}

void Albany::Application::loadBasicWorksetInfo(
       PHAL::Workset& workset, RCP<Epetra_Vector> overlapped_x,
       RCP<Epetra_Vector> overlapped_xdot, double current_time)
{
    workset.x        = overlapped_x;
    workset.xdot     = overlapped_xdot;
    workset.current_time = current_time;
    //workset.delta_time = delta_time;
    if (overlapped_xdot != Teuchos::null) workset.transientTerms = true;
}

void Albany::Application::loadWorksetJacobianInfo(PHAL::Workset& workset,
                                 const double& alpha, const double& beta)
{
    workset.m_coeff      = alpha;
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

void Albany::Application::setupBasicWorksetInfo(
  PHAL::Workset& workset,
  double current_time,
  const Epetra_Vector* xdot, 
  const Epetra_Vector* x,
  const Teuchos::Array<ParamVec>& p
  )
{
  // Scatter x and xdot to the overlapped distrbution
  overlapped_x->Import(*x, *importer, Insert);
  if (xdot != NULL) overlapped_xdot->Import(*xdot, *importer, Insert);

  // Set parameters
  for (int i=0; i<p.size(); i++)
    for (unsigned int j=0; j<p[i].size(); j++)
      p[i][j].family->setRealValueForAllTypes(p[i][j].baseValue);

  workset.x = overlapped_x;
  workset.xdot = overlapped_xdot;
  if (!paramLib->isParameter("Time"))
    workset.current_time = current_time;
  else 
    workset.current_time = 
      paramLib->getRealValue<PHAL::AlbanyTraits::Residual>("Time");
  if (overlapped_xdot != Teuchos::null) workset.transientTerms = true;

  // Create Teuchos::Comm from Epetra_Comm
  const Epetra_Comm& comm = x->Map().Comm();
  workset.comm = Albany::createTeuchosCommFromMpiComm(
                  Albany::getMpiCommFromEpetraComm(comm));

  workset.x_importer = importer;
}

void Albany::Application::setupBasicWorksetInfo(
  PHAL::Workset& workset,
  double current_time,
  const Stokhos::EpetraVectorOrthogPoly* sg_xdot, 
  const Stokhos::EpetraVectorOrthogPoly* sg_x,
  const Teuchos::Array<ParamVec>& p,
  const Teuchos::Array<int>& sg_p_index,
  const Teuchos::Array< Teuchos::Array<SGType> >& sg_p_vals)
{
  // Scatter x and xdot to the overlapped distrbution
  for (int i=0; i<sg_x->size(); i++) {
    (*sg_overlapped_x)[i].Import((*sg_x)[i], *importer, Insert);
    if (sg_xdot != NULL) (*sg_overlapped_xdot)[i].Import((*sg_xdot)[i], *importer, Insert);
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
  if (!paramLib->isParameter("Time"))
    workset.current_time = current_time;
  else 
    workset.current_time = 
      paramLib->getRealValue<PHAL::AlbanyTraits::Residual>("Time");
  if (sg_xdot != NULL) workset.transientTerms = true;

  // Create Teuchos::Comm from Epetra_Comm
  const Epetra_Comm& comm = sg_x->coefficientMap()->Comm();
  workset.comm = Albany::createTeuchosCommFromMpiComm(
                  Albany::getMpiCommFromEpetraComm(comm));

  workset.x_importer = importer;
}

void Albany::Application::setupBasicWorksetInfo(
  PHAL::Workset& workset,
  double current_time,
  const Stokhos::ProductEpetraVector* mp_xdot, 
  const Stokhos::ProductEpetraVector* mp_x,
  const Teuchos::Array<ParamVec>& p,
  const Teuchos::Array<int>& mp_p_index,
  const Teuchos::Array< Teuchos::Array<MPType> >& mp_p_vals)
{
  // Scatter x and xdot to the overlapped distrbution
  for (int i=0; i<mp_x->size(); i++) {
    (*mp_overlapped_x)[i].Import((*mp_x)[i], *importer, Insert);
    if (mp_xdot != NULL) (*mp_overlapped_xdot)[i].Import((*mp_xdot)[i], *importer, Insert);
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
  if (!paramLib->isParameter("Time"))
    workset.current_time = current_time;
  else 
    workset.current_time = 
      paramLib->getRealValue<PHAL::AlbanyTraits::Residual>("Time");
  if (mp_xdot != NULL) workset.transientTerms = true;

  // Create Teuchos::Comm from Epetra_Comm
  const Epetra_Comm& comm = mp_x->coefficientMap()->Comm();
  workset.comm = Albany::createTeuchosCommFromMpiComm(
                  Albany::getMpiCommFromEpetraComm(comm));

  workset.x_importer = importer;
}

void Albany::Application::setupTangentWorksetInfo(
  PHAL::Workset& workset, 
  double current_time,
  bool sum_derivs,
  const Epetra_Vector* xdot, 
  const Epetra_Vector* x,
  const Teuchos::Array<ParamVec>& p,
  ParamVec* deriv_p,
  const Epetra_MultiVector* Vxdot,
  const Epetra_MultiVector* Vx,
  const Epetra_MultiVector* Vp)
{
  setupBasicWorksetInfo(workset, current_time, xdot, x, p);

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
    overlapped_Vxdot = 
      rcp(new Epetra_MultiVector(*(disc->getOverlapMap()), 
				 Vxdot->NumVectors()));
    overlapped_Vxdot->Import(*Vxdot, *importer, Insert);
  }

  RCP<const Epetra_MultiVector > vp = rcp(Vp, false);
  RCP<ParamVec> params = rcp(deriv_p, false);

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

  workset.params = params;
  workset.Vx = overlapped_Vx;
  workset.Vxdot = overlapped_Vxdot;
  workset.Vp = vp;
  workset.num_cols_x = num_cols_x;
  workset.num_cols_p = num_cols_p;
  workset.param_offset = param_offset;
}

void Albany::Application::setupTangentWorksetInfo(
  PHAL::Workset& workset, 
  double current_time,
  bool sum_derivs,
  const Stokhos::EpetraVectorOrthogPoly* sg_xdot, 
  const Stokhos::EpetraVectorOrthogPoly* sg_x,
  const Teuchos::Array<ParamVec>& p,
  ParamVec* deriv_p,
  const Teuchos::Array<int>& sg_p_index,
  const Teuchos::Array< Teuchos::Array<SGType> >& sg_p_vals,
  const Epetra_MultiVector* Vxdot,
  const Epetra_MultiVector* Vx,
  const Epetra_MultiVector* Vp)
{
  setupBasicWorksetInfo(workset, current_time, sg_xdot, sg_x, p, 
			sg_p_index, sg_p_vals);

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
    overlapped_Vxdot = 
      rcp(new Epetra_MultiVector(*(disc->getOverlapMap()), 
				 Vxdot->NumVectors()));
    overlapped_Vxdot->Import(*Vxdot, *importer, Insert);
  }

  RCP<const Epetra_MultiVector > vp = rcp(Vp, false);
  RCP<ParamVec> params = rcp(deriv_p, false);

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
  workset.Vp = vp;
  workset.num_cols_x = num_cols_x;
  workset.num_cols_p = num_cols_p;
  workset.param_offset = param_offset;
}

void Albany::Application::setupTangentWorksetInfo(
  PHAL::Workset& workset, 
  double current_time,
  bool sum_derivs,
  const Stokhos::ProductEpetraVector* mp_xdot, 
  const Stokhos::ProductEpetraVector* mp_x,
  const Teuchos::Array<ParamVec>& p,
  ParamVec* deriv_p,
  const Teuchos::Array<int>& mp_p_index,
  const Teuchos::Array< Teuchos::Array<MPType> >& mp_p_vals,
  const Epetra_MultiVector* Vxdot,
  const Epetra_MultiVector* Vx,
  const Epetra_MultiVector* Vp)
{
  setupBasicWorksetInfo(workset, current_time, mp_xdot, mp_x, p, 
			mp_p_index, mp_p_vals);

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
    overlapped_Vxdot = 
      rcp(new Epetra_MultiVector(*(disc->getOverlapMap()), 
				 Vxdot->NumVectors()));
    overlapped_Vxdot->Import(*Vxdot, *importer, Insert);
  }

  RCP<const Epetra_MultiVector > vp = rcp(Vp, false);
  RCP<ParamVec> params = rcp(deriv_p, false);

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
  workset.Vp = vp;
  workset.num_cols_x = num_cols_x;
  workset.num_cols_p = num_cols_p;
  workset.param_offset = param_offset;
}
