//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "QCAD_CoupledPoissonSchrodinger.hpp"
#include "QCAD_CoupledPSJacobian.hpp"
#include "Piro_Epetra_LOCASolver.hpp"
#include "Stokhos.hpp"
#include "Stokhos_Epetra.hpp"
#include "Sacado_PCE_OrthogPoly.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#include "Teuchos_TestForException.hpp"

//needed?
//#include "Teuchos_RCP.hpp"
//#include "Teuchos_VerboseObject.hpp"
//#include "Teuchos_FancyOStream.hpp"

#include "Teuchos_ParameterList.hpp"

#include "Albany_ModelFactory.hpp"
#include "Albany_Utils.hpp"
#include "Albany_SolverFactory.hpp"
#include "Albany_StateInfoStruct.hpp"
#include "Albany_EigendataInfoStruct.hpp"

#include "QCAD_CoupledPSJacobian.hpp"
  

QCAD::CoupledPoissonSchrodinger::
CoupledPoissonSchrodinger(const Teuchos::RCP<Teuchos::ParameterList>& appParams,
			  const Teuchos::RCP<const Epetra_Comm>& comm,
			  const Teuchos::RCP<const Epetra_Vector>& initial_guess)
{
  using std::string;

  const Albany_MPI_Comm& mcomm = Albany::getMpiCommFromEpetraComm(*comm);
  Teuchos::RCP<Teuchos::Comm<int> > tcomm = Albany::createTeuchosCommFromMpiComm(mcomm);

  // Get sub-problem input xml files from problem parameters
  Teuchos::ParameterList& problemParams = appParams->sublist("Problem");

  // Validate Problem parameters against list for this specific problem
  //problemParams.validateParameters(getValidProblemParameters(),0); //TODO: copy over getValidProblemParams?

  string poissonInputFile = problemParams.get<string>("Poisson Input Filename");
  string schrodingerInputFile = problemParams.get<string>("Schrodinger Input Filename");
  nEigenvals = problemParams.get<int>("Number of Eigenvalues");

  //TODO: need to add meshmover initialization, as in Albany::Application constructor??

  //TODO: need to carry over any more of initialization from Albany::Application constructor -- I don't think so
  //       since this will be done by individual Application constructors called below.

  ////////////////////////////////////////////////////////////////////////////////////////////////////////
  //! Create Poisson application object (similar logic in Albany::SolverFactory::createAlbanyAppAndModel)
  ////////////////////////////////////////////////////////////////////////////////////////////////////////

  // Set up application parameters: read and broadcast XML file, and set defaults
  Teuchos::RCP<Teuchos::ParameterList> poissonAppParams = Teuchos::createParameterList("Albany Parameters");
  Teuchos::updateParametersFromXmlFileAndBroadcast(poissonInputFile, poissonAppParams.ptr(), *tcomm);

  //Techos::RCP<ParameterList> defaultSolverParams = Teuchos::rcp(new ParameterList());
  //setSolverParamDefaults(defaultSolverParams.get(), tcomm->getRank());
  //poissonAppParams->setParametersNotAlreadySet(*defaultSolverParams);

  poissonAppParams->validateParametersAndSetDefaults(*getValidAppParameters(),0);
  poissonApp = Teuchos::rcp(new Albany::Application(comm, poissonAppParams, initial_guess));

  // Validate Response list: may move inside individual Problem class
  const Teuchos::RCP<Teuchos::ParameterList> poissonProblemParams = Teuchos::sublist(poissonAppParams, "Problem");
  //poissonProblemParams->sublist("Response Functions").
  //  validateParameters(*getValidResponseParameters(),0);  //TODO: copy getValieResponseParameters too...

  //Note: Solverfactory determines which Piro solver to use here, but I think this is uncessary -- we ignore all solver directives in the 
  // poisson and schrodinger files and use the ones in the coupled-PS input file (i.e. under appParams)

  // Create model evaluator
  Albany::ModelFactory poissonModelFactory(poissonAppParams, poissonApp);
  poissonModel = poissonModelFactory.create();


  ////////////////////////////////////////////////////////////////////////////////////////////////////////
  //! Create Schrodinger application object (similar logic in Albany::SolverFactory::createAlbanyAppAndModel)
  ////////////////////////////////////////////////////////////////////////////////////////////////////////

  // Set up application parameters: read and broadcast XML file, and set defaults
  Teuchos::RCP<Teuchos::ParameterList> schrodingerAppParams = Teuchos::createParameterList("Albany Parameters");
  Teuchos::updateParametersFromXmlFileAndBroadcast(schrodingerInputFile, schrodingerAppParams.ptr(), *tcomm);

  //Techos::RCP<ParameterList> defaultSolverParams = Teuchos::rcp(new ParameterList());
  //setSolverParamDefaults(defaultSolverParams.get(), tcomm->getRank());
  //schrodingerAppParams->setParametersNotAlreadySet(*defaultSolverParams);

  schrodingerAppParams->validateParametersAndSetDefaults(*getValidAppParameters(),0);
  schrodingerApp = Teuchos::rcp(new Albany::Application(comm, schrodingerAppParams, Teuchos::null)); //initial_guess = null for schrodinger apps -- perhaps we need to break up initial_guess into parts... (TODO)

  // Validate Response list: may move inside individual Problem class
  const Teuchos::RCP<Teuchos::ParameterList> schrodingerProblemParams = Teuchos::sublist(schrodingerAppParams, "Problem");
  //schrodingerProblemParams->sublist("Response Functions").
  //  validateParameters(*getValidResponseParameters(),0);  //TODO: copy getValidResponseParameters?

  //Note: Solverfactory determines which Piro solver to use here, but I think this is uncessary -- we ignore all solver directives in the 
  // schrodinger and schrodinger files and use the ones in the coupled-PS input file (i.e. under appParams)

  // Create model evaluator
  Albany::ModelFactory schrodingerModelFactory(schrodingerAppParams, schrodingerApp);
  schrodingerModel = schrodingerModelFactory.create();

  //Save the discretization's maps for convenience (should be the same for Poisson and Schrodinger apps)
  disc_map = poissonApp->getMap();
  disc_overlap_map =  poissonApp->getStateMgr().getDiscretization()->getOverlapMap();
  
  //Create map for the entire coupled S-P application from the maps from the individual Poisson and Schrodinger applications
  //  We need to create a map which is the product of 1 disc_map (for P), N disc_maps (for S's), +N extra (for norm. eqns)
  //  in such a way that the elements for each disc_map are contiguous in index space (so that we can easily get Epetra vector views
  //  to them separately)

  int myRank = comm->MyPID();
  int nProcs = comm->NumProc();
  int nScalarEqns = nEigenvals;  // number of "extra" scalar equations, one per eigenvalue
  int nExtra = nScalarEqns % nProcs;

  //int my_nScalar = nScalarEqns / nProcs + (myRank < nExtra) ? 1 : 0;
  int my_nScalar = (nScalarEqns / nProcs) + ((myRank < nExtra) ? 1 : 0);
  std::cout << "INitial my_nScalar = " << nScalarEqns << "/" << nProcs << " = " << my_nScalar << " (" << myRank << " <> " << nExtra << ")" << std::endl;
  int my_scalar_offset = myRank * (nScalarEqns / nProcs) + (myRank < nExtra) ? myRank : nExtra;
  int my_nElements = disc_map->NumMyElements() * (1 + nEigenvals) + my_nScalar;
  std::vector<int> my_global_elements(my_nElements);  //global element indices for this processor

  int disc_nGlobalElements = disc_map->NumGlobalElements();
  int disc_nElements = disc_map->NumMyElements();
  std::vector<int> disc_global_elements(disc_nElements);
  disc_map->MyGlobalElements(&disc_global_elements[0]);
  
  for(int k=0; k<(1+nEigenvals); k++) {
    for(int l=0; l<disc_nElements; l++) {
      my_global_elements[k*disc_nElements + l] = k*disc_nGlobalElements + disc_global_elements[l];
    }
  }

  for(int l=0; l < my_nScalar; l++) {
    my_global_elements[(1+nEigenvals)*disc_nElements + l] = (1+nEigenvals)*disc_nGlobalElements + my_scalar_offset + l;
  }
  
  int global_nElements = (1+nEigenvals)*disc_nGlobalElements + nScalarEqns;
  std::cout << "Global Elements = " << global_nElements << ", nScalar = " << nScalarEqns << std::endl;
  std::cout << "My Elements = " << my_nElements << ", nScalar = " << my_nScalar << " (" << nProcs << " procs)" << std::endl;
  combined_SP_map = Teuchos::rcp(new Epetra_Map(global_nElements, my_nElements, &my_global_elements[0], 0, *comm));


  // Parameter vectors:  Parameter vectors of coupled PS model evaluator are just the parameter vectors
  //   of the Poisson then Schrodinger model evaluators (in order).

  //Get the number of parameter vectors of the Poisson model evaluator (is there a better way to do this??? - yes: get InArgs)
  num_poisson_param_vecs = 0;
  try {
    while(true) { //find highest available index by waiting for exception
      poissonModel->get_p_map(num_poisson_param_vecs);
      num_poisson_param_vecs++;
    }
  }
  catch(...) { }

  //Get the number of parameter vectors of the Schrodginer model evaluator (is there a better way to do this???)
  num_schrodinger_param_vecs = 0;
  try {
    while(true) { //find highest available index by waiting for exception
      schrodingerModel->get_p_map(num_schrodinger_param_vecs);
      num_schrodinger_param_vecs++;
    }
  }
  catch(...) { }

  num_param_vecs = num_poisson_param_vecs + num_schrodinger_param_vecs;

  // Create sacado parameter vectors of appropriate size for use in evalModel
  poisson_sacado_param_vec.resize(num_poisson_param_vecs);
  schrodinger_sacado_param_vec.resize(num_schrodinger_param_vecs);

  // Response vectors:  Response vectors of coupled PS model evaluator are just the response vectors
  //   of the Poisson then Schrodinger model evaluators (in order).
  num_response_vecs = poissonApp->getNumResponses() + schrodingerApp->getNumResponses();



  /*OLD:  // Create Epetra map for parameter vector (only one since num_p always == 1)
  epetra_param_map = Teuchos::rcp(new Epetra_LocalMap(static_cast<GlobalIndex>(nParameters), 0, *comm));

  // Create Epetra map for (first) response vector
  epetra_response_map = Teuchos::rcp(new Epetra_LocalMap(static_cast<GlobalIndex>(nResponseDoubles), 0, *comm));
     //ANDY: if (nResponseDoubles > 0) needed ??

  // Create Epetra map for solution vector (second response vector).  Assume 
  //  each subSolver has the same map, so just get the first one.
  Teuchos::RCP<const Epetra_Map> sub_x_map = (subSolvers.begin()->second).app->getMap();
  TEUCHOS_TEST_FOR_EXCEPT( sub_x_map == Teuchos::null );
  epetra_x_map = Teuchos::rcp(new Epetra_Map( *sub_x_map ));
  */


  myComm = comm;
}

QCAD::CoupledPoissonSchrodinger::~CoupledPoissonSchrodinger()
{
}


Teuchos::RCP<const Epetra_Map> QCAD::CoupledPoissonSchrodinger::get_x_map() const
{
  return combined_SP_map;
}

Teuchos::RCP<const Epetra_Map> QCAD::CoupledPoissonSchrodinger::get_f_map() const
{
  return combined_SP_map;
}

Teuchos::RCP<const Epetra_Map> QCAD::CoupledPoissonSchrodinger::get_p_map(int l) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(l >= num_param_vecs || l < 0, Teuchos::Exceptions::InvalidParameter,
                     std::endl <<
                     "Error in QCAD::CoupledPoissonSchrodinger::get_p_map():  " <<
                     "Invalid parameter index l = " << l << std::endl);
  if(l < num_poisson_param_vecs)
    return poissonModel->get_p_map(l);
  else
    return schrodingerModel->get_p_map(l - num_poisson_param_vecs);
}

Teuchos::RCP<const Epetra_Map> QCAD::CoupledPoissonSchrodinger::get_g_map(int j) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(j > num_response_vecs || j < 0, Teuchos::Exceptions::InvalidParameter,
                     std::endl <<
                     "Error in QCAD::CoupledPoissonSchrodinger::get_g_map():  " <<
                     "Invalid response index j = " << j << std::endl);
  
  if(j < poissonApp->getNumResponses())
    return poissonModel->get_g_map(j);
  else
    return schrodingerModel->get_g_map(j - poissonApp->getNumResponses());
}

Teuchos::RCP<const Teuchos::Array<std::string> > QCAD::CoupledPoissonSchrodinger::get_p_names(int l) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(l >= num_param_vecs || l < 0, 
		     Teuchos::Exceptions::InvalidParameter,
                     std::endl << 
                     "Error!  Albany::ModelEvaluator::get_p_names():  " <<
                     "Invalid parameter index l = " << l << std::endl);
  if(l < num_poisson_param_vecs)
    return poissonModel->get_p_names(l);
  else
    return schrodingerModel->get_p_names(l - num_poisson_param_vecs);
}


Teuchos::RCP<const Epetra_Vector> QCAD::CoupledPoissonSchrodinger::get_x_init() const
{
  //Put together x_init's from Poisson and Schrodinger for now (but does this make sense for eigenvectors?) -- TODO: discuss
  Teuchos::RCP<const Epetra_Vector> poisson_x_init = poissonModel->get_x_init(); // should have disc_map
  Teuchos::RCP<const Epetra_Vector> schrodinger_x_init = schrodingerModel->get_x_init(); // should have disc_map
  
  int disc_nMyElements = disc_map->NumMyElements();

  double *x_init_data;
  Teuchos::RCP<Epetra_Vector> x_init = Teuchos::rcp(new Epetra_Vector(*combined_SP_map));
  Teuchos::RCP<Epetra_Vector> x_init_poisson;
  Teuchos::RCP<Epetra_MultiVector> x_init_schrodinger;

  if(x_init->ExtractView(&x_init_data) != 0)
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
			       "Error!  QCAD::CoupledPoissonSchrodinger -- cannot extract x_init vector views");
  x_init_poisson = Teuchos::rcp(new Epetra_Vector(::View, *disc_map, &x_init_data[0]));
  x_init_schrodinger = Teuchos::rcp(new Epetra_MultiVector(::View, *disc_map, &x_init_data[disc_nMyElements], disc_nMyElements, nEigenvals));

  std::vector<int> localInds( poisson_x_init->MyLength() );
  for(int i=0; i < poisson_x_init->MyLength(); i++) localInds[i] = i;

  x_init_poisson->ReplaceMyValues( poisson_x_init->MyLength(), &(*poisson_x_init)[0], &localInds[0] );
  for(int k=0; k < nEigenvals; k++)
    (*x_init_schrodinger)(k)->ReplaceMyValues( schrodinger_x_init->MyLength(), &(*schrodinger_x_init)[0], &localInds[0] ); //same localInds are the same
  
  return x_init;
}

Teuchos::RCP<const Epetra_Vector> QCAD::CoupledPoissonSchrodinger::get_x_dot_init() const
{
  //Put together x_dot_init's from Poisson and Schrodinger for now (but does this make sense for eigenvectors?) -- TODO: discuss
  Teuchos::RCP<const Epetra_Vector> poisson_x_dot_init = poissonModel->get_x_dot_init(); // should have disc_map
  Teuchos::RCP<const Epetra_Vector> schrodinger_x_dot_init = schrodingerModel->get_x_dot_init(); // should have disc_map
  
  int disc_nMyElements = disc_map->NumMyElements();

  double *x_dot_init_data;
  Teuchos::RCP<Epetra_Vector> x_dot_init = Teuchos::rcp(new Epetra_Vector(*combined_SP_map));
  Teuchos::RCP<Epetra_Vector> x_dot_init_poisson;
  Teuchos::RCP<Epetra_MultiVector> x_dot_init_schrodinger;

  if(x_dot_init->ExtractView(&x_dot_init_data) != 0)
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
			       "Error!  QCAD::CoupledPoissonSchrodinger -- cannot extract x_dot_init vector views");
  x_dot_init_poisson = Teuchos::rcp(new Epetra_Vector(::View, *disc_map, &x_dot_init_data[0]));
  x_dot_init_schrodinger = Teuchos::rcp(new Epetra_MultiVector(::View, *disc_map, &x_dot_init_data[disc_nMyElements], disc_nMyElements, nEigenvals));

  std::vector<int> localInds( poisson_x_dot_init->MyLength() );
  for(int i=0; i < poisson_x_dot_init->MyLength(); i++) localInds[i] = i;

  x_dot_init_poisson->ReplaceMyValues( poisson_x_dot_init->MyLength(), &(*poisson_x_dot_init)[0], &localInds[0] );
  for(int k=0; k < nEigenvals; k++)
    (*x_dot_init_schrodinger)(k)->ReplaceMyValues( schrodinger_x_dot_init->MyLength(), &(*schrodinger_x_dot_init)[0], &localInds[0] ); //same localInds are the same
  
  return x_dot_init;
}


Teuchos::RCP<const Epetra_Vector> QCAD::CoupledPoissonSchrodinger::get_p_init(int l) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(l >= num_param_vecs || l < 0, Teuchos::Exceptions::InvalidParameter,
                     std::endl <<
                     "Error in QCAD::CoupledPoissonSchrodinger::get_p_init():  " <<
                     "Invalid parameter index l = " << l << std::endl);

  if(l < num_poisson_param_vecs)
    return poissonModel->get_p_init(l);
  else
    return schrodingerModel->get_p_init(l - num_poisson_param_vecs);
}


Teuchos::RCP<Epetra_Operator>
QCAD::CoupledPoissonSchrodinger::create_W() const
{
  return Teuchos::rcp( new QCAD::CoupledPSJacobian(nEigenvals, disc_map, combined_SP_map, myComm) );
}

Teuchos::RCP<EpetraExt::ModelEvaluator::Preconditioner>
QCAD::CoupledPoissonSchrodinger::create_WPrec() const
{
  //TODO LATER -- create jacobian preconditioner
  Teuchos::RCP<Epetra_Operator> precOp; // = app->getPreconditioner();

  // bool is answer to: "Prec is already inverted?"
  return Teuchos::rcp(new EpetraExt::ModelEvaluator::Preconditioner(precOp,true));
}

Teuchos::RCP<Epetra_Operator>
QCAD::CoupledPoissonSchrodinger::create_DgDx_op(int j) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(
    j >= num_response_vecs || j < 0, 
    Teuchos::Exceptions::InvalidParameter,
    std::endl << 
    "Error!  Albany::ModelEvaluator::create_DgDx_op():  " << 
    "Invalid response index j = " << j << std::endl);
  
  if(j < poissonApp->getNumResponses())
    return poissonApp->getResponse(j)->createGradientOp();
  else
    return schrodingerApp->getResponse(j - poissonApp->getNumResponses())->createGradientOp();
}

Teuchos::RCP<Epetra_Operator>
QCAD::CoupledPoissonSchrodinger::create_DgDx_dot_op(int j) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(
    j >= num_response_vecs || j < 0, 
    Teuchos::Exceptions::InvalidParameter,
    std::endl << 
    "Error!  Albany::ModelEvaluator::create_DgDx_dot_op():  " << 
    "Invalid response index j = " << j << std::endl);
  
  if(j < poissonApp->getNumResponses())
    return poissonApp->getResponse(j)->createGradientOp();
  else
    return schrodingerApp->getResponse(j - poissonApp->getNumResponses())->createGradientOp();
}


EpetraExt::ModelEvaluator::InArgs QCAD::CoupledPoissonSchrodinger::createInArgs() const
{
  InArgsSetup inArgs;
  inArgs.setModelEvalDescription("QCAD Coupled Poisson-Schrodinger Model Evaluator");

  inArgs.setSupports(IN_ARG_t,true);
  inArgs.setSupports(IN_ARG_x,true);
  inArgs.setSupports(IN_ARG_x_dot,true);
  inArgs.setSupports(IN_ARG_alpha,true);
  inArgs.setSupports(IN_ARG_beta,true);
  inArgs.set_Np(num_param_vecs);

  // Note: no SG support yet...

  return inArgs;
}

EpetraExt::ModelEvaluator::OutArgs QCAD::CoupledPoissonSchrodinger::createOutArgs() const
{
  OutArgsSetup outArgs;
  outArgs.setModelEvalDescription("QCAD Coupled Poisson-Schrodinger Model Evaluator");

  int n_g = num_response_vecs;
  bool bScalarResponse;

  // Deterministic
  outArgs.setSupports(OUT_ARG_f,true);
  outArgs.setSupports(OUT_ARG_W,true);
  outArgs.set_W_properties(
    DerivativeProperties(DERIV_LINEARITY_UNKNOWN, DERIV_RANK_FULL, true));
  outArgs.setSupports(OUT_ARG_WPrec, true); //TODO: set to false initially?
  outArgs.set_Np_Ng(num_param_vecs, n_g);

  for (int i=0; i<num_param_vecs; i++)
    outArgs.setSupports(OUT_ARG_DfDp, i, DerivativeSupport(DERIV_MV_BY_COL));
  for (int i=0; i<n_g; i++) {

    if(i < poissonApp->getNumResponses())
      bScalarResponse = poissonApp->getResponse(i)->isScalarResponse();
    else
      bScalarResponse = schrodingerApp->getResponse(i - poissonApp->getNumResponses())->isScalarResponse();

    if(bScalarResponse) {
      outArgs.setSupports(OUT_ARG_DgDx, i,
                          DerivativeSupport(DERIV_TRANS_MV_BY_ROW));
      outArgs.setSupports(OUT_ARG_DgDx_dot, i,
                          DerivativeSupport(DERIV_TRANS_MV_BY_ROW));
    }
    else {
      outArgs.setSupports(OUT_ARG_DgDx, i,
                          DerivativeSupport(DERIV_LINEAR_OP));
      outArgs.setSupports(OUT_ARG_DgDx_dot, i,
                          DerivativeSupport(DERIV_LINEAR_OP));
    }

    for (int j=0; j<num_param_vecs; j++)
      outArgs.setSupports(OUT_ARG_DgDp, i, j,
                          DerivativeSupport(DERIV_MV_BY_COL));
  }

  //Note: no SG support yet...

  return outArgs;
}


void 
QCAD::CoupledPoissonSchrodinger::evalModel(const InArgs& inArgs,
			const OutArgs& outArgs ) const
{
  //?? Teuchos::TimeMonitor Timer(*timer); //start timer

  //
  // Get the input arguments
  //
  Teuchos::RCP<const Epetra_Vector> x = inArgs.get_x();
  Teuchos::RCP<const Epetra_Vector> x_dot;
  double alpha     = 0.0;  // M coeff
  double beta      = 1.0;  // J coeff
  double curr_time = 0.0;
  x_dot = inArgs.get_x_dot();
  if (x_dot != Teuchos::null) {
    alpha = inArgs.get_alpha();
    beta = inArgs.get_beta();
    curr_time  = inArgs.get_t();
  }
  for (int i=0; i<inArgs.Np(); i++) {
    Teuchos::RCP<const Epetra_Vector> p = inArgs.get_p(i);
    if (p != Teuchos::null) {
      if(i < num_poisson_param_vecs) {
	for (unsigned int j=0; j<poisson_sacado_param_vec[i].size(); j++)
	  poisson_sacado_param_vec[i][j].baseValue = (*p)[j];
      }
      else {
	for (unsigned int j=0; j<schrodinger_sacado_param_vec[i-num_poisson_param_vecs].size(); j++)
	  schrodinger_sacado_param_vec[i-num_poisson_param_vecs][j].baseValue = (*p)[j];
      }
    }
  }

  //
  // Get the output arguments
  //
  EpetraExt::ModelEvaluator::Evaluation<Epetra_Vector> f_out = outArgs.get_f();
  Teuchos::RCP<Epetra_Operator> W_out = outArgs.get_W();

  // Get preconditioner operator, if requested
  Teuchos::RCP<Epetra_Operator> WPrec_out;
  if (outArgs.supports(OUT_ARG_WPrec)) WPrec_out = outArgs.get_WPrec();



  //
  // Get views into vectors 'f' and 'x' (and 'xdot'?) to use for separate poisson and schrodinger application object calls
  //
  int disc_nMyElements = disc_map->NumMyElements();

  Teuchos::RCP<Epetra_Vector> x_poisson, xdot_poisson, f_poisson;
  Teuchos::RCP<Epetra_MultiVector> x_schrodinger, xdot_schrodinger, f_schrodinger;

  double *x_data, *f_data;
  if(x->ExtractView(&x_data) != 0 || f_out.get()->ExtractView(&f_data) != 0) 
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
			       "Error!  QCAD::CoupledPoissonSchrodinger -- cannot extract vector views");
  x_poisson = Teuchos::rcp(new Epetra_Vector(::View, *disc_map, &x_data[0]));
  x_schrodinger = Teuchos::rcp(new Epetra_MultiVector(::View, *disc_map, &x_data[disc_nMyElements], disc_nMyElements, nEigenvals));

  f_poisson = Teuchos::rcp(new Epetra_Vector(::View, *disc_map, &f_data[0]));
  f_schrodinger = Teuchos::rcp(new Epetra_MultiVector(::View, *disc_map, &f_data[disc_nMyElements], disc_nMyElements, nEigenvals));

  if (x_dot != Teuchos::null) {  //maybe unnecessary if we hardcode alpha & beta - it seems that the coupled PS model evaluator shouldn't support x_dot ...
    double *xdot_data;
    if(x_dot->ExtractView(&xdot_data) != 0)
      TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
				 "Error!  QCAD::CoupledPoissonSchrodinger -- cannot extract vector views");
    xdot_poisson = Teuchos::rcp(new Epetra_Vector(::View, *disc_map, &xdot_data[0]));
    xdot_schrodinger = Teuchos::rcp(new Epetra_MultiVector(::View, *disc_map, &xdot_data[disc_nMyElements], disc_nMyElements, nEigenvals));
  }

  //Communicate all the eigenvalues to every processor, since all parts of the mesh need them
  int my_nEigenvals = combined_SP_map->NumMyElements() - disc_nMyElements * (1+nEigenvals);
  Epetra_Map dist_eigenval_map(nEigenvals, my_nEigenvals, 0, *myComm);
  Epetra_LocalMap local_eigenval_map(nEigenvals, 0, *myComm);
  Epetra_Import eigenval_importer(local_eigenval_map, dist_eigenval_map);

  Teuchos::RCP<Epetra_Vector> eigenvals_dist = 
    Teuchos::rcp(new Epetra_Vector(::View, dist_eigenval_map, &x_data[(1+nEigenvals)*disc_nMyElements]));
  Teuchos::RCP<Epetra_Vector> eigenvals =  Teuchos::rcp(new Epetra_Vector(local_eigenval_map));

  eigenvals->Import(*eigenvals_dist, eigenval_importer, Insert);
  Teuchos::RCP<std::vector<double> > stdvec_eigenvals = Teuchos::rcp(new std::vector<double>(&(*eigenvals)[0], &(*eigenvals)[0] + nEigenvals));


  // Create an eigendata struct for passing the eigenvectors to the poisson app
  //  -- note that this requires the *overlapped* eigenvectors
  Teuchos::RCP<Albany::EigendataStruct> eigenData = Teuchos::rcp( new Albany::EigendataStruct );

  eigenData->eigenvalueRe = stdvec_eigenvals;
  eigenData->eigenvectorRe = 
    Teuchos::rcp(new Epetra_MultiVector(*disc_overlap_map, nEigenvals));
  eigenData->eigenvectorIm = Teuchos::null; // no imaginary eigenvalue data... 

    // Importer for overlapped data
  Teuchos::RCP<Epetra_Import> overlap_importer =
    Teuchos::rcp(new Epetra_Import(*disc_overlap_map, *disc_map));

    // Overlapped eigenstate vectors
  for(int i=0; i<nEigenvals; i++)
    (*(eigenData->eigenvectorRe))(i)->Import( *((*x_schrodinger)(i)), *overlap_importer, Insert );

    // set eigenvalues / eigenvectors for use in poisson problem:
  poissonApp->getStateMgr().setEigenData(eigenData);


  // Get overlapped version of potential (x_poisson) for passing as auxData to schrodinger app
  Teuchos::RCP<Epetra_MultiVector> overlapped_V = Teuchos::rcp(new Epetra_MultiVector(*disc_overlap_map, 1));
  (*overlapped_V)(0)->Import( *x_poisson, *overlap_importer, Insert );

  // set potential for use in schrodinger problem
  schrodingerApp->getStateMgr().setAuxData(overlapped_V);


  
  //
  // Compute the functions
  //
  bool f_already_computed = false;

  // W 
  if (W_out != Teuchos::null) { 
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,"Error!  Jacobians in QCAD::CoupledPoissonSchrodinger are not implemented yet!!");

    // W = alpha*M + beta*J where M is mass mx and J is jacobian

    //if we need to compute the jacobian, get the jacobians of the poisson and schrodinger
    //  applications (as crs matrices), as well as the mass matrix (from the schrodinger problem,
    //  since it includes xdot - maybe need to fabricate this??) and from these construct a CoupledPoissonSchrodingerJacobian object (an Epetra_Operator)
    
    //TODO - hardcode alpha=0 beta=1 below for jacobian, and change W -> J to make more explicit what we're doing (or what we want to do)

    // Compute poisson Jacobian
    Teuchos::RCP<Epetra_Operator> W_out_poisson = poissonModel->create_W(); //maybe re-use this and not create it every time?
    Teuchos::RCP<Epetra_CrsMatrix> W_out_poisson_crs = Teuchos::rcp_dynamic_cast<Epetra_CrsMatrix>(W_out_poisson, true);

    poissonApp->computeGlobalJacobian(alpha, beta, curr_time, xdot_poisson.get(), *x_poisson, 
				      poisson_sacado_param_vec, f_poisson.get(), *W_out_poisson_crs);

    
    TEUCHOS_TEST_FOR_EXCEPTION(nEigenvals <= 0, Teuchos::Exceptions::InvalidParameter,"Error! The number of eigenvalues must be greater than zero.");
      
    //Compute schrodinger Jacobian using first eigenvector -- independent of eigenvector since Schro. eqn is linear
    Teuchos::RCP<Epetra_Operator> W_out_schrodinger = schrodingerModel->create_W(); //maybe re-use this and not create it every time?
    Teuchos::RCP<Epetra_CrsMatrix> W_out_schrodinger_crs = Teuchos::rcp_dynamic_cast<Epetra_CrsMatrix>(W_out_schrodinger, true);
    schrodingerApp->computeGlobalJacobian(alpha, beta, curr_time, (*xdot_schrodinger)(0), *((*x_schrodinger)(0)), 
					  schrodinger_sacado_param_vec, (*f_schrodinger)(0), *W_out_schrodinger_crs);

    //Compute mass matrix using schrodinger equation -- again independent of eigenvector so can just use 0th
    Teuchos::RCP<Epetra_Operator> M_out_schrodinger = schrodingerModel->create_W(); //maybe re-use this and not create it every time?
    Teuchos::RCP<Epetra_CrsMatrix> M_out_schrodinger_crs = Teuchos::rcp_dynamic_cast<Epetra_CrsMatrix>(M_out_schrodinger, true);
    schrodingerApp->computeGlobalJacobian(1.0, 0.0, curr_time, (*xdot_schrodinger)(0), *((*x_schrodinger)(0)), 
					    schrodinger_sacado_param_vec, (*f_schrodinger)(0), *M_out_schrodinger_crs);

    Teuchos::RCP<QCAD::CoupledPSJacobian> W_out_psj = Teuchos::rcp_dynamic_cast<QCAD::CoupledPSJacobian>(W_out, true);
    W_out_psj->initialize(W_out_poisson_crs, W_out_schrodinger_crs, M_out_schrodinger_crs, eigenvals, x_schrodinger);

    //I don't think the full residual (f) has been computed, since we only needed to evaluate the schrodinger app on the first evec...
    //f_already_computed=true; 
  }


  if (WPrec_out != Teuchos::null) {
    //TODO -- preconditioner
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
			       "Error!  Jacobian preconditioners in QCAD::CoupledPoissonSchrodinger are not implemented yet!!");
    
    //app->computeGlobalJacobian(alpha, beta, curr_time, x_dot.get(), *x, 
    //			       sacado_param_vec, f_out.get(), *Extra_W_crs);
    //f_already_computed=true;
    //
    //app->computeGlobalPreconditioner(Extra_W_crs, WPrec_out);
  }

  // df/dp
  for (int i=0; i<outArgs.Np(); i++) {
    Teuchos::RCP<Epetra_MultiVector> dfdp_out = 
      outArgs.get_DfDp(i).getMultiVector();
    if (dfdp_out != Teuchos::null) {


      // Get views into dfdp_out vectors for poisson and schrodinger parts
      //  Note that df/dp will be zero for parts of f corresponding to an app
      //    different from the one owning the p vector.  E.g. if i==0 corresponds
      //    to p being a poisson parameter vector then df/dp == 0 for all the schrodinger
      //    parts of f.

      int nParamComponents = dfdp_out->NumVectors();
      Teuchos::RCP<Epetra_MultiVector> dfdp_poisson;
      std::vector< Teuchos::RCP<Epetra_MultiVector> > dfdp_schrodinger(nEigenvals);

      double *dfdp_data; int myLDA;
      if(dfdp_out->ExtractView(&dfdp_data, &myLDA) != 0) 
	TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
				   "Error!  QCAD::CoupledPoissonSchrodinger -- cannot extract dgdp vector view");

      dfdp_poisson = Teuchos::rcp(new Epetra_MultiVector(::View, *disc_map, &(dfdp_data[0]), myLDA, nParamComponents));
      for(int k=0; k<nEigenvals; k++)
	dfdp_schrodinger[k] = Teuchos::rcp(new Epetra_MultiVector(::View, *disc_map, &(dfdp_data[(k+1)*disc_nMyElements]), myLDA, nParamComponents));

      // Assemble p_vec
      Teuchos::Array<int> p_indexes = 
        outArgs.get_DfDp(i).getDerivativeMultiVector().getParamIndexes();
      Teuchos::RCP<ParamVec> p_vec;

      Teuchos::Array<ParamVec>& sacado_param_vec = 
	(i < num_poisson_param_vecs) ? poisson_sacado_param_vec : schrodinger_sacado_param_vec;
      int offset = (i < num_poisson_param_vecs) ? 0 : num_poisson_param_vecs;

      if (p_indexes.size() == 0)
        p_vec = Teuchos::rcp(&sacado_param_vec[i-offset],false);
      else {
        p_vec = Teuchos::rcp(new ParamVec);
        for (int j=0; j<p_indexes.size(); j++)
          p_vec->addParam(sacado_param_vec[i-offset][p_indexes[j]].family, 
                          sacado_param_vec[i-offset][p_indexes[j]].baseValue);
      }

      dfdp_out->PutScalar(0.0);

      // Compute full dfdp by computing non-zero parts and leaving zeros in others
      if (i < num_poisson_param_vecs) {
	// "Poisson-owned" param vector, so only poisson part of dfdp vector can be nonzero
	poissonApp->computeGlobalTangent(0.0, 0.0, curr_time, false, xdot_poisson.get(), *x_poisson, 
				  poisson_sacado_param_vec, p_vec.get(),
				  NULL, NULL, NULL, f_poisson.get(), NULL, 
				  dfdp_poisson.get());
      }
      else {
	// "Schrodinger-owned" param vector, so only schrodinger parts of dfdp vector can be nonzero
	for(int k=0; k<nEigenvals; k++)
	  schrodingerApp->computeGlobalTangent(0.0, 0.0, curr_time, false, (*xdot_schrodinger)(k), *((*x_schrodinger)(k)),
				    schrodinger_sacado_param_vec, p_vec.get(),
				    NULL, NULL, NULL, (*f_schrodinger)(k), NULL, 
				    dfdp_schrodinger[k].get());	
      }

      //f_already_computed=true;
    }
  }

  // f
  /*if (app->is_adjoint) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
				   "Error!  QCAD::CoupledPoissonSchrodinger -- adjoints not implemented yet");
    Derivative f_deriv(f_out, DERIV_TRANS_MV_BY_ROW);
    int response_index = 0; // need to add capability for sending this in
    app->evaluateResponseDerivative(response_index, curr_time, x_dot.get(), *x, 
				    sacado_param_vec, NULL, 
				    NULL, f_deriv, Derivative(), Derivative());
  }
  else {  */
    if (f_out != Teuchos::null && !f_already_computed) {
      poissonApp->computeGlobalResidual(curr_time, xdot_poisson.get(), *x_poisson, 
					poisson_sacado_param_vec, *f_poisson);
      for(int i=0; i<nEigenvals; i++)      
	schrodingerApp->computeGlobalResidual(curr_time, (*xdot_schrodinger)(i), *((*x_schrodinger)(i)), 
					      schrodinger_sacado_param_vec, *((*f_schrodinger)(i)) );
    }
  //}

  // Response functions
  for (int i=0; i<outArgs.Ng(); i++) {
    Teuchos::RCP<Epetra_Vector> g_out = outArgs.get_g(i);
   
    bool g_computed = false;

    Derivative dgdx_out = outArgs.get_DgDx(i);
    Derivative dgdxdot_out = outArgs.get_DgDx_dot(i);

    // dg/dx, dg/dxdot
    if (!dgdx_out.isEmpty() || !dgdxdot_out.isEmpty()) {
      if(i < poissonApp->getNumResponses()) {
	poissonApp->evaluateResponseDerivative(i, curr_time, xdot_poisson.get(), *(x_poisson),
                                      poisson_sacado_param_vec, NULL,
                                      g_out.get(), dgdx_out,
                                      dgdxdot_out, Derivative());
      }
      else {
	// take response derivatives using lowest eigenstate only (is there something better??)
	schrodingerApp->evaluateResponseDerivative(i, curr_time, (*xdot_schrodinger)(0), *((*x_schrodinger)(0)),
                                      schrodinger_sacado_param_vec, NULL,
                                      g_out.get(), dgdx_out,
                                      dgdxdot_out, Derivative());
      }
      //g_computed = true;
    }

    // dg/dp
    for (int j=0; j<outArgs.Np(); j++) {
      Teuchos::RCP<Epetra_MultiVector> dgdp_out =
        outArgs.get_DgDp(i,j).getMultiVector();
      if (dgdp_out != Teuchos::null) {
        Teuchos::Array<int> p_indexes =
          outArgs.get_DgDp(i,j).getDerivativeMultiVector().getParamIndexes();

        Teuchos::RCP<ParamVec> p_vec;

	Teuchos::Array<ParamVec>& sacado_param_vec = 
	  (j < num_poisson_param_vecs) ? poisson_sacado_param_vec : schrodinger_sacado_param_vec;
	int offset = (j < num_poisson_param_vecs) ? 0 : num_poisson_param_vecs;

        if (p_indexes.size() == 0)
          p_vec = Teuchos::rcp(&sacado_param_vec[j-offset],false);
        else {
          p_vec = Teuchos::rcp(new ParamVec);
          for (int k=0; k<p_indexes.size(); k++)
            p_vec->addParam(sacado_param_vec[j-offset][p_indexes[k]].family,
                            sacado_param_vec[j-offset][p_indexes[k]].baseValue);
        }

	if(i < poissonApp->getNumResponses() && j < num_poisson_param_vecs) {
	  //both response and param vectors belong to poisson problem
	  poissonApp->evaluateResponseTangent(i, alpha, beta, curr_time, false,
					      xdot_poisson.get(), *x_poisson,
					      poisson_sacado_param_vec, p_vec.get(),
					      NULL, NULL, NULL, g_out.get(), NULL,
					      dgdp_out.get());
	}
	else if(i >= poissonApp->getNumResponses() && j >= num_poisson_param_vecs) {
	  //both response and param vectors belong to schrodinger problem -- evaluate dg/dp using first eigenvector
	  schrodingerApp->evaluateResponseTangent(i, alpha, beta, curr_time, false,
						  (*xdot_schrodinger)(0), *((*x_schrodinger)(0)),
						  schrodinger_sacado_param_vec, p_vec.get(),
						  NULL, NULL, NULL, g_out.get(), NULL,
						  dgdp_out.get());
	}
	else {
	  // response and param vectors belong to different sub-problems (Poisson or Schrodinger)
	  dgdp_out->PutScalar(0.0);
	}

        //g_computed = true;
      }
    }

    if (g_out != Teuchos::null && !g_computed) {
      if(i < poissonApp->getNumResponses()) {
	poissonApp->evaluateResponse(i, curr_time, xdot_poisson.get(), *x_poisson,
				     poisson_sacado_param_vec, *g_out);
      }
      else {
	schrodingerApp->evaluateResponse(i, curr_time, (*xdot_schrodinger)(0), *((*x_schrodinger)(0)), 
					 schrodinger_sacado_param_vec, *g_out);
      }
    }

  }

}

Teuchos::RCP<Albany::Application>
QCAD::CoupledPoissonSchrodinger::getPoissonApp() const
{
  return poissonApp;
}


//Copied from Albany::SolverFactory
Teuchos::RCP<const Teuchos::ParameterList>
QCAD::CoupledPoissonSchrodinger::getValidAppParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL = Teuchos::rcp(new Teuchos::ParameterList("ValidAppParams"));;
  validPL->sublist("Problem",            false, "Problem sublist");
  validPL->sublist("Debug Output",       false, "Debug Output sublist");
  validPL->sublist("Discretization",     false, "Discretization sublist");
  validPL->sublist("Quadrature",         false, "Quadrature sublist");
  validPL->sublist("Regression Results", false, "Regression Results sublist");
  validPL->sublist("VTK",                false, "DEPRECATED  VTK sublist");
  validPL->sublist("Piro",               false, "Piro sublist");
  validPL->sublist("Coupled System",     false, "Coupled system sublist");

  // validPL->set<string>("Jacobian Operator", "Have Jacobian", "Flag to allow Matrix-Free specification in Piro");
  // validPL->set<double>("Matrix-Free Perturbation", 3.0e-7, "delta in matrix-free formula");

  return validPL;
}
