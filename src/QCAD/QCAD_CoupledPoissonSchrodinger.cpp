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
#include "QCAD_CoupledPSPreconditioner.hpp"

//For creating discretiation object without a problem object
#include "Albany_DiscretizationFactory.hpp"
#include "Albany_StateInfoStruct.hpp"
#include "Albany_AbstractFieldContainer.hpp"
#include "Piro_NullSpaceUtils.hpp"

//Ifpack includes
#include "Ifpack_ConfigDefs.h"
#include "Ifpack.h"


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

  numDims = 0;
  string name = problemParams.get<string>("Name");
  if(name == "Poisson Schrodinger 1D") numDims = 1;
  else if(name == "Poisson Schrodinger 2D") numDims = 2;
  else if(name == "Poisson Schrodinger 3D") numDims = 3;
  else TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, std::endl 
			  << "Error!  Invalid problem name " << name << std::endl);

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

  saved_initial_guess = initial_guess;
  poissonAppParams->validateParametersAndSetDefaults(*getValidAppParameters(),0);
  poissonApp = Teuchos::rcp(new Albany::Application(comm, poissonAppParams, Teuchos::null));

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
  schrodingerApp = Teuchos::rcp(new Albany::Application(comm, schrodingerAppParams, Teuchos::null));

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

  //Get the number of parameter vectors of the Poisson model evaluator
  EpetraExt::ModelEvaluator::InArgs poisson_inArgs = poissonModel->createInArgs();
  num_poisson_param_vecs = poisson_inArgs.Np();

  //Get the number of parameter vectors of the Schrodginer model evaluator
  EpetraExt::ModelEvaluator::InArgs schrodinger_inArgs = schrodingerModel->createInArgs();
  num_schrodinger_param_vecs = schrodinger_inArgs.Np();

  num_param_vecs = num_poisson_param_vecs + num_schrodinger_param_vecs;

  // Create sacado parameter vectors of appropriate size for use in evalModel
  poisson_sacado_param_vec.resize(num_poisson_param_vecs);
  schrodinger_sacado_param_vec.resize(num_schrodinger_param_vecs);

  // Response vectors:  Response vectors of coupled PS model evaluator are just the response vectors
  //   of the Poisson then Schrodinger model evaluators (in order).
  num_response_vecs = poissonApp->getNumResponses() + schrodingerApp->getNumResponses();


  // Additional parameters from main list
  temperature = problemParams.get<double>("Temperature");
  length_unit_in_m = problemParams.get<double>("LengthUnitInMeters");


  // Get conduction band offset from reference level (solution to poisson problem), as this
  //   is needed to convert the poisson solution vector to conduction band values expected by the schrodinger problem

    // Material database
  std::string mtrlDbFilename = "materials.xml";
  if(problemParams.isType<std::string>("MaterialDB Filename"))
    mtrlDbFilename = problemParams.get<std::string>("MaterialDB Filename");
  materialDB = Teuchos::rcp(new QCAD::MaterialDatabase(mtrlDbFilename, comm));
  
  std::string refMtrlName = materialDB->getParam<std::string>("Reference Material");
  double refmatChi = materialDB->getMaterialParam<double>(refMtrlName,"Electron Affinity");

  // compute energy reference
  double qPhiRef;
  {
    const double kB = 8.617332e-5;  // eV/K
    std::string category = materialDB->getMaterialParam<std::string>(refMtrlName,"Category");
    if (category == "Semiconductor") 
    {
      // Same qPhiRef needs to be used for the entire structure
      double mdn = materialDB->getMaterialParam<double>(refMtrlName,"Electron DOS Effective Mass");
      double mdp = materialDB->getMaterialParam<double>(refMtrlName,"Hole DOS Effective Mass");
      double Chi = materialDB->getMaterialParam<double>(refMtrlName,"Electron Affinity");
      double Eg0 = materialDB->getMaterialParam<double>(refMtrlName,"Zero Temperature Band Gap");
      double alpha = materialDB->getMaterialParam<double>(refMtrlName,"Band Gap Alpha Coefficient");
      double beta = materialDB->getMaterialParam<double>(refMtrlName,"Band Gap Beta Coefficient");
      
      double Eg = Eg0 - alpha*pow(temperature,2.0)/(beta+temperature); // in [eV]
      double kbT = kB * temperature;      // in [eV]
      double Eic = -Eg/2. + 3./4.*kbT*log(mdp/mdn);  // (Ei-Ec) in [eV]
      qPhiRef = Chi - Eic;  // (Evac-Ei) in [eV] where Evac = vacuum level
    }
    else 
      TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, std::endl 
			  << "Error!  Invalid category " << category << " for reference material !" << std::endl);
  } 

  // NOTE: this works for element blocks of the reference material, but really needs to have refmatChi replaced by the
  //     chi (electron affinity) for the element block that owns each node...
  this->offset_to_CB = qPhiRef - refmatChi; // Conduction Band = offset - poisson_solution


  // Create discretization object solely for producing collected output
  Albany::DiscretizationFactory discFactory(appParams, comm);

  // Get mesh specification object: worksetSize, cell topology, etc
  Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> > meshSpecs =
    discFactory.createMeshSpecs();

  Teuchos::RCP<Albany::StateInfoStruct> stateInfo = Teuchos::rcp(new Albany::StateInfoStruct); //empty
  Albany::AbstractFieldContainer::FieldContainerRequirements requirements; //empty?

  int neq = 1+nEigenvals; // number of mesh-equations
  Teuchos::RCP<Piro::MLRigidBodyModes> rigidBodyModes(Teuchos::rcp(new Piro::MLRigidBodyModes(neq)));
  disc = discFactory.createDiscretization(neq, stateInfo,requirements,rigidBodyModes);

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
  if(saved_initial_guess != Teuchos::null) {
    std::cout << "DEBUG CPS: returning saved initial guess!" << std::endl;
    return saved_initial_guess;
  }

  //Put together x_init's from Poisson and Schrodinger for now (but does this make sense for eigenvectors?) -- TODO: discuss
  Teuchos::RCP<const Epetra_Vector> poisson_x_init = poissonModel->get_x_init(); // should have disc_map
  Teuchos::RCP<const Epetra_Vector> schrodinger_x_init = schrodingerModel->get_x_init(); // should have disc_map
  
  Teuchos::RCP<Epetra_Vector> x_init = Teuchos::rcp(new Epetra_Vector(*combined_SP_map));
  Teuchos::RCP<Epetra_Vector> x_init_poisson;
  Teuchos::RCP<Epetra_MultiVector> x_init_schrodinger;

  separateCombinedVector(x_init, x_init_poisson, x_init_schrodinger);

  std::vector<int> localInds( poisson_x_init->MyLength() );
  for(int i=0; i < poisson_x_init->MyLength(); i++) localInds[i] = i;

  x_init_poisson->ReplaceMyValues( poisson_x_init->MyLength(), &(*poisson_x_init)[0], &localInds[0] );
  for(int k=0; k < nEigenvals; k++)
    (*x_init_schrodinger)(k)->ReplaceMyValues( schrodinger_x_init->MyLength(), &(*schrodinger_x_init)[0], &localInds[0] ); //localInds are the same
  
  return x_init;
}

Teuchos::RCP<const Epetra_Vector> QCAD::CoupledPoissonSchrodinger::get_x_dot_init() const
{
  //Put together x_dot_init's from Poisson and Schrodinger for now (but does this make sense for eigenvectors?) -- TODO: discuss
  Teuchos::RCP<const Epetra_Vector> poisson_x_dot_init = poissonModel->get_x_dot_init(); // should have disc_map
  Teuchos::RCP<const Epetra_Vector> schrodinger_x_dot_init = schrodingerModel->get_x_dot_init(); // should have disc_map
  
  Teuchos::RCP<Epetra_Vector> x_dot_init = Teuchos::rcp(new Epetra_Vector(*combined_SP_map));
  Teuchos::RCP<Epetra_Vector> x_dot_init_poisson;
  Teuchos::RCP<Epetra_MultiVector> x_dot_init_schrodinger;

  separateCombinedVector(x_dot_init, x_dot_init_poisson, x_dot_init_schrodinger);

  std::vector<int> localInds( poisson_x_dot_init->MyLength() );
  for(int i=0; i < poisson_x_dot_init->MyLength(); i++) localInds[i] = i;

  x_dot_init_poisson->ReplaceMyValues( poisson_x_dot_init->MyLength(), &(*poisson_x_dot_init)[0], &localInds[0] );
  for(int k=0; k < nEigenvals; k++)
    (*x_dot_init_schrodinger)(k)->ReplaceMyValues( schrodinger_x_dot_init->MyLength(), &(*schrodinger_x_dot_init)[0], &localInds[0] ); //same localInds are the same
  
  //Teuchos::RCP<const Epetra_Vector> const_x_dot_init = Teuchos::rcp(new const Epetra_Vector(*x_dot_init));
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
  // Get material parameters for quantum region, used in computing quantum density
  std::string quantumMtrlName = materialDB->getParam<std::string>("Quantum Material");
  int valleyDegeneracyFactor = materialDB->getMaterialParam<int>(quantumMtrlName,"Number of conduction band min",2);
  double effMass = materialDB->getMaterialParam<double>(quantumMtrlName,"Transverse Electron Effective Mass");

  return Teuchos::rcp( new QCAD::CoupledPSJacobian(nEigenvals, disc_map, combined_SP_map, myComm, 
						   numDims, valleyDegeneracyFactor, temperature,
						   length_unit_in_m, effMass, offset_to_CB) );
}

Teuchos::RCP<EpetraExt::ModelEvaluator::Preconditioner>
QCAD::CoupledPoissonSchrodinger::create_WPrec() const
{
  //TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
  //			     "create_WPrec Error!  Jacobian preconditioners in QCAD::CoupledPoissonSchrodinger are not implemented yet!!");
  //return Teuchos::null;

  std::cout << "DEBUG:  CPS create_WPrec called!!" << std::endl;
  Teuchos::RCP<Epetra_Operator> precOp = Teuchos::rcp( new QCAD::CoupledPSPreconditioner(nEigenvals, disc_map, combined_SP_map, myComm) );

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
  outArgs.setSupports(OUT_ARG_WPrec, true);
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
    std::cout << "DEBUG: WARNING: x_dot given to CoupledPoissonSchrodinger evalModel!!" << std::endl;
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
  // Get views into 'x' (and 'xdot'?) vectors to use for separate poisson and schrodinger application object calls
  //
  int disc_nMyElements = disc_map->NumMyElements();

  Teuchos::RCP<const Epetra_Vector> x_poisson, xdot_poisson, eigenvals_dist;
  Teuchos::RCP<const Epetra_MultiVector> x_schrodinger, xdot_schrodinger;
  std::vector<const Epetra_Vector*> xdot_schrodinger_vec(nEigenvals);
  separateCombinedVector(x, x_poisson, x_schrodinger, eigenvals_dist);
    
  if (x_dot != Teuchos::null) {  //maybe unnecessary - it seems that the coupled PS model evaluator shouldn't support x_dot ...
    separateCombinedVector(x_dot, xdot_poisson, xdot_schrodinger);
    for(int i=0; i<nEigenvals; i++) xdot_schrodinger_vec[i] = (*xdot_schrodinger)(i);
  }
  else {
    xdot_poisson = Teuchos::null;
    for(int i=0; i<nEigenvals; i++) 
      xdot_schrodinger_vec[i] = NULL;
  }

  //
  // Communicate all the eigenvalues to every processor, since all parts of the mesh need them
  //
  int my_nEigenvals = combined_SP_map->NumMyElements() - disc_nMyElements * (1+nEigenvals);
  Epetra_Map dist_eigenval_map(nEigenvals, my_nEigenvals, 0, *myComm);
  Epetra_LocalMap local_eigenval_map(nEigenvals, 0, *myComm);
  Epetra_Import eigenval_importer(local_eigenval_map, dist_eigenval_map);

  Teuchos::RCP<Epetra_Vector> eigenvals =  Teuchos::rcp(new Epetra_Vector(local_eigenval_map));
  eigenvals->Import(*eigenvals_dist, eigenval_importer, Insert);
  Teuchos::RCP<std::vector<double> > stdvec_eigenvals = Teuchos::rcp(new std::vector<double>(&(*eigenvals)[0], &(*eigenvals)[0] + nEigenvals));

  //
  // Get views into 'f' residual vector to use for separate poisson and schrodinger application object calls
  //
  Teuchos::RCP<Epetra_Vector>   f_poisson, f_norm_local, f_norm_dist;
  Teuchos::RCP<Epetra_MultiVector> f_schrodinger;
  std::vector<Epetra_Vector*> f_schrodinger_vec(nEigenvals);

  if(f_out != Teuchos::null) {
    separateCombinedVector(f_out, f_poisson, f_schrodinger, f_norm_dist);
    for(int i=0; i<nEigenvals; i++) f_schrodinger_vec[i] = (*f_schrodinger)(i);

    // Create local vector for holding the residual of the normalization equations on each proc.
    //   (later we sum all procs contributions together and copy into distributed f_norm_dist vector)
    f_norm_local = Teuchos::rcp(new Epetra_Vector(local_eigenval_map));
  }
  else {
    f_poisson = Teuchos::null;
    for(int i=0; i<nEigenvals; i++) f_schrodinger_vec[i] = NULL;
    f_norm_local = f_norm_dist = Teuchos::null;
  }


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
  for(int i=0; i<nEigenvals; i++) {
    (*(eigenData->eigenvectorRe))(i)->Import( *((*x_schrodinger)(i)), *overlap_importer, Insert );
    //(*(eigenData->eigenvectorRe))(i)->PutScalar(0.0); //DEBUG - zero out eigenvectors passed to Poisson
  }

    // set eigenvalues / eigenvectors for use in poisson problem:
  poissonApp->getStateMgr().setEigenData(eigenData);


  // Get overlapped version of potential (x_poisson) for passing as auxData to schrodinger app
  Teuchos::RCP<Epetra_MultiVector> overlapped_V = Teuchos::rcp(new Epetra_MultiVector(*disc_overlap_map, 1));
  Teuchos::RCP<Epetra_Vector> ones_vec = Teuchos::rcp(new Epetra_Vector(*disc_overlap_map));
  ones_vec->PutScalar(1.0);
  (*overlapped_V)(0)->Import( *x_poisson, *overlap_importer, Insert );
  (*overlapped_V)(0)->Update(offset_to_CB, *ones_vec, -1.0);
  //std::cout << "DEBUG: Offset to conduction band = " << offset_to_CB << std::endl;

  // set potential for use in schrodinger problem
  schrodingerApp->getStateMgr().setAuxData(overlapped_V);


  
  //
  // Compute the functions
  //
  bool f_already_computed = false;

  // Mass Matrix -- needed even if we don't need to compute the Jacobian, since it enters into the normalization equations
  //   --> Compute mass matrix using schrodinger equation -- independent of eigenvector so can just use 0th
  //       Note: to compute this, we need to evaluate the schrodinger problem as a transient problem, so create a dummy xdot...
  Teuchos::RCP<Epetra_Operator> M_out_schrodinger = schrodingerModel->create_W(); //maybe re-use this and not create it every time?
  Teuchos::RCP<Epetra_CrsMatrix> M_out_schrodinger_crs = Teuchos::rcp_dynamic_cast<Epetra_CrsMatrix>(M_out_schrodinger, true);
  Teuchos::RCP<const Epetra_Vector> dummy_xdot = schrodingerModel->get_x_dot_init(); // I think this would work as well: Teuchos::rcp(new Epetra_Vector(*disc_map)) 
  schrodingerApp->computeGlobalJacobian(1.0, 0.0, curr_time, dummy_xdot.get(), *((*x_schrodinger)(0)), 
					    schrodinger_sacado_param_vec, f_schrodinger_vec[0], *M_out_schrodinger_crs);


  // Hamiltionan Matrix -- needed even if we don't need to compute the Jacobian, since this is how we compute the schrodinger residuals
  //   --> Computed as jacobian matrix of schrodinger equation -- independent of eigenvector so can just use 0th
  Teuchos::RCP<Epetra_Operator> J_out_schrodinger = schrodingerModel->create_W(); //maybe re-use this and not create it every time?
  Teuchos::RCP<Epetra_CrsMatrix> J_out_schrodinger_crs = Teuchos::rcp_dynamic_cast<Epetra_CrsMatrix>(J_out_schrodinger, true);
  schrodingerApp->computeGlobalJacobian(0.0, 1.0, curr_time, dummy_xdot.get(), *((*x_schrodinger)(0)), 
					    schrodinger_sacado_param_vec, f_schrodinger_vec[0], *J_out_schrodinger_crs);


  // W 
  if (W_out != Teuchos::null) { 
    // W = alpha*M + beta*J where M is mass mx and J is jacobian

    //if we need to compute the jacobian, get the jacobians of the poisson and schrodinger
    //  applications (as crs matrices), as well as the mass matrix (from the schrodinger problem,
    //  since it includes xdot - maybe need to fabricate this??) and from these construct a CoupledPoissonSchrodingerJacobian object (an Epetra_Operator)
    
    //TODO - how to allow general alpha and beta?  This won't work given current logic, so we should test that alpha=0, beta=1 and throw an error otherwise...

    // Compute poisson Jacobian
    Teuchos::RCP<Epetra_Operator> W_out_poisson = poissonModel->create_W(); //maybe re-use this and not create it every time?
    Teuchos::RCP<Epetra_CrsMatrix> W_out_poisson_crs = Teuchos::rcp_dynamic_cast<Epetra_CrsMatrix>(W_out_poisson, true);

    poissonApp->computeGlobalJacobian(alpha, beta, curr_time, xdot_poisson.get(), *x_poisson, 
				      poisson_sacado_param_vec, f_poisson.get(), *W_out_poisson_crs);

    
    TEUCHOS_TEST_FOR_EXCEPTION(nEigenvals <= 0, Teuchos::Exceptions::InvalidParameter,"Error! The number of eigenvalues must be greater than zero.");
      
    //Is this necessary? Done above for hard-coded jacobian since we need schrodinger hamiltonian matrix for residual computation...
    //Compute schrodinger Jacobian using first eigenvector -- independent of eigenvector since Schro. eqn is linear
    Teuchos::RCP<Epetra_Operator> W_out_schrodinger = schrodingerModel->create_W(); //maybe re-use this and not create it every time?
    Teuchos::RCP<Epetra_CrsMatrix> W_out_schrodinger_crs = Teuchos::rcp_dynamic_cast<Epetra_CrsMatrix>(W_out_schrodinger, true);
    schrodingerApp->computeGlobalJacobian(alpha, beta, curr_time, xdot_schrodinger_vec[0], *((*x_schrodinger)(0)), 
					  schrodinger_sacado_param_vec, f_schrodinger_vec[0], *W_out_schrodinger_crs);
    
    Teuchos::RCP<QCAD::CoupledPSJacobian> W_out_psj = Teuchos::rcp_dynamic_cast<QCAD::CoupledPSJacobian>(W_out, true);
    W_out_psj->initialize(W_out_poisson_crs, W_out_schrodinger_crs, M_out_schrodinger_crs, eigenvals, x_schrodinger);

    //I don't think the full residual (f) has been computed, since we only needed to evaluate the schrodinger app on the first evec...
    //f_already_computed=true; 



    /*
    // DEBUG --- JACOBIAN TEST -----------------------------------------------------------------------------------------
     
    Teuchos::RCP<Epetra_Vector> x_plus_dx = Teuchos::rcp( new Epetra_Vector(*combined_SP_map) );
    Teuchos::RCP<Epetra_Vector> dx_test = Teuchos::rcp( new Epetra_Vector(*combined_SP_map) );
    double eps;
    
    //Init dx_test
    Teuchos::RCP<Epetra_Vector> dx_test_poisson, dx_test_eigenvals;
    Teuchos::RCP<Epetra_MultiVector> dx_test_schrodinger;
    separateCombinedVector(dx_test, dx_test_poisson, dx_test_schrodinger, dx_test_eigenvals);

    eps = 1e-7;
    dx_test->PutScalar(1.0);
    //dx_test_poisson->PutScalar(1.0);
    //dx_test_schrodinger->PutScalar(1.0);
    //dx_test_eigenvals->PutScalar(1.0);

    Teuchos::RCP<Epetra_Vector> f_test_manual = Teuchos::rcp( new Epetra_Vector(*combined_SP_map) );
    Teuchos::RCP<Epetra_Vector> f_test_tmp = Teuchos::rcp( new Epetra_Vector(*combined_SP_map) );
    Teuchos::RCP<Epetra_Vector> f_test_jac = Teuchos::rcp( new Epetra_Vector(*combined_SP_map) );

    computeResidual(x, f_test_manual, M_out_schrodinger_crs);
    //f_test_manual->Print( std::cout << "JACOBIAN TEST:  MANUAL 1" << std::endl);
    x_plus_dx->Update(1.0, *x, eps, *dx_test, 0.0); // x + eps*dx
    computeResidual(x_plus_dx, f_test_tmp, M_out_schrodinger_crs);
    //f_test_tmp->Print( std::cout << "JACOBIAN TEST:  MANUAL 2" << std::endl);
    f_test_manual->Update(1.0/eps, *f_test_tmp, -1.0/eps); // f_test_manual = (resid(x+eps*dx) - resid(x)) / eps ~ jacobian * dx

    W_out_psj->Apply(*dx_test, *f_test_jac);


    //x_schrodinger->Print( std::cout << "JACOBIAN TEST:  X_SCHRODINGER" << std::endl);
    f_test_manual->Print( std::cout << "JACOBIAN TEST:  MANUAL DIFF" << std::endl);
    f_test_jac->Print( std::cout << "JACOBIAN TEST:  JACOBIAN DIFF" << std::endl);

    f_test_tmp->Update(1.0, *f_test_manual, -1.0, *f_test_jac, 0.0);
    f_test_tmp->Print( std::cout << "JACOBIAN TEST:  COMPARISON VECTOR" << std::endl);
    
    double test_norm;
    f_test_tmp->Norm2(&test_norm);
    std::cout << "JACOBIAN TEST: COMPARISON VECTOR 2-NORM = " << test_norm << std::endl;

    // DEBUG --- JACOBIAN TEST -----------------------------------------------------------------------------------------
    */
  }


  if (WPrec_out != Teuchos::null) {
     // Get Poisson Preconditioner
    Teuchos::RCP<Epetra_Operator> WPrec_poisson;
     
       // Get the jacobian -- TODO: better to just copy the Jacobian if we already computed it?
     Teuchos::RCP<Epetra_CrsMatrix> Extra_W_crs_poisson = Teuchos::rcp_dynamic_cast<Epetra_CrsMatrix>(poissonModel->create_W(), true);
     poissonApp->computeGlobalJacobian(alpha, beta, curr_time, xdot_poisson.get(), *x_poisson, 
				       poisson_sacado_param_vec, f_poisson.get(), *Extra_W_crs_poisson);
     //f_already_computed=true;

     bool poisson_supports_teko_prec = false;  // I think this should = whether poisson outargs supports OUT_ARG_WPrec
     if( poisson_supports_teko_prec ) {
       Teuchos::RCP<EpetraExt::ModelEvaluator::Preconditioner> WPrec_poisson_pre = poissonModel->create_WPrec(); //maybe re-use this and not create it every time?
       WPrec_poisson = WPrec_poisson_pre->PrecOp;       
       poissonApp->computeGlobalPreconditioner(Extra_W_crs_poisson, WPrec_poisson);
     }
     else {
       // Use Ifpack to get a pseudo inverse of Extra_W_crs_poisson
       Teuchos::ParameterList Ifpack_list;
       Ifpack Ifpack_factory; // allocate an IFPACK factory.

       // create the preconditioner. -- maybe pull this info from input file in FUTURE
       std::string PrecType = "ILU"; // incomplete LU
       int OverlapLevel = 1; // must be >= 0. If Comm.NumProc() == 1, it is ignored.

       Teuchos::RCP<Ifpack_Preconditioner> WPrec_poisson_pre = Teuchos::rcp( Ifpack_factory.Create(PrecType, &*Extra_W_crs_poisson, OverlapLevel) );
       assert(WPrec_poisson_pre != Teuchos::null);

       // specify parameters for ILU -- maybe pull this info from input file in FUTURE
       Ifpack_list.set("fact: drop tolerance", 1e-9);
       Ifpack_list.set("fact: level-of-fill", 1);
       // the combine mode is on the following:
       // "Add", "Zero", "Insert", "InsertAdd", "Average", "AbsMax"
       // Their meaning is as defined in file Epetra_CombineMode.h   
       Ifpack_list.set("schwarz: combine mode", "Add");


       if( WPrec_poisson_pre->SetParameters(Ifpack_list) != 0 ) // sets the parameters
	 TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,"Error! Invalid IFPACK Parameters.");	 
       if( WPrec_poisson_pre->Initialize() != 0)                // initialize preconditioner (must fillComplete matrix by now)
	 TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,"Error Inializing Ifpack preconditioner");	 
       if( WPrec_poisson_pre->Compute() != 0)                   // compute preconditioner
	 TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,"Error Computing Ifpack preconditioner");	 

       WPrec_poisson = Teuchos::rcp_dynamic_cast<Epetra_Operator>(WPrec_poisson_pre, true);
     }

     // Get Schrodinger Preconditioner
    Teuchos::RCP<Epetra_Operator> WPrec_schrodinger;

       // Get another copy of the jacobian -- TODO: better to just copy the Jacobian if we already computed it?
     Teuchos::RCP<Epetra_CrsMatrix> Extra_W_crs_schrodinger = Teuchos::rcp_dynamic_cast<Epetra_CrsMatrix>(schrodingerModel->create_W(), true);
     schrodingerApp->computeGlobalJacobian(alpha, beta, curr_time, xdot_schrodinger_vec[0], *((*x_schrodinger)(0)), 
					    schrodinger_sacado_param_vec, f_schrodinger_vec[0], *Extra_W_crs_schrodinger);
     //f_already_computed=true;

     bool schrodinger_supports_teko_prec = false;  // I think this should = whether poisson outargs supports OUT_ARG_WPrec
     if( schrodinger_supports_teko_prec ) {
       Teuchos::RCP<EpetraExt::ModelEvaluator::Preconditioner> WPrec_schrodinger_pre = schrodingerModel->create_WPrec(); //maybe re-use this and not create every time?
       WPrec_schrodinger = WPrec_schrodinger_pre->PrecOp;
       schrodingerApp->computeGlobalPreconditioner(Extra_W_crs_schrodinger, WPrec_schrodinger);
     }
     else {
       // Use Ifpack to get a pseudo inverse of Extra_W_crs_schrodinger
       Teuchos::ParameterList Ifpack_list;
       Ifpack Ifpack_factory; // allocate an IFPACK factory.

       // create the preconditioner. -- maybe pull this info from input file in FUTURE
       std::string PrecType = "ILU"; // incomplete LU
       int OverlapLevel = 1; // must be >= 0. If Comm.NumProc() == 1, it is ignored.

       Teuchos::RCP<Ifpack_Preconditioner> WPrec_schrodinger_pre = Teuchos::rcp( Ifpack_factory.Create(PrecType, &*Extra_W_crs_schrodinger, OverlapLevel) );
       assert(WPrec_schrodinger_pre != Teuchos::null);

       // specify parameters for ILU -- maybe pull this info from input file in FUTURE
       Ifpack_list.set("fact: drop tolerance", 1e-9);
       Ifpack_list.set("fact: level-of-fill", 1);
       // the combine mode is on the following:
       // "Add", "Zero", "Insert", "InsertAdd", "Average", "AbsMax"
       // Their meaning is as defined in file Epetra_CombineMode.h   
       Ifpack_list.set("schwarz: combine mode", "Add");


       if( WPrec_schrodinger_pre->SetParameters(Ifpack_list) != 0 ) // sets the parameters
	 TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,"Error! Invalid IFPACK Parameters.");	 
       if( WPrec_schrodinger_pre->Initialize() != 0)                // initialize preconditioner (must fillComplete matrix by now)
	 TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,"Error Inializing Ifpack preconditioner");	 
       if( WPrec_schrodinger_pre->Compute() != 0)                   // compute preconditioner
	 TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,"Error Computing Ifpack preconditioner");	 

       WPrec_schrodinger = Teuchos::rcp_dynamic_cast<Epetra_Operator>(WPrec_schrodinger_pre, true);
     }

     Teuchos::RCP<QCAD::CoupledPSPreconditioner> WPrec_out_psp = Teuchos::rcp_dynamic_cast<QCAD::CoupledPSPreconditioner>(WPrec_out, true);
     WPrec_out_psp->initialize(WPrec_poisson, WPrec_schrodinger);
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
	  schrodingerApp->computeGlobalTangent(0.0, 0.0, curr_time, false, xdot_schrodinger_vec[k], *((*x_schrodinger)(k)),
				    schrodinger_sacado_param_vec, p_vec.get(),
				    NULL, NULL, NULL, f_schrodinger_vec[k], NULL, 
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
      Epetra_Vector M_vec(*disc_map);  //temp storage for mass matrix times vec -- maybe don't allocate this on the stack??

      poissonApp->computeGlobalResidual(curr_time, xdot_poisson.get(), *x_poisson, 
					poisson_sacado_param_vec, *f_poisson);
      
      for(int i=0; i<nEigenvals; i++) {

	// Compute Mass_matrix * eigenvector[i]
	const Epetra_Vector& vec = *((*x_schrodinger)(i));
	M_out_schrodinger_crs->Multiply(false, vec, M_vec);  


	// Compute the schrodinger residual f_schrodinger_vec[i]: H*eigenvector[i] - eigenvalue[i] * M * eigenvector[i]
	/*schrodingerApp->computeGlobalResidual(curr_time, xdot_schrodinger_vec[i], *((*x_schrodinger)(i)), 
					      schrodinger_sacado_param_vec, *(f_schrodinger_vec[i]) );  // -H*evec[i]
	*/

	// H * Psi - E * M * Psi
	const Epetra_CrsMatrix& Hamiltonian_crs =  *J_out_schrodinger_crs;
	Hamiltonian_crs.Multiply(false, vec, *(f_schrodinger_vec[i]));  

	/*
	//DEBUG
	double He_norm, Me_norm, H_expect;
	f_schrodinger_vec[i]->Norm2(&He_norm);
	M_vec.Norm2(&Me_norm);
	std::cout << "EGN DEBUG " << i << ": norm(-H*evec) = " << He_norm << ", norm(M*evec) = " << Me_norm 
	   << ", eval = " << (*stdvec_eigenvals)[i] << std::endl;
	f_schrodinger_vec[i]->Dot(vec, &H_expect);
	//DEBUG
	*/

        // add -eval[i]*M*evec[i] to H*evec[i] (recall evals are really negative_evals)
        f_schrodinger_vec[i]->Update( (*stdvec_eigenvals)[i], M_vec, 1.0); 

        // Compute normalization equation residuals:  f_norm[i] = abs(1 - evec[i] . M . evec[i])
	double vec_M_vec;
	vec.Dot( M_vec, &vec_M_vec );
	(*f_norm_local)[i] = 1.0 - vec_M_vec;
      }

      // Fill elements of f_norm_dist that belong to this processor, i.e. loop over
      // eigenvalue indices "owned" by the current proc in the combined distributed map
      std::vector<int> eval_global_elements(my_nEigenvals);
      dist_eigenval_map.MyGlobalElements(&eval_global_elements[0]);
      for(int i=0; i<my_nEigenvals; i++)
	(*f_norm_dist)[i] = (*f_norm_local)[eval_global_elements[i]];

      

      //DEBUG -- print residual in gory detail to debugging
      std::cout << "DEBUG: ----------------- Coupled Schrodinger Poisson Info Dump ---------------------" << std::endl;
      double norm, mean;

      /*std::cout << "x map has " << x->Map().NumGlobalElements() << " global els" << std::endl;
      std::cout << "x_poisson map has " << x_poisson->Map().NumGlobalElements() << " global els" << std::endl;
      std::cout << "x_schrodinger map has " << x_schrodinger->Map().NumGlobalElements() << " global els (each vec)" << std::endl;
      std::cout << "dist_eval_map has " << dist_eigenval_map.NumGlobalElements() << " global els" << std::endl;
      */

      x->Norm2(&norm); x->MeanValue(&mean);
      std::cout << std::setprecision(10);
      std::cout << "X Norm & Mean = " << norm << " , " << mean << std::endl;

      x_poisson->Norm2(&norm); x_poisson->MeanValue(&mean);
      std::cout << "Poisson-part X Norm & Mean = " << norm << " , " << mean << std::endl;
      for(int i=0; i<nEigenvals; i++) {
	(*x_schrodinger)(i)->Norm2(&norm);
	std::cout << "Schrodinger[" << i << "]-part X Norm = " << norm << std::endl;
      }
      for(int i=0; i<nEigenvals; i++) 
	std::cout << "Eigenvalue[" << i << "] = " << (*stdvec_eigenvals)[i] << std::endl;

      f_poisson->Norm2(&norm);
      std::cout << "Poisson-part Residual Norm = " << norm << std::endl; //f_poisson->Print(std::cout);
      for(int i=0; i<nEigenvals; i++) {
	if(f_schrodinger_vec[i] != NULL) {
	  f_schrodinger_vec[i]->Norm2(&norm);
	  std::cout << "Schrodinger[" << i << "]-part Residual Norm = " << norm << std::endl; //f_schrodinger_vec[i]->Print(std::cout);
	}
      }
      std::cout << "Eigenvalue-part Residual: " << std::endl; f_norm_dist->Print(std::cout);
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
	schrodingerApp->evaluateResponseDerivative(i - poissonApp->getNumResponses(), curr_time, xdot_schrodinger_vec[0], *((*x_schrodinger)(0)),
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
	  schrodingerApp->evaluateResponseTangent(i - poissonApp->getNumResponses(), alpha, beta, curr_time, false,
						  xdot_schrodinger_vec[0], *((*x_schrodinger)(0)),
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
	schrodingerApp->evaluateResponse(i - poissonApp->getNumResponses(), curr_time, xdot_schrodinger_vec[0], *((*x_schrodinger)(0)), 
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

Teuchos::RCP<Albany::Application>
QCAD::CoupledPoissonSchrodinger::getSchrodingerApp() const
{
  return schrodingerApp;
}



void QCAD::CoupledPoissonSchrodinger::separateCombinedVector(const Teuchos::RCP<Epetra_Vector>& combinedVector,
							     Teuchos::RCP<Epetra_Vector>& poisson_part,
							     Teuchos::RCP<Epetra_MultiVector>& schrodinger_part) const
{
  double* data;
  int disc_nMyElements = disc_map->NumMyElements();

  if(combinedVector->ExtractView(&data) != 0)
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
			       "Error!  QCAD::CoupledPoissonSchrodinger cannot extract vector views");

  poisson_part = Teuchos::rcp(new Epetra_Vector(::View, *disc_map, &data[0]));
  schrodinger_part = Teuchos::rcp(new Epetra_MultiVector(::View, *disc_map, &data[disc_nMyElements], disc_nMyElements, nEigenvals));
}


void QCAD::CoupledPoissonSchrodinger::separateCombinedVector(const Teuchos::RCP<Epetra_Vector>& combinedVector,
							     Teuchos::RCP<Epetra_Vector>& poisson_part,
							     Teuchos::RCP<Epetra_MultiVector>& schrodinger_part,
							     Teuchos::RCP<Epetra_Vector>& eigenvalue_part) const
{
  this->separateCombinedVector(combinedVector, poisson_part, schrodinger_part);

  double* data;
  int disc_nMyElements = disc_map->NumMyElements();
  int my_nEigenvals = combined_SP_map->NumMyElements() - disc_nMyElements * (1+nEigenvals);
  Epetra_Map dist_eigenval_map(nEigenvals, my_nEigenvals, 0, *myComm);

  combinedVector->ExtractView(&data); //above call tests for failure
  eigenvalue_part = Teuchos::rcp(new Epetra_Vector(::View, dist_eigenval_map, &data[(1+nEigenvals)*disc_nMyElements]));
}



void QCAD::CoupledPoissonSchrodinger::separateCombinedVector(const Teuchos::RCP<const Epetra_Vector>& combinedVector,
							     Teuchos::RCP<const Epetra_Vector>& poisson_part,
							     Teuchos::RCP<const Epetra_MultiVector>& schrodinger_part) const
{
  double* data;
  int disc_nMyElements = disc_map->NumMyElements();

  if(combinedVector->ExtractView(&data) != 0)
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
			       "Error!  QCAD::CoupledPoissonSchrodinger cannot extract vector views");

  poisson_part = Teuchos::rcp(new const Epetra_Vector(::View, *disc_map, &data[0]));
  schrodinger_part = Teuchos::rcp(new const Epetra_MultiVector(::View, *disc_map, &data[disc_nMyElements], disc_nMyElements, nEigenvals));
}


void QCAD::CoupledPoissonSchrodinger::separateCombinedVector(const Teuchos::RCP<const Epetra_Vector>& combinedVector,
							     Teuchos::RCP<const Epetra_Vector>& poisson_part,
							     Teuchos::RCP<const Epetra_MultiVector>& schrodinger_part,
							     Teuchos::RCP<const Epetra_Vector>& eigenvalue_part) const
{
  this->separateCombinedVector(combinedVector, poisson_part, schrodinger_part);

  double* data;
  int disc_nMyElements = disc_map->NumMyElements();
  int my_nEigenvals = combined_SP_map->NumMyElements() - disc_nMyElements * (1+nEigenvals);
  Epetra_Map dist_eigenval_map(nEigenvals, my_nEigenvals, 0, *myComm);

  combinedVector->ExtractView(&data); //above call tests for failure
  eigenvalue_part = Teuchos::rcp(new const Epetra_Vector(::View, dist_eigenval_map, &data[(1+nEigenvals)*disc_nMyElements]));
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




void QCAD::CoupledPoissonSchrodinger::computeResidual(const Teuchos::RCP<const Epetra_Vector>& x,
						      Teuchos::RCP<Epetra_Vector>& f,
						      Teuchos::RCP<Epetra_CrsMatrix>& massMx) const
{
  double curr_time = 0.0;
  Epetra_Vector M_vec(*disc_map);  //temp storage for mass matrix times vec -- maybe don't allocate this on the stack??
  Epetra_LocalMap local_eigenval_map(nEigenvals, 0, *myComm);

  int disc_nMyElements = disc_map->NumMyElements();
  int my_nEigenvals = combined_SP_map->NumMyElements() - disc_nMyElements * (1+nEigenvals);

  Teuchos::RCP<const Epetra_Vector> x_poisson, eigenvals_dist;
  Teuchos::RCP<const Epetra_MultiVector> x_schrodinger;
  separateCombinedVector(x, x_poisson, x_schrodinger, eigenvals_dist);

  Teuchos::RCP<Epetra_Vector>   f_poisson, f_norm_local, f_norm_dist;
  Teuchos::RCP<Epetra_MultiVector> f_schrodinger;
  std::vector<Epetra_Vector*> f_schrodinger_vec(nEigenvals);
  separateCombinedVector(f, f_poisson, f_schrodinger, f_norm_dist);
  for(int i=0; i<nEigenvals; i++) f_schrodinger_vec[i] = (*f_schrodinger)(i);
  f_norm_local = Teuchos::rcp(new Epetra_Vector(local_eigenval_map));


  //update schrodinger wavefunctions for poisson

  Epetra_Import eigenval_importer(local_eigenval_map, eigenvals_dist->Map() );
  Teuchos::RCP<Epetra_Vector> eigenvals =  Teuchos::rcp(new Epetra_Vector(local_eigenval_map));
  eigenvals->Import(*eigenvals_dist, eigenval_importer, Insert);
  Teuchos::RCP<std::vector<double> > stdvec_eigenvals = Teuchos::rcp(new std::vector<double>(&(*eigenvals)[0], &(*eigenvals)[0] + nEigenvals));

  Teuchos::RCP<Albany::EigendataStruct> eigenData = Teuchos::rcp( new Albany::EigendataStruct );
  eigenData->eigenvalueRe = stdvec_eigenvals;
  eigenData->eigenvectorRe = Teuchos::rcp(new Epetra_MultiVector(*disc_overlap_map, nEigenvals));
  eigenData->eigenvectorIm = Teuchos::null;
  Teuchos::RCP<Epetra_Import> overlap_importer = Teuchos::rcp(new Epetra_Import(*disc_overlap_map, *disc_map));

  for(int i=0; i<nEigenvals; i++)
    (*(eigenData->eigenvectorRe))(i)->Import( *((*x_schrodinger)(i)), *overlap_importer, Insert );

    // set eigenvalues / eigenvectors for use in poisson problem:
  poissonApp->getStateMgr().setEigenData(eigenData);
  poissonApp->computeGlobalResidual(curr_time, NULL, *x_poisson, 
				    poisson_sacado_param_vec, *f_poisson);
      

    // Get overlapped version of potential (x_poisson) for passing as auxData to schrodinger app
  Teuchos::RCP<Epetra_MultiVector> overlapped_V = Teuchos::rcp(new Epetra_MultiVector(*disc_overlap_map, 1));
  Teuchos::RCP<Epetra_Vector> ones_vec = Teuchos::rcp(new Epetra_Vector(*disc_overlap_map));
  ones_vec->PutScalar(1.0);
  (*overlapped_V)(0)->Import( *x_poisson, *overlap_importer, Insert );
  (*overlapped_V)(0)->Update(offset_to_CB, *ones_vec, -1.0);
  schrodingerApp->getStateMgr().setAuxData(overlapped_V);

    // compute schrodinger Hamiltonian
  Teuchos::RCP<Epetra_Operator> hamMx = schrodingerModel->create_W(); //maybe re-use this and not create it every time?
  Teuchos::RCP<Epetra_CrsMatrix> hamMx_crs = Teuchos::rcp_dynamic_cast<Epetra_CrsMatrix>(hamMx, true);
  Teuchos::RCP<const Epetra_Vector> dummy_xdot = schrodingerModel->get_x_dot_init();
  schrodingerApp->computeGlobalJacobian(0.0, 1.0, curr_time, dummy_xdot.get(), *((*x_schrodinger)(0)), 
					schrodinger_sacado_param_vec, f_schrodinger_vec[0], *hamMx_crs);

  for(int i=0; i<nEigenvals; i++) {
    const Epetra_Vector& vec = *((*x_schrodinger)(i));
    massMx->Multiply(false, vec, M_vec);  
    hamMx_crs->Multiply(false, vec, *(f_schrodinger_vec[i]));  
    f_schrodinger_vec[i]->Update( (*stdvec_eigenvals)[i], M_vec, 1.0); 

    // Compute normalization equation residuals:  f_norm[i] = abs(1 - evec[i] . M . evec[i])
    double vec_M_vec;
    vec.Dot( M_vec, &vec_M_vec );
    (*f_norm_local)[i] = 1.0 - vec_M_vec;
  }

  // Fill elements of f_norm_dist that belong to this processor, i.e. loop over
  // eigenvalue indices "owned" by the current proc in the combined distributed map
  std::vector<int> eval_global_elements(my_nEigenvals);
  eigenvals_dist->Map().MyGlobalElements(&eval_global_elements[0]);
  for(int i=0; i<my_nEigenvals; i++)
    (*f_norm_dist)[i] = (*f_norm_local)[eval_global_elements[i]];
}
