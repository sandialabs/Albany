//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "SchwarzMultiscale.hpp"
#include "Albany_SolverFactory.hpp" 
#include "Albany_ModelFactory.hpp" 

LCM::
SchwarzMultiscale::
SchwarzMultiscale(const Teuchos::RCP<Teuchos::ParameterList>& appParams,
                  const Teuchos::RCP<const Teuchos::Comm<int> >& commT,  
                  const Teuchos::RCP<const Tpetra_Vector>& initial_guessT)
{
  std::cout << "Initializing Schwarz Multiscale constructor!" << std::endl;
  commT_ = commT; 
  // Get "Coupled Schwarz" parameter sublist
  Teuchos::ParameterList& coupledSystemParams = appParams->sublist("Coupled System");
  // Get names of individual model xml input files from problem parameterlist
  Teuchos::Array<std::string> model_filenames =
      coupledSystemParams.get<Teuchos::Array<std::string> >("Model XML Files");
  //number of models
  n_models_ = model_filenames.size(); 
  std::cout << "DEBUG: n_models_: " << n_models_ << std::endl;
  Teuchos::Array< Teuchos::RCP<Albany::Application> > apps(n_models_);
  models_.resize(n_models_);
  Teuchos::Array< Teuchos::RCP<Teuchos::ParameterList> > modelAppParams(n_models_);
  Teuchos::Array< Teuchos::RCP<Teuchos::ParameterList> > modelProblemParams(n_models_);
  Teuchos::Array< Teuchos::RCP<const Tpetra_Map> > disc_maps(n_models_); 
  Teuchos::Array< Teuchos::RCP<const Tpetra_Map> > disc_overlap_maps(n_models_);
  material_dbs_.resize(n_models_);  
  std::string mtrDbFilename = "materials.xml";
  //char mtrDbFilename[100];  //create string for file name
  //
  
  //Create a dummy solverFactory for validating application parameter lists 
  //(see QCAD::CoupledPoissonSchorodinger)
  //FIXME: look into how this is used, uncomment if necessary 
  /*Albany::SolverFactory validFactory( Teuchos::createParameterList("Empty dummy for Validation"), commT );
  Teuchos::RCP<const Teuchos::ParameterList> validAppParams = validFactory.getValidAppParameters();
  Teuchos::RCP<const Teuchos::ParameterList> validParameterParams = validFactory.getValidParameterParameters();
  Teuchos::RCP<const Teuchos::ParameterList> validResponseParams = validFactory.getValidResponseParameters();
  */

  //Set up each application and model object in Teuchos::Array
  //(similar logic to that in Albany::SolverFactory::createAlbanyAppAndModelT) 
  for (int m=0; m<n_models_; m++) {
    //get parameterlist from mth model *.xml file 
    Albany::SolverFactory slvrfctry(model_filenames[m], commT_);
    Teuchos::ParameterList& appParams_m = slvrfctry.getParameters(); 
    modelAppParams[m] = Teuchos::rcp(&(appParams_m)); 
    Teuchos::RCP<Teuchos::ParameterList> problemParams_m = Teuchos::sublist(modelAppParams[m], "Problem"); 
    modelProblemParams[m] = problemParams_m;
    std::string &problem_name = problemParams_m->get("Name", "");
    //FIXME: fix the following line to material name gets incremented
    //sprintf(mtrDbFilename, "materials%i.xml", m);
    if (problemParams_m->isType<std::string>("MaterialDB Filename")) 
      mtrDbFilename = problemParams_m->get<std::string>("MaterialDB Filename");
    material_dbs_[m] = Teuchos::rcp(new QCAD::MaterialDatabase(mtrDbFilename, commT_));
    //FIXME: should we throw an error if the problem names in the input files don't match??
    std::cout << "DEBUG: name of problem #" << m <<": " << problem_name << std::endl; 
    std::cout << "DEBUG: material of problem #" << m << ": " << mtrDbFilename << std::endl; 
    //create application for mth model 
    //FIXME: initial_guessT needs to be made the right one for the mth model!  Or can it be null?  
    apps[m] = Teuchos::rcp(new Albany::Application(commT, modelAppParams[m], initial_guessT));
    //Validate parameter lists
    //FIXME: add relevant things to validate to getValid* functions below
    //Uncomment
    //problemParams_m->validateParameters(*getValidProblemParameters(),0); 
    //problemParams_m->sublist("Parameters").validateParameters(*validParameterParams, 0);
    //problemParams_m->sublist("Response Functions").validateParameters(*validResponseParams, 0);
    //Create model evaluator
    Albany::ModelFactory modelFactory(modelAppParams[m], apps[m]); 
    models_[m] = modelFactory.createT();    
   }
   std::cout <<"Finished creating Albany apps and models!" << std::endl;

   //Now get maps, InArgs, OutArgs for each model.
   //Calculate how many parameters, responses there are total. 
   solver_inargs_.resize(n_models_); 
   solver_outargs_.resize(n_models_); 
   num_params_.resize(n_models_); 
   num_responses_.resize(n_models_); 
   num_params_total_ = 0; 
   num_responses_total_ = 0; 
   for (int m=0; m<n_models_; m++) {
    //FIXME: set 
    disc_maps[m] = apps[m]->getMapT(); 
    //FIXME: set 
    disc_overlap_maps[m] = apps[m]->getStateMgr().getDiscretization()->getOverlapMapT(); 
    //FIXME: set 
    solver_inargs_[m] = models_[m]->createInArgs(); 
    //FIXME: set 
    solver_outargs_[m] = models_[m]->createOutArgs(); 
    //FIXME: set 
    num_params_[m] = solver_inargs_[m].Np(); 
    //FIXME: set 
    num_responses_[m] = solver_outargs_[m].Ng(); 
    //Does it make sense for num_params_total and num_responses_total to be the sum 
    //of these values for each model?  I guess so. 
    //FIXME: set 
    num_params_total_ += num_params_[m]; 
    //FIXME: set 
    num_responses_total_ += num_responses_[m]; 
   }
   std::cout << "Total # parameters, responses: " << num_params_total_ << ", " << num_responses_total_ << std::endl; 
   //FIXME: define coupled_disc_map, a map for the entire coupled ME solution, 
   //created from the entries of the disc_map array (individual maps).
   //coupled_disc_map_ = ...  (write separate function to compute this map?) 

   //
   //FIXME: Add discretization parameterlist and discretization object for the "combined" solution 
   //vector from all the coupled Model Evaluators.  Refer to QCAD_CoupledPoissonSchrodinger.cpp. 
   //FIXME: How are we going to collect output?  Write exodus files for each model evaluator?  Joined 
   //exodus file?  
  
}


LCM::SchwarzMultiscale::~SchwarzMultiscale()
{
}

// Overridden from Thyra::ModelEvaluator<ST>
Teuchos::RCP<const Thyra::VectorSpaceBase<ST> >
LCM::SchwarzMultiscale::get_x_space() const
{
  //IK, 2/10/15: this function is done!
  std::cout << "DEBUG: In LCM::SchwarzMultiScale::get_x_space()!" << std::endl; 
  Teuchos::RCP<const Tpetra_Map> map = Teuchos::rcp(new const Tpetra_Map(*coupled_disc_map_));
  Teuchos::RCP<const Thyra::VectorSpaceBase<ST> > coupled_x_space = Thyra::createVectorSpace<ST>(map);
  return coupled_x_space;
}


Teuchos::RCP<const Thyra::VectorSpaceBase<ST> >
LCM::SchwarzMultiscale::get_f_space() const
{
  //IK, 2/10/15: this function is done!
  std::cout << "DEBUG: In LCM::SchwarzMultiScale::get_f_space()!" << std::endl; 
  Teuchos::RCP<const Tpetra_Map> map = Teuchos::rcp(new const Tpetra_Map(*coupled_disc_map_));
  Teuchos::RCP<const Thyra::VectorSpaceBase<ST> > coupled_f_space = Thyra::createVectorSpace<ST>(map);
  return coupled_f_space;
}


Teuchos::RCP<const Thyra::VectorSpaceBase<ST> >
LCM::SchwarzMultiscale::get_p_space(int l) const
{
  //FIXME: fill in!
  /*TEUCHOS_TEST_FOR_EXCEPTION(
    l >= num_param_vecs + num_dist_param_vecs || l < 0,
    Teuchos::Exceptions::InvalidParameter,
    std::endl <<
    "Error!  LCM::SchwarzMultiscale::get_p_space():  " <<
    "Invalid parameter index l = " << l << std::endl);
  Teuchos::RCP<const Tpetra_Map> map; 
  if (l < num_param_vecs)
    map = tpetra_param_map[l];  
  //IK, 7/1/14: commenting this out for now
  //map = distParamLib->get(dist_param_names[l-num_param_vecs])->map(); 
  Teuchos::RCP<const Thyra::VectorSpaceBase<ST> > tpetra_param_space = Thyra::createVectorSpace<ST>(map);
  return tpetra_param_space;*/
}


Teuchos::RCP<const Thyra::VectorSpaceBase<ST> >
LCM::SchwarzMultiscale::get_g_space(int l) const
{
  //FIXME: fill in!
  /*TEUCHOS_TEST_FOR_EXCEPTION(
      l >= app->getNumResponses() || l < 0,
      Teuchos::Exceptions::InvalidParameter,
      std::endl <<
      "Error!  LCM::SchwarzMultiscale::get_g_space():  " <<
      "Invalid response index l = " << l << std::endl);

  Teuchos::RCP<const Tpetra_Map> mapT = app->getResponse(l)->responseMapT();
  Teuchos::RCP<const Thyra::VectorSpaceBase<ST> > gT_space = Thyra::createVectorSpace<ST>(mapT);
  return gT_space;*/
}


Teuchos::RCP<const Teuchos::Array<std::string> >
LCM::SchwarzMultiscale::get_p_names(int l) const
{
  //FIXME: fill in!
/*
  TEUCHOS_TEST_FOR_EXCEPTION(
    l >= num_param_vecs + num_dist_param_vecs || l < 0,
    Teuchos::Exceptions::InvalidParameter,
    std::endl <<
    "Error!  LCM::SchwarzMultiscale::get_p_names():  " <<
    "Invalid parameter index l = " << l << std::endl);

  if (l < num_param_vecs)
    return param_names[l];
  return Teuchos::rcp(new Teuchos::Array<std::string>(1, dist_param_names[l-num_param_vecs]));
*/
}


Thyra::ModelEvaluatorBase::InArgs<ST>
LCM::SchwarzMultiscale::getNominalValues() const
{
  //IK, 2/11/15: this function is done!
  return nominal_values_;
}


Thyra::ModelEvaluatorBase::InArgs<ST>
LCM::SchwarzMultiscale::getLowerBounds() const
{
  //IK, 2/10/15: this function is done!
  return Thyra::ModelEvaluatorBase::InArgs<ST>(); // Default value
}


Thyra::ModelEvaluatorBase::InArgs<ST>
LCM::SchwarzMultiscale::getUpperBounds() const
{
  //IK, 2/10/15: this function is done!
  return Thyra::ModelEvaluatorBase::InArgs<ST>(); // Default value
}


Teuchos::RCP<Thyra::LinearOpBase<ST> >
LCM::SchwarzMultiscale::create_W_op() const
{
  //FIXME: fill in!
  //Here we'll need to create the Jacobian for the coupled system.
  /*const Teuchos::RCP<Tpetra_Operator> W =
    Teuchos::rcp(new Tpetra_CrsMatrix(app->getJacobianGraphT()));
  return Thyra::createLinearOp(W);
  */
}

Teuchos::RCP<Thyra::PreconditionerBase<ST> >
LCM::SchwarzMultiscale::create_W_prec() const
{
  //IK, 2/10/15: this function is done for now...
  //Analog of EpetraExt::ModelEvaluator::Preconditioner does not exist in Thyra yet!  
  //So problem will run for now with no preconditioner...
  const bool W_prec_not_supported = true;
  TEUCHOS_TEST_FOR_EXCEPT(W_prec_not_supported);
  return Teuchos::null;
}

Teuchos::RCP<Thyra::LinearOpBase<ST> >
LCM::SchwarzMultiscale::create_DfDp_op_impl(int j) const
{
  //FIXME: fill in!
  /*TEUCHOS_TEST_FOR_EXCEPTION(
    j >= num_param_vecs+num_dist_param_vecs || j < num_param_vecs,
    Teuchos::Exceptions::InvalidParameter,
    std::endl <<
    "Error!  LCM::SchwarzMultiscale::create_DfDp_op_impl():  " <<
    "Invalid parameter index j = " << j << std::endl);

  const Teuchos::RCP<Tpetra_Operator> DfDp = Teuchos::rcp(new DistributedParameterDerivativeOpT(
                      app, dist_param_names[j-num_param_vecs]));

  return Thyra::createLinearOp(DfDp); */
}


Teuchos::RCP<const Thyra::LinearOpWithSolveFactoryBase<ST> >
LCM::SchwarzMultiscale::get_W_factory() const
{
  //IK, 2/10/15: this function is done!
  return Teuchos::null;
}


Thyra::ModelEvaluatorBase::InArgs<ST>
LCM::SchwarzMultiscale::createInArgs() const
{
  //IK, 2/11/15: this function is done! 
  return this->createInArgsImpl();

}


void
LCM::SchwarzMultiscale::reportFinalPoint(
    const Thyra::ModelEvaluatorBase::InArgs<ST>& finalPoint,
    const bool wasSolved)
{
  //IK, 2/11/15: this function is done! 
  TEUCHOS_TEST_FOR_EXCEPTION(true,
     Teuchos::Exceptions::InvalidParameter,
     "Calling reportFinalPoint in CoupledSchwarz.cpp" << std::endl);
}

void
LCM::SchwarzMultiscale::
allocateVectors()
{
  //FIXME: is this function necessary? 
}


/// Create operator form of dg/dx for distributed responses
Teuchos::RCP<Thyra::LinearOpBase<ST> >
LCM::SchwarzMultiscale::
create_DgDx_op_impl(int j) const
{
  //FIXME: fill in!
}

/// Create operator form of dg/dx_dot for distributed responses
Teuchos::RCP<Thyra::LinearOpBase<ST> >
LCM::SchwarzMultiscale::
create_DgDx_dot_op_impl(int j) const
{
  //FIXME: fill in!
}

/// Create OutArgs
Thyra::ModelEvaluatorBase::OutArgs<ST>
LCM::SchwarzMultiscale::
createOutArgsImpl() const
{
  //FIXME: fill in!
}

/// Evaluate model on InArgs
void 
LCM::SchwarzMultiscale::
evalModelImpl(Thyra::ModelEvaluatorBase::InArgs<ST> const & in_args,
      Thyra::ModelEvaluatorBase::OutArgs<ST> const & out_args) const 
{
  //FIXME: fill in!
}

//Copied from QCAD::CoupledPoissonSchrodinger -- used to validate applicaton parameters of applications not created via a SolverFactory
//Check usage and whether necessary...
Teuchos::RCP<const Teuchos::ParameterList>
LCM::SchwarzMultiscale::
getValidAppParameters() const
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

  return validPL;
}

//Copied from QCAD::CoupledPoissonSchrodinger
//Check usage and whether neessary...  
Teuchos::RCP<const Teuchos::ParameterList> 
LCM::SchwarzMultiscale::
getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL = Teuchos::createParameterList("ValidCoupledSchwarzProblemParams");

  validPL->set<std::string>("Name", "", "String to designate Problem Class");
  validPL->set<int>("Phalanx Graph Visualization Detail", 0, "Flag to select output of Phalanx Graph and level of detail");
  //FIXME: anything else to validate? 
  validPL->set<std::string>("Solution Method", "Steady", "Flag for Steady, Transient, or Continuation");
  
  return validPL;
}
