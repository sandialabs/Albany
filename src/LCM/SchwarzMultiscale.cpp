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
SchwarzMultiscale(Teuchos::RCP<Teuchos::ParameterList> const & app_params,
    Teuchos::RCP<Teuchos::Comm<int> const> const & commT,
    Teuchos::RCP<Tpetra_Vector const> const & initial_guessT)
{
  std::cout << "Initializing Schwarz Multiscale constructor!" << std::endl;
  commT_ = commT;
  //IK, 2/11/15: I am assuming for now we don't have any distributed parameters.
  //TODO: Check with Alejandro.
  num_dist_params_total_ = 0;
  // Get "Coupled Schwarz" parameter sublist
  Teuchos::ParameterList& coupledSystemParams = app_params->sublist(
      "Coupled System");
  // Get names of individual model xml input files from problem parameterlist
  Teuchos::Array<std::string> model_filenames =
      coupledSystemParams.get<Teuchos::Array<std::string> >("Model XML Files");
  //number of models
  num_models_ = model_filenames.size();
  std::cout << "DEBUG: n_models_: " << num_models_ << std::endl;
  apps_.resize(num_models_);
  models_.resize(num_models_);
  Teuchos::Array<Teuchos::RCP<Teuchos::ParameterList> > modelAppParams(
      num_models_);
  Teuchos::Array<Teuchos::RCP<Teuchos::ParameterList> > modelProblemParams(
      num_models_);
  Teuchos::Array<Teuchos::RCP<const Tpetra_Map> > disc_maps(num_models_);
  Teuchos::Array<Teuchos::RCP<const Tpetra_Map> > disc_overlap_maps(
      num_models_);
  material_dbs_.resize(num_models_);
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
  for (int m = 0; m < num_models_; m++) {
    //get parameterlist from mth model *.xml file 
    Albany::SolverFactory slvrfctry(model_filenames[m], commT_);
    Teuchos::ParameterList& appParams_m = slvrfctry.getParameters();
    modelAppParams[m] = Teuchos::rcp(&(appParams_m));
    Teuchos::RCP<Teuchos::ParameterList> problemParams_m = Teuchos::sublist(
        modelAppParams[m],
        "Problem");
    modelProblemParams[m] = problemParams_m;
    std::string &problem_name = problemParams_m->get("Name", "");
    //FIXME: fix the following line to material name gets incremented
    //sprintf(mtrDbFilename, "materials%i.xml", m);
    if (problemParams_m->isType<std::string>("MaterialDB Filename"))
      mtrDbFilename = problemParams_m->get<std::string>("MaterialDB Filename");
    material_dbs_[m] = Teuchos::rcp(
        new QCAD::MaterialDatabase(mtrDbFilename, commT_));
    //FIXME: should we throw an error if the problem names in the input files don't match??
    std::cout << "DEBUG: name of problem #" << m << ": " << problem_name
        << std::endl;
    std::cout << "DEBUG: material of problem #" << m << ": " << mtrDbFilename
        << std::endl;
    //create application for mth model 
    //FIXME: initial_guessT needs to be made the right one for the mth model!  Or can it be null?  
    apps_[m] = Teuchos::rcp(
        new Albany::Application(commT, modelAppParams[m], initial_guessT));
    //Validate parameter lists
    //FIXME: add relevant things to validate to getValid* functions below
    //Uncomment
    //problemParams_m->validateParameters(*getValidProblemParameters(),0); 
    //problemParams_m->sublist("Parameters").validateParameters(*validParameterParams, 0);
    //problemParams_m->sublist("Response Functions").validateParameters(*validResponseParams, 0);
    //Create model evaluator
    Albany::ModelFactory modelFactory(modelAppParams[m], apps_[m]);
    models_[m] = modelFactory.createT();
  }
  std::cout << "Finished creating Albany apps_ and models!" << std::endl;

  //Now get maps, InArgs, OutArgs for each model.
  //Calculate how many parameters, responses there are total.
  solver_inargs_.resize(num_models_);
  solver_outargs_.resize(num_models_);
  num_params_.resize(num_models_);
  num_responses_.resize(num_models_);
  num_params_total_ = 0;
  num_responses_total_ = 0;
  for (int m = 0; m < num_models_; m++) {
    disc_maps[m] = apps_[m]->getMapT();
    disc_overlap_maps[m] = apps_[m]->getStateMgr().getDiscretization()
        ->getOverlapMapT();
    solver_inargs_[m] = models_[m]->createInArgs();
    solver_outargs_[m] = models_[m]->createOutArgs();
    num_params_[m] = solver_inargs_[m].Np();
    num_responses_[m] = solver_outargs_[m].Ng();
    //Does it make sense for num_params_total and num_responses_total to be the sum 
    //of these values for each model?  I guess so. 
    num_params_total_ += num_params_[m];
    num_responses_total_ += num_responses_[m];
  }

  std::cout << "Total # parameters, responses: " << num_params_total_ << ", "
      << num_responses_total_ << std::endl;

  // Setup nominal values
  {
    nominal_values_ = this->createInArgsImpl();

    // All the ME vectors are allocated/unallocated here
    // Calling allocateVectors() will set x and x_dot in nominal_values_
    allocateVectors();

    coupled_sacado_param_vec_.resize(num_params_total_);
    coupled_param_map_.resize(num_params_total_);
    coupled_param_vec_.resize(num_params_total_);
    //! Create sacado parameter vector for coupled problem
    //This is for setting p_init in nominal_values_
    //First get each model's parameter vector and put them in an array 
    Teuchos::Array<Teuchos::Array<ParamVec> > sacado_param_vec_array(
        num_models_);
    for (int m = 0; m < num_models_; m++) {
      Teuchos::Array<ParamVec> sacado_param_vec_m;
      sacado_param_vec_m.resize(num_params_[m]);
      for (int i = 0; i < solver_inargs_[m].Np(); i++) {
        Teuchos::RCP<const Tpetra_Vector> p = ConverterT::getConstTpetraVector(
            solver_inargs_[m].get_p(i));
        Teuchos::ArrayRCP<const ST> p_constView = p->get1dView();
        if (p != Teuchos::null) {
          for (unsigned int j = 0; j < sacado_param_vec_m[i].size(); j++)
            sacado_param_vec_m[i][j].baseValue = p_constView[j];
        }
      }
      sacado_param_vec_array[m] = sacado_param_vec_m;
    }
    //FIXME: populate coupled_sacado_param_vec_, the parameter vec for the coupled model
    //coupled_sacado_param_vec_ = ... 

    //Create Tpetra map and Tpetra vectors for coupled parameters
    //TODO: check with Alejandro that this makes sense for the parameters 
    Tpetra::LocalGlobal lg = Tpetra::LocallyReplicated;
    for (int l = 0; l < coupled_sacado_param_vec_.size(); ++l) {
      coupled_param_map_[l] = Teuchos::rcp(
          new Tpetra_Map(coupled_sacado_param_vec_[l].size(), 0, commT_, lg));
      coupled_param_vec_[l] = Teuchos::rcp(
          new Tpetra_Vector(coupled_param_map_[l]));
      for (unsigned int k = 0; k < coupled_sacado_param_vec_[l].size(); ++k) {
        const Teuchos::ArrayRCP<ST> coupled_param_vec_nonConstView =
            coupled_param_vec_[l]->get1dViewNonConst();
        coupled_param_vec_nonConstView[k] = coupled_sacado_param_vec_[l][k]
            .baseValue;
      }
    }

    // TODO: Check if correct nominal values for parameters
    for (int l = 0; l < num_params_total_; ++l) {
      Teuchos::RCP<const Tpetra_Map> map = coupled_param_map_[l];
      Teuchos::RCP<const Thyra::VectorSpaceBase<ST> > coupled_param_space =
          Thyra::createVectorSpace<ST>(map);
      nominal_values_.set_p(
          l,
          Thyra::createVector(coupled_param_vec_[l], coupled_param_space));
    }
  } //end setting of nominal values

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
  Teuchos::RCP<const Tpetra_Map> map = Teuchos::rcp(
      new const Tpetra_Map(*coupled_disc_map_));
  Teuchos::RCP<const Thyra::VectorSpaceBase<ST> > coupled_x_space =
      Thyra::createVectorSpace<ST>(map);
  return coupled_x_space;
}

Teuchos::RCP<const Thyra::VectorSpaceBase<ST> >
LCM::SchwarzMultiscale::get_f_space() const
{
  //IK, 2/10/15: this function is done!
  std::cout << "DEBUG: In LCM::SchwarzMultiScale::get_f_space()!" << std::endl;
  Teuchos::RCP<const Tpetra_Map> map = Teuchos::rcp(
      new const Tpetra_Map(*coupled_disc_map_));
  Teuchos::RCP<const Thyra::VectorSpaceBase<ST> > coupled_f_space =
      Thyra::createVectorSpace<ST>(map);
  return coupled_f_space;
}

Teuchos::RCP<const Thyra::VectorSpaceBase<ST> >
LCM::SchwarzMultiscale::get_p_space(int l) const
    {
  TEUCHOS_TEST_FOR_EXCEPTION(
      l >= num_params_total_ + num_dist_params_total_ < 0,
      Teuchos::Exceptions::InvalidParameter,
      std::endl <<
      "Error!  LCM::SchwarzMultiscale::get_p_space():  " <<
      "Invalid parameter index l = " << l << std::endl);
  Teuchos::RCP<const Tpetra_Map> map;
  if (l < num_params_total_)
    map = coupled_param_map_[l];
  //IK, 7/1/14: commenting this out for now
  //map = distParamLib->get(dist_param_names[l-num_param_vecs])->map(); 
  Teuchos::RCP<const Thyra::VectorSpaceBase<ST> > coupled_param_space =
      Thyra::createVectorSpace<ST>(map);
  return coupled_param_space;
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
  //In this function, we create and set x_init and x_dot_init in nominal_values_ for the coupled model.

  //Create Teuchos::Arrays of hte x_init and x_dot init Tpetra_Vectors for each of the models 
  Teuchos::Array<Teuchos::RCP<const Tpetra_Vector> > x_inits(num_models_);
  Teuchos::Array<Teuchos::RCP<const Tpetra_Vector> > x_dot_inits(num_models_);

  //Populate the arrays with the x_init and x_dot_init for each model.
  for (int m = 0; m < num_models_; m++) {
    x_inits[m] = apps_[m]->getInitialSolutionT();
    x_dot_inits[m] = apps_[m]->getInitialSolutionDotT();
  }

  // Create Tpetra objects to be wrapped in Thyra for coupled model
  const Teuchos::RCP<const Tpetra_Vector> coupled_x_init; //FIXME: implement by concatenating individual x_inits 
  const Teuchos::RCP<const Tpetra_Vector> coupled_x_dot_init; //FIXME: implement by concatenating individual x_dot_inits
  Teuchos::RCP<const Tpetra_Map> map = Teuchos::rcp(
      new const Tpetra_Map(*coupled_disc_map_));
  Teuchos::RCP<const Thyra::VectorSpaceBase<ST> > coupled_x_space =
      Thyra::createVectorSpace<ST>(map);

  // Create non-const versions of xT_init and x_dotT_init
  const Teuchos::RCP<Tpetra_Vector> coupled_x_init_nonconst = Teuchos::rcp(
      new Tpetra_Vector(*coupled_x_init));
  const Teuchos::RCP<Tpetra_Vector> coupled_x_dot_init_nonconst = Teuchos::rcp(
      new Tpetra_Vector(*coupled_x_dot_init));

  nominal_values_.set_x(
      Thyra::createVector(coupled_x_init_nonconst, coupled_x_space));
  nominal_values_.set_x_dot(
      Thyra::createVector(coupled_x_dot_init_nonconst, coupled_x_space));

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

Thyra::ModelEvaluatorBase::InArgs<ST>
LCM::SchwarzMultiscale::
createInArgsImpl() const
{
  Thyra::ModelEvaluatorBase::InArgsSetup<ST> result;
  result.setModelEvalDescription(this->description());

  result.setSupports(Thyra::ModelEvaluatorBase::IN_ARG_x, true);

  result.setSupports(Thyra::ModelEvaluatorBase::IN_ARG_x_dot, true);
  // AGS: x_dotdot time integrators not imlemented in Thyra ME yet
  //result.setSupports(Thyra::ModelEvaluatorBase::IN_ARG_x_dotdot, true);
  result.setSupports(Thyra::ModelEvaluatorBase::IN_ARG_t, true);
  result.setSupports(Thyra::ModelEvaluatorBase::IN_ARG_alpha, true);
  result.setSupports(Thyra::ModelEvaluatorBase::IN_ARG_beta, true);
  // AGS: x_dotdot time integrators not imlemented in Thyra ME yet
  //result.setSupports(Thyra::ModelEvaluatorBase::IN_ARG_omega, true);

  result.set_Np(num_params_total_ + num_dist_params_total_);

  return result;
}

//Copied from QCAD::CoupledPoissonSchrodinger -- used to validate applicaton parameters of applications not created via a SolverFactory
//Check usage and whether necessary...
Teuchos::RCP<const Teuchos::ParameterList>
LCM::SchwarzMultiscale::
getValidAppParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL = Teuchos::rcp(
      new Teuchos::ParameterList("ValidAppParams"));
  ;
  validPL->sublist("Problem", false, "Problem sublist");
  validPL->sublist("Debug Output", false, "Debug Output sublist");
  validPL->sublist("Discretization", false, "Discretization sublist");
  validPL->sublist("Quadrature", false, "Quadrature sublist");
  validPL->sublist("Regression Results", false, "Regression Results sublist");
  validPL->sublist("VTK", false, "DEPRECATED  VTK sublist");
  validPL->sublist("Piro", false, "Piro sublist");
  validPL->sublist("Coupled System", false, "Coupled system sublist");

  return validPL;
}

//Copied from QCAD::CoupledPoissonSchrodinger
//Check usage and whether neessary...  
Teuchos::RCP<const Teuchos::ParameterList>
LCM::SchwarzMultiscale::
getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL = Teuchos::createParameterList(
      "ValidCoupledSchwarzProblemParams");

  validPL->set<std::string>("Name", "", "String to designate Problem Class");
  validPL->set<int>(
      "Phalanx Graph Visualization Detail",
      0,
      "Flag to select output of Phalanx Graph and level of detail");
  //FIXME: anything else to validate? 
  validPL->set<std::string>(
      "Solution Method",
      "Steady",
      "Flag for Steady, Transient, or Continuation");

  return validPL;
}
