//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "SchwarzMultiscale.hpp"
#include "Albany_SolverFactory.hpp"
#include "Albany_ModelFactory.hpp"
#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"

//uncomment the following to write stuff out to matrix market to debug
#define WRITE_TO_MATRIX_MARKET

//string for storing name of first problem, for error checking
std::string problem_name0;

LCM::
SchwarzMultiscale::
SchwarzMultiscale(Teuchos::RCP<Teuchos::ParameterList> const & app_params,
    Teuchos::RCP<Teuchos::Comm<int> const> const & commT,
    Teuchos::RCP<Tpetra_Vector const> const & initial_guessT)
{
  std::cout << "Initializing Schwarz Multiscale constructor!\n";
  ;

  commT_ = commT;

  //IK, 2/11/15: I am assuming for now we don't have any distributed parameters.
  //TODO: Check with Alejandro.
  num_dist_params_total_ = 0;

  // Get "Coupled Schwarz" parameter sublist
  Teuchos::ParameterList &
  coupled_system_params = app_params->sublist("Coupled System");

  // Get names of individual model xml input files from problem parameterlist
  Teuchos::Array<std::string>
  model_filenames =
      coupled_system_params.get<Teuchos::Array<std::string> >(
          "Model XML Files");

  //number of models
  num_models_ = model_filenames.size();

  std::cout << "DEBUG: num_models_: " << num_models_ << '\n';

  apps_.resize(num_models_);
  models_.resize(num_models_);

  Teuchos::Array<Teuchos::RCP<Teuchos::ParameterList> >
  model_app_params(num_models_);

  Teuchos::Array<Teuchos::RCP<Teuchos::ParameterList> >
  model_problem_params(num_models_);

  Teuchos::Array<Teuchos::RCP<Tpetra_Map const> >
  disc_maps(num_models_);

  Teuchos::Array<Teuchos::RCP<Tpetra_Map const> >
  disc_overlap_maps(num_models_);

  material_dbs_.resize(num_models_);
  

  //Set up each application and model object in Teuchos::Array
  //(similar logic to that in Albany::SolverFactory::createAlbanyAppAndModelT)
  for (int m = 0; m < num_models_; ++m) {

    //get parameterlist from mth model *.xml file
    Albany::SolverFactory
    solver_factory(model_filenames[m], commT_);

    Teuchos::ParameterList &
    app_params_m = solver_factory.getParameters();

    model_app_params[m] = Teuchos::rcp(&(app_params_m), false);

    Teuchos::RCP<Teuchos::ParameterList>
    problem_params_m = Teuchos::sublist(model_app_params[m], "Problem");

    model_problem_params[m] = problem_params_m;

    std::string &
    problem_name = problem_params_m->get("Name", "");
    std::cout << "Name of problem #" << m << ": " << problem_name << '\n';

    if (m == 0) problem_name0 = problem_params_m->get("Name", "");

    if (problem_name0.compare(problem_name)) {
      std::cerr << "\nError in LCM::CoupledSchwarz constructor: ";
      std::cerr << "attempting go couple different models ";
      std::cerr << problem_name0 << " and " << problem_name << "!\n\n";
      exit(1);
      //FIXME: the above is not a very elegant way to exit,
      // but somehow the below line using Teuchos
      // exceptions does not seem to work...
      /*
       TEUCHOS_TEST_FOR_EXCEPTION(
       true, std::runtime_error,
       std::endl << "Error in LCM::CoupledSchwarz constructor:  " <<
       "attempting go couple different models " << problem_name0 << " and " <<
       problem_name << std::endl);
       */
    }

    std::ostringstream
    oss("materials");

    oss << '_' << m << ".xml";

    std::string
    matdb_filename(oss.str());

    if (problem_params_m->isType<std::string>("MaterialDB Filename")) {
      matdb_filename =
          problem_params_m->get<std::string>("MaterialDB Filename");
    }

    material_dbs_[m] =
        Teuchos::rcp(new QCAD::MaterialDatabase(matdb_filename, commT_));

    std::cout << "Materials #" << m << ": " << matdb_filename << '\n';

    //create application for mth model
    //FIXME: initial_guessT needs to be made the right one for the mth model!
    // Or can it be null?
    apps_[m] = Teuchos::rcp(
        new Albany::Application(commT, model_app_params[m], initial_guessT));

    //Create model evaluator
    Albany::ModelFactory
    model_factory(model_app_params[m], apps_[m]);

    models_[m] = model_factory.createT();
  }

  std::cout << "Finished creating Albany apps and models!\n";

  //Now get maps, InArgs, OutArgs for each model.
  //Calculate how many parameters, responses there are in total.
  solver_inargs_.resize(num_models_);
  solver_outargs_.resize(num_models_);
  num_params_.resize(num_models_);
  num_responses_.resize(num_models_);
  num_responses_partial_sum_.resize(num_models_); 
  num_params_total_ = 0;
  num_responses_total_ = 0;

  for (int m = 0; m < num_models_; ++m) {
    disc_maps[m] = apps_[m]->getMapT();

    disc_overlap_maps[m] =
        apps_[m]->getStateMgr().getDiscretization()->getOverlapMapT();

    solver_inargs_[m] = models_[m]->createInArgs();

    solver_outargs_[m] = models_[m]->createOutArgs();

    num_params_[m] = solver_inargs_[m].Np();

    num_responses_[m] = solver_outargs_[m].Ng();

    if (m == 0) num_responses_partial_sum_[m] = num_responses_[m]; 
    else num_responses_partial_sum_[m] = num_responses_partial_sum_[m-1] + num_responses_[m]; 

    //Does it make sense for num_params_total and num_responses_total to be
    //the sum of these values for each model?  I guess so.
    num_params_total_ += num_params_[m];

    num_responses_total_ += num_responses_[m];
  }

  std::cout << "Total # parameters, responses: " << num_params_total_;
  std::cout << ", " << num_responses_total_ << '\n';
 
  //Set coupled parameter names array, for get_p_names by concatenating parameter names
  //from each model.  
  coupled_param_names_.resize(num_params_total_ + num_dist_params_total_);
  int counter = 0; 
  for (int m=0; m<num_models_; m++) {
    for (int l=0; l<num_params_[m]; l++) {
      int num_parameters = models_[m]->get_p_names(l)->size();
      coupled_param_names_[l+counter] = Teuchos::rcp(new Teuchos::Array<std::string>(num_parameters));
      for (int j=0; j<num_parameters; j++) {
        (*coupled_param_names_[l+counter])[j] = (*models_[m]->get_p_names(l))[j]; 
        
      } 
      counter += num_parameters; 
    }
  }
  /*for (int m=0; m<n_models_; m++) {
    for (int i=0; i<num_params_[m]; i++) {

    }
  }*/

  //Create ccoupled_disc_map, a map for the entire coupled ME solution,
  //created from the entries of the disc_maps array (individual maps).
  coupled_disc_map_ = createCoupledMap(disc_maps, commT_);

  std::cout << "LCM::CoupledSchwarz constructor DEBUG: created coupled map!\n";

#ifdef WRITE_TO_MATRIX_MARKET
  // For debug, write the coupled map to matrix market file to
  // look at in vi or matlab
  Tpetra_MatrixMarket_Writer::writeMapFile(
      "coupled_disc_map.mm",
      *coupled_disc_map_);
#endif

  //IKT, 2/13/15: commenting out the following for now as it'll cause a seg fault due 
  //to some things not being allocated yet.  This is to facilitate implementation of 
  //other things, like parameter list validation.
  // Setup nominal values
 /* {
    nominal_values_ = this->createInArgsImpl();

    // All the ME vectors are allocated/unallocated here
    // Calling allocateVectors() will set x and x_dot in nominal_values_
    allocateVectors();

    coupled_sacado_param_vec_.resize(num_params_total_);
    coupled_param_map_.resize(num_params_total_);
    coupled_param_vec_.resize(num_params_total_);

    /// Create sacado parameter vector for coupled problem
    // This is for setting p_init in nominal_values_
    // First get each model's parameter vector and put them in an array
    Teuchos::Array<Teuchos::Array<ParamVec> >
    sacado_param_vec_array(num_models_);

    for (int m = 0; m < num_models_; ++m) {

      Teuchos::Array<ParamVec>
      sacado_param_vec_m;

      sacado_param_vec_m.resize(num_params_[m]);

      for (int i = 0; i < solver_inargs_[m].Np(); ++i) {
        Teuchos::RCP<Tpetra_Vector const>
        p = ConverterT::getConstTpetraVector(solver_inargs_[m].get_p(i));

        Teuchos::ArrayRCP<const ST> p_constView = p->get1dView();
        if (p != Teuchos::null) {
          for (unsigned int j = 0; j < sacado_param_vec_m[i].size(); j++)
            sacado_param_vec_m[i][j].baseValue = p_constView[j];
        }
      }
      sacado_param_vec_array[m] = sacado_param_vec_m;
    }
    //FIXME: populate coupled_sacado_param_vec_,
    //the parameter vec for the coupled model
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
*/
  //FIXME: Add discretization parameterlist and discretization object
  //for the "combined" solution vector from all the coupled Model
  //Evaluators.  Refer to QCAD_CoupledPoissonSchrodinger.cpp.

  //FIXME: How are we going to collect output?  Write exodus files for
  //each model evaluator?  Joined exodus file?

}

LCM::SchwarzMultiscale::~SchwarzMultiscale()
{
}

Teuchos::RCP<const Tpetra_Map>
LCM::SchwarzMultiscale::createCoupledMap(
    Teuchos::Array<Teuchos::RCP<Tpetra_Map const> > maps,
    Teuchos::RCP<Teuchos::Comm<int> const> const & commT)
{
  int
  n_maps = maps.size();

  std::cout << "DEBUG: LCM::SchwarzMultiscale::createCoupledMap, # maps: ";
  std::cout << n_maps << '\n';

  // Figure out how many local and global elements are in the
  // coupled map by summing these quantities over each model's map.
  LO
  local_num_elements = 0;

  GO
  global_num_elements = 0;

  for (int m = 0; m < n_maps; ++m) {
    local_num_elements += maps[m]->getNodeNumElements();
    global_num_elements += maps[m]->getGlobalNumElements();

    std::cout << "DEBUG: map #" << m << " has ";
    std::cout << maps[m]->getGlobalNumElements() << " global elements.\n";
  }
  //Create global element indices array for coupled map for this processor,
  //to be used to create the coupled map.
  GO *
  my_global_elements = new GO[local_num_elements];

  LO
  counter_local = 0;

  GO
  counter_global = 0;

  for (int m = 0; m < n_maps; ++m) {
    LO disc_nMyElements = maps[m]->getNodeNumElements();
    GO disc_nGlobalElements = maps[m]->getGlobalNumElements();
    Teuchos::ArrayView<const GO> disc_global_elements = maps[m]
        ->getNodeElementList();
    for (int l = 0; l < disc_nMyElements; l++) {
      my_global_elements[counter_local + l] = counter_global
          + disc_global_elements[l];
    }
    counter_local += disc_nMyElements;
    counter_global += disc_nGlobalElements;
  }
  const Teuchos::ArrayView<GO> my_global_elements_AV = Teuchos::arrayView(
      my_global_elements,
      local_num_elements);
  std::cout << "DEBUG: coupled map has " << global_num_elements
      << " global elements." << std::endl;
  Teuchos::RCP<const Tpetra_Map> coupled_map = Teuchos::rcp(
      new Tpetra_Map(global_num_elements, my_global_elements_AV, 0, commT_));
  return coupled_map;
}

// Overridden from Thyra::ModelEvaluator<ST>
Teuchos::RCP<Thyra::VectorSpaceBase<ST> const>
LCM::SchwarzMultiscale::get_x_space() const
{
  //IK, 2/10/15: this function is done!
  std::cout << "DEBUG: In LCM::SchwarzMultiScale::get_x_space()!\n";

  Teuchos::RCP<Tpetra_Map const>
  map = Teuchos::rcp(new (Tpetra_Map const)(*coupled_disc_map_));

  Teuchos::RCP<Thyra::VectorSpaceBase<ST> const>
  coupled_x_space = Thyra::createVectorSpace<ST>(map);

  return coupled_x_space;
}

Teuchos::RCP<Thyra::VectorSpaceBase<ST> const>
LCM::SchwarzMultiscale::get_f_space() const
{
  //IK, 2/10/15: this function is done!
  std::cout << "DEBUG: In LCM::SchwarzMultiScale::get_f_space()!\n";

  Teuchos::RCP<Tpetra_Map const>
  map = Teuchos::rcp(new (Tpetra_Map const)(*coupled_disc_map_));
  Teuchos::RCP<Thyra::VectorSpaceBase<ST> const> coupled_f_space =
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
  TEUCHOS_TEST_FOR_EXCEPTION(
   l >= num_responses_total_  || l < 0,
   Teuchos::Exceptions::InvalidParameter,
   std::endl <<
   "Error!  LCM::SchwarzMultiscale::get_g_space():  " <<
   "Invalid response index l = " << l << std::endl);

   if (l <= num_responses_partial_sum_[0]) 
     return models_[0]->get_g_space(l); 
   else {
     for (int m=1; m<num_models_; m++){
       if (l > num_responses_partial_sum_[m-1] && l <= num_responses_partial_sum_[m])
         return models_[m]->get_g_space(l - num_responses_partial_sum_[m-1]); 
     }
   } 
}

Teuchos::RCP<const Teuchos::Array<std::string> >
LCM::SchwarzMultiscale::get_p_names(int l) const
    {
   TEUCHOS_TEST_FOR_EXCEPTION(
   l >= num_params_total_ + num_dist_params_total_ || l < 0,
   Teuchos::Exceptions::InvalidParameter,
   std::endl <<
   "Error!  LCM::SchwarzMultiscale::get_p_names():  " <<
   "Invalid parameter index l = " << l << std::endl);

   return coupled_param_names_[l];
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

  //Analog of EpetraExt::ModelEvaluator::Preconditioner does not exist
  //in Thyra yet!  So problem will run for now with no
  //preconditioner...
  const bool W_prec_not_supported = true;
  TEUCHOS_TEST_FOR_EXCEPT(W_prec_not_supported);
  return Teuchos::null;
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
  //In this function, we create and set x_init and x_dot_init in
  //nominal_values_ for the coupled model.

  //Create Teuchos::Arrays of hte x_init and x_dot init Tpetra_Vectors
  //for each of the models
  Teuchos::Array<Teuchos::RCP<Tpetra_Vector const> > x_inits(num_models_);
  Teuchos::Array<Teuchos::RCP<Tpetra_Vector const> > x_dot_inits(num_models_);

  //Populate the arrays with the x_init and x_dot_init for each model.
  for (int m = 0; m < num_models_; m++) {
    x_inits[m] = apps_[m]->getInitialSolutionT();
    x_dot_inits[m] = apps_[m]->getInitialSolutionDotT();
  }

  // Create Tpetra objects to be wrapped in Thyra for coupled model
  //FIXME: implement by concatenating individual x_inits
  const Teuchos::RCP<Tpetra_Vector const> coupled_x_init;
  //FIXME: implement by concatenating individual x_dot_inits
  const Teuchos::RCP<Tpetra_Vector const> coupled_x_dot_init;
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

//Copied from QCAD::CoupledPoissonSchrodinger -- used to validate
//applicaton parameters of applications not created via a
//SolverFactory Check usage and whether necessary...
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
