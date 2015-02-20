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
#include "Schwarz_CoupledJacobian.hpp" 

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

  disc_maps_.resize(num_models_);

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
  num_params_partial_sum_.resize(num_models_);
  num_responses_.resize(num_models_);
  num_responses_partial_sum_.resize(num_models_);
  num_params_total_ = 0;
  num_responses_total_ = 0;

  for (int m = 0; m < num_models_; ++m) {
    disc_maps_[m] = apps_[m]->getMapT();

    disc_overlap_maps[m] =
        apps_[m]->getStateMgr().getDiscretization()->getOverlapMapT();

    solver_inargs_[m] = models_[m]->createInArgs();

    solver_outargs_[m] = models_[m]->createOutArgs();

    num_params_[m] = solver_inargs_[m].Np();

    num_responses_[m] = solver_outargs_[m].Ng();

    if (m == 0) {
      num_responses_partial_sum_[m] = num_responses_[m];
      num_params_partial_sum_[m] = num_params_[m];
    }
    else {
      num_responses_partial_sum_[m] = num_responses_partial_sum_[m - 1]
          + num_responses_[m];
      num_params_partial_sum_[m] = num_params_partial_sum_[m - 1]
          + num_params_[m];
    }

    //Does it make sense for num_params_total and num_responses_total to be
    //the sum of these values for each model?  I guess so.
    num_params_total_ += num_params_[m];

    num_responses_total_ += num_responses_[m];
  }

  // Create sacado parameter vectors of appropriate size for use in evalModelImpl
  sacado_param_vecs_.resize(num_models_);

  for (int m = 0; m < num_models_; ++m) {
    sacado_param_vecs_[m].resize(num_params_[m]);
  }

  std::cout << "Total # parameters, responses: " << num_params_total_;
  std::cout << ", " << num_responses_total_ << '\n';

  //Create ccoupled_disc_map, a map for the entire coupled ME solution,
  //created from the entries of the disc_maps array (individual maps).
  coupled_disc_map_ = createCoupledMap(disc_maps_, commT_);

  std::cout << "LCM::CoupledSchwarz constructor DEBUG: created coupled map!\n";

#ifdef WRITE_TO_MATRIX_MARKET
  // For debug, write the coupled map to matrix market file to
  // look at in vi or matlab
  Tpetra_MatrixMarket_Writer::writeMapFile(
      "coupled_disc_map.mm",
      *coupled_disc_map_);
#endif

  // Setup nominal values
  {
    nominal_values_ = this->createInArgsImpl();

    // All the ME vectors are allocated/unallocated here
    // Calling allocateVectors() will set x and x_dot in nominal_values_
    allocateVectors();

    //set p_init in nominal_values_  
    // TODO: Check if correct nominal values for parameters
    for (int l = 0; l < num_params_total_; ++l) {
      if (l < num_params_partial_sum_[0]) {
        nominal_values_.set_p(l, solver_inargs_[0].get_p(l));
      }
      else {
        for (int m = 1; m < num_models_; ++m) {
          if (l >= num_params_partial_sum_[m - 1]
              && l < num_params_partial_sum_[m]) {
            nominal_values_.set_p(
                l,
                solver_inargs_[m].get_p(l - num_params_partial_sum_[m - 1]));
          }
        }
      }
    }

  } //end setting of nominal values

  std::cout << "Set nominal_values_! \n";

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
    LO
    disc_nMyElements = maps[m]->getNodeNumElements();

    GO
    disc_nGlobalElements = maps[m]->getGlobalNumElements();

    Teuchos::ArrayView<const GO>
    disc_global_elements = maps[m]->getNodeElementList();

    for (int l = 0; l < disc_nMyElements; ++l) {
      my_global_elements[counter_local + l] = counter_global
          + disc_global_elements[l];
    }
    counter_local += disc_nMyElements;
    counter_global += disc_nGlobalElements;
  }

  Teuchos::ArrayView<GO> const
  my_global_elements_AV =
      Teuchos::arrayView(my_global_elements, local_num_elements);

  std::cout << "DEBUG: coupled map has " << global_num_elements;
  std::cout << " global elements.\n";

  Teuchos::RCP<Tpetra_Map const>
  coupled_map = Teuchos::rcp(
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

  Teuchos::RCP<Thyra::VectorSpaceBase<ST> const>
  coupled_f_space = Thyra::createVectorSpace<ST>(map);

  return coupled_f_space;
}

Teuchos::RCP<const Thyra::VectorSpaceBase<ST> >
LCM::SchwarzMultiscale::get_p_space(int l) const
{
  std::cout << "DEBUG: In LCM::SchwarzMultiScale::get_p_space()!\n";
  TEUCHOS_TEST_FOR_EXCEPTION(
      l >= num_params_total_ < 0,
      Teuchos::Exceptions::InvalidParameter,
      "\nError!  LCM::SchwarzMultiscale::get_p_space():  " <<
      "Invalid parameter index l = " << l << '\n');

  if (l < num_params_partial_sum_[0]) {
    return models_[0]->get_p_space(l);
  }
  else {
    for (int m = 1; m < num_models_; ++m) {
      if (l >= num_params_partial_sum_[m - 1]
          && l < num_params_partial_sum_[m]) {
        return models_[m]->get_p_space(l - num_params_partial_sum_[m - 1]);
      }
    }
  }
}

Teuchos::RCP<const Thyra::VectorSpaceBase<ST> >
LCM::SchwarzMultiscale::get_g_space(int l) const
{
  std::cout << "DEBUG: In LCM::SchwarzMultiScale::get_g_space()!\n";
  TEUCHOS_TEST_FOR_EXCEPTION(
      l >= num_responses_total_ || l < 0,
      Teuchos::Exceptions::InvalidParameter,
      std::endl <<
      "Error!  LCM::SchwarzMultiscale::get_g_space():  " <<
      "Invalid response index l = " << l << std::endl);

  if (l < num_responses_partial_sum_[0])
    return models_[0]->get_g_space(l);
  else {
    for (int m = 1; m < num_models_; m++) {
      if (l >= num_responses_partial_sum_[m - 1]
          && l < num_responses_partial_sum_[m])
        return models_[m]->get_g_space(l - num_responses_partial_sum_[m - 1]);
    }
  }
}

Teuchos::RCP<const Teuchos::Array<std::string> >
LCM::SchwarzMultiscale::get_p_names(int l) const
    {
  std::cout << "DEBUG: In LCM::SchwarzMultiScale::get_p_names()!\n";
  TEUCHOS_TEST_FOR_EXCEPTION(
      l >= num_params_total_ || l < 0,
      Teuchos::Exceptions::InvalidParameter,
      std::endl <<
      "Error!  LCM::SchwarzMultiscale::get_p_names():  " <<
      "Invalid parameter index l = " << l << std::endl);

  if (l < num_params_partial_sum_[0])
    return models_[0]->get_p_names(l);
  else {
    for (int m = 1; m < num_models_; m++) {
      if (l >= num_params_partial_sum_[m - 1]
          && l < num_params_partial_sum_[m]) {
        return models_[m]->get_p_names(l - num_params_partial_sum_[m - 1]);
      }
    }
  }
}

Thyra::ModelEvaluatorBase::InArgs<ST>
LCM::SchwarzMultiscale::getNominalValues() const
{
  //IK, 2/11/15: this function is done!
  std::cout << "DEBUG: In LCM::SchwarzMultiScale::getNominalValues()!\n";
  return nominal_values_;
}

Thyra::ModelEvaluatorBase::InArgs<ST>
LCM::SchwarzMultiscale::getLowerBounds() const
{
  //IK, 2/10/15: this function is done!
  std::cout << "DEBUG: In LCM::SchwarzMultiScale::getLowerBounds()!\n";
  return Thyra::ModelEvaluatorBase::InArgs<ST>(); // Default value
}

Thyra::ModelEvaluatorBase::InArgs<ST>
LCM::SchwarzMultiscale::getUpperBounds() const
{
  //IK, 2/10/15: this function is done!
  std::cout << "DEBUG: In LCM::SchwarzMultiScale::getUpperBounds()!\n";
  return Thyra::ModelEvaluatorBase::InArgs<ST>(); // Default value
}

Teuchos::RCP<Thyra::LinearOpBase<ST> >
LCM::SchwarzMultiscale::create_W_op() const
{
  std::cout << "DEBUG:  LCM::SchwarzMultizcale create_W_op() called! \n";
  const Teuchos::RCP<Tpetra_Operator> W_out_coupled = Teuchos::rcp(new LCM::Schwarz_CoupledJacobian(disc_maps_, coupled_disc_map_, commT_)); 
  return Thyra::createLinearOp(W_out_coupled);
}

Teuchos::RCP<Thyra::PreconditionerBase<ST> >
LCM::SchwarzMultiscale::create_W_prec() const
{
  std::cout << "DEBUG: In LCM::SchwarzMultiScale::create_W_prec()!\n";
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
  std::cout << "DEBUG: In LCM::SchwarzMultiScale::get_W_factory()!\n";
  //IK, 2/10/15: this function is done!
  return Teuchos::null;
}

Thyra::ModelEvaluatorBase::InArgs<ST>
LCM::SchwarzMultiscale::createInArgs() const
{
  std::cout << "DEBUG: In LCM::SchwarzMultiScale::createInArgs()!\n";
  //IK, 2/11/15: this function is done!
  return this->createInArgsImpl();
}

void
LCM::SchwarzMultiscale::reportFinalPoint(
    const Thyra::ModelEvaluatorBase::InArgs<ST>& finalPoint,
    const bool wasSolved)
{
  std::cout << "DEBUG: In LCM::SchwarzMultiScale::reportFinalPoint()!\n";
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
  std::cout << "DEBUG: In LCM::SchwarzMultiScale::allocateVectors()!\n";

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
  Teuchos::RCP<Tpetra_Vector> coupled_x_init = Teuchos::rcp(
      new Tpetra_Vector(coupled_disc_map_));
  //FIXME: implement by concatenating individual x_dot_inits
  Teuchos::RCP<Tpetra_Vector> coupled_x_dot_init = Teuchos::rcp(
      new Tpetra_Vector(coupled_disc_map_));

  //initialize coupled vecs to all 0s 
  coupled_x_init->putScalar(0.0);
  coupled_x_dot_init->putScalar(0.0);

  LO counter_local = 0;
  //get nonconst view of coupled_x_init & coupled_x_dot_init
  Teuchos::ArrayRCP<ST> coupled_x_init_nonconstView = coupled_x_init
      ->get1dViewNonConst();
  Teuchos::ArrayRCP<ST> coupled_x_dot_init_nonconstView = coupled_x_dot_init
      ->get1dViewNonConst();
  for (int m = 0; m < num_models_; m++) {
    //get const view of mth x_init & x_dot_init vector
    Teuchos::ArrayRCP<const ST> x_init_constView = x_inits[m]->get1dView();
    Teuchos::ArrayRCP<const ST> x_dot_init_constView =
        x_dot_inits[m]->get1dView();
    //The following assumes x_init and x_dot_init have same length.  
    //FIXME? Could have error checking to make sure. 
    for (int i = 0; i < x_inits[m]->getLocalLength(); i++) {
      coupled_x_init_nonconstView[counter_local + i] = x_init_constView[i];
      coupled_x_dot_init_nonconstView[counter_local + i] =
          x_dot_init_constView[i];
    }
    counter_local += x_inits[m]->getLocalLength();
  }

#ifdef WRITE_TO_MATRIX_MARKET
  //writing to MatrixMarket file for debug
  Tpetra_MatrixMarket_Writer::writeDenseFile("x_init0.mm", *(x_inits[0]));
  Tpetra_MatrixMarket_Writer::writeDenseFile(
      "x_dot_init0.mm",
      *(x_dot_inits[0]));
  if (num_models_ > 1) {
    Tpetra_MatrixMarket_Writer::writeDenseFile("x_init1.mm", *(x_inits[1]));
    Tpetra_MatrixMarket_Writer::writeDenseFile(
        "x_dot_init1.mm",
        *(x_dot_inits[1]));
  }
  Tpetra_MatrixMarket_Writer::writeDenseFile(
      "coupled_x_init.mm",
      *coupled_x_init);
  Tpetra_MatrixMarket_Writer::writeDenseFile(
      "coupled_x_dot_init.mm",
      *coupled_x_dot_init);
#endif

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
  std::cout << "DEBUG: In LCM::SchwarzMultiScale::create_DgDx_op_impl()!\n";
  TEUCHOS_TEST_FOR_EXCEPTION(
      j >= num_responses_total_ || j < 0,
      Teuchos::Exceptions::InvalidParameter,
      std::endl <<
      "Error!  LCM::SchwarzMultiscale::create_DgDx_op_impl():  " <<
      "Invalid response index j = " << j << std::endl);

  if (j < num_responses_partial_sum_[0])
    return Thyra::createLinearOp(apps_[0]->getResponse(j)->createGradientOpT());
  else {
    for (int m = 1; m < num_models_; m++) {
      if (j >= num_responses_partial_sum_[m - 1]
          && j < num_responses_partial_sum_[m])
        return Thyra::createLinearOp(
            apps_[m]->getResponse(j - num_responses_partial_sum_[m - 1])
                ->createGradientOpT());
    }
  }
}

/// Create operator form of dg/dx_dot for distributed responses
Teuchos::RCP<Thyra::LinearOpBase<ST> >
LCM::SchwarzMultiscale::
create_DgDx_dot_op_impl(int j) const
{
  std::cout << "DEBUG: In LCM::SchwarzMultiScale::create_DgDdx_dot_op_impl()!\n";
  TEUCHOS_TEST_FOR_EXCEPTION(
      j >= num_responses_total_ || j < 0,
      Teuchos::Exceptions::InvalidParameter,
      std::endl <<
      "Error!  LCM::SchwarzMultiscale::create_DgDx_dot_op():  " <<
      "Invalid response index j = " << j << std::endl);

  if (j < num_responses_partial_sum_[0])
    return Thyra::createLinearOp(apps_[0]->getResponse(j)->createGradientOpT());
  else {
    for (int m = 1; m < num_models_; m++) {
      if (j >= num_responses_partial_sum_[m - 1]
          && j < num_responses_partial_sum_[m])
        return Thyra::createLinearOp(
            apps_[m]->getResponse(j - num_responses_partial_sum_[m - 1])
                ->createGradientOpT());
    }
  }
}

/// Create OutArgs
Thyra::ModelEvaluatorBase::OutArgs<ST>
LCM::SchwarzMultiscale::
createOutArgsImpl() const
{
  std::cout << "DEBUG: In LCM::SchwarzMultiScale::createOutArgsImpl()!\n";
  Thyra::ModelEvaluatorBase::OutArgsSetup<ST> result;
  result.setModelEvalDescription(this->description());

  //Note: it is assumed her there are no distributed parameters.
  result.set_Np_Ng(num_params_total_, num_responses_total_);

  result.setSupports(Thyra::ModelEvaluatorBase::OUT_ARG_f, true);

  result.setSupports(Thyra::ModelEvaluatorBase::OUT_ARG_W_op, true);
  result.set_W_properties(
      Thyra::ModelEvaluatorBase::DerivativeProperties(
          Thyra::ModelEvaluatorBase::DERIV_LINEARITY_UNKNOWN,
          Thyra::ModelEvaluatorBase::DERIV_RANK_FULL,
          true));

  for (int l = 0; l < num_params_total_; ++l) {
    result.setSupports(
        Thyra::ModelEvaluatorBase::OUT_ARG_DfDp, l,
        Thyra::ModelEvaluatorBase::DERIV_MV_BY_COL);
  }
  for (int i = 0; i < num_responses_total_; ++i) {
    Thyra::ModelEvaluatorBase::DerivativeSupport dgdx_support;
    if (i < num_responses_partial_sum_[0]) {
      if (apps_[0]->getResponse(i)->isScalarResponse())
        dgdx_support = Thyra::ModelEvaluatorBase::DERIV_TRANS_MV_BY_ROW;
      else
        dgdx_support = Thyra::ModelEvaluatorBase::DERIV_LINEAR_OP;
    }
    else {
      for (int m = 1; m < num_models_; m++) {
        if (i >= num_responses_partial_sum_[m - 1]
            && i < num_responses_partial_sum_[m]) {
          if (apps_[m]->getResponse(i - num_responses_partial_sum_[m - 1])
              ->isScalarResponse())
            dgdx_support = Thyra::ModelEvaluatorBase::DERIV_TRANS_MV_BY_ROW;
          else
            dgdx_support = Thyra::ModelEvaluatorBase::DERIV_LINEAR_OP;
        }
      }
    }
    result.setSupports(
        Thyra::ModelEvaluatorBase::OUT_ARG_DgDx, i, dgdx_support);
    result.setSupports(
        Thyra::ModelEvaluatorBase::OUT_ARG_DgDx_dot, i, dgdx_support);
    // AGS: x_dotdot time integrators not imlemented in Thyra ME yet
    //result.setSupports(
    //    Thyra::ModelEvaluatorBase::OUT_ARG_DgDx_dotdot, i, dgdx_support);

    for (int l = 0; l < num_params_total_; l++)
      result.setSupports(
          Thyra::ModelEvaluatorBase::OUT_ARG_DgDp, i, l,
          Thyra::ModelEvaluatorBase::DERIV_MV_BY_COL);
  }
  return result;
}

/// Evaluate model on InArgs
void
LCM::SchwarzMultiscale::
evalModelImpl(Thyra::ModelEvaluatorBase::InArgs<ST> const & in_args,
    Thyra::ModelEvaluatorBase::OutArgs<ST> const & out_args) const
{
  //FIXME: finish filling in!
  
std::cout <<"DEBUG: in LCM::SchwarzMultiscale::evalModelImpl! \n"; 


  // Get the input arguments
  const Teuchos::RCP<const Tpetra_Vector> xT =
      ConverterT::getConstTpetraVector(in_args.get_x());

  const Teuchos::RCP<const Tpetra_Vector> x_dotT =
      Teuchos::nonnull(in_args.get_x_dot()) ?
          ConverterT::getConstTpetraVector(in_args.get_x_dot()) :
          Teuchos::null;

  //Create a Teuchos array of the xT and x_dotT for each model.
  Teuchos::Array<Teuchos::RCP<Tpetra_Vector> > xTs;
  Teuchos::Array<Teuchos::RCP<Tpetra_Vector> > x_dotTs;
  xTs.resize(num_models_);
  x_dotTs.resize(num_models_);
  for (int m = 0; m < num_models_; m++) {
    Teuchos::RCP<const Tpetra_Vector> xT_temp =
        ConverterT::getConstTpetraVector(solver_inargs_[m].get_x());
    xTs[m] = Teuchos::rcp(new Tpetra_Vector(*xT_temp));
    Teuchos::RCP<const Tpetra_Vector> x_dotT_temp =
        Teuchos::nonnull(solver_inargs_[m].get_x_dot()) ?
            ConverterT::getConstTpetraVector(solver_inargs_[m].get_x_dot()) :
            Teuchos::null;
    x_dotTs[m] = Teuchos::rcp(new Tpetra_Vector(*x_dotT_temp));
  }

  // AGS: x_dotdot time integrators not imlemented in Thyra ME yet
  //const Teuchos::RCP<const Tpetra_Vector> x_dotdotT =
  //  Teuchos::nonnull(in_args.get_x_dotdot()) ?
  //  ConverterT::getConstTpetraVector(in_args.get_x_dotdot()) :
  //  Teuchos::null;
  const Teuchos::RCP<const Tpetra_Vector> x_dotdotT = Teuchos::null;

  const double alpha =
      (Teuchos::nonnull(x_dotT) || Teuchos::nonnull(x_dotdotT)) ?
          in_args.get_alpha() : 0.0;
  // AGS: x_dotdot time integrators not imlemented in Thyra ME yet
  // const double omega = (Teuchos::nonnull(x_dotT) || Teuchos::nonnull(x_dotdotT)) ? in_args.get_omega() : 0.0;
  const double omega = 0.0;
  const double beta =
      (Teuchos::nonnull(x_dotT) || Teuchos::nonnull(x_dotdotT)) ?
          in_args.get_beta() : 1.0;
  const double curr_time =
      (Teuchos::nonnull(x_dotT) || Teuchos::nonnull(x_dotdotT)) ?
          in_args.get_t() : 0.0;

  for (int l = 0; l < in_args.Np(); ++l) {
    const Teuchos::RCP<const Thyra::VectorBase<ST> > p = in_args.get_p(l);
    const Teuchos::RCP<const Tpetra_Vector> pT =
        ConverterT::getConstTpetraVector(p);
    const Teuchos::ArrayRCP<const ST> pT_constView = pT->get1dView();
    if (Teuchos::nonnull(p)) {
      if (l < num_params_partial_sum_[0])
          {
        for (unsigned int j = 0; j < sacado_param_vecs_[0][l].size(); j++)
          sacado_param_vecs_[0][l][j].baseValue = pT_constView[j];
      }
      else {
        for (int m = 1; m < num_models_; m++) {
          if (l >= num_params_partial_sum_[m - 1]
              && l < num_params_partial_sum_[m]) {
            for (unsigned int j = 0;
                j
                    < sacado_param_vecs_[m][l - num_params_partial_sum_[m - 1]]
                        .size(); j++)
              sacado_param_vecs_[m][l - num_params_partial_sum_[m - 1]][j]
                  .baseValue = pT_constView[j];
          }
        }
      }
    }
  }

  //
  // Get the output arguments
  //
  const Teuchos::RCP<Tpetra_Vector> fT_out =
      Teuchos::nonnull(out_args.get_f()) ?
          ConverterT::getTpetraVector(out_args.get_f()) :
          Teuchos::null;

  //Create a Teuchos array of the fT_out for each model.
  Teuchos::Array<Teuchos::RCP<Tpetra_Vector> > fTs_out;
  fTs_out.resize(num_models_);
  for (int m = 0; m < num_models_; m++) {
    Teuchos::RCP<const Tpetra_Vector> fT_out_temp =
        Teuchos::nonnull(solver_outargs_[m].get_f()) ?
            ConverterT::getTpetraVector(solver_outargs_[m].get_f()) :
            Teuchos::null;
    fTs_out[m] = Teuchos::rcp(new Tpetra_Vector(*fT_out_temp));
  }

  const Teuchos::RCP<Tpetra_Operator> W_op_outT =
      Teuchos::nonnull(out_args.get_W_op()) ?
          ConverterT::getTpetraOperator(out_args.get_W_op()) :
          Teuchos::null;

  // Cast W to a CrsMatrix, throw an exception if this fails
  const Teuchos::RCP<Tpetra_CrsMatrix> W_op_out_crsT =
      Teuchos::nonnull(W_op_outT) ?
          Teuchos::rcp_dynamic_cast<Tpetra_CrsMatrix>(W_op_outT, true) :
          Teuchos::null;

  //
  // Compute the functions
  //
  // W matrix
  // Create Teuchos::Array of individual models' W matrices 
  const Teuchos::RCP<Tpetra_Operator> W_op_outT_temp;
  Teuchos::Array<Teuchos::RCP<Tpetra_CrsMatrix> > W_op_outs_crsT; 
  W_op_outs_crsT.resize(num_models_); 
  //get each of the models' W matrices 
  for (int m=0; m<num_models_; m++) {
    const Teuchos::RCP<Tpetra_Operator> W_op_outT_temp = Teuchos::nonnull(solver_outargs_[m].get_W_op()) ?
                                                         ConverterT::getTpetraOperator(out_args.get_W_op()) :
                                                         Teuchos::null;
    W_op_outs_crsT[m] = Teuchos::nonnull(W_op_outT_temp) ?
                        Teuchos::rcp_dynamic_cast<Tpetra_CrsMatrix>(W_op_outT_temp, true) :
                        Teuchos::null;
  }

  Teuchos::Array<bool> fs_already_computed; 
  fs_already_computed.resize(num_models_); 
  for (int m=0; m<num_models_; m++) fs_already_computed[m]= false;

  // W matrix for each individual model
  for (int m=0; m<num_models_; m++) { 
    if (Teuchos::nonnull(W_op_outs_crsT[m])) {
      //computeGlobalJacobianT sets fTs_out[m] and W_op_outs_crsT[m]
      apps_[m]->computeGlobalJacobianT(
                alpha, beta, omega, curr_time, x_dotTs[m].get(), x_dotdotT.get(),  *xTs[m],
                sacado_param_vecs_[m], fTs_out[m].get(), *W_op_outs_crsT[m]);
      fs_already_computed[m] = true;
    }
   }


  //FIXME: create coupled W matrix from array of model W matrices   
  Teuchos::RCP<LCM::Schwarz_CoupledJacobian> W_op_out_coupled = Teuchos::rcp_dynamic_cast<LCM::Schwarz_CoupledJacobian>(W_op_outT, true);
  W_op_out_coupled->initialize(W_op_outs_crsT);
  
  // Create fT_out from fTs_out[m]
  LO counter_local = 0;
  //get nonconst view of fT_out
  Teuchos::ArrayRCP<ST> fT_out_nonconstView = fT_out->get1dViewNonConst();
  for (int m = 0; m < num_models_; m++) {
    //get const view of mth x_init & x_dot_init vector
    Teuchos::ArrayRCP<ST> fT_out_nonconstView_m = fTs_out[m]->get1dViewNonConst(); 
    for (int i=0; i<fTs_out[m]->getLocalLength(); i++) {
      fT_out_nonconstView[counter_local + i] = fT_out_nonconstView_m[i];   
    }
    counter_local += fTs_out[m]->getLocalLength(); 
  } 
  

  //W prec matrix
  //FIXME: eventually will need to hook in Teko. 


   //FIXME: in the following, need to check logic involving looping over num_models_ -- here we're 
   //not creating arrays to store things in for each model.
   //TODO: understand better how evalModel is called and how g and f parameter arrays are set  
   // df/dp
  for (int l = 0; l < out_args.Np(); ++l) {
    int m_num; 
    for (int m=0; m<num_models_; m++) { 
      int index;
      if (l < num_params_partial_sum_[0]) { //m = 0 case 
        index = l;
        m_num = 0;  
       }
       else if (l >= num_params_partial_sum_[m-1] && l < num_params_partial_sum_[m]) {
          index = l - num_params_partial_sum_[m-1]; 
          m_num = m; 
       }
       const Teuchos::RCP<Thyra::MultiVectorBase<ST> > dfdp_out= solver_outargs_[m_num].get_DfDp(index).getMultiVector(); 
       const Teuchos::RCP<Tpetra_MultiVector> dfdp_outT = Teuchos::nonnull(dfdp_out) ?
                   ConverterT::getTpetraMultiVector(dfdp_out) :
                   Teuchos::null;
     
      if (Teuchos::nonnull(dfdp_outT)) {
        const Teuchos::RCP<ParamVec> p_vec = Teuchos::rcpFromRef(sacado_param_vecs_[m_num][index]);

        //computeGlobalTangentT sets last 3 arguments: fTs_out[m_num] and dfdp_outT
        apps_[m]->computeGlobalTangentT(
                  0.0, 0.0, 0.0, curr_time, false, x_dotTs[m_num].get(), x_dotdotT.get(), *xTs[m_num],
                  sacado_param_vecs_[m_num], p_vec.get(),
                  NULL, NULL, NULL, NULL, fTs_out[m_num].get(), NULL,
                  dfdp_outT.get());

        fs_already_computed[m_num] = true;
       }
    }
  }
  //FIXME: create fT_out from fTs_out 

  // Response functions
  for (int j = 0; j < out_args.Ng(); ++j) {
    const Teuchos::RCP<Thyra::VectorBase<ST> > g_out = out_args.get_g(j);
    Teuchos::RCP<Tpetra_Vector> gT_out = Teuchos::nonnull(g_out) ?
                                         ConverterT::getTpetraVector(g_out) :
                                         Teuchos::null;
    
    const Thyra::ModelEvaluatorBase::Derivative<ST> dgdxT_out = out_args.get_DgDx(j);
    const Thyra::ModelEvaluatorBase::Derivative<ST> dgdxdotT_out = out_args.get_DgDx_dot(j);
    // AGS: x_dotdot time integrators not imlemented in Thyra ME yet
    const Thyra::ModelEvaluatorBase::Derivative<ST> dgdxdotdotT_out;
    int m_num; 
    for (int m=0; m<num_models_; m++) { 
      int index; 
      if (j < num_responses_partial_sum_[0]) { //m = 0 case 
        index = j;
        m_num = 0; 
       }
       else if (j >= num_responses_partial_sum_[m-1] && j < num_responses_partial_sum_[m]) {
         index = j - num_responses_partial_sum_[m-1]; 
         m_num = m; 
       }
 
       // dg/dx, dg/dxdot
       if (!dgdxT_out.isEmpty() || !dgdxdotT_out.isEmpty()) {
         const Thyra::ModelEvaluatorBase::Derivative<ST> dummy_derivT;
         //sets gT_out 
         apps_[m_num]->evaluateResponseDerivativeT(
                    index, curr_time, x_dotTs[m_num].get(), x_dotdotT.get(), *xTs[m_num],
                    sacado_param_vecs_[m_num], NULL,
                    gT_out.get(), dgdxT_out,
                    dgdxdotT_out, dgdxdotdotT_out, dummy_derivT);
          // Set gT_out to null to indicate that g_out was evaluated.
          gT_out = Teuchos::null;
      }

      // dg/dp
      for (int l = 0; l < out_args.Np(); ++l) {
        const Teuchos::RCP<Thyra::MultiVectorBase<ST> > dgdp_out =
          out_args.get_DgDp(j, l).getMultiVector();
        const Teuchos::RCP<Tpetra_MultiVector> dgdpT_out =
          Teuchos::nonnull(dgdp_out) ?
          ConverterT::getTpetraMultiVector(dgdp_out) :
          Teuchos::null;
        int index2;
        if (l < num_params_partial_sum_[0]) { //m = 0 case 
          index2 = l;
        }
        else if (l >= num_params_partial_sum_[m_num-1] && l < num_params_partial_sum_[m_num]) {
          index2 = l - num_params_partial_sum_[m_num-1]; 
        }

        if (Teuchos::nonnull(dgdpT_out)) {
          const Teuchos::RCP<ParamVec> p_vec = Teuchos::rcpFromRef(sacado_param_vecs_[m_num][index2]);
          //sets gT_out, dgdpT_out 
          apps_[m_num]->evaluateResponseTangentT(
              index, alpha, beta, omega, curr_time, false,
              x_dotTs[m_num].get(), x_dotdotT.get(), *xTs[m_num],
              sacado_param_vecs_[m_num], p_vec.get(),
              NULL, NULL, NULL, NULL, gT_out.get(), NULL,
              dgdpT_out.get());
          gT_out = Teuchos::null;
        }
      }
    
      if (Teuchos::nonnull(gT_out)) {
        //sets gT_out 
        apps_[m_num]->evaluateResponseT(
            index, curr_time, x_dotTs[m_num].get(), x_dotdotT.get(), *xTs[m_num],
            sacado_param_vecs_[m_num], *gT_out);
      }
    }
  }

}

Thyra::ModelEvaluatorBase::InArgs<ST>
LCM::SchwarzMultiscale::
createInArgsImpl() const
{
  std::cout << "DEBUG: In LCM::SchwarzMultiScale::createInArgsImpl()!\n";
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

  result.set_Np(num_params_total_);

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
