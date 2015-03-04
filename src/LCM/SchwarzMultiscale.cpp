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

LCM::
SchwarzMultiscale::
SchwarzMultiscale(Teuchos::RCP<Teuchos::ParameterList> const & app_params,
    Teuchos::RCP<Teuchos::Comm<int> const> const & commT,
    Teuchos::RCP<Tpetra_Vector const> const & initial_guessT)
{
  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";

  commT_ = commT;

  //IK, 2/11/15: I am assuming for now we don't have any distributed parameters.
  num_dist_params_total_ = 0;

  // Get "Coupled Schwarz" parameter sublist
  Teuchos::ParameterList &
  coupled_system_params = app_params->sublist("Coupled System");

  // Get names of individual model xml input files from problem parameterlist
  Teuchos::Array<std::string>
  model_filenames =
    coupled_system_params.get<Teuchos::Array<std::string> >("Model XML Files");

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

  //string for storing name of first problem, for error checking
  std::string
  problem_name_0 = "";

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

    if (m == 0) {
      problem_name_0 = problem_params_m->get("Name", "");
    }

    assert(problem_name_0 == problem_name);

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

    int const
    sum_responses = m > 0 ? num_responses_partial_sum_[m - 1] : 0;

    int const
    sum_params = m > 0 ? num_responses_partial_sum_[m - 1] : 0;

    num_responses_partial_sum_[m] = num_responses_[m] + sum_responses;
    num_params_partial_sum_[m] = num_params_[m] + sum_params;

    //Does it make sense for num_params_total and num_responses_total to be
    //the sum of these values for each model?  I guess so.
    num_params_total_ += num_params_[m];
    num_responses_total_ += num_responses_[m];
  }

  // Create sacado parameter vectors of appropriate size
  // for use in evalModelImpl
  sacado_param_vecs_.resize(num_models_);

  for (int m = 0; m < num_models_; ++m) {
    sacado_param_vecs_[m].resize(num_params_[m]);
  }

  std::cout << "Total # parameters, responses: " << num_params_total_;
  std::cout << ", " << num_responses_total_ << '\n';

  //Create ccoupled_disc_map, a map for the entire coupled ME solution,
  //created from the entries of the disc_maps array (individual maps).
  coupled_disc_map_ = createCoupledMap(disc_maps_, commT_);

  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << ": created coupled map!\n";

#ifdef WRITE_TO_MATRIX_MARKET
  // For debug, write the coupled map to matrix market file to
  // look at in vi or matlab
  Tpetra_MatrixMarket_Writer::writeMapFile(
      "coupled_disc_map.mm",
      *coupled_disc_map_);
#endif

  // Setup nominal values
  nominal_values_ = this->createInArgsImpl();

  // All the ME vectors are allocated/unallocated here
  // Calling allocateVectors() will set x and x_dot in nominal_values_
  allocateVectors();

  // set p_init in nominal_values_
  // TODO: Check if these are correct nominal values for parameters
  for (int m = 0; m < num_models_; ++m) {

    int const
    num_params_m = num_params_[m];

    int const
    offset = m > 0 ? num_params_partial_sum_[m - 1] : 0;

    for (int l = 0; l < num_params_m; ++l) {
      Teuchos::RCP<Thyra::VectorBase<ST> const>
      p = models_[m]->getNominalValues().get_p(l);

      // Don't set it if there is nothing. Avoid null pointer.
      if (Teuchos::is_null(p) == true) continue;

      nominal_values_.set_p(l + offset, p);
    }
  }
  //end setting of nominal values

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
  std::vector<GO>
  my_global_elements(local_num_elements);

  LO
  counter_local = 0;

  GO
  counter_global = 0;

  for (int m = 0; m < n_maps; ++m) {
    LO
    local_num_elements_n = maps[m]->getNodeNumElements();

    GO
    global_num_elements_n = maps[m]->getGlobalNumElements();

    Teuchos::ArrayView<GO const>
    disc_global_elements = maps[m]->getNodeElementList();

    for (int l = 0; l < local_num_elements_n; ++l) {
      my_global_elements[counter_local + l] = counter_global
          + disc_global_elements[l];
    }
    counter_local += local_num_elements_n;
    counter_global += global_num_elements_n;
  }

  Teuchos::ArrayView<GO> const
  my_global_elements_AV =
      Teuchos::arrayView(&my_global_elements[0], local_num_elements);

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
  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";

  Teuchos::RCP<Tpetra_Map const>
  map = Teuchos::rcp(new (Tpetra_Map const)(*coupled_disc_map_));

  Teuchos::RCP<Thyra::VectorSpaceBase<ST> const>
  coupled_x_space = Thyra::createVectorSpace<ST>(map);

  return coupled_x_space;
}

Teuchos::RCP<Thyra::VectorSpaceBase<ST> const>
LCM::SchwarzMultiscale::get_f_space() const
{
  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";

  Teuchos::RCP<Tpetra_Map const>
  map = Teuchos::rcp(new (Tpetra_Map const)(*coupled_disc_map_));

  Teuchos::RCP<Thyra::VectorSpaceBase<ST> const>
  coupled_f_space = Thyra::createVectorSpace<ST>(map);

  return coupled_f_space;
}

Teuchos::RCP<const Thyra::VectorSpaceBase<ST> >
LCM::SchwarzMultiscale::get_p_space(int l) const
{
  assert(0 <= l && l < num_params_total_);

  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";

  for (int m = 0; m < num_models_; ++m) {
    int const
    lo = m > 0 ? num_params_partial_sum_[m - 1] : 0;

    int const
    hi = num_params_partial_sum_[m];

    bool const
    in_range = lo <= l && l < hi;

    if (in_range == true) {
      return models_[m]->get_p_space(l - lo);
    }
  }

  return Teuchos::null;
}

Teuchos::RCP<const Thyra::VectorSpaceBase<ST> >
LCM::SchwarzMultiscale::get_g_space(int l) const
{
  assert(0 <= l && l < num_responses_total_);

  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";

  for (int m = 0; m < num_models_; ++m) {
    int const
    lo = m > 0 ? num_responses_partial_sum_[m - 1] : 0;

    int const
    hi = num_responses_partial_sum_[m];

    bool const
    in_range = lo <= l && l < hi;

    if (in_range == true) {
      return models_[m]->get_g_space(l - lo);
    }
  }

  return Teuchos::null;
}

Teuchos::RCP<const Teuchos::Array<std::string> >
LCM::SchwarzMultiscale::get_p_names(int l) const
{
  assert(0 <= l && l < num_params_total_);

  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";

  for (int m = 0; m < num_models_; ++m) {
    int const
    lo = m > 0 ? num_params_partial_sum_[m - 1] : 0;

    int const
    hi = num_params_partial_sum_[m];

    bool const
    in_range = lo <= l && l < hi;

    if (in_range == true) {
      return models_[m]->get_p_names(l - lo);
    }
  }

  return Teuchos::null;
}

Thyra::ModelEvaluatorBase::InArgs<ST>
LCM::SchwarzMultiscale::getNominalValues() const
{
  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";

  return nominal_values_;
}

Thyra::ModelEvaluatorBase::InArgs<ST>
LCM::SchwarzMultiscale::getLowerBounds() const
{
  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";

  return Thyra::ModelEvaluatorBase::InArgs<ST>(); // Default value
}

Thyra::ModelEvaluatorBase::InArgs<ST>
LCM::SchwarzMultiscale::getUpperBounds() const
{
  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";

  return Thyra::ModelEvaluatorBase::InArgs<ST>(); // Default value
}

Teuchos::RCP<Thyra::LinearOpBase<ST> >
LCM::SchwarzMultiscale::create_W_op() const
{
  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";

  Teuchos::RCP<Tpetra_Operator> const
  W_out_coupled = Teuchos::rcp(
      new LCM::Schwarz_CoupledJacobian(disc_maps_, coupled_disc_map_, commT_));
  return Thyra::createLinearOp(W_out_coupled);
}

Teuchos::RCP<Thyra::PreconditionerBase<ST> >
LCM::SchwarzMultiscale::create_W_prec() const
{
  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";

  //Analog of EpetraExt::ModelEvaluator::Preconditioner does not exist
  //in Thyra yet!  So problem will run for now with no
  //preconditioner...
  bool const
  W_prec_not_supported = true;

  TEUCHOS_TEST_FOR_EXCEPT(W_prec_not_supported);
  return Teuchos::null;
}

Teuchos::RCP<const Thyra::LinearOpWithSolveFactoryBase<ST> >
LCM::SchwarzMultiscale::get_W_factory() const
{
  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";

  return Teuchos::null;
}

Thyra::ModelEvaluatorBase::InArgs<ST>
LCM::SchwarzMultiscale::createInArgs() const
{
  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";

  return this->createInArgsImpl();
}

void
LCM::SchwarzMultiscale::reportFinalPoint(
    Thyra::ModelEvaluatorBase::InArgs<ST> const & final_point,
    bool const was_solved)
{
  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";

  TEUCHOS_TEST_FOR_EXCEPTION(true,
      Teuchos::Exceptions::InvalidParameter,
      "Calling reportFinalPoint in CoupledSchwarz.cpp" << '\n');
}

void
LCM::SchwarzMultiscale::
allocateVectors()
{
  //In this function, we create and set x_init and x_dot_init in
  //nominal_values_ for the coupled model.
  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";

  //Create Teuchos::Arrays of hte x_init and x_dot init Tpetra_Vectors
  //for each of the models
  Teuchos::Array<Teuchos::RCP<Tpetra_Vector const> >
  x_inits(num_models_);

  Teuchos::Array<Teuchos::RCP<Tpetra_Vector const> >
  x_dot_inits(num_models_);

  //Populate the arrays with the x_init and x_dot_init for each model.
  for (int m = 0; m < num_models_; ++m) {
    x_inits[m] = apps_[m]->getInitialSolutionT();
    x_dot_inits[m] = apps_[m]->getInitialSolutionDotT();
  }

  // Create Tpetra objects to be wrapped in Thyra for coupled model
  //FIXME: implement by concatenating individual x_inits
  Teuchos::RCP<Tpetra_Vector>
  coupled_x_init = Teuchos::rcp(new Tpetra_Vector(coupled_disc_map_));

  //FIXME: implement by concatenating individual x_dot_inits
  Teuchos::RCP<Tpetra_Vector>
  coupled_x_dot_init = Teuchos::rcp(new Tpetra_Vector(coupled_disc_map_));

  //initialize coupled vecs to all 0s
  coupled_x_init->putScalar(0.0);
  coupled_x_dot_init->putScalar(0.0);

  LO
  counter_local = 0;

  //get nonconst view of coupled_x_init & coupled_x_dot_init
  Teuchos::ArrayRCP<ST>
  coupled_x_init_view = coupled_x_init->get1dViewNonConst();

  Teuchos::ArrayRCP<ST>
  coupled_x_dot_init_view = coupled_x_dot_init->get1dViewNonConst();

  for (int m = 0; m < num_models_; ++m) {
    //get const view of mth x_init & x_dot_init vector
    Teuchos::ArrayRCP<ST const>
    x_init_const_view = x_inits[m]->get1dView();

    Teuchos::ArrayRCP<ST const>
    x_dot_init_const_view = x_dot_inits[m]->get1dView();

    //The following assumes x_init and x_dot_init have same length.
    //FIXME? Could have error checking to make sure.
    for (int i = 0; i < x_inits[m]->getLocalLength(); ++i) {
      coupled_x_init_view[counter_local + i] = x_init_const_view[i];
      coupled_x_dot_init_view[counter_local + i] =
          x_dot_init_const_view[i];
    }
    counter_local += x_inits[m]->getLocalLength();
  }

#ifdef WRITE_TO_MATRIX_MARKET
  //writing to MatrixMarket file for debug
  Tpetra_MatrixMarket_Writer::writeDenseFile(
      "x_init0.mm",
      *(x_inits[0]));

  Tpetra_MatrixMarket_Writer::writeDenseFile(
      "x_dot_init0.mm",
      *(x_dot_inits[0]));

  if (num_models_ > 1) {
    Tpetra_MatrixMarket_Writer::writeDenseFile(
        "x_init1.mm",
        *(x_inits[1]));

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

  Teuchos::RCP<Tpetra_Map const>
  map = Teuchos::rcp(new (Tpetra_Map const)(*coupled_disc_map_));

  Teuchos::RCP<Thyra::VectorSpaceBase<ST> const>
  coupled_x_space = Thyra::createVectorSpace<ST>(map);

  // Create non-const versions of xT_init and x_dotT_init
  Teuchos::RCP<Tpetra_Vector> const
  coupled_x_init_nonconst = Teuchos::rcp(new Tpetra_Vector(*coupled_x_init));

  Teuchos::RCP<Tpetra_Vector> const
  coupled_x_dot_init_nonconst =
      Teuchos::rcp(new Tpetra_Vector(*coupled_x_dot_init));

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
  assert(0 <= j && j < num_responses_total_);

  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";

  for (int m = 0; m < num_models_; ++m) {
    int const
    lo = m > 0 ? num_responses_partial_sum_[m - 1] : 0;

    int const
    hi = num_responses_partial_sum_[m];

    bool const
    in_range = lo <= j && j < hi;

    if (in_range == true) {
      return Thyra::createLinearOp(
          apps_[m]->getResponse(j - lo)->createGradientOpT());
    }
  }

  return Teuchos::null;
}

/// Create operator form of dg/dx_dot for distributed responses
// FIXME: Is this correct? It seems the same as above.
Teuchos::RCP<Thyra::LinearOpBase<ST> >
LCM::SchwarzMultiscale::
create_DgDx_dot_op_impl(int j) const
{
  assert(0 <= j && j < num_responses_total_);

  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";

  for (int m = 0; m < num_models_; ++m) {
    int const
    lo = m > 0 ? num_responses_partial_sum_[m - 1] : 0;

    int const
    hi = num_responses_partial_sum_[m];

    bool const
    in_range = lo <= j && j < hi;

    if (in_range == true) {
      return Thyra::createLinearOp(
          apps_[m]->getResponse(j - lo)->createGradientOpT());
    }
  }

  return Teuchos::null;
}

/// Create OutArgs
Thyra::ModelEvaluatorBase::OutArgs<ST>
LCM::SchwarzMultiscale::
createOutArgsImpl() const
{
  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";

  Thyra::ModelEvaluatorBase::OutArgsSetup<ST>
  result;

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
    Thyra::ModelEvaluatorBase::DerivativeSupport
    dgdx_support;

    for (int m = 0; m < num_models_; ++m) {
      int const
      lo = m > 0 ? num_responses_partial_sum_[m - 1] : 0;

      int const
      hi = num_responses_partial_sum_[m];

      bool const
      in_range = lo <= i && i < hi;

      if (in_range == true) {
        if (apps_[m]->getResponse(i - lo)->isScalarResponse()) {
          dgdx_support = Thyra::ModelEvaluatorBase::DERIV_TRANS_MV_BY_ROW;
        }
        else {
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

    for (int l = 0; l < num_params_total_; ++l) {
      result.setSupports(
          Thyra::ModelEvaluatorBase::OUT_ARG_DgDp, i, l,
          Thyra::ModelEvaluatorBase::DERIV_MV_BY_COL);
    }
  }

  return result;
}

/// Evaluate model on InArgs
void
LCM::SchwarzMultiscale::
evalModelImpl(
    Thyra::ModelEvaluatorBase::InArgs<ST> const & in_args,
    Thyra::ModelEvaluatorBase::OutArgs<ST> const & out_args) const
{
  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << '\n';

  //Create a Teuchos array of the xT and x_dotT for each model.
  Teuchos::Array<Teuchos::RCP<Tpetra_Vector> >
  xTs(num_models_);

  Teuchos::Array<Teuchos::RCP<Tpetra_Vector> >
  x_dotTs(num_models_);

  for (int m = 0; m < num_models_; ++m) {
    Teuchos::RCP<Tpetra_Vector const>
    xT_temp = ConverterT::getConstTpetraVector(models_[m]->getNominalValues().get_x());

    xTs[m] = Teuchos::rcp(new Tpetra_Vector(*xT_temp));

    Teuchos::RCP<Tpetra_Vector const>
    x_dotT_temp = Teuchos::nonnull(models_[m]->getNominalValues().get_x_dot()) ?
            ConverterT::getConstTpetraVector(models_[m]->getNominalValues().get_x_dot()) :
            Teuchos::null;

    x_dotTs[m] = Teuchos::rcp(new Tpetra_Vector(*x_dotT_temp));
  }

  // AGS: x_dotdot time integrators not imlemented in Thyra ME yet
  //const Teuchos::RCP<const Tpetra_Vector> x_dotdotT =
  //  Teuchos::nonnull(in_args.get_x_dotdot()) ?
  //  ConverterT::getConstTpetraVector(in_args.get_x_dotdot()) :
  //  Teuchos::null;
  // Get the input arguments
  Teuchos::RCP<Tpetra_Vector const> const
  xT = ConverterT::getConstTpetraVector(in_args.get_x());

  Teuchos::RCP<Tpetra_Vector const> const
  x_dotT = Teuchos::nonnull(in_args.get_x_dot()) ?
      ConverterT::getConstTpetraVector(in_args.get_x_dot()) :
      Teuchos::null;

  Teuchos::RCP<Tpetra_Vector const> const
  x_dotdotT = Teuchos::null;

  double const
  alpha = (Teuchos::nonnull(x_dotT) || Teuchos::nonnull(x_dotdotT)) ?
          in_args.get_alpha() : 0.0;

  // AGS: x_dotdot time integrators not imlemented in Thyra ME yet
  // const double omega = (Teuchos::nonnull(x_dotT) ||
  // Teuchos::nonnull(x_dotdotT)) ? in_args.get_omega() : 0.0;

  double const
  omega = 0.0;

  double const beta =
      (Teuchos::nonnull(x_dotT) || Teuchos::nonnull(x_dotdotT)) ?
          in_args.get_beta() : 1.0;

  double const curr_time =
      (Teuchos::nonnull(x_dotT) || Teuchos::nonnull(x_dotdotT)) ?
          in_args.get_t() : 0.0;

  for (int m = 0; m < num_models_; ++m) {
    int const
    num_params_m = num_params_[m];

    int const
    offset = m > 0 ? num_params_partial_sum_[m - 1] : 0;

    for (int l = 0; l < num_params_m; ++l) {
      Teuchos::RCP<Thyra::VectorBase<ST> const>
      p = in_args.get_p(l + offset);

      // Don't set it if there is nothing. Avoid null pointer.
      if (Teuchos::is_null(p) == true) continue;

      Teuchos::RCP<Tpetra_Vector const> const
      pT = ConverterT::getConstTpetraVector(p);

      Teuchos::ArrayRCP<ST const> const
      pT_view = pT->get1dView();

      for (unsigned int j = 0; j < sacado_param_vecs_[m][l].size(); j++) {
        sacado_param_vecs_[m][l][j].baseValue = pT_view[j];
      }
    }
  }

  //
  // Get the output arguments
  //
  Teuchos::RCP<Tpetra_Vector> const
  fT_out = Teuchos::nonnull(out_args.get_f()) ?
      ConverterT::getTpetraVector(out_args.get_f()) :
      Teuchos::null;

  //Create a Teuchos array of the fT_out for each model.
  Teuchos::Array<Teuchos::RCP<Tpetra_Vector> >
  fTs_out(num_models_);

  for (int m = 0; m < num_models_; ++m) {

    Teuchos::RCP<Tpetra_Vector const>
    fT_out_temp = Teuchos::nonnull(solver_outargs_[m].get_f()) ?
        ConverterT::getTpetraVector(solver_outargs_[m].get_f()) :
        Teuchos::null;
    fTs_out[m] = Teuchos::nonnull(fT_out_temp) ?
            Teuchos::rcp(new Tpetra_Vector(*fT_out_temp)) :
            Teuchos::null;
    //if (fT_out_temp != Teuchos::null) 
    //  fTs_out[m] = Teuchos::rcp(new Tpetra_Vector(*fT_out_temp));
  }

  Teuchos::RCP<Tpetra_Operator> const
  W_op_outT = Teuchos::nonnull(out_args.get_W_op()) ?
      ConverterT::getTpetraOperator(out_args.get_W_op()) :
      Teuchos::null;

  // Cast W to a CrsMatrix, throw an exception if this fails
  Teuchos::RCP<Tpetra_CrsMatrix> const
  W_op_out_crsT = Teuchos::nonnull(W_op_outT) ?
      Teuchos::rcp_dynamic_cast<Tpetra_CrsMatrix>(W_op_outT, true) :
      Teuchos::null;

  //
  // Compute the functions
  //
  // W matrix
  // Create Teuchos::Array of individual models' W matrices
  Teuchos::Array<Teuchos::RCP<Tpetra_CrsMatrix> >
  W_op_outs_crsT(num_models_);

  //get each of the models' W matrices
  for (int m = 0; m < num_models_; ++m) {
    Teuchos::RCP<Tpetra_Operator> const
    W_op_outT_temp =
        Teuchos::nonnull(models_[m]->create_W_op()) ?
            ConverterT::getTpetraOperator(models_[m]->create_W_op()) :
            Teuchos::null;

    W_op_outs_crsT[m] =
        Teuchos::nonnull(W_op_outT_temp) ?
            Teuchos::rcp_dynamic_cast<Tpetra_CrsMatrix>(W_op_outT_temp, true) :
            Teuchos::null;
  }

  Teuchos::Array<bool>
  fs_already_computed(num_models_, false);

  // W matrix for each individual model
  for (int m = 0; m < num_models_; ++m) {
    if (Teuchos::is_null(W_op_outs_crsT[m]) == false) {
      //computeGlobalJacobianT sets fTs_out[m] and W_op_outs_crsT[m]
      apps_[m]->computeGlobalJacobianT(
          alpha, beta, omega, curr_time,
          x_dotTs[m].get(), x_dotdotT.get(), *xTs[m],
          sacado_param_vecs_[m], fTs_out[m].get(), *W_op_outs_crsT[m]);

      fs_already_computed[m] = true;
    }
  }

  // FIXME: create coupled W matrix from array of model W matrices
  if (W_op_outT != Teuchos::null) { 
    Teuchos::RCP<LCM::Schwarz_CoupledJacobian> W_op_out_coupled =
      Teuchos::rcp_dynamic_cast<LCM::Schwarz_CoupledJacobian>(W_op_outT, true);
    W_op_out_coupled->initialize(W_op_outs_crsT);
  }

  // Create fT_out from fTs_out[m]
  LO
  counter_local = 0;

  //get nonconst view of fT_out
  Teuchos::ArrayRCP<ST> fT_out_nonconst_view = fT_out->get1dViewNonConst();

  for (int m = 0; m < num_models_; ++m) {
    //get const view of mth x_init & x_dot_init vector
    if (fTs_out[m] != Teuchos::null) {
      Teuchos::ArrayRCP<ST>
      fT_out_nonconst_view_m = fTs_out[m]->get1dViewNonConst();

      for (int i = 0; i < fTs_out[m]->getLocalLength(); ++i) {
        fT_out_nonconst_view[counter_local + i] = fT_out_nonconst_view_m[i];
      }
      counter_local += fTs_out[m]->getLocalLength();
    }
  }

  // W prec matrix
  // FIXME: eventually will need to hook in Teko.

  // FIXME: in the following, need to check logic involving looping over
  // num_models_ -- here we're not creating arrays to store things in
  // for each model.
  // TODO: understand better how evalModel is called and how g and f parameter
  // arrays are set in df/dp


  for (int l = 0; l < out_args.Np(); ++l) {
    for (int m = 0; m < num_models_; ++m) {

      int const
      lo = m > 0 ? num_params_partial_sum_[m - 1] : 0;

      int const
      hi = num_params_partial_sum_[m];

      bool const
      in_range = lo <= l && l < hi;

      if (in_range == false) continue;

      int const
      index = l - lo;

      Teuchos::RCP<Thyra::MultiVectorBase<ST> > const
      dfdp_out = solver_outargs_[m].get_DfDp(index).getMultiVector();

      Teuchos::RCP<Tpetra_MultiVector> const
      dfdp_outT = Teuchos::nonnull(dfdp_out) ?
          ConverterT::getTpetraMultiVector(dfdp_out) :
          Teuchos::null;

      if (Teuchos::nonnull(dfdp_outT)) {
        Teuchos::RCP<ParamVec> const
        p_vec = Teuchos::rcpFromRef(sacado_param_vecs_[m][index]);

        // computeGlobalTangentT sets last 3 arguments:
        // fTs_out[m_num] and dfdp_outT
        apps_[m]->computeGlobalTangentT(
            0.0, 0.0, 0.0, curr_time, false, x_dotTs[m].get(), x_dotdotT.get(),
            *xTs[m], sacado_param_vecs_[m], p_vec.get(),
            NULL, NULL, NULL, NULL, fTs_out[m].get(), NULL, dfdp_outT.get());

        fs_already_computed[m] = true;
      }
    }
  }

  //FIXME: create fT_out from fTs_out

  // Response functions
  for (int j = 0; j < out_args.Ng(); ++j) {

    Teuchos::RCP<Thyra::VectorBase<ST> > const
    g_out = out_args.get_g(j);

    Teuchos::RCP<Tpetra_Vector>
    gT_out = Teuchos::nonnull(g_out) ?
        ConverterT::getTpetraVector(g_out) :
        Teuchos::null;

    Thyra::ModelEvaluatorBase::Derivative<ST> const
    dgdxT_out = out_args.get_DgDx(j);

    Thyra::ModelEvaluatorBase::Derivative<ST> const
    dgdxdotT_out = out_args.get_DgDx_dot(j);

    // AGS: x_dotdot time integrators not imlemented in Thyra ME yet
    Thyra::ModelEvaluatorBase::Derivative<ST> const
    dgdxdotdotT_out;

    for (int m = 0; m < num_models_; ++m) {

      int const
      lo = m > 0 ? num_responses_partial_sum_[m - 1] : 0;

      int const
      hi = num_responses_partial_sum_[m];

      bool const
      in_range = lo <= j && j < hi;

      if (in_range == false) continue;

      int const
      index = j - lo;

      // dg/dx, dg/dxdot
      if (!dgdxT_out.isEmpty() || !dgdxdotT_out.isEmpty()) {
        Thyra::ModelEvaluatorBase::Derivative<ST> const
        dummy_derivT;

        //sets gT_out
        apps_[m]->evaluateResponseDerivativeT(
            index, curr_time, x_dotTs[m].get(), x_dotdotT.get(),
            *xTs[m], sacado_param_vecs_[m], NULL,
            gT_out.get(), dgdxT_out,
            dgdxdotT_out, dgdxdotdotT_out, dummy_derivT);

        // Set gT_out to null to indicate that g_out was evaluated.
        gT_out = Teuchos::null;
      }

      // dg/dp
      for (int l = 0; l < out_args.Np(); ++l) {

        Teuchos::RCP<Thyra::MultiVectorBase<ST> > const
        dgdp_out = out_args.get_DgDp(j, l).getMultiVector();

        Teuchos::RCP<Tpetra_MultiVector> const
        dgdpT_out = Teuchos::nonnull(dgdp_out) ?
                ConverterT::getTpetraMultiVector(dgdp_out) :
                Teuchos::null;

        int const
        min = m > 0 ? num_params_partial_sum_[m - 1] : 0;

        int const
        max = num_params_partial_sum_[m];

        bool const
        l_is_in_range = min <= l && l < max;

        if (l_is_in_range == false) continue;

        int const
        k = l - min;

        if (Teuchos::is_null(dgdpT_out) == false) {

          Teuchos::RCP<ParamVec> const
          p_vec = Teuchos::rcpFromRef(sacado_param_vecs_[m][k]);

          //sets gT_out, dgdpT_out
          apps_[m]->evaluateResponseTangentT(
              k, alpha, beta, omega, curr_time, false,
              x_dotTs[m].get(), x_dotdotT.get(), *xTs[m],
              sacado_param_vecs_[m], p_vec.get(),
              NULL, NULL, NULL, NULL, gT_out.get(), NULL,
              dgdpT_out.get());
          gT_out = Teuchos::null;
        }
      }

      if (Teuchos::is_null(gT_out) == false) {
        //sets gT_out
        apps_[m]->evaluateResponseT(
            index, curr_time, x_dotTs[m].get(), x_dotdotT.get(),
            *xTs[m], sacado_param_vecs_[m], *gT_out);
      }
    }
  }
}

Thyra::ModelEvaluatorBase::InArgs<ST>
LCM::SchwarzMultiscale::
createInArgsImpl() const
{
  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << '\n';

  Thyra::ModelEvaluatorBase::InArgsSetup<ST>
  result;

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
Teuchos::RCP<Teuchos::ParameterList const>
LCM::SchwarzMultiscale::
getValidAppParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList>
  list = Teuchos::rcp(new Teuchos::ParameterList("ValidAppParams"));

  list->sublist("Problem", false, "Problem sublist");
  list->sublist("Debug Output", false, "Debug Output sublist");
  list->sublist("Discretization", false, "Discretization sublist");
  list->sublist("Quadrature", false, "Quadrature sublist");
  list->sublist("Regression Results", false, "Regression Results sublist");
  list->sublist("VTK", false, "DEPRECATED  VTK sublist");
  list->sublist("Piro", false, "Piro sublist");
  list->sublist("Coupled System", false, "Coupled system sublist");

  return list;
}

//Copied from QCAD::CoupledPoissonSchrodinger
//Check usage and whether neessary...
Teuchos::RCP<Teuchos::ParameterList const>
LCM::SchwarzMultiscale::
getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList>
  list = Teuchos::createParameterList("ValidCoupledSchwarzProblemParams");

  list->set<std::string>("Name", "", "String to designate Problem Class");

  list->set<int>(
      "Phalanx Graph Visualization Detail",
      0,
      "Flag to select output of Phalanx Graph and level of detail");

  //FIXME: anything else to validate?
  list->set<std::string>(
      "Solution Method",
      "Steady",
      "Flag for Steady, Transient, or Continuation");

  return list;
}
