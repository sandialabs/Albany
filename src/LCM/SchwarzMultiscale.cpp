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

#ifdef WRITE_TO_MATRIX_MARKET
static
int mm_counter = 0;
#endif // WRITE_TO_MATRIX_MARKET

LCM::
SchwarzMultiscale::
SchwarzMultiscale(
    Teuchos::RCP<Teuchos::ParameterList> const & app_params,
    Teuchos::RCP<Teuchos::Comm<int> const> const & commT,
    Teuchos::RCP<Tpetra_Vector const> const & initial_guessT,
    Teuchos::RCP<Thyra::LinearOpWithSolveFactoryBase<ST> const> const &
    solver_factory)
{
  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";

  commT_ = commT;

  solver_factory_ = solver_factory;

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

  jacs_.resize(num_models_);

  //FIXME: jacs_boundary_ will be smaller than num_models_^2 in practice
  int
  num_models2 = num_models_ * num_models_;
  jacs_boundary_.resize(num_models2);

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

    // Add application array for later use in Schwarz BC.
    apps_[m]->setCoupledApplications(apps_);

    //Create model evaluator
    Albany::ModelFactory
    model_factory(model_app_params[m], apps_[m]);

    models_[m] = model_factory.createT();

    //create array of individual model jacobians
    Teuchos::RCP<Tpetra_Operator> const jac_temp =
        Teuchos::nonnull(models_[m]->create_W_op()) ?
            ConverterT::getTpetraOperator(models_[m]->create_W_op()) :
            Teuchos::null;

    jacs_[m] =
        Teuchos::nonnull(jac_temp) ?
            Teuchos::rcp_dynamic_cast<Tpetra_CrsMatrix>(jac_temp, true) :
            Teuchos::null;
  }

  //Initialize each entry of jacs_boundary_ 
  //FIXME: the loops don't need to go through num_models_*num_models_
  //FIXME: allocate array(s) of indices identifying where entries of
  // jacs_boundary_ will go
  for (int i = 0; i < num_models_; ++i) {
    for (int j = 0; j < num_models_; ++j) {
      //Check if have this term?  Put into Teuchos array?
      jacs_boundary_[i * num_models_ + j] = Teuchos::rcp(
          new LCM::Schwarz_BoundaryJacobian(commT_, apps_));
    }
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

// Overridden from Thyra::ModelEvaluator<ST>
Teuchos::RCP<Thyra::VectorSpaceBase<ST> const>
LCM::SchwarzMultiscale::get_x_space() const
{
  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";
  return getThyraDomainSpace();
}

Teuchos::RCP<Thyra::VectorSpaceBase<ST> const>
LCM::SchwarzMultiscale::get_f_space() const
{
  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";
  return getThyraRangeSpace();
}

Teuchos::RCP<const Thyra::VectorSpaceBase<ST> >
LCM::SchwarzMultiscale::getThyraRangeSpace() const
{
  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";
  if (range_space_ == Teuchos::null) {
    // loop over all vectors and build the vector space
    std::vector<Teuchos::RCP<const Thyra::VectorSpaceBase<ST> > > vsArray;
    for (std::size_t i = 0; i < num_models_; i++)
      vsArray.push_back(
          Thyra::createVectorSpace<ST, LO, GO, KokkosNode>(disc_maps_[i]));
    range_space_ = Thyra::productVectorSpace<ST>(vsArray);
  }
  return range_space_;
}

Teuchos::RCP<const Thyra::VectorSpaceBase<ST> >
LCM::SchwarzMultiscale::getThyraDomainSpace() const
{
  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";
  if (domain_space_ == Teuchos::null) {
    // loop over all vectors and build the vector space
    std::vector<Teuchos::RCP<const Thyra::VectorSpaceBase<ST> > > vsArray;
    for (std::size_t i = 0; i < num_models_; i++)
      vsArray.push_back(
          Thyra::createVectorSpace<ST, LO, GO, KokkosNode>(disc_maps_[i]));
    domain_space_ = Thyra::productVectorSpace<ST>(vsArray);
  }
  return domain_space_;
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

  LCM::Schwarz_CoupledJacobian csJac(commT_);
  return csJac.getThyraCoupledJacobian(jacs_, jacs_boundary_);
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

  return solver_factory_;
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

  Teuchos::Array<Teuchos::RCP<const Thyra::VectorSpaceBase<ST> > > spaces(
      num_models_);
  for (int m = 0; m < num_models_; m++)
    spaces[m] = Thyra::createVectorSpace<ST>(disc_maps_[m]);
  Teuchos::RCP<const Thyra::DefaultProductVectorSpace<ST> >
  space = Thyra::productVectorSpace<ST>(spaces);

  std::vector<Teuchos::RCP<Thyra::VectorBase<ST> > >
  xT_vecs;

  std::vector<Teuchos::RCP<Thyra::VectorBase<ST> > >
  x_dotT_vecs;

  xT_vecs.resize(num_models_);
  x_dotT_vecs.resize(num_models_);

  for (int m = 0; m < num_models_; m++) {
    Teuchos::RCP<Tpetra_Vector>
    xT_vec = Teuchos::rcp(new Tpetra_Vector(*apps_[m]->getInitialSolutionT()));

    Teuchos::RCP<Tpetra_Vector>
    x_dotT_vec = Teuchos::rcp(
        new Tpetra_Vector(*apps_[m]->getInitialSolutionDotT()));

    xT_vecs[m] = Thyra::createVector(xT_vec, spaces[m]);
    x_dotT_vecs[m] = Thyra::createVector(x_dotT_vec, spaces[m]);
  }

  Teuchos::ArrayView<const Teuchos::RCP<Thyra::VectorBase<ST> > >
  xT_vecs_AV = Teuchos::arrayViewFromVector(xT_vecs);

  Teuchos::ArrayView<const Teuchos::RCP<Thyra::VectorBase<ST> > >
  x_dotT_vecs_AV = Teuchos::arrayViewFromVector(x_dotT_vecs);

  Teuchos::RCP<Thyra::DefaultProductVector<ST> >
  xT_prod_vec = Thyra::defaultProductVector<ST>(space, xT_vecs_AV);

  Teuchos::RCP<Thyra::DefaultProductVector<ST> >
  x_dotT_prod_vec = Thyra::defaultProductVector<ST>(space, x_dotT_vecs_AV);

  nominal_values_.set_x(xT_prod_vec);
  nominal_values_.set_x_dot(x_dotT_prod_vec);

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

  //Get xT and x_dotT from in_args
  Teuchos::RCP<const Thyra::ProductVectorBase<ST> > xT =
      Teuchos::rcp_dynamic_cast<const Thyra::ProductVectorBase<ST> >(
          in_args.get_x(),
          true);

  Teuchos::RCP<const Thyra::ProductVectorBase<ST> > x_dotT =
      Teuchos::nonnull(in_args.get_x_dot()) ?
          Teuchos::rcp_dynamic_cast<const Thyra::ProductVectorBase<ST> >(
              in_args.get_x_dot(),
              true) :
          Teuchos::null;

  // Create a Teuchos array of the xT and x_dotT for each model,
  // casting to Tpetra_Vector
  Teuchos::Array<Teuchos::RCP<const Tpetra_Vector> >
  xTs(num_models_);

  Teuchos::Array<Teuchos::RCP<const Tpetra_Vector> >
  x_dotTs(num_models_);

  for (int m = 0; m < num_models_; m++) {
    //Get each Tpetra vector
    xTs[m] = Teuchos::rcp_dynamic_cast<const ThyraVector>(
        xT->getVectorBlock(m),
        true)->getConstTpetraVector();
  }
  if (x_dotT != Teuchos::null) {
    for (int m = 0; m < num_models_; m++) {
      //Get each Tpetra vector
      x_dotTs[m] = Teuchos::rcp_dynamic_cast<const ThyraVector>(
          x_dotT->getVectorBlock(m),
          true)->getConstTpetraVector();
    }
  }

  // AGS: x_dotdot time integrators not imlemented in Thyra ME yet
  Teuchos::RCP<Tpetra_Vector const> const x_dotdotT = Teuchos::null;

  double const
  alpha =
      (Teuchos::nonnull(x_dotT) || Teuchos::nonnull(x_dotdotT)) ?
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
  Teuchos::RCP<Thyra::ProductVectorBase<ST> > fT_out =
      Teuchos::nonnull(out_args.get_f()) ?
          Teuchos::rcp_dynamic_cast<Thyra::ProductVectorBase<ST> >(
              out_args.get_f(),
              true) :
          Teuchos::null;

  //Create a Teuchos array of the fT_out for each model.
  Teuchos::Array<Teuchos::RCP<Tpetra_Vector> > fTs_out(num_models_);
  if (fT_out != Teuchos::null) {
    for (int m = 0; m < num_models_; m++) {
      //Get each Tpetra vector
      fTs_out[m] = Teuchos::rcp_dynamic_cast<ThyraVector>(
          fT_out->getNonconstVectorBlock(m),
          true)->getTpetraVector();
    }
  }

  Teuchos::RCP<Thyra::LinearOpBase<ST> >
  W_op_outT = Teuchos::nonnull(out_args.get_W_op()) ?
                                                      out_args.get_W_op() :
                                                      Teuchos::null;

  // Compute the functions

  Teuchos::Array<bool> fs_already_computed(num_models_, false);

  // W matrix for each individual model
  for (int m = 0; m < num_models_; ++m) {
    if (Teuchos::nonnull(W_op_outT)) {
      //computeGlobalJacobianT sets fTs_out[m] and jacs_[m]
      apps_[m]->computeGlobalJacobianT(
          alpha, beta, omega, curr_time,
          x_dotTs[m].get(), x_dotdotT.get(), *xTs[m],
          sacado_param_vecs_[m], fTs_out[m].get(), *jacs_[m]);

      fs_already_computed[m] = true;
    }
  }

  // FIXME: create coupled W matrix from array of model W matrices
  if (W_op_outT != Teuchos::null) {
    //FIXME: create boundary operators 
    for (int i = 0; i < jacs_boundary_.size(); ++i) {
      //FIXME: initialize will have arguments (index array?)!
      jacs_boundary_[i]->initialize();
    }

    LCM::Schwarz_CoupledJacobian csJac(commT_);
    //FIXME: add boundary operators array to coupled Jacobian parameter list 
    W_op_outT = csJac.getThyraCoupledJacobian(jacs_, jacs_boundary_);
  }

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
      dfdp_outT =
          Teuchos::nonnull(dfdp_out) ?
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

  // f
  for (int m = 0; m < num_models_; ++m) {
    if (apps_[m]->is_adjoint) {
      Thyra::ModelEvaluatorBase::Derivative<ST> const
      f_derivT(solver_outargs_[m].get_f(),
          Thyra::ModelEvaluatorBase::DERIV_TRANS_MV_BY_ROW);

      Thyra::ModelEvaluatorBase::Derivative<ST> const dummy_derivT;

      // need to add capability for sending this in
      int const
      response_index = 0;

      apps_[m]->evaluateResponseDerivativeT(
          response_index, curr_time, x_dotTs[m].get(), x_dotdotT.get(), *xTs[m],
          sacado_param_vecs_[m], NULL,
          NULL, f_derivT, dummy_derivT, dummy_derivT, dummy_derivT);
    }
    else {
      if (Teuchos::nonnull(fTs_out[m]) && !fs_already_computed[m]) {
        apps_[m]->computeGlobalResidualT(
            curr_time, x_dotTs[m].get(), x_dotdotT.get(), *xTs[m],
            sacado_param_vecs_[m], *fTs_out[m]);
      }
    }
  }

#ifdef WRITE_TO_MATRIX_MARKET
  //writing to MatrixMarket file for debug
  char name[100];  //create string for file name
  sprintf(name, "f0_%i.mm", mm_counter);
  if (fTs_out[0] != Teuchos::null) {
    Tpetra_MatrixMarket_Writer::writeDenseFile(
        name,
        *(fTs_out[0]));
  }
  if (num_models_ > 1 && fTs_out[1] != Teuchos::null) {
    sprintf(name, "f1_%i.mm", mm_counter);
    Tpetra_MatrixMarket_Writer::writeDenseFile(
        name,
        *(fTs_out[1]));
  }
  sprintf(name, "x0_%i.mm", mm_counter);
  Tpetra_MatrixMarket_Writer::writeDenseFile(name, *(xTs[0]));
  if (num_models_ > 1) {
    sprintf(name, "x1_%i.mm", mm_counter);
    Tpetra_MatrixMarket_Writer::writeDenseFile(name, *(xTs[1]));
  }
  mm_counter++;
#endif

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
        dgdpT_out =
            Teuchos::nonnull(dgdp_out) ?
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
