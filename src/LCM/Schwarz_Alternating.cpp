//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include <sstream>

#include "Albany_ModelFactory.hpp"
#include "Albany_SolverFactory.hpp"
#include "Schwarz_CoupledJacobian.hpp"
#include "Schwarz_Alternating.hpp"
#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "NOXSolverPrePostOperator.h"

//uncomment the following to write stuff out to matrix market to debug
//define WRITE_TO_MATRIX_MARKET

#ifdef WRITE_TO_MATRIX_MARKET
static int mm_counter_sol = 0;
static int mm_counter_res = 0;
static int mm_counter_pre = 0;
static int mm_counter_jac = 0;
#endif // WRITE_TO_MATRIX_MARKET

namespace LCM {

SchwarzAlternating::
SchwarzAlternating(
    Teuchos::RCP<Teuchos::ParameterList> const & app_params,
    Teuchos::RCP<Teuchos::Comm<int> const> const & comm,
    Teuchos::RCP<Tpetra_Vector const> const & initial_guess,
    Teuchos::RCP<Thyra::LinearOpWithSolveFactoryBase<ST> const> const &
    lowsfb)
{
  comm_ = comm;

  lowsfb_ = lowsfb;

  Teuchos::ParameterList &
  alt_system_params = app_params->sublist("Alternating System");

  // Get names of individual model input files
  Teuchos::Array<std::string>
  model_filenames =
      alt_system_params.get<Teuchos::Array<std::string>>("Model Input Files");

  //number of models
  num_models_ = model_filenames.size();

  // Create application name-index map used for Schwarz BC.
  Teuchos::RCP<std::map<std::string, int>>
  app_name_index_map = Teuchos::rcp(new std::map<std::string, int>);

  for (auto app_index = 0; app_index < num_models_; ++app_index) {

    std::string const &
    app_name = model_filenames[app_index];

    std::pair<std::string, int>
    app_name_index = std::make_pair(app_name, app_index);

    app_name_index_map->insert(app_name_index);
  }

  //----------------Parameters------------------------
  //Get "Problem" parameter list
  Teuchos::ParameterList &
  problem_params = app_params->sublist("Problem");

  Teuchos::RCP<Teuchos::ParameterList>
  parameter_params;

  Teuchos::RCP<Teuchos::ParameterList>
  response_params;

  num_params_total_ = 0;

  //Get "Parameters" parameter sublist, if it exists
  if (problem_params.isSublist("Parameters")) {
    parameter_params = Teuchos::rcp(
        &(problem_params.sublist("Parameters")), false);

    auto const
    num_parameters =
        parameter_params->isType<int>("Number") == true ?
            parameter_params->get<int>("Number") : 0;

    bool const
    using_old_parameter_list = num_parameters > 0 ? true : false;

    num_params_total_ =
        num_parameters > 0 ?
            1 : parameter_params->get("Number of Parameter Vectors", 0);

    ALBANY_ASSERT(num_params_total_, "Parameters not supported.");

    //Get parameter names
    param_names_.resize(num_params_total_);
    for (auto l = 0; l < num_params_total_; ++l) {

      Teuchos::RCP<Teuchos::ParameterList const>
      p_list =
          using_old_parameter_list == true ?
              Teuchos::rcp(new Teuchos::ParameterList(*parameter_params)) :
              Teuchos::rcp(&(parameter_params->sublist(
                  Albany::strint("Parameter Vector", l))), false);

      auto const
      num_parameters = p_list->get<int>("Number");

      ALBANY_EXPECT(num_parameters > 0);

      param_names_[l] =
          Teuchos::rcp(new Teuchos::Array<std::string>(num_parameters));

      for (auto k = 0; k < num_parameters; ++k) {
        (*param_names_[l])[k] =
            p_list->get<std::string>(Albany::strint("Parameter", k));
      }
      std::cout << "Number of parameters in parameter vector ";
      std::cout << l << " = " << num_parameters << '\n';
    }
  }

  std::cout << "Number of parameter vectors = " << num_params_total_ << '\n';

  //---------------End Parameters---------------------

  //----------------Responses------------------------
  //Get "Response functions" parameter sublist
  if (problem_params.isSublist("Response Functions")) {
    response_params =
        Teuchos::rcp(&(problem_params.sublist("Response Functions")), false);

    auto const
    num_parameters = response_params->isType<int>("Number") == true ?
        response_params->get<int>("Number") : 0;

    ALBANY_ASSERT(num_parameters == 0, "No responses allowed.");
  }

  //----------- end Responses-----------------------

  apps_.resize(num_models_);
  models_.resize(num_models_);
  model_app_params_.resize(num_models_);

  Teuchos::Array<Teuchos::RCP<Teuchos::ParameterList>>
  model_problem_params(num_models_);

  disc_maps_.resize(num_models_);

  jacs_.resize(num_models_);

  precs_.resize(num_models_);

  Teuchos::Array<Teuchos::RCP<Tpetra_Map const>>
  disc_overlap_maps(num_models_);

  material_dbs_.resize(num_models_);

  //Set up each application and model object
  for (auto m = 0; m < num_models_; ++m) {

    //get parameterlist from mth model
    Albany::SolverFactory
    solver_factory(model_filenames[m], comm_);

    // solver_factory will go out of scope, so get a copy of the PL. We take
    // ownership and give weak pointers to everyone else.
    model_app_params_[m] = Teuchos::rcp(
        new Teuchos::ParameterList(solver_factory.getParameters()));

    Teuchos::RCP<Teuchos::ParameterList>
    problem_params_m = Teuchos::sublist(model_app_params_[m], "Problem");

    model_problem_params[m] = problem_params_m;

    std::string const &
    problem_name = problem_params_m->get("Name", "");

    std::cout << "Name of problem #" << m << ": " << problem_name << '\n';

    bool const
    have_matdb = problem_params_m->isType<std::string>("MaterialDB Filename");

    ALBANY_ASSERT(have_matdb == true, "Material database required.");

    std::string const &
    matdb_file = problem_params_m->get<std::string>("MaterialDB Filename");

    material_dbs_[m] = Teuchos::rcp(new MaterialDatabase(matdb_file, comm_));

    std::cout << "Materials #" << m << ": " << matdb_file << '\n';

    // Pass these on the parameter list because the are needed before
    // BC evaluators are built.

    // Add application array for later use in Schwarz BC.
    model_app_params_[m]->set("Application Array", apps_);

    // See application index for use with Schwarz BC.
    model_app_params_[m]->set("Application Index", m);

    // App application name-index map for later use in Schwarz BC.
    model_app_params_[m]->set("Application Name Index Map", app_name_index_map);

    //create application for mth model
    apps_[m] = Teuchos::rcp(new Albany::Application(
        comm, model_app_params_[m].create_weak(), initial_guess));

    //Create model evaluator
    Albany::ModelFactory
    model_factory(model_app_params_[m].create_weak(), apps_[m]);

    models_[m] = model_factory.createT();

    //create array of individual model jacobians
    Teuchos::RCP<Tpetra_Operator> const
    jac_temp = Teuchos::nonnull(models_[m]->create_W_op()) ?
            ConverterT::getTpetraOperator(models_[m]->create_W_op()) :
            Teuchos::null;

    jacs_[m] = Teuchos::nonnull(jac_temp) ?
            Teuchos::rcp_dynamic_cast<Tpetra_CrsMatrix>(jac_temp, true) :
            Teuchos::null;

    // create array of individual model preconditioners
    // these will have same graph as Jacobians for now
    precs_[m] = Teuchos::nonnull(jac_temp) ?
            Teuchos::rcp_dynamic_cast<Tpetra_CrsMatrix>(jac_temp, true) :
            Teuchos::null;

    if (precs_[m]->isFillActive()) {
      precs_[m]->fillComplete();
    }
  }

  //Now get maps, InArgs, OutArgs for each model.
  //Calculate how many parameters, responses there are in total.
  solver_inargs_.resize(num_models_);
  solver_outargs_.resize(num_models_);

  for (auto m = 0; m < num_models_; ++m) {
    disc_maps_[m] = apps_[m]->getMapT();

    disc_overlap_maps[m] =
        apps_[m]->getStateMgr().getDiscretization()->getOverlapMapT();

    solver_inargs_[m] = models_[m]->createInArgs();
    solver_outargs_[m] = models_[m]->createOutArgs();
  }

  //----------------Parameters------------------------
  // Create sacado parameter vectors of appropriate size
  // for use in evalModelImpl
  tpetra_param_map_.resize(num_params_total_);

  // FIXME: Copied from Schwarz Coupled. Need to revisit.
  sacado_param_vecs_.resize(num_models_);

  for (auto m = 0; m < num_models_; ++m) {
    sacado_param_vecs_[m].resize(num_params_total_);
  }

  for (auto m = 0; m < num_models_; ++m) {
    for (auto l = 0; l < num_params_total_; ++l) {
      apps_[m]->getParamLib()->fillVector<PHAL::AlbanyTraits::Residual>(
          *(param_names_[l]), sacado_param_vecs_[m][l]);
    }
  }

  //----------- end Parameters-----------------------

  //------------------Setup nominal values----------------
  nominal_values_ = this->createInArgsImpl();

  Teuchos::RCP<Thyra::DefaultProductVector<ST>>
  x = Teuchos::null;

  Teuchos::RCP<Thyra::DefaultProductVector<ST>>
  x_dot = Teuchos::null;

  nominal_values_.set_x(x);
  nominal_values_.set_x_dot(x_dot);

  // set p_init in nominal_values_ --
  // create product vector that concatenates parameters from each model.
  // TODO: Check if these are correct nominal values for parameters

  for (auto l = 0; l < num_params_total_; ++l) {

    Teuchos::RCP<Thyra::DefaultProductVector<ST>>
    p_prod_vec = Teuchos::null;

    if (Teuchos::is_null(p_prod_vec) == true) continue;

    nominal_values_.set_p(l, p_prod_vec);
  }

  //--------------End setting of nominal values------------------

}

SchwarzAlternating::
~SchwarzAlternating()
{
  return;
}

// Overridden from Thyra::ModelEvaluator<ST>
Teuchos::RCP<Thyra::VectorSpaceBase<ST> const>
SchwarzAlternating::
get_x_space() const
{
  Teuchos::RCP<Thyra::ProductVectorSpaceBase<ST>>
  unused = Teuchos::null;

  return unused;
}

Teuchos::RCP<Thyra::VectorSpaceBase<ST> const>
SchwarzAlternating::
get_f_space() const
{
  Teuchos::RCP<Thyra::ProductVectorSpaceBase<ST>>
  unused = Teuchos::null;

  return unused;
}

Teuchos::RCP<Thyra::VectorSpaceBase<ST> const>
SchwarzAlternating::
get_p_space(int l) const
{
  ALBANY_EXPECT(0 <= l && l < num_params_total_);

  std::vector<Teuchos::RCP<Thyra::VectorSpaceBase<ST> const>>
  p_space_array;

  auto
  vs = Thyra::createVectorSpace<ST, LO, GO, KokkosNode>(tpetra_param_map_[l]);

  p_space_array.push_back(vs);

  return Thyra::productVectorSpace<ST>(p_space_array);
}

Teuchos::RCP<Thyra::VectorSpaceBase<ST> const>
SchwarzAlternating::
get_g_space(int l) const
{
  ALBANY_EXPECT(0 <= l);

  Teuchos::Array<Teuchos::RCP<Thyra::VectorSpaceBase<ST> const>>
  vs_array;

  // create product space for lth response by concatenating lth response
  // from all the models.
  for (auto m = 0; m < num_models_; ++m) {
    vs_array.push_back(models_[m]->get_g_space(l));
  }

  return Thyra::productVectorSpace<ST>(vs_array);
}

Teuchos::RCP<const Teuchos::Array<std::string>>
SchwarzAlternating::
get_p_names(int l) const
{
  ALBANY_EXPECT(0 <= l && l < num_params_total_);
  return param_names_[l];
}

Teuchos::ArrayView<const std::string>
SchwarzAlternating::
get_g_names(int l) const
{
  ALBANY_ASSERT(false, "not implemented");
  return Teuchos::ArrayView<const std::string>();
}

Thyra::ModelEvaluatorBase::InArgs<ST>
SchwarzAlternating::
getNominalValues() const
{
  return nominal_values_;
}

Thyra::ModelEvaluatorBase::InArgs<ST>
SchwarzAlternating::
getLowerBounds() const
{
  return Thyra::ModelEvaluatorBase::InArgs<ST>(); // Default value
}

Thyra::ModelEvaluatorBase::InArgs<ST>
SchwarzAlternating::
getUpperBounds() const
{
  return Thyra::ModelEvaluatorBase::InArgs<ST>(); // Default value
}

Teuchos::RCP<Thyra::LinearOpBase<ST>>
SchwarzAlternating::
create_W_op() const
{
  Schwarz_CoupledJacobian
  jac(comm_);

  return jac.getThyraCoupledJacobian(jacs_, apps_);
}

Teuchos::RCP<Thyra::PreconditionerBase<ST>>
SchwarzAlternating::
create_W_prec() const
{
  Teuchos::RCP<Thyra::DefaultPreconditioner<ST>>
  W_prec = Teuchos::null;

  return W_prec;
}

Teuchos::RCP<const Thyra::LinearOpWithSolveFactoryBase<ST>>
SchwarzAlternating::
get_W_factory() const
{
  return lowsfb_;
}

Thyra::ModelEvaluatorBase::InArgs<ST>
SchwarzAlternating::
createInArgs() const
{
  return this->createInArgsImpl();
}

void
SchwarzAlternating::
reportFinalPoint(
    Thyra::ModelEvaluatorBase::InArgs<ST> const & final_point,
    bool const was_solved)
{
  ALBANY_ASSERT(false, "reportFinalPoint not allowed");
}

Teuchos::ArrayRCP<Teuchos::RCP<Albany::Application>>
SchwarzAlternating::
getApps() const
{
  return apps_;
}

/// Create operator form of dg/dx for distributed responses
Teuchos::RCP<Thyra::LinearOpBase<ST>>
SchwarzAlternating::
create_DgDx_op_impl(int j) const
{
  ALBANY_ASSERT(0 <= j);
  return Teuchos::null;
}

/// Create operator form of dg/dx_dot for distributed responses
Teuchos::RCP<Thyra::LinearOpBase<ST>>
SchwarzAlternating::
create_DgDx_dot_op_impl(int j) const
{
  ALBANY_ASSERT(0 <= j);
  return Teuchos::null;
}

/// Create OutArgs
Thyra::ModelEvaluatorBase::OutArgs<ST>
SchwarzAlternating::
createOutArgsImpl() const
{
  Thyra::ModelEvaluatorBase::OutArgsSetup<ST>
  result;

  result.setModelEvalDescription(this->description());

  //Note: it is assumed her there are no distributed parameters.
  result.set_Np_Ng(num_params_total_, 0);

  result.setSupports(Thyra::ModelEvaluatorBase::OUT_ARG_f, true);
  result.setSupports(Thyra::ModelEvaluatorBase::OUT_ARG_W_op, true);
  result.setSupports(Thyra::ModelEvaluatorBase::OUT_ARG_W_prec, false);

  result.set_W_properties(
      Thyra::ModelEvaluatorBase::DerivativeProperties(
          Thyra::ModelEvaluatorBase::DERIV_LINEARITY_UNKNOWN,
          Thyra::ModelEvaluatorBase::DERIV_RANK_FULL,
          true));

  return result;
}

/// Evaluate model on InArgs
void
SchwarzAlternating::
evalModelImpl(
    Thyra::ModelEvaluatorBase::InArgs<ST> const & in_args,
    Thyra::ModelEvaluatorBase::OutArgs<ST> const & out_args) const
{
  SchwarzLoop(in_args, out_args);
  return;
}

Thyra::ModelEvaluatorBase::InArgs<ST>
SchwarzAlternating::
createInArgsImpl() const
{
  Thyra::ModelEvaluatorBase::InArgsSetup<ST>
  result;

  result.setModelEvalDescription(this->description());

  result.setSupports(Thyra::ModelEvaluatorBase::IN_ARG_x, true);
  result.setSupports(Thyra::ModelEvaluatorBase::IN_ARG_x_dot, true);
  result.setSupports(Thyra::ModelEvaluatorBase::IN_ARG_x_dot_dot, true);
  result.setSupports(Thyra::ModelEvaluatorBase::IN_ARG_t, true);
  result.setSupports(Thyra::ModelEvaluatorBase::IN_ARG_alpha, true);
  result.setSupports(Thyra::ModelEvaluatorBase::IN_ARG_beta, true);
  result.setSupports(Thyra::ModelEvaluatorBase::IN_ARG_W_x_dot_dot_coeff, true);

  result.set_Np(num_params_total_);

  return result;
}

// Validate applicaton parameters of applications not created via a
// SolverFactory Check usage and whether necessary.
Teuchos::RCP<Teuchos::ParameterList const>
SchwarzAlternating::
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
  list->set<std::string>(
      "Matrix-Free Preconditioner",
      "",
      "Matrix-Free Preconditioner Type");

  return list;
}

// Check usage and whether neessary.
Teuchos::RCP<Teuchos::ParameterList const>
SchwarzAlternating::
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

//
// Schwarz Alternating loop
//
void
SchwarzAlternating::
SchwarzLoop(
    Thyra::ModelEvaluatorBase::InArgs<ST> const & in_args,
    Thyra::ModelEvaluatorBase::OutArgs<ST> const & out_args) const
{
  bool
  converged{false};

  while (converged == false) {

    for (auto m = 0; m < num_models_; ++m) {

      auto &
      model = *(models_[m]);

      auto
      in_args_m = model.createInArgs();

      auto
      out_args_m = model.createOutArgs();

      model.evalModel(in_args_m, out_args_m);
    }

  }

  return;
}

} // namespace LCM
