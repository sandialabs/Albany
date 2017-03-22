//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Albany_ModelFactory.hpp"
#include "Albany_SolverFactory.hpp"
#include "MiniTensor.h"
#include "Schwarz_Alternating.hpp"

namespace LCM {

//
//
//
SchwarzAlternating::
SchwarzAlternating(
    Teuchos::RCP<Teuchos::ParameterList> const & app_params,
    Teuchos::RCP<Teuchos::Comm<int> const> const & comm,
    Teuchos::RCP<Tpetra_Vector const> const & initial_guess)
{
  comm_ = comm;

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

  //
  // Parameters
  //
  Teuchos::ParameterList &
  problem_params = app_params->sublist("Problem");

  bool const
  have_parameters = problem_params.isSublist("Parameters");

  ALBANY_ASSERT(have_parameters == false, "Parameters not supported.");

  //
  // Responses
  //
  bool const
  have_responses = problem_params.isSublist("Response Functions");

  ALBANY_ASSERT(have_responses == false, "No responses allowed.");

  //
  //
  //
  apps_.resize(num_models_);
  solvers_.resize(num_models_);

  //Set up each application and model object
  for (auto m = 0; m < num_models_; ++m) {

    //get parameterlist from mth model
    Albany::SolverFactory
    solver_factory(model_filenames[m], comm);

    Teuchos::ParameterList &
    solver_factory_params = solver_factory.getParameters();

    // Add application array for later use in Schwarz BC.
    solver_factory_params.set("Application Array", apps_);

    // See application index for use with Schwarz BC.
    solver_factory_params.set("Application Index", m);

    // App application name-index map for later use in Schwarz BC.
    solver_factory_params.set("Application Name Index Map", app_name_index_map);

    Teuchos::RCP<Albany::Application>
    app;

    Teuchos::RCP<Thyra::ResponseOnlyModelEvaluatorBase<ST>>
    solver = solver_factory.createAndGetAlbanyAppT(app, comm, comm);

    solvers_[m] = solver;

    apps_[m] = app;
  }

  //
  // Setup nominal values
  //
  nominal_values_ = this->createInArgsImpl();
  nominal_values_.set_x(Teuchos::null);
  nominal_values_.set_x_dot(Teuchos::null);
}

//
//
//
SchwarzAlternating::
~SchwarzAlternating()
{
  return;
}

//
//
//
Teuchos::RCP<Thyra::VectorSpaceBase<ST> const>
SchwarzAlternating::
get_x_space() const
{
  return Teuchos::null;
}

//
//
//
Teuchos::RCP<Thyra::VectorSpaceBase<ST> const>
SchwarzAlternating::
get_f_space() const
{
  return Teuchos::null;
}

//
//
//
Teuchos::RCP<Thyra::VectorSpaceBase<ST> const>
SchwarzAlternating::
get_p_space(int) const
{
  return Teuchos::null;
}

//
//
//
Teuchos::RCP<Thyra::VectorSpaceBase<ST> const>
SchwarzAlternating::
get_g_space(int) const
{
  return Teuchos::null;
}

//
//
//
Teuchos::RCP<const Teuchos::Array<std::string>>
SchwarzAlternating::
get_p_names(int) const
{
  return Teuchos::null;
}

//
//
//
Teuchos::ArrayView<const std::string>
SchwarzAlternating::
get_g_names(int l) const
{
  ALBANY_ASSERT(false, "not implemented");
  return Teuchos::ArrayView<const std::string>();
}

//
//
//
Thyra::ModelEvaluatorBase::InArgs<ST>
SchwarzAlternating::
getNominalValues() const
{
  return nominal_values_;
}

//
//
//
Thyra::ModelEvaluatorBase::InArgs<ST>
SchwarzAlternating::
getLowerBounds() const
{
  return Thyra::ModelEvaluatorBase::InArgs<ST>(); // Default value
}

//
//
//
Thyra::ModelEvaluatorBase::InArgs<ST>
SchwarzAlternating::
getUpperBounds() const
{
  return Thyra::ModelEvaluatorBase::InArgs<ST>(); // Default value
}

//
//
//
Teuchos::RCP<Thyra::LinearOpBase<ST>>
SchwarzAlternating::
create_W_op() const
{
  return Teuchos::null;
}

//
//
//
Teuchos::RCP<Thyra::PreconditionerBase<ST>>
SchwarzAlternating::
create_W_prec() const
{
  return Teuchos::null;
}

//
//
//
Teuchos::RCP<const Thyra::LinearOpWithSolveFactoryBase<ST>>
SchwarzAlternating::
get_W_factory() const
{
  return Teuchos::null;
}

//
//
//
Thyra::ModelEvaluatorBase::InArgs<ST>
SchwarzAlternating::
createInArgs() const
{
  return this->createInArgsImpl();
}

//
//
//
Teuchos::ArrayRCP<Teuchos::RCP<Albany::Application>>
SchwarzAlternating::
getApps() const
{
  return apps_;
}

//
// Create operator form of dg/dx for distributed responses
//
Teuchos::RCP<Thyra::LinearOpBase<ST>>
SchwarzAlternating::
create_DgDx_op_impl(int j) const
{
  return Teuchos::null;
}

//
// Create operator form of dg/dx_dot for distributed responses
//
Teuchos::RCP<Thyra::LinearOpBase<ST>>
SchwarzAlternating::
create_DgDx_dot_op_impl(int j) const
{
  return Teuchos::null;
}

//
// Create InArgs
//
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

  sub_inargs_.resize(num_models_);
  for (auto m = 0; m < num_models_; ++m) {
    sub_inargs_[m] = solvers_[m]->createInArgs();
  }

  return result;
}

//
// Create OutArgs
//
Thyra::ModelEvaluatorBase::OutArgs<ST>
SchwarzAlternating::
createOutArgsImpl() const
{
  Thyra::ModelEvaluatorBase::OutArgsSetup<ST>
  result;

  result.setModelEvalDescription(this->description());

  result.setSupports(Thyra::ModelEvaluatorBase::OUT_ARG_f, true);
  result.setSupports(Thyra::ModelEvaluatorBase::OUT_ARG_W_op, true);
  result.setSupports(Thyra::ModelEvaluatorBase::OUT_ARG_W_prec, false);

  result.set_W_properties(
      Thyra::ModelEvaluatorBase::DerivativeProperties(
          Thyra::ModelEvaluatorBase::DERIV_LINEARITY_UNKNOWN,
          Thyra::ModelEvaluatorBase::DERIV_RANK_FULL,
          true));

  sub_outargs_.resize(num_models_);
  for (auto m = 0; m < num_models_; ++m) {
    sub_outargs_[m] = solvers_[m]->createOutArgs();
  }

  return result;
}

//
// Evaluate model on InArgs
//
void
SchwarzAlternating::
evalModelImpl(
    Thyra::ModelEvaluatorBase::InArgs<ST> const & in_args,
    Thyra::ModelEvaluatorBase::OutArgs<ST> const & out_args) const
{
  SchwarzLoop(in_args, out_args);
  return;
}

//
// Validate applicaton parameters of applications not created via a
// SolverFactory Check usage and whether necessary.
//
Teuchos::RCP<Teuchos::ParameterList const>
SchwarzAlternating::
getValidAppParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList>
  list = Teuchos::rcp(new Teuchos::ParameterList("ValidAppParams"));

  list->sublist("Problem", false, "Problem sublist");
  list->sublist("Debug Output", false, "Debug Output sublist");
  list->sublist("Alternating System", false, "Alternating system sublist");

  return list;
}

//
// Check usage and whether necessary.
//
Teuchos::RCP<Teuchos::ParameterList const>
SchwarzAlternating::
getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList>
  list = Teuchos::createParameterList("ValidSchwarzAlternatingProblemParams");

  list->set<std::string>("Name", "", "String to designate Problem Class");

  list->set<int>(
      "Phalanx Graph Visualization Detail",
      0,
      "Phalanx Graph and level of detail");

  //FIXME: anything else to validate?
  list->set<std::string>(
      "Solution Method",
      "Steady",
      "Steady, Transient, or Continuation");

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
  constexpr ST
  abs_tol = 1.0e-12;

  constexpr ST
  rel_tol = 1.0e-12;

  minitensor::Vector<ST>
  norms_diff(num_models_, minitensor::ZEROS);

  minitensor::Vector<ST>
  norms_soln(num_models_, minitensor::ZEROS);

  bool
  converged{false};

  int const
  max_iter = 64;

  int
  num_iter = 0;

  while (converged == false) {

    for (auto m = 0; m < num_models_; ++m) {

      Thyra::ResponseOnlyModelEvaluatorBase<ST> &
      solver = *(solvers_[m]);

      Thyra::ModelEvaluatorBase::InArgs<ST> &
      in_args = sub_inargs_[m];

      Thyra::ModelEvaluatorBase::OutArgs<ST> &
      out_args = sub_outargs_[m];

      solver.evalModel(in_args, out_args);

      norms_soln(m) = 0.0;
      norms_diff(m) = 0.0;
    }

    ST const
    norm_soln = minitensor::norm(norms_soln);

    ST const
    norm_diff = minitensor::norm(norms_diff);

    ST const
    abs_error = norm_diff;

    ST const
    rel_error = norm_soln > 0.0 ? norm_diff / norm_soln : norm_diff;

    ++num_iter;

    converged = num_iter >= max_iter;
  }

  return;
}

} // namespace LCM
