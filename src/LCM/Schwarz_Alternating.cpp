//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Albany_ModelFactory.hpp"
#include "Albany_SolverFactory.hpp"
#include "Albany_STKDiscretization.hpp"
#include "MiniTensor.h"
#include "Teuchos_FancyOStream.hpp"
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
  Teuchos::ParameterList &
  alt_system_params = app_params->sublist("Alternating System");

  // Get names of individual model input files
  Teuchos::Array<std::string>
  model_filenames =
      alt_system_params.get<Teuchos::Array<std::string>>("Model Input Files");

  min_iters_ = alt_system_params.get<int>("Minimum Iterations", 1);
  max_iters_ = alt_system_params.get<int>("Maximum Iterations", 1024);
  rel_tol_ = alt_system_params.get<ST>("Relative Tolerance", 1.0e-08);
  abs_tol_ = alt_system_params.get<ST>("Absolute Tolerance", 1.0e-08);
  output_interval_ = alt_system_params.get<int>("Exodus Write Interval", 1);

  //number of models
  num_subdomains_ = model_filenames.size();

  // Create application name-index map used for Schwarz BC.
  Teuchos::RCP<std::map<std::string, int>>
  app_name_index_map = Teuchos::rcp(new std::map<std::string, int>);

  // Arrays to cache useful info for each subdomain for later use
  apps_.resize(num_subdomains_);
  solvers_.resize(num_subdomains_);
  convergence_ops_.resize(num_subdomains_);
  stk_mesh_structs_.resize(num_subdomains_);

  //Set up each application and model object
  for (auto subdomain = 0; subdomain < num_subdomains_; ++subdomain) {

    std::string const &
    app_name = model_filenames[subdomain];

    std::pair<std::string, int>
    app_name_index = std::make_pair(app_name, subdomain);

    app_name_index_map->insert(app_name_index);

    //get parameterlist from each model
    Albany::SolverFactory
    solver_factory(model_filenames[subdomain], comm);

    Teuchos::ParameterList &
    params = solver_factory.getParameters();

    // Add application array for later use in Schwarz BC.
    params.set("Application Array", apps_);

    // See application index for use with Schwarz BC.
    params.set("Application Index", subdomain);

    // App application name-index map for later use in Schwarz BC.
    params.set("Application Name Index Map", app_name_index_map);

    // Add NOX pre-post-operator for Schwarz loop convergence criterion.
    bool const
    have_piro = params.isSublist("Piro");

    ALBANY_ASSERT(have_piro == true);

    Teuchos::ParameterList &
    piro_params = params.sublist("Piro");

    bool const
    have_nox = piro_params.isSublist("NOX");

    ALBANY_ASSERT(have_nox == true);

    Teuchos::ParameterList &
    nox_params = piro_params.sublist("NOX");

    bool const
    have_solver_opts = nox_params.isSublist("Solver Options");

    ALBANY_ASSERT(have_solver_opts == true);

    Teuchos::ParameterList &
    solver_opts = nox_params.sublist("Solver Options");

    std::string const
    ppo_str{"User Defined Pre/Post Operator"};

    bool const
    have_ppo = solver_opts.isParameter(ppo_str);

    Teuchos::RCP<NOX::Abstract::PrePostOperator>
    ppo{Teuchos::null};

    if (have_ppo == true) {
      ppo = solver_opts.get<decltype(ppo)>(ppo_str);
    } else {
      ppo = Teuchos::rcp(new NOXSolverPrePostOperator);
      solver_opts.set(ppo_str, ppo);
      ALBANY_ASSERT(solver_opts.isParameter(ppo_str) == true);
    }

    Teuchos::RCP<NOXSolverPrePostOperator>
    convergence_op = Teuchos::rcp_dynamic_cast<NOXSolverPrePostOperator>(ppo);

    convergence_ops_[subdomain] = convergence_op;

    Teuchos::RCP<Albany::Application>
    app{Teuchos::null};

    Teuchos::RCP<Thyra::ResponseOnlyModelEvaluatorBase<ST>>
    solver = solver_factory.createAndGetAlbanyAppT(app, comm, comm);

    solvers_[subdomain] = solver;

    apps_[subdomain] = app;

    // Get STK mesh structs to control Exodus output interval
    Teuchos::RCP<Albany::AbstractDiscretization>
    disc = app->getDiscretization();

    Albany::STKDiscretization &
    stk_disc = *static_cast<Albany::STKDiscretization *>(disc.get());

    Teuchos::RCP<Albany::AbstractSTKMeshStruct>
    ams = stk_disc.getSTKMeshStruct();

    stk_mesh_structs_[subdomain] = ams;
  }

  //
  // Setup nominal values
  //
  nominal_values_ = this->createInArgsImpl();
  nominal_values_.set_x(Teuchos::null);
  nominal_values_.set_x_dot(Teuchos::null);
  nominal_values_.set_x_dot_dot(Teuchos::null);

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

  ALBANY_ASSERT(have_responses == false, "Responses not supported.");

  return;
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

  sub_inargs_.resize(num_subdomains_);
  for (auto subdomain = 0; subdomain < num_subdomains_; ++subdomain) {
    sub_inargs_[subdomain] = solvers_[subdomain]->createInArgs();
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

  sub_outargs_.resize(num_subdomains_);
  for (auto subdomain = 0; subdomain < num_subdomains_; ++subdomain) {
    sub_outargs_[subdomain] = solvers_[subdomain]->createOutArgs();
  }

  return result;
}

//
// Evaluate model on InArgs
//
void
SchwarzAlternating::
evalModelImpl(
    Thyra::ModelEvaluatorBase::InArgs<ST> const &,
    Thyra::ModelEvaluatorBase::OutArgs<ST> const &) const
{
  SchwarzLoop();
  return;
}

namespace {

std::string
centered(std::string const & str, int width)
{
  assert(width >= 0);

  int const
  length = static_cast<int>(str.size());

  int const
  padding = width - length;

  if (padding <= 0) return str;

  int const
  left = padding / 2;

  int const
  right = padding - left;

  return std::string(left, ' ') + str + std::string(right, ' ');
}

} // anonymous

//
// Schwarz Alternating loop
//
void
SchwarzAlternating::
SchwarzLoop() const
{
  minitensor::Vector<ST>
  norms_init(num_subdomains_, minitensor::Filler::ZEROS);

  minitensor::Vector<ST>
  norms_final(num_subdomains_, minitensor::Filler::ZEROS);

  minitensor::Vector<ST>
  norms_diff(num_subdomains_, minitensor::Filler::ZEROS);

  int const
  iter_limit = std::max(min_iters_, max_iters_);

  std::string const
  delim(72, '=');

  Teuchos::FancyOStream &
  fos{*Teuchos::VerboseObjectBase::getDefaultOStream()};

  fos << delim << '\n';
  fos << "Schwarz Alternating Method with " << num_subdomains_;
  fos << " subdomains\n";
  fos << std::scientific << std::setprecision(17);

  for (auto iter = 0; iter < iter_limit; ++iter) {

    for (auto subdomain = 0; subdomain < num_subdomains_; ++subdomain) {

      fos << delim << '\n';
      fos << "Schwarz iteration  :" << iter << '\n';
      fos << "Subdomain          :" << subdomain << '\n';
      fos << delim << '\n';

      Thyra::ResponseOnlyModelEvaluatorBase<ST> &
      solver = *(solvers_[subdomain]);

      Thyra::ModelEvaluatorBase::InArgs<ST> &
      in_args = sub_inargs_[subdomain];

      Thyra::ModelEvaluatorBase::OutArgs<ST> &
      out_args = sub_outargs_[subdomain];

      solver.evalModel(in_args, out_args);

      Teuchos::RCP<NOXSolverPrePostOperator>
      convergence_op = convergence_ops_[subdomain];

      norms_init(subdomain) = convergence_op->getInitialNorm();
      norms_final(subdomain) = convergence_op->getFinalNorm();
      norms_diff(subdomain) = convergence_op->getDifferenceNorm();

      // Ouput handling
      Albany::Application const &
      app = *apps_[subdomain];

      Teuchos::RCP<Albany::AbstractDiscretization>
      disc = app.getDiscretization();

      Albany::STKDiscretization &
      stk_disc = *static_cast<Albany::STKDiscretization *>(disc.get());

      Albany::AbstractSTKMeshStruct &
      ams = *stk_disc.getSTKMeshStruct();

      ams.exoOutput = false;
   }


    ST const
    norm_init = minitensor::norm(norms_init);

    ST const
    norm_final = minitensor::norm(norms_final);

    ST const
    norm_diff = minitensor::norm(norms_diff);

    ST const
    abs_error = norm_diff;

    ST const
    rel_error = norm_final > 0.0 ? norm_diff / norm_final : norm_diff;

    fos << delim << '\n';
    fos << "Schwarz iteration         :" << iter << '\n';

    std::string const
    line(72, '-');

    fos << line << '\n';

    fos << centered("Sub", 4);
    fos << centered("Initial norm", 24);
    fos << centered("Final norm", 24);
    fos << centered("Difference norm", 24);
    fos << '\n';

    fos << centered("dom", 4);
    fos << centered("||X0||", 24);
    fos << centered("||Xf||", 24);
    fos << centered("||Xf-X0||", 24);
    fos << '\n';

    fos << line << '\n';

    for (auto m = 0; m < num_subdomains_; ++m) {
      fos << std::setw(4) << m;
      fos << std::setw(24) << norms_init(m);
      fos << std::setw(24) << norms_final(m);
      fos << std::setw(24) << norms_diff(m);
      fos << '\n';
    }

    fos << line << '\n';

    fos << centered("Norm", 4);
    fos << std::setw(24) << norm_init;
    fos << std::setw(24) << norm_final;
    fos << std::setw(24) << norm_diff;
    fos << '\n';

    fos << line << '\n';

    fos << "Absolute error     :" << abs_error << '\n';
    fos << "Absolute tolerance :" << abs_tol_ << '\n';
    fos << "Relative error     :" << rel_error << '\n';
    fos << "Relative tolerance :" << rel_tol_ << '\n';
    fos << delim << '\n';

    if (abs_error < abs_tol_ || rel_error < rel_tol_) {
      fos << "Schwarz loop converged.\n";
      fos << "Total iterations   :" << iter + 1 << '\n';
      fos << "Last absolute error:" << abs_error << '\n';
      fos << "Absolute tolerance :" << abs_tol_ << '\n';
      fos << "Last relative error:" << rel_error << '\n';
      fos << "Relative tolerance :" << rel_tol_ << '\n';
      fos << delim << '\n';
      break;
    }
  }

  return;
}

} // namespace LCM
