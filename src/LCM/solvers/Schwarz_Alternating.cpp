//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Schwarz_Alternating.hpp"
#include "Albany_ModelFactory.hpp"
#include "Albany_STKDiscretization.hpp"
#include "Albany_SolverFactory.hpp"
#include "MiniTensor.h"
#include "Piro_LOCASolver.hpp"
#include "Piro_TempusSolver.hpp"

//#define DEBUG

namespace LCM {

//
//
//
SchwarzAlternating::SchwarzAlternating(
    Teuchos::RCP<Teuchos::ParameterList> const&   app_params,
    Teuchos::RCP<Teuchos::Comm<int> const> const& comm,
    Teuchos::RCP<Tpetra_Vector const> const&      initial_guess)
{
  Teuchos::ParameterList& alt_system_params =
      app_params->sublist("Alternating System");

  // Get names of individual model input files
  Teuchos::Array<std::string> model_filenames =
      alt_system_params.get<Teuchos::Array<std::string>>("Model Input Files");

  min_iters_         = alt_system_params.get<int>("Minimum Iterations", 1);
  max_iters_         = alt_system_params.get<int>("Maximum Iterations", 1024);
  rel_tol_           = alt_system_params.get<ST>("Relative Tolerance", 1.0e-08);
  abs_tol_           = alt_system_params.get<ST>("Absolute Tolerance", 1.0e-08);
  maximum_steps_     = alt_system_params.get<int>("Maximum Steps", 0);
  initial_time_      = alt_system_params.get<ST>("Initial Time", 0.0);
  final_time_        = alt_system_params.get<ST>("Final Time", 0.0);
  initial_time_step_ = alt_system_params.get<ST>("Initial Time Step", 1.0);

  auto const dt  = initial_time_step_;
  auto const dt2 = dt * dt;

  min_time_step_    = alt_system_params.get<ST>("Minimum Time Step", dt);
  max_time_step_    = alt_system_params.get<ST>("Maximum Time Step", dt);
  reduction_factor_ = alt_system_params.get<ST>("Reduction Factor", 1.0);
  increase_factor_  = alt_system_params.get<ST>("Increase Factor", 1.0);
  output_interval_  = alt_system_params.get<int>("Exodus Write Interval", 1);
  std_init_guess_ =
      alt_system_params.get<bool>("Standard Initial Guess", false);

  tol_factor_vel_ = alt_system_params.get<ST>("Tolerance Factor Velocity", dt);
  tol_factor_acc_ =
      alt_system_params.get<ST>("Tolerance Factor Acceleration", dt2);

  std::string convergence_str =
      alt_system_params.get<std::string>("Convergence Criterion", "BOTH");

  std::transform(
      convergence_str.begin(),
      convergence_str.end(),
      convergence_str.begin(),
      ::toupper);

  if (convergence_str == "ABSOLUTE") {
    criterion_ = ConvergenceCriterion::ABSOLUTE;
  } else if (convergence_str == "RELATIVE") {
    criterion_ = ConvergenceCriterion::RELATIVE;
  } else if (convergence_str == "BOTH") {
    criterion_ = ConvergenceCriterion::BOTH;
  } else {
    ALBANY_ASSERT(false, "Unknown Convergence Criterion");
  }

  std::string operator_str =
      alt_system_params.get<std::string>("Convergence Operator", "AND");

  std::transform(
      operator_str.begin(),
      operator_str.end(),
      operator_str.begin(),
      ::toupper);

  if (operator_str == "AND") {
    operator_ = ConvergenceLogicalOperator::AND;
  } else if (operator_str == "OR") {
    operator_ = ConvergenceLogicalOperator::OR;
  } else {
    ALBANY_ASSERT(false, "Unknown Convergence Logical Operator");
  }

  // Firewalls
  ALBANY_ASSERT(min_iters_ >= 1);
  ALBANY_ASSERT(max_iters_ >= 1);
  ALBANY_ASSERT(max_iters_ >= min_iters_);
  ALBANY_ASSERT(rel_tol_ >= 0.0);
  ALBANY_ASSERT(abs_tol_ >= 0.0);
  ALBANY_ASSERT(maximum_steps_ >= 1);
  ALBANY_ASSERT(final_time_ >= initial_time_);
  ALBANY_ASSERT(initial_time_step_ > 0.0);
  ALBANY_ASSERT(max_time_step_ > 0.0);
  ALBANY_ASSERT(min_time_step_ > 0.0);
  ALBANY_ASSERT(max_time_step_ >= min_time_step_);
  ALBANY_ASSERT(reduction_factor_ <= 1.0);
  ALBANY_ASSERT(reduction_factor_ > 0.0);
  ALBANY_ASSERT(increase_factor_ >= 1.0);
  ALBANY_ASSERT(output_interval_ >= 1);

  // number of models
  num_subdomains_ = model_filenames.size();

  // Create application name-index map used for Schwarz BC.
  Teuchos::RCP<std::map<std::string, int>> app_name_index_map =
      Teuchos::rcp(new std::map<std::string, int>);

  for (auto subdomain = 0; subdomain < num_subdomains_; ++subdomain) {
    std::string const& app_name = model_filenames[subdomain];

    std::pair<std::string, int> app_name_index =
        std::make_pair(app_name, subdomain);

    app_name_index_map->insert(app_name_index);
  }

  // Arrays to cache useful info for each subdomain for later use
  apps_.resize(num_subdomains_);
  solvers_.resize(num_subdomains_);
  stk_mesh_structs_.resize(num_subdomains_);
  discs_.resize(num_subdomains_);
  model_evaluators_.resize(num_subdomains_);
  sub_inargs_.resize(num_subdomains_);
  sub_outargs_.resize(num_subdomains_);
  curr_disp_.resize(num_subdomains_);
  prev_step_disp_.resize(num_subdomains_);
  internal_states_.resize(num_subdomains_);
  // the following 9 arrays are for dynamics
  ics_disp_.resize(num_subdomains_);
  ics_velo_.resize(num_subdomains_);
  ics_acce_.resize(num_subdomains_);
  prev_disp_.resize(num_subdomains_);
  prev_velo_.resize(num_subdomains_);
  prev_acce_.resize(num_subdomains_);
  this_disp_.resize(num_subdomains_);
  this_velo_.resize(num_subdomains_);
  this_acce_.resize(num_subdomains_);
  do_outputs_.resize(num_subdomains_);
  do_outputs_init_.resize(num_subdomains_);

  bool is_static{false};

  bool is_dynamic{false};

  // Initialization
  for (auto subdomain = 0; subdomain < num_subdomains_; ++subdomain) {
    // Get parameters for each subdomain
    Albany::SolverFactory solver_factory(model_filenames[subdomain], comm);

    solver_factory.setSchwarz(true);

    Teuchos::ParameterList& params = solver_factory.getParameters();

    // Add application array for later use in Schwarz BC.
    params.set("Application Array", apps_);

    // See application index for use with Schwarz BC.
    params.set("Application Index", subdomain);

    // Add application name-index map for later use in Schwarz BC.
    params.set("Application Name Index Map", app_name_index_map);

    // Add NOX pre-post-operator for Schwarz loop convergence criterion.
    bool const have_piro = params.isSublist("Piro");

    ALBANY_ASSERT(have_piro == true);

    Teuchos::ParameterList& piro_params = params.sublist("Piro");

    std::string const msg{
        "All subdomains must have the same solution method (NOX or Tempus)"};

    if (subdomain == 0) {
      is_dynamic  = piro_params.isSublist("Tempus");
      is_static   = !is_dynamic;
      is_static_  = is_static;
      is_dynamic_ = is_dynamic;
    }
    if (is_static == true) {
      ALBANY_ASSERT(piro_params.isSublist("NOX") == true, msg);
    }
    if (is_dynamic == true) {
      ALBANY_ASSERT(piro_params.isSublist("Tempus") == true, msg);

      Teuchos::ParameterList& tempus_params = piro_params.sublist("Tempus");

      tempus_params.set("Abort on Failure", false);

      Teuchos::ParameterList& time_step_control_params =
          piro_params.sublist("Tempus")
              .sublist("Tempus Integrator")
              .sublist("Time Step Control");

      std::string const integrator_step_type =
          time_step_control_params.get("Integrator Step Type", "Constant");

      std::string const msg2{
          "Non-constant time-stepping through Tempus not supported "
          "with dynamic alternating Schwarz; \n"
          "In this case, variable time-stepping is "
          "handled within the Schwarz loop.\n"
          "Please rerun with 'Integrator Step Type: "
          "Constant' in 'Time Step Control' sublist.\n"};
      ALBANY_ASSERT(integrator_step_type == "Constant", msg2);
    }

    Teuchos::RCP<Albany::Application> app{Teuchos::null};

    Teuchos::RCP<Thyra::ResponseOnlyModelEvaluatorBase<ST>> solver =
        solver_factory.createAndGetAlbanyAppT(app, comm, comm);

    solvers_[subdomain] = solver;

    app->setSchwarzAlternating(true);

    apps_[subdomain] = app;

    // Get STK mesh structs to control Exodus output interval
    Teuchos::RCP<Albany::AbstractDiscretization> disc =
        app->getDiscretization();

    discs_[subdomain] = disc;

    Albany::STKDiscretization& stk_disc =
        *static_cast<Albany::STKDiscretization*>(disc.get());

    Teuchos::RCP<Albany::AbstractSTKMeshStruct> ams =
        stk_disc.getSTKMeshStruct();

    do_outputs_[subdomain]      = ams->exoOutput;
    do_outputs_init_[subdomain] = ams->exoOutput;

    stk_mesh_structs_[subdomain] = ams;

    model_evaluators_[subdomain] = solver_factory.returnModelT();

    curr_disp_[subdomain] = Teuchos::null;
  }

  //
  // Parameters
  //
  Teuchos::ParameterList& problem_params = app_params->sublist("Problem");

  bool const have_parameters = problem_params.isSublist("Parameters");

  ALBANY_ASSERT(have_parameters == false, "Parameters not supported.");

  //
  // Responses
  //
  bool const have_responses = problem_params.isSublist("Response Functions");

  ALBANY_ASSERT(have_responses == false, "Responses not supported.");

  return;
}

//
//
//
SchwarzAlternating::~SchwarzAlternating() { return; }

//
//
//
Teuchos::RCP<Thyra::VectorSpaceBase<ST> const>
SchwarzAlternating::get_x_space() const
{
  return Teuchos::null;
}

//
//
//
Teuchos::RCP<Thyra::VectorSpaceBase<ST> const>
SchwarzAlternating::get_f_space() const
{
  return Teuchos::null;
}

//
//
//
Teuchos::RCP<Thyra::VectorSpaceBase<ST> const>
SchwarzAlternating::get_p_space(int) const
{
  return Teuchos::null;
}

//
//
//
Teuchos::RCP<Thyra::VectorSpaceBase<ST> const>
SchwarzAlternating::get_g_space(int) const
{
  return Teuchos::null;
}

//
//
//
Teuchos::RCP<const Teuchos::Array<std::string>>
SchwarzAlternating::get_p_names(int) const
{
  return Teuchos::null;
}

//
//
//
Teuchos::ArrayView<const std::string>
SchwarzAlternating::get_g_names(int) const
{
  ALBANY_ASSERT(false, "not implemented");
  return Teuchos::ArrayView<const std::string>(Teuchos::null);
}

//
//
//
Thyra::ModelEvaluatorBase::InArgs<ST>
SchwarzAlternating::getNominalValues() const
{
  return this->createInArgsImpl();
}

//
//
//
Thyra::ModelEvaluatorBase::InArgs<ST>
SchwarzAlternating::getLowerBounds() const
{
  return Thyra::ModelEvaluatorBase::InArgs<ST>();  // Default value
}

//
//
//
Thyra::ModelEvaluatorBase::InArgs<ST>
SchwarzAlternating::getUpperBounds() const
{
  return Thyra::ModelEvaluatorBase::InArgs<ST>();  // Default value
}

//
//
//
Teuchos::RCP<Thyra::LinearOpBase<ST>>
SchwarzAlternating::create_W_op() const
{
  return Teuchos::null;
}

//
//
//
Teuchos::RCP<Thyra::PreconditionerBase<ST>>
SchwarzAlternating::create_W_prec() const
{
  return Teuchos::null;
}

//
//
//
Teuchos::RCP<const Thyra::LinearOpWithSolveFactoryBase<ST>>
SchwarzAlternating::get_W_factory() const
{
  return Teuchos::null;
}

//
//
//
Thyra::ModelEvaluatorBase::InArgs<ST>
SchwarzAlternating::createInArgs() const
{
  return this->createInArgsImpl();
}

//
//
//
Teuchos::ArrayRCP<Teuchos::RCP<Albany::Application>>
SchwarzAlternating::getApps() const
{
  return apps_;
}

//
//
//
void
SchwarzAlternating::set_failed(char const* msg)
{
  failed_          = true;
  failure_message_ = msg;
  return;
}

//
//
//
void
SchwarzAlternating::clear_failed()
{
  failed_ = false;
  return;
}

//
//
//
bool
SchwarzAlternating::get_failed() const
{
  return failed_;
}

//
// Create operator form of dg/dx for distributed responses
//
Teuchos::RCP<Thyra::LinearOpBase<ST>>
SchwarzAlternating::create_DgDx_op_impl(int j) const
{
  return Teuchos::null;
}

//
// Create operator form of dg/dx_dot for distributed responses
//
Teuchos::RCP<Thyra::LinearOpBase<ST>>
SchwarzAlternating::create_DgDx_dot_op_impl(int j) const
{
  return Teuchos::null;
}

//
// Create InArgs
//
Thyra::ModelEvaluatorBase::InArgs<ST>
SchwarzAlternating::createInArgsImpl() const
{
  Thyra::ModelEvaluatorBase::InArgsSetup<ST> ias;

  ias.setModelEvalDescription(this->description());

  ias.setSupports(Thyra::ModelEvaluatorBase::IN_ARG_x, true);
  ias.setSupports(Thyra::ModelEvaluatorBase::IN_ARG_x_dot, true);
  ias.setSupports(Thyra::ModelEvaluatorBase::IN_ARG_x_dot_dot, true);
  ias.setSupports(Thyra::ModelEvaluatorBase::IN_ARG_t, true);
  ias.setSupports(Thyra::ModelEvaluatorBase::IN_ARG_alpha, true);
  ias.setSupports(Thyra::ModelEvaluatorBase::IN_ARG_beta, true);
  ias.setSupports(Thyra::ModelEvaluatorBase::IN_ARG_W_x_dot_dot_coeff, true);

  return ias;
}

//
// Create OutArgs
//
Thyra::ModelEvaluatorBase::OutArgs<ST>
SchwarzAlternating::createOutArgsImpl() const
{
  Thyra::ModelEvaluatorBase::OutArgsSetup<ST> oas;

  oas.setModelEvalDescription(this->description());

  oas.setSupports(Thyra::ModelEvaluatorBase::OUT_ARG_f, true);
  oas.setSupports(Thyra::ModelEvaluatorBase::OUT_ARG_W_op, true);
  oas.setSupports(Thyra::ModelEvaluatorBase::OUT_ARG_W_prec, false);

  oas.set_W_properties(Thyra::ModelEvaluatorBase::DerivativeProperties(
      Thyra::ModelEvaluatorBase::DERIV_LINEARITY_UNKNOWN,
      Thyra::ModelEvaluatorBase::DERIV_RANK_FULL,
      true));

  return oas;
}

//
// Evaluate model on InArgs
//
void
SchwarzAlternating::evalModelImpl(
    Thyra::ModelEvaluatorBase::InArgs<ST> const&,
    Thyra::ModelEvaluatorBase::OutArgs<ST> const&) const
{
  if (is_dynamic_ == true) { SchwarzLoopDynamics(); }
  if (is_static_ == true) { SchwarzLoopQuasistatics(); }
  return;
}

namespace {

std::string
centered(std::string const& str, int width)
{
  assert(width >= 0);

  int const length = static_cast<int>(str.size());

  int const padding = width - length;

  if (padding <= 0) return str;

  int const left = padding / 2;

  int const right = padding - left;

  return std::string(left, ' ') + str + std::string(right, ' ');
}

void
printInternalElementState(
    Albany::StateArrayVec& esa,
    std::string const&     statename,
    std::string const&     init_type,
    int const              size,
    int const              ws)
{
  // IKT, 2/7/18: this is cut/paste from Albany::StateManager.
  // Note we are only printing states at first cell, quad point, dimension,
  // etc., to suppress amount of debug output. This can be changed, as desired,
  // by modifying the code here.
  auto& fos = *Teuchos::VerboseObjectBase::getDefaultOStream();
  if (size == 0) return;
  int cell = 0;
  int qp   = 0;
  int i    = 0;
  int j    = 0;
  int k    = 0;
  if (init_type == "scalar") {
    switch (size) {
      case 1:
        fos << "   DEBUG: case 1, " << statename << " = "
            << esa[ws][statename](cell) << "\n";
        break;
      case 2:
        fos << "   DEBUG: case 2, " << statename << " = "
            << esa[ws][statename](cell, qp) << "\n";
        break;
      case 3:
        fos << "   DEBUG: case 3, " << statename << " = "
            << esa[ws][statename](cell, qp, i) << "\n";
        break;
      case 4:
        fos << "   DEBUG: case 4, " << statename << " = "
            << esa[ws][statename](cell, qp, i, j) << "\n";
        break;
      case 5:
        fos << "   DEBUG: case 5, " << statename << " = "
            << esa[ws][statename](cell, qp, i, j, k) << "\n";
        break;
      default: ALBANY_ASSERT(1 <= size && size <= 5); break;
    }
  } else if (init_type == "identity") {
    fos << "   DEBUG: " << statename << " = "
        << esa[ws][statename](cell, qp, i, j) << "\n";
  }
}

void
printInternalElementStates(
    Albany::StateArrays&                  sa,
    Teuchos::RCP<Albany::StateInfoStruct> sis)
{
  Albany::StateArrayVec& esa = sa.elemStateArrays;
  // Print stuff for only workset 0
  int const ws = 0;
  for (auto i = 0; i < sis->size(); i++) {
    std::string const&             state_name = (*sis)[i]->name;
    std::string const&             init_type  = (*sis)[i]->initType;
    Albany::StateStruct::FieldDims dims;
    esa[ws][state_name].dimensions(dims);
    int size = dims.size();
    printInternalElementState(esa, state_name, init_type, size, ws);
  }
}

void
toFrom(LCM::StateArrayVec& dst, Albany::StateArrayVec const& src)
{
  auto const num_maps = src.size();

  dst.resize(num_maps);

  for (auto i = 0; i < num_maps; ++i) {
    auto&& src_map = src[i];

    auto&& dst_map = dst[i];

    for (auto&& kv : src_map) {
      auto&& state_name = kv.first;

      auto&& src_states = kv.second;

      auto&& dst_states = dst_map[state_name];

      auto const num_states = src_states.size();

      dst_states.resize(num_states);

      for (auto j = 0; j < num_states; ++j) { dst_states[j] = src_states[j]; }
    }
  }
  return;
}

void
toFrom(Albany::StateArrayVec& dst, LCM::StateArrayVec const& src)
{
  auto const num_maps = src.size();

  ALBANY_ASSERT(
      num_maps == dst.size(),
      "Inconsistent number of state maps from LCM to Albany");

  for (auto i = 0; i < num_maps; ++i) {
    auto&& src_map = src[i];

    auto&& dst_map = dst[i];

    for (auto&& kv : src_map) {
      auto&& state_name = kv.first;

      auto&& src_states = kv.second;

      ALBANY_ASSERT(
          dst_map.find(state_name) != dst_map.end(),
          "Missing state name in transfer from LCM to Albany: " + state_name);

      auto&& dst_states = dst_map[state_name];

      auto const num_states = src_states.size();

      ALBANY_ASSERT(
          num_states == dst_states.size(),
          "Inconsistent number of state entries from LCM to Albany");

      for (auto j = 0; j < num_states; ++j) { dst_states[j] = src_states[j]; }
    }
  }
  return;
}

void
toFrom(LCM::StateArrays& dst, Albany::StateArrays const& src)
{
  toFrom(dst.element_state_arrays, src.elemStateArrays);
  toFrom(dst.node_state_arrays, src.nodeStateArrays);
  return;
}

void
toFrom(Albany::StateArrays& dst, LCM::StateArrays const& src)
{
  toFrom(dst.elemStateArrays, src.element_state_arrays);
  toFrom(dst.nodeStateArrays, src.node_state_arrays);
  return;
}

}  // namespace

//
//
//
void
SchwarzAlternating::updateConvergenceCriterion() const
{
  abs_error_ = norm_diff_;
  rel_error_ = norm_final_ > 0.0 ? norm_diff_ / norm_final_ : norm_diff_;

  bool const converged_absolute = abs_error_ <= abs_tol_;
  bool const converged_relative = rel_error_ <= rel_tol_;

  switch (criterion_) {
    default: ALBANY_ASSERT(false, "Unknown Convergence Criterion"); break;
    case ConvergenceCriterion::ABSOLUTE: converged_ = converged_absolute; break;
    case ConvergenceCriterion::RELATIVE: converged_ = converged_relative; break;
    case ConvergenceCriterion::BOTH:
      switch (operator_) {
        default:
          ALBANY_ASSERT(false, "Unknown Convergence Logical Operator");
          break;
        case ConvergenceLogicalOperator::AND:
          converged_ = converged_absolute && converged_relative;
          break;
        case ConvergenceLogicalOperator::OR:
          converged_ = converged_absolute || converged_relative;
          break;
      }
      break;
  }

  return;
}

//
//
//
bool
SchwarzAlternating::continueSolve() const
{
  ++num_iter_;

  // If failure has occurred, stop immediately.
  if (failed_ == true) return false;

  // Regardless of other criteria, if error is zero stop solving.
  bool const zero_error = ((abs_error_ > 0.0) == false);

  if (zero_error == true) return false;

  // Minimum iterations takes precedence over maximum iterations and
  // convergence. Continue solving if not exceeded.
  bool const exceeds_min_iter = num_iter_ >= min_iters_;

  if (exceeds_min_iter == false) return true;

  // Maximum iterations takes precedence over convergence.
  // Stop solving if exceeded.
  bool const exceeds_max_iter = num_iter_ >= max_iters_;

  if (exceeds_max_iter == true) return false;

  // Lastly check for convergence.
  bool const continue_solve = (converged_ == false);

  return continue_solve;
}

//
//
//
void
SchwarzAlternating::reportFinals(std::ostream& os) const
{
  std::string const conv_str = converged_ == true ? "YES" : "NO";

  os << '\n';
  os << "Schwarz Alternating Method converged: " << conv_str << '\n';
  os << "Minimum iterations :" << min_iters_ << '\n';
  os << "Maximum iterations :" << max_iters_ << '\n';
  os << "Total iterations   :" << num_iter_ << '\n';
  os << "Last absolute error:" << abs_error_ << '\n';
  os << "Absolute tolerance :" << abs_tol_ << '\n';
  os << "Last relative error:" << rel_error_ << '\n';
  os << "Relative tolerance :" << rel_tol_ << '\n';
  os << std::endl;
  return;
}

//
// Schwarz Alternating loop, dynamic
//
void
SchwarzAlternating::SchwarzLoopDynamics() const
{
  minitensor::Vector<ST> norms_init(num_subdomains_, minitensor::Filler::ZEROS);

  minitensor::Vector<ST> norms_final(
      num_subdomains_, minitensor::Filler::ZEROS);

  minitensor::Vector<ST> norms_diff(num_subdomains_, minitensor::Filler::ZEROS);

  std::string const delim(72, '=');

  auto& fos = *Teuchos::VerboseObjectBase::getDefaultOStream();

  fos << delim << std::endl;
  fos << "Schwarz Alternating Method with " << num_subdomains_;
  fos << " subdomains\n";
  fos << std::scientific << std::setprecision(17);

  ST time_step{initial_time_step_};

  int stop{0};

  ST current_time{initial_time_};

  // Set ICs and PrevSoln vecs and write initial configuration to Exodus file
  setDynamicICVecsAndDoOutput(initial_time_);

  // Time-stepping loop
  while (stop < maximum_steps_ && current_time < final_time_) {
    fos << delim << std::endl;
    fos << "Time stop          :" << stop << '\n';
    fos << "Time               :" << current_time << '\n';
    fos << "Time step          :" << time_step << '\n';
    fos << delim << std::endl;

    // Before the Schwarz loop, get internal states
    for (auto subdomain = 0; subdomain < num_subdomains_; ++subdomain) {
      auto& app = *apps_[subdomain];

      auto& state_mgr = app.getStateMgr();

#ifdef DEBUG
      fos << "DEBUG: Getting internal states subdomain = " << subdomain
          << "...\n";
#endif
      toFrom(internal_states_[subdomain], state_mgr.getStateArrays());
#ifdef DEBUG
      printInternalElementStates(
          internal_states_[subdomain], state_mgr.getStateInfoStruct());
      fos << "DEBUG: ...done setting internal states subdomain = " << subdomain
          << ".\n";
#endif
    }

    ST const next_time{current_time + time_step};

    num_iter_ = 0;

    // Schwarz loop
    do {
      bool const is_initial_state = stop == 0 && num_iter_ == 0;

      for (auto subdomain = 0; subdomain < num_subdomains_; ++subdomain) {
        fos << delim << std::endl;
        fos << "Schwarz iteration  :" << num_iter_ << '\n';
        fos << "Subdomain          :" << subdomain << '\n';
        fos << delim << std::endl;

        // Restore solution from previous Schwarz iteration before solve
        if (is_initial_state == true) {
          auto& me = dynamic_cast<Albany::ModelEvaluatorT&>(
              *model_evaluators_[subdomain]);
          auto const& nv        = me.getNominalValues();
          prev_disp_[subdomain] = Thyra::createMember(me.get_x_space());
          Thyra::copy(*(nv.get_x()), prev_disp_[subdomain].ptr());
          prev_velo_[subdomain] = Thyra::createMember(me.get_x_space());
          Thyra::copy(*(nv.get_x_dot()), prev_velo_[subdomain].ptr());
          prev_acce_[subdomain] = Thyra::createMember(me.get_x_space());
          Thyra::copy(*(nv.get_x_dot_dot()), prev_acce_[subdomain].ptr());
        } else {
          Thyra::put_scalar(0.0, prev_disp_[subdomain].ptr());
          Thyra::copy(*this_disp_[subdomain], prev_disp_[subdomain].ptr());
          Thyra::put_scalar(0.0, prev_velo_[subdomain].ptr());
          Thyra::copy(*this_velo_[subdomain], prev_velo_[subdomain].ptr());
          Thyra::put_scalar(0.0, prev_acce_[subdomain].ptr());
          Thyra::copy(*this_acce_[subdomain], prev_acce_[subdomain].ptr());
        }

        // Solve for each subdomain
        Thyra::ResponseOnlyModelEvaluatorBase<ST>& solver =
            *(solvers_[subdomain]);

        Piro::TempusSolver<ST, LO, Tpetra_GO, KokkosNode>& piro_tempus_solver =
            dynamic_cast<Piro::TempusSolver<ST, LO, Tpetra_GO, KokkosNode>&>(
                solver);

        piro_tempus_solver.setStartTime(current_time);
        piro_tempus_solver.setFinalTime(next_time);
        piro_tempus_solver.setInitTimeStep(time_step);

        fos << "Initial time       :" << current_time << '\n';
        fos << "Final time         :" << next_time << '\n';
        fos << "Time step          :" << time_step << '\n';
        fos << delim << std::endl;

        Thyra::ModelEvaluatorBase::InArgs<ST> in_args = solver.createInArgs();

        Thyra::ModelEvaluatorBase::OutArgs<ST> out_args =
            solver.createOutArgs();

        auto& me = dynamic_cast<Albany::ModelEvaluatorT&>(
            *model_evaluators_[subdomain]);

        // Restore internal states
        auto& app = *apps_[subdomain];

        auto& state_mgr = app.getStateMgr();

#ifdef DEBUG
        fos << "DEBUG: Setting internal states subdomain = " << subdomain
            << "...\n";
#endif
        toFrom(state_mgr.getStateArrays(), internal_states_[subdomain]);
#ifdef DEBUG
        printInternalElementStates(
            internal_states_[subdomain], state_mgr.getStateInfoStruct());
        fos << "DEBUG: ...done setting internal states subdomain = "
            << subdomain << ".\n";
#endif

        Teuchos::RCP<Tempus::SolutionHistory<ST>> solution_history;

        Teuchos::RCP<Tempus::SolutionState<ST>> current_state;

        Teuchos::RCP<Thyra::VectorBase<ST>> ic_disp_rcp =
            Thyra::createMember(me.get_x_space());

        Teuchos::RCP<Thyra::VectorBase<ST>> ic_velo_rcp =
            Thyra::createMember(me.get_x_space());

        Teuchos::RCP<Thyra::VectorBase<ST>> ic_acce_rcp =
            Thyra::createMember(me.get_x_space());

        // set ic_disp_rcp, ic_velo_rcp and ic_acce_rcp
        // by making copy of what is in ics_disp_[subdomain], etc.
        Thyra::VectorBase<ST>& ic_disp = *ics_disp_[subdomain];

        Thyra::VectorBase<ST>& ic_velo = *ics_velo_[subdomain];

        Thyra::VectorBase<ST>& ic_acce = *ics_acce_[subdomain];

        Thyra::copy(ic_disp, ic_disp_rcp.ptr());

        Thyra::copy(ic_velo, ic_velo_rcp.ptr());

        Thyra::copy(ic_acce, ic_acce_rcp.ptr());

        piro_tempus_solver.setInitialState(
            current_time, ic_disp_rcp, ic_velo_rcp, ic_acce_rcp);

        if (std_init_guess_ == false) {
          piro_tempus_solver.setInitialGuess(prev_disp_[subdomain]);
        }

        solver.evalModel(in_args, out_args);

        // Allocate current solution vectors

        this_disp_[subdomain] = Thyra::createMember(me.get_x_space());
        this_velo_[subdomain] = Thyra::createMember(me.get_x_space());
        this_acce_[subdomain] = Thyra::createMember(me.get_x_space());

#if defined(DEBUG)
        Teuchos::RCP<Tpetra_Vector> prev_disp_tpetra;

        fos << "\n*** Thyra: Previous solution ***\n";
        prev_disp_[subdomain]->describe(fos, Teuchos::VERB_EXTREME);
        fos << "\n*** NORM: " << Thyra::norm(*prev_disp_[subdomain]) << '\n';
        if (subdomain == 0) {
          prev_disp_tpetra = ConverterT::getTpetraVector(prev_disp_[0]);
          Albany::writeMatrixMarket(prev_disp_tpetra, "prev_disp0", num_iter_);
        } else if (subdomain == 1) {
          prev_disp_tpetra = ConverterT::getTpetraVector(prev_disp_[1]);
          Albany::writeMatrixMarket(prev_disp_tpetra, "prev_disp1", num_iter_);
        }
        fos << "\n*** Thyra: Previous solution ***\n";
#endif  // DEBUG

        // Check whether solver did OK.

        auto const status = piro_tempus_solver.getTempusIntegratorStatus();

        // if (status == NOX::StatusTest::Failed) {
        if (status == Tempus::Status::FAILED) {
          fos << "\nINFO: Unable to solve for subdomain " << subdomain << '\n';
          failed_ = true;
          // Break out of the subdomain loop
          break;
        }

        // If solver is OK, extract solution

        solution_history = piro_tempus_solver.getSolutionHistory();

        current_state = solution_history->getCurrentState();

        Thyra::copy(*current_state->getX(), this_disp_[subdomain].ptr());

        Thyra::copy(*current_state->getXDot(), this_velo_[subdomain].ptr());

        Thyra::copy(*current_state->getXDotDot(), this_acce_[subdomain].ptr());

#if defined(DEBUG)
        fos << "\n*** Thyra: Current solution ***\n";
        this_disp_[subdomain]->describe(fos, Teuchos::VERB_EXTREME);
        fos << "\n*** NORM: " << Thyra::norm(*this_disp_[subdomain]) << '\n';
        fos << "\n*** Thyra: Current solution ***\n";

        Teuchos::RCP<Tpetra_Vector> curr_disp_tpetra;

        if (subdomain == 0) {
          curr_disp_tpetra = ConverterT::getTpetraVector(this_disp_[0]);
          Albany::writeMatrixMarket(curr_disp_tpetra, "curr_disp0", num_iter_);
        } else if (subdomain == 1) {
          curr_disp_tpetra = ConverterT::getTpetraVector(this_disp_[1]);
          Albany::writeMatrixMarket(curr_disp_tpetra, "curr_disp1", num_iter_);
        }
#endif  // DEBUG

        Teuchos::RCP<Thyra::VectorBase<ST>> disp_diff_rcp =
            Thyra::createMember(me.get_x_space());
        Thyra::put_scalar<ST>(0.0, disp_diff_rcp.ptr());
        Thyra::V_VpStV(
            disp_diff_rcp.ptr(),
            *this_disp_[subdomain],
            -1.0,
            *prev_disp_[subdomain]);

        Teuchos::RCP<Thyra::VectorBase<ST>> velo_diff_rcp =
            Thyra::createMember(me.get_x_space());
        Thyra::put_scalar<ST>(0.0, velo_diff_rcp.ptr());
        Thyra::V_VpStV(
            velo_diff_rcp.ptr(),
            *this_velo_[subdomain],
            -1.0,
            *prev_velo_[subdomain]);

        Teuchos::RCP<Thyra::VectorBase<ST>> acce_diff_rcp =
            Thyra::createMember(me.get_x_space());
        Thyra::put_scalar<ST>(0.0, acce_diff_rcp.ptr());
        Thyra::V_VpStV(
            acce_diff_rcp.ptr(),
            *this_acce_[subdomain],
            -1.0,
            *prev_acce_[subdomain]);

#if defined(DEBUG)
        fos << "\n*** Thyra: Solution difference ***\n";
        disp_diff_rcp->describe(fos, Teuchos::VERB_EXTREME);
        fos << "\n*** NORM: " << Thyra::norm(*disp_diff_rcp) << '\n';
        fos << "\n*** Thyra: Solution difference ***\n";

        Teuchos::RCP<Tpetra_Vector> disp_diff_tpetra;

        if (subdomain == 0) {
          disp_diff_tpetra = ConverterT::getTpetraVector(disp_diff_rcp);
          Albany::writeMatrixMarket(disp_diff_tpetra, "disp_diff0", num_iter_);
        } else if (subdomain == 1) {
          disp_diff_tpetra = ConverterT::getTpetraVector(disp_diff_rcp);
          Albany::writeMatrixMarket(disp_diff_tpetra, "disp_diff1", num_iter_);
        }
#endif  // DEBUG

        // After solve, save solution and get info to check convergence
        norms_init(subdomain)  = Thyra::norm(*prev_disp_[subdomain]);
        norms_final(subdomain) = Thyra::norm(*this_disp_[subdomain]);
        norms_diff(subdomain)  = Thyra::norm(*disp_diff_rcp);

        auto const dt = tol_factor_vel_;

        norms_init(subdomain) += dt * Thyra::norm(*prev_velo_[subdomain]);
        norms_final(subdomain) += dt * Thyra::norm(*this_velo_[subdomain]);
        norms_diff(subdomain) += dt * Thyra::norm(*velo_diff_rcp);

        auto const dt2 = tol_factor_acc_;

        norms_init(subdomain) += dt2 * Thyra::norm(*prev_acce_[subdomain]);
        norms_final(subdomain) += dt2 * Thyra::norm(*this_acce_[subdomain]);
        norms_diff(subdomain) += dt2 * Thyra::norm(*acce_diff_rcp);

      }  // Subdomains loop

      if (failed_ == true) {
        fos << "INFO: Unable to continue Schwarz iteration " << num_iter_;
        fos << "\n";
        // Break out of the Schwarz loop.
        break;
      }

      norm_init_  = minitensor::norm(norms_init);
      norm_final_ = minitensor::norm(norms_final);
      norm_diff_  = minitensor::norm(norms_diff);

      updateConvergenceCriterion();

      fos << delim << std::endl;
      fos << "Schwarz iteration         :" << num_iter_ << '\n';

      std::string const line(72, '-');

      fos << line << std::endl;

      fos << centered("Sub", 6);
      fos << centered("Initial norm", 22);
      fos << centered("Final norm", 22);
      fos << centered("Difference norm", 22);
      fos << std::endl;
      fos << centered("dom", 6);
      fos << centered("||X0||", 22);
      fos << centered("||Xf||", 22);
      fos << centered("||Xf-X0||", 22);
      fos << std::endl;
      fos << line << std::endl;

      for (auto m = 0; m < num_subdomains_; ++m) {
        fos << std::setw(6) << m;
        fos << std::setw(22) << norms_init(m);
        fos << std::setw(22) << norms_final(m);
        fos << std::setw(22) << norms_diff(m);
        fos << std::endl;
      }

      fos << line << std::endl;
      fos << centered("Norm", 6);
      fos << std::setw(22) << norm_init_;
      fos << std::setw(22) << norm_final_;
      fos << std::setw(22) << norm_diff_;
      fos << std::endl;
      fos << line << std::endl;
      fos << "Absolute error     :" << abs_error_ << '\n';
      fos << "Absolute tolerance :" << abs_tol_ << '\n';
      fos << "Relative error     :" << rel_error_ << '\n';
      fos << "Relative tolerance :" << rel_tol_ << '\n';
      fos << delim << std::endl;

    } while (continueSolve() == true);

    // One of the subdomains failed to solve. Reduce step.
    if (failed_ == true) {
      failed_ = false;

      auto const reduced_step = reduction_factor_ * time_step;

      if (time_step <= min_time_step_) {
        fos << "ERROR: Cannot reduce step. Stopping execution.\n";
        fos << "INFO: Requested step    :" << reduced_step << '\n';
        fos << "INFO: Minimum time step :" << min_time_step_ << '\n';
        return;
      }

      if (reduced_step > min_time_step_) {
        fos << "INFO: Reducing step from " << time_step << " to ";
        fos << reduced_step << '\n';
        time_step = reduced_step;
      } else {
        fos << "INFO: Reducing step from " << time_step << " to ";
        fos << min_time_step_ << '\n';
        time_step = min_time_step_;
      }

      // Restore previous solutions
      for (auto subdomain = 0; subdomain < num_subdomains_; ++subdomain) {
        Thyra::put_scalar(0.0, this_disp_[subdomain].ptr());
        Thyra::copy(*ics_disp_[subdomain], this_disp_[subdomain].ptr());
        Thyra::put_scalar(0.0, this_velo_[subdomain].ptr());
        Thyra::copy(*ics_velo_[subdomain], this_velo_[subdomain].ptr());
        Thyra::put_scalar(0.0, this_acce_[subdomain].ptr());
        Thyra::copy(*ics_acce_[subdomain], this_acce_[subdomain].ptr());

        // restore the state manager with the state variables from the previous
        // loadstep.
        auto& app = *apps_[subdomain];

        auto& state_mgr = app.getStateMgr();

        toFrom(state_mgr.getStateArrays(), internal_states_[subdomain]);

        // restore the solution in the discretization so the schwarz solver gets
        // the right boundary conditions!
        Teuchos::RCP<Tpetra_Vector const> disp_rcp_tpetra;

        Teuchos::RCP<Tpetra_Vector const> velo_rcp_tpetra;

        Teuchos::RCP<Tpetra_Vector const> acce_rcp_tpetra;

        disp_rcp_tpetra =
            ConverterT::getConstTpetraVector(ics_disp_[subdomain]);
        velo_rcp_tpetra =
            ConverterT::getConstTpetraVector(ics_velo_[subdomain]);
        acce_rcp_tpetra =
            ConverterT::getConstTpetraVector(ics_acce_[subdomain]);
        Teuchos::RCP<Albany::AbstractDiscretization> const& app_disc =
            app.getDiscretization();

        app_disc->writeSolutionToMeshDatabaseT(
            *disp_rcp_tpetra, *velo_rcp_tpetra, *acce_rcp_tpetra, current_time);
      }

      // Jump to the beginning of the time-step loop without advancing
      // time to try to use a reduced step.
      continue;
    }

    reportFinals(fos);

    // Update IC vecs and output solution to exodus file

    for (auto subdomain = 0; subdomain < num_subdomains_; ++subdomain) {
      if (do_outputs_init_[subdomain] == true) {
        do_outputs_[subdomain] =
            output_interval_ > 0 ? (stop + 1) % output_interval_ == 0 : false;
      }
    }

    setDynamicICVecsAndDoOutput(next_time);

    ++stop;
    current_time += time_step;

    // Step successful. Try to increase the time step.
    auto const increased_step =
        std::min(max_time_step_, increase_factor_ * time_step);

    if (increased_step > time_step) {
      fos << "\nINFO: Increasing step from " << time_step << " to ";
      fos << increased_step << '\n';
      time_step = increased_step;
    } else {
      fos << "\nINFO: Cannot increase step. Using " << time_step << '\n';
    }

  }  // Time-step loop

  return;
}

void
SchwarzAlternating::setExplicitUpdateInitialGuessForSchwarz(
    ST const current_time,
    ST const time_step) const
{
  // do an explicit update to form the initial guess for the schwarz
  // iteration
  for (auto subdomain = 0; subdomain < num_subdomains_; ++subdomain) {
    auto& app = *apps_[subdomain];

    auto&                  state_mgr = app.getStateMgr();
    Thyra::VectorBase<ST>& ic_disp   = *ics_disp_[subdomain];

    Thyra::VectorBase<ST>& ic_velo = *ics_velo_[subdomain];

    Thyra::VectorBase<ST>& ic_acce = *ics_acce_[subdomain];

    auto& me =
        dynamic_cast<Albany::ModelEvaluatorT&>(*model_evaluators_[subdomain]);
    if (current_time == 0) {
      this_disp_[subdomain] = Thyra::createMember(me.get_x_space());
      this_velo_[subdomain] = Thyra::createMember(me.get_x_space());
      this_acce_[subdomain] = Thyra::createMember(me.get_x_space());
    }

    const ST aConst = time_step * time_step / 2.0;
    Thyra::V_StVpStV(
        this_disp_[subdomain].ptr(), time_step, ic_velo, aConst, ic_acce);
    Thyra::Vp_V(this_disp_[subdomain].ptr(), ic_disp, 1.0);

    // This is the initial guess that I want to apply to the subdomains before
    // the schwarz solver starts
    auto disp_rcp = this_disp_[subdomain];

    auto velo_rcp = this_velo_[subdomain];

    auto acce_rcp = this_acce_[subdomain];

    Teuchos::RCP<Tpetra_Vector const> disp_rcp_tpetra;

    Teuchos::RCP<Tpetra_Vector const> velo_rcp_tpetra;

    Teuchos::RCP<Tpetra_Vector const> acce_rcp_tpetra;

    disp_rcp_tpetra = ConverterT::getConstTpetraVector(disp_rcp);
    velo_rcp_tpetra = ConverterT::getConstTpetraVector(velo_rcp);
    acce_rcp_tpetra = ConverterT::getConstTpetraVector(acce_rcp);
    // setting the displacement in the albany application
    app.setX(disp_rcp_tpetra);
    app.setXdot(velo_rcp_tpetra);
    app.setXdotdot(acce_rcp_tpetra);

    // in order to get the Schwarz boundary conditions right, we need to set the
    // state in the discretization
    Teuchos::RCP<Albany::AbstractDiscretization> const& app_disc =
        app.getDiscretization();

    app_disc->writeSolutionToMeshDatabaseT(
        *disp_rcp_tpetra, *velo_rcp_tpetra, *acce_rcp_tpetra, current_time);
  }
}

void
SchwarzAlternating::setDynamicICVecsAndDoOutput(ST const time) const
{
  bool is_initial_time = false;

  if (time == initial_time_) is_initial_time = true;

  for (auto subdomain = 0; subdomain < num_subdomains_; ++subdomain) {
    Albany::AbstractSTKMeshStruct& stk_mesh_struct =
        *stk_mesh_structs_[subdomain];

    Albany::AbstractDiscretization& abs_disc = *discs_[subdomain];

    Albany::STKDiscretization& stk_disc =
        static_cast<Albany::STKDiscretization&>(abs_disc);

    stk_mesh_struct.exoOutputInterval = 1;

    stk_mesh_struct.exoOutput = do_outputs_[subdomain];

    if (is_initial_time == true) {  // initial time-step: get initial solution
                                    // from nominalValues in ME

      auto& me =
          dynamic_cast<Albany::ModelEvaluatorT&>(*model_evaluators_[subdomain]);

      auto const& nv = me.getNominalValues();

      ics_disp_[subdomain] = Thyra::createMember(me.get_x_space());
      Thyra::copy(*(nv.get_x()), ics_disp_[subdomain].ptr());

      ics_velo_[subdomain] = Thyra::createMember(me.get_x_space());
      Thyra::copy(*(nv.get_x_dot()), ics_velo_[subdomain].ptr());

      ics_acce_[subdomain] = Thyra::createMember(me.get_x_space());
      Thyra::copy(*(nv.get_x_dot_dot()), ics_acce_[subdomain].ptr());

      // Write initial condition to STK mesh
      Teuchos::RCP<Tpetra_MultiVector const> const xMV =
          apps_[subdomain]->getAdaptSolMgrT()->getOverlappedSolution();

      stk_disc.writeSolutionMV(*xMV, initial_time_, true);

    }

    else {  // subsequent time steps: update ic vecs based on fields in stk
            // discretization

      Teuchos::RCP<Tpetra_MultiVector> disp_mv = stk_disc.getSolutionMV();

      // Update ics_disp_ and its time-derivatives
      ics_disp_[subdomain] = Thyra::createVector(disp_mv->getVectorNonConst(0));

      ics_velo_[subdomain] = Thyra::createVector(disp_mv->getVectorNonConst(1));

      ics_acce_[subdomain] = Thyra::createVector(disp_mv->getVectorNonConst(2));

      if (do_outputs_[subdomain] == true) {  // write solution to Exodus

        stk_disc.writeSolutionMV(*disp_mv, time);
      }
    }

    stk_mesh_struct.exoOutput = false;
  }
  return;
}

void
SchwarzAlternating::doQuasistaticOutput(ST const time) const
{
  for (auto subdomain = 0; subdomain < num_subdomains_; ++subdomain) {
    if (do_outputs_[subdomain] == true) {
      auto& stk_mesh_struct = *stk_mesh_structs_[subdomain];

      stk_mesh_struct.exoOutputInterval = 1;
      stk_mesh_struct.exoOutput         = true;

      auto& abs_disc = *discs_[subdomain];

      auto& stk_disc = static_cast<Albany::STKDiscretization&>(abs_disc);

      // Do not dereference this RCP. Leads to SEGFAULT (!?)
      auto disp_mv_rcp = stk_disc.getSolutionMV();

      stk_disc.writeSolutionMV(*disp_mv_rcp, time);

      stk_mesh_struct.exoOutput = false;
    }
  }

  return;
}

//
// Schwarz Alternating loop, quasistatic
//
void
SchwarzAlternating::SchwarzLoopQuasistatics() const
{
  minitensor::Vector<ST> norms_init(num_subdomains_, minitensor::Filler::ZEROS);

  minitensor::Vector<ST> norms_final(
      num_subdomains_, minitensor::Filler::ZEROS);

  minitensor::Vector<ST> norms_diff(num_subdomains_, minitensor::Filler::ZEROS);

  std::string const delim(72, '=');

  auto& fos = *Teuchos::VerboseObjectBase::getDefaultOStream();

  fos << delim << std::endl;
  fos << "Schwarz Alternating Method with " << num_subdomains_;
  fos << " subdomains\n";
  fos << std::scientific << std::setprecision(17);

  ST time_step{initial_time_step_};

  int stop{0};

  ST current_time{initial_time_};

  // Output initial configuration.
  doQuasistaticOutput(current_time);

  // Continuation loop. We do continuation manually for Schwarz instead
  // using LOCA. It turned out to be too complicated to use LOCA to sync
  // continuation between different subdomains.
  while (stop < maximum_steps_ && current_time < final_time_) {
    ST const next_time{current_time + time_step};

    fos << delim << std::endl;
    fos << "Global time stop   :" << stop << '\n';
    fos << "Start time         :" << current_time << '\n';
    fos << "Stop time          :" << next_time << '\n';
    fos << "Time step          :" << time_step << '\n';
    fos << delim << std::endl;

    // This object is necessary to be able to set an initial solution
    // for the model evaluator to a desired value. This is used to save
    // previous values of the solution at each Schwarz iteration for
    // each subdomain.
    Thyra::ModelEvaluatorBase::InArgsSetup<ST> nv;
    nv.setModelEvalDescription(this->description());
    nv.setSupports(Thyra::ModelEvaluatorBase::IN_ARG_x, true);

    // Before the Schwarz loop, save the solutions for each subdomain in case
    // the solve fails. Then the load step is reduced and the Schwarz
    // loop is restarted from scratch.
    for (auto subdomain = 0; subdomain < num_subdomains_; ++subdomain) {
      // Set these initial values explicitly to zero so that no
      // extra logic is necessary for initial values in the
      // Schwarz and subdomain loops.
      if (stop == 0) {
        auto& me = dynamic_cast<Albany::ModelEvaluatorT&>(
            *model_evaluators_[subdomain]);

        auto zero_disp_rcp = Thyra::createMember(me.get_x_space());

        auto zero_disp_ptr = zero_disp_rcp.ptr();

        Thyra::put_scalar<ST>(0.0, zero_disp_ptr);

        prev_step_disp_[subdomain] = zero_disp_rcp;
        curr_disp_[subdomain]      = zero_disp_rcp;
      } else {
        prev_step_disp_[subdomain] = curr_disp_[subdomain];
      }

      auto& app = *apps_[subdomain];

      auto& state_mgr = app.getStateMgr();

      toFrom(internal_states_[subdomain], state_mgr.getStateArrays());
    }

    num_iter_ = 0;

    // Schwarz loop
    do {
      // Subdomain loop
      for (auto subdomain = 0; subdomain < num_subdomains_; ++subdomain) {
        fos << delim << std::endl;
        fos << "Schwarz iteration  :" << num_iter_ << '\n';
        fos << "Subdomain          :" << subdomain << '\n';
        fos << "Start time         :" << current_time << '\n';
        fos << "Stop time          :" << next_time << '\n';
        fos << "Time step          :" << time_step << '\n';
        fos << delim << std::endl;

        // Save solution from previous Schwarz iteration before solve
        auto& me = dynamic_cast<Albany::ModelEvaluatorT&>(
            *model_evaluators_[subdomain]);

        auto prev_disp_rcp = curr_disp_[subdomain];

        auto const& prev_disp = *prev_disp_rcp;

        // Restore internal states
        auto& app = *apps_[subdomain];

        auto& state_mgr = app.getStateMgr();

        toFrom(state_mgr.getStateArrays(), internal_states_[subdomain]);

        // Restore solution from previous time step
        auto prev_step_disp_rcp = prev_step_disp_[subdomain];

        nv.set_x(prev_disp_rcp);
        me.setNominalValues(nv);

        // Target time
        me.setCurrentTime(next_time);

        // Solve for each subdomain
        auto& solver = *(solvers_[subdomain]);

        auto& piro_nox_solver = dynamic_cast<Piro::NOXSolver<ST>&>(solver);

        auto in_args = solver.createInArgs();

        auto out_args = solver.createOutArgs();

        solver.evalModel(in_args, out_args);

        // Check whether solver did OK.
        auto const& thyra_nox_solver = *piro_nox_solver.getSolver();

        auto const& const_nox_solver = *thyra_nox_solver.getNOXSolver();

        auto& nox_solver = const_cast<NOX::Solver::Generic&>(const_nox_solver);

        auto const status = nox_solver.getStatus();

        if (status == NOX::StatusTest::Failed) {
          fos << "\nINFO: Unable to solve for subdomain " << subdomain << '\n';
          failed_ = true;
          // Break out of the subdomain loop
          break;
        }

        // Solver OK, extract solution
        auto curr_disp_rcp = thyra_nox_solver.get_current_x()->clone_v();

        auto const& curr_disp = *curr_disp_rcp;

        // Compute difference between previous and current solutions
        auto disp_diff_rcp = Thyra::createMember(me.get_x_space());

        auto disp_diff_ptr = disp_diff_rcp.ptr();

        Thyra::put_scalar<ST>(0.0, disp_diff_ptr);

        Thyra::V_VpStV(disp_diff_ptr, curr_disp, -1.0, prev_disp);

        auto& disp_diff = *disp_diff_rcp;

        // After solve, save solution and get info to check convergence
        curr_disp_[subdomain]  = curr_disp_rcp;
        norms_init(subdomain)  = Thyra::norm(prev_disp);
        norms_final(subdomain) = Thyra::norm(curr_disp);
        norms_diff(subdomain)  = Thyra::norm(disp_diff);

      }  // Subdomain loop

      if (failed_ == true) {
        fos << "INFO: Unable to continue Schwarz iteration " << num_iter_;
        fos << "\n";
        // Break out of the Schwarz loop.
        break;
      }

      norm_init_  = minitensor::norm(norms_init);
      norm_final_ = minitensor::norm(norms_final);
      norm_diff_  = minitensor::norm(norms_diff);

      updateConvergenceCriterion();

      fos << delim << std::endl;
      fos << "Schwarz iteration         :" << num_iter_ << '\n';

      std::string const line(72, '-');

      fos << line << std::endl;

      fos << centered("Sub", 6);
      fos << centered("Initial norm", 22);
      fos << centered("Final norm", 22);
      fos << centered("Difference norm", 22);
      fos << std::endl;
      fos << centered("dom", 6);
      fos << centered("||X0||", 22);
      fos << centered("||Xf||", 22);
      fos << centered("||Xf-X0||", 22);
      fos << std::endl;
      fos << line << std::endl;

      for (auto m = 0; m < num_subdomains_; ++m) {
        fos << std::setw(6) << m;
        fos << std::setw(22) << norms_init(m);
        fos << std::setw(22) << norms_final(m);
        fos << std::setw(22) << norms_diff(m);
        fos << std::endl;
      }

      fos << line << std::endl;
      fos << centered("Norm", 6);
      fos << std::setw(22) << norm_init_;
      fos << std::setw(22) << norm_final_;
      fos << std::setw(22) << norm_diff_;
      fos << std::endl;
      fos << line << std::endl;
      fos << "Absolute error     :" << abs_error_ << '\n';
      fos << "Absolute tolerance :" << abs_tol_ << '\n';
      fos << "Relative error     :" << rel_error_ << '\n';
      fos << "Relative tolerance :" << rel_tol_ << '\n';
      fos << delim << std::endl;

    } while (continueSolve() == true);  // Schwarz loop

    // One or more of the subdomains failed to solve. Reduce step.
    if (failed_ == true) {
      failed_ = false;

      auto const reduced_step = reduction_factor_ * time_step;

      if (time_step <= min_time_step_) {
        fos << "ERROR: Cannot reduce step. Stopping execution.\n";
        fos << "INFO: Requested step    :" << reduced_step << '\n';
        fos << "INFO: Minimum time step :" << min_time_step_ << '\n';
        return;
      }

      if (reduced_step > min_time_step_) {
        fos << "INFO: Reducing step from " << time_step << " to ";
        fos << reduced_step << '\n';
        time_step = reduced_step;
      } else {
        fos << "INFO: Reducing step from " << time_step << " to ";
        fos << min_time_step_ << '\n';
        time_step = min_time_step_;
      }

      // Restore previous solutions
      for (auto subdomain = 0; subdomain < num_subdomains_; ++subdomain) {
        curr_disp_[subdomain] = prev_step_disp_[subdomain];

        // restore the state manager with the state variables from the previous
        // loadstep.
        auto& app = *apps_[subdomain];

        auto& state_mgr = app.getStateMgr();

        toFrom(state_mgr.getStateArrays(), internal_states_[subdomain]);

        // restore the solution in the discretization so the schwarz solver gets
        // the right boundary conditions!
        Teuchos::RCP<Tpetra_Vector const> disp_rcp_tpetra;

        disp_rcp_tpetra =
            ConverterT::getConstTpetraVector(curr_disp_[subdomain]);
        Teuchos::RCP<Albany::AbstractDiscretization> const& app_disc =
            app.getDiscretization();

        app_disc->writeSolutionToMeshDatabaseT(*disp_rcp_tpetra, current_time);
      }

      // Jump to the beginning of the continuation loop without advancing
      // time to try to use a reduced step.
      continue;
    }

    reportFinals(fos);

    // Output converged solution if at specified interval

    for (auto subdomain = 0; subdomain < num_subdomains_; ++subdomain) {
      if (do_outputs_init_[subdomain] == true) {
        do_outputs_[subdomain] =
            output_interval_ > 0 ? (stop + 1) % output_interval_ == 0 : false;
      }
    }

    doQuasistaticOutput(next_time);

    ++stop;
    current_time += time_step;

    // Step successful. Try to increase the time step.
    auto const increased_step =
        std::min(max_time_step_, increase_factor_ * time_step);

    if (increased_step > time_step) {
      fos << "\nINFO: Increasing step from " << time_step << " to ";
      fos << increased_step << '\n';
      time_step = increased_step;
    } else {
      fos << "\nINFO: Cannot increase step. Using " << time_step << '\n';
    }

  }  // Continuation loop

  return;
}

}  // namespace LCM
