//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Albany_ModelFactory.hpp"
#include "Albany_SolverFactory.hpp"
#include "Albany_STKDiscretization.hpp"
#include "MiniTensor.h"
#include "Piro_LOCASolver.hpp"
#include "Piro_TempusSolver.hpp"
#include "Schwarz_Alternating.hpp"

#define DEBUG

namespace LCM {

//
//
//
SchwarzAlternating::
SchwarzAlternating(
    Teuchos::RCP<Teuchos::ParameterList> const & app_params,
    Teuchos::RCP<Teuchos::Comm<int> const> const & comm,
    Teuchos::RCP<Tpetra_Vector const> const & initial_guess) :
    fos_(Teuchos::VerboseObjectBase::getDefaultOStream())
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
  maximum_steps_ = alt_system_params.get<int>("Maximum Steps", 0);
  initial_time_ = alt_system_params.get<ST>("Initial Time", 0.0);
  final_time_ = alt_system_params.get<ST>("Final Time", 0.0);
  initial_time_step_ = alt_system_params.get<ST>("Initial Time Step", 0.0);
  output_interval_ = alt_system_params.get<int>("Exodus Write Interval", 1);

  //number of models
  num_subdomains_ = model_filenames.size();

  // Create application name-index map used for Schwarz BC.
  Teuchos::RCP<std::map<std::string, int>>
  app_name_index_map = Teuchos::rcp(new std::map<std::string, int>);

  for (auto subdomain = 0; subdomain < num_subdomains_; ++subdomain) {

    std::string const &
    app_name = model_filenames[subdomain];

    std::pair<std::string, int>
    app_name_index = std::make_pair(app_name, subdomain);

    app_name_index_map->insert(app_name_index);
  }

  // Arrays to cache useful info for each subdomain for later use
  apps_.resize(num_subdomains_);
  solvers_.resize(num_subdomains_);
  solution_sniffers_.resize(num_subdomains_);
  stk_mesh_structs_.resize(num_subdomains_);
  model_evaluators_.resize(num_subdomains_);
  sub_inargs_.resize(num_subdomains_);
  sub_outargs_.resize(num_subdomains_);
  solutions_.resize(num_subdomains_);
  have_loca_ = false;
  have_tempus_ = false;

  // Initialization
  for (auto subdomain = 0; subdomain < num_subdomains_; ++subdomain) {
    // Get parameters for each subdomain
    Albany::SolverFactory
    solver_factory(model_filenames[subdomain], comm);

    Teuchos::ParameterList &
    params = solver_factory.getParameters();

    // Add application array for later use in Schwarz BC.
    params.set("Application Array", apps_);

    // See application index for use with Schwarz BC.
    params.set("Application Index", subdomain);

    // Add application name-index map for later use in Schwarz BC.
    params.set("Application Name Index Map", app_name_index_map);

    // Add NOX pre-post-operator for Schwarz loop convergence criterion.
    bool const
    have_piro = params.isSublist("Piro");

    ALBANY_ASSERT(have_piro == true);

    Teuchos::ParameterList &
    piro_params = params.sublist("Piro");

    if (subdomain == 0) { 
      have_loca_ = piro_params.isSublist("LOCA");
    }
    else {
      bool const 
      have_loca = piro_params.isSublist("LOCA"); 
      if (have_loca != have_loca_) {
        TEUCHOS_TEST_FOR_EXCEPTION(true,
          std::logic_error, "Error in Schwarz::Alternating!  All problems "
          << "must have same solution method (LOCA or Tempus)!");
      } 
    }

    if (have_loca_ == true) {
      Teuchos::ParameterList &
      loca_params = piro_params.sublist("LOCA");

      Teuchos::ParameterList &
      stepper_params = loca_params.sublist("Stepper");

      start_stop_params_.emplace_back(stepper_params);
      init_str_.emplace_back("Initial Value");
      start_str_.emplace_back("Min Value");
      stop_str_.emplace_back("Max Value");
    }

    if (subdomain == 0) { 
      have_tempus_ = piro_params.isSublist("Tempus");
    }
    else {
      bool const 
      have_tempus = piro_params.isSublist("Tempus"); 
      if (have_tempus != have_tempus_) {
        TEUCHOS_TEST_FOR_EXCEPTION(true,
          std::logic_error, "Error in Schwarz::Alternating!  All problems "
          << "must have same solution method (LOCA or Tempus)!");
      } 
    }

    if (have_tempus_ == true) {
      Teuchos::ParameterList &
      tempus_params = piro_params.sublist("Tempus");

      Teuchos::ParameterList &
      integrator_params = tempus_params.sublist("Tempus Integrator");

      Teuchos::ParameterList &
      step_params = integrator_params.sublist("Time Step Control");

      start_stop_params_.emplace_back(step_params);
      init_str_.emplace_back("");
      start_str_.emplace_back("Initial Time");
      stop_str_.emplace_back("Final Time");
    }

    ALBANY_ASSERT(have_loca_ == true || have_tempus_ == true);

    //IKT, 8/11/17: we are only requiring NOX / creating SolutionSniffer
    //for LOCA, not Tempus.
    if (have_loca_ == true) {

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
        ppo = Teuchos::rcp(new SolutionSniffer);
        solver_opts.set(ppo_str, ppo);
        ALBANY_ASSERT(solver_opts.isParameter(ppo_str) == true);
      }

      constexpr bool
      throw_on_fail{true};

      Teuchos::RCP<SolutionSniffer>
      solution_sniffer = Teuchos::rcp_dynamic_cast<SolutionSniffer>
      (ppo, throw_on_fail);

      solution_sniffers_[subdomain] = solution_sniffer;

    }

    Teuchos::RCP<Albany::Application>
    app{Teuchos::null};

    Teuchos::RCP<Thyra::ResponseOnlyModelEvaluatorBase<ST>>
    solver = solver_factory.createAndGetAlbanyAppT(app, comm, comm);

    app->setAlternatingSchwarz(true);

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

    model_evaluators_[subdomain] = solver_factory.returnModelT();

    solutions_[subdomain] = Teuchos::null;
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
get_g_names(int) const
{
  ALBANY_ASSERT(false, "not implemented");
  return Teuchos::ArrayView<const std::string>(Teuchos::null);
}

//
//
//
Thyra::ModelEvaluatorBase::InArgs<ST>
SchwarzAlternating::
getNominalValues() const
{
  return this->createInArgsImpl();
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
//
//
void
SchwarzAlternating::
set_failed(char const * msg)
{
  failed_ = true;
  failure_message_ = msg;
  return;
}

//
//
//
void
SchwarzAlternating::
clear_failed()
{
  failed_ = false;
  return;
}

//
//
//
bool
SchwarzAlternating::
get_failed() const
{
  return failed_;
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
  ias;

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
SchwarzAlternating::
createOutArgsImpl() const
{
  Thyra::ModelEvaluatorBase::OutArgsSetup<ST>
  oas;

  oas.setModelEvalDescription(this->description());

  oas.setSupports(Thyra::ModelEvaluatorBase::OUT_ARG_f, true);
  oas.setSupports(Thyra::ModelEvaluatorBase::OUT_ARG_W_op, true);
  oas.setSupports(Thyra::ModelEvaluatorBase::OUT_ARG_W_prec, false);

  oas.set_W_properties(
      Thyra::ModelEvaluatorBase::DerivativeProperties(
          Thyra::ModelEvaluatorBase::DERIV_LINEARITY_UNKNOWN,
          Thyra::ModelEvaluatorBase::DERIV_RANK_FULL,
          true));

  return oas;
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
  if (have_loca_ == true) {
    SchwarzLoopLOCA();
  }
  if (have_tempus_ == true) {
    SchwarzLoopTempus();
  }
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
//
//
void
SchwarzAlternating::
updateConvergenceCriterion() const
{
  abs_error_ = norm_diff_;
  rel_error_ = norm_final_ > 0.0 ? norm_diff_ / norm_final_ : norm_diff_;

  bool const
  converged_absolute = abs_error_ <= abs_tol_;

  bool const
  converged_relative = rel_error_ <= rel_tol_;

  converged_ = converged_absolute || converged_relative;

  return;
}

//
//
//
bool
SchwarzAlternating::
continueSolve() const
{
  ++num_iter_;

  // If failure has occurred, stop immediately.
  if (failed_ == true) return false;

  // Regardless of other criteria, if error is zero stop solving.
  bool const
  zero_error = ((abs_error_ > 0.0) == false);

  if (zero_error == true) return false;

  // Minimum iterations takes precedence over maximum iterations and
  // convergence. Continue solving if not exceeded.
  bool const
  exceeds_min_iter = num_iter_ >= min_iters_;

  if (exceeds_min_iter == false) return true;

  // Maximum iterations takes precedence over convergence.
  // Stop solving if exceeded.
  bool const
  exceeds_max_iter = num_iter_ >= max_iters_;

  if (exceeds_max_iter == true) return false;

  // Lastly check for convergence.
  bool const
  continue_solve = (converged_ == false);

  return continue_solve;
}

//
//
//
void
SchwarzAlternating::
reportFinals(std::ostream & os) const
{
  std::string const
  conv_str = converged_ == true ? "YES" : "NO";

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
// Schwarz Alternating loop
//
void
SchwarzAlternating::
SchwarzLoopTempus() const
{
  minitensor::Vector<ST>
  norms_init(num_subdomains_, minitensor::Filler::ZEROS);

  minitensor::Vector<ST>
  norms_final(num_subdomains_, minitensor::Filler::ZEROS);

  minitensor::Vector<ST>
  norms_diff(num_subdomains_, minitensor::Filler::ZEROS);

  std::string const
  delim(72, '=');

  *fos_ << delim << std::endl;
  *fos_ << "Schwarz Alternating Method with " << num_subdomains_;
  *fos_ << " subdomains\n";
  *fos_ << std::scientific << std::setprecision(17);

  ST
  time_step{initial_time_step_};

  int
  stop{0};

  ST
  current_time{initial_time_};

  // Continuation loop
  while (stop < maximum_steps_ && current_time < final_time_) {

    *fos_ << delim << std::endl;
    *fos_ << "Time stop          :" << stop << '\n';
    *fos_ << "Time               :" << current_time << '\n';
    *fos_ << "Time step          :" << time_step << '\n';
    *fos_ << delim << std::endl;

    ST const
    next_time{current_time + time_step};

    num_iter_ = 0;

    // Disble output. Handle it after Schwarz iteration.
    for (auto subdomain = 0; subdomain < num_subdomains_; ++subdomain) {

      Albany::AbstractSTKMeshStruct &
      ams = *stk_mesh_structs_[subdomain];

      ams.exoOutput = false;
    }

    do {
    
      bool const
      is_initial_state = stop == 0 && num_iter_ == 0;

      for (auto subdomain = 0; subdomain < num_subdomains_; ++subdomain) {

        *fos_ << delim << std::endl;
        *fos_ << "Schwarz iteration  :" << num_iter_ << '\n';
        *fos_ << "Subdomain          :" << subdomain << '\n';
        *fos_ << delim << std::endl;

        // Solve for each subdomain
        Thyra::ResponseOnlyModelEvaluatorBase<ST> &
        solver = *(solvers_[subdomain]);

        *fos_ << "IKT creating Piro::TempusSolver...\n"; 
        Piro::TempusSolver<ST,LO,GO,KokkosNode> &
        piro_tempus_solver = dynamic_cast<Piro::TempusSolver<ST,LO,GO,KokkosNode> &>(solver);
        *fos_ << "done! \n"; 

        piro_tempus_solver.setStartTime(current_time); 
        piro_tempus_solver.setFinalTime(next_time); 
        //piro_tempus_solver.setInitTimeStep(time_step);

        *fos_ << "*** PIRO accessors/mutators ***\n";
        *fos_ << "Initial time       :" << piro_tempus_solver.getStartTime() << '\n';
        *fos_ << "Final time         :" << piro_tempus_solver.getFinalTime() << '\n';
        *fos_ << delim << std::endl;

        // For time dependent DBCs, set the time to be next time
        auto &
        app = *apps_[subdomain];

        app.setDBCTime(next_time);

        Thyra::ModelEvaluatorBase::InArgs<ST>
        in_args = solver.createInArgs();

        Thyra::ModelEvaluatorBase::OutArgs<ST>
        out_args = solver.createOutArgs();

        auto &
        me = dynamic_cast<Albany::ModelEvaluatorT &>
        (*model_evaluators_[subdomain]);

        me.getNominalValues().set_t(current_time);

        //IKT, 8/11/17: the following is a temporary assert to prevent user from 
        //running SchwarzLoopTempus before it is complete.
        ALBANY_ASSERT(have_tempus_ == false, "SchwarzLoopTempus() not fully implemented!");  
      }
    }  while (continueSolve() == true);
  }

}

void
SchwarzAlternating::
SchwarzLoopLOCA() const
{
  minitensor::Vector<ST>
  norms_init(num_subdomains_, minitensor::Filler::ZEROS);

  minitensor::Vector<ST>
  norms_final(num_subdomains_, minitensor::Filler::ZEROS);

  minitensor::Vector<ST>
  norms_diff(num_subdomains_, minitensor::Filler::ZEROS);

  std::string const
  delim(72, '=');

  *fos_ << delim << std::endl;
  *fos_ << "Schwarz Alternating Method with " << num_subdomains_;
  *fos_ << " subdomains\n";
  *fos_ << std::scientific << std::setprecision(17);

  ST
  time_step{initial_time_step_};

  int
  stop{0};

  ST
  current_time{initial_time_};

  // Continuation loop
  while (stop < maximum_steps_ && current_time < final_time_) {

    *fos_ << delim << std::endl;
    *fos_ << "Time stop          :" << stop << '\n';
    *fos_ << "Time               :" << current_time << '\n';
    *fos_ << "Time step          :" << time_step << '\n';
    *fos_ << delim << std::endl;

    ST const
    next_time{current_time + time_step};

    num_iter_ = 0;

    // Disble output. Handle it after Schwarz iteration.
    for (auto subdomain = 0; subdomain < num_subdomains_; ++subdomain) {

      Albany::AbstractSTKMeshStruct &
      ams = *stk_mesh_structs_[subdomain];

      ams.exoOutput = false;
    }

    do {

      bool const
      is_initial_state = stop == 0 && num_iter_ == 0;

      for (auto subdomain = 0; subdomain < num_subdomains_; ++subdomain) {

        *fos_ << delim << std::endl;
        *fos_ << "Schwarz iteration  :" << num_iter_ << '\n';
        *fos_ << "Subdomain          :" << subdomain << '\n';
        *fos_ << delim << std::endl;

        // Solve for each subdomain
        Thyra::ResponseOnlyModelEvaluatorBase<ST> &
        solver = *(solvers_[subdomain]);

        Piro::LOCASolver<ST> &
        piro_loca_solver = dynamic_cast<Piro::LOCASolver<ST> &>(solver);

        Teuchos::ParameterList &
        start_stop_params = piro_loca_solver.getStepperParams();

        std::string const &
        init_str = init_str_[subdomain];

        std::string const &
        start_str = start_str_[subdomain];

        std::string const &
        stop_str = stop_str_[subdomain];

        piro_loca_solver.setStartValue(current_time);
        piro_loca_solver.setMinValue(current_time);
        piro_loca_solver.setMaxValue(next_time);

        start_stop_params.set(init_str, current_time);
        start_stop_params.set(start_str, current_time);
        start_stop_params.set(stop_str, next_time);

        *fos_ << "*** PIRO accessors/mutators ***\n";
        *fos_ << "Initial time       :" << piro_loca_solver.getStartValue() << '\n';
        *fos_ << "Start time         :" << piro_loca_solver.getMinValue() << '\n';
        *fos_ << "Stop time          :" << piro_loca_solver.getMaxValue() << '\n';
        *fos_ << delim << std::endl;
        *fos_ << "*** ParameterList accessors/mutators ***\n";
        *fos_ << "Initial time       :" << start_stop_params.get<double>(init_str) << '\n';
        *fos_ << "Start time         :" << start_stop_params.get<double>(start_str) << '\n';
        *fos_ << "Stop time          :" << start_stop_params.get<double>(stop_str) << '\n';
        *fos_ << delim << std::endl;

        // For time dependent DBCs, set the time to be next time
        auto &
        app = *apps_[subdomain];

        app.setDBCTime(next_time);

        Thyra::ModelEvaluatorBase::InArgs<ST>
        in_args = solver.createInArgs();

        Thyra::ModelEvaluatorBase::OutArgs<ST>
        out_args = solver.createOutArgs();

        auto &
        me = dynamic_cast<Albany::ModelEvaluatorT &>
        (*model_evaluators_[subdomain]);

        me.getNominalValues().set_t(current_time);

        NOX::Solver::Generic &
        nox_solver = *piro_loca_solver.getSolver();

        Teuchos::RCP<NOX::Abstract::Vector>
        prev_soln_rcp = is_initial_state == true ?
            nox_solver.getPreviousSolutionGroup().getX().clone(NOX::DeepCopy) :
            solutions_[subdomain];

        NOX::Abstract::Vector &
        prev_soln = *prev_soln_rcp;

#if defined(DEBUG)
        *fos_ << "\n*** NOX: Previous solution ***\n";
        prev_soln.print(*fos_);
        *fos_ << "\n*** NORM: " << prev_soln.norm() << '\n';
        *fos_ << "\n*** NOX: Previous solution ***\n";
#endif //DEBUG

        NOX::Abstract::Group &
        nox_group =
            const_cast<NOX::Abstract::Group &>(nox_solver.getSolutionGroup());

        // Use previous solution as initial condition for next step
        if (is_initial_state == false) {
          nox_group.setX(prev_soln);
        }

        solver.evalModel(in_args, out_args);

        Teuchos::RCP<SolutionSniffer>
        solution_sniffer = solution_sniffers_[subdomain];

        Teuchos::RCP<NOX::Abstract::Vector>
        curr_soln_rcp = solution_sniffer->getLastSoln();

        NOX::Abstract::Vector const &
        curr_soln = *curr_soln_rcp;

#if defined(DEBUG)
        *fos_ << "\n*** NOX: Current solution ***\n";
        curr_soln.print(*fos_);
        *fos_ << "\n*** NORM: " << curr_soln.norm() << '\n';
        *fos_ << "\n*** NOX: Current solution ***\n";
#endif //DEBUG

        Teuchos::RCP<NOX::Abstract::Vector>
        soln_diff_rcp = curr_soln.clone(NOX::DeepCopy);

        NOX::Abstract::Vector &
        soln_diff = *(soln_diff_rcp);

        soln_diff.update(1.0, curr_soln, -1.0, prev_soln, 0.0);

#if defined(DEBUG)
        *fos_ << "\n*** NOX: Solution difference ***\n";
        soln_diff.print(*fos_);
        *fos_ << "\n*** NORM: " << soln_diff.norm() << '\n';
        *fos_ << "\n*** NOX: Solution difference ***\n";
#endif //DEBUG

        // After solve, save solution and get info to check convergence
        solutions_[subdomain] = curr_soln_rcp;
        norms_init(subdomain) = prev_soln.norm();
        norms_final(subdomain) = curr_soln.norm();
        norms_diff(subdomain) = soln_diff.norm();
      }

      norm_init_ = minitensor::norm(norms_init);
      norm_final_ = minitensor::norm(norms_final);
      norm_diff_ = minitensor::norm(norms_diff);

      updateConvergenceCriterion();

      *fos_ << delim << std::endl;
      *fos_ << "Schwarz iteration         :" << num_iter_ << '\n';

      std::string const
      line(72, '-');

      *fos_ << line << std::endl;

      *fos_ << centered("Sub", 4);
      *fos_ << centered("Initial norm", 24);
      *fos_ << centered("Final norm", 24);
      *fos_ << centered("Difference norm", 24);
      *fos_ << std::endl;

      *fos_ << centered("dom", 4);
      *fos_ << centered("||X0||", 24);
      *fos_ << centered("||Xf||", 24);
      *fos_ << centered("||Xf-X0||", 24);
      *fos_ << std::endl;

      *fos_ << line << std::endl;

      for (auto m = 0; m < num_subdomains_; ++m) {
        *fos_ << std::setw(4) << m;
        *fos_ << std::setw(24) << norms_init(m);
        *fos_ << std::setw(24) << norms_final(m);
        *fos_ << std::setw(24) << norms_diff(m);
        *fos_ << std::endl;
      }

      *fos_ << line << std::endl;

      *fos_ << centered("Norm", 4);
      *fos_ << std::setw(24) << norm_init_;
      *fos_ << std::setw(24) << norm_final_;
      *fos_ << std::setw(24) << norm_diff_;
      *fos_ << std::endl;

      *fos_ << line << std::endl;

      *fos_ << "Absolute error     :" << abs_error_ << '\n';
      *fos_ << "Absolute tolerance :" << abs_tol_ << '\n';
      *fos_ << "Relative error     :" << rel_error_ << '\n';
      *fos_ << "Relative tolerance :" << rel_tol_ << '\n';
      *fos_ << delim << std::endl;

    }  while (continueSolve() == true);

    reportFinals(*fos_);

    // Print converged solution if at specified interval
    for (auto subdomain = 0; subdomain < num_subdomains_; ++subdomain) {

      Albany::AbstractSTKMeshStruct &
      ams = *stk_mesh_structs_[subdomain];

      ams.exoOutputInterval = 1;

      ams.exoOutput = output_interval_ > 0 ?
          (stop + 1) % output_interval_ == 0 : false;

      Thyra::ResponseOnlyModelEvaluatorBase<ST> &
      solver = *(solvers_[subdomain]);

      Piro::LOCASolver<ST> &
      piro_loca_solver = dynamic_cast<Piro::LOCASolver<ST> &>(solver);

      piro_loca_solver.printSolution();
    }

    ++stop;
    current_time += time_step;
  }


  return;
}

} // namespace LCM
