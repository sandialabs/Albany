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
  maximum_steps_ = alt_system_params.get<int>("Maximum Steps", 0);
  initial_time_ = alt_system_params.get<ST>("Initial Time", 0.0);
  final_time_ = alt_system_params.get<ST>("Final Time", 0.0);
  initial_time_step_ = alt_system_params.get<ST>("Initial Time Step", 1.0);
  use_velo_in_conv_criterion_ = alt_system_params.get<bool>("Use Velocity in Convergence Criterion", true); 
  use_acce_in_conv_criterion_ = alt_system_params.get<bool>("Use Acceleration in Convergence Criterion", true); 

#ifdef DEBUG
  auto &
  fos = *Teuchos::VerboseObjectBase::getDefaultOStream();
  fos << "Use Velocity in Convergence Criterion = " << use_velo_in_conv_criterion_ << "\n";
  fos << "Use Acceleration in Convergence Criterion = " << use_acce_in_conv_criterion_ << "\n";
#endif

  ST const
  dt = initial_time_step_;

  min_time_step_ = alt_system_params.get<ST>("Minimum Time Step", dt);
  max_time_step_ = alt_system_params.get<ST>("Maximum Time Step", dt);
  reduction_factor_ = alt_system_params.get<ST>("Reduction Factor", 1.0);
  increase_factor_ = alt_system_params.get<ST>("Increase Factor", 1.0);
  output_interval_ = alt_system_params.get<int>("Exodus Write Interval", 1);

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
  stk_mesh_structs_.resize(num_subdomains_);
  discs_.resize(num_subdomains_);
  model_evaluators_.resize(num_subdomains_);
  sub_inargs_.resize(num_subdomains_);
  sub_outargs_.resize(num_subdomains_);
  disp_nox_.resize(num_subdomains_);
  prev_disp_nox_.resize(num_subdomains_);
  ics_disp_.resize(num_subdomains_);
  ics_velo_.resize(num_subdomains_);
  ics_acce_.resize(num_subdomains_);
  prev_disp_thyra_.resize(num_subdomains_);
  prev_velo_thyra_.resize(num_subdomains_);
  prev_acce_thyra_.resize(num_subdomains_);
  internal_states_.resize(num_subdomains_);

  bool
  have_nox{false};

  bool
  have_tempus{false};

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

    std::string const
    msg{"All subdomains must have the same solution method (NOX or Tempus)"};

    if (subdomain == 0) { 
      have_nox = piro_params.isSublist("NOX");
      have_tempus = piro_params.isSublist("Tempus");
      ALBANY_ASSERT(have_nox != have_tempus, "Must have either NOX or Tempus");
      have_nox_ = have_nox;
      have_tempus_ = have_tempus;
    }
    else {
      ALBANY_ASSERT(have_nox == piro_params.isSublist("NOX"), msg);
      ALBANY_ASSERT(have_tempus == piro_params.isSublist("Tempus"), msg);
    }

    Teuchos::RCP<Albany::Application>
    app{Teuchos::null};

    Teuchos::RCP<Thyra::ResponseOnlyModelEvaluatorBase<ST>>
    solver = solver_factory.createAndGetAlbanyAppT(app, comm, comm);

    solvers_[subdomain] = solver;

    app->setSchwarzAlternating(true);

    apps_[subdomain] = app;

    // Get STK mesh structs to control Exodus output interval
    Teuchos::RCP<Albany::AbstractDiscretization>
    disc = app->getDiscretization();
    
    discs_[subdomain] = disc; 

    Albany::STKDiscretization &
    stk_disc = *static_cast<Albany::STKDiscretization *>(disc.get());

    Teuchos::RCP<Albany::AbstractSTKMeshStruct>
    ams = stk_disc.getSTKMeshStruct();

    stk_mesh_structs_[subdomain] = ams;

    model_evaluators_[subdomain] = solver_factory.returnModelT();

    disp_nox_[subdomain] = Teuchos::null;
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
  if (have_nox_ == true) {
    SchwarzLoopQuasistatics();
  }
  if (have_tempus_ == true) {
    SchwarzLoopDynamics();
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
// Schwarz Alternating loop, dynamic
//
void
SchwarzAlternating::
SchwarzLoopDynamics() const
{
  minitensor::Vector<ST>
  norms_init(num_subdomains_, minitensor::Filler::ZEROS);

  minitensor::Vector<ST>
  norms_final(num_subdomains_, minitensor::Filler::ZEROS);

  minitensor::Vector<ST>
  norms_diff(num_subdomains_, minitensor::Filler::ZEROS);

  std::string const
  delim(72, '=');

  auto &
  fos = *Teuchos::VerboseObjectBase::getDefaultOStream();

  fos << delim << std::endl;
  fos << "Schwarz Alternating Method with " << num_subdomains_;
  fos << " subdomains\n";
  fos << std::scientific << std::setprecision(17);

  ST
  time_step{initial_time_step_};

  int
  stop{0};

  ST
  current_time{initial_time_};

  // Output initial configuration. Then disable output.
  // Handle it after Schwarz iteration.

  for (auto subdomain = 0; subdomain < num_subdomains_; ++subdomain) {

 
    Albany::AbstractSTKMeshStruct &
    ams = *stk_mesh_structs_[subdomain];

    ams.exoOutputInterval = 1;
    ams.exoOutput = true;

    Albany::AbstractDiscretization &
    abs_disc = *discs_[subdomain];

    Albany::STKDiscretization &
    stk_disc = static_cast<Albany::STKDiscretization &>(abs_disc);

    // Populate ics_disp_ and its time-derivatives with values of IC
    // from nominal values in model evaluator.
    auto &
    me = dynamic_cast<Albany::ModelEvaluatorT &>
    (*model_evaluators_[subdomain]);

    auto const &
    nv = me.getNominalValues();
        
    ics_disp_[subdomain] = Thyra::createMember(me.get_x_space());
    Thyra::copy(*(nv.get_x()), ics_disp_[subdomain].ptr());
    //fos << "IKT subdomain = " << subdomain << ", ics_disp_  = " << std::endl; 
    //const Teuchos::RCP<const Tpetra_Vector> ic =
    //ConverterT::getConstTpetraVector(ics_disp_[subdomain]);
    //ic->describe(fos, Teuchos::VERB_EXTREME);

    ics_velo_[subdomain] = Thyra::createMember(me.get_x_space());
    Thyra::copy(*(nv.get_x_dot()), ics_velo_[subdomain].ptr());

    ics_acce_[subdomain] = Thyra::createMember(me.get_x_space());
    Thyra::copy(*(nv.get_x_dot_dot()), ics_acce_[subdomain].ptr());

    prev_disp_thyra_[subdomain] = Thyra::createMember(me.get_x_space());
    Thyra::copy(*(nv.get_x()), prev_disp_thyra_[subdomain].ptr());
    
    prev_velo_thyra_[subdomain] = Thyra::createMember(me.get_x_space());
    Thyra::copy(*(nv.get_x_dot()), prev_velo_thyra_[subdomain].ptr());

    prev_acce_thyra_[subdomain] = Thyra::createMember(me.get_x_space());
    Thyra::copy(*(nv.get_x_dot_dot()), prev_acce_thyra_[subdomain].ptr());

    //Write initial condition to STK mesh 
    Teuchos::RCP<Tpetra_Vector const> const
    prev_disp_tpetra = ConverterT::getConstTpetraVector(prev_disp_thyra_[subdomain]);

    Teuchos::RCP<Tpetra_Vector const> const
    prev_velo_tpetra = ConverterT::getConstTpetraVector(prev_velo_thyra_[subdomain]);

    Teuchos::RCP<Tpetra_Vector const> const
    prev_acce_tpetra = ConverterT::getConstTpetraVector(prev_acce_thyra_[subdomain]);

#if defined(DEBUG)
    if (subdomain == 0) {
      Tpetra_MatrixMarket_Writer::writeDenseFile("init_disp0.mm", prev_disp_tpetra);
    }
    else if (subdomain == 1) {
      Tpetra_MatrixMarket_Writer::writeDenseFile("init_disp1.mm", prev_disp_tpetra);
    }
#endif

    const Teuchos::RCP<const Tpetra_MultiVector> xMV =
      apps_[subdomain]->getAdaptSolMgrT()->getOverlappedSolution();

    stk_disc.writeSolutionMV(*xMV, initial_time_, true); 

    ams.exoOutput = false;

  }


  // Continuation loop
  while (stop < maximum_steps_ && current_time < final_time_) {

    fos << delim << std::endl;
    fos << "Time stop          :" << stop << '\n';
    fos << "Time               :" << current_time << '\n';
    fos << "Time step          :" << time_step << '\n';
    fos << delim << std::endl;

    ST const
    next_time{current_time + time_step};

    num_iter_ = 0;

    do {
    
      bool const
      is_initial_state = stop == 0 && num_iter_ == 0;

      for (auto subdomain = 0; subdomain < num_subdomains_; ++subdomain) {

        fos << delim << std::endl;
        fos << "Schwarz iteration  :" << num_iter_ << '\n';
        fos << "Subdomain          :" << subdomain << '\n';
        fos << delim << std::endl;

        // Solve for each subdomain
        Thyra::ResponseOnlyModelEvaluatorBase<ST> &
        solver = *(solvers_[subdomain]);

        Piro::TempusSolver<ST, LO, GO, KokkosNode> &
        piro_tempus_solver =
            dynamic_cast<Piro::TempusSolver<ST, LO, GO, KokkosNode> &>(solver);

        piro_tempus_solver.setStartTime(current_time); 
        piro_tempus_solver.setFinalTime(next_time); 
        piro_tempus_solver.setInitTimeStep(time_step);

        double const
        tempus_start_time = piro_tempus_solver.getStartTime();

        double const
        tempus_final_time = piro_tempus_solver.getFinalTime();

        double const
        tempus_time_step = piro_tempus_solver.getInitTimeStep();

        fos << "Initial time       :" << tempus_start_time << '\n';
        fos << "Final time         :" << tempus_final_time << '\n';
        fos << "Time step          :" << tempus_time_step << '\n';
        fos << delim << std::endl;

        auto &
        app = *apps_[subdomain];

        Thyra::ModelEvaluatorBase::InArgs<ST>
        in_args = solver.createInArgs();

        Thyra::ModelEvaluatorBase::OutArgs<ST>
        out_args = solver.createOutArgs();

        auto &
        me = dynamic_cast<Albany::ModelEvaluatorT &>
        (*model_evaluators_[subdomain]);

        me.getNominalValues().set_t(current_time);

        Teuchos::RCP<Tempus::SolutionHistory<ST> > 
        solution_history; 
          
        Teuchos::RCP<Tempus::SolutionState<ST>> 
        current_state;

        Teuchos::RCP<Thyra::VectorBase<ST>> 
        ic_disp_rcp = Thyra::createMember(me.get_x_space());
        
        Teuchos::RCP<Thyra::VectorBase<ST>> 
        ic_velo_rcp = Thyra::createMember(me.get_x_space());
        
        Teuchos::RCP<Thyra::VectorBase<ST>> 
        ic_acce_rcp = Thyra::createMember(me.get_x_space());

        //set ic_disp_rcp, ic_velo_rcp and ic_acce_rcp 
        //by making copy of what is in ics_disp_[subdomain], etc.
        Thyra::VectorBase<ST> &
        ic_disp = *ics_disp_[subdomain];

        Thyra::VectorBase<ST> &
        ic_velo = *ics_velo_[subdomain];

        Thyra::VectorBase<ST> &
        ic_acce = *ics_acce_[subdomain];

        Thyra::copy(ic_disp, ic_disp_rcp.ptr());

        Thyra::copy(ic_velo, ic_velo_rcp.ptr());

        Thyra::copy(ic_acce, ic_acce_rcp.ptr());

        piro_tempus_solver.setInitialState(
            current_time,
            ic_disp_rcp,
            ic_velo_rcp,
            ic_acce_rcp);

        solver.evalModel(in_args, out_args);  
        
#if defined(DEBUG)
        Teuchos::RCP<Tpetra_Vector>
        prev_disp_tpetra;

        fos << "\n*** Thyra: Previous solution ***\n";
        prev_disp_thyra_[subdomain]->describe(fos, Teuchos::VERB_EXTREME);
        fos << "\n*** NORM: " << Thyra::norm(*prev_disp_thyra_[subdomain]) << '\n';
        if (subdomain == 0) {
          prev_disp_tpetra = ConverterT::getTpetraVector(prev_disp_thyra_[0]);
          Albany::writeMatrixMarket(prev_disp_tpetra, "prev_disp_subd0", num_iter_);
        }
        else if (subdomain == 1) {
          prev_disp_tpetra = ConverterT::getTpetraVector(prev_disp_thyra_[1]);
          Albany::writeMatrixMarket(prev_disp_tpetra, "prev_disp_subd1", num_iter_);
        }
        fos << "\n*** Thyra: Previous solution ***\n";
#endif //DEBUG
        
        solution_history = piro_tempus_solver.getSolutionHistory();

        current_state = solution_history->getCurrentState();

        Teuchos::RCP<Thyra::VectorBase<ST>> 
        curr_disp_rcp = Thyra::createMember(me.get_x_space());
        Thyra::copy(*current_state->getX(), curr_disp_rcp.ptr());

        Teuchos::RCP<Thyra::VectorBase<ST>> 
        curr_velo_rcp = Thyra::createMember(me.get_x_space());
        Thyra::copy(*current_state->getXDot(), curr_velo_rcp.ptr());

        Teuchos::RCP<Thyra::VectorBase<ST>> 
        curr_acce_rcp = Thyra::createMember(me.get_x_space());
        Thyra::copy(*current_state->getXDotDot(), curr_acce_rcp.ptr());

#if defined(DEBUG)
        fos << "\n*** Thyra: Current solution ***\n";
        curr_disp_rcp->describe(fos, Teuchos::VERB_EXTREME);
        fos << "\n*** NORM: " << Thyra::norm(*curr_disp_rcp) << '\n';
        fos << "\n*** Thyra: Current solution ***\n";

        Teuchos::RCP<Tpetra_Vector>
        curr_disp_tpetra;

        if (subdomain == 0) {
          curr_disp_tpetra = ConverterT::getTpetraVector(curr_disp_rcp); 
          Albany::writeMatrixMarket(curr_disp_tpetra, "curr_disp_subd0", num_iter_);
        }
        else if (subdomain == 1) {
          curr_disp_tpetra = ConverterT::getTpetraVector(curr_disp_rcp); 
          Albany::writeMatrixMarket(curr_disp_tpetra, "curr_disp_subd1", num_iter_);
        }
#endif //DEBUG

        Teuchos::RCP<Thyra::VectorBase<ST>>
        disp_diff_rcp = Thyra::createMember(me.get_x_space());
        Thyra::put_scalar<ST>(0.0, disp_diff_rcp.ptr()); 
        Thyra::V_VpStV(
            disp_diff_rcp.ptr(),
            *curr_disp_rcp,
            -1.0,
            *prev_disp_thyra_[subdomain]);

        Teuchos::RCP<Thyra::VectorBase<ST>>
        velo_diff_rcp = Thyra::createMember(me.get_x_space());
        Thyra::put_scalar<ST>(0.0, velo_diff_rcp.ptr()); 
        Thyra::V_VpStV(
            velo_diff_rcp.ptr(),
            *curr_velo_rcp,
            -1.0,
            *prev_velo_thyra_[subdomain]);

        Teuchos::RCP<Thyra::VectorBase<ST>>
        acce_diff_rcp = Thyra::createMember(me.get_x_space());
        Thyra::put_scalar<ST>(0.0, acce_diff_rcp.ptr()); 
        Thyra::V_VpStV(
            acce_diff_rcp.ptr(),
            *curr_acce_rcp,
            -1.0,
            *prev_acce_thyra_[subdomain]);

#if defined(DEBUG)
        fos << "\n*** Thyra: Solution difference ***\n"; 
        disp_diff_rcp->describe(fos, Teuchos::VERB_EXTREME); 
        fos << "\n*** NORM: " << Thyra::norm(*disp_diff_rcp) << '\n';
        fos << "\n*** Thyra: Solution difference ***\n";

        Teuchos::RCP<Tpetra_Vector>
        disp_diff_tpetra;

        if (subdomain == 0) {
          disp_diff_tpetra = ConverterT::getTpetraVector(disp_diff_rcp); 
          Albany::writeMatrixMarket(disp_diff_tpetra, "disp_diff_subd0", num_iter_);
        }
        else if (subdomain == 1) {
          disp_diff_tpetra = ConverterT::getTpetraVector(disp_diff_rcp); 
          Albany::writeMatrixMarket(disp_diff_tpetra, "disp_diff_subd1", num_iter_);
        }
#endif //DEBUG

        //After solve, save solution and get info to check convergence
        norms_init(subdomain) = Thyra::norm(*prev_disp_thyra_[subdomain]); 
        norms_final(subdomain) = Thyra::norm(*curr_disp_rcp);
        norms_diff(subdomain) = Thyra::norm(*disp_diff_rcp);
         
        if (use_velo_in_conv_criterion_ == true) {
          norms_init(subdomain)  += time_step*Thyra::norm(*prev_velo_thyra_[subdomain]); 
          norms_final(subdomain) += time_step*Thyra::norm(*curr_velo_rcp); 
          norms_diff(subdomain)  += time_step*Thyra::norm(*velo_diff_rcp);
        }
       
        if (use_acce_in_conv_criterion_ == true) { 
          norms_init(subdomain)  += time_step*time_step*Thyra::norm(*prev_acce_thyra_[subdomain]); 
          norms_final(subdomain) += time_step*time_step*Thyra::norm(*curr_acce_rcp); 
          norms_diff(subdomain)  += time_step*time_step*Thyra::norm(*acce_diff_rcp); 
        }
 
        //Update prev_disp_thyra_. 
        Thyra::put_scalar(0.0, prev_disp_thyra_[subdomain].ptr());  
        Thyra::copy(*curr_disp_rcp, prev_disp_thyra_[subdomain].ptr());
        Thyra::put_scalar(0.0, prev_velo_thyra_[subdomain].ptr());  
        Thyra::copy(*curr_velo_rcp, prev_velo_thyra_[subdomain].ptr());
        Thyra::put_scalar(0.0, prev_acce_thyra_[subdomain].ptr());  
        Thyra::copy(*curr_acce_rcp, prev_acce_thyra_[subdomain].ptr());

      } //subdomains loop 

      norm_init_ = minitensor::norm(norms_init);
      norm_final_ = minitensor::norm(norms_final);
      norm_diff_ = minitensor::norm(norms_diff);

      updateConvergenceCriterion(); 

      fos << delim << std::endl;
      fos << "Schwarz iteration         :" << num_iter_ << '\n';
 
      std::string const
      line(72, '-');

      fos << line << std::endl;

      fos << centered("Sub", 4);
      fos << centered("Initial norm", 24);
      fos << centered("Final norm", 24);
      fos << centered("Difference norm", 24);
      fos << std::endl;

      fos << centered("dom", 4);
      fos << centered("||X0||", 24);
      fos << centered("||Xf||", 24);
      fos << centered("||Xf-X0||", 24);
      fos << std::endl;

      fos << line << std::endl;

      for (auto m = 0; m < num_subdomains_; ++m) {
        fos << std::setw(4) << m;
        fos << std::setw(24) << norms_init(m);
        fos << std::setw(24) << norms_final(m);
        fos << std::setw(24) << norms_diff(m);
        fos << std::endl;
      }

      fos << line << std::endl;

      fos << centered("Norm", 4);
      fos << std::setw(24) << norm_init_;
      fos << std::setw(24) << norm_final_;
      fos << std::setw(24) << norm_diff_;
      fos << std::endl;

      fos << line << std::endl;

      fos << "Absolute error     :" << abs_error_ << '\n';
      fos << "Absolute tolerance :" << abs_tol_ << '\n';
      fos << "Relative error     :" << rel_error_ << '\n';
      fos << "Relative tolerance :" << rel_tol_ << '\n';
      fos << delim << std::endl;
 
    }  while (continueSolve() == true);

    reportFinals(fos);

    //Output converged solution if at specified interval 
    for (auto subdomain = 0; subdomain < num_subdomains_; ++subdomain) {

      Albany::AbstractSTKMeshStruct &
      ams = *stk_mesh_structs_[subdomain];

      ams.exoOutputInterval = 1;

      ams.exoOutput = output_interval_ > 0 ?
          (stop + 1) % output_interval_ == 0 : false;
      
      Albany::AbstractDiscretization &
      abs_disc = *discs_[subdomain];

      Albany::STKDiscretization &
      stk_disc = static_cast<Albany::STKDiscretization &>(abs_disc);

      Teuchos::RCP<Tpetra_MultiVector>
      disp_mv = stk_disc.getSolutionMV();

      //Update ics_disp_ and its time-derivatives
      ics_disp_[subdomain] =
          Thyra::createVector(disp_mv->getVectorNonConst(0));

      ics_velo_[subdomain] =
          Thyra::createVector(disp_mv->getVectorNonConst(1));

      ics_acce_[subdomain] =
          Thyra::createVector(disp_mv->getVectorNonConst(2));

      if (ams.exoOutput == true) {
        stk_disc.writeSolutionMV(*disp_mv, current_time + time_step);
      }

      ams.exoOutput = false;
    }

    ++stop;
    current_time += time_step;
  }

  return; 
}

//
// Schwarz Alternating loop, quasistatic
//
void
SchwarzAlternating::
SchwarzLoopQuasistatics() const
{
  minitensor::Vector<ST>
  norms_init(num_subdomains_, minitensor::Filler::ZEROS);

  minitensor::Vector<ST>
  norms_final(num_subdomains_, minitensor::Filler::ZEROS);

  minitensor::Vector<ST>
  norms_diff(num_subdomains_, minitensor::Filler::ZEROS);

  std::string
  init_str{"Initial Value"};

  std::string
  start_str{"Min Value"};

  std::string
  stop_str{"Max Value"};

  std::string
  step_str{"Initial Step Size"};

  std::string const
  delim(72, '=');

  auto &
  fos = *Teuchos::VerboseObjectBase::getDefaultOStream();

  fos << delim << std::endl;
  fos << "Schwarz Alternating Method with " << num_subdomains_;
  fos << " subdomains\n";
  fos << std::scientific << std::setprecision(17);

  ST
  time_step{initial_time_step_};

  int
  stop{0};

  ST
  current_time{initial_time_};

  // Output initial configuration. Then disable output.
  // Handle it after Schwarz iteration.

  for (auto subdomain = 0; subdomain < num_subdomains_; ++subdomain) {

    Albany::AbstractSTKMeshStruct &
    ams = *stk_mesh_structs_[subdomain];

    ams.exoOutputInterval = 1;
    ams.exoOutput = true;

    Albany::AbstractDiscretization &
    abs_disc = *discs_[subdomain];
  
    Albany::STKDiscretization &
    stk_disc = static_cast<Albany::STKDiscretization &>(abs_disc);

    Teuchos::RCP<Tpetra_MultiVector>
    disp_mv_rcp = stk_disc.getSolutionMV();

    stk_disc.writeSolutionMV(*disp_mv_rcp, initial_time_);

    ams.exoOutput = false;

  }

  // Continuation loop
  while (stop < maximum_steps_ && current_time < final_time_) {

    fos << delim << std::endl;
    fos << "Time stop          :" << stop << '\n';
    fos << "Time               :" << current_time << '\n';
    fos << "Time step          :" << time_step << '\n';
    fos << delim << std::endl;

    ST const
    next_time{current_time + time_step};

    num_iter_ = 0;

    bool const
    is_initial_state = stop == 0 && num_iter_ == 0;

    // Before the Schwarz loop, save the solutions for each subdomain in case
    // the solve phase fails. Then the load step is reduced and the Schwarz
    // loop is restarted from scratch.
    for (auto subdomain = 0; subdomain < num_subdomains_; ++subdomain) {

      prev_disp_nox_[subdomain] =
          is_initial_state == true ? Teuchos::null : disp_nox_[subdomain];

      auto &
      app = *apps_[subdomain];

      auto &
      state_mgr = app.getStateMgr();

      internal_states_[subdomain] = state_mgr.getStateArrays();
    }

    // Schwarz loop
    do {

      // Subdomain loop
      for (auto subdomain = 0; subdomain < num_subdomains_; ++subdomain) {

        fos << delim << std::endl;
        fos << "Schwarz iteration  :" << num_iter_ << '\n';
        fos << "Subdomain          :" << subdomain << '\n';
        fos << delim << std::endl;

        // Solve for each subdomain
        auto &
        solver = *(solvers_[subdomain]);

        auto &
        piro_nox_solver = dynamic_cast<Piro::NOXSolver<ST> &>(solver);

        fos << "Start time         :" << current_time << '\n';
        fos << "Stop time          :" << next_time << '\n';
        fos << "Step size          :" << time_step << '\n';
        fos << delim << std::endl;

        auto &
        app = *apps_[subdomain];

        auto &
        state_mgr = app.getStateMgr();

        state_mgr.setStateArrays(internal_states_[subdomain]);

        auto
        in_args = solver.createInArgs();

        auto
        out_args = solver.createOutArgs();

        auto &
        me = dynamic_cast<Albany::ModelEvaluatorT &>
        (*model_evaluators_[subdomain]);

        me.setCurrentTime(current_time);

        auto
        prev_disp_rcp = is_initial_state == true ?
            me.getNominalValues().get_x() :
            prev_disp_nox_[subdomain];

        auto const &
        prev_disp = *prev_disp_rcp;

        me.getNominalValues().set_x(prev_disp_rcp);

#if defined(DEBUG)
        fos << "\n*** NOX: Previous solution ***\n";
        prev_disp.describe(fos, Teuchos::VERB_EXTREME);
        fos << "\n*** NORM: " << Thyra::norm(prev_disp) << '\n';
        fos << "\n*** NOX: Previous solution ***\n";
#endif //DEBUG

        solver.evalModel(in_args, out_args);

        auto const &
        thyra_nox_solver = *piro_nox_solver.getSolver();

        auto const &
        const_nox_solver = *thyra_nox_solver.getNOXSolver();

        auto &
        nox_solver = const_cast<NOX::Solver::Generic &>(const_nox_solver);

        auto const
        status = nox_solver.getStatus();

        if (status == NOX::StatusTest::Failed) {
          fos << "\nINFO: Unable to solve for subdomain " << subdomain << '\n';
          failed_ = true;
          // Break out of the subdomain loop
          break;
        }

        auto
        curr_disp_rcp = thyra_nox_solver.get_current_x();

        auto const &
        curr_disp = *curr_disp_rcp;

#if defined(DEBUG)
        fos << "\n*** NOX: Current solution ***\n";
        curr_disp.describe(fos, Teuchos::VERB_EXTREME);
        fos << "\n*** NORM: " << Thyra::norm(curr_disp) << '\n';
        fos << "\n*** NOX: Current solution ***\n";
#endif //DEBUG

        Teuchos::RCP<Thyra::VectorBase<ST>>
        disp_diff_rcp = Thyra::createMember(me.get_x_space());

        Thyra::put_scalar<ST>(0.0, disp_diff_rcp.ptr());

        Thyra::V_VpStV(
            disp_diff_rcp.ptr(),
            curr_disp,
            -1.0,
            prev_disp);

        auto &
        disp_diff = *disp_diff_rcp;

#if defined(DEBUG)
        fos << "\n*** NOX: Solution difference ***\n";
        disp_diff.describe(fos, Teuchos::VERB_EXTREME);
        fos << "\n*** NORM: " << Thyra::norm(disp_diff) << '\n';
        fos << "\n*** NOX: Solution difference ***\n";
#endif //DEBUG

        // After solve, save solution and get info to check convergence
        disp_nox_[subdomain] = curr_disp_rcp;
        norms_init(subdomain) = Thyra::norm(prev_disp);
        norms_final(subdomain) = Thyra::norm(curr_disp);
        norms_diff(subdomain) = Thyra::norm(disp_diff);
      } // Subdomain loop

      if (failed_ == true) {
        fos << "INFO: Unable to continue Schwarz iteration " << num_iter_;
        fos << "\n";
        // Break out of the Schwarz loop.
        break;
      }

      norm_init_ = minitensor::norm(norms_init);
      norm_final_ = minitensor::norm(norms_final);
      norm_diff_ = minitensor::norm(norms_diff);

      updateConvergenceCriterion();

      fos << delim << std::endl;
      fos << "Schwarz iteration         :" << num_iter_ << '\n';

      std::string const
      line(72, '-');

      fos << line << std::endl;

      fos << centered("Sub", 4);
      fos << centered("Initial norm", 24);
      fos << centered("Final norm", 24);
      fos << centered("Difference norm", 24);
      fos << std::endl;

      fos << centered("dom", 4);
      fos << centered("||X0||", 24);
      fos << centered("||Xf||", 24);
      fos << centered("||Xf-X0||", 24);
      fos << std::endl;

      fos << line << std::endl;

      for (auto m = 0; m < num_subdomains_; ++m) {
        fos << std::setw(4) << m;
        fos << std::setw(24) << norms_init(m);
        fos << std::setw(24) << norms_final(m);
        fos << std::setw(24) << norms_diff(m);
        fos << std::endl;
      }

      fos << line << std::endl;

      fos << centered("Norm", 4);
      fos << std::setw(24) << norm_init_;
      fos << std::setw(24) << norm_final_;
      fos << std::setw(24) << norm_diff_;
      fos << std::endl;

      fos << line << std::endl;

      fos << "Absolute error     :" << abs_error_ << '\n';
      fos << "Absolute tolerance :" << abs_tol_ << '\n';
      fos << "Relative error     :" << rel_error_ << '\n';
      fos << "Relative tolerance :" << rel_tol_ << '\n';
      fos << delim << std::endl;

    }  while (continueSolve() == true); // Schwarz loop

    // One of the subdomains failed to solve. Reduce step.
    if (failed_ == true) {
      failed_ = false;

      auto const
      reduced_step = std::max(min_time_step_, reduction_factor_ * time_step);

      if (reduced_step < time_step) {
        fos << "INFO: Reducing step from " << time_step << " to ";
        fos << reduced_step << '\n';
      } else {
        fos << "INFO: Cannot reduce step. Using " << reduced_step << '\n';
      }

      time_step = reduced_step;

      // Restore previous solutions
      for (auto subdomain = 0; subdomain < num_subdomains_; ++subdomain) {
        disp_nox_[subdomain] = prev_disp_nox_[subdomain];
      }

      // Jump to the beginning of the continuation loop without advancing
      // time to try to use a reduced step.
      continue;
    }

    reportFinals(fos);

    // Output converged solution if at specified interval
    bool const
    do_output = output_interval_ > 0 ?
        (stop + 1) % output_interval_ == 0 : false;

    if (do_output == true) {

      for (auto subdomain = 0; subdomain < num_subdomains_; ++subdomain) {

        Albany::AbstractSTKMeshStruct &
        ams = *stk_mesh_structs_[subdomain];

        ams.exoOutputInterval = 1;
        ams.exoOutput = true;

        Albany::AbstractDiscretization &
        abs_disc = *discs_[subdomain];

        Albany::STKDiscretization &
        stk_disc = static_cast<Albany::STKDiscretization &>(abs_disc);

        Teuchos::RCP<Tpetra_MultiVector>
        disp_mv_rcp = stk_disc.getSolutionMV();

        stk_disc.writeSolutionMV(*disp_mv_rcp, next_time);

        ams.exoOutput = false;
      }

    }

    ++stop;
    current_time += time_step;

    // Step successful. Try to increase the time step.
    auto const
    increased_step = std::min(max_time_step_, increase_factor_ * time_step);

    if (increased_step > time_step) {
      fos << "\nINFO: Increasing step from " << time_step << " to ";
      fos << increased_step << '\n';
      time_step = increased_step;
    } else {
      fos << "\nINFO: Cannot increase step. Using " << time_step << '\n';
    }


  } // Continuation loop

  return;
}

} // namespace LCM
