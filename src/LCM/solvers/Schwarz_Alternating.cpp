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

//#define DEBUG

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
  stk_mesh_structs_.resize(num_subdomains_);
  discs_.resize(num_subdomains_);
  model_evaluators_.resize(num_subdomains_);
  sub_inargs_.resize(num_subdomains_);
  sub_outargs_.resize(num_subdomains_);
  solutions_nox_.resize(num_subdomains_);
  solutions_thyra_.resize(num_subdomains_);
  solutions_dot_thyra_.resize(num_subdomains_);
  solutions_dotdot_thyra_.resize(num_subdomains_);

  bool
  have_loca{false};

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
    msg{"All subdomains must have the same solution method (LOCA or Tempus)"};

    if (subdomain == 0) { 
      have_loca = piro_params.isSublist("LOCA");
      have_tempus = piro_params.isSublist("Tempus");
      ALBANY_ASSERT(have_loca != have_tempus, "Must have either LOCA or Tempus");
      have_loca_ = have_loca;
      have_tempus_ = have_tempus;
    }
    else {
      ALBANY_ASSERT(have_loca == piro_params.isSublist("LOCA"), msg);
      ALBANY_ASSERT(have_tempus == piro_params.isSublist("Tempus"), msg);
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
    
    discs_[subdomain] = disc; 

    Albany::STKDiscretization &
    stk_disc = *static_cast<Albany::STKDiscretization *>(disc.get());

    Teuchos::RCP<Albany::AbstractSTKMeshStruct>
    ams = stk_disc.getSTKMeshStruct();

    stk_mesh_structs_[subdomain] = ams;

    model_evaluators_[subdomain] = solver_factory.returnModelT();

    solutions_nox_[subdomain] = Teuchos::null;
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

    Albany::STKDiscretization &
    stk_disc = *static_cast<Albany::STKDiscretization *>(discs_[subdomain].get());

    Teuchos::RCP<Tpetra_MultiVector> soln_mv = stk_disc.getSolutionMV();

    stk_disc.writeSolutionMV(*soln_mv, initial_time_);

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

        Piro::TempusSolver<ST,LO,GO,KokkosNode> &
        piro_tempus_solver = dynamic_cast<Piro::TempusSolver<ST,LO,GO,KokkosNode> &>(solver);

        piro_tempus_solver.setStartTime(current_time); 
        piro_tempus_solver.setFinalTime(next_time); 
        piro_tempus_solver.setInitTimeStep(time_step);

        fos << "Initial time       :" << piro_tempus_solver.getStartTime() << '\n';
        fos << "Final time         :" << piro_tempus_solver.getFinalTime() << '\n';
        fos << "Time step          :" << piro_tempus_solver.getInitTimeStep() << '\n';
        fos << delim << std::endl;

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

        Teuchos::RCP<Tempus::SolutionHistory<ST> > 
        solution_history; 
          
        Teuchos::RCP<Tempus::SolutionState<ST>> 
        current_state;

        Teuchos::RCP<Thyra::VectorBase<ST>> 
        prev_soln_rcp;
        
        Teuchos::RCP<Thyra::VectorBase<ST>> 
        prev_soln_dot_rcp;
        
        Teuchos::RCP<Thyra::VectorBase<ST>> 
        prev_soln_dotdot_rcp;

        if (is_initial_state == true) {
 
          solution_history = piro_tempus_solver.getSolutionHistory();
          current_state = solution_history->getCurrentState();
          prev_soln_rcp = Thyra::createMember(me.get_x_space()); 
          prev_soln_dot_rcp = Thyra::createMember(me.get_x_space()); 
          prev_soln_dotdot_rcp = Thyra::createMember(me.get_x_space()); 
          prev_soln_rcp->assign(*(me.getNominalValues().get_x()));
          prev_soln_dot_rcp->assign(*(me.getNominalValues().get_x_dot()));
          prev_soln_dotdot_rcp->assign(*(me.getNominalValues().get_x_dot_dot()));
 
        }
        else {
          prev_soln_rcp = solutions_thyra_[subdomain];
          prev_soln_dot_rcp = solutions_dot_thyra_[subdomain];
          prev_soln_dotdot_rcp = solutions_dotdot_thyra_[subdomain];
          //IKT, FIXME: check with Alejandro if current_time is the correct argument to use 
          //in the call below.
          piro_tempus_solver.setInitialState(current_time, prev_soln_rcp, 
                                            prev_soln_dot_rcp, prev_soln_dotdot_rcp);
        }

//#if defined(DEBUG)
        fos << "\n*** Thyra: Previous solution ***\n";
        prev_soln_rcp->describe(fos, Teuchos::VERB_EXTREME);
        fos << "\n*** NORM: " << Thyra::norm(*prev_soln_rcp) << '\n';
        fos << "\n*** Thyra: Previous solution ***\n";
//#endif //DEBUG


        solver.evalModel(in_args, out_args);  
        
        solution_history = piro_tempus_solver.getSolutionHistory();

        current_state = solution_history->getCurrentState();

        Teuchos::RCP<Thyra::VectorBase<ST>> 
        curr_soln_rcp = Thyra::createMember(me.get_x_space());
 
        curr_soln_rcp->assign(*current_state->getX());
 
        Teuchos::RCP<Thyra::VectorBase<ST>> 
        curr_soln_dot_rcp = Thyra::createMember(me.get_x_space());
 
        curr_soln_dot_rcp->assign(*current_state->getXDot()); 

        Teuchos::RCP<Thyra::VectorBase<ST>> 
        curr_soln_dotdot_rcp = Thyra::createMember(me.get_x_space());
 
        curr_soln_dotdot_rcp->assign(*current_state->getXDotDot()); 

//#if defined(DEBUG)
        fos << "\n*** Thyra: Current solution ***\n";
        curr_soln_rcp->describe(fos, Teuchos::VERB_EXTREME);
        fos << "\n*** NORM: " << Thyra::norm(*curr_soln_rcp) << '\n';
        fos << "\n*** Thyra: Current solution ***\n";
//#endif //DEBUG

        Teuchos::RCP<Thyra::VectorBase<ST>>
        soln_diff_rcp = Thyra::createMember(me.get_x_space());

        Thyra::put_scalar<ST>(0.0, soln_diff_rcp.ptr()); 

        //soln_diff = curr_soln - prev_soln 
        Thyra::V_VpStV(soln_diff_rcp.ptr(), *curr_soln_rcp, -1.0, *prev_soln_rcp);        

//#if defined(DEBUG)
        fos << "\n*** Thyra: Solution difference ***\n"; 
        soln_diff_rcp->describe(fos, Teuchos::VERB_EXTREME); 
        fos << "\n*** NORM: " << Thyra::norm(*soln_diff_rcp) << '\n';
        fos << "\n*** Thyra: Solution difference ***\n";
//#endif //DEBUG 

        //After solve, save solution and get info to check convergence
        solutions_thyra_[subdomain] = curr_soln_rcp; 
        solutions_dot_thyra_[subdomain] = curr_soln_dot_rcp; 
        solutions_dotdot_thyra_[subdomain] = curr_soln_dotdot_rcp; 
        norms_init(subdomain) = Thyra::norm(*prev_soln_rcp); 
        norms_final(subdomain) = Thyra::norm(*curr_soln_rcp); 
        norms_diff(subdomain) = Thyra::norm(*soln_diff_rcp); 

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

      if (ams.exoOutput == true) {

        Albany::STKDiscretization &
        stk_disc = *static_cast<Albany::STKDiscretization *>(discs_[subdomain].get());

        Teuchos::RCP<Tpetra_MultiVector> soln_mv = stk_disc.getSolutionMV();

        //IKT, 8/16/17: Uncomment for debug output.
        /*Teuchos::RCP<const Tpetra_Vector> soln = soln_mv->getVector(0); 
        fos << "\n*** Thyra: soln ***\n"; 
        soln->describe(fos, Teuchos::VERB_EXTREME); 
        fos << "\n*** Thyra: soln ***\n";
        */

        //IKT, 8/16/17: The following block of code would be another way of writing the solution and its 
        //time-derivatives to the Exodus file.  However, with this approach other fields living 
        //on the mesh like the Cauchy stresses would not be written to the file.
        /*Teuchos::RCP<const Tpetra_Vector> 
        soln = ConverterT::getConstTpetraVector(solutions_thyra_[subdomain]);
        Teuchos::RCP<const Tpetra_Vector> 
        soln_dot = ConverterT::getConstTpetraVector(solutions_dot_thyra_[subdomain]);
        Teuchos::RCP<const Tpetra_Vector> 
        soln_dotdot = ConverterT::getConstTpetraVector(solutions_dotdot_thyra_[subdomain]);
        stk_disc.writeSolutionT(*soln, *soln_dot, *soln_dotdot, current_time + time_step); */

        stk_disc.writeSolutionMV(*soln_mv, current_time + time_step);

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
    soln_mv_rcp = stk_disc.getSolutionMV();

    stk_disc.writeSolutionMV(*soln_mv_rcp, initial_time_);

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

        Piro::LOCASolver<ST> &
        piro_loca_solver = dynamic_cast<Piro::LOCASolver<ST> &>(solver);

        Teuchos::ParameterList &
        start_stop_params = piro_loca_solver.getStepperParams();

        start_stop_params.set(init_str, current_time);
        start_stop_params.set(start_str, current_time);
        start_stop_params.set(stop_str, next_time);

        Teuchos::ParameterList &
        time_step_params = piro_loca_solver.getStepSizeParams();

        time_step_params.set(step_str, time_step);
        time_step_params.set("Method", "Constant");

        double const
        init_time = start_stop_params.get<double>(init_str);

        double const
        start_time = start_stop_params.get<double>(start_str);

        double const
        stop_time = start_stop_params.get<double>(stop_str);

        double const
        step_size = time_step_params.get<double>(step_str);

        fos << "Initial time       :" << init_time << '\n';
        fos << "Start time         :" << start_time << '\n';
        fos << "Stop time          :" << stop_time << '\n';
        fos << "Step size          :" << step_size << '\n';
        fos << delim << std::endl;

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
            solutions_nox_[subdomain];

        NOX::Abstract::Vector const &
        prev_soln = *prev_soln_rcp;

#if defined(DEBUG)
        fos << "\n*** NOX: Previous solution ***\n";
        prev_soln.print(fos);
        fos << "\n*** NORM: " << prev_soln.norm() << '\n';
        fos << "\n*** NOX: Previous solution ***\n";
#endif //DEBUG

        nox_solver.reset(prev_soln);

        solver.evalModel(in_args, out_args);

        auto const &
        soln_group = piro_loca_solver.getSolver()->getSolutionGroup();

        Teuchos::RCP<NOX::Abstract::Vector>
        curr_soln_rcp = soln_group.getX().clone(NOX::DeepCopy);

        NOX::Abstract::Vector const &
        curr_soln = *curr_soln_rcp;

#if defined(DEBUG)
        fos << "\n*** NOX: Current solution ***\n";
        curr_soln.print(fos);
        fos << "\n*** NORM: " << curr_soln.norm() << '\n';
        fos << "\n*** NOX: Current solution ***\n";
#endif //DEBUG

        Teuchos::RCP<NOX::Abstract::Vector>
        soln_diff_rcp = curr_soln.clone(NOX::DeepCopy);

        NOX::Abstract::Vector &
        soln_diff = *(soln_diff_rcp);

        soln_diff.update(1.0, curr_soln, -1.0, prev_soln, 0.0);

#if defined(DEBUG)
        fos << "\n*** NOX: Solution difference ***\n";
        soln_diff.print(fos);
        fos << "\n*** NORM: " << soln_diff.norm() << '\n';
        fos << "\n*** NOX: Solution difference ***\n";
#endif //DEBUG

        // After solve, save solution and get info to check convergence
        solutions_nox_[subdomain] = curr_soln_rcp;
        norms_init(subdomain) = prev_soln.norm();
        norms_final(subdomain) = curr_soln.norm();
        norms_diff(subdomain) = soln_diff.norm();
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

    }  while (continueSolve() == true);

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
        soln_mv_rcp = stk_disc.getSolutionMV();

        stk_disc.writeSolutionMV(*soln_mv_rcp, next_time);

        ams.exoOutput = false;
      }

    }

    ++stop;
    current_time += time_step;
  }

  return;
}

} // namespace LCM
