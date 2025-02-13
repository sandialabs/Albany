// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include "LandIce_SequentialCoupling.hpp"

#include <algorithm>
#include <fstream>

#include "Albany_PiroObserver.hpp"
#include "Albany_STKDiscretization.hpp"
#include "Albany_SolverFactory.hpp"
#include "Albany_Utils.hpp"
#include "Piro_TempusSolver.hpp"

namespace {

//IKT 1/7/2025 TODO: check if these functions are needed.  If not, remove.

bool
fileExists(std::string const& filename)
{
  std::ifstream file(filename.c_str());
  return (file.good());
}

bool
fileExistsParallel(std::string const& filename, Teuchos::RCP<Teuchos::Comm<int> const> comm)
{
  int const   num_ranks = comm->getSize();
  int const   this_rank = comm->getRank();
  std::string full_filename{""};
  if (num_ranks > 1) {
    full_filename = filename + "." + std::to_string(num_ranks) + "." + std::to_string(this_rank);
  } else {
    full_filename = filename;
  }
  return fileExists(full_filename);
}

// In "Poisson System 2D", extract "Poisson System".
// IKT 1/14/2025: this routine is needed.
inline std::string
getName(std::string const& method)
{
  if (method.size() < 3) return method;
  return method.substr(0, method.size() - 3);
}

void
deleteParallel(std::string const& filename, Teuchos::RCP<Teuchos::Comm<int> const> comm)
{
  int const num_ranks = comm->getSize();
  int const this_rank = comm->getRank();
  if (num_ranks > 1) {
    std::string const full_filename = filename + "." + std::to_string(num_ranks) + "." + std::to_string(this_rank);
    auto const        file_removed  = remove(full_filename.c_str());
    ALBANY_ASSERT(file_removed == 0, "Could not remove file : " << full_filename);
  } else {
    auto const file_removed = remove(filename.c_str());
    ALBANY_ASSERT(file_removed == 0, "Could not remove file : " << filename);
  }
}

void
renameParallel(std::string const& old_filename, std::string const& new_filename, Teuchos::RCP<Teuchos::Comm<int> const> comm)
{
  if (old_filename == new_filename) return;
  int const num_ranks = comm->getSize();
  int const this_rank = comm->getRank();
  if (num_ranks > 1) {
    std::string const full_old_filename = old_filename + "." + std::to_string(num_ranks) + "." + std::to_string(this_rank);
    std::string const full_new_filename = new_filename + "." + std::to_string(num_ranks) + "." + std::to_string(this_rank);
    auto const        file_renamed      = rename(full_old_filename.c_str(), full_new_filename.c_str());
    ALBANY_ASSERT(file_renamed == 0, "Could not rename file : " << full_old_filename << " to " << full_new_filename);
  } else {
    auto const file_renamed = rename(old_filename.c_str(), new_filename.c_str());
    ALBANY_ASSERT(file_renamed == 0, "Could not rename file : " << old_filename << " to " << new_filename);
  }
}

void
validate_intervals(std::vector<ST> const& initial_times, std::vector<ST> const& final_times, std::vector<ST> const& time_steps)
{
  auto const initials_size = initial_times.size();
  auto const finals_size   = final_times.size();
  auto const steps_size    = time_steps.size();
  auto const all_equal     = initials_size == finals_size == steps_size;
  ALBANY_ASSERT(
      all_equal == true,
      "Event interval arrays have different sizes, "
          << "Initial Times : " << initials_size << ", Final Times : " << finals_size << ", Steps Size" << steps_size);
  auto prev_ti = initial_times[0];
  auto prev_tf = final_times[0];
  for (auto i = 0; i < steps_size; ++i) {
    auto const ti = initial_times[i];
    auto const tf = final_times[i];
    auto const dt = time_steps[i];
    if (i > 0) {
      ALBANY_ASSERT(
          ti > prev_ti,
          "Initial Times must be monotonically increasing. "
              << "At index " << i << " found initial time " << ti << " less or equal than previous " << prev_ti);
      ALBANY_ASSERT(
          tf > prev_tf,
          "Final Times must be monotonically increasing. "
              << "At index " << i << " found final time " << tf << " less or equal than previous " << prev_tf);
      ALBANY_ASSERT(ti > prev_tf, "Intervals must not overlap. At index " << i << " found initial time less or equal than previous final time " << prev_tf);
    }
    ALBANY_ASSERT(tf > ti, "At event interval index " << i << " found initial time " << ti << " greater or equal than final time " << tf);
    ALBANY_ASSERT(dt > 0.0, "At event interval index " << i << " found non-positive time step " << dt);
    ALBANY_ASSERT(dt <= tf - ti, "At event interval index " << i << " found time step " << dt << "greater than interval " << tf - ti);
    prev_ti = ti;
    prev_tf = tf;
  }
}

int
find_time_interval_index(std::vector<ST> const& initial_times, std::vector<ST> const& final_times, ST time)
{
  for (size_t i = 0; i < initial_times.size(); ++i) {
    if (time >= initial_times[i] && time <= final_times[i]) {
      return i;  // Time value is within this interval
    }
  }
  return -1;  // Time value is not within any interval
}

int
find_next_interval_index(std::vector<ST> const& initial_times, std::vector<ST> const& final_times, ST time)
{
  for (size_t i = 0; i < initial_times.size(); ++i) {
    if (time < initial_times[i]) {
      return i;
    }
  }
  return initial_times.size();
}

bool
is_within_interval(std::vector<ST> const& initial_times, std::vector<ST> const& final_times, ST time, int interval_index)
{
  if (interval_index == -1) return false;
  return initial_times[interval_index] <= time && time <= final_times[interval_index];
}

std::string
centered(std::string const& str, int width)
{
  assert(width >= 0);
  int const length  = static_cast<int>(str.size());
  int const padding = width - length;
  if (padding <= 0) return str;
  int const left  = padding / 2;
  int const right = padding - left;
  return std::string(left, ' ') + str + std::string(right, ' ');
}

void
renameExodusFile(int const file_index, std::string& filename)
{
  if (filename.find(".e") != std::string::npos) {
    std::ostringstream ss;
    ss << ".e-s." << file_index;
    filename.replace(filename.find(".e"), std::string::npos, ss.str());
  } else {
    ALBANY_ABORT("Exodus output file does not end in '.e*' - cannot rename!\n");
  }
}

}  // anonymous namespace

namespace LandIce {

SequentialCoupling::SequentialCoupling(Teuchos::RCP<Teuchos::ParameterList> const& app_params, Teuchos::RCP<Teuchos::Comm<int> const> const& comm)
    : fos_(Teuchos::VerboseObjectBase::getDefaultOStream()), comm_(comm)
{
  std::cout << "IKT in LandIce::SequentialCoupling constructor!\n"; 
  //IKT 1/7/2025 TODO: implement new guts of this function 
  alt_system_params_ = Teuchos::sublist(app_params, "Alternating System");
  // Get names of individual model input files
  model_filenames_ = alt_system_params_->get<Teuchos::Array<std::string>>("Model Input Files");

  maximum_steps_     = alt_system_params_->get<int>("Maximum Steps", 0);
  initial_time_      = alt_system_params_->get<ST>("Initial Time", 0.0);
  final_time_        = alt_system_params_->get<ST>("Final Time", 0.0);
  initial_time_step_ = alt_system_params_->get<ST>("Initial Time Step", 1.0);

  auto const dt  = initial_time_step_;

  min_time_step_    = alt_system_params_->get<ST>("Minimum Time Step", dt);
  max_time_step_    = alt_system_params_->get<ST>("Maximum Time Step", dt);
  reduction_factor_ = alt_system_params_->get<ST>("Reduction Factor", 1.0);
  increase_factor_  = alt_system_params_->get<ST>("Amplification Factor", 1.0);
  output_interval_  = alt_system_params_->get<int>("Exodus Write Interval", 1);

  std::cout << "IKT dt = " << dt << "\n"; 
  // Firewalls
  ALBANY_ASSERT(maximum_steps_ >= 1, "");
  ALBANY_ASSERT(final_time_ >= initial_time_, "");
  ALBANY_ASSERT(initial_time_step_ > 0.0, "");
  ALBANY_ASSERT(max_time_step_ > 0.0, "");
  ALBANY_ASSERT(min_time_step_ > 0.0, "");
  ALBANY_ASSERT(max_time_step_ >= min_time_step_, "");
  ALBANY_ASSERT(reduction_factor_ <= 1.0, "");
  ALBANY_ASSERT(reduction_factor_ > 0.0, "");
  ALBANY_ASSERT(increase_factor_ >= 1.0, "");
  ALBANY_ASSERT(output_interval_ >= 1, "");

  // number of models
  num_subdomains_ = model_filenames_.size();

  // throw error if number of model filenames provided is != 2
  ALBANY_ASSERT(num_subdomains_ == 2, "SequentialCoupling solver requires 2 models!");

  // Arrays to cache useful info for each subdomain for later use
  apps_.resize(num_subdomains_);
  solvers_.resize(num_subdomains_);
  solver_factories_.resize(num_subdomains_);
  stk_mesh_structs_.resize(num_subdomains_);
  discs_.resize(num_subdomains_);
  model_evaluators_.resize(num_subdomains_);
  sub_inargs_.resize(num_subdomains_);
  sub_outargs_.resize(num_subdomains_);
  curr_x_.resize(num_subdomains_);
  prev_step_x_.resize(num_subdomains_);
 
  // IKT NOTE: the "dot" arrays are
  // not relevant for the steady Poisson problem,
  // but they are constructed anyway.
  ics_x_.resize(num_subdomains_);
  ics_xdot_.resize(num_subdomains_);
  prev_x_.resize(num_subdomains_);
  prev_xdot_.resize(num_subdomains_);
  this_x_.resize(num_subdomains_);
  this_xdot_.resize(num_subdomains_);
  do_outputs_.resize(num_subdomains_);
  do_outputs_init_.resize(num_subdomains_);
  prob_types_.resize(num_subdomains_);

  // Create solver factories once at the beginning
  for (auto subdomain = 0; subdomain < num_subdomains_; ++subdomain) {
    solver_factories_[subdomain]   = Teuchos::rcp(new Albany::SolverFactory(model_filenames_[subdomain], comm_));
    //Teuchos::ParameterList& params = solver_factories_[subdomain]->getParameters();
    Teuchos::RCP<Teuchos::ParameterList> params = solver_factories_[subdomain]->getParameters();

    Teuchos::ParameterList& problem_params = params->sublist("Problem", true);
    //IKT 1/14/2025: I believe we won't need an analog of the following here, but 
    //keeping commented out just in case.
    //problem_params.set<bool>("LandIce Sequential Solver", true);

    std::string const problem_name = getName(problem_params.get<std::string>("Name"));
    if (problem_name == "Poisson System") {
      prob_types_[subdomain] = POISSON;
    } else if (problem_name == "AdvDiff System") {
      prob_types_[subdomain] = ADVDIFF;
    } else {
      //IKT 1/14/2025 TODO: will need to change this when we switch over to the true landice couplings
      ALBANY_ABORT("LandIce Sequential solver only supports coupling of 'Poisson System' and 'Adv Diff System' problems!");
    }

    auto const problem_type = prob_types_[subdomain];

    // Error checks - only needs to be done once at the beginning
    bool const have_piro = params->isSublist("Piro");
    ALBANY_ASSERT(have_piro == true, "Error! Piro sublist not found.\n");

    Teuchos::ParameterList& piro_params = params->sublist("Piro");

    if (problem_type == ADVDIFF) { //This will be thickness equation for LandIce, since it is dynamic 
      auto const is_dynamic = piro_params.isSublist("Tempus");
      ALBANY_ASSERT(is_dynamic == true, "LandIce Sequential Coupling requires Tempus for advection-diffusion solve.");

      Teuchos::ParameterList& tempus_params = piro_params.sublist("Tempus");
      tempus_params.set("Abort on Failure", false);

      Teuchos::ParameterList& time_step_control_params =
          piro_params.sublist("Tempus").sublist("Tempus Integrator").sublist("Time Step Control").sublist("Time Step Control Strategy");

      std::string const integrator_step_type = time_step_control_params.get("Strategy Type", "Constant");
      std::string const msg{
          "Non-constant time-stepping through Tempus not supported "
          "with LandIce Sequential Coupling; \n"
          "In this case, variable time-stepping is "
          "handled within the coupling loop.\n"
          "Please rerun with 'Strategy Type: "
          "Constant' in 'Time Step Control Strategy' sublist.\n"};
      ALBANY_ASSERT(integrator_step_type == "Constant", msg);
    }
    else if (problem_type == POISSON) { //This will be steady velocity solve for LandIce 
      std::string const sol_method    = problem_params.get<std::string>("Solution Method"); 
      bool is_steady = false; 
      if (sol_method == "Steady") {
        is_steady = true; 
      }
      auto const is_loca              = piro_params.isSublist("LOCA");
      auto const is_nox               = piro_params.isSublist("NOX");
      if (!is_steady && !is_loca && !is_nox) {  
        ALBANY_ABORT("LandIce Sequential Coupling requires LOCA or NOX steady solver for Poisson solve.");
      }
    } else {
      //IKT 2/11/2025: will need to change the following for LandIce 
      ALBANY_ABORT("LandIce Sequential Coupling only supports coupling of Poisson and AdvDiff problems.");
    }

    //IKT 2/11/2025: I don't think we actually can't have parameters and responses, so the following
    //maybe should be deleted.  I am keeping it as left-over from ACE.

    // Parameters
    int const        num_params = problem_params.sublist("Parameters").get<int>("Number Of Parameters");
    ALBANY_ASSERT(num_params == 0, "Parameters not supported.");

    // Responses
    int const     num_responses = problem_params.sublist("Response Functions").get<int>("Number Of Responses");
    ALBANY_ASSERT(num_responses == 0, "Responses not supported.");

  } //End subdomain loop 

  // Check that the first problem type is Poisson and the second problem type is AdvDiff,
  // as the current version of the coupling algorithm requires this ordering.
  // If ordering is wrong, through error.
  PROB_TYPE prob_type = prob_types_[0];
  ALBANY_ASSERT(prob_type == POISSON, "The first problem type needs to be 'Poisson System 2D'!");
  prob_type = prob_types_[1];
  ALBANY_ASSERT(prob_type == ADVDIFF, "The second problem type needs to be 'AdvDiff System 2D'!");

  return;
}

SequentialCoupling::~SequentialCoupling() { return; }

Teuchos::RCP<Thyra_VectorSpace const>
SequentialCoupling::get_x_space() const
{
  return Teuchos::null;
}

Teuchos::RCP<Thyra_VectorSpace const>
SequentialCoupling::get_f_space() const
{
  return Teuchos::null;
}

Teuchos::RCP<Thyra_VectorSpace const>
SequentialCoupling::get_p_space(int) const
{
  return Teuchos::null;
}

Teuchos::RCP<Thyra_VectorSpace const>
SequentialCoupling::get_g_space(int) const
{
  return Teuchos::null;
}

Teuchos::RCP<const Teuchos::Array<std::string>>
SequentialCoupling::get_p_names(int) const
{
  return Teuchos::null;
}

Teuchos::ArrayView<std::string const>
SequentialCoupling::get_g_names(int) const
{
  ALBANY_ABORT("not implemented");
  return Teuchos::ArrayView<std::string const>(Teuchos::null);
}

Thyra_ModelEvaluator::InArgs<ST>
SequentialCoupling::getNominalValues() const
{
  return this->createInArgsImpl();
}

Thyra_ModelEvaluator::InArgs<ST>
SequentialCoupling::getLowerBounds() const
{
  return Thyra_ModelEvaluator::InArgs<ST>();  // Default value
}

Thyra_ModelEvaluator::InArgs<ST>
SequentialCoupling::getUpperBounds() const
{
  return Thyra_ModelEvaluator::InArgs<ST>();  // Default value
}

Teuchos::RCP<Thyra::LinearOpBase<ST>>
SequentialCoupling::create_W_op() const
{
  return Teuchos::null;
}

Teuchos::RCP<Thyra::PreconditionerBase<ST>>
SequentialCoupling::create_W_prec() const
{
  return Teuchos::null;
}

Teuchos::RCP<const Thyra::LinearOpWithSolveFactoryBase<ST>>
SequentialCoupling::get_W_factory() const
{
  return Teuchos::null;
}

Thyra_ModelEvaluator::InArgs<ST>
SequentialCoupling::createInArgs() const
{
  return this->createInArgsImpl();
}

Teuchos::ArrayRCP<Teuchos::RCP<Albany::Application>>
SequentialCoupling::getApps() const
{
  return apps_;
}


// Create operator form of dg/dx for distributed responses
Teuchos::RCP<Thyra::LinearOpBase<ST>>
SequentialCoupling::create_DgDx_op_impl(int /* j */) const
{
  return Teuchos::null;
}

// Create operator form of dg/dx_dot for distributed responses
Teuchos::RCP<Thyra::LinearOpBase<ST>>
SequentialCoupling::create_DgDx_dot_op_impl(int /* j */) const
{
  return Teuchos::null;
}

// Create InArgs
Thyra_InArgs
SequentialCoupling::createInArgsImpl() const
{
  Thyra::ModelEvaluatorBase::InArgsSetup<ST> ias;

  ias.setModelEvalDescription(this->description());

  ias.setSupports(Thyra_ModelEvaluator::IN_ARG_x, true);
  ias.setSupports(Thyra_ModelEvaluator::IN_ARG_x_dot, true);
  ias.setSupports(Thyra_ModelEvaluator::IN_ARG_x_dot_dot, true);
  ias.setSupports(Thyra_ModelEvaluator::IN_ARG_t, true);
  ias.setSupports(Thyra_ModelEvaluator::IN_ARG_alpha, true);
  ias.setSupports(Thyra_ModelEvaluator::IN_ARG_beta, true);
  ias.setSupports(Thyra_ModelEvaluator::IN_ARG_W_x_dot_dot_coeff, true);

  return static_cast<Thyra_InArgs>(ias);
}

// Create OutArgs
Thyra_OutArgs
SequentialCoupling::createOutArgsImpl() const
{
  Thyra::ModelEvaluatorBase::OutArgsSetup<ST> oas;

  oas.setModelEvalDescription(this->description());

  oas.setSupports(Thyra_ModelEvaluator::OUT_ARG_f, true);
  oas.setSupports(Thyra_ModelEvaluator::OUT_ARG_W_op, true);
  oas.setSupports(Thyra_ModelEvaluator::OUT_ARG_W_prec, false);

  oas.set_W_properties(Thyra_ModelEvaluator::DerivativeProperties(Thyra_ModelEvaluator::DERIV_LINEARITY_UNKNOWN, Thyra_ModelEvaluator::DERIV_RANK_FULL, true));

  return static_cast<Thyra_OutArgs>(oas);
}

// Evaluate model on InArgs
void
SequentialCoupling::evalModelImpl(Thyra_ModelEvaluator::InArgs<ST> const&, Thyra_ModelEvaluator::OutArgs<ST> const&) const
{
  std::cout << "IKT entering evalModelImpl!\n"; 
  SequentialCouplingLoop();
}


void
SequentialCoupling::createPoissonSolverAppDiscME(int const file_index, double const current_time) const
{
  std::cout << "IKT in createPoissonSolverAppDiscME!\n"; 
  auto const              subdomain      = 0;
  Teuchos::RCP<Teuchos::ParameterList> params         = solver_factories_[subdomain]->getParameters();
  Teuchos::ParameterList& problem_params = params->sublist("Problem", true);
  Teuchos::ParameterList& disc_params    = params->sublist("Discretization", true);
  //IKT 2/13/2025: we are not going to want to do solution transfer through output files eventually
  //but making it this way first to mimic what LCM is doing.
  std::string             filename       = disc_params.get<std::string>("Exodus Output File Name");
  renameExodusFile(file_index, filename);
  *fos_ << "Renaming output file to - " << filename << '\n';
  disc_params.set<std::string>("Exodus Output File Name", filename);
  disc_params.set<std::string>("Exodus Solution Name", "temperature");
  disc_params.set<std::string>("Exodus SolutionDot Name", "temperature_dot");
  disc_params.set<bool>("Output DTK Field to Exodus", false);
  if (!disc_params.isParameter("Disable Exodus Output Initial Time")) {
    disc_params.set<bool>("Disable Exodus Output Initial Time", true);
  }
  int const thermal_exo_write_interval = disc_params.get<int>("Exodus Write Interval", 1);
  ALBANY_ASSERT(
      thermal_exo_write_interval == 1,
      "'Exodus Write Interval' for Thermal Problem must be 1!  This parameter is controlled by variables in coupled "
      "input file.");
  if (file_index > 0) {
    // Change input Exodus file to previous mechanical Exodus output file, for restarts.
    disc_params.set<std::string>("Exodus Input File Name", prev_mechanical_exo_outfile_name_);
    // Set restart index based on 'disable exodus output initial time' variable
    // provided in input file
    const bool disable_exo_out_init_time = disc_params.get<bool>("Disable Exodus Output Initial Time");
    if (disable_exo_out_init_time == true) {
      disc_params.set<int>("Restart Index", 1);
    } else {
      disc_params.set<int>("Restart Index", 2);
    }
    // Remove Initial Condition sublist
    problem_params.remove("Initial Condition", true);
  }
  //IKT 2/13/2025: may need to hook in the following to problem...
  problem_params.set<double>("LandIce Sequential Coupling Problem Current Time", current_time);

  // Create app (null initial guess)
  Teuchos::RCP<Albany::Application> app = solver_factories_[subdomain]->createApplication(comm_);
  //Forward model model evaluator
  Teuchos::RCP<Albany::ModelEvaluator> model = solver_factories_[subdomain]->createModel(app, false);
  //Adjoint model model evaluator - assuming this is null for now 
  Teuchos::RCP<Albany::ModelEvaluator> adjoint_model = Teuchos::null; 
  // Create solver   
  Teuchos::RCP<Thyra::ResponseOnlyModelEvaluatorBase<ST>> solver = solver_factories_[subdomain]->createSolver(model, adjoint_model, true); 

  solvers_[subdomain] = solver;
  apps_[subdomain]    = app;
  auto num_dims       = app->getSpatialDimension();
  //IKT 2/13/2025: probably won't need something like the following throw... 
  /*if (num_dims != 3) {
    ALBANY_ABORT("LandIce Sequential Coupling solver only works in 3D!  Poisson problem has " << num_dims << " dimensions.");
  }*/
  // Get STK mesh structs to control Exodus output interval
  Teuchos::RCP<Albany::AbstractDiscretization> disc = app->getDiscretization();
  discs_[subdomain]                                 = disc;

  Albany::STKDiscretization& stk_disc = *static_cast<Albany::STKDiscretization*>(disc.get());
  if (file_index == 0) {
    //IKT TO DO: implement the following routine in the discretization! 
    stk_disc.outputExodusSolutionInitialTime(true);
  }

  
  auto  abs_stk_mesh_struct_rcp  = stk_disc.getSTKMeshStruct();
  auto& abs_stk_mesh_struct      = *abs_stk_mesh_struct_rcp;
  do_outputs_[subdomain]         = abs_stk_mesh_struct.exoOutput;
  do_outputs_init_[subdomain]    = abs_stk_mesh_struct.exoOutput;
  stk_mesh_structs_[subdomain]   = abs_stk_mesh_struct_rcp;
  model_evaluators_[subdomain]   = solver_factories_[subdomain]->returnModel();
  curr_x_[subdomain]             = Teuchos::null;
  prev_thermal_exo_outfile_name_ = filename;
  // Delete previously-written Exodus files to not have inundation of output files
  if (file_index > 0 && ((file_index - 1) % output_interval_) != 0) {
    deleteParallel(prev_mechanical_exo_outfile_name_, comm_);
  }
}

void
SequentialCoupling::createAdvDiffSolverAppDiscME(int const file_index, double const current_time, double const next_time, double const time_step) const
{
  std::cout << "IKT in createAdvDiffSolverAppDiscME!\n"; 
  //IKT 1/7/2025 TODO: fill in this function
  /*auto const              subdomain      = 1;
  Teuchos::ParameterList& params         = solver_factories_[subdomain]->getParameters();
  Teuchos::ParameterList& problem_params = params.sublist("Problem", true);
  Teuchos::ParameterList& disc_params    = params.sublist("Discretization", true);
  // Check if using wave pressure NBC, and if so, inject zmin_ value into that PL
  if (problem_params.isSublist("Neumann BCs")) {
    Teuchos::ParameterList&               nbc_params = problem_params.sublist("Neumann BCs");
    Teuchos::ParameterList::ConstIterator it;
    for (it = nbc_params.begin(); it != nbc_params.end(); it++) {
      const std::string nbc_sublist = nbc_params.name(it);
      const std::string wp          = "wave_pressure";
      std::size_t       found       = nbc_sublist.find(wp);
      if (found != std::string::npos) {
        Teuchos::ParameterList& pnbc_sublist = nbc_params.sublist(nbc_sublist);
        pnbc_sublist.set<double>("Min z-Value", zmin_);
      }
    }
  }
  std::string filename = disc_params.get<std::string>("Exodus Output File Name");
  renameExodusFile(file_index + init_file_index_, filename);
  *fos_ << "Renaming output file to - " << filename << '\n';
  disc_params.set<std::string>("Exodus Output File Name", filename);
  disc_params.set<std::string>("Exodus Solution Name", "disp");
  disc_params.set<std::string>("Exodus SolutionDot Name", "disp_dot");
  disc_params.set<std::string>("Exodus SolutionDotDot Name", "disp_dotdot");
  disc_params.set<bool>("Output DTK Field to Exodus", false);
  int const mechanics_exo_write_interval = disc_params.get<int>("Exodus Write Interval", 1);
  ALBANY_ASSERT(
      mechanics_exo_write_interval == 1,
      "'Exodus Write Interval' for Mechanics Problem must be 1!  This parameter is controlled by variables in coupled "
      "input file.");

  // After the initial run, we will do restarts from the previously written Exodus output file.
  // Change input Exodus file to previous thermal Exodus output file, for restarts.
  disc_params.set<std::string>("Exodus Input File Name", prev_thermal_exo_outfile_name_);
  if (!disc_params.isParameter("Disable Exodus Output Initial Time")) {
    disc_params.set<bool>("Disable Exodus Output Initial Time", true);
  }
  // Set restart index based on where we are in the simulation
  if (file_index == 0) {  // Initially, restart index = 2, since initial file will have 2 snapshots
                          // and the second one is the one we want to restart from
    disc_params.set<int>("Restart Index", 2);
  } else {
    // Set restart index based on 'disable exodus output initial time' variable
    // after initial time step
    const bool disable_exo_out_init_time = disc_params.get<bool>("Disable Exodus Output Initial Time");
    if (disable_exo_out_init_time == true) {
      disc_params.set<int>("Restart Index", 1);
    } else {
      disc_params.set<int>("Restart Index", 2);
    }
  }
  // Remove Initial Condition sublist
  problem_params.remove("Initial Condition", true);
  // Set flag to tell code that we have an ACE Sequential Thermomechanical Problem
  problem_params.set("ACE Sequential Thermomechanical", true, "ACE Sequential Thermomechanical Problem");

  if (mechanical_solver_ == MechanicalSolver::TrapezoidRule) {
    Teuchos::ParameterList& piro_params = params.sublist("Piro", true);
    Teuchos::ParameterList& tr_params   = piro_params.sublist("Trapezoid Rule", true);
    tr_params.set<int>("Num Time Steps", 1);
    tr_params.set<double>("Initial Time", current_time);
    tr_params.set<double>("Final Time", next_time);
    tr_params.remove("Write Only Converged Solution", false);
    tr_params.remove("Sensitivity Method", false);
    tr_params.remove("Jacobian Operator", false);
    tr_params.remove("Exit on Failed NOX Solve", false);
    tr_params.remove("On Failure Solve With Zero Initial Guess", false);
  }

  Teuchos::RCP<Albany::Application>                       app{Teuchos::null};
  Teuchos::RCP<Thyra::ResponseOnlyModelEvaluatorBase<ST>> solver = solver_factories_[subdomain]->createAndGetAlbanyApp(app, comm_, comm_);

  solvers_[subdomain] = solver;
  apps_[subdomain]    = app;
  auto num_dims       = app->getSpatialDimension();
  if (num_dims != 3) {
    ALBANY_ABORT("ACE Thermo-Mechanical solver only works in 3D!  Mechanics problem has " << num_dims << " dimensions.");
  }

  // Get STK mesh structs to control Exodus output interval
  Teuchos::RCP<Albany::AbstractDiscretization> disc = app->getDiscretization();
  discs_[subdomain]                                 = disc;

  Albany::STKDiscretization& stk_disc = *static_cast<Albany::STKDiscretization*>(disc.get());
  if (file_index == 0) {
    stk_disc.outputExodusSolutionInitialTime(true);
  }

  auto  abs_stk_mesh_struct_rcp     = stk_disc.getSTKMeshStruct();
  auto& abs_stk_mesh_struct         = *abs_stk_mesh_struct_rcp;
  do_outputs_[subdomain]            = abs_stk_mesh_struct.exoOutput;
  do_outputs_init_[subdomain]       = abs_stk_mesh_struct.exoOutput;
  stk_mesh_structs_[subdomain]      = abs_stk_mesh_struct_rcp;
  model_evaluators_[subdomain]      = solver_factories_[subdomain]->returnModel();
  curr_x_[subdomain]                = Teuchos::null;
  prev_mechanical_exo_outfile_name_ = filename;
  // Delete previously-written Exodus files to not have inundation of output files
  if ((file_index % output_interval_) != 0) {
    deleteParallel(prev_thermal_exo_outfile_name_, comm_);
  }
  */
}

bool
SequentialCoupling::continueSolve() const
{
  ++num_iter_;
  // IKT 6/5/2020: right now, we want to do just 1 coupling iteration.
  // Therefore return false if we've hit num_iter_ = 1;
  // Also set converged_ to true, which is equally irrelevant unless doing
  // Schwarz-like coupling
  converged_ = true;
  if (num_iter_ > 0) return false;
  else return true;
}

// Sequential coupling loop, dynamic
void
SequentialCoupling::SequentialCouplingLoop() const
{
  //IKT 1/7/2025 TODO: implement this function!
  std::cout << "IKT entering SequentialCouplingLoop()!\n"; 
  std::string const delim(72, '=');

  *fos_ << std::scientific << std::setprecision(17);

  // If initial time is within an interval, reset to its beginning
  ST   time_step{initial_time_step_};
  //IKT 2/12/2025: check with Alejandro what the event_initial_times_ and event_final_times_ were for
  //I don't recall ever using them in LCM.
  /*auto interval_index = find_time_interval_index(event_initial_times_, event_final_times_, initial_time_);
  if (interval_index != -1) {
    initial_time_ = event_initial_times_[interval_index];
    time_step     = event_time_steps_[interval_index];
  }*/
  int stop{0};
  ST  current_time{initial_time_};

  // Time-stepping loop
  while (stop < maximum_steps_ && current_time < final_time_) {
    //IKT 2/15/2025: commenting out the following for now until figure out how it's supposed to work.
    /*if (interval_index != -1) {
      *fos_ << delim << std::endl;
      *fos_ << "Subclycling within an event interval.\n";
    }*/

    *fos_ << delim << std::endl;
    *fos_ << "Time stop          :" << stop << '\n';
    *fos_ << "Time               :" << current_time << '\n';
    *fos_ << "Time step          :" << time_step << '\n';
    *fos_ << delim << std::endl;

    ST const next_time{current_time + time_step};
    num_iter_ = 0;

    // Coupling loop
    do {
      bool const is_initial_state = stop == 0 && num_iter_ == 0;
      for (auto subdomain = 0; subdomain < num_subdomains_; ++subdomain) {
        // Create new solvers, apps, discs and model evaluators
        auto const prob_type = prob_types_[subdomain];
        if (prob_type == POISSON) {
          createPoissonSolverAppDiscME(stop, current_time);
        }
        if (prob_type == ADVDIFF && failed_ == false) {
          createAdvDiffSolverAppDiscME(stop, current_time, next_time, time_step);
        }

	/*
        // Before the coupling loop, get internal states, and figure out whether
        // output needs to be done or not.
        if (num_iter_ == 0) {
          auto& app       = *apps_[subdomain];
          auto& state_mgr = app.getStateMgr();
          fromTo(state_mgr.getStateArrays(), internal_states_[subdomain]);
          do_outputs_[subdomain] = true;  // We always want output in the initial step
        } else {
          if (do_outputs_init_[subdomain] == true) {
            do_outputs_[subdomain] = output_interval_ > 0 ? (stop + 1) % output_interval_ == 0 : false;
          }
        }
        *fos_ << delim << std::endl;
        *fos_ << "Subdomain          :" << subdomain << '\n';
        if (prob_type == MECHANICAL) {
          *fos_ << "Problem            :Mechanical\n";
          AdvanceMechanicalDynamics(subdomain, is_initial_state, current_time, next_time, time_step);
          if (failed_ == false) {
            doDynamicInitialOutput(next_time, subdomain);
            renamePrevWrittenExoFiles(subdomain, stop);
          }
        }
        if (prob_type == THERMAL) {
          *fos_ << "Problem            :Thermal\n";
          AdvanceThermalDynamics(subdomain, is_initial_state, current_time, next_time, time_step);
          if (failed_ == false) {
            doDynamicInitialOutput(next_time, subdomain);
            renamePrevWrittenExoFiles(subdomain, stop);
          }
        }
        if (failed_ == true) {
          // Break out of the subdomain loop
          break;
        }*/
      }  // Subdomains loop

      exit(1); 
      if (failed_ == true) {
        // Break out of the coupling loop.
        break;
      }
    } while (continueSolve() == true);
  } //Time-stepping loop.  IKT 2/12/2025: dummy for now!  
/*
    // One of the subdomains failed to solve. Reduce step.
    if (failed_ == true) {
      auto const reduced_step = reduction_factor_ * time_step;

      if (time_step <= min_time_step_) {
        *fos_ << "ERROR: Cannot reduce step. Stopping execution.\n";
        *fos_ << "INFO: Requested step    :" << reduced_step << '\n';
        *fos_ << "INFO: Minimum time step :" << min_time_step_ << '\n';
        return;
      }

      if (reduced_step > min_time_step_) {
        *fos_ << "INFO: Reducing step from " << time_step << " to ";
        *fos_ << reduced_step << '\n';
        time_step = reduced_step;
      } else {
        *fos_ << "INFO: Reducing step from " << time_step << " to ";
        *fos_ << min_time_step_ << '\n';
        time_step = min_time_step_;
      }

      // Jump to the beginning of the time-step loop without advancing
      // time to try to use a reduced step.
      continue;
    }

    // Update IC vecs
    for (auto subdomain = 0; subdomain < num_subdomains_; ++subdomain) {
      setICVecs(next_time, subdomain);
    }

    if (interval_index != -1) {
      time_step = std::min(time_step, event_final_times_[interval_index] - current_time);
      time_step = std::max(time_step, min_time_step_);
    } else {
      auto const next_interval_index = find_time_interval_index(event_initial_times_, event_final_times_, current_time);
      if (next_interval_index < event_initial_times_.size()) {
        time_step = std::min(time_step, event_initial_times_[next_interval_index] - current_time);
        time_step = std::max(time_step, min_time_step_);
      }
    }

    ++stop;
    current_time += time_step;
    interval_index         = find_time_interval_index(event_initial_times_, event_final_times_, current_time);
    auto const in_interval = is_within_interval(event_initial_times_, event_final_times_, current_time, interval_index);

    // Step successful. Try to increase the time step.
    auto const increased_step = std::min(max_time_step_, increase_factor_ * time_step);

    if (increased_step > time_step && in_interval == false) {
      *fos_ << "\nINFO: Increasing step from " << time_step << " to ";
      *fos_ << increased_step << '\n';
      time_step = increased_step;
    } else {
      *fos_ << "\nINFO: Cannot increase step. Using " << time_step << '\n';
    }

  }  // Time-step loop

  // Rename final Exodus output file
  for (auto subdomain = 0; subdomain < num_subdomains_; ++subdomain) {
    renamePrevWrittenExoFiles(subdomain, stop);
  }
  return;
  */
}

void
SequentialCoupling::AdvancePoisson(
    int const    subdomain,
    bool const   is_initial_state,
    double const current_time,
    double const next_time,
    double const time_step) const
{
  //IKT 1/7/2025 TODO: fill in this function.
  /* failed_ = false;
  // Solve for each subdomain
  Thyra::ResponseOnlyModelEvaluatorBase<ST>& solver = *(solvers_[subdomain]);

  Piro::TempusSolver<ST>& piro_tempus_solver = dynamic_cast<Piro::TempusSolver<ST>&>(solver);

  piro_tempus_solver.setStartTime(current_time);
  piro_tempus_solver.setFinalTime(next_time);
  piro_tempus_solver.setInitTimeStep(time_step);

  std::string const delim(72, '=');
  *fos_ << "Initial time       :" << current_time << '\n';
  *fos_ << "Final time         :" << next_time << '\n';
  *fos_ << "Time step          :" << time_step << '\n';
  *fos_ << delim << std::endl;

  Thyra_ModelEvaluator::InArgs<ST>  in_args  = solver.createInArgs();
  Thyra_ModelEvaluator::OutArgs<ST> out_args = solver.createOutArgs();

  auto& me = dynamic_cast<Albany::ModelEvaluator&>(*model_evaluators_[subdomain]);

  // Restore internal states
  auto& app       = *apps_[subdomain];
  auto& state_mgr = app.getStateMgr();
  fromTo(internal_states_[subdomain], state_mgr.getStateArrays());

  Teuchos::RCP<Tempus::SolutionHistory<ST>> solution_history;
  Teuchos::RCP<Tempus::SolutionState<ST>>   current_state;

  solver.evalModel(in_args, out_args);

  // Allocate current solution vectors
  this_x_[subdomain]    = Thyra::createMember(me.get_x_space());
  this_xdot_[subdomain] = Thyra::createMember(me.get_x_space());

  // Check whether solver did OK.
  auto const status = piro_tempus_solver.getTempusIntegratorStatus();

  if (status == Tempus::Status::FAILED) {
    *fos_ << "\nINFO: Unable to solve Thermal problem for subdomain " << subdomain << '\n';
    failed_ = true;
    return;
  }

  // If solver is OK, extract solution

  solution_history = piro_tempus_solver.getSolutionHistory();
  current_state    = solution_history->getCurrentState();

  Thyra::copy(*current_state->getX(), this_x_[subdomain].ptr());
  Thyra::copy(*current_state->getXDot(), this_xdot_[subdomain].ptr());

  failed_ = false;
  */
}

void
SequentialCoupling::AdvanceAdvDiff(
    int const    subdomain,
    bool const   is_initial_state,
    double const current_time,
    double const next_time,
    double const time_step) const
{
  //IKT 1/7/2025 TODO: fill in this function
  /*failed_ = false;
  // Solve for each subdomain
  Thyra::ResponseOnlyModelEvaluatorBase<ST>& solver = *(solvers_[subdomain]);

  // Restore internal states
  auto& app       = *apps_[subdomain];
  auto& state_mgr = app.getStateMgr();
  fromTo(internal_states_[subdomain], state_mgr.getStateArrays());

  if (mechanical_solver_ == MechanicalSolver::Tempus) {
    auto& piro_tempus_solver = dynamic_cast<Piro::TempusSolver<ST>&>(solver);
    piro_tempus_solver.setStartTime(current_time);
    piro_tempus_solver.setFinalTime(next_time);
    piro_tempus_solver.setInitTimeStep(time_step);

    std::string const delim(72, '=');
    *fos_ << "Initial time       :" << current_time << '\n';
    *fos_ << "Final time         :" << next_time << '\n';
    *fos_ << "Time step          :" << time_step << '\n';
    *fos_ << delim << std::endl;

    Thyra_ModelEvaluator::InArgs<ST>  in_args  = solver.createInArgs();
    Thyra_ModelEvaluator::OutArgs<ST> out_args = solver.createOutArgs();

    auto& me = dynamic_cast<Albany::ModelEvaluator&>(*model_evaluators_[subdomain]);

    Teuchos::RCP<Tempus::SolutionHistory<ST>> solution_history;
    Teuchos::RCP<Tempus::SolutionState<ST>>   current_state;

    solver.evalModel(in_args, out_args);

    // Allocate current solution vectors
    this_x_[subdomain]       = Thyra::createMember(me.get_x_space());
    this_xdot_[subdomain]    = Thyra::createMember(me.get_x_space());
    this_xdotdot_[subdomain] = Thyra::createMember(me.get_x_space());

    // Check whether solver did OK.
    auto const status = piro_tempus_solver.getTempusIntegratorStatus();

    if (status == Tempus::Status::FAILED) {
      *fos_ << "\nINFO: Unable to solve Mechanical problem for subdomain " << subdomain << '\n';
      failed_ = true;
      return;
    }
    // If solver is OK, extract solution

    solution_history = piro_tempus_solver.getSolutionHistory();
    current_state    = solution_history->getCurrentState();

    Thyra::copy(*current_state->getX(), this_x_[subdomain].ptr());
    Thyra::copy(*current_state->getXDot(), this_xdot_[subdomain].ptr());
    Thyra::copy(*current_state->getXDotDot(), this_xdotdot_[subdomain].ptr());

  } else if (mechanical_solver_ == MechanicalSolver::TrapezoidRule) {
    auto&             piro_tr_solver = dynamic_cast<Piro::TrapezoidRuleSolver<ST>&>(solver);
    std::string const delim(72, '=');
    *fos_ << "Initial time       :" << current_time << '\n';
    *fos_ << "Final time         :" << next_time << '\n';
    *fos_ << "Time step          :" << time_step << '\n';
    *fos_ << delim << std::endl;
    // Disable initial acceleration solve unless in initial time-step.
    // This should speed up code by ~2x.
    if (current_time != initial_time_) {
      piro_tr_solver.disableCalcInitAccel();
    }

    Thyra_ModelEvaluator::InArgs<ST>  in_args  = solver.createInArgs();
    Thyra_ModelEvaluator::OutArgs<ST> out_args = solver.createOutArgs();

    auto& me = dynamic_cast<Albany::ModelEvaluator&>(*model_evaluators_[subdomain]);

    auto const& nv     = me.getNominalValues();
    auto        x_init = nv.get_x();
    auto        v_init = nv.get_x_dot();
    auto        a_init = me.get_x_dotdot();
    auto&       app    = *apps_[subdomain];

    // Make sure there is no leftover adapted mesh from before.
    std::string const tmp_adapt_filename{"ace_adapt_temporary.e"};
    if (fileExistsParallel(tmp_adapt_filename, comm_) == true) {
      deleteParallel(tmp_adapt_filename, comm_);
    }

    solver.evalModel(in_args, out_args);

    // Check whether there was adaptation by testing whether an adapted mesh exists,
    // and if so, rename it.
    if (fileExistsParallel(tmp_adapt_filename, comm_) == true) {
      renameParallel(tmp_adapt_filename, prev_mechanical_exo_outfile_name_, comm_);
    }

    // Check whether solver did OK.
    auto&      tr_nox_solver              = *(piro_tr_solver.getNOXSolver());
    auto&      thyra_nox_nonlinear_solver = *(tr_nox_solver.getSolver());
    auto&      const_nox_generic_solver   = *(thyra_nox_nonlinear_solver.getNOXSolver());
    auto&      nox_generic_solver         = const_cast<NOX::Solver::Generic&>(const_nox_generic_solver);
    auto const status                     = nox_generic_solver.getStatus();

    // Hack: fix status test parameter list. For some reason a dummy test gets added.
    // Extract the correct list and set it as the status test list.
    {
      auto const app_params_rcp = app.getAppPL();
      auto&      piro_params    = app_params_rcp->sublist("Piro");
      auto&      nox_params     = piro_params.sublist("NOX");
      auto&      st_params      = nox_params.sublist("Status Tests");
      auto&      old_params     = st_params.sublist("Test 1");
      auto       new_params     = old_params;
      st_params.remove("Test 0");
      st_params.remove("Test 1");
      st_params.setParameters(new_params);
    }

    if (status == NOX::StatusTest::Failed) {
      *fos_ << "\nINFO: Unable to solve Mechanical problem for subdomain " << subdomain << '\n';
      failed_ = true;
      return;
    }

    // Obtain the solution and its time derivatives
    auto& tr_decorator = *(piro_tr_solver.getDecorator());
    auto  x_rcp        = tr_decorator.get_x();
    auto  xdot_rcp     = tr_decorator.get_x_dot();
    auto  xdotdot_rcp  = tr_decorator.get_x_dotdot();
    auto  a_nrm        = norm_2(*xdotdot_rcp);

    this_x_[subdomain]       = Thyra::createMember(me.get_x_space());
    this_xdot_[subdomain]    = Thyra::createMember(me.get_x_space());
    this_xdotdot_[subdomain] = Thyra::createMember(me.get_x_space());

    Thyra::copy(*x_rcp, this_x_[subdomain].ptr());
    Thyra::copy(*xdot_rcp, this_xdot_[subdomain].ptr());
    Thyra::copy(*xdotdot_rcp, this_xdotdot_[subdomain].ptr());

    // Acummulate values of failed cells and report if applicable
    auto thyra_solution_manager_rcp = piro_tr_solver.getSolutionManager();
    if (thyra_solution_manager_rcp != Teuchos::null) {
      auto& thyra_solution_manager    = *(piro_tr_solver.getSolutionManager());
      auto& adaptive_solution_manager = static_cast<AAdapt::AdaptiveSolutionManager&>(thyra_solution_manager);
      auto& adapter                   = *(adaptive_solution_manager.getAdapter());
      auto& erosion_adapter           = static_cast<AAdapt::Erosion&>(adapter);
      auto& topology                  = *(erosion_adapter.getTopology());
      auto& failure_criterion         = *(topology.get_failure_criterion());
      auto& bulk_failure_criterion    = static_cast<BulkFailureCriterion&>(failure_criterion);

      count_displacement += bulk_failure_criterion.count_displacement;
      count_angle += bulk_failure_criterion.count_angle;
      count_yield += bulk_failure_criterion.count_yield;
      count_strain += bulk_failure_criterion.count_strain;
      count_tension += bulk_failure_criterion.count_tension;

      auto&      fos                 = *Teuchos::VerboseObjectBase::getDefaultOStream();
      auto const failed_displacement = count_displacement / bulk_failure_criterion.failed_threshold;
      auto const failed_angle        = count_angle / bulk_failure_criterion.failed_threshold;
      auto const failed_yield        = count_yield / bulk_failure_criterion.failed_threshold;
      auto const failed_strain       = count_strain / bulk_failure_criterion.failed_threshold;
      auto const failed_tension      = count_tension / bulk_failure_criterion.failed_threshold;
      fos << "INFO: Failed element count";
      fos << ": displacement:" << failed_displacement;
      fos << ", angle:" << failed_angle;
      fos << ", yield:" << failed_yield;
      fos << ", strain:" << failed_strain;
      fos << ", tension:" << failed_tension;
      fos << '\n';
    }
  } else {
    ALBANY_ABORT("Unknown time integrator for mechanics. Only Tempus and Piro Trapezoid Rule supported.");
  }

  failed_ = false;
  */
}


void
SequentialCoupling::setICVecs(ST const time, int const subdomain) const
{
  //IKT 1/7/2025 TODO: fill in this routine 
  /*auto const prob_type       = prob_types_[subdomain];
  auto const is_initial_time = time <= initial_time_ + initial_time_step_;

  if (is_initial_time == true) {
    // initial time-step: get initial solution from nominalValues in ME
    auto&       me = dynamic_cast<Albany::ModelEvaluator&>(*model_evaluators_[subdomain]);
    auto const& nv = me.getNominalValues();

    ics_x_[subdomain] = Thyra::createMember(me.get_x_space());
    Thyra::copy(*(nv.get_x()), ics_x_[subdomain].ptr());

    ics_xdot_[subdomain] = Thyra::createMember(me.get_x_space());
    Thyra::copy(*(nv.get_x_dot()), ics_xdot_[subdomain].ptr());

    if (prob_type == MECHANICAL) {
      ics_xdotdot_[subdomain] = Thyra::createMember(me.get_x_space());
      Thyra::copy(*(nv.get_x_dot_dot()), ics_xdotdot_[subdomain].ptr());
      auto nrm = norm_2(*ics_xdotdot_[subdomain]);
    }
  }

  else {
    // subsequent time steps: update ic vecs based on fields in stk discretization
    auto& abs_disc = *discs_[subdomain];
    auto& stk_disc = static_cast<Albany::STKDiscretization&>(abs_disc);
    auto  x_mv     = stk_disc.getSolutionMV();

    // Update ics_x_ and its time-derivatives
    ics_x_[subdomain] = Thyra::createMember(x_mv->col(0)->space());
    Thyra::copy(*x_mv->col(0), ics_x_[subdomain].ptr());

    ics_xdot_[subdomain] = Thyra::createMember(x_mv->col(1)->space());
    Thyra::copy(*x_mv->col(1), ics_xdot_[subdomain].ptr());

    if (prob_type == MECHANICAL) {
      ics_xdotdot_[subdomain] = Thyra::createMember(x_mv->col(2)->space());
      Thyra::copy(*x_mv->col(2), ics_xdotdot_[subdomain].ptr());
      auto nrm = norm_2(*ics_xdotdot_[subdomain]);
    }
  }*/
}

//IKT 1/7/2025 TODO: may not need the following routines anymore
void
SequentialCoupling::renamePrevWrittenExoFiles(int const subdomain, int const file_index) const
{
  //IKT 1/7/2025 TODO: fill in routine if needed 
  /*if (((file_index - 1) % output_interval_) == 0) {
    Teuchos::ParameterList& params         = solver_factories_[subdomain]->getParameters();
    Teuchos::ParameterList& problem_params = params.sublist("Problem", true);
    Teuchos::ParameterList& disc_params    = params.sublist("Discretization", true);
    std::string             filename_old   = disc_params.get<std::string>("Exodus Output File Name");
    renameExodusFile(file_index - 1 + init_file_index_, filename_old);
    std::string filename_new = filename_old;
    renameExodusFile((file_index - 1) / output_interval_ + init_file_index_, filename_new);
    if (file_index > 0) {
      renameParallel(filename_old, filename_new, comm_);
    }
  }*/
}

void
SequentialCoupling::doDynamicInitialOutput(ST const time, int const subdomain) const
{
  //IKT 1/7/2025 TODO: fill in routine if needed 
  /*auto const is_initial_time = time <= initial_time_ + initial_time_step_;
  if (is_initial_time == false) {
    return;
  }
  // Write solution at specified time to STK mesh
  auto const xMV_rcp         = apps_[subdomain]->getAdaptSolMgr()->getOverlappedSolution();
  auto&      abs_disc        = *discs_[subdomain];
  auto&      stk_disc        = static_cast<Albany::STKDiscretization&>(abs_disc);
  auto&      stk_mesh_struct = *stk_mesh_structs_[subdomain];

  stk_mesh_struct.exoOutputInterval = 1;
  stk_mesh_struct.exoOutput         = do_outputs_[subdomain];
  stk_disc.writeSolutionMV(*xMV_rcp, time, true);
  stk_mesh_struct.exoOutput = false;
  */
}

}  // namespace LandIce
