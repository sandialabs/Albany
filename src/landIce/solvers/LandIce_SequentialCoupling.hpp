
// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#if !defined(LandIce_SequentialCoupling_hpp)
#define LandIce_SequentialCoupling_hpp

#include <functional>

#include "Albany_AbstractDiscretization.hpp"
#include "Albany_AbstractSTKMeshStruct.hpp"
#include "Albany_Application.hpp"
#include "Albany_ModelEvaluator.hpp"
#include "Albany_SolverFactory.hpp"
#include "Piro_NOXSolver.hpp"
#include "Thyra_DefaultProductVector.hpp"
#include "Thyra_DefaultProductVectorSpace.hpp"
#include "Thyra_ModelEvaluatorDelegatorBase.hpp"
#include "Thyra_ResponseOnlyModelEvaluatorBase.hpp"

namespace LandIce {

///
/// SequentialCoupling coupling class
///
class SequentialCoupling : public Thyra::ResponseOnlyModelEvaluatorBase<ST>
{
 public:
  /// Constructor
  SequentialCoupling(Teuchos::RCP<Teuchos::ParameterList> const& app_params, Teuchos::RCP<Teuchos::Comm<int> const> const& comm);

  /// Destructor
  ~SequentialCoupling();

  /// Return solution vector map
  Teuchos::RCP<Thyra::VectorSpaceBase<ST> const>
  get_x_space() const;

  /// Return residual vector map
  Teuchos::RCP<Thyra::VectorSpaceBase<ST> const>
  get_f_space() const;

  /// Return parameter vector map
  Teuchos::RCP<Thyra::VectorSpaceBase<ST> const>
  get_p_space(int l) const;

  /// Return response function map
  Teuchos::RCP<Thyra::VectorSpaceBase<ST> const>
  get_g_space(int j) const;

  /// Return array of parameter names
  Teuchos::RCP<Teuchos::Array<std::string> const>
  get_p_names(int l) const;

  Teuchos::ArrayView<std::string const>
  get_g_names(int j) const;

  Thyra::ModelEvaluatorBase::InArgs<ST>
  getNominalValues() const;

  Thyra::ModelEvaluatorBase::InArgs<ST>
  getLowerBounds() const;

  Thyra::ModelEvaluatorBase::InArgs<ST>
  getUpperBounds() const;

  Teuchos::RCP<Thyra::LinearOpBase<ST>>
  create_W_op() const;

  /// Create preconditioner operator
  Teuchos::RCP<Thyra::PreconditionerBase<ST>>
  create_W_prec() const;

  Teuchos::RCP<Thyra::LinearOpWithSolveFactoryBase<ST> const>
  get_W_factory() const;

  /// Create InArgs
  Thyra::ModelEvaluatorBase::InArgs<ST>
  createInArgs() const;

  Teuchos::ArrayRCP<Teuchos::RCP<Albany::Application>>
  getApps() const;

  enum class ConvergenceCriterion
  {
    ABSOLUTE,
    RELATIVE,
    BOTH
  };
  enum class ConvergenceLogicalOperator
  {
    AND,
    OR
  };

 private:
  /// Create operator form of dg/dx for distributed responses
  Teuchos::RCP<Thyra::LinearOpBase<ST>>
  create_DgDx_op_impl(int j) const;

  /// Create operator form of dg/dx_dot for distributed responses
  Teuchos::RCP<Thyra::LinearOpBase<ST>>
  create_DgDx_dot_op_impl(int j) const;

  /// Create OutArgs
  Thyra::ModelEvaluatorBase::OutArgs<ST>
  createOutArgsImpl() const;

  /// Evaluate model on InArgs
  void
  evalModelImpl(Thyra::ModelEvaluatorBase::InArgs<ST> const& in_args, Thyra::ModelEvaluatorBase::OutArgs<ST> const& out_args) const;

  Thyra::ModelEvaluatorBase::InArgs<ST>
  createInArgsImpl() const;

  void
  SequentialCouplingLoop() const;

  void
  AdvancePoisson(int const subdomain, bool const is_initial_state, double const current_time, double const next_time, double const time_step) const;

  void
  AdvanceAdvDiff(int const subdomain, bool const is_initial_state, double const current_time, double const next_time, double const time_step) const;

  bool
  continueSolve() const;

  void
  createPoissonSolverAppDiscME(int const file_index, double const current_time) const;

  void
  createAdvDiffSolverAppDiscME(int const file_index, double const current_time, double const next_time, double const time_step) const;

  void
  doDynamicInitialOutput(ST const time, int const subdomain) const;

  void
  renamePrevWrittenExoFiles(const int subdomain, const int file_index) const;

  void
  setICVecs(ST const time, int const subdomain) const;

  std::vector<Teuchos::RCP<Albany::SolverFactory>>                             solver_factories_;
  mutable std::vector<Teuchos::RCP<Thyra::ResponseOnlyModelEvaluatorBase<ST>>> solvers_;
  mutable Teuchos::ArrayRCP<Teuchos::RCP<Albany::Application>>                 apps_;
  mutable std::vector<Teuchos::RCP<Albany::AbstractSTKMeshStruct>>             stk_mesh_structs_;
  mutable std::vector<Teuchos::RCP<Albany::AbstractDiscretization>>            discs_;
  std::vector<Teuchos::RCP<Teuchos::ParameterList>>                            init_pls_;

  int          num_subdomains_{0};
  int          maximum_steps_{0};
  mutable ST   initial_time_{0.0};
  mutable ST   final_time_{0.0};
  ST           initial_time_step_{0.0};
  ST           min_time_step_{0.0};
  ST           max_time_step_{0.0};
  ST           reduction_factor_{0.0};
  ST           increase_factor_{0.0};
  int          output_interval_{1};
  mutable bool failed_{false};
  mutable bool converged_{false};
  mutable int  num_iter_{0};

  mutable ConvergenceCriterion       criterion_{ConvergenceCriterion::BOTH};
  mutable ConvergenceLogicalOperator operator_{ConvergenceLogicalOperator::AND};

  mutable std::vector<Teuchos::RCP<Thyra::VectorBase<ST> const>> curr_x_;
  mutable std::vector<Teuchos::RCP<Thyra::VectorBase<ST> const>> prev_step_x_;

  mutable std::vector<Thyra::ModelEvaluatorBase::InArgs<ST>>   sub_inargs_;
  mutable std::vector<Thyra::ModelEvaluatorBase::OutArgs<ST>>  sub_outargs_;
  mutable std::vector<Teuchos::RCP<Thyra::ModelEvaluator<ST>>> model_evaluators_;
  mutable std::vector<Teuchos::RCP<Thyra::VectorBase<ST>>>     ics_x_;
  mutable std::vector<Teuchos::RCP<Thyra::VectorBase<ST>>>     ics_xdot_;
  mutable std::vector<Teuchos::RCP<Thyra::VectorBase<ST>>>     prev_x_;
  mutable std::vector<Teuchos::RCP<Thyra::VectorBase<ST>>>     prev_xdot_;
  mutable std::vector<Teuchos::RCP<Thyra::VectorBase<ST>>>     this_x_;
  mutable std::vector<Teuchos::RCP<Thyra::VectorBase<ST>>>     this_xdot_;

  // variable with previous thermal Exodus output file, for mechanical restarts
  mutable std::string prev_poisson_exo_outfile_name_{""};
  // variable with previous mechanical Exodus output file, for thermal restarts
  mutable std::string prev_advdiff_exo_outfile_name_{""};

  mutable std::vector<bool>             do_outputs_;
  mutable std::vector<bool>             do_outputs_init_;

  enum PROB_TYPE
  {
    POISSON,
    ADVDIFF
  };

  //IKT 1/7/2025 TODO: delete/rename the following class and function
  enum class MechanicalSolver
  {
    Tempus,
    LOCA,
    TrapezoidRule
  };
  mutable MechanicalSolver mechanical_solver_{MechanicalSolver::TrapezoidRule};

  // std::vector mapping subdomain number to PROB_TYPE;
  mutable std::vector<PROB_TYPE> prob_types_;

  Teuchos::RCP<Teuchos::FancyOStream> fos_;

  Teuchos::RCP<Teuchos::ParameterList>   alt_system_params_;
  Teuchos::RCP<Teuchos::Comm<int> const> comm_;
  Teuchos::Array<std::string>            model_filenames_;

};

}  // namespace LCM

#endif  // LandIce_SequentialCoupling_hpp
