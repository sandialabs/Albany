//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_SchwarzAlternating_hpp)
#define LCM_SchwarzAlternating_hpp

#include <functional>

#include "Albany_AbstractSTKMeshStruct.hpp"
#include "Albany_DataTypes.hpp"
#include "Albany_MaterialDatabase.hpp"
#include "Albany_ModelEvaluatorT.hpp"
#include "Piro_NOXSolver.hpp"
#include "Thyra_DefaultProductVector.hpp"
#include "Thyra_DefaultProductVectorSpace.hpp"
#include "Thyra_ResponseOnlyModelEvaluatorBase.hpp"

namespace LCM {

//
// These are to mirror Albany::StateArrays, which are shards:Arrays
// under the hood, which in turn use for storage a raw pointer that comes
// from the depths of STK. Thus, to make a copy of the states without
// touching that pointer, we create these so that the values can be
// passed back and forth between LCM::StateArrays and Albany::StateArrays
// whenever we need to reset states.
//
using StateArray    = std::map<std::string, std::vector<ST>>;
using StateArrayVec = std::vector<StateArray>;

struct StateArrays
{
  StateArrayVec element_state_arrays;

  StateArrayVec node_state_arrays;
};

///
/// SchwarzAlternating coupling class
///
class SchwarzAlternating : public Thyra::ResponseOnlyModelEvaluatorBase<ST>
{
 public:
  /// Constructor
  SchwarzAlternating(
      Teuchos::RCP<Teuchos::ParameterList> const&   app_params,
      Teuchos::RCP<Teuchos::Comm<int> const> const& comm,
      Teuchos::RCP<Tpetra_Vector const> const&      initial_guess);

  /// Destructor
  ~SchwarzAlternating();

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

  Teuchos::ArrayView<const std::string>
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

  void
  set_failed(char const* msg);

  void
  clear_failed();

  bool
  get_failed() const;

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
  evalModelImpl(
      Thyra::ModelEvaluatorBase::InArgs<ST> const&  in_args,
      Thyra::ModelEvaluatorBase::OutArgs<ST> const& out_args) const;

  Thyra::ModelEvaluatorBase::InArgs<ST>
  createInArgsImpl() const;

  /// Schwarz Alternating loops
  void
  SchwarzLoopQuasistatics() const;

  void
  SchwarzLoopDynamics() const;

  void
  updateConvergenceCriterion() const;

  bool
  continueSolve() const;

  void
  doQuasistaticOutput(ST const time) const;

  void
  setExplicitUpdateInitialGuessForSchwarz(
      ST const current_time,
      ST const time_step) const;

  void
  setDynamicICVecsAndDoOutput(ST const time) const;

  void
  reportFinals(std::ostream& os) const;

  std::vector<Teuchos::RCP<Thyra::ResponseOnlyModelEvaluatorBase<ST>>> solvers_;
  Teuchos::ArrayRCP<Teuchos::RCP<Albany::Application>>                 apps_;
  std::vector<Teuchos::RCP<Albany::AbstractSTKMeshStruct>>  stk_mesh_structs_;
  std::vector<Teuchos::RCP<Albany::AbstractDiscretization>> discs_;

  char const*  failure_message_{"No failure detected"};
  int          num_subdomains_{0};
  int          min_iters_{0};
  int          max_iters_{0};
  ST           rel_tol_{0.0};
  ST           abs_tol_{0.0};
  int          maximum_steps_{0};
  ST           initial_time_{0.0};
  ST           final_time_{0.0};
  ST           initial_time_step_{0.0};
  ST           tol_factor_vel_{0.0};
  ST           tol_factor_acc_{0.0};
  ST           min_time_step_{0.0};
  ST           max_time_step_{0.0};
  ST           reduction_factor_{0.0};
  ST           increase_factor_{0.0};
  int          output_interval_{1};
  mutable bool failed_{false};
  mutable bool converged_{false};
  mutable int  num_iter_{0};
  mutable ST   rel_error_{0.0};
  mutable ST   abs_error_{0.0};
  mutable ST   norm_init_{0.0};
  mutable ST   norm_final_{0.0};
  mutable ST   norm_diff_{0.0};

  mutable ConvergenceCriterion       criterion_{ConvergenceCriterion::BOTH};
  mutable ConvergenceLogicalOperator operator_{ConvergenceLogicalOperator::AND};

  mutable std::vector<Teuchos::RCP<Thyra::VectorBase<ST> const>> curr_disp_;
  mutable std::vector<Teuchos::RCP<Thyra::VectorBase<ST> const>>
      prev_step_disp_;

  mutable std::vector<Thyra::ModelEvaluatorBase::InArgs<ST>>  sub_inargs_;
  mutable std::vector<Thyra::ModelEvaluatorBase::OutArgs<ST>> sub_outargs_;
  mutable std::vector<Teuchos::RCP<Thyra::ModelEvaluator<ST>>>
                                                           model_evaluators_;
  mutable std::vector<Teuchos::RCP<Thyra::VectorBase<ST>>> ics_disp_;
  mutable std::vector<Teuchos::RCP<Thyra::VectorBase<ST>>> ics_velo_;
  mutable std::vector<Teuchos::RCP<Thyra::VectorBase<ST>>> ics_acce_;
  mutable std::vector<Teuchos::RCP<Thyra::VectorBase<ST>>> prev_disp_;
  mutable std::vector<Teuchos::RCP<Thyra::VectorBase<ST>>> prev_velo_;
  mutable std::vector<Teuchos::RCP<Thyra::VectorBase<ST>>> prev_acce_;
  mutable std::vector<Teuchos::RCP<Thyra::VectorBase<ST>>> this_disp_;
  mutable std::vector<Teuchos::RCP<Thyra::VectorBase<ST>>> this_velo_;
  mutable std::vector<Teuchos::RCP<Thyra::VectorBase<ST>>> this_acce_;

  mutable std::vector<LCM::StateArrays> internal_states_;
  mutable std::vector<bool>             do_outputs_;
  mutable std::vector<bool>             do_outputs_init_;

  // Used if solving with loca or tempus
  bool is_static_{false};
  bool is_dynamic_{false};
  bool std_init_guess_{false};
};

}  // namespace LCM

#endif  // LCM_SchwarzAlternating_hpp
