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
#include "NOX_PrePostOperator_Vector.H"
#include "SolutionSniffer.hpp"
#include "Thyra_DefaultProductVector.hpp"
#include "Thyra_DefaultProductVectorSpace.hpp"

namespace LCM {

///
/// SchwarzAlternating coupling class
///
class SchwarzAlternating: public Thyra::ResponseOnlyModelEvaluatorBase<ST> {

public:

  /// Constructor
  SchwarzAlternating(
      Teuchos::RCP<Teuchos::ParameterList> const & app_params,
      Teuchos::RCP<Teuchos::Comm<int> const> const & comm,
      Teuchos::RCP<Tpetra_Vector const> const & initial_guess);

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
  set_failed(char const * msg);

  void
  clear_failed();

  bool
  get_failed() const;
  
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
      Thyra::ModelEvaluatorBase::InArgs<ST> const & in_args,
      Thyra::ModelEvaluatorBase::OutArgs<ST> const & out_args) const;
  
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
  reportFinals(std::ostream & os) const;

  Teuchos::Array<Teuchos::RCP<Thyra::ResponseOnlyModelEvaluatorBase<ST>>>
  solvers_;

  Teuchos::ArrayRCP<Teuchos::RCP<Albany::Application>>
  apps_;

  Teuchos::Array<Teuchos::RCP<SolutionSniffer>>
  solution_sniffers_;

  Teuchos::Array<Teuchos::RCP<Albany::AbstractSTKMeshStruct>>
  stk_mesh_structs_;
  
  Teuchos::Array<Teuchos::RCP<Albany::AbstractDiscretization>>
  discs_;

  char const *
  failure_message_{"No failure detected"};

  int
  num_subdomains_{0};

  int
  min_iters_{0};

  int
  max_iters_{0};

  ST
  rel_tol_{0.0};

  ST
  abs_tol_{0.0};

  int
  maximum_steps_{0};

  ST
  initial_time_{0.0};

  ST
  final_time_{0.0};

  ST
  initial_time_step_{0.0};

  int
  output_interval_{1};

  mutable bool
  failed_{false};

  mutable bool
  converged_{false};

  mutable int
  num_iter_{0};

  mutable ST
  rel_error_{0.0};

  mutable ST
  abs_error_{0.0};

  mutable ST
  norm_init_{0.0};

  mutable ST
  norm_final_{0.0};

  mutable ST
  norm_diff_{0.0};

  mutable std::vector<Thyra::ModelEvaluatorBase::InArgs<ST>>
  sub_inargs_;

  mutable std::vector<Thyra::ModelEvaluatorBase::OutArgs<ST>>
  sub_outargs_;

  mutable std::vector<Teuchos::RCP<Thyra::ModelEvaluator<ST>>>
  model_evaluators_;

  mutable std::vector<Teuchos::RCP<NOX::Abstract::Vector>>
  solutions_nox_;

  mutable std::vector<Teuchos::RCP<Thyra::VectorBase<ST>>>
  solutions_thyra_;
  
  mutable std::vector<Teuchos::RCP<Thyra::VectorBase<ST>>>
  solutions_dot_thyra_;

  mutable std::vector<Teuchos::RCP<Thyra::VectorBase<ST>>>
  solutions_dotdot_thyra_;

  // Used if solving with loca or tempus
  bool
  have_loca_{false};

  bool
  have_tempus_{false};

  mutable std::vector<std::reference_wrapper<Teuchos::ParameterList>>
  start_stop_params_;

  mutable std::string
  init_str_{""};

  mutable std::string
  start_str_{""};

  mutable std::string
  stop_str_{""};
};

} // namespace LCM

#endif // LCM_SchwarzAlternating_hpp
