//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_SchwarzAlternating_hpp)
#define LCM_SchwarzAlternating_hpp

#include "Albany_DataTypes.hpp"
#include "Albany_ModelEvaluatorT.hpp"
#include "MaterialDatabase.h"
#include "NOX_Abstract_PrePostOperator.H"
#include "Thyra_DefaultProductVector.hpp"
#include "Thyra_DefaultProductVectorSpace.hpp"

namespace LCM {

///
/// NOX PrePostOperator used for Schwarz loop convergence criterion.
///
class SchwarzConvergenceCriterion : public NOX::Abstract::PrePostOperator {

public:

  SchwarzConvergenceCriterion();

  virtual void
  runPreIterate(NOX::Solver::Generic const & solver);

  virtual void
  runPostIterate(NOX::Solver::Generic const & solver);

  virtual void
  runPreSolve(NOX::Solver::Generic const & solver);

  virtual void
  runPostSolve(NOX::Solver::Generic const & solver);

  ST
  getInitialNorm();

  ST
  getFinalNorm();

  ST
  getDifferenceNorm();

private:

  Teuchos::RCP<NOX::Abstract::Vector>
  soln_init_{Teuchos::null};

  ST
  norm_init_{0.0};

  ST
  norm_final_{0.0};

  ST
  norm_diff_{0.0};
};

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

protected:

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
  
private:

  Teuchos::RCP<Teuchos::ParameterList const>
  getValidAppParameters() const;

  Teuchos::RCP<Teuchos::ParameterList const>
  getValidProblemParameters() const;

  Thyra::ModelEvaluatorBase::InArgs<ST>
  createInArgsImpl() const;

  /// Schwarz Alternating loop
  void
  SchwarzLoop() const;

  Teuchos::Array<Teuchos::RCP<Thyra::ResponseOnlyModelEvaluatorBase<ST>>>
  solvers_;

  Teuchos::ArrayRCP<Teuchos::RCP<Albany::Application>>
  apps_;

  /// Cached nominal values -- this contains stuff like x_init, x_dot_init, etc.
  Thyra::ModelEvaluatorBase::InArgs<ST>
  nominal_values_;
  
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

  mutable Teuchos::Array<Thyra::ModelEvaluatorBase::InArgs<ST>>
  sub_inargs_;

  mutable Teuchos::Array<Thyra::ModelEvaluatorBase::OutArgs<ST>>
  sub_outargs_;
};

} // namespace LCM

#endif // LCM_SchwarzAlternating_hpp
