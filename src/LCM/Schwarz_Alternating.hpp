//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_SchwarzAlternating_hpp)
#define LCM_SchwarzAlternating_hpp

#include "Albany_AbstractSTKMeshStruct.hpp"
#include "Albany_DataTypes.hpp"
#include "Albany_ModelEvaluatorT.hpp"
#include "NOXSolverPrePostOperator.h"
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

  /// Schwarz Alternating loop
  void
  SchwarzLoop() const;

  Teuchos::Array<Teuchos::RCP<Thyra::ResponseOnlyModelEvaluatorBase<ST>>>
  solvers_;

  Teuchos::ArrayRCP<Teuchos::RCP<Albany::Application>>
  apps_;

  Teuchos::Array<Teuchos::RCP<NOXSolverPrePostOperator>>
  convergence_ops_;

  Teuchos::Array<Teuchos::RCP<Albany::AbstractSTKMeshStruct>>
  stk_mesh_structs_;

  /// Cached nominal values -- this contains stuff like x_init, x_dot_init, etc.
  Thyra::ModelEvaluatorBase::InArgs<ST>
  nominal_values_;
  
  int
  num_subdomains_{0};

  int
  min_iters_{0};

  int
  max_iters_{0};

  int
  output_interval_{1};

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
