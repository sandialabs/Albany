//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_SchwarzCoupled_hpp)
#define LCM_SchwarzCoupled_hpp

#include "Albany_DataTypes.hpp"
#include "Albany_MaterialDatabase.hpp"
#include "Albany_ModelEvaluatorT.hpp"
#include "Schwarz_BoundaryJacobian.hpp"
#include "Thyra_DefaultProductVector.hpp"
#include "Thyra_DefaultProductVectorSpace.hpp"

namespace LCM {

///
/// SchwarzCoupled coupling class
///
class SchwarzCoupled : public Thyra::ModelEvaluatorDefaultBase<ST>
{
 public:
  /// Constructor
  SchwarzCoupled(
      Teuchos::RCP<Teuchos::ParameterList> const&   app_params,
      Teuchos::RCP<Teuchos::Comm<int> const> const& comm,
      Teuchos::RCP<Tpetra_Vector const> const&      initial_guessT,
      Teuchos::RCP<Thyra::LinearOpWithSolveFactoryBase<ST> const> const&
          lowsfb);

  /// Destructor
  ~SchwarzCoupled();

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
  Teuchos::RCP<Thyra_Preconditioner>
  create_W_prec() const;

  Teuchos::RCP<Thyra::LinearOpWithSolveFactoryBase<ST> const>
  get_W_factory() const;

  /// Create InArgs
  Thyra::ModelEvaluatorBase::InArgs<ST>
  createInArgs() const;

  void
  reportFinalPoint(
      Thyra::ModelEvaluatorBase::InArgs<ST> const& final_point,
      bool const                                   was_solved);

  void
  allocateVectors();

  Teuchos::RCP<Thyra::VectorSpaceBase<ST> const>
  getThyraRangeSpace() const;

  Teuchos::RCP<Thyra::VectorSpaceBase<ST> const>
  getThyraDomainSpace() const;

  Teuchos::RCP<Thyra::VectorSpaceBase<ST> const>
  getThyraResponseSpace(int l) const;

  Teuchos::RCP<Thyra::VectorSpaceBase<ST> const>
  getThyraParamSpace(int l) const;

  Teuchos::ArrayRCP<Teuchos::RCP<Albany::Application>>
  getApps() const
  {
    return apps_;
  }

 protected:
  mutable Teuchos::RCP<Thyra::ProductVectorSpaceBase<ST>> range_space_;

  mutable Teuchos::RCP<Thyra::ProductVectorSpaceBase<ST>> domain_space_;

  mutable Teuchos::RCP<Thyra::ProductVectorSpaceBase<ST>> response_space_;

  mutable Teuchos::RCP<Thyra::ProductVectorSpaceBase<ST>> param_space_;

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

 private:
  Teuchos::RCP<Teuchos::ParameterList const>
  getValidAppParameters() const;

  Teuchos::RCP<Teuchos::ParameterList const>
  getValidProblemParameters() const;

  Thyra::ModelEvaluatorBase::InArgs<ST>
  createInArgsImpl() const;

  /// List of free parameter names
  Teuchos::Array<Teuchos::RCP<Teuchos::Array<std::string>>> param_names_;

  /// RCP to matDB object
  Teuchos::Array<Teuchos::RCP<Albany::MaterialDatabase>> material_dbs_;

  Teuchos::Array<Teuchos::RCP<Thyra::ModelEvaluator<ST>>> models_;

  /// Own the application parameters.
  Teuchos::Array<Teuchos::RCP<Teuchos::ParameterList>> model_app_params_;

  Teuchos::ArrayRCP<Teuchos::RCP<Albany::Application>> apps_;

  Teuchos::RCP<Teuchos::Comm<int> const> comm_;

  /// Cached nominal values -- this contains stuff like x_init, x_dot_init, etc.
  Thyra::ModelEvaluatorBase::InArgs<ST> nominal_values_;

  Teuchos::Array<Teuchos::RCP<Tpetra_Map const>> disc_maps_;

  /// Teuchos array holding main diagonal jacobians (non-coupled models)
  Teuchos::Array<Teuchos::RCP<Tpetra_CrsMatrix>> jacs_;

  /// Teuchos array holding main diagonal preconditioners (non-coupled models)
  Teuchos::Array<Teuchos::RCP<Tpetra_CrsMatrix>> precs_;

  int num_models_;

  /// Like num_param_vecs
  int num_params_total_;

  /// Like dist_param_vecs
  int num_dist_params_total_;

  /// Like num_response_vecs
  int num_responses_total_;

  /// For setting get_W_factory()
  Teuchos::RCP<Thyra::LinearOpWithSolveFactoryBase<ST> const> lowsfb_;

  /// Array of Sacado parameter vectors
  mutable Teuchos::Array<Teuchos::Array<ParamVec>> sacado_param_vecs_;

  mutable Teuchos::Array<Thyra::ModelEvaluatorBase::InArgs<ST>> solver_inargs_;

  mutable Teuchos::Array<Thyra::ModelEvaluatorBase::OutArgs<ST>>
      solver_outargs_;

  bool w_prec_supports_;

  bool supports_xdot_;

  enum MF_PREC_TYPE
  {
    NONE,
    JACOBI,
    ABS_ROW_SUM,
    ID
  };

  MF_PREC_TYPE mf_prec_type_;
};

}  // namespace LCM

#endif  // LCM_SchwarzCoupled_hpp
