//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_SchwarzMultiscale_hpp)
#define LCM_SchwarzMultiscale_hpp

#include "Albany_ModelEvaluatorT.hpp"

namespace LCM {

///
/// \brief Definition for the SchwarzMultiscale
///
class SchwarzMultiscale : public Albany::ModelEvaluatorT {

public:

  /// Constructor
  SchwarzMultiscale(
      const Teuchos::RCP<Albany::Application> & app,
      const Teuchos::RCP<Teuchos::ParameterList> & app_params);

  /// Return solution vector map
  Teuchos::RCP<const Thyra::VectorSpaceBase<ST> >
  get_x_space() const;

  /// Return residual vector map
  Teuchos::RCP<const Thyra::VectorSpaceBase<ST> >
  get_f_space() const;

  /// Return parameter vector map
  Teuchos::RCP<const Thyra::VectorSpaceBase<ST> >
  get_p_space(int l) const;

  /// Return response function map
  Teuchos::RCP<const Thyra::VectorSpaceBase<ST> >
  get_g_space(int j) const;

  /// Return array of parameter names
  Teuchos::RCP<const Teuchos::Array<std::string> >
  get_p_names(int l) const;

  Thyra::ModelEvaluatorBase::InArgs<ST>
  getNominalValues() const;

  Thyra::ModelEvaluatorBase::InArgs<ST>
  getLowerBounds() const;

  Thyra::ModelEvaluatorBase::InArgs<ST>
  getUpperBounds() const;

  Teuchos::RCP<Thyra::LinearOpBase<ST> >
  create_W_op() const;

  /// Create preconditioner operator
  Teuchos::RCP<Thyra::PreconditionerBase<ST> >
  create_W_prec() const;

  Teuchos::RCP<const Thyra::LinearOpWithSolveFactoryBase<ST> >
  get_W_factory() const;

  /// Create InArgs
  Thyra::ModelEvaluatorBase::InArgs<ST>
  createInArgs() const;

  void
  reportFinalPoint(
      Thyra::ModelEvaluatorBase::InArgs<ST> const & final_point,
      bool const was_solved);

  void
  allocateVectors();

protected:
  /// Create operator form of df/dp for distributed parameters
  Teuchos::RCP<Thyra::LinearOpBase<ST> >
  create_DfDp_op_impl(int j) const;

  /// Create operator form of dg/dx for distributed responses
  Teuchos::RCP<Thyra::LinearOpBase<ST> >
  create_DgDx_op_impl(int j) const;

  /// Create operator form of dg/dx_dot for distributed responses
  Teuchos::RCP<Thyra::LinearOpBase<ST> >
  create_DgDx_dot_op_impl(int j) const;

  /// Create OutArgs
  Thyra::ModelEvaluatorBase::OutArgs<ST>
  createOutArgsImpl() const;

  /// Evaluate model on InArgs
  void evalModelImpl(
      Thyra::ModelEvaluatorBase::InArgs<ST> const & in_args,
      Thyra::ModelEvaluatorBase::OutArgs<ST> const & out_args) const;

private:
  ///
  /// RCP to matDB object
  ///
  Teuchos::RCP<QCAD::MaterialDatabase> material_db_;

};

} // namespace LCM

#endif // LCM_SchwarzMultiscale_hpp
