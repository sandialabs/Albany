//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(Aeras_HyperViscosityDecorator_hpp)
#define Aeras_HyperViscosityDecorator_hpp

#include "Albany_ModelEvaluatorT.hpp"
#include "Albany_DataTypes.hpp"
#include "Thyra_DefaultProductVector.hpp"
#include "Thyra_DefaultProductVectorSpace.hpp"

namespace Aeras {

///
/// \brief Definition for the HyperViscosityDecorator
///
class HyperViscosityDecorator: public Thyra::ModelEvaluatorDefaultBase<ST> {

public:

  /// Constructor
  HyperViscosityDecorator(Teuchos::RCP<Teuchos::ParameterList> const & app_params,
      Teuchos::RCP<Teuchos::Comm<int> const > const & commT,
      Teuchos::RCP<Tpetra_Vector const > const & initial_guessT, 
      Teuchos::RCP<Thyra::LinearOpWithSolveFactoryBase<ST> const > const &
      solver_factory);

  ///Destructor
  ~HyperViscosityDecorator();

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

  Teuchos::ArrayView<const std::string> get_g_names(int j) const
  { TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "not impl'ed"); }

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

  Teuchos::RCP<Thyra::LinearOpWithSolveFactoryBase<ST> const>
  get_W_factory() const;

  /// Create InArgs
  Thyra::ModelEvaluatorBase::InArgs<ST>
  createInArgs() const;

  void
  reportFinalPoint(
      Thyra::ModelEvaluatorBase::InArgs<ST> const & final_point,
      bool const was_solved);

protected:

  /// Create operator form of dg/dx for distributed responses
  Teuchos::RCP<Thyra::LinearOpBase<ST> >
  create_DgDx_op_impl(int j) const;

  /// Create operator form of dg/dx_dot for distributed responses
  Teuchos::RCP<Thyra::LinearOpBase<ST> >
  create_DgDx_dot_op_impl(int j) const;

  /// Create OutArgs
  Thyra::ModelEvaluatorBase::OutArgs<ST>
  createOutArgsImpl() const;
  
  //! Evaluate model on InArgs
  void evalModel(
      const Thyra::ModelEvaluatorBase::InArgs<ST>& inArgs,
      const Thyra::ModelEvaluatorBase::OutArgs<ST>& outArgs) const;

private:

  Thyra::ModelEvaluatorBase::InArgs<ST>
  createInArgsImpl() const;

  Teuchos::RCP<Thyra::ModelEvaluator<ST> > model_; 

  Teuchos::RCP<Albany::Application> app_;

  Teuchos::RCP<Teuchos::Comm<int> const>
  commT_;

  Teuchos::Array<Teuchos::RCP<Tpetra_Map const> >
  disc_maps_;

  int num_models_;

  int num_param_vecs_; 
  
  int num_dist_param_vecs_; 

  //for setting get_W_factory() 
  Teuchos::RCP<Thyra::LinearOpWithSolveFactoryBase<ST> const>
  solver_factory_;
    

};

} // namespace Aeras

#endif // Aeras_HyperViscosityDecorator_hpp
