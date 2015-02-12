//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_SchwarzMultiscale_hpp)
#define LCM_SchwarzMultiscale_hpp

#include "Albany_ModelEvaluatorT.hpp"
#include "Albany_DataTypes.hpp"

namespace LCM {

///
/// \brief Definition for the SchwarzMultiscale
///
class SchwarzMultiscale : public Thyra::ModelEvaluatorDefaultBase<ST> {

public:

  /// Constructor
      /** \brief . */
  SchwarzMultiscale(const Teuchos::RCP<Teuchos::ParameterList>& appParams,
                    const Teuchos::RCP<const Teuchos::Comm<int> >& commT,  
                    const Teuchos::RCP<const Tpetra_Vector>& initial_guessT);

  ///Destructor
  ~SchwarzMultiscale();

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
  Teuchos::RCP<const Teuchos::ParameterList> getValidAppParameters() const;
  Teuchos::RCP<const Teuchos::ParameterList> getValidProblemParameters() const; 
  /// RCP to matDB object
  Teuchos::Array<Teuchos::RCP<QCAD::MaterialDatabase> > material_dbs_;
  
  Teuchos::Array<Teuchos::RCP<Thyra::ModelEvaluator<ST> > > models_;
  Teuchos::Array< Teuchos::RCP<Albany::Application> > apps_;
  Teuchos::RCP<const Teuchos::Comm<int> > commT_;

  //! Cached nominal values -- this contains stuff like x_init, x_dot_init, etc.
  Thyra::ModelEvaluatorBase::InArgs<ST> nominal_values_;

  Thyra::ModelEvaluatorBase::InArgs<ST> createInArgsImpl() const;
      
  Teuchos::RCP<Tpetra_Map> coupled_disc_map_; 
  int n_models_;
  Teuchos::Array<int> num_params_; 
  Teuchos::Array<int> num_responses_; 
  int num_params_total_; //like num_param_vecs
  int num_dist_params_total_; //like dist_param_vecs
  int num_responses_total_; //like num_response_vecs

  //! Sacado parameter vector
  mutable Teuchos::Array<ParamVec> coupled_sacado_param_vec_;

  //! Tpetra map for parameter vector
  Teuchos::Array<Teuchos::RCP<Tpetra_Map> > coupled_param_map_;

  //! Tpetra parameter vector
  Teuchos::Array<Teuchos::RCP<Tpetra_Vector> > coupled_param_vec_;

  mutable Teuchos::Array<Thyra::ModelEvaluatorBase::InArgs<ST> > solver_inargs_; 
  mutable Teuchos::Array<Thyra::ModelEvaluatorBase::OutArgs<ST> > solver_outargs_;


};

} // namespace LCM

#endif // LCM_SchwarzMultiscale_hpp
