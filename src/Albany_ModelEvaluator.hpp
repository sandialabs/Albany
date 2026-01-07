//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_MODEL_EVALUATOR_HPP
#define ALBANY_MODEL_EVALUATOR_HPP

#include "Albany_SacadoTypes.hpp"
#include "Albany_TpetraTypes.hpp"
#include "Albany_ThyraTypes.hpp"
#include "Albany_DistributedParameterLibrary.hpp"
#include "Albany_DistributedParameterDerivativeOp.hpp"

#include "Piro_TransientDecorator.hpp"

namespace Albany {

// Forward declarations
class Application;
class DistributedParameterLibrary;

class ModelEvaluator : public Piro::TransientDecorator<ST> {
public:
  // Constructor
  ModelEvaluator(
      const Teuchos::RCP<Application>& app,
      const Teuchos::RCP<Teuchos::ParameterList>& appParams,
      const bool adjoint_model);

  /** \name Overridden from Thyra::ModelEvaluator<ST> . */
  //@{

  //! Return solution vector map
  Teuchos::RCP<const Thyra_VectorSpace>
  get_x_space() const;

  //! Return residual vector map
  Teuchos::RCP<const Thyra_VectorSpace>
  get_f_space() const;

  //! Return parameter vector map
  Teuchos::RCP<const Thyra_VectorSpace>
  get_p_space(int l) const;

  //! Return response function map
  Teuchos::RCP<const Thyra_VectorSpace>
  get_g_space(int j) const;

  //! Return array of parameter names
  Teuchos::RCP<const Teuchos::Array<std::string>>
  get_p_names(int l) const;

  Teuchos::ArrayView<const std::string>
  get_g_names(int /* j */) const {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "not impl'ed");
  }

  Thyra_InArgs getNominalValues () const { return nominalValues; }
  Thyra_InArgs getLowerBounds   () const { return lowerBounds; }
  Thyra_InArgs getUpperBounds   () const { return upperBounds; }

  Teuchos::RCP<Thyra_LinearOp>  create_W_op() const;

  //! Create preconditioner operator
  Teuchos::RCP<Thyra_Preconditioner> create_W_prec() const;

  Teuchos::RCP<const Thyra_LOWS_Factory>  get_W_factory() const;

  //! Create Hessian operator
  Teuchos::RCP<Thyra_LinearOp>  create_hess_g_pp( int j, int l1, int l2 ) const;

  //! Set nominal value
  void setNominalValue(int j, Teuchos::RCP<Thyra_Vector> p);

  //! Create InArgs
  Thyra_InArgs createInArgs() const;

  void reportFinalPoint(const Thyra_InArgs& finalPoint,
                        const bool wasSolved);

  void allocateVectors();

  //@}

  Teuchos::RCP<Application> getAlbanyApp () const { return app; }
  
  Teuchos::RCP<const DistributedParameter> setDistParamVec(const std::string p_name, const Teuchos::ParameterList param_list);

 protected:
  /** \name Overridden from Thyra::ModelEvaluatorDefaultBase<ST> . */
  //@{

  //! Create operator form of df/dp for distributed parameters
  Teuchos::RCP<Thyra::LinearOpBase<ST>>
  create_DfDp_op_impl(int j) const;

  //! Create operator form of dg/dx for distributed responses
  Teuchos::RCP<Thyra::LinearOpBase<ST>>
  create_DgDx_op_impl(int j) const;

  //! Create operator form of dg/dx_dot for distributed responses
  Teuchos::RCP<Thyra::LinearOpBase<ST>>
  create_DgDx_dot_op_impl(int j) const;

  //! Create operator form of dg/dx_dotdot for distributed responses
  Teuchos::RCP<Thyra::LinearOpBase<ST>>
  create_DgDx_dotdot_op_impl(int j) const;

  //! Create OutArgs
  Thyra::ModelEvaluatorBase::OutArgs<ST>
  createOutArgsImpl() const;

  //! Evaluate model on InArgs
  void
  evalModelImpl(
      const Thyra_InArgs& inArgs,
      const Thyra::ModelEvaluatorBase::OutArgs<ST>& outArgs) const;

  //! Application object
  Teuchos::RCP<Albany::Application> app;
  Teuchos::RCP<Teuchos::ParameterList> appParams;

  Teuchos::RCP<Teuchos::Time> timer;

  //! Sacado parameter vector
  mutable Teuchos::Array<ParamVec> sacado_param_vec;

  //! Jacobian for sending to user preconditioner
  mutable Teuchos::RCP<Thyra_LinearOp> Extra_W_op;

  //! Whether the problem supplies its own preconditioner
  bool supplies_prec;

  //! Boolean marking whether Tempus is used 
  bool use_tempus{false}; 

  //@}

  //! Total number of parameter vectors (num_param_vecs+num_dist_param_vecs)
  int total_num_param_vecs;

  //! Number of parameter vectors
  int num_param_vecs;

  Thyra_InArgs createInArgsImpl() const;

  //! Cached nominal values and lower/upper bounds
  Thyra_InArgs nominalValues;
  Thyra_InArgs lowerBounds;
  Thyra_InArgs upperBounds;

  //! List of free parameter names
  Teuchos::Array<Teuchos::RCP<Teuchos::Array<std::string>>> param_names;

  //! Thyra vector spaces for parameter vector
  Teuchos::Array<Teuchos::RCP<const Thyra_VectorSpace>> param_vss;

  //! Thyra vectors for parameters and their bounds
  Teuchos::Array<Teuchos::RCP<Thyra_Vector>> param_vecs;
  Teuchos::Array<Teuchos::RCP<Thyra_Vector>> param_lower_bds;
  Teuchos::Array<Teuchos::RCP<Thyra_Vector>> param_upper_bds;

  //! Thyra response vector
  Teuchos::Array<Teuchos::RCP<Thyra_Vector>> thyra_response_vec;

  //! Number of distributed parameter vectors
  int num_dist_param_vecs;

  //! List of enabled distributed parameters
  Teuchos::Array<std::string> dist_param_names;

  //! Distributed parameter library
  Teuchos::RCP<DistributedParameterLibrary> distParamLib;

  //! Model uses time integration (velocities)
  bool supports_xdot;

  //! Model uses time integration (accelerations)
  bool supports_xdotdot;

  //! As it says, when reportFinalPoint is called.
  bool overwriteNominalValuesWithFinalPoint;

  //! Tells code whether to construct forward or adjoint ME
  //The latter has the Jacobian transposed
  bool adjoint_model;
};

} // namespace Albany

#endif // ALBANY_MODEL_EVALUATOR_HPP
