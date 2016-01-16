//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AADAPT_THYRA_EPETRA_MODEL_EVALUATOR_HPP
#define AADAPT_THYRA_EPETRA_MODEL_EVALUATOR_HPP

#include "Albany_DataTypes.hpp"
#include "Thyra_EpetraModelEvaluator.hpp"

namespace AAdapt {

/** \brief Concrete Adapter subclass that takes an
 * <tt>EpetraExt::ModelEvaluator</tt> object and wraps it as a
 * <tt>AAdapt::ThyraAdaptiveModelEvaluator</tt> object.
 *
 * Note that this is an adaptive subclass of Thyra::ModelEvaluator
 * most of the interface is a pass-through to underlying Thyra::ModelEvaluator functions or duplicated
 * from that class
 */


class ThyraAdaptiveModelEvaluator
  : public Thyra::EpetraModelEvaluator
{
public:

  /** \name Constructors/initializers/accessors/utilities. */
  //@{

  /** \brief . */
  ThyraAdaptiveModelEvaluator();

  /** \brief . */
  ThyraAdaptiveModelEvaluator(
    const Teuchos::RCP<const EpetraExt::ModelEvaluator> &epetraModel,
    const Teuchos::RCP<Thyra::LinearOpWithSolveFactoryBase<ST> > &W_factory
    );

  /** \brief . */
  void initialize(
    const Teuchos::RCP<const EpetraExt::ModelEvaluator> &epetraModel,
    const Teuchos::RCP<Thyra::LinearOpWithSolveFactoryBase<ST> > &W_factory
    );

  /** \brief . */
  Teuchos::RCP<const EpetraExt::ModelEvaluator> getEpetraModel() const;

  /** \brief Set the nominal values.
   *
   * Warning, if scaling is being used, these must be according to the scaled
   * values, not the original unscaled values.
   */
  void setNominalValues( const Thyra::ModelEvaluatorBase::InArgs<ST>& nominalValues );
  
  /** \brief Set the state variable scaling vector <tt>s_x</tt> (see above).
   *
   * This function must be called after <tt>intialize()</tt> or the
   * constructur in order to set the scaling vector correctly!
   *
   * ToDo: Move this into an external strategy class object!
   */
  void setStateVariableScalingVec(
    const Teuchos::RCP<const Epetra_Vector> &stateVariableScalingVec
    );
  
  /** \brief Get the state variable scaling vector <tt>s_x</tt> (see
   * above). */
  Teuchos::RCP<const Epetra_Vector>
  getStateVariableInvScalingVec() const;
  
  /** \brief Get the inverse state variable scaling vector <tt>inv_s_x</tt>
   * (see above). */
  Teuchos::RCP<const Epetra_Vector>
  getStateVariableScalingVec() const;
  
  /** \brief Set the state function scaling vector <tt>s_f</tt> (see
   * above). */
  void setStateFunctionScalingVec(
    const Teuchos::RCP<const Epetra_Vector> &stateFunctionScalingVec
    );
  
  /** \brief Get the state function scaling vector <tt>s_f</tt> (see
   * above). */
  Teuchos::RCP<const Epetra_Vector>
  getStateFunctionScalingVec() const;

  /** \brief . */
  void uninitialize(
    Teuchos::RCP<const EpetraExt::ModelEvaluator> *epetraModel = NULL,
    Teuchos::RCP<Thyra::LinearOpWithSolveFactoryBase<ST> > *W_factory = NULL
    );
  
  /** \brief . */
  const Thyra::ModelEvaluatorBase::InArgs<ST>& getFinalPoint() const;

  /** \brief . */
  bool finalPointWasSolved() const;

  /** \brief . */
  const Teuchos::RCP<Thyra::VectorBase<ST> >
    resize_g_space(int index, Teuchos::RCP<const Epetra_Map> map);

  //@}

  /** \name Public functions overridden from Teuchos::Describable. */
  //@{

  /** \brief . */
  std::string description() const;

  //@}

  /** @name Overridden from ParameterListAcceptor */
  //@{

  /** \brief . */
  void setParameterList(Teuchos::RCP<Teuchos::ParameterList> const& paramList);
  /** \brief . */
  Teuchos::RCP<Teuchos::ParameterList> getNonconstParameterList();
  /** \brief . */
  Teuchos::RCP<Teuchos::ParameterList> unsetParameterList();
  /** \brief . */
  Teuchos::RCP<const Teuchos::ParameterList> getParameterList() const;
  /** \brief . */
  Teuchos::RCP<const Teuchos::ParameterList> getValidParameters() const;

  //@}

  /** \name Public functions overridden from ModelEvaulator. */
  //@{

  /** \brief . */
  int Np() const;
  /** \brief . */
  int Ng() const;
  /** \brief . */
  Teuchos::RCP<const Thyra::VectorSpaceBase<ST> > get_x_space() const;
  /** \brief . */
  Teuchos::RCP<const Thyra::VectorSpaceBase<ST> > get_f_space() const;
  /** \brief . */
  Teuchos::RCP<const Thyra::VectorSpaceBase<ST> > get_p_space(int l) const;
  /** \brief . */
  Teuchos::RCP<const Teuchos::Array<std::string> > get_p_names(int l) const;
  /** \brief . */
  Teuchos::RCP<const Thyra::VectorSpaceBase<ST> > get_g_space(int j) const;
  /** \brief . */
  Thyra::ModelEvaluatorBase::InArgs<ST> getNominalValues() const;
  /** \brief . */
  Thyra::ModelEvaluatorBase::InArgs<ST> getLowerBounds() const;
  /** \brief . */
  Thyra::ModelEvaluatorBase::InArgs<ST> getUpperBounds() const;
  /** \brief . */
  Teuchos::RCP<Thyra::LinearOpBase<ST> > create_W_op() const;
  /** \brief Returns null currently. */
  Teuchos::RCP<Thyra::PreconditionerBase<ST> > create_W_prec() const;
  /** \breif . */
  Teuchos::RCP<const Thyra::LinearOpWithSolveFactoryBase<ST> > get_W_factory() const;
  /** \brief . */
  Thyra::ModelEvaluatorBase::InArgs<ST> createInArgs() const;
  /** \brief . */
  void reportFinalPoint(
    const Thyra::ModelEvaluatorBase::InArgs<ST>      &finalPoint
    ,const bool                                   wasSolved
    );

  //@}

  // Made public to simplify implementation but this is harmless to be public.
  // Clients should not deal with this type.
  enum EStateFunctionScaling { STATE_FUNC_SCALING_NONE, STATE_FUNC_SCALING_ROW_SUM };

private:

  /** \name Private functions overridden from ModelEvaulatorDefaultBase. */
  //@{

  /** \brief . */
  Teuchos::RCP<Thyra::LinearOpBase<ST> > create_DfDp_op_impl(int l) const;
  /** \brief . */
  Teuchos::RCP<Thyra::LinearOpBase<ST> > create_DgDx_dot_op_impl(int j) const;
  /** \brief . */
  Teuchos::RCP<Thyra::LinearOpBase<ST> > create_DgDx_op_impl(int j) const;
  /** \brief . */
  Teuchos::RCP<Thyra::LinearOpBase<ST> > create_DgDp_op_impl(int j, int l) const;
  /** \brief . */
  Thyra::ModelEvaluatorBase::OutArgs<ST> createOutArgsImpl() const;
  /** \brief . */
  void evalModelImpl(
    const Thyra::ModelEvaluatorBase::InArgs<ST> &inArgs,
    const Thyra::ModelEvaluatorBase::OutArgs<ST> &outArgs
    ) const;

  //@}

private:

  // ////////////////////
  // Private types

  typedef Teuchos::Array<Teuchos::RCP<const Epetra_Map> > p_map_t;
  typedef Teuchos::Array<Teuchos::RCP<const Epetra_Map> > g_map_t;
  typedef std::vector<bool> p_map_is_local_t;
  typedef std::vector<bool> g_map_is_local_t;

  typedef Teuchos::Array<Teuchos::RCP<const Thyra::VectorSpaceBase<ST> > >
  p_space_t;
  typedef Teuchos::Array<Teuchos::RCP<const Thyra::VectorSpaceBase<ST> > >
  g_space_t;

  // /////////////////////
  // Private data members

  Teuchos::RCP<const EpetraExt::ModelEvaluator> epetraModel_;

  Teuchos::RCP<Teuchos::ParameterList> paramList_;

  Teuchos::RCP<Thyra::LinearOpWithSolveFactoryBase<ST> > W_factory_;

  Teuchos::RCP<const Epetra_Map> x_map_;
  p_map_t p_map_;
  g_map_t g_map_;
  p_map_is_local_t p_map_is_local_;
  p_map_is_local_t g_map_is_local_;
  Teuchos::RCP<const Epetra_Map> f_map_;

  Teuchos::RCP<const Thyra::VectorSpaceBase<ST> > x_space_;
  p_space_t p_space_;
  Teuchos::RCP<const Thyra::VectorSpaceBase<ST> > f_space_;
  g_space_t g_space_;

  mutable Thyra::ModelEvaluatorBase::InArgs<ST> nominalValues_;
  mutable Thyra::ModelEvaluatorBase::InArgs<ST> lowerBounds_;
  mutable Thyra::ModelEvaluatorBase::InArgs<ST> upperBounds_;
  mutable bool nominalValuesAndBoundsAreUpdated_;

  Thyra::ModelEvaluatorBase::InArgs<ST> finalPoint_;

  EStateFunctionScaling stateFunctionScaling_;
  mutable Teuchos::RCP<const Epetra_Vector> stateFunctionScalingVec_;

  Teuchos::RCP<const Epetra_Vector> stateVariableScalingVec_; // S_x
  mutable Teuchos::RCP<const Epetra_Vector> invStateVariableScalingVec_; // inv(S_x)
  mutable EpetraExt::ModelEvaluator::InArgs epetraInArgsScaling_;
  mutable EpetraExt::ModelEvaluator::OutArgs epetraOutArgsScaling_;
  
  mutable Teuchos::RCP<Epetra_Vector> x_unscaled_;
  mutable Teuchos::RCP<Epetra_Vector> x_dot_unscaled_;

  mutable Thyra::ModelEvaluatorBase::InArgs<ST> prototypeInArgs_;
  mutable Thyra::ModelEvaluatorBase::OutArgs<ST> prototypeOutArgs_;
  mutable bool currentInArgsOutArgs_;

  bool finalPointWasSolved_;

  // Share the outArgs between the resize and evaluate functions
  mutable EpetraExt::ModelEvaluator::OutArgs evaluated_epetraUnscaledOutArgs;

  // //////////////////////////
  // Private member functions

  /** \brief . */
  void convertInArgsFromEpetraToThyra(
    const EpetraExt::ModelEvaluator::InArgs &epetraInArgs,
    Thyra::ModelEvaluatorBase::InArgs<ST> *inArgs
    ) const;

  /** \brief . */
  void convertInArgsFromThyraToEpetra(
    const Thyra::ModelEvaluatorBase::InArgs<ST> &inArgs,
    EpetraExt::ModelEvaluator::InArgs *epetraInArgs
    ) const;

  /** \brief . */
  void convertOutArgsFromThyraToEpetra(
    // Thyra form of the outArgs
    const Thyra::ModelEvaluatorBase::OutArgs<ST> &outArgs,
    // Epetra form of the unscaled output arguments 
    EpetraExt::ModelEvaluator::OutArgs *epetraUnscaledOutArgs,
    // The passed-in form of W
    Teuchos::RCP<Thyra::LinearOpBase<ST> > *W_op,
    Teuchos::RCP<Thyra::EpetraLinearOp> *efwdW,
    // The actual Epetra object passed to the underylying EpetraExt::ModelEvaluator
    Teuchos::RCP<Epetra_Operator> *eW
    ) const;

  /** \brief . */
  void preEvalScalingSetup(
    EpetraExt::ModelEvaluator::InArgs *epetraInArgs,
    EpetraExt::ModelEvaluator::OutArgs *epetraUnscaledOutArgs,
    const Teuchos::RCP<Teuchos::FancyOStream> &out,
    const Teuchos::EVerbosityLevel verbLevel
    ) const;

  /** \brief . */
  void postEvalScalingSetup(
    const EpetraExt::ModelEvaluator::OutArgs &epetraUnscaledOutArgs,
    const Teuchos::RCP<Teuchos::FancyOStream> &out,
    const Teuchos::EVerbosityLevel verbLevel
    ) const;

  /** \brief . */
  void finishConvertingOutArgsFromEpetraToThyra(
    const EpetraExt::ModelEvaluator::OutArgs &epetraOutArgs,
    Teuchos::RCP<Thyra::LinearOpBase<ST> > &W_op,
    Teuchos::RCP<Thyra::EpetraLinearOp> &efwdW,
    Teuchos::RCP<Epetra_Operator> &eW,
    const Thyra::ModelEvaluatorBase::OutArgs<ST> &outArgs // Output!
    ) const;
  // 2007/08/03: rabartl: Above, I pass many of the RCP objects by non-const
  // reference since I don't want the compiler to perform any implicit
  // conversions on this RCP objects.

  /** \brief . */
  void updateNominalValuesAndBounds() const;

  /** \brief . */
  void updateInArgsOutArgs() const;

  /** \brief . */
  Teuchos::RCP<Thyra::EpetraLinearOp> create_epetra_W_op() const;
  
};



} // namespace AAdapt


#endif // THYRA_ADAPTIVE_MODEL_EVALUATOR_HPP
