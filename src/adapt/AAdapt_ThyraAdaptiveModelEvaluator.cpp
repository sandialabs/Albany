//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "AAdapt_ThyraAdaptiveModelEvaluator.hpp"

#include "Thyra_EpetraModelEvaluator.hpp"
#include "Thyra_LinearOpWithSolveFactoryHelpers.hpp"
#include "Thyra_EpetraThyraWrappers.hpp"
#include "Thyra_EpetraLinearOp.hpp"
#include "Thyra_DetachedMultiVectorView.hpp"
#include "Thyra_ModelEvaluatorDelegatorBase.hpp" // Gives verbose macros!
#include "EpetraExt_ModelEvaluatorScalingTools.h"
#include "Epetra_RowMatrix.h"
#include "Teuchos_Time.hpp"
#include "Teuchos_implicit_cast.hpp"
#include "Teuchos_Assert.hpp"
#include "Teuchos_StandardParameterEntryValidators.hpp"
#include "Teuchos_VerboseObjectParameterListHelpers.hpp"


namespace {


const std::string StateFunctionScaling_name = "State Function Scaling";
Teuchos::RCP<
  Teuchos::StringToIntegralParameterEntryValidator<
    AAdapt::ThyraAdaptiveModelEvaluator::EStateFunctionScaling
    >
  >
stateFunctionScalingValidator;
const std::string StateFunctionScaling_default = "None";


// Extract out the Epetra_RowMatrix from the set W in an Epetra OutArgs object
Teuchos::RCP<Epetra_RowMatrix>
get_Epetra_RowMatrix(
  const EpetraExt::ModelEvaluator::OutArgs &epetraOutArgs
  )
{
  using Teuchos::RCP;
  const RCP<Epetra_Operator>
    eW = epetraOutArgs.get_W();
  const RCP<Epetra_RowMatrix>
    ermW = Teuchos::rcp_dynamic_cast<Epetra_RowMatrix>(eW,false);
  TEUCHOS_TEST_FOR_EXCEPTION(
    is_null(ermW), std::logic_error,
    "AAdapt::ThyraAdaptiveModelEvaluator::evalModel(...): Error, if\n"
    "scaling is turned on, the underlying Epetra_Operator created\n"
    "an initialized by the underlying epetra model evaluator\n"
    "\"" << epetraOutArgs.modelEvalDescription() << "\"\n"
    "must support the Epetra_RowMatrix interface through a dynamic cast.\n"
    "The concrete type " << Teuchos::typeName(*eW) << " does not support\n"
    "Epetra_RowMatrix!"
    );
  return ermW;
}


Teuchos::RCP<Epetra_Operator>
create_and_assert_W( 
  const EpetraExt::ModelEvaluator &epetraModel
  )
{
  using Teuchos::RCP;
  RCP<Epetra_Operator>
    eW = epetraModel.create_W();
  TEUCHOS_TEST_FOR_EXCEPTION(
    is_null(eW), std::logic_error,
    "Error, the call to create_W() returned null on the "
    "EpetraExt::ModelEvaluator object "
    "\"" << epetraModel.description() << "\".  This may mean that "
    "the underlying model does not support more than one copy of "
    "W at one time!" );
  return eW;
}


} // namespace


namespace AAdapt {

using Teuchos::RCP;
using Thyra::LinearOpWithSolveFactoryBase;
using Thyra::LinearOpBase;
using Thyra::create_VectorSpace;
using Thyra::ModelEvaluatorBase;
using Thyra::VectorSpaceBase;
using Thyra::VectorBase;
using Thyra::PreconditionerBase;
using Thyra::EpetraLinearOp;
using Thyra::get_Epetra_Vector;
using Thyra::convert;

// Constructors/initializers/accessors.


ThyraAdaptiveModelEvaluator::ThyraAdaptiveModelEvaluator()
  :nominalValuesAndBoundsAreUpdated_(false), stateFunctionScaling_(STATE_FUNC_SCALING_NONE),
   currentInArgsOutArgs_(false), finalPointWasSolved_(false)
{}


ThyraAdaptiveModelEvaluator::ThyraAdaptiveModelEvaluator(
  const RCP<const EpetraExt::ModelEvaluator> &epetraModel,
  const RCP<LinearOpWithSolveFactoryBase<ST> > &W_factory
  )
  :nominalValuesAndBoundsAreUpdated_(false), stateFunctionScaling_(STATE_FUNC_SCALING_NONE),
   currentInArgsOutArgs_(false), finalPointWasSolved_(false)
{
  initialize(epetraModel,W_factory);
}


void ThyraAdaptiveModelEvaluator::initialize(
  const RCP<const EpetraExt::ModelEvaluator> &epetraModel,
  const RCP<LinearOpWithSolveFactoryBase<ST> > &W_factory
  )
{
  using Teuchos::implicit_cast;
  typedef ModelEvaluatorBase MEB;
  //
  epetraModel_ = epetraModel;
  //
  W_factory_ = W_factory;
  //
  x_map_ = epetraModel_->get_x_map();
  f_map_ = epetraModel_->get_f_map();
  if (!is_null(x_map_)) {
    x_space_ = create_VectorSpace(x_map_);
    f_space_ = create_VectorSpace(f_map_);
  }
  //
  EpetraExt::ModelEvaluator::InArgs inArgs = epetraModel_->createInArgs();
  p_map_.resize(inArgs.Np()); p_space_.resize(inArgs.Np());
  p_map_is_local_.resize(inArgs.Np(),false);
  for( int l = 0; l < implicit_cast<int>(p_space_.size()); ++l ) {
    RCP<const Epetra_Map>
      p_map_l = ( p_map_[l] = epetraModel_->get_p_map(l) );
#ifdef TEUCHOS_DEBUG
    TEUCHOS_TEST_FOR_EXCEPTION(
      is_null(p_map_l), std::logic_error,
      "Error, the the map p["<<l<<"] for the model \""
      <<epetraModel->description()<<"\" can not be null!");
#endif

    p_map_is_local_[l] = !p_map_l->DistributedGlobal();
    p_space_[l] = create_VectorSpace(p_map_l);
  }
  //
  EpetraExt::ModelEvaluator::OutArgs outArgs = epetraModel_->createOutArgs();
  g_map_.resize(outArgs.Ng()); g_space_.resize(outArgs.Ng());
  g_map_is_local_.resize(outArgs.Ng(),false);
  for( int j = 0; j < implicit_cast<int>(g_space_.size()); ++j ) {
    RCP<const Epetra_Map>
      g_map_j = ( g_map_[j] = epetraModel_->get_g_map(j) );
    g_map_is_local_[j] = !g_map_j->DistributedGlobal();
    g_space_[j] = create_VectorSpace( g_map_j );
  }
  //
  epetraInArgsScaling_ = epetraModel_->createInArgs();
  epetraOutArgsScaling_ = epetraModel_->createOutArgs();
  nominalValuesAndBoundsAreUpdated_ = false;
  finalPointWasSolved_ = false;
  stateFunctionScalingVec_ = Teuchos::null; // Must set new scaling!
  //
  currentInArgsOutArgs_ = false;
}


RCP<const EpetraExt::ModelEvaluator>
ThyraAdaptiveModelEvaluator::getEpetraModel() const
{
  return epetraModel_;
}


void ThyraAdaptiveModelEvaluator::setNominalValues(
  const ModelEvaluatorBase::InArgs<ST>& nominalValues
 )
{
  nominalValues_.setArgs(nominalValues);
  // Note: These must be the scaled values so we don't need to scale!
}


void ThyraAdaptiveModelEvaluator::setStateVariableScalingVec(
  const RCP<const Epetra_Vector> &stateVariableScalingVec
  )
{
  typedef ModelEvaluatorBase MEB;
#ifdef TEUCHOS_DEBUG
  TEUCHOS_TEST_FOR_EXCEPT( !this->createInArgs().supports(MEB::IN_ARG_x) );
#endif  
  stateVariableScalingVec_ = stateVariableScalingVec.assert_not_null();
  invStateVariableScalingVec_ = Teuchos::null;
  nominalValuesAndBoundsAreUpdated_ = false;
}


RCP<const Epetra_Vector>
ThyraAdaptiveModelEvaluator::getStateVariableScalingVec() const
{
  return stateVariableScalingVec_;
}


RCP<const Epetra_Vector>
ThyraAdaptiveModelEvaluator::getStateVariableInvScalingVec() const
{
  updateNominalValuesAndBounds();
  return invStateVariableScalingVec_;
}


void ThyraAdaptiveModelEvaluator::setStateFunctionScalingVec(
  const RCP<const Epetra_Vector> &stateFunctionScalingVec
  )
{
  stateFunctionScalingVec_ = stateFunctionScalingVec;
}


RCP<const Epetra_Vector>
ThyraAdaptiveModelEvaluator::getStateFunctionScalingVec() const
{
  return stateFunctionScalingVec_;
}


void ThyraAdaptiveModelEvaluator::uninitialize(
  RCP<const EpetraExt::ModelEvaluator> *epetraModel,
  RCP<LinearOpWithSolveFactoryBase<ST> > *W_factory
  )
{
  if(epetraModel) *epetraModel = epetraModel_;
  if(W_factory) *W_factory = W_factory_;
  epetraModel_ = Teuchos::null;
  W_factory_ = Teuchos::null;
  stateFunctionScalingVec_ = Teuchos::null;
  stateVariableScalingVec_ = Teuchos::null;
  invStateVariableScalingVec_ = Teuchos::null;
  currentInArgsOutArgs_ = false;
}


const ModelEvaluatorBase::InArgs<ST>&
ThyraAdaptiveModelEvaluator::getFinalPoint() const
{
  return finalPoint_;
}


bool ThyraAdaptiveModelEvaluator::finalPointWasSolved() const
{
  return finalPointWasSolved_;
}

const Teuchos::RCP<Thyra::VectorBase<ST> >
ThyraAdaptiveModelEvaluator::resize_g_space(int index, Teuchos::RCP<const Epetra_Map> map){

    RCP<const Epetra_Map>
      g_map_j = ( g_map_[index] = map );
    g_map_is_local_[index] = !g_map_j->DistributedGlobal();
    g_space_[index] = create_VectorSpace( g_map_j );
    const Teuchos::RCP<Thyra::VectorBase<ST> > g_j = Thyra::createMember(*g_space_[index]);

    RCP<Epetra_Vector> davector = get_Epetra_Vector(*g_map_[index], g_j);

    // replace the vector in the outArgs being used in epetraModel->evalModel
    evaluated_epetraUnscaledOutArgs.set_g(index, davector);

    return g_j;

/* Ordinarily, we would update the maps, but Piro_Epetra_ME returns NULL maps.
    x_map_ = epetraModel_->get_x_map();
    f_map_ = epetraModel_->get_f_map();
    if (!is_null(x_map_)) {
      x_space_ = create_VectorSpace(x_map_);
      f_space_ = create_VectorSpace(f_map_);
    }
*/

}


// Public functions overridden from Teuchos::Describable


std::string ThyraAdaptiveModelEvaluator::description() const
{
  std::ostringstream oss;
  oss << "Thyra::ThyraAdaptiveModelEvaluator{";
  oss << "epetraModel=";
  if(epetraModel_.get())
    oss << "\'"<<epetraModel_->description()<<"\'";
  else
    oss << "NULL";
  oss << ",W_factory=";
  if(W_factory_.get())
    oss << "\'"<<W_factory_->description()<<"\'";
  else
    oss << "NULL";
  oss << "}";
  return oss.str();
}


// Overridden from Teuchos::ParameterListAcceptor


void ThyraAdaptiveModelEvaluator::setParameterList(
  RCP<Teuchos::ParameterList> const& paramList
  )
{
  TEUCHOS_TEST_FOR_EXCEPT(is_null(paramList));
  paramList->validateParameters(*getValidParameters(),0); // Just validate my params
  paramList_ = paramList;
  const EStateFunctionScaling stateFunctionScaling_old = stateFunctionScaling_; 
  stateFunctionScaling_ = stateFunctionScalingValidator->getIntegralValue(
    *paramList_, StateFunctionScaling_name, StateFunctionScaling_default
    );
  if( stateFunctionScaling_ != stateFunctionScaling_old )
    stateFunctionScalingVec_ = Teuchos::null;
  Teuchos::readVerboseObjectSublist(&*paramList_,this);
#ifdef TEUCHOS_DEBUG
  paramList_->validateParameters(*getValidParameters(),0);
#endif // TEUCHOS_DEBUG
}


RCP<Teuchos::ParameterList>
ThyraAdaptiveModelEvaluator::getNonconstParameterList()
{
  return paramList_;
}


RCP<Teuchos::ParameterList>
ThyraAdaptiveModelEvaluator::unsetParameterList()
{
  RCP<Teuchos::ParameterList> _paramList = paramList_;
  paramList_ = Teuchos::null;
  return _paramList;
}


RCP<const Teuchos::ParameterList>
ThyraAdaptiveModelEvaluator::getParameterList() const
{
  return paramList_;
}


RCP<const Teuchos::ParameterList>
ThyraAdaptiveModelEvaluator::getValidParameters() const
{
  using Teuchos::rcp;
  using Teuchos::StringToIntegralParameterEntryValidator;
  using Teuchos::tuple;
  using Teuchos::rcp_implicit_cast;
  typedef Teuchos::ParameterEntryValidator PEV;
  static RCP<const Teuchos::ParameterList> validPL;
  if(is_null(validPL)) {
    RCP<Teuchos::ParameterList>
      pl = Teuchos::rcp(new Teuchos::ParameterList());
    stateFunctionScalingValidator = rcp(
      new StringToIntegralParameterEntryValidator<EStateFunctionScaling>(
        tuple<std::string>(
          "None",
          "Row Sum"
          ),
        tuple<std::string>(
          "Do not scale the state function f(...) in this class.",

          "Scale the state function f(...) and all its derivatives\n"
          "using the row sum scaling from the initial Jacobian\n"
          "W=d(f)/d(x).  Note, this only works with Epetra_CrsMatrix\n"
          "currently."
          ),
        tuple<EStateFunctionScaling>(
          STATE_FUNC_SCALING_NONE,
          STATE_FUNC_SCALING_ROW_SUM
          ),
        StateFunctionScaling_name
        )
      );
    pl->set(StateFunctionScaling_name,StateFunctionScaling_default,
      "Determines if and how the state function f(...) and all of its\n"
      "derivatives are scaled.  The scaling is done explicitly so there should\n"
      "be no impact on the meaning of inner products or tolerances for\n"
      "linear solves.",
      rcp_implicit_cast<const PEV>(stateFunctionScalingValidator)
      );
    Teuchos::setupVerboseObjectSublist(&*pl);
    validPL = pl;
  }
  return validPL;
}


// Overridden from ModelEvaulator.


int ThyraAdaptiveModelEvaluator::Np() const
{
  return p_space_.size();
}


int ThyraAdaptiveModelEvaluator::Ng() const
{
  return g_space_.size();
}


RCP<const VectorSpaceBase<ST> >
ThyraAdaptiveModelEvaluator::get_x_space() const
{
  return x_space_;
}


RCP<const VectorSpaceBase<ST> >
ThyraAdaptiveModelEvaluator::get_f_space() const
{
  return f_space_;
}


RCP<const VectorSpaceBase<ST> >
ThyraAdaptiveModelEvaluator::get_p_space(int l) const
{
#ifdef TEUCHOS_DEBUG
  TEUCHOS_ASSERT_IN_RANGE_UPPER_EXCLUSIVE( l, 0, this->Np() );
#endif
  return p_space_[l];
}


RCP<const Teuchos::Array<std::string> >
ThyraAdaptiveModelEvaluator::get_p_names(int l) const
{
#ifdef TEUCHOS_DEBUG
  TEUCHOS_ASSERT_IN_RANGE_UPPER_EXCLUSIVE( l, 0, this->Np() );
#endif
  return epetraModel_->get_p_names(l);
}


RCP<const VectorSpaceBase<ST> >
ThyraAdaptiveModelEvaluator::get_g_space(int j) const
{
  TEUCHOS_TEST_FOR_EXCEPT( ! ( 0 <= j && j < this->Ng() ) );
  return g_space_[j];
}


ModelEvaluatorBase::InArgs<ST>
ThyraAdaptiveModelEvaluator::getNominalValues() const
{
  updateNominalValuesAndBounds();
  return nominalValues_;
}


ModelEvaluatorBase::InArgs<ST>
ThyraAdaptiveModelEvaluator::getLowerBounds() const
{
  updateNominalValuesAndBounds();
  return lowerBounds_;
}


ModelEvaluatorBase::InArgs<ST>
ThyraAdaptiveModelEvaluator::getUpperBounds() const
{
  updateNominalValuesAndBounds();
  return upperBounds_;
}


RCP<LinearOpBase<ST> >
ThyraAdaptiveModelEvaluator::create_W_op() const
{
  return this->create_epetra_W_op();
}


RCP<PreconditionerBase<ST> >
ThyraAdaptiveModelEvaluator::create_W_prec() const
{
  return Teuchos::null;
}


RCP<const LinearOpWithSolveFactoryBase<ST> >
ThyraAdaptiveModelEvaluator::get_W_factory() const
{
  return W_factory_;
}


ModelEvaluatorBase::InArgs<ST> ThyraAdaptiveModelEvaluator::createInArgs() const
{
  if (!currentInArgsOutArgs_)
    updateInArgsOutArgs();
  return prototypeInArgs_;
}


void ThyraAdaptiveModelEvaluator::reportFinalPoint(
  const ModelEvaluatorBase::InArgs<ST> &finalPoint,
  const bool wasSolved
  )
{
  finalPoint_ = this->createInArgs();
  finalPoint_.setArgs(finalPoint);
  finalPointWasSolved_ = wasSolved;
}


// Private functions overridden from ModelEvaulatorDefaultBase


RCP<LinearOpBase<ST> >
ThyraAdaptiveModelEvaluator::create_DfDp_op_impl(int l) const
{
  TEUCHOS_TEST_FOR_EXCEPT(true);
  return Teuchos::null;
}


RCP<LinearOpBase<ST> >
ThyraAdaptiveModelEvaluator::create_DgDx_dot_op_impl(int j) const
{
  TEUCHOS_TEST_FOR_EXCEPT(true);
  return Teuchos::null;
}


RCP<LinearOpBase<ST> >
ThyraAdaptiveModelEvaluator::create_DgDx_op_impl(int j) const
{
  TEUCHOS_TEST_FOR_EXCEPT(true);
  return Teuchos::null;
}


RCP<LinearOpBase<ST> >
ThyraAdaptiveModelEvaluator::create_DgDp_op_impl( int j, int l ) const
{
  TEUCHOS_TEST_FOR_EXCEPT(true);
  return Teuchos::null;
}


ModelEvaluatorBase::OutArgs<ST>
ThyraAdaptiveModelEvaluator::createOutArgsImpl() const
{
  if (!currentInArgsOutArgs_)
    updateInArgsOutArgs();
  return prototypeOutArgs_;
}


void ThyraAdaptiveModelEvaluator::evalModelImpl(
  const ModelEvaluatorBase::InArgs<ST>& inArgs_in,
  const ModelEvaluatorBase::OutArgs<ST>& outArgs
  ) const
{

  using Teuchos::rcp;
  using Teuchos::rcp_const_cast;
  using Teuchos::rcp_dynamic_cast;
  using Teuchos::OSTab;
  using Teuchos::includesVerbLevel;
  typedef EpetraExt::ModelEvaluator EME;

  //
  // A) Initial setup
  //

  // Make sure that we are fully initialized!
  this->updateNominalValuesAndBounds();

  // Make sure we grab the initial guess first!
  InArgs<ST> inArgs = this->getNominalValues();
  // Now, copy the parameters from the input inArgs_in object to the inArgs
  // object.  Any input objects that are set in inArgs_in will overwrite those
  // in inArgs that will already contain the nominal values.  This will insure
  // that all input parameters are set and those that are not set by the
  // client will be at their nominal values (as defined by the underlying
  // EpetraExt::ModelEvaluator object).  The full set of Thyra input arguments
  // must be set before these can be translated into Epetra input arguments.
  inArgs.setArgs(inArgs_in);

  // This is a special exception: see evalModel() in Thyra::ME
  // documentation.  If inArgs() supports x_dot but the evaluate call
  // passes in a null value, then we need to make sure the null value
  // gets passed on instead of the nominal value.
  if (inArgs.supports(Thyra::ModelEvaluatorBase::IN_ARG_x_dot)) {
    if (is_null(inArgs_in.get_x_dot()))
      inArgs.set_x_dot(Teuchos::null);
  }

  // Print the header and the values of the inArgs and outArgs objects!
  typedef ST Scalar; // Needed for below macro!
  THYRA_MODEL_EVALUATOR_DECORATOR_EVAL_MODEL_GEN_BEGIN(
    "Thyra::ThyraAdaptiveModelEvaluator",inArgs,outArgs,Teuchos::null
    );

  // State function Scaling
  const bool firstTimeStateFuncScaling
    = (
      stateFunctionScaling_ != STATE_FUNC_SCALING_NONE
      && is_null(stateFunctionScalingVec_)
      );
  
  typedef Teuchos::VerboseObjectTempState<LinearOpWithSolveFactoryBase<ST> > VOTSLOWSF;
  VOTSLOWSF W_factory_outputTempState(W_factory_,out,verbLevel);

  Teuchos::Time timer("");

  //
  // B) Prepressess the InArgs and OutArgs in preparation to call
  // the underlying EpetraExt::ModelEvaluator
  //

  //
  // B.1) Translate InArgs from Thyra to Epetra objects
  //
  
  if (out.get() && includesVerbLevel(verbLevel,Teuchos::VERB_LOW))
    *out << "\nSetting-up/creating input arguments ...\n";
  timer.start(true);

  // Unwrap input Thyra objects to get Epetra objects
  EME::InArgs epetraScaledInArgs = epetraModel_->createInArgs();
  convertInArgsFromThyraToEpetra( inArgs, &epetraScaledInArgs );

  // Unscale the input Epetra objects which will be passed to the underlying
  // EpetraExt::ModelEvaluator object.
  EME::InArgs epetraInArgs = epetraModel_->createInArgs();
  EpetraExt::unscaleModelVars(
    epetraScaledInArgs, epetraInArgsScaling_, &epetraInArgs,
    out.get(), verbLevel
    );

  timer.stop();
  if (out.get() && includesVerbLevel(verbLevel,Teuchos::VERB_LOW))
    OSTab(out).o() << "\nTime to setup InArgs = "<<timer.totalElapsedTime()<<" sec\n";

  //
  // B.2) Convert from Thyra to Epetra OutArgs
  //
  
  if (out.get() && includesVerbLevel(verbLevel,Teuchos::VERB_LOW))
    *out << "\nSetting-up/creating output arguments ...\n";
  timer.start(true);
  
  // The unscaled Epetra OutArgs that will be passed to the
  // underlying EpetraExt::ModelEvaluator object
  // This is a private class member to allow resizing the solution data array
  evaluated_epetraUnscaledOutArgs = epetraModel_->createOutArgs();

  // Various objects that are needed later (see documentation in
  // the function convertOutArgsFromThyraToEpetra(...)
  RCP<LinearOpBase<ST> > W_op;
  RCP<EpetraLinearOp> efwdW;
  RCP<Epetra_Operator> eW;
  
  // Convert from Thyra to Epetra OutArgs and grap some of the intermediate
  // objects accessed along the way that are needed later.
  convertOutArgsFromThyraToEpetra(
    outArgs,
    &evaluated_epetraUnscaledOutArgs,
    &W_op, &efwdW, &eW
    );

  //
  // B.3) Setup OutArgs to computing scaling if needed
  //

  if (firstTimeStateFuncScaling) {
    preEvalScalingSetup(&epetraInArgs,&evaluated_epetraUnscaledOutArgs,out,verbLevel);
  }

  timer.stop();
  if (out.get() && includesVerbLevel(verbLevel,Teuchos::VERB_LOW))
    OSTab(out).o()
      << "\nTime to setup OutArgs = "
      << timer.totalElapsedTime() <<" sec\n";

  //
  // C) Evaluate the underlying EpetraExt model to compute the Epetra outputs
  //
  
  if (out.get() && includesVerbLevel(verbLevel,Teuchos::VERB_LOW))
    *out << "\nEvaluating the Epetra output functions ...\n";
  timer.start(true);

  epetraModel_->evalModel(epetraInArgs, evaluated_epetraUnscaledOutArgs);

  timer.stop();
  if (out.get() && includesVerbLevel(verbLevel,Teuchos::VERB_LOW))
    OSTab(out).o()
      << "\nTime to evaluate Epetra output functions = "
      << timer.totalElapsedTime() <<" sec\n";

  //
  // D) Postprocess the output objects
  //

  //
  // D.1) Compute the scaling factors if needed
  //
  
  if (out.get() && includesVerbLevel(verbLevel,Teuchos::VERB_LOW))
    *out << "\nCompute scale factors if needed ...\n";
  timer.start(true);

  if (firstTimeStateFuncScaling) {
    postEvalScalingSetup(evaluated_epetraUnscaledOutArgs,out,verbLevel);
  }

  timer.stop();
  if (out.get() && includesVerbLevel(verbLevel,Teuchos::VERB_LOW))
    OSTab(out).o()
      << "\nTime to compute scale factors = "
      << timer.totalElapsedTime() <<" sec\n";

  //
  // D.2) Scale the output Epetra objects
  //

  if (out.get() && includesVerbLevel(verbLevel,Teuchos::VERB_LOW))
    *out << "\nScale the output objects ...\n";
  timer.start(true);

  EME::OutArgs epetraOutArgs = epetraModel_->createOutArgs();
  bool allFuncsWhereScaled = false;
  EpetraExt::scaleModelFuncs(
    evaluated_epetraUnscaledOutArgs, epetraInArgsScaling_, epetraOutArgsScaling_,
    &epetraOutArgs, &allFuncsWhereScaled,
    out.get(), verbLevel
    );
  TEUCHOS_TEST_FOR_EXCEPTION(
    !allFuncsWhereScaled, std::logic_error,
    "Error, we can not currently handle epetra output objects that could not be"
    " scaled.  Special code will have to be added to handle this (i.e. using"
    " implicit diagonal and multiplied linear operators to implicitly do"
    " the scaling."
    );

  timer.stop();
  if (out.get() && includesVerbLevel(verbLevel,Teuchos::VERB_LOW))
    OSTab(out).o()
      << "\nTime to scale the output objects = "
      << timer.totalElapsedTime() << " sec\n";

  //
  // D.3) Convert any Epetra objects to Thyra OutArgs objects that still need to
  // be converted
  //

  if (out.get() && includesVerbLevel(verbLevel,Teuchos::VERB_LOW))
    *out << "\nFinish processing and wrapping the output objects ...\n";
  timer.start(true);

  finishConvertingOutArgsFromEpetraToThyra(
    epetraOutArgs, W_op, efwdW, eW,
    outArgs
    );

  timer.stop();
  if (out.get() && includesVerbLevel(verbLevel,Teuchos::VERB_LOW))
    OSTab(out).o()
      << "\nTime to finish processing and wrapping the output objects = "
      << timer.totalElapsedTime() <<" sec\n";

  //
  // E) Print footer to end the function
  //

  THYRA_MODEL_EVALUATOR_DECORATOR_EVAL_MODEL_END();
  
}


// private


void ThyraAdaptiveModelEvaluator::convertInArgsFromEpetraToThyra(
  const EpetraExt::ModelEvaluator::InArgs &epetraInArgs,
  ModelEvaluatorBase::InArgs<ST> *inArgs
  ) const
{
  
  using Teuchos::implicit_cast;
  typedef ModelEvaluatorBase MEB;

  TEUCHOS_TEST_FOR_EXCEPT(!inArgs);

  if(inArgs->supports(MEB::IN_ARG_x)) {
    inArgs->set_x( create_Vector( epetraInArgs.get_x(), x_space_ ) );
  }
  
  if(inArgs->supports(MEB::IN_ARG_x_dot)) {
    inArgs->set_x_dot( create_Vector( epetraInArgs.get_x_dot(), x_space_ ) );
  }

  const int l_Np = inArgs->Np();
  for( int l = 0; l < l_Np; ++l ) {
    inArgs->set_p( l, create_Vector( epetraInArgs.get_p(l), p_space_[l] ) );
  }
  
  if(inArgs->supports(MEB::IN_ARG_t)) {
    inArgs->set_t(epetraInArgs.get_t());
  }
  
}


void ThyraAdaptiveModelEvaluator::convertInArgsFromThyraToEpetra(
  const ModelEvaluatorBase::InArgs<ST> &inArgs,
  EpetraExt::ModelEvaluator::InArgs *epetraInArgs
  ) const
{

  using Teuchos::rcp;
  using Teuchos::rcp_const_cast;
#ifdef HAVE_THYRA_ME_POLYNOMIAL
  using Teuchos::Polynomial;
#endif // HAVE_THYRA_ME_POLYNOMIAL


  TEUCHOS_TEST_FOR_EXCEPT(0==epetraInArgs);

  RCP<const VectorBase<ST> > x_dot;
  if( inArgs.supports(IN_ARG_x_dot) && (x_dot = inArgs.get_x_dot()).get() ) {
    RCP<const Epetra_Vector> e_x_dot = get_Epetra_Vector(*x_map_,x_dot);
    epetraInArgs->set_x_dot(e_x_dot);
  }

  RCP<const VectorBase<ST> > x;
  if( inArgs.supports(IN_ARG_x) && (x = inArgs.get_x()).get() ) {
    RCP<const Epetra_Vector> e_x = get_Epetra_Vector(*x_map_,x);
    epetraInArgs->set_x(e_x);
  }

  RCP<const VectorBase<ST> > p_l;
  for(int l = 0;  l < inArgs.Np(); ++l ) {
    p_l = inArgs.get_p(l);
    if(p_l.get()) epetraInArgs->set_p(l,get_Epetra_Vector(*p_map_[l],p_l));
  }

#ifdef HAVE_THYRA_ME_POLYNOMIAL

  RCP<const Polynomial< VectorBase<ST> > > x_dot_poly;
  RCP<Epetra_Vector> epetra_ptr;
  if(
    inArgs.supports(IN_ARG_x_dot_poly)
    && (x_dot_poly = inArgs.get_x_dot_poly()).get()
    )
  {
    RCP<Polynomial<Epetra_Vector> > epetra_x_dot_poly = 
      rcp(new Polynomial<Epetra_Vector>(x_dot_poly->degree()));
    for (unsigned int i=0; i<=x_dot_poly->degree(); i++) {
      epetra_ptr = rcp_const_cast<Epetra_Vector>(
        get_Epetra_Vector(*x_map_, x_dot_poly->getCoefficient(i)) );
      epetra_x_dot_poly->setCoefficientPtr(i,epetra_ptr);
    }
    epetraInArgs->set_x_dot_poly(epetra_x_dot_poly);
  }
  
  RCP<const Polynomial< VectorBase<ST> > > x_poly;
  if(
    inArgs.supports(IN_ARG_x_poly)
    && (x_poly = inArgs.get_x_poly()).get()
    )
  {
    RCP<Polynomial<Epetra_Vector> > epetra_x_poly = 
      rcp(new Polynomial<Epetra_Vector>(x_poly->degree()));
    for (unsigned int i=0; i<=x_poly->degree(); i++) {
      epetra_ptr = rcp_const_cast<Epetra_Vector>(
        get_Epetra_Vector(*x_map_, x_poly->getCoefficient(i)) );
      epetra_x_poly->setCoefficientPtr(i,epetra_ptr);
    }
    epetraInArgs->set_x_poly(epetra_x_poly);
  }

#endif // HAVE_THYRA_ME_POLYNOMIAL

  if( inArgs.supports(IN_ARG_t) )
    epetraInArgs->set_t(inArgs.get_t());
  
  if( inArgs.supports(IN_ARG_alpha) )
    epetraInArgs->set_alpha(inArgs.get_alpha());
  
  if( inArgs.supports(IN_ARG_beta) )
    epetraInArgs->set_beta(inArgs.get_beta());

}


void ThyraAdaptiveModelEvaluator::convertOutArgsFromThyraToEpetra(
  const ModelEvaluatorBase::OutArgs<ST> &outArgs,
  EpetraExt::ModelEvaluator::OutArgs *epetraUnscaledOutArgs_inout,
  RCP<LinearOpBase<ST> > *W_op_out,
  RCP<EpetraLinearOp> *efwdW_out,
  RCP<Epetra_Operator> *eW_out
  ) const
{

  using Teuchos::rcp;
  using Teuchos::rcp_const_cast;
  using Teuchos::rcp_dynamic_cast;
  using Teuchos::OSTab;
  using Teuchos::implicit_cast;
  using Thyra::get_Epetra_Vector;
  typedef EpetraExt::ModelEvaluator EME;

  // Assert input
#ifdef TEUCHOS_DEBUG
  TEUCHOS_ASSERT(epetraUnscaledOutArgs_inout);
  TEUCHOS_ASSERT(W_op_out);
  TEUCHOS_ASSERT(efwdW_out);
  TEUCHOS_ASSERT(eW_out);
#endif

  // Create easy to use references
  EpetraExt::ModelEvaluator::OutArgs &epetraUnscaledOutArgs = *epetraUnscaledOutArgs_inout;
  RCP<LinearOpBase<ST> > &W_op = *W_op_out;
  RCP<EpetraLinearOp> &efwdW = *efwdW_out;
  RCP<Epetra_Operator> &eW = *eW_out;

  // f
  { 
    RCP<VectorBase<ST> > f;
    if( outArgs.supports(OUT_ARG_f) && (f = outArgs.get_f()).get() )
      epetraUnscaledOutArgs.set_f(get_Epetra_Vector(*f_map_,f));
  }
    
  // g
  {
    RCP<VectorBase<ST> > g_j;
    for(int j = 0;  j < outArgs.Ng(); ++j ) {
      g_j = outArgs.get_g(j);
      if(g_j.get()) epetraUnscaledOutArgs.set_g(j,get_Epetra_Vector(*g_map_[j],g_j));
    }
  }
  
  // W_op
  {

    if (outArgs.supports(OUT_ARG_W_op) && nonnull(W_op = outArgs.get_W_op())) {
      if (nonnull(W_op) && is_null(efwdW)) {
        efwdW = rcp_const_cast<EpetraLinearOp>(
          rcp_dynamic_cast<const EpetraLinearOp>(W_op, true));
      }
    }
    
    if (nonnull(efwdW)) {
      // By the time we get here, if we have an object in efwdW, then it
      // should already be embeadded with an underlying Epetra_Operator object
      // that was allocated by the EpetraExt::ModelEvaluator object.
      // Therefore, we should just have to grab this object and be on our way.
      eW = efwdW->epetra_op();
      epetraUnscaledOutArgs.set_W(eW);
    }
    
    // Note: The following derivative objects update in place!

  }

  // DfDp
  {
    Derivative<ST> DfDp_l;
    for(int l = 0;  l < outArgs.Np(); ++l ) {
      if( !outArgs.supports(OUT_ARG_DfDp,l).none()
        && !(DfDp_l = outArgs.get_DfDp(l)).isEmpty() )
      {
        epetraUnscaledOutArgs.set_DfDp(l,convert(DfDp_l,f_map_,p_map_[l]));
      }
    }
  }

  // DgDx_dot
  {
    Derivative<ST> DgDx_dot_j;
    for(int j = 0;  j < outArgs.Ng(); ++j ) {
      if( !outArgs.supports(OUT_ARG_DgDx_dot,j).none()
        && !(DgDx_dot_j = outArgs.get_DgDx_dot(j)).isEmpty() )
      {
        epetraUnscaledOutArgs.set_DgDx_dot(j,convert(DgDx_dot_j,g_map_[j],x_map_));
      }
    }
  }

  // DgDx
  {
    Derivative<ST> DgDx_j;
    for(int j = 0;  j < outArgs.Ng(); ++j ) {
      if( !outArgs.supports(OUT_ARG_DgDx,j).none()
        && !(DgDx_j = outArgs.get_DgDx(j)).isEmpty() )
      {
        epetraUnscaledOutArgs.set_DgDx(j,convert(DgDx_j,g_map_[j],x_map_));
      }
    }
  }

  // DgDp
  {
    DerivativeSupport DgDp_j_l_support;
    Derivative<ST> DgDp_j_l;
    for (int j = 0;  j < outArgs.Ng(); ++j ) {
      for (int l = 0;  l < outArgs.Np(); ++l ) {
        if (!(DgDp_j_l_support = outArgs.supports(OUT_ARG_DgDp,j,l)).none()
          && !(DgDp_j_l = outArgs.get_DgDp(j,l)).isEmpty() )
        {
          epetraUnscaledOutArgs.set_DgDp(j,l,convert(DgDp_j_l,g_map_[j],p_map_[l]));
        }
      }
    }
  }

#ifdef HAVE_THYRA_ME_POLYNOMIAL

  // f_poly
  RCP<const Teuchos::Polynomial< VectorBase<ST> > > f_poly;
  if( outArgs.supports(OUT_ARG_f_poly) && (f_poly = outArgs.get_f_poly()).get() )
  {
    RCP<Teuchos::Polynomial<Epetra_Vector> > epetra_f_poly = 
      Teuchos::rcp(new Teuchos::Polynomial<Epetra_Vector>(f_poly->degree()));
    for (unsigned int i=0; i<=f_poly->degree(); i++) {
      RCP<Epetra_Vector> epetra_ptr
        = Teuchos::rcp_const_cast<Epetra_Vector>(get_Epetra_Vector(*f_map_,
            f_poly->getCoefficient(i)));
      epetra_f_poly->setCoefficientPtr(i,epetra_ptr);
    }
    epetraUnscaledOutArgs.set_f_poly(epetra_f_poly);
  }

#endif // HAVE_THYRA_ME_POLYNOMIAL

}


void ThyraAdaptiveModelEvaluator::preEvalScalingSetup(
  EpetraExt::ModelEvaluator::InArgs *epetraInArgs_inout,
  EpetraExt::ModelEvaluator::OutArgs *epetraUnscaledOutArgs_inout,
  const RCP<Teuchos::FancyOStream> &out,
  const Teuchos::EVerbosityLevel verbLevel
  ) const
{
  
  typedef EpetraExt::ModelEvaluator EME;
  
#ifdef TEUCHOS_DEBUG
  TEUCHOS_ASSERT(epetraInArgs_inout);
  TEUCHOS_ASSERT(epetraUnscaledOutArgs_inout);
#endif

  EpetraExt::ModelEvaluator::InArgs
    &epetraInArgs = *epetraInArgs_inout;
  EpetraExt::ModelEvaluator::OutArgs
    &epetraUnscaledOutArgs = *epetraUnscaledOutArgs_inout;

  if (
    ( stateFunctionScaling_ == STATE_FUNC_SCALING_ROW_SUM )
    &&
    (
      epetraUnscaledOutArgs.supports(EME::OUT_ARG_f) 
      &&
      epetraUnscaledOutArgs.funcOrDerivesAreSet(EME::OUT_ARG_f)
      )
    &&
    (
      epetraUnscaledOutArgs.supports(EME::OUT_ARG_W)
      &&
      is_null(epetraUnscaledOutArgs.get_W())
      )
    )
  {
    // This is the first pass through with scaling turned on and the client
    // turned on automatic scaling but did not ask for W.  We must compute W
    // in order to compute the scale factors so we must allocate a temporary W
    // just to compute the scale factors and then throw it away.  If the
    // client wants to evaluate W at the same point, then it should have
    // passed W in but that is not our problem here.  The ModelEvaluator
    // relies on the client to set up the calls to allow for efficient
    // evaluation.

    if(out.get() && verbLevel >= Teuchos::VERB_LOW)
      *out
        << "\nCreating a temporary Epetra W to compute scale factors"
        << " for f(...) ...\n";
    epetraUnscaledOutArgs.set_W(create_and_assert_W(*epetraModel_));
    if( epetraInArgs.supports(EME::IN_ARG_beta) )
      epetraInArgs.set_beta(1.0);
    if( epetraInArgs.supports(EME::IN_ARG_alpha) )
      epetraInArgs.set_alpha(0.0);
  }
  
}


void ThyraAdaptiveModelEvaluator::postEvalScalingSetup(
  const EpetraExt::ModelEvaluator::OutArgs &epetraUnscaledOutArgs,
  const RCP<Teuchos::FancyOStream> &out,
  const Teuchos::EVerbosityLevel verbLevel
  ) const
{

  using Teuchos::OSTab;
  using Teuchos::rcp;
  using Teuchos::rcp_const_cast;
  using Teuchos::includesVerbLevel;

  // Compute the scale factors for the state function f(...)
  switch(stateFunctionScaling_) {

    case STATE_FUNC_SCALING_ROW_SUM: {

      // Compute the inverse row-sum scaling from W

      const RCP<Epetra_RowMatrix>
        ermW = get_Epetra_RowMatrix(epetraUnscaledOutArgs);
      // Note: Above, we get the Epetra W object directly from the Epetra
      // OutArgs object since this might be a temporary matrix just to
      // compute scaling factors.  In this case, the stack funtion variable
      // eW might be empty!

      RCP<Epetra_Vector>
        invRowSums = rcp(new Epetra_Vector(ermW->OperatorRangeMap()));
      // Above: From the documentation is seems that the RangeMap should be
      // okay but who knows for sure!

      ermW->InvRowSums(*invRowSums);

      if (out.get() && includesVerbLevel(verbLevel,Teuchos::VERB_LOW)) {
        *out
          << "\nComputed inverse row sum scaling from W that"
          " will be used to scale f(...) and its derivatives:\n";
        ST minVal = 0, maxVal = 0, avgVal = 0;
        invRowSums->MinValue(&minVal);
        invRowSums->MaxValue(&maxVal);
        invRowSums->MeanValue(&avgVal);
        OSTab tab(out);
        *out
          << "min(invRowSums) = " << minVal << "\n"
          << "max(invRowSums) = " << maxVal << "\n"
          << "avg(invRowSums) = " << avgVal << "\n";
      }

      stateFunctionScalingVec_ = invRowSums;

      break;

    }

    default:
      TEUCHOS_TEST_FOR_EXCEPT("Should never get here!");

  }

  epetraOutArgsScaling_ = epetraModel_->createOutArgs();

  epetraOutArgsScaling_.set_f(
    rcp_const_cast<Epetra_Vector>(stateFunctionScalingVec_) );

}


void ThyraAdaptiveModelEvaluator::finishConvertingOutArgsFromEpetraToThyra(
  const EpetraExt::ModelEvaluator::OutArgs &epetraOutArgs,
  RCP<LinearOpBase<ST> > &W_op,
  RCP<EpetraLinearOp> &efwdW,
  RCP<Epetra_Operator> &eW,
  const ModelEvaluatorBase::OutArgs<ST> &outArgs
  ) const
{

  using Teuchos::rcp_dynamic_cast;
  typedef EpetraExt::ModelEvaluator EME;

  if (nonnull(efwdW)) {
    efwdW->setFullyInitialized(true); 
    // NOTE: Above will directly update W_op also if W.get()==NULL!
  }

  if (nonnull(W_op)) {
    if (W_op.shares_resource(efwdW)) {
      // W_op was already updated above since *efwdW is the same object as *W_op
    }
    else {
      rcp_dynamic_cast<EpetraLinearOp>(W_op, true)->setFullyInitialized(true);
    }
  }
  
}


void ThyraAdaptiveModelEvaluator::updateNominalValuesAndBounds() const
{

  using Teuchos::rcp;
  using Teuchos::implicit_cast;
  typedef ModelEvaluatorBase MEB;
  typedef EpetraExt::ModelEvaluator EME;

  if( !nominalValuesAndBoundsAreUpdated_ ) {

    // Gather the nominal values and bounds into Epetra InArgs objects

    EME::InArgs epetraOrigNominalValues;
    EpetraExt::gatherModelNominalValues(
      *epetraModel_, &epetraOrigNominalValues );

    EME::InArgs epetraOrigLowerBounds;
    EME::InArgs epetraOrigUpperBounds;
    EpetraExt::gatherModelBounds(
      *epetraModel_, &epetraOrigLowerBounds, &epetraOrigUpperBounds );

    // Set up Epetra InArgs scaling object

    epetraInArgsScaling_ = epetraModel_->createInArgs();

    if( !is_null(stateVariableScalingVec_) ) {
      invStateVariableScalingVec_
        = EpetraExt::createInverseModelScalingVector(stateVariableScalingVec_);
      if( epetraOrigNominalValues.supports(EME::IN_ARG_x_dot) ) {
        epetraInArgsScaling_.set_x_dot(invStateVariableScalingVec_);
      }
      if( epetraOrigNominalValues.supports(EME::IN_ARG_x) ) {
        epetraInArgsScaling_.set_x(invStateVariableScalingVec_);
      }
    }
    
    // Scale the original variables and bounds

    EME::InArgs epetraScaledNominalValues = epetraModel_->createInArgs();
    EpetraExt::scaleModelVars(
      epetraOrigNominalValues, epetraInArgsScaling_, &epetraScaledNominalValues
      );

    EME::InArgs epetraScaledLowerBounds = epetraModel_->createInArgs();
    EME::InArgs epetraScaledUpperBounds = epetraModel_->createInArgs();
    EpetraExt::scaleModelBounds(
      epetraOrigLowerBounds, epetraOrigUpperBounds, epetraModel_->getInfBound(),
      epetraInArgsScaling_,
      &epetraScaledLowerBounds, &epetraScaledUpperBounds
      );

    // Wrap the scaled epetra InArgs objects as Thyra InArgs objects!

    nominalValues_ = this->createInArgs();
    lowerBounds_ = this->createInArgs();
    upperBounds_ = this->createInArgs();
    convertInArgsFromEpetraToThyra(epetraScaledNominalValues, &nominalValues_);
    convertInArgsFromEpetraToThyra(epetraScaledLowerBounds, &lowerBounds_);
    convertInArgsFromEpetraToThyra(epetraScaledUpperBounds, &upperBounds_);

    nominalValuesAndBoundsAreUpdated_ = true;

  }
  else {

    // The nominal values and bounds should already be updated an should have
    // the currect scaling!

  }

}


void ThyraAdaptiveModelEvaluator::updateInArgsOutArgs() const
{

  typedef EpetraExt::ModelEvaluator EME;

  const EpetraExt::ModelEvaluator &epetraModel = *epetraModel_;
  EME::InArgs  epetraInArgs  = epetraModel.createInArgs();
  EME::OutArgs epetraOutArgs = epetraModel.createOutArgs();
  const int l_Np = epetraOutArgs.Np();
  const int l_Ng = epetraOutArgs.Ng();

  //
  // InArgs
  //

  InArgsSetup<ST> inArgs;
  inArgs.setModelEvalDescription(this->description());
  inArgs.set_Np(epetraInArgs.Np());
  inArgs.setSupports(IN_ARG_x_dot, epetraInArgs.supports(EME::IN_ARG_x_dot));
  inArgs.setSupports(IN_ARG_x, epetraInArgs.supports(EME::IN_ARG_x));
#ifdef HAVE_THYRA_ME_POLYNOMIAL
  inArgs.setSupports(IN_ARG_x_dot_poly,
    epetraInArgs.supports(EME::IN_ARG_x_dot_poly));
  inArgs.setSupports(IN_ARG_x_poly, epetraInArgs.supports(EME::IN_ARG_x_poly));
#endif // HAVE_THYRA_ME_POLYNOMIAL
  inArgs.setSupports(IN_ARG_t, epetraInArgs.supports(EME::IN_ARG_t));
  inArgs.setSupports(IN_ARG_alpha, epetraInArgs.supports(EME::IN_ARG_alpha));
  inArgs.setSupports(IN_ARG_beta, epetraInArgs.supports(EME::IN_ARG_beta));
  prototypeInArgs_ = inArgs;

  //
  // OutArgs
  //

  OutArgsSetup<ST> outArgs;
  outArgs.setModelEvalDescription(this->description());
  outArgs.set_Np_Ng(l_Np, l_Ng);
  // f
  outArgs.setSupports(OUT_ARG_f, epetraOutArgs.supports(EME::OUT_ARG_f));
  if (outArgs.supports(OUT_ARG_f)) {
    // W_op
    outArgs.setSupports(OUT_ARG_W_op,  epetraOutArgs.supports(EME::OUT_ARG_W));
    outArgs.set_W_properties(convert(epetraOutArgs.get_W_properties()));
    // DfDp
    for(int l=0; l<l_Np; ++l) {
      outArgs.setSupports(OUT_ARG_DfDp, l,
        convert(epetraOutArgs.supports(EME::OUT_ARG_DfDp, l)));
      if(!outArgs.supports(OUT_ARG_DfDp, l).none())
        outArgs.set_DfDp_properties(l,
          convert(epetraOutArgs.get_DfDp_properties(l)));
    }
  }
  // DgDx_dot and DgDx
  for(int j=0; j<l_Ng; ++j) {
    if (inArgs.supports(IN_ARG_x_dot))
      outArgs.setSupports(OUT_ARG_DgDx_dot, j,
        convert(epetraOutArgs.supports(EME::OUT_ARG_DgDx_dot, j)));
    if(!outArgs.supports(OUT_ARG_DgDx_dot, j).none())
      outArgs.set_DgDx_dot_properties(j,
        convert(epetraOutArgs.get_DgDx_dot_properties(j)));
    if (inArgs.supports(IN_ARG_x))
      outArgs.setSupports(OUT_ARG_DgDx, j,
        convert(epetraOutArgs.supports(EME::OUT_ARG_DgDx, j)));
    if(!outArgs.supports(OUT_ARG_DgDx, j).none())
      outArgs.set_DgDx_properties(j,
        convert(epetraOutArgs.get_DgDx_properties(j)));
  }
  // DgDp
  for(int j=0; j < l_Ng; ++j) for(int l=0; l < l_Np; ++l) {
    const EME::DerivativeSupport epetra_DgDp_j_l_support =
      epetraOutArgs.supports(EME::OUT_ARG_DgDp, j, l);
    outArgs.setSupports(OUT_ARG_DgDp, j, l,
      convert(epetra_DgDp_j_l_support));
    if(!outArgs.supports(OUT_ARG_DgDp, j, l).none())
      outArgs.set_DgDp_properties(j, l,
        convert(epetraOutArgs.get_DgDp_properties(j, l)));
  }
#ifdef HAVE_THYRA_ME_POLYNOMIAL
  outArgs.setSupports(OUT_ARG_f_poly,
    epetraOutArgs.supports(EME::OUT_ARG_f_poly));
#endif // HAVE_THYRA_ME_POLYNOMIAL
  prototypeOutArgs_ = outArgs;

  // We are current!
  currentInArgsOutArgs_ = true;

}


RCP<EpetraLinearOp>
ThyraAdaptiveModelEvaluator::create_epetra_W_op() const
{
  return Thyra::partialNonconstEpetraLinearOp(
    this->get_f_space(), this->get_x_space(),
    create_and_assert_W(*epetraModel_)
    );
}


} // namespace AAdapt

