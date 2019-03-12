//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/12/14: this is Epetra (Albany) function.
//Not compiled if ALBANY_EPETRA_EXE is off.

#include "Albany_ModelEvaluator.hpp"

#include "Albany_DistributedParameterLibrary.hpp"
#include "Albany_DistributedParameterDerivativeOp.hpp"
#include "Teuchos_ScalarTraits.hpp"
#include "Teuchos_TestForException.hpp"

// To wrap thyra op inside epetra op
#include "Thyra_EpetraOperatorWrapper.hpp"

#include "Albany_EpetraThyraUtils.hpp"
#include "Albany_CommUtils.hpp"

//IK, 7/15/14: adding option to write the mass matrix to matrix market file, which is needed
//for some applications.  Uncomment the following line to turn on.
//#define WRITE_MASS_MATRIX_TO_MM_FILE

#ifdef WRITE_MASS_MATRIX_TO_MM_FILE
#include "EpetraExt_RowMatrixOut.h"
#include "EpetraExt_BlockMapOut.h"
#endif

namespace Albany
{

ModelEvaluator::
ModelEvaluator(const Teuchos::RCP<Albany::Application>& app_,
               const Teuchos::RCP<Teuchos::ParameterList>& appParams)
 : app(app_)
 , supplies_prec(app_->suppliesPreconditioner())
{
  Teuchos::RCP<Teuchos::FancyOStream> out =
    Teuchos::VerboseObjectBase::getDefaultOStream();

    Teuchos::ParameterList& discParams = appParams->sublist("Discretization");

//IKT, 5/18/15:
//Test for Spectral elements requested, which do not work now with Albany executable.
#ifdef ALBANY_AERAS
    std::string& method = discParams.get("Method", "STK1D");
    if (method == "Ioss Aeras" || method == "Exodus Aeras" || method == "STK1D Aeras"){
       TEUCHOS_TEST_FOR_EXCEPTION(true,
                               Teuchos::Exceptions::InvalidParameter,
                               "Error: Albany executable does not support discretization method " << method
                               << "!  Please re-run with AlbanyT executable." << std::endl);
    }
#endif

  // Get number of time derivatives
  num_time_deriv = discParams.get<int>("Number Of Time Derivatives");

  // Parameters (e.g., for sensitivities, SG expansions, ...)
  Teuchos::ParameterList& problemParams = appParams->sublist("Problem");
  Teuchos::ParameterList& parameterParams =
    problemParams.sublist("Parameters");
  num_param_vecs =
    parameterParams.get("Number of Parameter Vectors", 0);
  bool using_old_parameter_list = false;
  if (parameterParams.isType<int>("Number")) {
    int numParameters = parameterParams.get<int>("Number");
    if (numParameters > 0) {
      num_param_vecs = 1;
      using_old_parameter_list = true;
    }
  }
  param_names.resize(num_param_vecs);
  param_lower_bd.resize(num_param_vecs);
  param_upper_bd.resize(num_param_vecs);

  *out << "Number of parameters vectors  = " << num_param_vecs << std::endl;
  for (int i=0; i<num_param_vecs; i++) {
    Teuchos::ParameterList* pList;
    if (using_old_parameter_list)
      pList = &parameterParams;
    else
      pList = &(parameterParams.sublist(Albany::strint("Parameter Vector",i)));
    int numParameters = pList->get<int>("Number");
    TEUCHOS_TEST_FOR_EXCEPTION(
      numParameters == 0,
      Teuchos::Exceptions::InvalidParameter,
      std::endl << "Error!  In Albany::ModelEvaluator constructor:  " <<
      "Parameter vector " << i << " has zero parameters!" << std::endl);
    param_names[i] =
      Teuchos::rcp(new Teuchos::Array<std::string>(numParameters));
    for (int j=0; j<numParameters; j++) {
      (*param_names[i])[j] =
        pList->get<std::string>(Albany::strint("Parameter",j));
    }
    *out << "Number of parameters in parameter vector " << i << " = "
         << numParameters << std::endl;

  }

  // Setup sacado and epetra storage for parameters
  sacado_param_vec.resize(num_param_vecs);
  epetra_param_map.resize(num_param_vecs);
  epetra_param_vec.resize(num_param_vecs);
  Teuchos::RCP<const Epetra_Comm> comm = createEpetraCommFromTeuchosComm(app->getComm());
  for (int i=0; i<num_param_vecs; i++) {

    // Initialize Sacado parameter vector
    app->getParamLib()->fillVector<PHAL::AlbanyTraits::Residual>(
      *(param_names[i]), sacado_param_vec[i]);

    // Create Epetra map for parameter vector
    epetra_param_map[i] = Teuchos::rcp(new Epetra_LocalMap((int) sacado_param_vec[i].size(), 0, *comm));

    // Create Epetra vector for parameters
    epetra_param_vec[i] = Teuchos::rcp(new Epetra_Vector(*(epetra_param_map[i])));

    Teuchos::ParameterList* pList;
    if (using_old_parameter_list)
      pList = &parameterParams;
    else
      pList = &(parameterParams.sublist(Albany::strint("Parameter Vector",i)));

    int numParameters = epetra_param_map[i]->NumMyElements();

    // Loading lower bounds (if any)
    if (pList->isParameter("Lower Bounds"))
    {
      param_lower_bd[i] = Teuchos::rcp(new Epetra_Vector(*(epetra_param_map[i])));
      Teuchos::Array<double> lb = pList->get<Teuchos::Array<double>>("Lower Bounds");
      TEUCHOS_TEST_FOR_EXCEPTION (lb.size()!=numParameters, Teuchos::Exceptions::InvalidParameter,
                                  "Error! Input lower bounds array has the wrong dimension.\n");

      for (int j=0; j<numParameters; ++j)
        (*param_lower_bd[i])[j] = lb[j];
    }

    // Loading upper bounds (if any)
    if (pList->isParameter("Upper Bounds"))
    {
      param_upper_bd[i] = Teuchos::rcp(new Epetra_Vector(*(epetra_param_map[i])));
      Teuchos::Array<double> ub = pList->get<Teuchos::Array<double>>("Upper Bounds");
      TEUCHOS_TEST_FOR_EXCEPTION (ub.size()!=numParameters, Teuchos::Exceptions::InvalidParameter,
                                  "Error! Input upper bounds array has the wrong dimension.\n");

      for (int j=0; j<numParameters; ++j)
        (*param_upper_bd[i])[j] = ub[j];
    }

    // Loading nominal values (if any)
    if (pList->isParameter("Nominal Values"))
    {
      Teuchos::Array<double> nvals = pList->get<Teuchos::Array<double>>("Nominal Values");
      TEUCHOS_TEST_FOR_EXCEPTION (nvals.size()!=numParameters, Teuchos::Exceptions::InvalidParameter,
                                  "Error! Input nominal values array has the wrong dimension.\n");

      for (int j=0; j<numParameters; ++j) {
        sacado_param_vec[i][j].baseValue = (*(epetra_param_vec[i]))[j] = nvals[j];
      }
    }
    else
    {
      for (int j=0; j<numParameters; ++j)
        (*(epetra_param_vec[i]))[j] = sacado_param_vec[i][j].baseValue;
    }

  }

  // Setup distributed parameters
  distParamLib = app->getDistributedParameterLibrary();
  Teuchos::ParameterList& distParameterParams =
    problemParams.sublist("Distributed Parameters");
  Teuchos::ParameterList* param_list;
  num_dist_param_vecs =
    distParameterParams.get("Number of Parameter Vectors", 0);
  dist_param_names.resize(num_dist_param_vecs);
  *out << "Number of distributed parameters vectors  = " << num_dist_param_vecs
       << std::endl;
  const std::string* p_name_ptr;
  const std::string emptyString("");
  dfdp_ops.resize(num_dist_param_vecs);
  for (int i=0; i<num_dist_param_vecs; i++) {
    const std::string& p_sublist_name = Albany::strint("Distributed Parameter",i);
    param_list = distParameterParams.isSublist(p_sublist_name) ? &distParameterParams.sublist(p_sublist_name) : NULL;

    p_name_ptr = &distParameterParams.get<std::string>(Albany::strint("Parameter",i),emptyString);

    if(param_list != NULL) {
    const std::string& name_from_list = param_list->get<std::string>("Name",emptyString);

    p_name_ptr = (*p_name_ptr != emptyString) ? p_name_ptr : &name_from_list;

    TEUCHOS_TEST_FOR_EXCEPTION(
        (*p_name_ptr != name_from_list) && (name_from_list != emptyString),
        Teuchos::Exceptions::InvalidParameter,
          std::endl << "Error!  In Albany::ModelEvaluator constructor:  Provided two different names for same parameter in Distributed Parameters list: \"" <<
          *p_name_ptr << "\" and \"" << name_from_list << "\"" << std::endl);
    }

    TEUCHOS_TEST_FOR_EXCEPTION(
      !distParamLib->has(*p_name_ptr),
      Teuchos::Exceptions::InvalidParameter,
      std::endl << "Error!  In Albany::ModelEvaluator constructor:  " <<
      "Invalid distributed parameter name \"" << *p_name_ptr << "\""<<std::endl);

    dist_param_names[i] = *p_name_ptr;
    //set parameters bonuds
    if(param_list) {
      Teuchos::RCP<const DistributedParameter> distParam = distParamLib->get(*p_name_ptr);
      if(param_list->isParameter("Lower Bound") && (distParam->lower_bounds_vector() != Teuchos::null))
        distParam->lower_bounds_vector()->assign(param_list->get<double>("Lower Bound"));
      if(param_list->isParameter("Upper Bound") && (distParam->upper_bounds_vector() != Teuchos::null))
        distParam->upper_bounds_vector()->assign(param_list->get<double>("Upper Bound"));
      if(param_list->isParameter("Initial Uniform Value") && (distParam->vector() != Teuchos::null))
        distParam->vector()->assign(param_list->get<double>("Initial Uniform Value"));
    }
    // We create all the dist param deriv ops here, since
    //  1) we need to keep a copy of them (cause the Thyra::EpetraOperatorWrapper does not allow to change the stored op)
    //  2) the create_DfDp_op method is marked const
    dfdp_ops[i] = Teuchos::rcp(new DistributedParameterDerivativeOp(app, dist_param_names[i]));
  }

  timer = Teuchos::TimeMonitor::getNewTimer("Albany: **Total Fill Time**");
}

ModelEvaluator::~ModelEvaluator(){
#ifdef ALBANY_DEBUG
  std::cout << "Calling destructor for Albany_ModelEvaluator" << std::endl;
#endif
}

// Overridden from EpetraExt::ModelEvaluator

Teuchos::RCP<const Epetra_Map>
ModelEvaluator::get_x_map() const
{
  return getEpetraMap(app->getVectorSpace());
}

Teuchos::RCP<const Epetra_Map>
ModelEvaluator::get_f_map() const
{
  return getEpetraMap(app->getVectorSpace());
}

Teuchos::RCP<const Epetra_Map>
ModelEvaluator::get_p_map(int l) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(
    l >= num_param_vecs + num_dist_param_vecs || l < 0,
    Teuchos::Exceptions::InvalidParameter,
    std::endl <<
    "Error!  Albany::ModelEvaluator::get_p_map():  " <<
    "Invalid parameter index l = " << l << std::endl);
  if (l < num_param_vecs) {
    return epetra_param_map[l];
  }
  auto vs = distParamLib->get(dist_param_names[l-num_param_vecs])->vector_space();
  return getEpetraMap(vs);
}

Teuchos::RCP<const Epetra_Map>
ModelEvaluator::get_g_map(int l) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(
    l >= app->getNumResponses() || l < 0,
    Teuchos::Exceptions::InvalidParameter,
    std::endl <<
    "Error!  Albany::ModelEvaluator::get_g_map():  " <<
    "Invalid response index l = " << l << std::endl);

  return getEpetraMap(app->getResponse(l)->responseVectorSpace());
}

Teuchos::RCP<const Teuchos::Array<std::string> >
ModelEvaluator::get_p_names(int l) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(
    l >= num_param_vecs + num_dist_param_vecs || l < 0,
    Teuchos::Exceptions::InvalidParameter,
    std::endl <<
    "Error!  Albany::ModelEvaluator::get_p_names():  " <<
    "Invalid parameter index l = " << l << std::endl);

  if (l < num_param_vecs)
    return param_names[l];
  return Teuchos::rcp(new Teuchos::Array<std::string>(1, dist_param_names[l-num_param_vecs]));
}

Teuchos::RCP<const Epetra_Vector>
ModelEvaluator::get_x_init() const
{
#ifdef ALBANY_MOVE_MEMBER_FN_ADAPTSOLMGR_TPETRA
  return app->getAdaptSolMgr()->getInitialSolution();
#endif
  return app->getInitialSolution();
}

Teuchos::RCP<const Epetra_Vector>
ModelEvaluator::get_x_dot_init() const
{
#ifdef ALBANY_MOVE_MEMBER_FN_ADAPTSOLMGR_TPETRA
  return app->getAdaptSolMgr()->getInitialSolutionDot();
#endif
   return app->getInitialSolutionDot();
}

Teuchos::RCP<const Epetra_Vector>
ModelEvaluator::get_x_dotdot_init() const
{
#ifdef ALBANY_MOVE_MEMBER_FN_ADAPTSOLMGR_TPETRA
  return app->getAdaptSolMgr()->getInitialSolutionDotDot();
#endif
   return app->getInitialSolutionDotDot();
}


Teuchos::RCP<const Epetra_Vector>
ModelEvaluator::get_p_init(int l) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(
    l >= num_param_vecs + num_dist_param_vecs || l < 0,
    Teuchos::Exceptions::InvalidParameter,
    std::endl <<
    "Error!  Albany::ModelEvaluator::get_p_init():  " <<
    "Invalid parameter index l = " << l << std::endl);

  if (l < num_param_vecs) {
    return epetra_param_vec[l];
  }

  return getEpetraVector(distParamLib->get(dist_param_names[l-num_param_vecs])->vector());
}

Teuchos::RCP<const Epetra_Vector>
ModelEvaluator::get_p_lower_bounds(int l) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(
    l >= num_param_vecs + num_dist_param_vecs || l < 0,
    Teuchos::Exceptions::InvalidParameter,
    std::endl <<
    "Error!  Albany::ModelEvaluator::get_p_init():  " <<
    "Invalid parameter index l = " << l << std::endl);

  if (l < num_param_vecs) {
    return param_lower_bd[l];
  }
  return getEpetraVector(distParamLib->get(dist_param_names[l-num_param_vecs])->lower_bounds_vector());
}

Teuchos::RCP<const Epetra_Vector>
ModelEvaluator::get_p_upper_bounds(int l) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(
    l >= num_param_vecs + num_dist_param_vecs || l < 0,
    Teuchos::Exceptions::InvalidParameter,
    std::endl <<
    "Error!  Albany::ModelEvaluator::get_p_init():  " <<
    "Invalid parameter index l = " << l << std::endl);

  if (l < num_param_vecs) {
    return param_upper_bd[l];
  }

  return getEpetraVector(distParamLib->get(dist_param_names[l-num_param_vecs])->upper_bounds_vector());
}

Teuchos::RCP<Epetra_Operator>
ModelEvaluator::create_W() const
{
  return getEpetraOperator(app->getDisc()->createJacobianOp());
}

Teuchos::RCP<EpetraExt::ModelEvaluator::Preconditioner>
ModelEvaluator::create_WPrec() const
{
  Teuchos::RCP<Epetra_Operator> precOp = app->getPreconditioner();

  // Teko prec needs space for Jacobian as well
  Extra_W_crs = Teuchos::rcp_dynamic_cast<Epetra_CrsMatrix>(create_W(), true);

  // bool is answer to: "Prec is already inverted?"
  return Teuchos::rcp(new EpetraExt::ModelEvaluator::Preconditioner(precOp,true));
}

Teuchos::RCP<Epetra_Operator>
ModelEvaluator::create_DfDp_op(int l) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(
    l >= num_param_vecs+num_dist_param_vecs || l < num_param_vecs,
    Teuchos::Exceptions::InvalidParameter,
    std::endl <<
    "Error!  Albany::ModelEvaluator::create_DfDp_op():  " <<
    "Invalid parameter index l = " << l << std::endl);

  return Teuchos::rcp(new Thyra::EpetraOperatorWrapper(dfdp_ops[l-num_param_vecs]));
}

Teuchos::RCP<Epetra_Operator>
ModelEvaluator::create_DgDx_op(int j) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(
    j >= app->getNumResponses() || j < 0,
    Teuchos::Exceptions::InvalidParameter,
    std::endl <<
    "Error!  Albany::ModelEvaluator::create_DgDx_op():  " <<
    "Invalid response index j = " << j << std::endl);

  auto dgdx_op = app->getResponse(j)->createGradientOp();
  return getEpetraOperator(dgdx_op);
}

Teuchos::RCP<Epetra_Operator>
ModelEvaluator::create_DgDx_dot_op(int j) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(
    j >= app->getNumResponses() || j < 0,
    Teuchos::Exceptions::InvalidParameter,
    std::endl <<
    "Error!  Albany::ModelEvaluator::create_DgDx_dot_op():  " <<
    "Invalid response index j = " << j << std::endl);

  auto dgdxdot_op = app->getResponse(j)->createGradientOp();
  return getEpetraOperator(dgdxdot_op);
}

Teuchos::RCP<Epetra_Operator>
ModelEvaluator::create_DgDx_dotdot_op(int j) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(
    j >= app->getNumResponses() || j < 0,
    Teuchos::Exceptions::InvalidParameter,
    std::endl <<
    "Error!  Albany::ModelEvaluator::create_DgDx_dotdot_op():  " <<
    "Invalid response index j = " << j << std::endl);

  auto dgdxdotdot_op = app->getResponse(j)->createGradientOp();
  return getEpetraOperator(dgdxdotdot_op);
}

EpetraExt::ModelEvaluator::InArgs
ModelEvaluator::createInArgs() const
{
  InArgsSetup inArgs;
  inArgs.setModelEvalDescription(this->description());

  inArgs.setSupports(IN_ARG_x,true);
  if(num_time_deriv > 0){
    inArgs.setSupports(IN_ARG_t,true);
    inArgs.setSupports(IN_ARG_x_dot,true);
    inArgs.setSupports(IN_ARG_alpha,true);
    inArgs.setSupports(IN_ARG_beta,true);
  }
  if(num_time_deriv > 1){
    inArgs.setSupports(IN_ARG_x_dotdot,true);
    inArgs.setSupports(IN_ARG_omega,true);
  }
  inArgs.set_Np(num_param_vecs+num_dist_param_vecs);

  return static_cast<EpetraExt::ModelEvaluator::InArgs>(inArgs);
}

EpetraExt::ModelEvaluator::OutArgs
ModelEvaluator::createOutArgs() const
{
  OutArgsSetup outArgs;
  outArgs.setModelEvalDescription(this->description());

  int n_g = app->getNumResponses();

  // Deterministic
  outArgs.setSupports(OUT_ARG_f,true);
  outArgs.setSupports(OUT_ARG_W,true);
  outArgs.set_W_properties(
    DerivativeProperties(DERIV_LINEARITY_UNKNOWN, DERIV_RANK_FULL, true));
  if (supplies_prec) outArgs.setSupports(OUT_ARG_WPrec, true);
  outArgs.set_Np_Ng(num_param_vecs+num_dist_param_vecs, n_g);

  for (int i=0; i<num_param_vecs; i++)
    outArgs.setSupports(OUT_ARG_DfDp, i, DerivativeSupport(DERIV_MV_BY_COL));
  for (int i=0; i<num_dist_param_vecs; i++)
    outArgs.setSupports(OUT_ARG_DfDp, i+num_param_vecs,
                        DerivativeSupport(DERIV_LINEAR_OP));
  for (int i=0; i<n_g; i++) {
    if (app->getResponse(i)->isScalarResponse()) {
      outArgs.setSupports(OUT_ARG_DgDx, i,
                          DerivativeSupport(DERIV_TRANS_MV_BY_ROW));
      outArgs.setSupports(OUT_ARG_DgDx_dot, i,
                          DerivativeSupport(DERIV_TRANS_MV_BY_ROW));
      outArgs.setSupports(OUT_ARG_DgDx_dotdot, i,
                          DerivativeSupport(DERIV_TRANS_MV_BY_ROW));
    }
    else {
      outArgs.setSupports(OUT_ARG_DgDx, i,
                          DerivativeSupport(DERIV_LINEAR_OP));
      outArgs.setSupports(OUT_ARG_DgDx_dot, i,
                          DerivativeSupport(DERIV_LINEAR_OP));
      outArgs.setSupports(OUT_ARG_DgDx_dotdot, i,
                          DerivativeSupport(DERIV_LINEAR_OP));
    }

    for (int j=0; j<num_param_vecs; j++) {
      outArgs.setSupports(OUT_ARG_DgDp, i, j,
                          DerivativeSupport(DERIV_MV_BY_COL));
    }
    if (app->getResponse(i)->isScalarResponse()) {
      for (int j=0; j<num_dist_param_vecs; j++)
        outArgs.setSupports(OUT_ARG_DgDp, i, j+num_param_vecs,
                            DerivativeSupport(DERIV_TRANS_MV_BY_ROW));
    }
    else {
      for (int j=0; j<num_dist_param_vecs; j++)
        outArgs.setSupports(OUT_ARG_DgDp, i, j+num_param_vecs,
                            DerivativeSupport(DERIV_LINEAR_OP));
    }
  }

  return static_cast<EpetraExt::ModelEvaluator::OutArgs>(outArgs);
}

void ModelEvaluator::evalModel(const InArgs& inArgs,
                               const OutArgs& outArgs) const
{
  Teuchos::TimeMonitor Timer(*timer); //start timer
  //
  // Get the input arguments
  //
  Teuchos::RCP<const Epetra_Vector> xE = inArgs.get_x();
  Teuchos::RCP<const Epetra_Vector> xE_dot;
  Teuchos::RCP<const Epetra_Vector> xE_dotdot;
  if(num_time_deriv > 0) {
    xE_dot = inArgs.get_x_dot();
  }
  if(num_time_deriv > 1) {
    xE_dotdot = inArgs.get_x_dotdot();
  }

  //Create Thyra wrappers of xE, xE_dot, and xE_dotdot
  Teuchos::RCP<const Thyra_Vector> x, x_dot, x_dotdot;
  if (x != Teuchos::null) {
    x  = createConstThyraVector(xE);
  }
  if (Teuchos::nonnull(xE_dot)) {
    x_dot = createConstThyraVector(xE_dot);
  }
  if (Teuchos::nonnull(xE_dotdot)) {
    x_dotdot = createConstThyraVector(xE_dotdot);
  }

  double alpha     = 0.0;
  double omega     = 0.0;
  double beta      = 1.0;
  double curr_time = 0.0;

  if (Teuchos::nonnull(x_dot)){
    alpha = inArgs.get_alpha();
    beta = inArgs.get_beta();
    curr_time  = inArgs.get_t();
  }
  if (Teuchos::nonnull(x_dotdot)) {
    omega = inArgs.get_omega();
  }

  for (int i=0; i<num_param_vecs; i++) {
    Teuchos::RCP<const Epetra_Vector> p = inArgs.get_p(i);
    if (p != Teuchos::null) {
      for (unsigned int j=0; j<sacado_param_vec[i].size(); j++) {
        sacado_param_vec[i][j].baseValue = (*p)[j];
      }
    }
  }

  for (int i=0; i<num_dist_param_vecs; i++) {
    Teuchos::RCP<const Epetra_Vector> pE = inArgs.get_p(i+num_param_vecs);
    if (pE != Teuchos::null) {
      //*(distParamLib->get(dist_param_names[i])->vector()) = *p;
      distParamLib->get(dist_param_names[i])->vector()->assign(*createConstThyraVector(pE));
    }
  }

  //
  // Get the output arguments
  //
  EpetraExt::ModelEvaluator::Evaluation<Epetra_Vector> f_outE = outArgs.get_f();
  // Wrap residual into thyra vector
  Teuchos::RCP<Thyra_Vector> f_out;
  if (Teuchos::nonnull(f_outE)) {
    f_out = createThyraVector(f_outE);
  }

  Teuchos::RCP<Epetra_Operator> W_out = outArgs.get_W();

  // Cast W to a CrsMatrix, throw an exception if this fails
  Teuchos::RCP<Epetra_CrsMatrix> W_out_crs;
  Teuchos::RCP<Thyra_LinearOp> W_out_op;
  Teuchos::RCP<Thyra_LinearOp> Extra_W_op;
#ifdef WRITE_MASS_MATRIX_TO_MM_FILE
  //IK, 7/15/14: adding object to hold mass matrix to be written to matrix market file
  Teuchos::RCP<Epetra_CrsMatrix> Mass;
  Teuchos::RCP<Thyra_LinearOp> MassOp;
  //IK, 7/15/14: needed for writing mass matrix out to matrix market file
  EpetraExt::ModelEvaluator::Evaluation<Epetra_Vector> f_tmp = outArgs.get_f();
  Teuchos::RCP<Thyra_Vector> ftmp;
  if (Teuchos::nonnull(f_tmp)) {
    f_tmp = createThyraVector(f_tmp);
  }
#endif

  if (W_out != Teuchos::null) {
    W_out_crs = Teuchos::rcp_dynamic_cast<Epetra_CrsMatrix>(W_out, true);
    W_out_op = createThyraLinearOp(W_out_crs);
#ifdef WRITE_MASS_MATRIX_TO_MM_FILE
    //IK, 7/15/14: adding object to hold mass matrix to be written to matrix market file
    Mass = Teuchos::rcp_dynamic_cast<Epetra_CrsMatrix>(W_out, true);
    MassOp = createThyraLinearOp(Mass);
#endif
  }

  if(nonnull(Extra_W_crs)) {
    Extra_W_op = createThyraLinearOp(Extra_W_crs);
  }

int test_var = 0;
if(test_var != 0){
std::cout << "The current solution length is: " << xE->MyLength() << std::endl;
xE->Print(std::cout);

}

  // Get preconditioner operator, if requested
  Teuchos::RCP<Epetra_Operator> WPrec_out;
  if (outArgs.supports(OUT_ARG_WPrec)) {
    WPrec_out = outArgs.get_WPrec();
  }

  //
  // Compute the functions
  //
  bool f_already_computed = false;

  // W matrix
  if (W_out != Teuchos::null) {
#ifdef WRITE_MASS_MATRIX_TO_MM_FILE
    //IK, 7/15/14: write mass matrix to matrix market file
    //Warning: to read this in to MATLAB correctly, code must be run in serial.
    //Otherwise Mass will have a distributed Map which would also need to be read in to MATLAB for proper
    //reading in of Mass.
    // NOTE: do this BEFORE computing the actual jacobian, since we are using the same matrix/vector for W and f.
    //       if you do this after the computation of the actual jacobian, you'll end up with a mass matrix instead of J.
    app->computeGlobalJacobian(1.0, 0.0, 0.0, curr_time,
                               x,x_dot,x_dotdot,
                               sacado_param_vec,
                               ftmp, MassOp);

    EpetraExt::RowMatrixToMatrixMarketFile("mass.mm", *Mass);
    EpetraExt::BlockMapToMatrixMarketFile("rowmap.mm", Mass->RowMap());
    EpetraExt::BlockMapToMatrixMarketFile("colmap.mm", Mass->ColMap());
#endif

    app->computeGlobalJacobian(alpha, beta, omega, curr_time,
                               x, x_dot, x_dotdot,
                               sacado_param_vec,
                               f_out, W_out_op);

    f_already_computed=true;
    W_out_crs->FillComplete(true);


//    app->computeGlobalJacobian(alpha, beta, omega, curr_time, x_dot.get(), x_dotdot.get(),*x,
//                               sacado_param_vec, f_out.get(), *W_out_crs);
    if(test_var != 0) {
      //std::cout << "The current rhs length is: " << f_out->MyLength() << std::endl;
      //f_out->Print(std::cout);
      std::cout << "The current Jacobian length is: " << W_out_crs->NumGlobalRows() << std::endl;
      W_out_crs->Print(std::cout);
    }
  }

  if (WPrec_out != Teuchos::null) {
    app->computeGlobalJacobian(alpha, beta, omega, curr_time,
                               x, x_dot, x_dotdot,
                               sacado_param_vec,
                               f_out, Extra_W_op);

//    app->computeGlobalJacobian(alpha, beta, omega, curr_time, x_dot.get(), x_dotdot.get(), *x,
//                               sacado_param_vec, f_out.get(), *Extra_W_crs);
    f_already_computed=true;
    Extra_W_crs->FillComplete(true);


  if(test_var != 0) {
    //std::cout << "The current rhs length is: " << f_out->MyLength() << std::endl;
    //f_out->Print(std::cout);
    std::cout << "The current preconditioner length is: " << Extra_W_crs->NumGlobalRows() << std::endl;
    Extra_W_crs->Print(std::cout);
  }

    app->computeGlobalPreconditioner(Extra_W_crs, WPrec_out);
  }

  // scalar df/dp
  for (int i=0; i<num_param_vecs; i++) {
    Teuchos::RCP<Epetra_MultiVector> dfdp_outE =
      outArgs.get_DfDp(i).getMultiVector();
    Teuchos::RCP<Thyra_MultiVector> dfdp_out;
    if (dfdp_outE != Teuchos::null) {
      dfdp_out = createThyraMultiVector(dfdp_outE);

      Teuchos::Array<int> p_indexes = outArgs.get_DfDp(i).getDerivativeMultiVector().getParamIndexes();
      Teuchos::RCP<ParamVec> p_vec;
      if (p_indexes.size() == 0)
        p_vec = Teuchos::rcp(&sacado_param_vec[i],false);
      else {
        p_vec = Teuchos::rcp(new ParamVec);
        for (int j=0; j<p_indexes.size(); j++)
          p_vec->addParam(sacado_param_vec[i][p_indexes[j]].family,
                          sacado_param_vec[i][p_indexes[j]].baseValue);
      }

      app->computeGlobalTangent(0.0, 0.0, 0.0, curr_time, false,
                                x, x_dot, x_dotdot,
                                sacado_param_vec, p_vec.get(),
                                Teuchos::null, Teuchos::null, Teuchos::null, Teuchos::null,
                                f_out,Teuchos::null,dfdp_out);
      f_already_computed=true;

if(test_var != 0){
std::cout << "The current rhs length is: " << f_outE->MyLength() << std::endl;
f_outE->Print(std::cout);
}
    }
  }

  // distributed df/dp
  for (int i=0; i<num_dist_param_vecs; i++) {
    Teuchos::RCP<Epetra_Operator> dfdp_out =
      outArgs.get_DfDp(i+num_param_vecs).getLinearOp();
    if (dfdp_out != Teuchos::null) {
      dfdp_ops[i]->set(curr_time,x,x_dot,x_dotdot,Teuchos::rcp(&sacado_param_vec,false));
    }
  }

  // f
  if (app->is_adjoint) {
    //Derivative f_deriv(f_out, DERIV_TRANS_MV_BY_ROW);
    const Thyra_Derivative f_deriv(f_out, Thyra::ModelEvaluatorBase::DERIV_TRANS_MV_BY_ROW);

    int response_index = 0; // need to add capability for sending this in

    const Thyra_Derivative dummy_deriv;

    app->evaluateResponseDerivative(response_index, curr_time,
                                    x, x_dot, x_dotdot,
                                    sacado_param_vec, NULL,
                                    Teuchos::null,
                                    f_deriv, dummy_deriv, dummy_deriv, dummy_deriv);
//    app->evaluateResponseDerivative(response_index, curr_time, x_dot.get(), x_dotdot.get(), *x,
//                                    sacado_param_vec, NULL,
//                                    NULL, f_deriv, Derivative(), Derivative(), Derivative());

  } else {
    if (f_out != Teuchos::null && !f_already_computed) {
      app->computeGlobalResidual(curr_time, x, x_dot, x_dotdot, sacado_param_vec, f_out);

if(test_var != 0){
std::cout << "The current rhs length is: " << f_outE->MyLength() << std::endl;
f_outE->Print(std::cout);
}
    }
  }

  // Response functions
  for (int i=0; i<outArgs.Ng(); i++) {
    //Set curr_time to final time at which response occurs.
    if(num_time_deriv > 0)
      curr_time  = inArgs.get_t();
    Teuchos::RCP<Epetra_Vector> g_outE = outArgs.get_g(i);
    // Wrap response into thyra vector
    Teuchos::RCP<Thyra_Vector> g_out;
    if (Teuchos::nonnull(g_outE)) {
      g_out = createThyraVector(g_outE);
    }

    bool g_computed = false;

    Teuchos::RCP<Thyra_Derivative> dgdx_out;
    Teuchos::RCP<Thyra_Derivative> dgdxdot_out;
    Teuchos::RCP<Thyra_Derivative> dgdxdotdot_out;
    const Thyra_Derivative dummy_deriv;

    Derivative dgdx_outE = outArgs.get_DgDx(i);
    Derivative dgdxdot_outE = outArgs.get_DgDx_dot(i);
    Derivative dgdxdotdot_outE = outArgs.get_DgDx_dotdot(i);

    Teuchos::RCP<Thyra_MultiVector> dgdx_out_vec;
    Teuchos::RCP<Epetra_MultiVector> dgdx_out_vecE = dgdx_outE.getMultiVector();
    if (dgdx_out_vecE != Teuchos::null) {
      dgdx_out_vec = createThyraMultiVector(dgdx_out_vecE);
      dgdx_out = Teuchos::rcp(new Thyra_Derivative(dgdx_out_vec, Thyra::convert(dgdx_outE.getMultiVectorOrientation())));
    } else {
      dgdx_out = Teuchos::rcp(new Thyra_Derivative());
    }

    Teuchos::RCP<Thyra_MultiVector> dgdxdot_out_vec;
    Teuchos::RCP<Epetra_MultiVector> dgdxdot_out_vecE = dgdxdot_outE.getMultiVector();
    if (dgdxdot_out_vecE != Teuchos::null) {
      dgdxdot_out_vec = createThyraMultiVector(dgdxdot_out_vecE);
      dgdxdot_out = Teuchos::rcp(new Thyra_Derivative(dgdxdot_out_vec, Thyra::convert(dgdxdot_outE.getMultiVectorOrientation())));
    } else {
      dgdxdot_out = Teuchos::rcp(new Thyra_Derivative());
    }

    Teuchos::RCP<Thyra_MultiVector> dgdxdotdot_out_vec;
    Teuchos::RCP<Epetra_MultiVector> dgdxdotdot_out_vecE = dgdxdotdot_outE.getMultiVector();
    if (dgdxdotdot_out_vecE != Teuchos::null) {
      dgdxdotdot_out_vec = createThyraMultiVector(dgdxdotdot_out_vecE);
      dgdxdotdot_out = Teuchos::rcp(new Thyra_Derivative(dgdxdotdot_out_vec, Thyra::convert(dgdxdotdot_outE.getMultiVectorOrientation())));
    } else {
      dgdxdotdot_out = Teuchos::rcp(new Thyra_Derivative());
    }

    // dg/dx, dg/dxdot
    if (!dgdx_outE.isEmpty() || !dgdxdot_outE.isEmpty() || !dgdxdotdot_outE.isEmpty() ) {
      app->evaluateResponseDerivative(i, curr_time, x, x_dot, x_dotdot,
                                      sacado_param_vec, NULL,
                                      g_out, *dgdx_out, *dgdxdot_out, *dgdxdotdot_out, dummy_deriv);
//      app->evaluateResponseDerivative(i, curr_time, x_dot.get(), x_dotdot.get(), *x,
//                                      sacado_param_vec, NULL,
//                                      g_out.get(), dgdx_out,
//                                      dgdxdot_out, dgdxdotdot_out, Derivative());
      g_computed = true;
    }

    // dg/dp
    for (int j=0; j<num_param_vecs; j++) {
      Teuchos::RCP<Epetra_MultiVector> dgdp_outE =
        outArgs.get_DgDp(i,j).getMultiVector();
      // Wrap into Thyra multivector
      Teuchos::RCP<Thyra_MultiVector> dgdp_out;
      if (dgdp_outE != Teuchos::null) {
        Teuchos::Array<int> p_indexes =
          outArgs.get_DgDp(i,j).getDerivativeMultiVector().getParamIndexes();
        Teuchos::RCP<ParamVec> p_vec;
        if (p_indexes.size() == 0) {
          p_vec = Teuchos::rcp(&sacado_param_vec[j],false);
        } else {
          p_vec = Teuchos::rcp(new ParamVec);
          for (int k=0; k<p_indexes.size(); k++)
            p_vec->addParam(sacado_param_vec[j][p_indexes[k]].family,
                            sacado_param_vec[j][p_indexes[k]].baseValue);
        }
        // wrap g_outE into a thyra vector (if not already done)
        if (g_outE != Teuchos::null && g_out.is_null()) {
           g_out = createThyraVector(g_outE);
        }
        // wrap dgdp_outE into a thyra multivector
        if (dgdp_outE != Teuchos::null) {
           dgdp_out = createThyraMultiVector(dgdp_outE);
        }
        app->evaluateResponseTangent(i, alpha, beta, omega, curr_time, false,
                                     x, x_dot, x_dotdot,
                                     sacado_param_vec, p_vec.get(),
                                     Teuchos::null, Teuchos::null, Teuchos::null, Teuchos::null,
                                     g_out,Teuchos::null,dgdp_out);

        g_computed = true;
      }
    }

    // Need to handle dg/dp for distributed p
    for(int j=0; j<num_dist_param_vecs; j++) {
      Teuchos::RCP<Epetra_MultiVector> dgdp_outE = outArgs.get_DgDp(i,j+num_param_vecs).getMultiVector();
      Teuchos::RCP<Thyra_MultiVector> dgdp_out;
      if (dgdp_outE != Teuchos::null) {
        dgdp_out = createThyraMultiVector(dgdp_outE);
        dgdp_out->assign(0.);
        app->evaluateResponseDistParamDeriv(i, curr_time,
                                            x, x_dot, x_dotdot,
                                            sacado_param_vec, dist_param_names[j],
                                            dgdp_out);
      }
    }

    if (g_outE != Teuchos::null && !g_computed) {
      // Wrap into Thyra vector
      g_out = createThyraVector(g_outE);
      app->evaluateResponse(i, curr_time, x, x_dot, x_dotdot, sacado_param_vec, g_out);
    }
  }
}

} // namespace Albany
