//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_ModelEvaluator.hpp"

#include "Albany_DistributedParameterLibrary.hpp"
#include "Albany_DistributedParameterDerivativeOp.hpp"
#include "Teuchos_ScalarTraits.hpp"
#include "Teuchos_TestForException.hpp"
#include "Albany_ObserverImpl.hpp"
#include "Albany_ThyraUtils.hpp"

#include "Albany_Application.hpp"

// uncomment the following to write stuff out to matrix market to debug
//#define WRITE_TO_MATRIX_MARKET

#ifdef WRITE_TO_MATRIX_MARKET
static int mm_counter_sol = 0;
static int mm_counter_res = 0;
static int mm_counter_jac = 0;
#endif  // WRITE_TO_MATRIX_MARKET

// IK, 4/24/15: adding option to write the mass matrix to matrix market file,
// which is needed
// for some applications.  Uncomment the following line to turn on.
//#define WRITE_MASS_MATRIX_TO_MM_FILE

namespace {
void sanitize_nans(const Thyra_Derivative& v)
{
  if (!v.isEmpty() && Teuchos::nonnull(v.getMultiVector())) {
    v.getMultiVector()->assign(0.0);
  }
}
}  // namespace

namespace Albany
{

ModelEvaluator::
ModelEvaluator (const Teuchos::RCP<Albany::Application>&    app_,
                const Teuchos::RCP<Teuchos::ParameterList>& appParams_)
 : app(app_)
 , appParams(appParams_)
 , supplies_prec(app_->suppliesPreconditioner())
 , supports_xdot(false)
 , supports_xdotdot(false)
{
  Teuchos::RCP<Teuchos::FancyOStream> out =
      Teuchos::VerboseObjectBase::getDefaultOStream();

  // Parameters (e.g., for sensitivities, SG expansions, ...)
  Teuchos::ParameterList& problemParams   = appParams->sublist("Problem");
  Teuchos::ParameterList& parameterParams = problemParams.sublist("Parameters");

  const std::string soln_method = problemParams.get("Solution Method", "Steady"); 
  if (soln_method == "Transient Tempus") {
    use_tempus = true; 
  }

  num_param_vecs = parameterParams.get("Number of Parameter Vectors", 0);
  bool using_old_parameter_list = false;
  if (parameterParams.isType<int>("Number")) {
    int numParameters = parameterParams.get<int>("Number");
    if (numParameters > 0) {
      num_param_vecs           = 1;
      using_old_parameter_list = true;
    }
  }

  *out << "Number of parameter vectors  = " << num_param_vecs << std::endl;

  Teuchos::ParameterList& responseParams =
      problemParams.sublist("Response Functions");

  int  num_response_vecs       = app->getNumResponses();
  bool using_old_response_list = false;
  if (responseParams.isType<int>("Number")) {
    int numParameters = responseParams.get<int>("Number");
    if (numParameters > 0) {
      num_response_vecs       = 1;
      using_old_response_list = true;
    }
  }

  param_names.resize(num_param_vecs);
  param_lower_bds.resize(num_param_vecs);
  param_upper_bds.resize(num_param_vecs);
  for (int l = 0; l < num_param_vecs; ++l) {
    const Teuchos::ParameterList* pList =
        using_old_parameter_list ?
            &parameterParams :
            &(parameterParams.sublist(Albany::strint("Parameter Vector", l)));

    const int numParameters = pList->get<int>("Number");
    TEUCHOS_TEST_FOR_EXCEPTION(
        numParameters == 0,
        Teuchos::Exceptions::InvalidParameter,
        std::endl
            << "Error!  In Albany::ModelEvaluator constructor:  "
            << "Parameter vector "
            << l
            << " has zero parameters!"
            << std::endl);

    param_names[l] =
        Teuchos::rcp(new Teuchos::Array<std::string>(numParameters));
    for (int k = 0; k < numParameters; ++k) {
      (*param_names[l])[k] =
          pList->get<std::string>(Albany::strint("Parameter", k));
    }

    *out << "Number of parameters in parameter vector " << l << " = "
         << numParameters << std::endl;
  }

  Teuchos::Array<Teuchos::RCP<Teuchos::Array<std::string>>> response_names;
  response_names.resize(num_response_vecs);
  for (int l = 0; l < num_response_vecs; ++l) {
    const Teuchos::ParameterList* pList =
        using_old_response_list ?
            &responseParams :
            &(responseParams.sublist(Albany::strint("Response Vector", l)));

    bool number_exists = pList->getEntryPtr("Number");

    if (number_exists) {
      const int numParameters = pList->get<int>("Number");
      TEUCHOS_TEST_FOR_EXCEPTION(
          numParameters == 0,
          Teuchos::Exceptions::InvalidParameter,
          std::endl
              << "Error!  In Albany::ModelEvaluator constructor:  "
              << "Response vector "
              << l
              << " has zero parameters!"
              << std::endl);

      response_names[l] =
          Teuchos::rcp(new Teuchos::Array<std::string>(numParameters));
      for (int k = 0; k < numParameters; ++k) {
        (*response_names[l])[k] =
            pList->get<std::string>(strint("Response", k));
      }
    }
  }

  *out << "Number of response vectors  = " << num_response_vecs << std::endl;

  // Setup sacado and thyra storage for parameters
  sacado_param_vec.resize(num_param_vecs);
  param_vecs.resize(num_param_vecs);
  param_vss.resize(num_param_vecs);
  thyra_response_vec.resize(num_response_vecs);

  Teuchos::RCP<const Teuchos::Comm<int>> commT = app->getComm();
  for (int l = 0; l < param_vecs.size(); ++l) {
    try {
      // Initialize Sacado parameter vector
      // The following call will throw, and it is often due to an incorrect
      // input line in the "Parameters" PL
      // in the input file. Give the user a hint about what might be happening
      app->getParamLib()->fillVector<PHAL::AlbanyTraits::Residual>(
          *(param_names[l]), sacado_param_vec[l]);
    } catch (const std::logic_error& le) {
      *out << "Error: exception thrown from ParamLib fillVector in file "
           << __FILE__ << " line " << __LINE__ << std::endl;
      *out << "This is probably due to something incorrect in the "
              "\"Parameters\" list in the input file, one of the lines:"
           << std::endl;
      for (int k = 0; k < param_names[l]->size(); ++k)
        *out << "      " << (*param_names[l])[k] << std::endl;

      throw le;  // rethrow to shut things down
    }

    // Create vector space for parameter vector
    param_vss[l] = createLocallyReplicatedVectorSpace(sacado_param_vec[l].size(), commT);

    // Create Thyra vector for parameters
    param_vecs[l] = Thyra::createMember(param_vss[l]);

    Teuchos::ParameterList* pList;
    if (using_old_parameter_list) {
      pList = &parameterParams;
    } else {
      pList = &(parameterParams.sublist(strint("Parameter Vector",l)));
    }

    int numParameters = param_vss[l]->dim();

    // Loading lower bounds (if any)
    if (pList->isParameter("Lower Bounds"))
    {
      param_lower_bds[l] = Thyra::createMember(param_vss[l]);
      Teuchos::Array<ST> lb = pList->get<Teuchos::Array<ST>>("Lower Bounds");
      TEUCHOS_TEST_FOR_EXCEPTION (lb.size()!=numParameters, Teuchos::Exceptions::InvalidParameter,
                                  "Error! Input lower bounds array has the wrong dimension.\n");

      auto param_lower_bd_nonConstView = getNonconstLocalData(param_lower_bds[l]);
      for (int k = 0; k < numParameters; ++k) {
        param_lower_bd_nonConstView[k] = lb[k];
      }
    }

    // Loading upper bounds (if any)
    if (pList->isParameter("Upper Bounds"))
    {
      param_upper_bds[l] = Thyra::createMember(param_vss[l]);
      Teuchos::Array<ST> ub = pList->get<Teuchos::Array<ST>>("Upper Bounds");
      TEUCHOS_TEST_FOR_EXCEPTION (ub.size()!=numParameters, Teuchos::Exceptions::InvalidParameter,
                                  "Error! Input upper bounds array has the wrong dimension.\n");

      auto param_upper_bd_nonConstView = getNonconstLocalData(param_upper_bds[l]);
      for (int k = 0; k < numParameters; ++k) {
        param_upper_bd_nonConstView[k] = ub[k];
      }
    }

    // Loading nominal values (if any)
    auto param_vec_nonConstView = getNonconstLocalData(param_vecs[l]);
    if (pList->isParameter("Nominal Values"))
    {
      Teuchos::Array<ST> nvals = pList->get<Teuchos::Array<ST>>("Nominal Values");
      TEUCHOS_TEST_FOR_EXCEPTION (nvals.size()!=numParameters, Teuchos::Exceptions::InvalidParameter,
                                  "Error! Input nominal values array has the wrong dimension.\n");

      for (int k = 0; k < numParameters; ++k) {
        sacado_param_vec[l][k].baseValue = param_vec_nonConstView[k] = nvals[k];
      }
    } else {
      for (int k = 0; k < numParameters; ++k) {
        param_vec_nonConstView[k] = sacado_param_vec[l][k].baseValue;
      }
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
  const std::string  emptyString("");
  for (int i = 0; i < num_dist_param_vecs; i++) {
    const std::string& p_sublist_name = strint("Distributed Parameter", i);
    param_list = distParameterParams.isSublist(p_sublist_name) ?
                     &distParameterParams.sublist(p_sublist_name) :
                     NULL;

    p_name_ptr = &distParameterParams.get<std::string>(strint("Parameter", i), emptyString);

    if (param_list != NULL) {
      const std::string& name_from_list =
          param_list->get<std::string>("Name", emptyString);

      p_name_ptr = (*p_name_ptr != emptyString) ? p_name_ptr : &name_from_list;

      TEUCHOS_TEST_FOR_EXCEPTION(
          (*p_name_ptr != name_from_list) && (name_from_list != emptyString),
          Teuchos::Exceptions::InvalidParameter,
          std::endl
              << "Error!  In Albany::ModelEvaluator constructor:  Provided "
                 "two different names for same parameter in Distributed "
                 "Parameters list: \""
              << *p_name_ptr
              << "\" and \""
              << name_from_list
              << "\""
              << std::endl);
    }

    TEUCHOS_TEST_FOR_EXCEPTION(
        !distParamLib->has(*p_name_ptr),
        Teuchos::Exceptions::InvalidParameter,
        std::endl
            << "Error!  In Albany::ModelEvaluator constructor:  "
            << "Invalid distributed parameter name \""
            << *p_name_ptr
            << "\""
            << std::endl);

    dist_param_names[i] = *p_name_ptr;
    // set parameters bonuds
    if (param_list) {
      Teuchos::RCP<const DistributedParameter> distParam = distParamLib->get(*p_name_ptr);
      if (param_list->isParameter("Lower Bound") &&
          (distParam->lower_bounds_vector() != Teuchos::null))
        distParam->lower_bounds_vector()->assign(
            param_list->get<ST>("Lower Bound"));
      if (param_list->isParameter("Upper Bound") &&
          (distParam->upper_bounds_vector() != Teuchos::null))
        distParam->upper_bounds_vector()->assign(
            param_list->get<ST>("Upper Bound"));
      if (param_list->isParameter("Initial Uniform Value") &&
          (distParam->vector() != Teuchos::null))
        distParam->vector()->assign(
            param_list->get<ST>("Initial Uniform Value"));
    }
  }

  for (int l = 0; l < app->getNumResponses(); ++l) {
    // Create Thyra vector for responses
    thyra_response_vec[l] = Thyra::createMember(app->getResponse(l)->responseVectorSpace());
  }

  // Determine the number of solution vectors (x, xdot, xdotdot)
  int num_sol_vectors = app->getAdaptSolMgr()->getInitialSolution()->domain()->dim();

  if (num_sol_vectors > 1) {  // have x dot
    supports_xdot = true;
    if (num_sol_vectors > 2)  // have both x dot and x dotdot
      supports_xdotdot = true;
  }

  // Setup nominal values, lower and upper bounds, and final point
  nominalValues = this->createInArgsImpl();
  lowerBounds = this->createInArgsImpl();
  upperBounds = this->createInArgsImpl();

  // All the ME vectors are unallocated here
  allocateVectors();

  // TODO: Check if correct nominal values for parameters
  for (int l = 0; l < num_param_vecs; ++l) {
    nominalValues.set_p(l, param_vecs[l]);
    if(Teuchos::nonnull(param_lower_bds[l])) {
      lowerBounds.set_p(l, param_lower_bds[l]);
    }
    if(Teuchos::nonnull(param_upper_bds[l])) {
      upperBounds.set_p(l, param_upper_bds[l]);
    }
  }
  for (int l = 0; l < num_dist_param_vecs; ++l) {
    nominalValues.set_p(l+num_param_vecs, distParamLib->get(dist_param_names[l])->vector());
    lowerBounds.set_p(l+num_param_vecs, distParamLib->get(dist_param_names[l])->lower_bounds_vector());
    upperBounds.set_p(l+num_param_vecs, distParamLib->get(dist_param_names[l])->upper_bounds_vector());
  }

  overwriteNominalValuesWithFinalPoint = appParams->get("Overwrite Nominal Values With Final Point",false);

  timer = Teuchos::TimeMonitor::getNewTimer("Albany: Total Fill Time");
}

void ModelEvaluator::allocateVectors()
{
  const Teuchos::RCP<const Thyra_MultiVector>   xMV  = app->getAdaptSolMgr()->getCurrentSolution();

  // Create non-const versions of x_init [and x_dot_init [and x_dotdot_init]]
  const Teuchos::RCP<const Thyra_Vector> x_init = xMV->col(0);
  const Teuchos::RCP<Thyra_Vector> x_init_nonconst = x_init->clone_v();
  nominalValues.set_x(x_init_nonconst);

  // Have xdot
  if (xMV->domain()->dim() > 1) {
    const Teuchos::RCP<const Thyra_Vector> x_dot_init = xMV->col(1);
    const Teuchos::RCP<Thyra_Vector>       x_dot_init_nonconst = x_dot_init->clone_v();
    nominalValues.set_x_dot(x_dot_init_nonconst);
  }

  // Have xdotdot
  if (xMV->domain()->dim() > 2) {
    // Set xdotdot in parent class to pass to time integrator

    // GAH set x_dotdot for transient simulations. Note that xDotDot is a member
    // of Piro::TransientDecorator<ST, LO, GO, KokkosNode>
    const Teuchos::RCP<const Thyra_Vector> x_dotdot_init = xMV->col(2);
    const Teuchos::RCP<Thyra_Vector>       x_dotdot_init_nonconst = x_dotdot_init->clone_v();
    // IKT, 3/30/17: set x_dotdot in nominalValues for Tempus, now that
    // it is available in Thyra::ModelEvaluator
    this->xDotDot = x_dotdot_init_nonconst;
    nominalValues.set_x_dot_dot(x_dotdot_init_nonconst);
  } else {
    this->xDotDot = Teuchos::null;
  }
}

// Overridden from Thyra::ModelEvaluator<ST>

Teuchos::RCP<const Thyra_VectorSpace>
ModelEvaluator::get_x_space() const
{
  return app->getVectorSpace();
}

Teuchos::RCP<const Thyra_VectorSpace>
ModelEvaluator::get_f_space() const
{
  return app->getVectorSpace();
}

Teuchos::RCP<const Thyra_VectorSpace>
ModelEvaluator::get_p_space(int l) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(
      l >= num_param_vecs + num_dist_param_vecs || l < 0,
      Teuchos::Exceptions::InvalidParameter,
      std::endl
          << "Error!  Albany::ModelEvaluator::get_p_space():  "
          << "Invalid parameter index l = "
          << l
          << std::endl);
  Teuchos::RCP<const Thyra_VectorSpace> vs;
  if (l < num_param_vecs) {
    vs = param_vss[l];
  } else {
    vs = distParamLib->get(dist_param_names[l-num_param_vecs])->vector_space();
  }
  return vs;
}

Teuchos::RCP<const Thyra_VectorSpace>
ModelEvaluator::get_g_space(int l) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(
      l >= app->getNumResponses() || l < 0,
      Teuchos::Exceptions::InvalidParameter,
      std::endl
          << "Error!  Albany::ModelEvaluator::get_g_space():  "
          << "Invalid response index l = "
          << l
          << std::endl);

  return app->getResponse(l)->responseVectorSpace();
}

Teuchos::RCP<const Teuchos::Array<std::string>>
ModelEvaluator::get_p_names(int l) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(
      l >= num_param_vecs + num_dist_param_vecs || l < 0,
      Teuchos::Exceptions::InvalidParameter,
      std::endl
          << "Error!  Albany::ModelEvaluator::get_p_names():  "
          << "Invalid parameter index l = "
          << l
          << std::endl);

  if (l < num_param_vecs) return param_names[l];
  return Teuchos::rcp(
      new Teuchos::Array<std::string>(1, dist_param_names[l - num_param_vecs]));
}

Teuchos::RCP<Thyra_LinearOp>
ModelEvaluator::create_W_op() const
{
  return app->getDisc()->createJacobianOp();
}

Teuchos::RCP<Thyra_Preconditioner>
ModelEvaluator::create_W_prec() const
{
  Teuchos::RCP<Thyra::DefaultPreconditioner<ST>> W_prec  = Teuchos::rcp(new Thyra::DefaultPreconditioner<ST>);
  Teuchos::RCP<Thyra_LinearOp>                   precOp  = app->getPreconditioner();

  W_prec->initializeRight(precOp);
  return W_prec;
}

Teuchos::RCP<Thyra_LinearOp>
ModelEvaluator::create_DfDp_op_impl(int j) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(
      j >= num_param_vecs + num_dist_param_vecs || j < num_param_vecs,
      Teuchos::Exceptions::InvalidParameter,
      std::endl
          << "Error!  Albany::ModelEvaluator::create_DfDp_op_impl():  "
          << "Invalid parameter index j = "
          << j
          << std::endl);

  return Teuchos::rcp( new DistributedParameterDerivativeOp(app, dist_param_names[j - num_param_vecs]) );
}

Teuchos::RCP<const Thyra_LOWS_Factory>
ModelEvaluator::get_W_factory() const
{
  return Teuchos::null;
}

Thyra_ModelEvaluator::InArgs<ST>
ModelEvaluator::createInArgs() const
{
  return this->createInArgsImpl();
}

void
ModelEvaluator::reportFinalPoint(
    const Thyra_ModelEvaluator::InArgs<ST>& finalPoint,
    const bool                              wasSolved)
{
  // Set nominal values to the final point, if the model was solved
  if (overwriteNominalValuesWithFinalPoint && wasSolved) {
    nominalValues = finalPoint;
  }
}

Teuchos::RCP<Thyra_LinearOp>
ModelEvaluator::create_DgDx_op_impl(int j) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(
      j >= app->getNumResponses() || j < 0,
      Teuchos::Exceptions::InvalidParameter,
      std::endl
          << "Error!  Albany::ModelEvaluator::create_DgDx_op_impl():  "
          << "Invalid response index j = "
          << j
          << std::endl);

  return app->getResponse(j)->createGradientOp();
}

// AGS: x_dotdot time integrators not imlemented in Thyra ME yet
Teuchos::RCP<Thyra_LinearOp>
ModelEvaluator::create_DgDx_dotdot_op_impl(int j) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(
      j >= app->getNumResponses() || j < 0,
      Teuchos::Exceptions::InvalidParameter,
      std::endl
          << "Error!  Albany::ModelEvaluator::create_DgDx_dotdot_op():  "
          << "Invalid response index j = "
          << j
          << std::endl);

  return app->getResponse(j)->createGradientOp();
}

Teuchos::RCP<Thyra_LinearOp>
ModelEvaluator::create_DgDx_dot_op_impl(int j) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(
      j >= app->getNumResponses() || j < 0,
      Teuchos::Exceptions::InvalidParameter,
      std::endl
          << "Error!  Albany::ModelEvaluator::create_DgDx_dot_op_impl():  "
          << "Invalid response index j = "
          << j
          << std::endl);

  return app->getResponse(j)->createGradientOp();
}

Thyra_OutArgs ModelEvaluator::createOutArgsImpl() const
{
  Thyra_ModelEvaluator::OutArgsSetup<ST> result;
  result.setModelEvalDescription(this->description());

  const int n_g = app->getNumResponses();
  result.set_Np_Ng(num_param_vecs + num_dist_param_vecs, n_g);

  result.setSupports(Thyra_ModelEvaluator::OUT_ARG_f, true);

  if (supplies_prec)
    result.setSupports(Thyra_ModelEvaluator::OUT_ARG_W_prec, true);

  result.setSupports(Thyra_ModelEvaluator::OUT_ARG_W_op, true);
  result.set_W_properties(Thyra_ModelEvaluator::DerivativeProperties(
      Thyra_ModelEvaluator::DERIV_LINEARITY_UNKNOWN,
      Thyra_ModelEvaluator::DERIV_RANK_FULL,
      true));

  for (int l = 0; l < num_param_vecs; ++l) {
    result.setSupports(
        Thyra_ModelEvaluator::OUT_ARG_DfDp,
        l,
        Thyra_ModelEvaluator::DERIV_MV_BY_COL);
  }
  for (int i = 0; i < num_dist_param_vecs; i++)
    result.setSupports(
        Thyra_ModelEvaluator::OUT_ARG_DfDp,
        i + num_param_vecs,
        Thyra_ModelEvaluator::DERIV_LINEAR_OP);

  for (int i = 0; i < n_g; ++i) {
    Thyra_ModelEvaluator::DerivativeSupport dgdx_support;
    if (app->getResponse(i)->isScalarResponse()) {
      dgdx_support = Thyra_ModelEvaluator::DERIV_TRANS_MV_BY_ROW;
    } else {
      dgdx_support = Thyra_ModelEvaluator::DERIV_LINEAR_OP;
    }

    result.setSupports(
        Thyra_ModelEvaluator::OUT_ARG_DgDx, i, dgdx_support);
    if (supports_xdot) {
      result.setSupports(
          Thyra_ModelEvaluator::OUT_ARG_DgDx_dot, i, dgdx_support);
    }

    // AGS: x_dotdot time integrators not imlemented in Thyra ME yet
    // result.setSupports(
    //    Thyra_ModelEvaluator::OUT_ARG_DgDx_dotdot, i, dgdx_support);

    for (int l = 0; l < num_param_vecs; l++)
      result.setSupports(
          Thyra_ModelEvaluator::OUT_ARG_DgDp,
          i,
          l,
          Thyra_ModelEvaluator::DERIV_MV_BY_COL);

    if (app->getResponse(i)->isScalarResponse()) {
      for (int j = 0; j < num_dist_param_vecs; j++)
        result.setSupports(
            Thyra_ModelEvaluator::OUT_ARG_DgDp,
            i,
            j + num_param_vecs,
            Thyra_ModelEvaluator::DERIV_TRANS_MV_BY_ROW);
    } else {
      for (int j = 0; j < num_dist_param_vecs; j++)
        result.setSupports(
            Thyra_ModelEvaluator::OUT_ARG_DgDp,
            i,
            j + num_param_vecs,
            Thyra_ModelEvaluator::DERIV_LINEAR_OP);
    }
  }

  return static_cast<Thyra_OutArgs>(result);
}

void ModelEvaluator::
evalModelImpl(const Thyra_InArgs&  inArgs,
              const Thyra_OutArgs& outArgs) const
{
#ifdef OUTPUT_TO_SCREEN
  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";
#endif

  Teuchos::TimeMonitor Timer(*timer);  // start timer
  //
  // Get the input arguments
  //

  //! If a parameter has changed in value, saved/unsaved fields must be updated
  auto out = Teuchos::VerboseObjectBase::getDefaultOStream();
  auto analysisParams = appParams->sublist("Piro").sublist("Analysis");
  if(analysisParams.isSublist("Optimization Status")) {
    auto& opt_paramList = analysisParams.sublist("Optimization Status");
    if(opt_paramList.isParameter("Optimization Variables Changed") && opt_paramList.get<bool>("Optimization Variables Changed")) {
      if(opt_paramList.isParameter("Parameter Names")) {
        auto& param_names = *opt_paramList.get<Teuchos::RCP<std::vector<std::string>>>("Parameter Names");
        for (int k=0; k < param_names.size(); ++k) {
          *out << param_names[k] << " has changed!" << std::endl;
          app->getPhxSetup()->init_unsaved_param(param_names[k]);
        }
      }
      opt_paramList.set("Optimization Variables Changed", false);
    }

    // When using the Old Reduced Space ROL Interface, the solution printing is 
    // handled in Piro using observers. Otherwise we take care of printing here.
    if(analysisParams.isSublist("ROL") && !analysisParams.sublist("ROL").get("Use Old Reduced Space Interface", false)) {
      int iter = opt_paramList.get("Optimizer Iteration Number", -1);
      static int iteration = -1;
      int write_interval = analysisParams.get("Write Interval",1);
      if((iter >= 0) && (iter != iteration) && (iteration%write_interval == 0))
      {
        Teuchos::TimeMonitor timer(*Teuchos::TimeMonitor::getNewTimer("Albany: Output to File"));
        const Teuchos::RCP<const Thyra_Vector> x = inArgs.get_x();
        ObserverImpl observer(app);
        observer.observeSolution(iter, *x, Teuchos::null, Teuchos::null);
        iteration = iter;
      }
    }
  }

  // Thyra vectors
  const Teuchos::RCP<const Thyra_Vector> x = inArgs.get_x();
  const Teuchos::RCP<const Thyra_Vector> x_dot =
      (supports_xdot ? inArgs.get_x_dot() : Teuchos::null);

  // IKT, 3/30/17: the following logic is meant to support both the Thyra
  // time-integrators in Piro
  //(e.g., trapezoidal rule) and the second order time-integrators in Tempus.
  Teuchos::RCP<const Thyra_Vector> x_dotdot; 
  ST                          omega = 0.0;
  if (supports_xdotdot == true) {
    if (use_tempus == true) 
      omega = inArgs.get_W_x_dot_dot_coeff();
    // The following case is to support second order time-integrators in Piro
    if (std::abs(omega) < 1.0e-14) {
      if (Teuchos::nonnull(this->get_x_dotdot())) {
        x_dotdot  =  this->get_x_dotdot(); 
        omega     = this->get_omega();
      } else {
        x_dotdot  = Teuchos::null; 
        omega     = 0.0;
      }
    }
    // The following case is for second-order time-integrators in Tempus
    else {
      if (inArgs.supports(Thyra_ModelEvaluator::IN_ARG_x_dot_dot)) {
        //x_dotdot = inArgs.get_x_dot_dot(); 
        x_dotdot = inArgs.get_x_dot_dot();
      } else {
        x_dotdot  = Teuchos::null; 
        omega     = 0.0;
      }
    }
  } else {
    x_dotdot  = Teuchos::null; 
    omega     = 0.0;
  }

  const ST alpha = (Teuchos::nonnull(x_dot) || Teuchos::nonnull(x_dotdot)) ?
                       inArgs.get_alpha() :
                       0.0;
  const ST beta = (Teuchos::nonnull(x_dot) || Teuchos::nonnull(x_dotdot)) ?
                      inArgs.get_beta() :
                      1.0;

  bool const is_dynamic =
      Teuchos::nonnull(x_dot) || Teuchos::nonnull(x_dotdot);

#if defined(ALBANY_LCM)
  ST const curr_time = is_dynamic == true ? inArgs.get_t() : getCurrentTime();
#else
  const ST curr_time =
      (Teuchos::nonnull(x_dot) || Teuchos::nonnull(x_dotdot)) ?
          inArgs.get_t() :
          0.0;
#endif  // ALBANY_LCM

  double dt = 0.0; //time step 
  if (is_dynamic == true) {
    dt = inArgs.get_step_size(); 
  }

  for (int l = 0; l < num_param_vecs+num_dist_param_vecs; ++l) {
    const Teuchos::RCP<const Thyra_Vector> p = inArgs.get_p(l);
    if (Teuchos::nonnull(p)) {

      if(l<num_param_vecs){
        auto p_constView = getLocalData(p);
        ParamVec& sacado_param_vector = sacado_param_vec[l];
        for (unsigned int k = 0; k < sacado_param_vector.size(); ++k)
          sacado_param_vector[k].baseValue = p_constView[k];
      } else {
        distParamLib->get(dist_param_names[l-num_param_vecs])->vector()->assign(*p);
      }
    }
  }

  //
  // Get the output arguments
  //
  auto f_out    = outArgs.get_f();
  auto W_op_out = outArgs.get_W_op();

  //
  // Compute the functions
  //

  // Setup Phalanx data before functions are computed
  app->getPhxSetup()->pre_eval();

#ifdef WRITE_STIFFNESS_MATRIX_TO_MM_FILE
    // IK, 4/24/15: write stiffness matrix to matrix market file
    // Warning: to read this in to MATLAB correctly, code must be run in serial.
    // Otherwise Mass will have a distributed Map which would also need to be
    // read in to MATLAB for proper reading in of Mass.
    // IMPORTANT NOTE: keep this call BEFORE the computation of the actual jacobian,
    //                 so you don't overwrite the jacobian.
    app->computeGlobalJacobian(
        0.0, 1.0, 0.0, curr_time,
        x, x_dot, x_dotdot,
        sacado_param_vec,
        Teuchos::null, W_op_out);

    writeMatrixMarket(W_op_out,"stiffness.mm");
    writeMatrixMarket(W_op_out->range(),"range_space.mm");
    writeMatrixMarket(W_op_out->domain(),"domain_space.mm");
#endif

#ifdef WRITE_MASS_MATRIX_TO_MM_FILE
    // IK, 4/24/15: write mass matrix to matrix market file
    // Warning: to read this in to MATLAB correctly, code must be run in serial.
    // Otherwise Mass will have a distributed Map which would also need to be
    // read in to MATLAB for proper reading in of Mass.
    // IMPORTANT NOTE: keep this call BEFORE the computation of the actual jacobian,
    //                 so you don't overwrite the jacobian.
    app->computeGlobalJacobian(
        1.0, 0.0, 0.0, curr_time,
        x, x_dot, x_dotdot,
        sacado_param_vec,
        Teuchos::null, W_op_out);

    writeMatrixMarket(W_op_out,"mass.mm");
    writeMatrixMarket(W_op_out->range(),"range_space.mm");
    writeMatrixMarket(W_op_out->domain(),"domain_space.mm");
#endif

  bool f_already_computed = false;

  // W matrix
  if (Teuchos::nonnull(W_op_out)) {
    app->computeGlobalJacobian(
        alpha, beta, omega, curr_time,
        x, x_dot, x_dotdot,
        sacado_param_vec,
        f_out, W_op_out, dt);
    f_already_computed = true;
  }

  // df/dp
  for (int l = 0; l < num_param_vecs; ++l) {
    const Teuchos::RCP<Thyra_MultiVector> dfdp_out =
        outArgs.get_DfDp(l).getMultiVector();

    if (Teuchos::nonnull(dfdp_out)) {
      const Teuchos::RCP<ParamVec> p_vec =
          Teuchos::rcpFromRef(sacado_param_vec[l]);

      app->computeGlobalTangent(
          0.0, 0.0, 0.0, curr_time, false,
          x, x_dot, x_dotdot, sacado_param_vec, p_vec.get(),
          Teuchos::null, Teuchos::null, Teuchos::null, Teuchos::null,
          f_out, Teuchos::null, dfdp_out);

      f_already_computed = true;
    }
  }

  // distributed df/dp
  for (int i=0; i<num_dist_param_vecs; i++) {
	  const Teuchos::RCP<Thyra_LinearOp> dfdp_out = outArgs.get_DfDp(i+num_param_vecs).getLinearOp();
    if (dfdp_out != Teuchos::null) {
      Teuchos::RCP<DistributedParameterDerivativeOp> dfdp_op =
        Teuchos::rcp_dynamic_cast<DistributedParameterDerivativeOp>(dfdp_out);
      dfdp_op->set(curr_time, x, x_dot, x_dotdot,
                   Teuchos::rcp(&sacado_param_vec,false));
    }
  }

  // f
  if (app->is_adjoint) {
    const Thyra_Derivative f_deriv(outArgs.get_f(), Thyra_ModelEvaluator::DERIV_TRANS_MV_BY_ROW);

    const Thyra_Derivative dummy_deriv;

    const int response_index = 0;  // need to add capability for sending this in
    app->evaluateResponseDerivative(
        response_index, curr_time,
        x, x_dot, x_dotdot,
        sacado_param_vec,
        NULL,
        Teuchos::null,
        f_deriv,
        dummy_deriv,
        dummy_deriv,
        dummy_deriv);
  } else if (Teuchos::nonnull(f_out) && !f_already_computed) {
    app->computeGlobalResidual(
        curr_time,
        x, x_dot, x_dotdot,
        sacado_param_vec,
        f_out,
        dt);
  }

  // Response functions
  for (int j = 0; j < outArgs.Ng(); ++j) {
    Teuchos::RCP<Thyra_Vector> g_out = outArgs.get_g(j);

    const Thyra_Derivative dgdx_out = outArgs.get_DgDx(j);
    Thyra_Derivative dgdxdot_out;

    if (supports_xdot) {
      dgdxdot_out = outArgs.get_DgDx_dot(j);
    }

    const Thyra_Derivative dgdxdotdot_out;

    sanitize_nans(dgdx_out);
    sanitize_nans(dgdxdot_out);
    sanitize_nans(dgdxdotdot_out);

    // dg/dx, dg/dxdot
    if (!dgdx_out.isEmpty() || !dgdxdot_out.isEmpty()) {
      const Thyra_Derivative dummy_deriv;
      app->evaluateResponseDerivative(
          j, curr_time, x, x_dot, x_dotdot,
          sacado_param_vec,
          NULL,
          g_out,
          dgdx_out,
          dgdxdot_out,
          dgdxdotdot_out,
          dummy_deriv);
      // Set g_out to null to indicate that g_out was evaluated.
      g_out = Teuchos::null;
    }

    // dg/dp
    for (int l = 0; l < num_param_vecs; ++l) {
      const Teuchos::RCP<Thyra_MultiVector> dgdp_out = outArgs.get_DgDp(j, l).getMultiVector();

      if (Teuchos::nonnull(dgdp_out)) {
        const Teuchos::RCP<ParamVec> p_vec = Teuchos::rcpFromRef(sacado_param_vec[l]);

        app->evaluateResponseTangent(
            j, alpha, beta, omega, curr_time, false,
            x, x_dot, x_dotdot, sacado_param_vec, p_vec.get(),
            Teuchos::null, Teuchos::null, Teuchos::null, Teuchos::null,
            g_out, Teuchos::null, dgdp_out);
        // Set g_out to null to indicate that g_out was evaluated.
        g_out = Teuchos::null;
      }
    }

    // Need to handle dg/dp for distributed p
    for (int l = 0; l < num_dist_param_vecs; l++) {
      const Teuchos::RCP<Thyra_MultiVector> dgdp_out = outArgs.get_DgDp(j, l + num_param_vecs).getMultiVector();

      if (Teuchos::nonnull(dgdp_out)) {
        dgdp_out->assign(0.);
        app->evaluateResponseDistParamDeriv(
            j, curr_time, x, x_dot, x_dotdot,
            sacado_param_vec,
            dist_param_names[l],
            dgdp_out);
      }
    }

    if (Teuchos::nonnull(g_out)) {
      app->evaluateResponse(j, curr_time, x, x_dot, x_dotdot, sacado_param_vec, g_out);
    }
  }

#ifdef WRITE_TO_MATRIX_MARKET
  Albany::writeMatrixMarket(x, "sol", mm_counter_sol);
  ++mm_counter_sol;
#endif

#ifdef WRITE_TO_MATRIX_MARKET
  Albany::writeMatrixMarket(f_out, "res", mm_counter_res);
  ++mm_counter_res;
#endif

#ifdef WRITE_TO_MATRIX_MARKET
  Albany::writeMatrixMarket(W_op_out, "jac", mm_counter_jac);
  ++mm_counter_jac;
#endif
}

Thyra_InArgs ModelEvaluator::createInArgsImpl() const
{
  Thyra::ModelEvaluatorBase::InArgsSetup<ST> result;
  result.setModelEvalDescription(this->description());

  result.setSupports(Thyra_ModelEvaluator::IN_ARG_x, true);

#if defined(ALBANY_LCM)
  result.setSupports(Thyra_ModelEvaluator::IN_ARG_t, true);
  result.setSupports(Thyra_ModelEvaluator::IN_ARG_step_size, true);
#endif

  if (supports_xdot) {
    result.setSupports(Thyra_ModelEvaluator::IN_ARG_x_dot, true);
    result.setSupports(Thyra_ModelEvaluator::IN_ARG_t, true);
    result.setSupports(Thyra_ModelEvaluator::IN_ARG_step_size, true);
    result.setSupports(Thyra_ModelEvaluator::IN_ARG_alpha, true);
    result.setSupports(Thyra_ModelEvaluator::IN_ARG_beta, true);
  }

  if (supports_xdotdot) {
    result.setSupports(Thyra_ModelEvaluator::IN_ARG_x_dot_dot, true);
    result.setSupports(
        Thyra_ModelEvaluator::IN_ARG_W_x_dot_dot_coeff, true);
  }
  result.set_Np(num_param_vecs + num_dist_param_vecs);

  return static_cast<Thyra_InArgs>(result);
}

} // namespace Albany
