//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_ModelEvaluator.hpp"

#include "Albany_ObserverImpl.hpp"
#include "Albany_ThyraUtils.hpp"

#include "Albany_Application.hpp"

#include "Albany_TpetraThyraUtils.hpp"
#include "Albany_Hessian.hpp"
#include "Albany_Utils.hpp"
#include "Albany_StringUtils.hpp"

#include "Teuchos_ScalarTraits.hpp"
#include "Teuchos_TestForException.hpp"

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
                const Teuchos::RCP<Teuchos::ParameterList>& appParams_,
		const bool adjoint_model_)
 : app(app_)
 , appParams(appParams_)
 , supplies_prec(app_->suppliesPreconditioner())
 , supports_xdot(false)
 , supports_xdotdot(false)
 , adjoint_model(adjoint_model_)
{
  Teuchos::RCP<Teuchos::FancyOStream> out =
      Teuchos::VerboseObjectBase::getDefaultOStream();

  // Parameters (e.g., for sensitivities, SG expansions, ...)
  Teuchos::ParameterList& problemParams   = appParams->sublist("Problem");
  const Teuchos::ParameterList& parameterParams = problemParams.sublist("Parameters");

  const std::string soln_method = problemParams.get("Solution Method", "Steady"); 
  if (soln_method == "Transient") {
    use_tempus = true; 
  }

  getParameterSizes(parameterParams, total_num_param_vecs, num_param_vecs, num_dist_param_vecs);

  *out << "Total number of parameters  = " << total_num_param_vecs << std::endl;
  *out << "Number of non-distributed parameters  = " << num_param_vecs << std::endl;

  int  num_response_vecs = app->getNumResponses();

  param_names.resize(num_param_vecs);
  param_lower_bds.resize(num_param_vecs);
  param_upper_bds.resize(num_param_vecs);
  for (int l = 0; l < num_param_vecs; ++l) {
    const Teuchos::ParameterList& pList = parameterParams.sublist(util::strint("Parameter", l));

    const std::string& parameterType = pList.isParameter("Type") ?
        pList.get<std::string>("Type") : std::string("Scalar");

    if(parameterType == "Scalar") {
      param_names[l] =
          Teuchos::rcp(new Teuchos::Array<std::string>(1));
      (*param_names[l])[0] =
          pList.get<std::string>("Name");
      *out << "Number of parameters in parameter vector " << l << " = 1" << std::endl;
    }
    if(parameterType == "Vector") {
      const int numParameters = pList.get<int>("Dimension");
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
            pList.sublist(util::strint("Scalar", k)).get<std::string>("Name");
      }
      *out << "Number of parameters in parameter vector " << l << " = "
          << numParameters << std::endl;
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

    const Teuchos::ParameterList& pList = parameterParams.sublist(util::strint("Parameter",l));

    int numParameters = param_vss[l]->dim();

    // Loading lower and upper bounds (if any)
    // IKT: I believe these parameters are only relevant for optimization
    const std::string& parameterType = pList.isParameter("Type") ?
        pList.get<std::string>("Type") : std::string("Scalar");

    if(parameterType == "Scalar") {
      // Loading lower bounds (if any)
      if (pList.isParameter("Lower Bound")) {
        param_lower_bds[l] = Thyra::createMember(param_vss[l]);
        ST lb = pList.get<ST>("Lower Bound");
        TEUCHOS_TEST_FOR_EXCEPTION (1!=numParameters, Teuchos::Exceptions::InvalidParameter,
                                    "Error! numParameters!=1.\n");

        auto param_lower_bd_nonConstView = getNonconstLocalData(param_lower_bds[l]);
        param_lower_bd_nonConstView[0] = lb;
      }

      // Loading upper bounds (if any)
      if (pList.isParameter("Upper Bound")) {
        param_upper_bds[l] = Thyra::createMember(param_vss[l]);
        ST ub = pList.get<ST>("Upper Bound");
        TEUCHOS_TEST_FOR_EXCEPTION (1!=numParameters, Teuchos::Exceptions::InvalidParameter,
                                    "Error! numParameters!=1.\n");

        auto param_upper_bd_nonConstView = getNonconstLocalData(param_upper_bds[l]);
        param_upper_bd_nonConstView[0] = ub;
      }

      // Loading nominal values (if any)
      auto param_vec_nonConstView = getNonconstLocalData(param_vecs[l]);
      if (pList.isParameter("Nominal Value")) {
        ST nvals = pList.get<ST>("Nominal Value");
        TEUCHOS_TEST_FOR_EXCEPTION (1!=numParameters, Teuchos::Exceptions::InvalidParameter,
                                    "Error! numParameters!=1.\n");

        sacado_param_vec[l][0].baseValue = param_vec_nonConstView[0] = nvals;
      } else {
        param_vec_nonConstView[0] = sacado_param_vec[l][0].baseValue;
      }
    }

    if(parameterType == "Vector") {
      param_lower_bds[l] = Thyra::createMember(param_vss[l]);
      param_upper_bds[l] = Thyra::createMember(param_vss[l]);

      auto param_lower_bd_nonConstView = getNonconstLocalData(param_lower_bds[l]);
      auto param_upper_bd_nonConstView = getNonconstLocalData(param_upper_bds[l]);
      auto param_vec_nonConstView = getNonconstLocalData(param_vecs[l]);

      for (int k = 0; k < numParameters; ++k) {
        std::string sublistName = util::strint("Scalar",k);
	//IKT: I believe the following parameters are only for optimization
        if (pList.sublist(sublistName).isParameter("Lower Bound")) {
          ST lb = pList.sublist(sublistName).get<ST>("Lower Bound");
          param_lower_bd_nonConstView[k] = lb;
        }
        if (pList.sublist(sublistName).isParameter("Upper Bound")) {
          ST ub = pList.sublist(sublistName).get<ST>("Upper Bound");
          param_upper_bd_nonConstView[k] = ub;
        }
        if (pList.sublist(sublistName).isParameter("Nominal Value")) {
          ST nvals = pList.sublist(sublistName).get<ST>("Nominal Value");
          sacado_param_vec[l][k].baseValue = param_vec_nonConstView[k] = nvals;
        } else {
          param_vec_nonConstView[k] = sacado_param_vec[l][k].baseValue;
        }
      }
    }
  }

  // Setup distributed parameters
  distParamLib = app->getDistributedParameterLibrary();
  dist_param_names.resize(num_dist_param_vecs);
  *out << "Number of distributed parameters vectors  = " << num_dist_param_vecs
       << std::endl;
  std::string p_name;
  std::string  emptyString("");
  for (int i = num_param_vecs; i < total_num_param_vecs; i++) {
    const std::string& p_sublist_name = util::strint("Parameter", i);
    Teuchos::ParameterList param_list = parameterParams.sublist(p_sublist_name);

    p_name = param_list.get<std::string>("Name");

    TEUCHOS_TEST_FOR_EXCEPTION(
        !distParamLib->has(p_name),
        Teuchos::Exceptions::InvalidParameter,
        std::endl
            << "Error!  In Albany::ModelEvaluator constructor:  "
            << "Invalid distributed parameter name \""
            << p_name
            << "\""
            << std::endl);

    dist_param_names[i-num_param_vecs] = p_name;
    Teuchos::RCP<const DistributedParameter> distParam = setDistParamVec(p_name, param_list);
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

void ModelEvaluator::setNominalValue(int j, Teuchos::RCP<Thyra_Vector> p)
{
  TEUCHOS_TEST_FOR_EXCEPTION(
      j >= num_param_vecs + num_dist_param_vecs || j < 0,
      Teuchos::Exceptions::InvalidParameter,
      std::endl
          << "Error!  Albany::ModelEvaluator::setNominalValue():  "
          << "Invalid parameter index j = "
          << j
          << std::endl);

  nominalValues.set_p(j, p);
}

Teuchos::RCP<const DistributedParameter> ModelEvaluator::setDistParamVec(const std::string p_name, 
		                                                         const Teuchos::ParameterList param_list)
{
  Teuchos::RCP<const DistributedParameter> distParam = distParamLib->get(p_name);
  // set parameters bounds - IKT: I think this is only relevant for the optimization
  if (param_list.isParameter("Lower Bound")) {
    TEUCHOS_TEST_FOR_EXCEPTION(
        distParam->lower_bounds_vector() == Teuchos::null,
        Teuchos::Exceptions::InvalidParameter,
               "\nError!  In Albany::ModelEvaluator constructor:  "
            << "distParam->lower_bounds_vector() == Teuchos::null\n"); 
    distParam->lower_bounds_vector()->assign(
        param_list.get<ST>("Lower Bound"));
  }
  if (param_list.isParameter("Upper Bound")) {
    TEUCHOS_TEST_FOR_EXCEPTION(
        distParam->upper_bounds_vector() == Teuchos::null,
        Teuchos::Exceptions::InvalidParameter,
             "\nError!  In Albany::ModelEvaluator constructor:  "
             "distParam->upper_bounds_vector() == Teuchos::null\n"); 
    distParam->upper_bounds_vector()->assign(
        param_list.get<ST>("Upper Bound"));
  }
  if (param_list.isParameter("Parameter Analytic Expression")) {
    TEUCHOS_TEST_FOR_EXCEPTION(
        distParam->vector() == Teuchos::null,
        Teuchos::Exceptions::InvalidParameter,
            "\nError!  In Albany::ModelEvaluator constructor:  "
            << "distParam->vector() == Teuchos::null.\n"); 
    if (app->getComm()->getSize() > 1) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
            "\nError!  In Albany::ModelEvaluator constructor:  "
            << "'Parameter Analytic Expression' option for initializing distributed"
            << " parameters only works in serial.\n"); 
    }
    //IKT, 7/2021: the following has been added to enable verification of 
    //sensitivities for distributed parameters using MMS problems.  Other 
    //analytical expressions for distributed parameters may be added besides
    //the currently-available 'Quadratic' one, if desired.
    const std::string param_expr = param_list.get<std::string>("Parameter Analytic Expression");
    Teuchos::RCP<Albany::AbstractDiscretization> disc = app->getDisc();
    const Teuchos::ArrayRCP<double>& ov_coords = disc->getCoordinates();
    const int num_dims = app->getSpatialDimension();
    const int num_nodes = disc->getVectorSpace()->dim();
    /*const int num_dofs = ov_coords.size();
      std::cout << "IKT num_dims, num_dofs, num_nodes = " << num_dims << ", " << num_dofs
                << ", " << num_nodes << "\n"; 
    for (int i=0; i<num_dofs; i++) {
      std::cout << "IKT i, ov_coords = " << i << ", " << ov_coords[i] << "\n"; 
    }*/
    Teuchos::ArrayRCP<double> coeffs(2);
    if (param_list.isParameter("Parameter Analytic Expression Coefficients")) {
      Teuchos::Array<double> coeffs_array = param_list.get<Teuchos::Array<double>>("Parameter Analytic Expression Coefficients");
      if (coeffs_array.size() != 2) {
        TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
              "\nError!  In Albany::ModelEvaluator constructor:  "
              << "'Parameter Analytic Expression Coefficients' array must have size 2."
              << " You have provided an array of size " << coeffs_array.size() << ".\n"); 
      }
      coeffs[0] = coeffs_array[0];
      coeffs[1] = coeffs_array[1];
    }
    else {
      coeffs[0] = 1.0; 
      coeffs[1] = 0.0;
    }

    Teuchos::ArrayRCP<ST> distParamVec_ArrayRCP = 
       getNonconstLocalData(distParam->vector()); 

    if (param_expr == "Linear") 
    {
      if (num_dims == 1) {
        for (int i=0; i < num_nodes; i++) {
          const double x = ov_coords[i]; 
          distParamVec_ArrayRCP[i] = x + coeffs[0]; 
        }
      }
      else if (num_dims == 2) {
        for (int i=0; i < num_nodes; i++) {
          const double x = ov_coords[2*i]; 
          const double y = ov_coords[2*i+1]; 
          distParamVec_ArrayRCP[i] = (x + coeffs[0]) + (y + coeffs[1]); 
        }
      }
      else {
        TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
            "\nError!  In Albany::ModelEvaluator constructor:  "
            << "Linear Parameter Analytic Expression not valid for >2D.\n"); 
      }
    }
    else if (param_expr == "Quadratic") 
    {
      if (num_dims == 1) {
        for (int i=0; i < num_nodes; i++) {
          const double x = ov_coords[i]; 
          distParamVec_ArrayRCP[i] = x*(coeffs[0]-x) + coeffs[1]; 
        }
      }
      else if (num_dims == 2) {
        for (int i=0; i < num_nodes; i++) {
          const double x = ov_coords[2*i]; 
          const double y = ov_coords[2*i+1]; 
          distParamVec_ArrayRCP[i] = x*(coeffs[0]-x)*y*(coeffs[0]-y) + coeffs[1]; 
        }
      }
      else {
        TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
            "\nError!  In Albany::ModelEvaluator constructor:  "
            << "Quadratic Parameter Analytic Expression not valid for >2D.\n"); 
      }
    }
    else {
      TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
            "\nError!  In Albany::ModelEvaluator constructor:  "
            << "Invalid value for 'Parameter Analytic Expression' = "
            << param_expr << ".  Valid expressions are: 'Linear', 'Quadratic'.\n");
    }
  } 
    
  if (param_list.isParameter("Initial Uniform Value")) {
    TEUCHOS_TEST_FOR_EXCEPTION(
        distParam->vector() == Teuchos::null,
        Teuchos::Exceptions::InvalidParameter,
            "\nError!  In Albany::ModelEvaluator constructor:  "
            << "distParam->vector() == Teuchos::null.\n"); 
    distParam->vector()->assign(
        param_list.get<ST>("Initial Uniform Value"));
  }
  
  return distParam; 
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
    // of Piro::TransientDecorator<ST>
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
ModelEvaluator::create_hess_g_pp( int j, int l1, int l2 ) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(
      l1 != l2,
      Teuchos::Exceptions::InvalidParameter,
      std::endl
          << "Error! Albany::ModelEvaluator::create_hess_g_pp():  "
          << "Parameter index l1 is not equal to l2"
          << "l1 = " << l1
          << "l2 = " << l2
          << std::endl);

  auto pl = app->getProblemPL()->sublist("Hessian").sublist(util::strint("Response",j)).sublist(util::strint("Parameter",l1));
  bool HessVecProdBasedOp = pl.get("Reconstruct H_pp using Hessian-vector products",true);

  if (l1 < num_param_vecs) {
    TEUCHOS_TEST_FOR_EXCEPTION(!HessVecProdBasedOp, std::logic_error, 
            std::endl
                << "Error!  Albany::ModelEvaluator::create_hess_g_pp():  "
                << "Hessian pp operator for response " << j << " and non-distributed parameter " << l1 << " can only be reconstructed via Hessian-vector products"
                << std::endl); 
    return Albany::createDenseHessianLinearOp(param_vss[l1]);
  } else {

    // distributed parameters
    TEUCHOS_TEST_FOR_EXCEPTION(
        l1 >= num_param_vecs + num_dist_param_vecs || l1 < num_param_vecs,
        Teuchos::Exceptions::InvalidParameter,
        std::endl
            << "Error!  Albany::ModelEvaluator::create_hess_g_pp():  "
            << "Invalid parameter index l1 = "
            << l1
            << std::endl);

    if(HessVecProdBasedOp) {
      const auto p = distParamLib->get(dist_param_names[l1-num_param_vecs]);
      return Albany::createSparseHessianLinearOp(p);
    } else {
      Teuchos::RCP<Thyra_LinearOp> linOp = app->getResponse(j)->get_Hess_pp_operator(dist_param_names[l1 - num_param_vecs]);
      TEUCHOS_TEST_FOR_EXCEPTION(Teuchos::is_null(linOp), std::logic_error, 
            std::endl
                << "Error!  Albany::ModelEvaluator::create_hess_g_pp():  "
                << "Hessian pp operator not defined for response " << j << " and parameter " << dist_param_names[l1 - num_param_vecs]
                << std::endl); 
      return linOp;
    }
  }
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
  Application::SolutionStatus status = wasSolved ? Application::SolutionStatus::Converged : Application::SolutionStatus::NotConverged;
  app->setSolutionStatus(status);
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

// AGS: x_dotdot time integrators not implemented in Thyra ME yet
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
        Thyra_ModelEvaluator::DERIV_MV_JACOBIAN_FORM);
  }
  for (int i = 0; i < num_dist_param_vecs; i++)
    result.setSupports(
        Thyra_ModelEvaluator::OUT_ARG_DfDp,
        i + num_param_vecs,
        Thyra_ModelEvaluator::DERIV_LINEAR_OP);

  for (int i = 0; i < n_g; ++i) {
    Thyra_ModelEvaluator::DerivativeSupport dgdx_support;
    //Check that responses are scalar; throw an error if they are not, 
    //as distributed responses are not supported yet.
    if (!app->getResponse(i)->isScalarResponse()) {
      TEUCHOS_TEST_FOR_EXCEPTION(
          true,
          std::logic_error,
          std::endl
              << "Error!  Albany::ModelEvaluator::createOutArgsImpl():  "
              << "The response associated to the index i = "
              << i
              << " is not a scalar response. Only scalar responses are currently supported."
              << std::endl);
    }
    if (app->getResponse(i)->isScalarResponse()) {
      dgdx_support = Thyra_ModelEvaluator::DERIV_MV_GRADIENT_FORM;
    } else {
      //IKT 6/30/2021: note that this case will not get hit ever because
      //distributed responses are not supported in Albany yet
      dgdx_support = Thyra_ModelEvaluator::DERIV_LINEAR_OP;
    }

    result.setSupports(
        Thyra_ModelEvaluator::OUT_ARG_DgDx, i, dgdx_support);
    if (supports_xdot) {
      result.setSupports(
          Thyra_ModelEvaluator::OUT_ARG_DgDx_dot, i, dgdx_support);
    }

    // AGS: x_dotdot time integrators not implemented in Thyra ME yet
    // result.setSupports(
    //    Thyra_ModelEvaluator::OUT_ARG_DgDx_dotdot, i, dgdx_support);

    for (int l1 = 0; l1 < num_param_vecs; l1++) {
      result.setSupports(
          Thyra_ModelEvaluator::OUT_ARG_DgDp,
          i,
          l1,
          Thyra_ModelEvaluator::DERIV_MV_JACOBIAN_FORM);
    }

    if (app->getResponse(i)->isScalarResponse()) {
      for (int j1 = 0; j1 < num_dist_param_vecs; j1++) {
        result.setSupports(
            Thyra_ModelEvaluator::OUT_ARG_DgDp,
            i,
            j1 + num_param_vecs,
            Thyra_ModelEvaluator::DERIV_MV_GRADIENT_FORM);
      }
    } 
  }

  // Set Hessian-related supports:

  const Teuchos::ParameterList& hessParams = appParams->sublist("Problem").sublist("Hessian");

  // Default value:
  const bool dADHessVec = hessParams.isParameter("Use AD for Hessian-vector products (default)") ?
      hessParams.get<bool>("Use AD for Hessian-vector products (default)") : true;

  // Default value for residual:
  bool dADHessVec_f; 
  
  if(hessParams.isSublist("Residual")) {
    dADHessVec_f = hessParams.sublist("Residual").isParameter("Use AD for Hessian-vector products (default)") ?
      hessParams.sublist("Residual").get<bool>("Use AD for Hessian-vector products (default)") : dADHessVec;
  }
  else {
    dADHessVec_f = dADHessVec;
  }

  const int num_params = num_param_vecs + num_dist_param_vecs;

  bool aDHessVec_f[num_params + 1][num_params + 1];
  bool aDHessVec_g[num_params + 1][num_params + 1];

  aDHessVec_f[0][0] = dADHessVec_f;

  for (int j1 = 0; j1 < num_params; j1++) {
    aDHessVec_f[0][j1+1] = dADHessVec_f;
    aDHessVec_f[j1+1][0] = dADHessVec_f;
    for (int j2 = 0; j2 < num_params; j2++) {
      aDHessVec_f[j1+1][j2+1] = dADHessVec_f;
    }
  }

  if(hessParams.isSublist("Residual")) {
    std::string toDisable = hessParams.sublist("Residual").isParameter("Disable AD for Hessian-vector product contributions of") ?
      hessParams.sublist("Residual").get<std::string>("Disable AD for Hessian-vector product contributions of") : "";
    std::string toEnable = hessParams.sublist("Residual").isParameter("Enable AD for Hessian-vector product contributions of") ?
      hessParams.sublist("Residual").get<std::string>("Enable AD for Hessian-vector product contributions of") : "";

    std::vector<std::string> toDisableVec, toEnableVec;

    util::splitStringOnDelim(toDisable,' ',toDisableVec);
    util::splitStringOnDelim(toEnable,' ',toEnableVec);

    for (auto toDisableEntry : toDisableVec)
      for (auto toEnableEntry : toEnableVec)
        TEUCHOS_TEST_FOR_EXCEPTION(
            toDisableEntry.compare(toEnableEntry)==0,
            Teuchos::Exceptions::InvalidParameter,
            std::endl
                << "Error!  Albany::ModelEvaluator::createOutArgsImpl():  "
                << "Hessian contribution of the residual = "
                << toDisableEntry
                << " is set both in the AD enabled and disabled list."
                << std::endl);
    int i1, i2;
    for (auto toDisableEntry : toDisableVec) {
      Albany::getHessianBlockIDs(i1, i2, toDisableEntry);
      TEUCHOS_TEST_FOR_EXCEPTION(
        i1 > num_params || i2 > num_params || i1 < 0 || i2 < 0,
        Teuchos::Exceptions::InvalidParameter,
        std::endl
            << "Error!  Albany::ModelEvaluator::createOutArgsImpl():  "
            << "Hessian contribution of the residual = "
            << toDisableEntry
            << " has an ID out of the range: [0, "
            << num_params
            << "]"
            << std::endl);
      aDHessVec_f[i1][i2] = false;
    }
    for (auto toEnableEntry : toEnableVec) {
      Albany::getHessianBlockIDs(i1, i2, toEnableEntry);
      TEUCHOS_TEST_FOR_EXCEPTION(
        i1 > num_params || i2 > num_params || i1 < 0 || i2 < 0,
        Teuchos::Exceptions::InvalidParameter,
        std::endl
            << "Error!  Albany::ModelEvaluator::createOutArgsImpl():  "
            << "Hessian contribution of the residual = "
            << toEnableEntry
            << " has an ID out of the range: [0, "
            << num_params
            << "]"
            << std::endl);
      aDHessVec_f[i1][i2] = true;
    }

    if (num_params > 1) {
      bool tmp = aDHessVec_f[0][1];
      for (int j1 = 2; j1 < num_params + 1; j1++) {
        TEUCHOS_TEST_FOR_EXCEPTION(
          aDHessVec_f[0][j1] != tmp,
          Teuchos::Exceptions::InvalidParameter,
          std::endl
              << "Error!  Albany::ModelEvaluator::createOutArgsImpl():  "
              << "AD support for xp Hessian contributions of the residual are not consistent; "
              << "AD is enable for some blocks but not for others."
              << std::endl);
      }
      tmp = aDHessVec_f[1][0];
      for (int j1 = 2; j1 < num_params + 1; j1++) {
        TEUCHOS_TEST_FOR_EXCEPTION(
          aDHessVec_f[j1][0] != tmp,
          Teuchos::Exceptions::InvalidParameter,
          std::endl
              << "Error!  Albany::ModelEvaluator::createOutArgsImpl():  "
              << "AD support for px Hessian contributions of the residual are not consistent; "
              << "AD is enable for some blocks but not for others."
              << std::endl);
      }
      tmp = aDHessVec_f[1][1];
      for (int j1 = 2; j1 < num_params + 1; j1++) {
        for (int j2 = 2; j2 < num_params + 1; j2++) {
          TEUCHOS_TEST_FOR_EXCEPTION(
            aDHessVec_f[j1][j2] != tmp,
            Teuchos::Exceptions::InvalidParameter,
            std::endl
                << "Error!  Albany::ModelEvaluator::createOutArgsImpl():  "
                << "AD support for pp Hessian contributions of the residual are not consistent; "
                << "AD is enable for some blocks but not for others."
                << std::endl);
        }
      }
    }
  }

  result.setSupports(
    Thyra_ModelEvaluator::OUT_ARG_hess_vec_prod_f_xx,
    aDHessVec_f[0][0]);

  for (int j1 = 0; j1 < num_params; j1++) {
    result.setSupports(
      Thyra_ModelEvaluator::OUT_ARG_hess_vec_prod_f_xp,
      j1,
      aDHessVec_f[0][j1+1]);
    result.setSupports(
      Thyra_ModelEvaluator::OUT_ARG_hess_vec_prod_f_px,
      j1,
      aDHessVec_f[j1+1][0]);
    for (int j2 = 0; j2 < num_params; j2++) {
      result.setSupports(
        Thyra_ModelEvaluator::OUT_ARG_hess_vec_prod_f_pp,
        j1,
        j2,
        aDHessVec_f[j1+1][j2+1]);
    }
  }

  for (int i = 0; i < n_g; ++i) {
    // Default value for response:
    bool dADHessVec_g, supportHpp;

    if(hessParams.isSublist(util::strint("Response", i))) {
      dADHessVec_g = hessParams.sublist(util::strint("Response", i)).isParameter("Use AD for Hessian-vector products (default)") ?
        hessParams.sublist(util::strint("Response", i)).get<bool>("Use AD for Hessian-vector products (default)") : dADHessVec;
      supportHpp = hessParams.sublist(util::strint("Response", i)).isParameter("Reconstruct H_pp") ?
        hessParams.sublist(util::strint("Response", i)).get<bool>("Reconstruct H_pp") : true;
    }
    else {
      dADHessVec_g = dADHessVec;
      supportHpp = true;
    }
    
    auto& analysisParams = appParams->sublist("Piro").sublist("Analysis");
    if(analysisParams.isSublist("ROL")) {
      bool reconstructHppROL = false;
      if(analysisParams.sublist("ROL").isSublist("Matrix Based Dot Product"))
        reconstructHppROL = reconstructHppROL ||
            (analysisParams.sublist("ROL").sublist("Matrix Based Dot Product").get<std::string>("Matrix Type") == "Hessian Of Response");

      if(analysisParams.sublist("ROL").isSublist("Custom Secant"))
        reconstructHppROL = reconstructHppROL ||
            (analysisParams.sublist("ROL").sublist("Custom Secant").get<std::string>("Initialization Type") == "Hessian Of Response");

      TEUCHOS_TEST_FOR_EXCEPTION(
          (supportHpp == false) && (reconstructHppROL == true),
          Teuchos::Exceptions::InvalidParameter,
          std::endl
              << "Error!  Albany::ModelEvaluator::createOutArgsImpl():  "
              << "The construction of H_pp is requested but not supported"
              << ". Please set the Option in the Hessian and ROL sublists consistently."
              << std::endl);

    }

    aDHessVec_g[0][0] = dADHessVec_g;

    for (int j1 = 0; j1 < num_params; j1++) {
      aDHessVec_g[0][j1+1] = dADHessVec_g;
      aDHessVec_g[j1+1][0] = dADHessVec_g;
      for (int j2 = 0; j2 < num_params; j2++) {
        aDHessVec_g[j1+1][j2+1] = dADHessVec_g;
      }
    }
    
    if(hessParams.isSublist(util::strint("Response", i))) {
      std::string toDisable = hessParams.sublist(util::strint("Response", i)).isParameter("Disable AD for Hessian-vector product contributions of") ?
        hessParams.sublist(util::strint("Response", i)).get<std::string>("Disable AD for Hessian-vector product contributions of") : "";
      std::string toEnable = hessParams.sublist(util::strint("Response", i)).isParameter("Enable AD for Hessian-vector product contributions of") ?
        hessParams.sublist(util::strint("Response", i)).get<std::string>("Enable AD for Hessian-vector product contributions of") : "";

      std::vector<std::string> toDisableVec, toEnableVec;

      util::splitStringOnDelim(toDisable,' ',toDisableVec);
      util::splitStringOnDelim(toEnable,' ',toEnableVec);

      for (auto toDisableEntry : toDisableVec)
        for (auto toEnableEntry : toEnableVec)
          TEUCHOS_TEST_FOR_EXCEPTION(
              toDisableEntry.compare(toEnableEntry)==0,
              Teuchos::Exceptions::InvalidParameter,
              std::endl
                  << "Error!  Albany::ModelEvaluator::createOutArgsImpl():  "
                  << "Hessian contribution of the response " << i << " = "
                  << toDisableEntry
                  << " is set both in the AD enabled and disabled list."
                  << std::endl);
      int i1, i2;
      for (auto toDisableEntry : toDisableVec) {
        Albany::getHessianBlockIDs(i1, i2, toDisableEntry);
        TEUCHOS_TEST_FOR_EXCEPTION(
          i1 > num_params || i2 > num_params || i1 < 0 || i2 < 0,
          Teuchos::Exceptions::InvalidParameter,
          std::endl
              << "Error!  Albany::ModelEvaluator::createOutArgsImpl():  "
              << "Hessian contribution of the response " << i << " = "
              << toDisableEntry
              << " has an ID out of the range: [0, "
              << num_params
              << "]"
              << std::endl);
        aDHessVec_g[i1][i2] = false;
      }
      for (auto toEnableEntry : toEnableVec) {
        Albany::getHessianBlockIDs(i1, i2, toEnableEntry);
        TEUCHOS_TEST_FOR_EXCEPTION(
          i1 > num_params || i2 > num_params || i1 < 0 || i2 < 0,
          Teuchos::Exceptions::InvalidParameter,
          std::endl
              << "Error!  Albany::ModelEvaluator::createOutArgsImpl():  "
              << "Hessian contribution of the response " << i << " = "
              << toEnableEntry
              << " has an ID out of the range: [0, "
              << num_params
              << "]"
              << std::endl);
        aDHessVec_g[i1][i2] = true;
      }

      if (num_params > 1) {
        bool tmp = aDHessVec_g[0][1];
        for (int j1 = 2; j1 < num_params + 1; j1++) {
          TEUCHOS_TEST_FOR_EXCEPTION(
            aDHessVec_g[0][j1] != tmp,
            Teuchos::Exceptions::InvalidParameter,
            std::endl
                << "Error!  Albany::ModelEvaluator::createOutArgsImpl():  "
                << "AD support for xp Hessian contributions of the response " << i << " are not consistent; "
                << "AD is enable for some blocks but not for others."
                << std::endl);
        }
        tmp = aDHessVec_g[1][0];
        for (int j1 = 2; j1 < num_params + 1; j1++) {
          TEUCHOS_TEST_FOR_EXCEPTION(
            aDHessVec_g[j1][0] != tmp,
            Teuchos::Exceptions::InvalidParameter,
            std::endl
                << "Error!  Albany::ModelEvaluator::createOutArgsImpl():  "
                << "AD support for px Hessian contributions of the response " << i << " are not consistent; "
                << "AD is enable for some blocks but not for others."
                << std::endl);
        }
        tmp = aDHessVec_g[1][1];
        for (int j1 = 2; j1 < num_params + 1; j1++) {
          for (int j2 = 2; j2 < num_params + 1; j2++) {
            TEUCHOS_TEST_FOR_EXCEPTION(
              aDHessVec_g[j1][j2] != tmp,
              Teuchos::Exceptions::InvalidParameter,
              std::endl
                  << "Error!  Albany::ModelEvaluator::createOutArgsImpl():  "
                  << "AD support for pp Hessian contributions of the response " << i << " are not consistent; "
                  << "AD is enable for some blocks but not for others."
                  << std::endl);
          }
        }
      }
    }

    result.setSupports(
      Thyra_ModelEvaluator::OUT_ARG_hess_vec_prod_g_xx,
      i,
      aDHessVec_g[0][0]);

    for (int j1 = 0; j1 < num_params; j1++) {
      result.setSupports(
        Thyra_ModelEvaluator::OUT_ARG_hess_vec_prod_g_xp,
        i,
        j1,
        aDHessVec_g[0][j1+1]);
      result.setSupports(
        Thyra_ModelEvaluator::OUT_ARG_hess_vec_prod_g_px,
        i,
        j1,
        aDHessVec_g[j1+1][0]);
      result.setSupports(
        Thyra_ModelEvaluator::OUT_ARG_hess_g_pp,
        i,
        j1,
        j1,
        supportHpp);
      for (int j2 = 0; j2 < num_params; j2++) {
        result.setSupports(
          Thyra_ModelEvaluator::OUT_ARG_hess_vec_prod_g_pp,
          i,
          j1,
          j2,
          aDHessVec_g[j1+1][j2+1]);
      }
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

  auto out = Teuchos::VerboseObjectBase::getDefaultOStream();

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

  const ST curr_time =
      (Teuchos::nonnull(x_dot) || Teuchos::nonnull(x_dotdot)) ?
          inArgs.get_t() :
          0.0;

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
        for (unsigned int k = 0; k < sacado_param_vector.size(); ++k) {
          sacado_param_vector[k].baseValue = p_constView[k];
          sacado_param_vector[k].family->setRealValueForAllTypes(sacado_param_vector[k].baseValue);
        }
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
    if(adjoint_model) {
      TEUCHOS_FUNC_TIME_MONITOR("Albany Transpose Jacobian");
      Albany::transpose(W_op_out);
    }

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
          x, x_dot, x_dotdot, sacado_param_vec, l,
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
  if (Teuchos::nonnull(f_out) && !f_already_computed) {
    app->computeGlobalResidual(
        curr_time,
        x, x_dot, x_dotdot,
        sacado_param_vec,
        f_out,
        dt);
  }

  // Need to handle hess_vec_prod_f

  const Teuchos::RCP<const Thyra_MultiVector> delta_x = inArgs.get_x_direction();
  const Teuchos::RCP<const Thyra_Vector> z = inArgs.get_f_multiplier();
  const Teuchos::RCP<Thyra_MultiVector> f_hess_xx_v =
    outArgs.supports(Thyra_ModelEvaluator::OUT_ARG_hess_vec_prod_f_xx) ?
    outArgs.get_hess_vec_prod_f_xx() : Teuchos::null;

  if (Teuchos::nonnull(f_hess_xx_v)) {
    TEUCHOS_TEST_FOR_EXCEPTION(adjoint_model, std::logic_error,
        std::endl << "ModelEvaluator::evalModelImpl " <<
        " adjoint Hessian not implemented." << std::endl);
    if (delta_x.is_null()) {
      TEUCHOS_TEST_FOR_EXCEPTION(
          true,
          Teuchos::Exceptions::InvalidParameter,
          "hess_vec_prod_f_xx() is set in outArgs but x_direction "
          << "is not set in inArgs.\n");
    }
    f_hess_xx_v->assign(0.);
    app->evaluateResidual_HessVecProd_xx(
        curr_time, delta_x, z, x, x_dot, x_dotdot,
        sacado_param_vec,
        f_hess_xx_v);
  }

  const int num_params = num_param_vecs + num_dist_param_vecs;
  std::vector<std::string> all_param_names(num_params);

  for (int l1 = 0; l1 < num_param_vecs; l1++) {
    all_param_names[l1] = util::strint("parameter_vector", l1);
  }
  for (int l1 = 0; l1 < num_dist_param_vecs; l1++) {
    all_param_names[l1 + num_param_vecs] = dist_param_names[l1];
  }

  for (int l1 = 0; l1 < num_params; l1++) {
    const Teuchos::RCP<const Thyra_MultiVector> delta_p_l1 = inArgs.get_p_direction(l1);
    const Teuchos::RCP<Thyra_MultiVector> f_hess_xp_v =
      outArgs.supports(Thyra_ModelEvaluator::OUT_ARG_hess_vec_prod_f_xp, l1) ?
      outArgs.get_hess_vec_prod_f_xp(l1) : Teuchos::null;
    const Teuchos::RCP<Thyra_MultiVector> f_hess_px_v =
      outArgs.supports(Thyra_ModelEvaluator::OUT_ARG_hess_vec_prod_f_px, l1) ?
      outArgs.get_hess_vec_prod_f_px(l1) : Teuchos::null;

    if (Teuchos::nonnull(f_hess_xp_v)) {
      TEUCHOS_TEST_FOR_EXCEPTION(adjoint_model, std::logic_error,
          std::endl << "ModelEvaluator::evalModelImpl " <<
          " adjoint Hessian not implemented." << std::endl);
      if (delta_p_l1.is_null()) {
        TEUCHOS_TEST_FOR_EXCEPTION(
            true,
            Teuchos::Exceptions::InvalidParameter,
            "hess_vec_prod_f_xp(" << l1 <<") is set in outArgs but "
            << "p_direction(" << l1  << ") is not set in inArgs.\n");
      }
      f_hess_xp_v->assign(0.);
      app->evaluateResidual_HessVecProd_xp(
          curr_time, delta_p_l1, z, x, x_dot, x_dotdot,
          sacado_param_vec,
          all_param_names[l1],
          f_hess_xp_v);
    }

    if (Teuchos::nonnull(f_hess_px_v)) {
      TEUCHOS_TEST_FOR_EXCEPTION(adjoint_model, std::logic_error,
          std::endl << "ModelEvaluator::evalModelImpl " <<
          " adjoint Hessian not implemented." << std::endl);
      if (delta_x.is_null()) {
        TEUCHOS_TEST_FOR_EXCEPTION(
            true,
            Teuchos::Exceptions::InvalidParameter,
            "hess_vec_prod_f_px(" << l1 <<") is set in outArgs but "
            << "x_direction is not set in inArgs.\n");
      }
      f_hess_px_v->assign(0.);
      app->evaluateResidual_HessVecProd_px(
          curr_time, delta_x, z, x, x_dot, x_dotdot,
          sacado_param_vec,
          all_param_names[l1],
          f_hess_px_v);
    }

    for (int l2 = 0; l2 < num_params; l2++) {
      const Teuchos::RCP<const Thyra_MultiVector> delta_p_l2 = inArgs.get_p_direction(l2);
      const Teuchos::RCP<Thyra_MultiVector> f_hess_pp_v =
        outArgs.supports(Thyra_ModelEvaluator::OUT_ARG_hess_vec_prod_f_pp, l1, l2) ?
        outArgs.get_hess_vec_prod_f_pp(l1, l2) : Teuchos::null;

      if (Teuchos::nonnull(f_hess_pp_v)) {
        if (delta_p_l2.is_null()) {
          TEUCHOS_TEST_FOR_EXCEPTION(
              true,
              Teuchos::Exceptions::InvalidParameter,
              "hess_vec_prod_f_pp(" << l1 <<","<< l2
              << ") is set in outArgs but p_direction(" << l2 << ") is not set in inArgs.\n");
        }
        f_hess_pp_v->assign(0.);
        app->evaluateResidual_HessVecProd_pp(
            curr_time, delta_p_l2, z, x, x_dot, x_dotdot,
            sacado_param_vec,
            all_param_names[l1],
            all_param_names[l2],
            f_hess_pp_v);
      }
    }
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
          -1,
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
            j, l, alpha, beta, omega, curr_time, false,
            x, x_dot, x_dotdot, sacado_param_vec,
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

    // Need to handle hess_vec_prod_g

    const Teuchos::RCP<const Thyra_MultiVector> delta_x = inArgs.get_x_direction();
    const Teuchos::RCP<Thyra_MultiVector> g_hess_xx_v =
      outArgs.supports(Thyra_ModelEvaluator::OUT_ARG_hess_vec_prod_g_xx, j) ?
      outArgs.get_hess_vec_prod_g_xx(j) : Teuchos::null;

    if (Teuchos::nonnull(g_hess_xx_v)) {
      if (delta_x.is_null()) {
        TEUCHOS_TEST_FOR_EXCEPTION(
            true,
            Teuchos::Exceptions::InvalidParameter,
            "hess_vec_prod_g_xx(" << j <<") is set in outArgs but x_direction "
            << "is not set in inArgs.\n");
      }
      g_hess_xx_v->assign(0.);
      app->evaluateResponse_HessVecProd_xx(
          j, curr_time, delta_x, x, x_dot, x_dotdot,
          sacado_param_vec,
          g_hess_xx_v);
    }

    for (int l1 = 0; l1 < num_params; l1++) {
      const Teuchos::RCP<const Thyra_MultiVector> delta_p_l1 = inArgs.get_p_direction(l1);
      const Teuchos::RCP<Thyra_MultiVector> g_hess_xp_v =
        outArgs.supports(Thyra_ModelEvaluator::OUT_ARG_hess_vec_prod_g_xp, j, l1) ?
        outArgs.get_hess_vec_prod_g_xp(j, l1) : Teuchos::null;
      const Teuchos::RCP<Thyra_MultiVector> g_hess_px_v =
        outArgs.supports(Thyra_ModelEvaluator::OUT_ARG_hess_vec_prod_g_px, j, l1) ?
        outArgs.get_hess_vec_prod_g_px(j, l1) : Teuchos::null;

      if (Teuchos::nonnull(g_hess_xp_v)) {
        if (delta_p_l1.is_null()) {
          TEUCHOS_TEST_FOR_EXCEPTION(
              true,
              Teuchos::Exceptions::InvalidParameter,
              "hess_vec_prod_g_xp(" << j <<","<< l1 <<") is set in outArgs but "
              << "p_direction(" << l1 << ") is not set in inArgs.\n");
        }
        g_hess_xp_v->assign(0.);
        app->evaluateResponse_HessVecProd_xp(
            j, curr_time, delta_p_l1, x, x_dot, x_dotdot,
            sacado_param_vec,
            all_param_names[l1],
            g_hess_xp_v);
      }

      if (Teuchos::nonnull(g_hess_px_v)) {
        if (delta_x.is_null()) {
          TEUCHOS_TEST_FOR_EXCEPTION(
              true,
              Teuchos::Exceptions::InvalidParameter,
              "hess_vec_prod_g_px(" << j <<","<< l1 <<") is set in outArgs but "
              << "x_direction is not set in inArgs.\n");
        }
        g_hess_px_v->assign(0.);
        app->evaluateResponse_HessVecProd_px(
            j, curr_time, delta_x, x, x_dot, x_dotdot,
            sacado_param_vec,
            all_param_names[l1],
            g_hess_px_v);
      }

      const Teuchos::RCP<Thyra_LinearOp> g_hess_pp =
        outArgs.supports(Thyra_ModelEvaluator::OUT_ARG_hess_g_pp, j, l1, l1) ?
        outArgs.get_hess_g_pp(j, l1, l1) : Teuchos::null;
      if (Teuchos::nonnull(g_hess_pp)) {
        auto hessParams = appParams->sublist("Problem").sublist("Hessian");
        auto hess_pp_matrix_op = Teuchos::rcp_dynamic_cast<MatrixBased_LOWS>(g_hess_pp);
        if(Teuchos::nonnull(hess_pp_matrix_op)) {
          app->evaluateResponseHessian_pp(j, l1, curr_time, x, x_dot, x_dotdot,
                    sacado_param_vec,
                    all_param_names[l1],
                    hess_pp_matrix_op->getMatrix());
          if(hessParams.get<bool>("Write Hessian MatrixMarket", false))
            Albany::writeMatrixMarket(Albany::getTpetraMatrix(hess_pp_matrix_op->getMatrix()).getConst(), "H", l1);
        }

        //Initialize Solver only if Solver sublist is present
        if(hessParams.sublist(util::strint("Response",j)).sublist(util::strint("Parameter",l1)).isSublist("H_pp Solver")) {
          auto hess_pp = Teuchos::rcp_dynamic_cast<Init_LOWS>(g_hess_pp);
          TEUCHOS_TEST_FOR_EXCEPTION(Teuchos::is_null(hess_pp), std::runtime_error, 
                  "hess_g_pp(" << j <<","<< l1  <<","<< l1 << ") is not derived from Hessian_LOWS.\n");
          auto pl = hessParams.sublist(util::strint("Response",j)).sublist(util::strint("Parameter",l1)).sublist("H_pp Solver");
          hess_pp->initializeSolver(Teuchos::rcpFromRef(pl));        
        }
      }

      for (int l2 = 0; l2 < num_params; l2++) {
        const Teuchos::RCP<const Thyra_MultiVector> delta_p_l2 = inArgs.get_p_direction(l2);
        const Teuchos::RCP<Thyra_MultiVector> g_hess_pp_v =
          outArgs.supports(Thyra_ModelEvaluator::OUT_ARG_hess_vec_prod_g_pp, j, l1, l2) ?
          outArgs.get_hess_vec_prod_g_pp(j, l1, l2) : Teuchos::null;

        if (Teuchos::nonnull(g_hess_pp_v)) {
          if (delta_p_l2.is_null()) {
            TEUCHOS_TEST_FOR_EXCEPTION(
                true,
                Teuchos::Exceptions::InvalidParameter,
                "hess_vec_prod_g_pp(" << j <<","<< l1  <<","<< l2
                << ") is set in outArgs but p_direction(" << l2  << ") is not set in inArgs.\n");
          }
          g_hess_pp_v->assign(0.);
          app->evaluateResponse_HessVecProd_pp(
              j, curr_time, delta_p_l2, x, x_dot, x_dotdot,
              sacado_param_vec,
              all_param_names[l1],
              all_param_names[l2],
              g_hess_pp_v);
        }
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
  const int n_g = app->getNumResponses();
  result.set_Np_Ng(num_param_vecs + num_dist_param_vecs, n_g);

  return static_cast<Thyra_InArgs>(result);
}

} // namespace Albany
