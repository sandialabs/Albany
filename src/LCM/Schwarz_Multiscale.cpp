//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include <sstream>

#include "Albany_ModelFactory.hpp"
#include "Albany_SolverFactory.hpp"
#include "Schwarz_CoupledJacobian.hpp"
#include "Schwarz_Multiscale.hpp"
#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "NOXSolverPrePostOperator.h"

//uncomment the following to write stuff out to matrix market to debug
#define WRITE_TO_MATRIX_MARKET

#ifdef WRITE_TO_MATRIX_MARKET
static int mm_counter_sol = 0;
static int mm_counter_res = 0;
static int mm_counter_pre = 0;
static int mm_counter_jac = 0;
#endif // WRITE_TO_MATRIX_MARKET

LCM::
SchwarzMultiscale::
SchwarzMultiscale(
    Teuchos::RCP<Teuchos::ParameterList> const & app_params,
    Teuchos::RCP<Teuchos::Comm<int> const> const & commT,
    Teuchos::RCP<Tpetra_Vector const> const & initial_guessT,
    Teuchos::RCP<Thyra::LinearOpWithSolveFactoryBase<ST> const> const &
    solver_factory)
{
  commT_ = commT;

  solver_factory_ = solver_factory;

  //IK, 2/11/15: I am assuming for now we don't have any distributed parameters.
  num_dist_params_total_ = 0;

  // Get "Coupled Schwarz" parameter sublist
  Teuchos::ParameterList &
  coupled_system_params = app_params->sublist("Coupled System");

  // Get names of individual model xml input files from problem parameterlist
  Teuchos::Array<std::string>
  model_filenames =
      coupled_system_params.get<Teuchos::Array<std::string>>(
          "Model XML Files");

  //number of models
  num_models_ = model_filenames.size();

  // Create application name-index map used for Schwarz BC.
  Teuchos::RCP<std::map<std::string, int>>
  app_name_index_map = Teuchos::rcp(new std::map<std::string, int>);

  for (auto app_index = 0; app_index < num_models_; ++app_index) {

    std::string const &
    app_name = model_filenames[app_index];

    std::pair<std::string, int>
    app_name_index = std::make_pair(app_name, app_index);

    app_name_index_map->insert(app_name_index);
  }

  //------------Determine whether to set OUT_ARG_W_prec supports or not--------
  //------------This is only relevant for matrix-free GMRES--------------------
  // Get "Piro" parameter sublist
  Teuchos::ParameterList &
  piroPL = app_params->sublist("Piro");

  w_prec_supports_ = false;

  //Check if problem is matrix-free  
  std::string const
  jacob_op = piroPL.isParameter("Jacobian Operator") == true ?
      piroPL.get<std::string>("Jacobian Operator") : "";

  //Get matrix-free preconditioner from input file
  std::string const
  mf_prec =
      coupled_system_params.isParameter("Matrix-Free Preconditioner") == true ?
      coupled_system_params.get<std::string>("Matrix-Free Preconditioner") :
      "None";

  if (mf_prec == "None") {
    mf_prec_type_ = NONE;
  }
  else if (mf_prec == "Jacobi") {
    mf_prec_type_ = JACOBI;
  }
  else if (mf_prec == "AbsRowSum") {
    mf_prec_type_ = ABS_ROW_SUM;
  }
  else if (mf_prec == "Identity") {
    mf_prec_type_ = ID;
  }
  else {
    ALBANY_ASSERT(false, "Unknown Matrix-Free Preconditioner type.");
  }

  // If using matrix-free, get NOX sublist and set "Preconditioner Type" to
  // "None" regardless  of what is specified in the input file.
  // Currently preconditioners for matrix-free  are implemented in this
  // ModelEvaluator, which requires the type to be "None".
  // Also set w_prec_supoprts_ to true if using matrix-free with a
  // preconditioner.
  if ((mf_prec != "None") && (jacob_op != "")) {
    //Set w_prec_supports_ to true
    w_prec_supports_ = true;
    //IKT, 11/14/16, FIXME: may want to add more cases, e.g., for Tempus
    if (piroPL.isSublist("NOX")) {
      Teuchos::ParameterList &noxPL = piroPL.sublist("NOX", true);
      if (noxPL.isSublist("Direction")) {
        Teuchos::ParameterList &dirPL = noxPL.sublist("Direction", true);
        if (dirPL.isSublist("Newton")) {
          Teuchos::ParameterList &newPL = dirPL.sublist("Newton", true);
          if (newPL.isSublist("Stratimikos Linear Solver")) {
            Teuchos::ParameterList &stratLSPL = newPL.sublist(
                "Stratimikos Linear Solver",
                true);
            if (stratLSPL.isSublist("Stratimikos")) {
              Teuchos::ParameterList &stratPL = stratLSPL.sublist(
                  "Stratimikos",
                  true);
              stratPL.set<std::string>("Preconditioner Type", "None");
            }
          }
        }
      }
    }
  }

  // Create a NOX status test and associated machinery for cutting the
  // global time step when the CrystalPlasticity constitutive model's state
  // update routine fails
  Teuchos::RCP<NOX::StatusTest::Generic> nox_status_test = Teuchos::rcp(
      new NOX::StatusTest::ModelEvaluatorFlag);
  Teuchos::RCP<NOX::Abstract::PrePostOperator> pre_post_operator = Teuchos::rcp(
      new NOXSolverPrePostOperator);
  Teuchos::RCP<NOXSolverPrePostOperator> nox_solver_pre_post_operator =
      Teuchos::rcp_dynamic_cast<NOXSolverPrePostOperator>(pre_post_operator);
  Teuchos::RCP<NOX::StatusTest::ModelEvaluatorFlag> statusTest =
      Teuchos::rcp_dynamic_cast<NOX::StatusTest::ModelEvaluatorFlag>(
          nox_status_test);

  // Acquire the NOX "Solver Options" and "Status Tests" parameter lists
  Teuchos::RCP<Teuchos::ParameterList> solverOptionsParameterList;
  Teuchos::RCP<Teuchos::ParameterList> statusTestsParameterList;
  if (app_params->isSublist("Piro")) {
    if (app_params->sublist("Piro").isSublist("NOX")) {
      if (app_params->sublist("Piro").sublist("NOX").isSublist(
          "Solver Options")) {
        solverOptionsParameterList = Teuchos::rcpFromRef(
            app_params->sublist("Piro").sublist("NOX").sublist(
                "Solver Options"));
      }
      if (app_params->sublist("Piro").sublist("NOX").isSublist(
          "Status Tests")) {
        statusTestsParameterList = Teuchos::rcpFromRef(
            app_params->sublist("Piro").sublist("NOX").sublist("Status Tests"));
      }
    }
  }

  if (!solverOptionsParameterList.is_null()
      && !statusTestsParameterList.is_null()) {

    // Add the model evaulator flag as a status test.
    Teuchos::ParameterList originalStatusTestParameterList =
        *statusTestsParameterList;
    Teuchos::ParameterList newStatusTestParameterList;
    newStatusTestParameterList.set<std::string>("Test Type", "Combo");
    newStatusTestParameterList.set<std::string>("Combo Type", "OR");
    newStatusTestParameterList.set<int>("Number of Tests", 2);
    newStatusTestParameterList.sublist("Test 0");
    newStatusTestParameterList.sublist("Test 0").set(
        "Test Type",
        "User Defined");
    newStatusTestParameterList.sublist("Test 0").set(
        "User Status Test",
        nox_status_test);
    newStatusTestParameterList.sublist("Test 1") =
        originalStatusTestParameterList;
    *statusTestsParameterList = newStatusTestParameterList;

    nox_solver_pre_post_operator->setStatusTest(statusTest);
    solverOptionsParameterList->set(
        "User Defined Pre/Post Operator",
        pre_post_operator);
  }

  //------------End getting of Preconditioner type-------------------------------------------------------------

  //----------------Parameters------------------------
  //Get "Problem" parameter list
  Teuchos::ParameterList &
  problem_params = app_params->sublist("Problem");

  Teuchos::RCP<Teuchos::ParameterList>
  parameter_params;

  Teuchos::RCP<Teuchos::ParameterList>
  response_params;

  num_params_total_ = 0;

  //Get "Parameters" parameter sublist, if it exists
  if (problem_params.isSublist("Parameters")) {
    parameter_params = Teuchos::rcp(
        &(problem_params.sublist("Parameters")), false);

    auto const
    num_parameters =
        parameter_params->isType<int>("Number") == true ?
            parameter_params->get<int>("Number") : 0;

    bool const
    using_old_parameter_list = num_parameters > 0 ? true : false;

    num_params_total_ =
        num_parameters > 0 ?
            1 : parameter_params->get("Number of Parameter Vectors", 0);

    //Get parameter names
    param_names_.resize(num_params_total_);
    for (auto l = 0; l < num_params_total_; ++l) {

      Teuchos::RCP<Teuchos::ParameterList const>
      p_list =
          using_old_parameter_list == true ?
              Teuchos::rcp(new Teuchos::ParameterList(*parameter_params)) :
              Teuchos::rcp(&(parameter_params->sublist(
                  Albany::strint("Parameter Vector", l))), false);

      auto const
      num_parameters = p_list->get<int>("Number");

      ALBANY_EXPECT(num_parameters > 0);

      param_names_[l] =
          Teuchos::rcp(new Teuchos::Array<std::string>(num_parameters));

      for (auto k = 0; k < num_parameters; ++k) {
        (*param_names_[l])[k] =
            p_list->get<std::string>(Albany::strint("Parameter", k));
      }
      std::cout << "Number of parameters in parameter vector ";
      std::cout << l << " = " << num_parameters << '\n';
    }
  }

  std::cout << "Number of parameter vectors = " << num_params_total_ << '\n';

  //---------------End Parameters---------------------

  //----------------Responses------------------------
  //Get "Response functions" parameter sublist
  num_responses_total_ = 0;

  if (problem_params.isSublist("Response Functions")) {
    response_params =
        Teuchos::rcp(&(problem_params.sublist("Response Functions")), false);

    auto const
    num_parameters =
        response_params->isType<int>("Number") == true ?
            response_params->get<int>("Number") : 0;

    bool const
    using_old_response_list = num_parameters > 0 ? true : false;

    num_responses_total_ =
        num_parameters > 0 ?
            1 : response_params->get("Number of Response Vectors", 0);

    Teuchos::Array<Teuchos::RCP<Teuchos::Array<std::string>>>
    response_names;

    response_names.resize(num_responses_total_);

    for (auto l = 0; l < num_responses_total_; ++l) {

      Teuchos::RCP<Teuchos::ParameterList const>
      p_list =
          using_old_response_list == true ?
              Teuchos::rcp(new Teuchos::ParameterList(*response_params)) :
              Teuchos::rcp(&(response_params->sublist(
                  Albany::strint("Response Vector", l))), false);

      auto const
      num_parameters =
          p_list->get<int>("Number") == true ?
                                               p_list->get<int>("Number") :
                                               0;

      if (num_parameters > 0) {
        response_names[l] =
            Teuchos::rcp(new Teuchos::Array<std::string>(num_parameters));

        for (auto k = 0; k < num_parameters; ++k) {
          (*response_names[l])[k] =
              p_list->get<std::string>(Albany::strint("Response", k));
        }

      }
    }
  }

  std::cout << "Number of response vectors = " << num_responses_total_ << '\n';

  //----------- end Responses-----------------------

  apps_.resize(num_models_);
  models_.resize(num_models_);
  model_app_params_.resize(num_models_);

  Teuchos::Array<Teuchos::RCP<Teuchos::ParameterList>>
  model_problem_params(num_models_);

  disc_maps_.resize(num_models_);

  jacs_.resize(num_models_);

  precs_.resize(num_models_);

  Teuchos::Array<Teuchos::RCP<Tpetra_Map const>>
  disc_overlap_maps(num_models_);

  material_dbs_.resize(num_models_);

  //Set up each application and model object in Teuchos::Array
  //(similar logic to that in Albany::SolverFactory::createAlbanyAppAndModelT)
  for (auto m = 0; m < num_models_; ++m) {

    //get parameterlist from mth model *.xml file
    Albany::SolverFactory
    solver_factory(model_filenames[m], commT_);

    // solver_factory will go out of scope, so get a copy of the PL. We take
    // ownership and give weak pointers to everyone else.
    model_app_params_[m] = Teuchos::rcp(
        new Teuchos::ParameterList(solver_factory.getParameters()));

    Teuchos::RCP<Teuchos::ParameterList>
    problem_params_m = Teuchos::sublist(model_app_params_[m], "Problem");

    // Set Parameter sublists for individual models 
    // to the parameters specified in the "master" coupled input file.
    if (parameter_params != Teuchos::null) {
      if (problem_params_m->isSublist("Parameters")) {
        std::cout << "parameters!" << '\n';
        TEUCHOS_TEST_FOR_EXCEPTION(
            true,
            std::logic_error,
            "Error in LCM::CoupledSchwarz! Model input file " <<
            model_filenames[m] <<
            " cannot have a 'Parameters' section!  " <<
            "Parameters must be specified in the 'master' input file " <<
            "driving the coupled problem.\n");
      }
      Teuchos::ParameterList &
      param_params_m = problem_params_m->sublist("Parameters", false);

      param_params_m.setParametersNotAlreadySet(*parameter_params);
    }

    // Overwrite Responses sublists for individual models,
    // if they are provided, to set them
    // to the parameters specified in the "master" coupled input file.
    if (response_params != Teuchos::null) {
      if (problem_params_m->isSublist("Response Functions")) {
        TEUCHOS_TEST_FOR_EXCEPTION(
            true,
            std::logic_error,
            "Error in LCM::CoupledSchwarz! Model input file " <<
            model_filenames[m] <<
            " cannot have a 'Response Functions' section!  " <<
            "Responses must be specified in the 'master' input file " <<
            "driving the coupled problem.\n");
      }
      Teuchos::ParameterList &
      response_params_m =
          problem_params_m->sublist("Response Functions", false);
      response_params_m.setParametersNotAlreadySet(*response_params);
    }

    model_problem_params[m] = problem_params_m;

    std::string const &
    problem_name = problem_params_m->get("Name", "");

    std::cout << "Name of problem #" << m << ": " << problem_name << '\n';

    bool const
    matdb_exists = problem_params_m->isType<std::string>("MaterialDB Filename");

    if (matdb_exists == false) {
      TEUCHOS_TEST_FOR_EXCEPTION(
          true,
          std::logic_error,
          "Error in LCM::CoupledSchwarz! " <<
          "Input file needs to have 'MaterialDB Filename' specified.\n");
    }

    std::string const &
    matdb_filename = problem_params_m->get<std::string>("MaterialDB Filename");

    material_dbs_[m] =
        Teuchos::rcp(new LCM::MaterialDatabase(matdb_filename, commT_));

    std::cout << "Materials #" << m << ": " << matdb_filename << '\n';

    // Pass these on the parameter list because the are needed before
    // BC evaluators are built.

    // Add application array for later use in Schwarz BC.
    model_app_params_[m]->set("Application Array", apps_);

    // See application index for use with Schwarz BC.
    model_app_params_[m]->set("Application Index", m);

    // App application name-index map for later use in Schwarz BC.
    model_app_params_[m]->set("Application Name Index Map", app_name_index_map);

    // Machinery for cutting the global time step from within the
    // CrystalPlasticity constitutive model
    model_app_params_[m]->sublist("Problem").set(
        "Constitutive Model NOX Status Test",
        nox_status_test);

    //create application for mth model
    apps_[m] = Teuchos::rcp(new Albany::Application(
        commT, model_app_params_[m].create_weak(), initial_guessT));

    //Create model evaluator
    Albany::ModelFactory
    model_factory(model_app_params_[m].create_weak(), apps_[m]);

    models_[m] = model_factory.createT();

    //create array of individual model jacobians
    Teuchos::RCP<Tpetra_Operator> const jac_temp =
        Teuchos::nonnull(models_[m]->create_W_op()) ?
            ConverterT::getTpetraOperator(models_[m]->create_W_op()) :
            Teuchos::null;

    jacs_[m] =
        Teuchos::nonnull(jac_temp) ?
            Teuchos::rcp_dynamic_cast<Tpetra_CrsMatrix>(jac_temp, true) :
            Teuchos::null;

    // create array of individual model preconditioners
    // these will have same graph as Jacobians for now
    precs_[m] =
        Teuchos::nonnull(jac_temp) ?
            Teuchos::rcp_dynamic_cast<Tpetra_CrsMatrix>(jac_temp, true) :
            Teuchos::null;
    if (precs_[m]->isFillActive())
      precs_[m]->fillComplete();
  }

  //Now get maps, InArgs, OutArgs for each model.
  //Calculate how many parameters, responses there are in total.
  solver_inargs_.resize(num_models_);
  solver_outargs_.resize(num_models_);

  for (auto m = 0; m < num_models_; ++m) {
    disc_maps_[m] = apps_[m]->getMapT();

    disc_overlap_maps[m] =
        apps_[m]->getStateMgr().getDiscretization()->getOverlapMapT();

    solver_inargs_[m] = models_[m]->createInArgs();
    solver_outargs_[m] = models_[m]->createOutArgs();
  }

  //----------------Parameters------------------------
  // Create sacado parameter vectors of appropriate size
  // for use in evalModelImpl
  sacado_param_vecs_.resize(num_models_);

  for (auto m = 0; m < num_models_; ++m) {
    sacado_param_vecs_[m].resize(num_params_total_);
  }

  for (auto m = 0; m < num_models_; ++m) {
    for (auto l = 0; l < num_params_total_; ++l) {
      try {
        // Initialize Sacado parameter vector
        // The following call will throw,
        // and it is often due to an incorrect input line in the
        // "Parameters" PL
        // in the input file.
        // Give the user a hint about what might be happening
        apps_[m]->getParamLib()->fillVector<PHAL::AlbanyTraits::Residual>(
            *(param_names_[l]), sacado_param_vecs_[m][l]);
      }
      catch (const std::logic_error & le) {
        std::cout << "Error: exception thrown from ParamLib fillVector in ";
        std::cout << __FILE__ << " line " << __LINE__ << '\n';
        std::cout << "This is probably due to something incorrect in the";
        std::cout << " \"Parameters\" list in the input file, ";
        std::cout << "one of the lines:" << '\n';
        for (auto k = 0; k < param_names_[l]->size(); ++k) {
          std::cout << "      " << (*param_names_[l])[k] << '\n';
        }
        throw le; // rethrow to shut things down
      }
    }
  }

  //----------- end Parameters-----------------------

  //------------------Setup nominal values----------------
  nominal_values_ = this->createInArgsImpl();

  // All the ME vectors are allocated/unallocated here
  // Calling allocateVectors() will set x and x_dot in nominal_values_
  allocateVectors();

  // set p_init in nominal_values_ --
  // create product vector that concatenates parameters from each model.
  // TODO: Check if these are correct nominal values for parameters

  for (auto l = 0; l < num_params_total_; ++l) {

    Teuchos::Array<Teuchos::RCP<Thyra::VectorSpaceBase<ST> const>>
    p_spaces(num_models_);

    for (auto m = 0; m < num_models_; ++m) {
      p_spaces[m] = models_[m]->get_p_space(l);
    }

    Teuchos::RCP<Thyra::DefaultProductVectorSpace<ST> const>
    p_space = Thyra::productVectorSpace<ST>(p_spaces);

    Teuchos::ArrayRCP<Teuchos::RCP<Thyra::VectorBase<ST> const>>
    p_vecs(num_models_);

    for (auto m = 0; m < num_models_; ++m) {
      p_vecs[m] = models_[m]->getNominalValues().get_p(l);
    }

    Teuchos::RCP<Thyra::DefaultProductVector<ST>>
    p_prod_vec = Thyra::defaultProductVector<ST>(p_space, p_vecs());

    if (Teuchos::is_null(p_prod_vec) == true) continue;

    nominal_values_.set_p(l, p_prod_vec);
  }

  //--------------End setting of nominal values------------------

}

LCM::SchwarzMultiscale::~SchwarzMultiscale()
{
}

// Overridden from Thyra::ModelEvaluator<ST>
Teuchos::RCP<Thyra::VectorSpaceBase<ST> const>
LCM::SchwarzMultiscale::get_x_space() const
{
  return getThyraDomainSpace();
}

Teuchos::RCP<Thyra::VectorSpaceBase<ST> const>
LCM::SchwarzMultiscale::get_f_space() const
{
  return getThyraRangeSpace();
}

Teuchos::RCP<Thyra::VectorSpaceBase<ST> const>
LCM::SchwarzMultiscale::getThyraRangeSpace() const
{
  if (range_space_ == Teuchos::null) {
    // loop over all vectors and build the vector space
    std::vector<Teuchos::RCP<Thyra::VectorSpaceBase<ST> const>>
    vs_array;

    for (auto m = 0; m < num_models_; ++m) {
      vs_array.push_back(
          Thyra::createVectorSpace<ST, LO, GO, KokkosNode>(disc_maps_[m]));
    }

    range_space_ = Thyra::productVectorSpace<ST>(vs_array);
  }
  return range_space_;
}

Teuchos::RCP<Thyra::VectorSpaceBase<ST> const>
LCM::SchwarzMultiscale::getThyraDomainSpace() const
{
  if (domain_space_ == Teuchos::null) {
    // loop over all vectors and build the vector space
    std::vector<Teuchos::RCP<Thyra::VectorSpaceBase<ST> const>>
    vs_array;

    for (auto m = 0; m < num_models_; ++m) {
      vs_array.push_back(
          Thyra::createVectorSpace<ST, LO, GO, KokkosNode>(disc_maps_[m]));
    }

    domain_space_ = Thyra::productVectorSpace<ST>(vs_array);
  }
  return domain_space_;
}

Teuchos::RCP<Thyra::VectorSpaceBase<ST> const>
LCM::SchwarzMultiscale::get_p_space(int l) const
{
  ALBANY_EXPECT(0 <= l && l < num_params_total_);

  std::vector<Teuchos::RCP<Thyra::VectorSpaceBase<ST> const>>
  vs_array;

  // create product space for lth parameter by concatenating lth parameter
  // from all the models.
  for (auto m = 0; m < num_models_; ++m) {
    vs_array.push_back(models_[m]->get_p_space(l));
  }

  return Thyra::productVectorSpace<ST>(vs_array);
}

Teuchos::RCP<Thyra::VectorSpaceBase<ST> const>
LCM::SchwarzMultiscale::get_g_space(int l) const
{
  ALBANY_EXPECT(0 <= l && l < num_responses_total_);

  Teuchos::Array<Teuchos::RCP<Thyra::VectorSpaceBase<ST> const>>
  vs_array;

  // create product space for lth response by concatenating lth response
  // from all the models.
  for (auto m = 0; m < num_models_; ++m) {
    vs_array.push_back(models_[m]->get_g_space(l));
  }

  Teuchos::RCP<Thyra::VectorSpaceBase<ST> const>
  Z = Thyra::productVectorSpace<ST>(vs_array);

  return Z;
}

Teuchos::RCP<const Teuchos::Array<std::string>>
LCM::SchwarzMultiscale::get_p_names(int l) const
{
  ALBANY_EXPECT(0 <= l && l < num_params_total_);
  return param_names_[l];
}

Thyra::ModelEvaluatorBase::InArgs<ST>
LCM::SchwarzMultiscale::getNominalValues() const
{
  return nominal_values_;
}

Thyra::ModelEvaluatorBase::InArgs<ST>
LCM::SchwarzMultiscale::getLowerBounds() const
{
  return Thyra::ModelEvaluatorBase::InArgs<ST>(); // Default value
}

Thyra::ModelEvaluatorBase::InArgs<ST>
LCM::SchwarzMultiscale::getUpperBounds() const
{
  return Thyra::ModelEvaluatorBase::InArgs<ST>(); // Default value
}

Teuchos::RCP<Thyra::LinearOpBase<ST>>
LCM::SchwarzMultiscale::create_W_op() const
{
  LCM::Schwarz_CoupledJacobian csJac(commT_);
  return csJac.getThyraCoupledJacobian(jacs_, apps_);
}

Teuchos::RCP<Thyra::PreconditionerBase<ST>>
LCM::SchwarzMultiscale::create_W_prec() const
{
  //Teuchos::RCP< Thyra::PreconditionerBase<ST>> W_prec;
  Teuchos::RCP<Thyra::DefaultPreconditioner<ST>> W_prec = Teuchos::rcp(
      new Thyra::DefaultPreconditioner<ST>);
  if (w_prec_supports_) {
    LCM::Schwarz_CoupledJacobian csJac(commT_);
    for (auto m = 0; m < num_models_; m++) {
      if (precs_[m]->isFillActive())
        precs_[m]->fillComplete();
    }
    Teuchos::RCP<Thyra::LinearOpBase<ST>> W_op = csJac.getThyraCoupledJacobian(
        precs_,
        apps_);
    W_prec->initializeRight(W_op);
    // IKT, 11/16/16: the following code is for Teko.
    // We may want to switch to this once I figure out how to hook up Teko
    // with natrix-free.
    /*
     // Get preconditioner factory from solver_factory_.
     // For Teko, this will get the TekoFactory.
     Teuchos::RCP<Thyra::PreconditionerFactoryBase<ST>>
     prec_factory = solver_factory_->getPreconditionerFactory();
     //Get the preconditioner operator from the prec_factory
     W_prec = prec_factory->createPrec();
     */
  }
  else {
    TEUCHOS_TEST_FOR_EXCEPT(w_prec_supports_);
    //W_prec = Teuchos::null; 
  }
  return W_prec;
}

Teuchos::RCP<const Thyra::LinearOpWithSolveFactoryBase<ST>>
LCM::SchwarzMultiscale::get_W_factory() const
{
  return solver_factory_;
}

Thyra::ModelEvaluatorBase::InArgs<ST>
LCM::SchwarzMultiscale::createInArgs() const
{
  return this->createInArgsImpl();
}

void
LCM::SchwarzMultiscale::reportFinalPoint(
    Thyra::ModelEvaluatorBase::InArgs<ST> const & final_point,
    bool const was_solved)
{
  ALBANY_ASSERT(false, "Calling reportFinalPoint");
}

void
LCM::SchwarzMultiscale::
allocateVectors()
{
  //In this function, we create and set x_init and x_dot_init in
  //nominal_values_ for the coupled model.
  Teuchos::Array<Teuchos::RCP<Thyra::VectorSpaceBase<ST> const>>
  spaces(num_models_);

  for (auto m = 0; m < num_models_; ++m) {
    spaces[m] = Thyra::createVectorSpace<ST>(disc_maps_[m]);
  }

  Teuchos::RCP<Thyra::DefaultProductVectorSpace<ST> const>
  space = Thyra::productVectorSpace<ST>(spaces);

  Teuchos::ArrayRCP<Teuchos::RCP<Thyra::VectorBase<ST>>>
  xT_vecs;

  Teuchos::ArrayRCP<Teuchos::RCP<Thyra::VectorBase<ST>>>
  x_dotT_vecs;

  xT_vecs.resize(num_models_);
  x_dotT_vecs.resize(num_models_);

  for (auto m = 0; m < num_models_; ++m) {

    Teuchos::RCP<Tpetra_MultiVector const> const
    xMV = apps_[m]->getAdaptSolMgrT()->getInitialSolution();

    Teuchos::RCP<Tpetra_Vector>
    xT_vec = Teuchos::rcp(new Tpetra_Vector(*xMV->getVector(0)));

    // Error if xdot isn't around
    ALBANY_ASSERT(xMV->getNumVectors() >= 2, "Time derivative is not present.");

    Teuchos::RCP<Tpetra_Vector>
    x_dotT_vec = Teuchos::rcp(new Tpetra_Vector(*xMV->getVector(1)));

    xT_vecs[m] = Thyra::createVector(xT_vec, spaces[m]);
    x_dotT_vecs[m] = Thyra::createVector(x_dotT_vec, spaces[m]);
  }

  Teuchos::RCP<Thyra::DefaultProductVector<ST>>
  xT_prod_vec = Thyra::defaultProductVector<ST>(space, xT_vecs());

  Teuchos::RCP<Thyra::DefaultProductVector<ST>>
  x_dotT_prod_vec = Thyra::defaultProductVector<ST>(space, x_dotT_vecs());

  nominal_values_.set_x(xT_prod_vec);
  nominal_values_.set_x_dot(x_dotT_prod_vec);

}

/// Create operator form of dg/dx for distributed responses
Teuchos::RCP<Thyra::LinearOpBase<ST>>
LCM::SchwarzMultiscale::
create_DgDx_op_impl(int j) const
{
  ALBANY_EXPECT(0 <= j && j < num_responses_total_);

  //FIX ME: re-implement using product vectors! 
  return Teuchos::null;
}

/// Create operator form of dg/dx_dot for distributed responses
Teuchos::RCP<Thyra::LinearOpBase<ST>>
LCM::SchwarzMultiscale::
create_DgDx_dot_op_impl(int j) const
{
  ALBANY_EXPECT(0 <= j && j < num_responses_total_);
  //FIXME: re-implement using product vectors!
  return Teuchos::null;
}

/// Create OutArgs
Thyra::ModelEvaluatorBase::OutArgs<ST>
LCM::SchwarzMultiscale::
createOutArgsImpl() const
{
  Thyra::ModelEvaluatorBase::OutArgsSetup<ST>
  result;

  result.setModelEvalDescription(this->description());

  //Note: it is assumed her there are no distributed parameters.
  result.set_Np_Ng(num_params_total_, num_responses_total_);

  result.setSupports(Thyra::ModelEvaluatorBase::OUT_ARG_f, true);
  result.setSupports(Thyra::ModelEvaluatorBase::OUT_ARG_W_op, true);
  result.setSupports(
      Thyra::ModelEvaluatorBase::OUT_ARG_W_prec,
      w_prec_supports_);

  result.set_W_properties(
      Thyra::ModelEvaluatorBase::DerivativeProperties(
          Thyra::ModelEvaluatorBase::DERIV_LINEARITY_UNKNOWN,
          Thyra::ModelEvaluatorBase::DERIV_RANK_FULL,
          true));

  return result;
}

/// Evaluate model on InArgs
void
LCM::SchwarzMultiscale::
evalModelImpl(
    Thyra::ModelEvaluatorBase::InArgs<ST> const & in_args,
    Thyra::ModelEvaluatorBase::OutArgs<ST> const & out_args) const
{
  //Get xT and x_dotT from in_args
  Teuchos::RCP<const Thyra::ProductVectorBase<ST>>
  xT = Teuchos::rcp_dynamic_cast<const Thyra::ProductVectorBase<ST>>(
      in_args.get_x(), true);

  Teuchos::RCP<const Thyra::ProductVectorBase<ST>>
  x_dotT =
      Teuchos::nonnull(in_args.get_x_dot()) ?
          Teuchos::rcp_dynamic_cast<const Thyra::ProductVectorBase<ST>>(
              in_args.get_x_dot(), true) :
          Teuchos::null;

  // Create a Teuchos array of the xT and x_dotT for each model,
  // casting to Tpetra_Vector
  Teuchos::Array<Teuchos::RCP<Tpetra_Vector const>>
  xTs(num_models_);

  Teuchos::Array<Teuchos::RCP<Tpetra_Vector const>>
  x_dotTs(num_models_);

  for (auto m = 0; m < num_models_; ++m) {
    //Get each Tpetra vector
    xTs[m] = Teuchos::rcp_dynamic_cast<const ThyraVector>(
        xT->getVectorBlock(m),
        true)->getConstTpetraVector();
  }
  if (x_dotT != Teuchos::null) {
    for (auto m = 0; m < num_models_; ++m) {
      //Get each Tpetra vector
      x_dotTs[m] = Teuchos::rcp_dynamic_cast<const ThyraVector>(
          x_dotT->getVectorBlock(m),
          true)->getConstTpetraVector();
    }
  }

  // AGS: x_dotdot time integrators not imlemented in Thyra ME yet
  Teuchos::RCP<Tpetra_Vector const> const
  x_dotdotT = Teuchos::null;

  double const
  alpha =
      (Teuchos::nonnull(x_dotT) || Teuchos::nonnull(x_dotdotT)) ?
          in_args.get_alpha() : 0.0;

  // AGS: x_dotdot time integrators not imlemented in Thyra ME yet
  // const double omega = (Teuchos::nonnull(x_dotT) ||
  // Teuchos::nonnull(x_dotdotT)) ? in_args.get_omega() : 0.0;

  double const
  omega = 0.0;

  double const beta =
      (Teuchos::nonnull(x_dotT) || Teuchos::nonnull(x_dotdotT)) ?
          in_args.get_beta() : 1.0;

  double const curr_time =
      (Teuchos::nonnull(x_dotT) || Teuchos::nonnull(x_dotdotT)) ?
          in_args.get_t() : 0.0;

  //Get parameters
  for (auto l = 0; l < num_params_total_; ++l) {
    //get p from in_args for each parameter
    Teuchos::RCP<Thyra::ProductVectorBase<ST> const> pT =
        Teuchos::rcp_dynamic_cast<const Thyra::ProductVectorBase<ST>>(
            in_args.get_p(l), true);
    // Don't set it if there is nothing. Avoid null pointer.
    if (Teuchos::is_null(pT) == true) continue;

    for (auto m = 0; m < num_models_; ++m) {
      ParamVec &
      sacado_param_vector = sacado_param_vecs_[m][l];

      // IKT FIXME: the following is somewhat of a hack:
      // we get the 0th block of p b/c with all the parameters
      // read from the master file, all the blocks are the same.
      // The code does not work if 0 is replaced by m:
      // only the first vector in the Thyra Product MultiVec is correct.
      // Why...?
      Teuchos::RCP<Tpetra_Vector const>
      pTm = Teuchos::rcp_dynamic_cast<const ThyraVector>(
          pT->getVectorBlock(0), true)->getConstTpetraVector();

      Teuchos::ArrayRCP<ST const> pTm_constView = pTm->get1dView();
      for (auto k = 0; k < sacado_param_vector.size(); ++k) {
        sacado_param_vector[k].baseValue = pTm_constView[k];
      }
    }
  }
  //
  // Get the output arguments
  //
  Teuchos::RCP<Thyra::ProductVectorBase<ST>> fT_out =
      Teuchos::nonnull(out_args.get_f()) ?
          Teuchos::rcp_dynamic_cast<Thyra::ProductVectorBase<ST>>(
              out_args.get_f(), true) :
          Teuchos::null;

  //Create a Teuchos array of the fT_out for each model.
  Teuchos::Array<Teuchos::RCP<Tpetra_Vector>> fTs_out(num_models_);
  if (fT_out != Teuchos::null) {
    for (auto m = 0; m < num_models_; ++m) {
      //Get each Tpetra vector
      fTs_out[m] = Teuchos::rcp_dynamic_cast<ThyraVector>(
          fT_out->getNonconstVectorBlock(m),
          true)->getTpetraVector();
    }
  }

  Teuchos::RCP<Thyra::LinearOpBase<ST>>
  W_op_outT = Teuchos::nonnull(out_args.get_W_op()) ?
                                                      out_args.get_W_op() :
                                                      Teuchos::null;

  // Compute the functions

  Teuchos::Array<bool>
  fs_already_computed(num_models_, false);

  // So that the Schwarz BC has the latest solution, we force here a
  // write of the solution to the mesh database. For STK, which we use,
  // the time parameter is ignored.
  for (auto m = 0; m < num_models_; ++m) {
    double const
    time = 0.0;

    Teuchos::RCP<Albany::AbstractDiscretization> const &
    app_disc = apps_[m]->getDiscretization();

    app_disc->writeSolutionToMeshDatabaseT(*xTs[m], time);
  }

  // W matrix for each individual model
  if (Teuchos::nonnull(W_op_outT) == true) {
    for (auto m = 0; m < num_models_; ++m) {
      //computeGlobalJacobianT sets fTs_out[m] and jacs_[m]
      apps_[m]->computeGlobalJacobianT(
          alpha, beta, omega, curr_time,
          x_dotTs[m].get(), x_dotdotT.get(), *xTs[m],
          sacado_param_vecs_[m], fTs_out[m].get(), *jacs_[m]);
      fs_already_computed[m] = true;
    }
    // FIXME: create coupled W matrix from array of model W matrices
    LCM::Schwarz_CoupledJacobian csJac(commT_);
    W_op_outT = csJac.getThyraCoupledJacobian(jacs_, apps_);
  }

  for (auto m = 0; m < num_models_; ++m) {
    if (apps_[m]->is_adjoint) {
      Thyra::ModelEvaluatorBase::Derivative<ST> const
      f_derivT(solver_outargs_[m].get_f(),
          Thyra::ModelEvaluatorBase::DERIV_TRANS_MV_BY_ROW);

      Thyra::ModelEvaluatorBase::Derivative<ST> const dummy_derivT;

      // need to add capability for sending this in
      auto const
      response_index = 0;

      apps_[m]->evaluateResponseDerivativeT(
          response_index, curr_time, x_dotTs[m].get(), x_dotdotT.get(), *xTs[m],
          sacado_param_vecs_[m], NULL,
          NULL, f_derivT, dummy_derivT, dummy_derivT, dummy_derivT);
    }
    else {
      if (Teuchos::nonnull(fTs_out[m]) && fs_already_computed[m] == false) {

        apps_[m]->computeGlobalResidualT(
            curr_time, x_dotTs[m].get(), x_dotdotT.get(), *xTs[m],
            sacado_param_vecs_[m], *fTs_out[m]);

      }
    }
  }

  //Create preconditioner if w_prec_supports_ are on
  if (w_prec_supports_ == true) {
    Teuchos::RCP<Thyra::PreconditionerBase<ST>>
    W_prec_outT =
        Teuchos::nonnull(out_args.get_W_prec()) ?
                                                  out_args.get_W_prec() :
                                                  Teuchos::null;

    //IKT, 11/16/16: it may be desirable to move the following code into a separate 
    //function, especially as we implement more preconditioners. 
    if (Teuchos::nonnull(W_prec_outT) == true) {
      for (auto m = 0; m < num_models_; ++m) {
        if (!precs_[m]->isFillActive())
          precs_[m]->resumeFill();
        if (mf_prec_type_ == JACOBI) {
          //With matrix-free, W_op_outT is null, so computeJacobianT does not
          //get called earlier.  We need to call it here to get the Jacobians.
          //Create fTtemp vector, so that this call to computeGlobalJacobianT 
          //doesn't overwrite the real residual.
          Teuchos::RCP<Tpetra_Vector> fTtemp;
          if (fT_out != Teuchos::null) {
            fTtemp = Teuchos::rcp_dynamic_cast<ThyraVector>(
                fT_out->getNonconstVectorBlock(m),
                true)->getTpetraVector();
          }
          apps_[m]->computeGlobalJacobianT(alpha, beta, omega, curr_time,
              x_dotTs[m].get(), x_dotdotT.get(), *xTs[m],
              sacado_param_vecs_[m], fTtemp.get(), *jacs_[m]);
          //Get diagonal of jacs_[m]  
          Teuchos::RCP<Tpetra_Vector> diag = Teuchos::rcp(
              new Tpetra_Vector(jacs_[m]->getRowMap()));
          jacs_[m]->getLocalDiagCopy(*diag);
          //Take reciprocal of diagonal 
          Teuchos::RCP<Tpetra_Vector> invdiag = Teuchos::rcp(
              new Tpetra_Vector(jacs_[m]->getRowMap()));
          invdiag->reciprocal(*diag);
          Teuchos::ArrayRCP<const ST> invdiag_constView = invdiag->get1dView();
          //Zero out precs_[m] 
          precs_[m]->resumeFill();
          precs_[m]->scale(0.0);
          //Create Jacobi preconditioner 
          for (auto i = 0; i < jacs_[m]->getNodeNumRows(); ++i) {
            GO global_row = jacs_[m]->getRowMap()->getGlobalElement(i);
            Teuchos::Array<ST> matrixEntriesT(1);
            Teuchos::Array<GO> matrixIndicesT(1);
            matrixEntriesT[0] = invdiag_constView[i];
            matrixIndicesT[0] = global_row;
            precs_[m]->replaceGlobalValues(
                global_row,
                matrixIndicesT(),
                matrixEntriesT());
          }
        }
        else if (mf_prec_type_ == ABS_ROW_SUM) {
          //With matrix-free, W_op_outT is null, so computeJacobianT does not
          //get called earlier.  We need to call it here to get the Jacobians.
          //Create fTtemp vector, so that this call to computeGlobalJacobianT 
          //doesn't overwrite the real residual.
          Teuchos::RCP<Tpetra_Vector> fTtemp;
          if (fT_out != Teuchos::null) {
            fTtemp = Teuchos::rcp_dynamic_cast<ThyraVector>(
                fT_out->getNonconstVectorBlock(m),
                true)->getTpetraVector();
          }
          apps_[m]->computeGlobalJacobianT(alpha, beta, omega, curr_time,
              x_dotTs[m].get(), x_dotdotT.get(), *xTs[m],
              sacado_param_vecs_[m], fTtemp.get(), *jacs_[m]);
          //Create vector to store absrowsum 
          Teuchos::RCP<Tpetra_Vector> absrowsum = Teuchos::rcp(
              new Tpetra_Vector(jacs_[m]->getRowMap()));
          absrowsum->putScalar(0.0);
          Teuchos::ArrayRCP<ST> absrowsum_nonconstView = absrowsum
              ->get1dViewNonConst();
          //Compute abs sum of each row and store in absrowsum vector 
          for (auto i = 0; i < jacs_[m]->getNodeNumRows(); ++i) {
            std::size_t NumEntries = jacs_[m]->getNumEntriesInLocalRow(i);
            Teuchos::Array<LO> Indices(NumEntries);
            Teuchos::Array<ST> Values(NumEntries);
            //Get local row
            jacs_[m]->getLocalRowCopy(i, Indices(), Values(), NumEntries);
            //Compute abs row rum 
            for (auto j = 0; j < NumEntries; j++)
              absrowsum_nonconstView[i] += abs(Values[j]);
          }
          //Invert absrowsum 
          Teuchos::RCP<Tpetra_Vector> invabsrowsum = Teuchos::rcp(
              new Tpetra_Vector(jacs_[m]->getRowMap()));
          invabsrowsum->reciprocal(*absrowsum);
          Teuchos::ArrayRCP<const ST> invabsrowsum_constView = invabsrowsum
              ->get1dView();
          //Zero out precs_[m] 
          precs_[m]->resumeFill();
          precs_[m]->scale(0.0);
          //Create diagonal abs row sum preconditioner 
          for (auto i = 0; i < jacs_[m]->getNodeNumRows(); ++i) {
            GO global_row = jacs_[m]->getRowMap()->getGlobalElement(i);
            Teuchos::Array<ST> matrixEntriesT(1);
            Teuchos::Array<GO> matrixIndicesT(1);
            matrixEntriesT[0] = invabsrowsum_constView[i];
            matrixIndicesT[0] = global_row;
            precs_[m]->replaceGlobalValues(
                global_row,
                matrixIndicesT(),
                matrixEntriesT());
          }
        }
        else if (mf_prec_type_ == ID) {
          //Create Identity
          for (auto i = 0; i < jacs_[m]->getNodeNumRows(); ++i) {
            GO global_row = jacs_[m]->getRowMap()->getGlobalElement(i);
            Teuchos::Array<ST> matrixEntriesT(1);
            Teuchos::Array<GO> matrixIndicesT(1);
            matrixEntriesT[0] = 1.0;
            matrixIndicesT[0] = global_row;
            precs_[m]->replaceGlobalValues(
                global_row,
                matrixIndicesT(),
                matrixEntriesT());
          }
        }
        if (precs_[m]->isFillActive())
          precs_[m]->fillComplete();
      }
      LCM::Schwarz_CoupledJacobian csJac(commT_);
      Teuchos::RCP<Thyra::LinearOpBase<ST>> W_op =
          csJac.getThyraCoupledJacobian(precs_, apps_);
      Teuchos::RCP<Thyra::DefaultPreconditioner<ST>> W_prec = Teuchos::rcp(
          new Thyra::DefaultPreconditioner<ST>);
      W_prec->initializeRight(W_op);
      W_prec_outT = Teuchos::rcp_dynamic_cast<Thyra::PreconditionerBase<ST>>(
          W_prec,
          true);
    }
  }

#ifdef WRITE_TO_MATRIX_MARKET
  Albany::writeMatrixMarket(xTs, "sol", mm_counter_sol);
  ++mm_counter_sol;
#endif

#ifdef WRITE_TO_MATRIX_MARKET
  Albany::writeMatrixMarket(fTs_out, "res", mm_counter_res);
  ++mm_counter_res;
#endif

#ifdef WRITE_TO_MATRIX_MARKET
  Albany::writeMatrixMarket(jacs_, "jac", mm_counter_jac);
  ++mm_counter_jac;
#endif

#ifdef WRITE_TO_MATRIX_MARKET
  Albany::writeMatrixMarket(precs_, "pre", mm_counter_pre);
  ++mm_counter_pre;
#endif
//Responses / sensitivities
//FIXME: need to implement DgDx, DgDp, etc for sensitivity analysis! 
// Response functions
  for (auto j = 0; j < out_args.Ng(); ++j) {

    Teuchos::RCP<Thyra::ProductVectorBase<ST>>
    gT_out =
        Teuchos::nonnull(out_args.get_g(j)) ?
            Teuchos::rcp_dynamic_cast<Thyra::ProductVectorBase<ST>>(
                out_args.get_g(j), true) :
            Teuchos::null;

    if (Teuchos::is_null(gT_out) == false) {
      for (auto m = 0; m < num_models_; ++m) {
        //Get each Tpetra vector
        Teuchos::RCP<Tpetra_Vector>
        gT_out_m = Teuchos::rcp_dynamic_cast<ThyraVector>(
            gT_out->getNonconstVectorBlock(m),
            true)->getTpetraVector();

        for (auto l = 0; l < out_args.Np(); ++l) {
          //sets gT_out
          apps_[m]->evaluateResponseT(
              l, curr_time, x_dotTs[m].get(), x_dotdotT.get(),
              *xTs[m], sacado_param_vecs_[m], *gT_out_m);
        }
      }
    }
  }
}

Thyra::ModelEvaluatorBase::InArgs<ST>
LCM::SchwarzMultiscale::
createInArgsImpl() const
{
  Thyra::ModelEvaluatorBase::InArgsSetup<ST>
  result;

  result.setModelEvalDescription(this->description());

  result.setSupports(Thyra::ModelEvaluatorBase::IN_ARG_x, true);

  result.setSupports(Thyra::ModelEvaluatorBase::IN_ARG_x_dot, true);
  // AGS: x_dotdot time integrators not imlemented in Thyra ME yet
  //result.setSupports(Thyra::ModelEvaluatorBase::IN_ARG_x_dotdot, true);
  result.setSupports(Thyra::ModelEvaluatorBase::IN_ARG_t, true);
  result.setSupports(Thyra::ModelEvaluatorBase::IN_ARG_alpha, true);
  result.setSupports(Thyra::ModelEvaluatorBase::IN_ARG_beta, true);
  // AGS: x_dotdot time integrators not imlemented in Thyra ME yet
  //result.setSupports(Thyra::ModelEvaluatorBase::IN_ARG_omega, true);

  result.set_Np(num_params_total_);

  return result;
}

//Copied from QCAD::CoupledPoissonSchrodinger -- used to validate
//applicaton parameters of applications not created via a
//SolverFactory Check usage and whether necessary...
Teuchos::RCP<Teuchos::ParameterList const>
LCM::SchwarzMultiscale::
getValidAppParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList>
  list = Teuchos::rcp(new Teuchos::ParameterList("ValidAppParams"));

  list->sublist("Problem", false, "Problem sublist");
  list->sublist("Debug Output", false, "Debug Output sublist");
  list->sublist("Discretization", false, "Discretization sublist");
  list->sublist("Quadrature", false, "Quadrature sublist");
  list->sublist("Regression Results", false, "Regression Results sublist");
  list->sublist("VTK", false, "DEPRECATED  VTK sublist");
  list->sublist("Piro", false, "Piro sublist");
  list->sublist("Coupled System", false, "Coupled system sublist");
  list->set<std::string>(
      "Matrix-Free Preconditioner",
      "",
      "Matrix-Free Preconditioner Type");

  return list;
}

//Copied from QCAD::CoupledPoissonSchrodinger
//Check usage and whether neessary...
Teuchos::RCP<Teuchos::ParameterList const>
LCM::SchwarzMultiscale::
getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList>
  list = Teuchos::createParameterList("ValidCoupledSchwarzProblemParams");

  list->set<std::string>("Name", "", "String to designate Problem Class");

  list->set<int>(
      "Phalanx Graph Visualization Detail",
      0,
      "Flag to select output of Phalanx Graph and level of detail");

  //FIXME: anything else to validate?
  list->set<std::string>(
      "Solution Method",
      "Steady",
      "Flag for Steady, Transient, or Continuation");

  return list;
}

