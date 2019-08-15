//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_SolverFactory.hpp"
#include "Albany_PiroObserver.hpp"
#include "Albany_ModelEvaluator.hpp"
#include "Albany_Application.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_Macros.hpp"

#ifdef ALBANY_ATO
#include "ATO_Solver.hpp"
#endif

#if defined(ALBANY_LCM) && defined(ALBANY_STK)
#include "Schwarz_Alternating.hpp"
#include "Schwarz_Coupled.hpp"
#include "Schwarz_PiroObserver.hpp"
#endif

#ifdef ALBANY_AERAS
#include "Aeras/Aeras_HVDecorator.hpp"
#endif

#include "Piro_ProviderBase.hpp"
#include "Piro_NOXSolver.hpp"
#include "Piro_SolverFactory.hpp"
#include "Piro_StratimikosUtils.hpp"

#include "Stratimikos_DefaultLinearSolverBuilder.hpp"

#ifdef ALBANY_IFPACK2
#include "Teuchos_AbstractFactoryStd.hpp"
#include "Thyra_Ifpack2PreconditionerFactory.hpp"
#endif /* ALBANY_IFPACK2 */

#ifdef ALBANY_MUELU
#include "Stratimikos_MueLuHelpers.hpp"
#endif /* ALBANY_MUELU */

#ifdef ALBANY_FROSCH
#include <Stratimikos_FROSchXpetra.hpp>
#endif /* ALBANY_FROSCH */

#ifdef ALBANY_TEKO
#include "Teko_StratimikosFactory.hpp"
#endif

#include "Thyra_DefaultModelEvaluatorWithSolveFactory.hpp"
#include "Thyra_DetachedVectorView.hpp"

#include "Teuchos_TestForException.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#include "Teuchos_YamlParameterListHelpers.hpp"

#if defined(ALBANY_RYTHMOS)
#include "Rythmos_IntegrationObserverBase.hpp"
#endif

namespace {

void
enableIfpack2(Stratimikos::DefaultLinearSolverBuilder& linearSolverBuilder)
{
#ifdef ALBANY_IFPACK2
  typedef Thyra::PreconditionerFactoryBase<ST>                  Base;
  typedef Thyra::Ifpack2PreconditionerFactory<Tpetra_CrsMatrix> Impl;
  linearSolverBuilder.setPreconditioningStrategyFactory(
      Teuchos::abstractFactoryStd<Base, Impl>(), "Ifpack2");
#endif
}

void
enableMueLu(
    Stratimikos::DefaultLinearSolverBuilder&    linearSolverBuilder)
{
#ifdef ALBANY_MUELU
  Stratimikos::enableMueLu<LO, Tpetra_GO, KokkosNode>(linearSolverBuilder);
#endif
}

void
enableFROSch(Stratimikos::DefaultLinearSolverBuilder&    linearSolverBuilder)
{
#ifdef ALBANY_FROSCH
    Stratimikos::enableFROSch<LO,Tpetra_GO, KokkosNode>(linearSolverBuilder);
#endif
}
}  // namespace

namespace Albany
{

SolverFactory::
SolverFactory(const std::string&                      inputFile,
              const Teuchos::RCP<const Teuchos_Comm>& comm)
 : out(Teuchos::VerboseObjectBase::getDefaultOStream())
{
  // Set up application parameters: read and broadcast XML file, and set
  // defaults
  // Teuchos::RCP<Teuchos::ParameterList> input_
  appParams = Teuchos::createParameterList("Albany Parameters");

  std::string const input_extension = getFileExtension(inputFile);

  if (input_extension == "yaml" || input_extension == "yml") {
    Teuchos::updateParametersFromYamlFileAndBroadcast(
        inputFile, appParams.ptr(), *comm);
  } else {
    Teuchos::updateParametersFromXmlFileAndBroadcast(
        inputFile, appParams.ptr(), *comm);
  }

  // do not set default solver parameters for ATO::Solver problems,
  // ... as they handle this themselves
  std::string solution_method =
      appParams->sublist("Problem").get("Solution Method", "Steady");
  if (solution_method != "ATO Problem") {
    Teuchos::RCP<Teuchos::ParameterList> defaultSolverParams = rcp(new Teuchos::ParameterList());
    setSolverParamDefaults(defaultSolverParams.get(), comm->getRank());
    appParams->setParametersNotAlreadySet(*defaultSolverParams);
  }

  if (!appParams->isParameter("Build Type")) {
    if (comm->getRank()==0) {
      *out << "\nWARNING! You have not set the entry 'Build Type' in the input file. This will cause Albany to *assume* a Tpetra build.\n"
           << "         If that's not ok, and you specified Epetra-based solvers/preconditioners, you will get an dynamic cast error like this:\n"
           << "\n"
           << "           dyn_cast<Thyra::EpetraLinearOpBase>(Thyra::LinearOpBase<double>) : Error, the object with the concrete type 'Thyra::TpetraLinearOp<[Some Template Args]>' (passed in through the interface type 'Thyra::LinearOpBase<double>')  does not support the interface 'Thyra::EpetraLinearOpBase' and the dynamic cast failed!\n"
           << "\n"
           << "         If that happens, all you have to do is to set 'Build Type: Epetra' in the main level of your input yaml file.\n\n";
    }
  }
  appParams->validateParametersAndSetDefaults(*getValidAppParameters(), 0);
  if (appParams->isSublist("Debug Output")) {
    Teuchos::RCP<Teuchos::ParameterList> debugPL = Teuchos::rcpFromRef(appParams->sublist("Debug Output", false)); 
    debugPL->validateParametersAndSetDefaults(*getValidDebugParameters(), 0);
  }
  if (appParams->isSublist("Scaling")) {
    Teuchos::RCP<Teuchos::ParameterList> scalingPL = Teuchos::rcpFromRef(appParams->sublist("Scaling", false)); 
    scalingPL->validateParametersAndSetDefaults(*getValidScalingParameters(), 0);
  }
}

SolverFactory::
SolverFactory(const Teuchos::RCP<Teuchos::ParameterList>& input_appParams,
              const Teuchos::RCP<const Teuchos_Comm>&     comm)
 : appParams(input_appParams)
 , out(Teuchos::VerboseObjectBase::getDefaultOStream())
{
  // do not set default solver parameters for ATO::Solver
  // problems,
  // ... as they handle this themselves
  std::string solution_method =
      appParams->sublist("Problem").get("Solution Method", "Steady");
  if (solution_method != "ATO Problem") {
    Teuchos::RCP<Teuchos::ParameterList> defaultSolverParams = rcp(new Teuchos::ParameterList());
    setSolverParamDefaults(defaultSolverParams.get(), comm->getRank());
    appParams->setParametersNotAlreadySet(*defaultSolverParams);
  }

  if (!appParams->isParameter("Build Type")) {
    if (comm->getRank()==0) {
      *out << "\nWARNING! You have not set the entry 'Build Type' in the input parameter list. This will cause Albany to *assume* a Tpetra build.\n"
           << "         If that's not ok, and you specified Epetra-based solvers/preconditioners, you will get an dynamic cast error like this:\n"
           << "\n"
           << "           dyn_cast<Thyra::EpetraLinearOpBase>(Thyra::LinearOpBase<double>) : Error, the object with the concrete type 'Thyra::TpetraLinearOp<[Some Template Args]>' (passed in through the interface type 'Thyra::LinearOpBase<double>')  does not support the interface 'Thyra::EpetraLinearOpBase' and the dynamic cast failed!\n"
           << "\n"
           << "         If that happens, all you have to do is to set 'Build Type: Epetra' in the main level of your parameter list.\n\n";
    }
  }
  appParams->validateParametersAndSetDefaults(*getValidAppParameters(), 0);
  if (appParams->isSublist("Debug Output")) {
    Teuchos::RCP<Teuchos::ParameterList> debugPL = Teuchos::rcpFromRef(appParams->sublist("Debug Output", false)); 
    debugPL->validateParametersAndSetDefaults(*getValidDebugParameters(), 0);
  }
}

Teuchos::RCP<Thyra::ResponseOnlyModelEvaluatorBase<ST>>
SolverFactory::create(
    const Teuchos::RCP<const Teuchos_Comm>& appComm,
    const Teuchos::RCP<const Teuchos_Comm>& solverComm,
    const Teuchos::RCP<const Thyra_Vector>& initial_guess)
{
  Teuchos::RCP<Application> dummyAlbanyApp;
  return createAndGetAlbanyApp(dummyAlbanyApp, appComm, solverComm, initial_guess);
}

Teuchos::RCP<Thyra::ResponseOnlyModelEvaluatorBase<ST>>
SolverFactory::createAndGetAlbanyApp(
    Teuchos::RCP<Application>&              albanyApp,
    const Teuchos::RCP<const Teuchos_Comm>& appComm,
    const Teuchos::RCP<const Teuchos_Comm>& solverComm,
    const Teuchos::RCP<const Thyra_Vector>& initial_guess,
    bool                                    createAlbanyApp)
{
  const Teuchos::RCP<Teuchos::ParameterList> problemParams = Teuchos::sublist(appParams, "Problem");
  const std::string solutionMethod = problemParams->get("Solution Method", "Steady");

  if (solutionMethod == "ATO Problem") {
#ifdef ALBANY_ATO
    return rcp(new ATO::Solver(appParams, solverComm, initial_guess));
#else  /* ALBANY_ATO */
    TEUCHOS_TEST_FOR_EXCEPTION(
        true,
        std::logic_error,
        "Must activate ATO (topological optimization)\n");
#endif /* ALBANY_ATO */
  }

#ifdef ALBANY_AERAS
  if (solutionMethod == "Aeras Hyperviscosity") {
    // std::cout <<"In Albany_SolverFactory: solutionMethod = Aeras
    // Hyperviscosity" << std::endl;
    // Check if HV coefficient tau is zero of "Explicit HV" is false. Then there
    // is no need for Aeras HVDecorator.

    double tau;
    bool   useExplHyperviscosity;

    std::string swProblem_name    = "Shallow Water Problem",
                hydroProblem_name = "Hydrostatic Problem";

    bool swProblem    = problemParams->isSublist(swProblem_name);
    bool hydroProblem = problemParams->isSublist(hydroProblem_name);

    if ((!swProblem) && (!hydroProblem)) {
      *out << "Error: Hyperviscosity can only be used with Aeras:Shallow Water "
              "or Aeras:Hydrostatic."
           << std::endl;
      TEUCHOS_TEST_FOR_EXCEPTION(
          true,
          std::logic_error,
          "Error: cannot locate " << swProblem_name << " or "
                                  << hydroProblem_name
                                  << " sublist in the input file."
                                  << "\n");
    }

    if (swProblem) {
      tau = problemParams->sublist(swProblem_name)
                .get<double>("Hyperviscosity Tau", 0.0);
      useExplHyperviscosity =
          problemParams->sublist(swProblem_name)
              .get<bool>("Use Explicit Hyperviscosity", false);
      *out << "Reading Shallow Water Problem List: Using explicit "
              "hyperviscosity? "
           << useExplHyperviscosity << "\n";
    }
    if (hydroProblem) {
      tau = problemParams->sublist(hydroProblem_name)
                .get<double>("Hyperviscosity Tau", 0.0);
      useExplHyperviscosity =
          problemParams->sublist(hydroProblem_name)
              .get<bool>("Use Explicit Hyperviscosity", false);
      *out
          << "Reading Hydrostatic Problem List: Using explicit hyperviscosity? "
          << useExplHyperviscosity << "\n";
    }

    if ((useExplHyperviscosity) && (tau != 0.0)) {
      ///// make a solver, repeated code
      const Teuchos::RCP<Teuchos::ParameterList> piroParams = Teuchos::sublist(appParams, "Piro");
      const Teuchos::RCP<Teuchos::ParameterList> stratList =
          Piro::extractStratimikosParams(piroParams);
      // Create and setup the Piro solver factory
      Piro::SolverFactory piroFactory;
      // Setup linear solver
      Stratimikos::DefaultLinearSolverBuilder linearSolverBuilder;
      enableIfpack2(linearSolverBuilder);
      enableMueLu(linearSolverBuilder);
      linearSolverBuilder.setParameterList(stratList);
      const Teuchos::RCP<Thyra_LOWS_Factory> lowsFactory = createLinearSolveStrategy(linearSolverBuilder);

      ///// create an app and a model evaluator

      albanyApp = Teuchos::rcp (new Application(appComm, appParams, initial_guess, is_schwarz_));
      Teuchos::RCP<Thyra_ModelEvaluator> modelHV(new Aeras::HVDecorator(albanyApp, appParams));

      Teuchos::RCP<Thyra_ModelEvaluator> modelWithSolve;

      modelWithSolve = Teuchos::rcp(new Thyra::DefaultModelEvaluatorWithSolveFactory<ST>(modelHV, lowsFactory));

      observer_ = Teuchos::rcp(new PiroObserver(albanyApp, modelWithSolve));

      // Piro::SolverFactory
      return piroFactory.createSolver<ST, LO, Tpetra_GO, KokkosNode>(
          piroParams, modelWithSolve, Teuchos::null, observer_);

    }  // if useExplHV=true and tau <>0.

  }  // if Aeras HyperViscosity
#endif

#if defined(ALBANY_LCM) && defined(ALBANY_STK) 
  bool const is_schwarz = solutionMethod == "Coupled Schwarz" ||
                          solutionMethod == "Schwarz Alternating";

  if (is_schwarz == true) {
#if !defined(ALBANY_DTK)
    ALBANY_ASSERT(appComm->getSize() == 1, "Parallel Schwarz requires DTK");
#endif  // ALBANY_DTK
  }
  if (solutionMethod == "Coupled Schwarz") {
    // IKT: We are assuming the "Piro" list will come from the main coupled
    // Schwarz input file (not the sub-input
    // files for each model).
    const Teuchos::RCP<Teuchos::ParameterList> piroParams = Teuchos::sublist(appParams, "Piro");

    const Teuchos::RCP<Teuchos::ParameterList> stratList =
        Piro::extractStratimikosParams(piroParams);
    // Create and setup the Piro solver factory
    Piro::SolverFactory piroFactory;
    // Setup linear solver
    Stratimikos::DefaultLinearSolverBuilder linearSolverBuilder;
    enableIfpack2(linearSolverBuilder);
    enableMueLu(linearSolverBuilder);

#if defined(ALBANY_TEKO)
    Teko::addTekoToStratimikosBuilder(linearSolverBuilder, "Teko");
#endif
    linearSolverBuilder.setParameterList(stratList);

    const Teuchos::RCP<Thyra_LOWS_Factory> lowsFactory =
        createLinearSolveStrategy(linearSolverBuilder);

    const Teuchos::RCP<LCM::SchwarzCoupled> coupled_model_with_solve =
        Teuchos::rcp(new LCM::SchwarzCoupled(
            appParams, solverComm, initial_guess, lowsFactory));

    observer_ = Teuchos::rcp(new LCM::Schwarz_PiroObserver(coupled_model_with_solve));

    // WARNING: Coupled Schwarz does not contain a primary Application
    // instance and so albanyApp is null.
    return piroFactory.createSolver<ST, LO, Tpetra_GO, KokkosNode>(
        piroParams, coupled_model_with_solve, Teuchos::null, observer_);
  }

  if (solutionMethod == "Schwarz Alternating") {
    return Teuchos::rcp(new LCM::SchwarzAlternating(appParams, solverComm));
  }
#endif /* LCM */

  model_ = createAlbanyAppAndModel(albanyApp, appComm, initial_guess, createAlbanyApp);

  const Teuchos::RCP<Teuchos::ParameterList> piroParams = Teuchos::sublist(appParams, "Piro");
  const Teuchos::RCP<Teuchos::ParameterList> stratList =
      Piro::extractStratimikosParams(piroParams);

  if (Teuchos::is_null(stratList)) {
    *out << "Error: cannot locate Stratimikos solver parameters in the input "
            "file."
         << std::endl;
    *out << "Printing the Piro parameter list:" << std::endl;
    piroParams->print(*out);
    // GAH: this is an error - should be fatal
    TEUCHOS_TEST_FOR_EXCEPTION(
        true,
        std::logic_error,
        "Error: cannot locate Stratimikos solver parameters in the input file."
            << "\n");
  }

  Teuchos::RCP<Thyra_ModelEvaluator> modelWithSolve;
  if (Teuchos::nonnull(model_->get_W_factory())) {
    modelWithSolve = model_;
  } else {
    // Setup linear solver
    Stratimikos::DefaultLinearSolverBuilder linearSolverBuilder;
    enableIfpack2(linearSolverBuilder);
    enableMueLu(linearSolverBuilder);
    enableFROSch(linearSolverBuilder);
#ifdef ALBANY_TEKO
    Teko::addTekoToStratimikosBuilder(linearSolverBuilder, "Teko");
#endif
    linearSolverBuilder.setParameterList(stratList);

    const Teuchos::RCP<Thyra_LOWS_Factory> lowsFactory =
        createLinearSolveStrategy(linearSolverBuilder);

    modelWithSolve = rcp(new Thyra::DefaultModelEvaluatorWithSolveFactory<ST>(model_, lowsFactory));
  }

  const auto solMgr = albanyApp->getAdaptSolMgr();

  Piro::SolverFactory piroFactory;
  observer_ = Teuchos::rcp(new PiroObserver(albanyApp, modelWithSolve));
  if (solMgr->isAdaptive()) {
    return piroFactory.createSolver<ST, LO, Tpetra_GO, KokkosNode>(
        piroParams, modelWithSolve, solMgr, observer_);
  } else {
    return piroFactory.createSolver<ST, LO, Tpetra_GO, KokkosNode>(
        piroParams, modelWithSolve, Teuchos::null, observer_);
  }
  TEUCHOS_TEST_FOR_EXCEPTION(
      true,
      std::logic_error,
      "Reached end of createAndGetAlbanyAppT()"
          << "\n");

  // Silence compiler warning in case it wasn't used (due to ifdef logic)
  (void) solverComm;

  return Teuchos::null;
}

Teuchos::RCP<Thyra_ModelEvaluator>
SolverFactory::createAlbanyAppAndModel(
    Teuchos::RCP<Application>&      albanyApp,
    const Teuchos::RCP<const Teuchos_Comm>& appComm,
    const Teuchos::RCP<const Thyra_Vector>& initial_guess,
    const bool                              createAlbanyApp)
{
  if (createAlbanyApp) {
    // Create application
    albanyApp = Teuchos::rcp(new Application(
        appComm, appParams, initial_guess, is_schwarz_));
    //  albanyApp = rcp(new ApplicationT(appComm, appParams,
    //  initial_guess));
  }

  // Validate Response list: may move inside individual Problem class
  Teuchos::RCP<Teuchos::ParameterList> problemParams = Teuchos::sublist(appParams, "Problem");
  problemParams->sublist("Response Functions")
      .validateParameters(*getValidResponseParameters(), 0);

  // If not explicitly specified, determine which Piro solver to use from the
  // problem parameters
  const Teuchos::RCP<Teuchos::ParameterList> piroParams = Teuchos::sublist(appParams, "Piro");
  if (!piroParams->getPtr<std::string>("Solver Type")) {
    const std::string solutionMethod =
        problemParams->get("Solution Method", "Steady");

    /* TODO: this should be a boolean, not a string ! */
    const std::string secondOrder = problemParams->get("Second Order", "No");
    TEUCHOS_TEST_FOR_EXCEPTION(
        secondOrder != "No",
        std::logic_error,
        "Second Order is not supported"
            << "\n");

    // Populate the Piro parameter list accordingly to inform the Piro solver
    // factory
    std::string piroSolverToken;
    if (solutionMethod == "Steady") {
      piroSolverToken = "NOX";
#ifdef ALBANY_RYTHMOS
    } else if (solutionMethod == "Transient") {
      piroSolverToken = "Rythmos";
#endif
#ifdef ALBANY_TEMPUS
    } else if (solutionMethod == "Transient Tempus") {
      piroSolverToken = "Tempus";
#endif
    } else {
      // Piro cannot handle the corresponding problem
      piroSolverToken = "Unsupported";
    }

    ALBANY_ASSERT(
        piroSolverToken != "Unsupported",
        "Unsupported Solution Method: " << solutionMethod);

    piroParams->set("Solver Type", piroSolverToken);
  }

  // Create model evaluator
  return Teuchos::rcp(new ModelEvaluator(albanyApp,appParams));
}

int
SolverFactory::
checkSolveTestResults(
    int                                           response_index,
    int                                           parameter_index,
    const Teuchos::RCP<const Thyra_Vector>&       g,
    const Teuchos::RCP<const Thyra_MultiVector>&  dgdp) const
{
  Teuchos::ParameterList* testParams = getTestParameters(response_index);

  int          failures    = 0;
  int          comparisons = 0;
  const double relTol      = testParams->get<double>("Relative Tolerance");
  const double absTol      = testParams->get<double>("Absolute Tolerance");

  // Get number of responses (g) to test
  const int numResponseTests = testParams->get<int>("Number of Comparisons");
  if (numResponseTests > 0) {
    ALBANY_ASSERT(
        g != Teuchos::null,
        "There are Response Tests but the response vector is null!");
    ALBANY_ASSERT(
        numResponseTests <= g->space()->dim(),
        "Number of Response Tests (" << numResponseTests
                                     << ") greater than number of responses ("
                                     << g->space()->dim() << ") !");
    Teuchos::Array<double> testValues =
        testParams->get<Teuchos::Array<double>>("Test Values");

    ALBANY_ASSERT(
        numResponseTests == testValues.size(),
        "Number of Response Tests (" << numResponseTests
                                     << ") != number of Test Values ("
                                     << testValues.size() << ") !");

    Teuchos::ArrayRCP<const ST> g_view = getLocalData(g);
    for (int i = 0; i < testValues.size(); i++) {
      auto s = std::string("Response Test ") + std::to_string(i);
      failures += scaledCompare(g_view[i], testValues[i], relTol, absTol, s);
      comparisons++;
    }
  }

  // Repeat comparisons for sensitivities
  Teuchos::ParameterList* sensitivityParams = 0;
  std::string             sensitivity_sublist_name =
      strint("Sensitivity Comparisons", parameter_index);
  if (parameter_index == 0 && !testParams->isSublist(sensitivity_sublist_name))
    sensitivityParams = testParams;
  else if (testParams->isSublist(sensitivity_sublist_name))
    sensitivityParams = &(testParams->sublist(sensitivity_sublist_name));
  int numSensTests = 0;
  if (sensitivityParams != 0) {
    numSensTests =
        sensitivityParams->get<int>("Number of Sensitivity Comparisons", 0);
  }
  if (numSensTests > 0) {
    ALBANY_ASSERT(
        dgdp != Teuchos::null,
        "There are Sensitivity Tests but the sensitivity vector ("
            << response_index << ", " << parameter_index << ") is null!");
    ALBANY_ASSERT(
        numSensTests <= dgdp->range()->dim(),
        "Number of sensitivity tests ("
            << numSensTests << ") != number of sensitivities ["
            << response_index << "][" << parameter_index << "] ("
            << dgdp->range()->dim() << ") !");
  }
  for (int i = 0; i < numSensTests; i++) {
    const int numVecs = dgdp->domain()->dim();
    Teuchos::Array<double> testSensValues =
        sensitivityParams->get<Teuchos::Array<double>>(
            strint("Sensitivity Test Values", i));
    ALBANY_ASSERT(
        numVecs== testSensValues.size(),
        "Number of Sensitivity Test Values ("
            << testSensValues.size() << " != number of sensitivity vectors ("
            << numVecs << ") !");
    auto dgdp_view = getLocalData(dgdp);
    for (int jvec = 0; jvec < numVecs; jvec++) {
      auto s = std::string("Sensitivity Test ") + std::to_string(i) + "," +
               std::to_string(jvec);
      failures +=
          scaledCompare(dgdp_view[jvec][i], testSensValues[jvec], relTol, absTol, s);
      comparisons++;
    }
  }

  storeTestResults(testParams, failures, comparisons);

  return failures;
}

int
SolverFactory::checkDakotaTestResults(
    int                                            response_index,
    const Teuchos::SerialDenseVector<int, double>* drdv) const
{
  Teuchos::ParameterList* testParams = getTestParameters(response_index);

  int          failures    = 0;
  int          comparisons = 0;
  const double relTol      = testParams->get<double>("Relative Tolerance");
  const double absTol      = testParams->get<double>("Absolute Tolerance");

  const int numDakotaTests =
      testParams->get<int>("Number of Dakota Comparisons");
  if (numDakotaTests > 0 && drdv != NULL) {
    ALBANY_ASSERT(
        numDakotaTests <= drdv->length(),
        "more Dakota Tests (" << numDakotaTests << ") than derivatives ("
                              << drdv->length() << ") !\n");
    // Read accepted test results
    Teuchos::Array<double> testValues =
        testParams->get<Teuchos::Array<double>>("Dakota Test Values");

    TEUCHOS_TEST_FOR_EXCEPT(numDakotaTests != testValues.size());
    for (int i = 0; i < numDakotaTests; i++) {
      auto s = std::string("Dakota Test ") + std::to_string(i);
      failures += scaledCompare((*drdv)[i], testValues[i], relTol, absTol, s);
      comparisons++;
    }
  }

  storeTestResults(testParams, failures, comparisons);

  return failures;
}

int
SolverFactory::checkAnalysisTestResults(
    int                                            response_index,
    const Teuchos::RCP<Thyra::VectorBase<double>>& tvec) const
{
  Teuchos::ParameterList* testParams = getTestParameters(response_index);

  int          failures    = 0;
  int          comparisons = 0;
  const double relTol      = testParams->get<double>("Relative Tolerance");
  const double absTol      = testParams->get<double>("Absolute Tolerance");

  int numPiroTests =
      testParams->get<int>("Number of Piro Analysis Comparisons");
  if (numPiroTests > 0 && tvec != Teuchos::null) {
    // Create indexable thyra vector
    ::Thyra::DetachedVectorView<double> p(tvec);

    ALBANY_ASSERT(
        numPiroTests <= p.subDim(),
        "more Piro Analysis Comparisons (" << numPiroTests << ") than values ("
                                           << p.subDim() << ") !\n");
    // Read accepted test results
    Teuchos::Array<double> testValues =
        testParams->get<Teuchos::Array<double>>("Piro Analysis Test Values");

    TEUCHOS_TEST_FOR_EXCEPT(numPiroTests != testValues.size());
    for (int i = 0; i < numPiroTests; i++) {
      auto s = std::string("Piro Analysis Test ") + std::to_string(i);
      failures += scaledCompare(p[i], testValues[i], relTol, absTol, s);
      comparisons++;
    }
  }

  storeTestResults(testParams, failures, comparisons);

  return failures;
}

Teuchos::ParameterList*
SolverFactory::getTestParameters(int response_index) const
{
  Teuchos::ParameterList* result;

  if (response_index == 0 && appParams->isSublist("Regression Results")) {
    result = &(appParams->sublist("Regression Results"));
  } else {
    result = &(appParams->sublist(
        strint("Regression Results", response_index)));
  }

  TEUCHOS_TEST_FOR_EXCEPTION(
      result->isType<std::string>("Test Values"),
      std::logic_error,
      "Array information in XML file must now be of type Array(double)\n");
  result->validateParametersAndSetDefaults(
      *getValidRegressionResultsParameters(), 0);

  return result;
}

void
SolverFactory::storeTestResults(
    Teuchos::ParameterList* testParams,
    int            failures,
    int            comparisons) const
{
  // Store failures in param list (this requires mutable appParams!)
  testParams->set("Number of Failures", failures);
  testParams->set("Number of Comparisons Attempted", comparisons);
  *out << "\nCheckTestResults: Number of Comparisons Attempted = "
       << comparisons << std::endl;
}

bool
SolverFactory::scaledCompare(
    double             x1,
    double             x2,
    double             relTol,
    double             absTol,
    std::string const& name) const
{
  auto d       = fabs(x1 - x2);
  auto avg_mag = (0.5 * (fabs(x1) + fabs(x2)));
  auto rel_ok  = (d <= (avg_mag * relTol));
  auto abs_ok  = (d <= fabs(absTol));
  auto ok      = rel_ok || abs_ok;
  if (!ok) {
    *out << name << ": " << x1 << " != " << x2 << " (rel " << relTol << " abs "
         << absTol << ")\n";
  }
  return !ok;
}

void
SolverFactory::setSolverParamDefaults(
    Teuchos::ParameterList* appParams_,
    int            myRank)
{
  // Set the nonlinear solver method
  Teuchos::ParameterList& piroParams = appParams_->sublist("Piro");
  Teuchos::ParameterList& noxParams  = piroParams.sublist("NOX");
  noxParams.set("Nonlinear Solver", "Line Search Based");

  // Set the printing parameters in the "Printing" sublist
  Teuchos::ParameterList& printParams = noxParams.sublist("Printing");
  printParams.set("MyPID", myRank);
  printParams.set("Output Precision", 3);
  printParams.set("Output Processor", 0);
  printParams.set(
      "Output Information",
      NOX::Utils::OuterIteration + NOX::Utils::OuterIterationStatusTest +
          NOX::Utils::InnerIteration + NOX::Utils::Parameters +
          NOX::Utils::Details + NOX::Utils::LinearSolverDetails +
          NOX::Utils::Warning + NOX::Utils::Error);

  // Sublist for line search
  Teuchos::ParameterList& searchParams = noxParams.sublist("Line Search");
  searchParams.set("Method", "Full Step");

  // Sublist for direction
  Teuchos::ParameterList& dirParams = noxParams.sublist("Direction");
  dirParams.set("Method", "Newton");
  Teuchos::ParameterList& newtonParams = dirParams.sublist("Newton");
  newtonParams.set("Forcing Term Method", "Constant");

  // Sublist for linear solver for the Newton method
  Teuchos::ParameterList& lsParams = newtonParams.sublist("Linear Solver");
  lsParams.set("Max Iterations", 43);
  lsParams.set("Tolerance", 1e-4);

  // Sublist for status tests
  Teuchos::ParameterList& statusParams = noxParams.sublist("Status Tests");
  statusParams.set("Test Type", "Combo");
  statusParams.set("Combo Type", "OR");
  statusParams.set("Number of Tests", 2);
  Teuchos::ParameterList& normF = statusParams.sublist("Test 0");
  normF.set("Test Type", "NormF");
  normF.set("Tolerance", 1.0e-8);
  normF.set("Norm Type", "Two Norm");
  normF.set("Scale Type", "Unscaled");
  Teuchos::ParameterList& maxiters = statusParams.sublist("Test 1");
  maxiters.set("Test Type", "MaxIters");
  maxiters.set("Maximum Iterations", 10);
}

Teuchos::RCP<const Teuchos::ParameterList>
SolverFactory::getValidAppParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL = rcp(new Teuchos::ParameterList("ValidAppParams"));

  validPL->set("Build Type", "Tpetra", "The type of run (e.g., Epetra, Tpetra)");

  validPL->sublist("Problem", false, "Problem sublist");
  validPL->sublist("Debug Output", false, "Debug Output sublist");
  validPL->sublist("Scaling", false, "Jacobian/Residual Scaling sublist");
  validPL->sublist("DataTransferKit", false, "DataTransferKit sublist");
  validPL->sublist("DataTransferKit", false, "DataTransferKit sublist")
      .sublist(
          "Consistent Interpolation",
          false,
          "DTK Consistent Interpolation sublist");
  validPL->sublist("DataTransferKit", false, "DataTransferKit sublist")
      .sublist("Search", false, "DTK Search sublist");
  validPL->sublist("DataTransferKit", false, "DataTransferKit sublist")
      .sublist("L2 Projection", false, "DTK L2 Projection sublist");
  validPL->sublist("DataTransferKit", false, "DataTransferKit sublist")
      .sublist("Point Cloud", false, "DTK Point Cloud sublist");
  validPL->sublist("Discretization", false, "Discretization sublist");
  validPL->sublist("Quadrature", false, "Quadrature sublist");
  validPL->sublist("Regression Results", false, "Regression Results sublist");
  validPL->sublist("VTK", false, "DEPRECATED  VTK sublist");
  validPL->sublist("Piro", false, "Piro sublist");
  validPL->sublist("Coupled System", false, "Coupled system sublist");
  validPL->sublist("Alternating System", false, "Alternating system sublist");

  // validPL->set<std::string>("Jacobian Operator", "Have Jacobian", "Flag to
  // allow Matrix-Free specification in Piro");
  // validPL->set<double>("Matrix-Free Perturbation", 3.0e-7, "delta in
  // matrix-free formula");

  return validPL;
}


Teuchos::RCP<const Teuchos::ParameterList>
SolverFactory::getValidDebugParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL = rcp(new Teuchos::ParameterList("ValidDebugParams"));
  validPL->set<int>("Write Jacobian to MatrixMarket", 0, "Jacobian Number to Dump to MatrixMarket");
  validPL->set<int>("Compute Jacobian Condition Number", 0, "Jacobian Condition Number to Compute");
  validPL->set<int>("Write Residual to MatrixMarket", 0, "Residual Number to Dump to MatrixMarket");
  validPL->set<int>("Write Jacobian to Standard Output", 0, "Jacobian Number to Dump to Standard Output");
  validPL->set<int>("Write Residual to Standard Output", 0, "Residual Number to Dump to Standard Output");
  validPL->set<int>("Derivative Check", 0, "Derivative check");
  validPL->set<bool>("Write Solution to MatrixMarket", false, "Flag to Write Solution to MatrixMarket"); 
  validPL->set<bool>("Write Distributed Solution and Map to MatrixMarket", false, "Flag to Write Distributed Solution and Map to MatrixMarket"); 
  validPL->set<bool>("Write Solution to Standard Output", false, "Flag to Write Sotion to Standard Output");
  validPL->set<bool>("Analyze Memory", false, "Flag to Analyze Memory");
  return validPL; 
}

Teuchos::RCP<const Teuchos::ParameterList>
SolverFactory::getValidScalingParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL = rcp(new Teuchos::ParameterList("ValidScalingParams"));
  validPL->set<double>("Scale", 0.0, "Value of Scaling to Apply to Jacobian/Residual");
  validPL->set<bool>("Scale BC Dofs", false, "Flag to Scale Jacobian/Residual Rows Corresponding to DBC Dofs");
  validPL->set<std::string>("Type", "Constant", "Scaling Type (Constant, Diagonal, AbsRowSum)"); 
  return validPL; 
}

Teuchos::RCP<const Teuchos::ParameterList>
SolverFactory::getValidRegressionResultsParameters() const
{
  using Teuchos::Array;
  Teuchos::RCP<Teuchos::ParameterList> validPL = rcp(new Teuchos::ParameterList("ValidRegressionParams"));
  ;
  Array<double> ta;
  ;  // std::string to be converted to teuchos array

  validPL->set<double>(
      "Relative Tolerance",
      1.0e-4,
      "Relative Tolerance used in regression testing");
  validPL->set<double>(
      "Absolute Tolerance",
      1.0e-8,
      "Absolute Tolerance used in regression testing");

  validPL->set<int>(
      "Number of Comparisons", 0, "Number of responses to regress against");
  validPL->set<Array<double>>(
      "Test Values", ta, "Array of regression values for responses");

  validPL->set<int>(
      "Number of Sensitivity Comparisons",
      0,
      "Number of sensitivity vectors to regress against");

  const int maxSensTests = 10;
  for (int i = 0; i < maxSensTests; i++) {
    validPL->set<Array<double>>(
        strint("Sensitivity Test Values", i),
        ta,
        strint(
            "Array of regression values for Sensitivities w.r.t parameter", i));
    validPL->sublist(
        strint("Sensitivity Comparisons", i),
        false,
        "Sensitivity Comparisons sublist");
  }

  validPL->set<int>(
      "Number of Dakota Comparisons",
      0,
      "Number of paramters from Dakota runs to regress against");
  validPL->set<Array<double>>(
      "Dakota Test Values",
      ta,
      "Array of regression values for final parameters from Dakota runs");

  validPL->set<int>(
      "Number of Piro Analysis Comparisons",
      0,
      "Number of paramters from Analysis to regress against");
  validPL->set<Array<double>>(
      "Piro Analysis Test Values",
      ta,
      "Array of regression values for final parameters from Analysis runs");

  // Should deprecate these options, but need to remove them from all input
  // files
  validPL->set<int>(
      "Number of Stochastic Galerkin Comparisons",
      0,
      "Number of stochastic Galerkin expansions to regress against");

  const int maxSGTests = 10;
  for (int i = 0; i < maxSGTests; i++) {
    validPL->set<Array<double>>(
        strint("Stochastic Galerkin Expansion Test Values", i),
        ta,
        strint(
            "Array of regression values for stochastic Galerkin expansions",
            i));
  }

  validPL->set<int>(
      "Number of Stochastic Galerkin Mean Comparisons",
      0,
      "Number of SG mean responses to regress against");
  validPL->set<Array<double>>(
      "Stochastic Galerkin Mean Test Values",
      ta,
      "Array of regression values for SG mean responses");
  validPL->set<int>(
      "Number of Stochastic Galerkin Standard Deviation Comparisons",
      0,
      "Number of SG standard deviation responses to regress against");
  validPL->set<Array<double>>(
      "Stochastic Galerkin Standard Deviation Test Values",
      ta,
      "Array of regression values for SG standard deviation responses");
  // End of deprecated Stochastic Galerkin Options

  // These two are typically not set on input, just output.
  validPL->set<int>(
      "Number of Failures",
      0,
      "Output information from regression tests reporting number of failed "
      "tests");
  validPL->set<int>(
      "Number of Comparisons Attempted",
      0,
      "Output information from regression tests reporting number of "
      "comparisons attempted");

  return validPL;
}

Teuchos::RCP<const Teuchos::ParameterList>
SolverFactory::getValidParameterParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL = rcp(new Teuchos::ParameterList("ValidParameterParams"));
  ;

  validPL->set<int>("Number", 0);
  const int maxParameters = 100;
  for (int i = 0; i < maxParameters; i++) {
    validPL->set<std::string>(strint("Parameter", i), "");
  }
  return validPL;
}

Teuchos::RCP<const Teuchos::ParameterList>
SolverFactory::getValidResponseParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL = rcp(new Teuchos::ParameterList("ValidResponseParams"));
  ;
  validPL->set<std::string>("Collection Method", "Sum Responses");
  validPL->set<int>("Number of Response Vectors", 0);
  validPL->set<bool>("Observe Responses", true);
  validPL->set<int>("Responses Observation Frequency", 1);
  Teuchos::Array<unsigned int> defaultDataUnsignedInt;
  validPL->set<Teuchos::Array<unsigned int>>(
      "Relative Responses Markers",
      defaultDataUnsignedInt,
      "Array of responses for which relative change will be obtained");

  validPL->set<int>("Number", 0);
  validPL->set<int>("Equation", 0);
  const int maxParameters = 500;
  for (int i = 0; i < maxParameters; i++) {
    validPL->set<std::string>(strint("Response", i), "");
    validPL->sublist(strint("ResponseParams", i));
    validPL->sublist(strint("Response Vector", i));
  }
  return validPL;
}

} // namespace Albany
