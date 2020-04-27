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
#else
  (void) linearSolverBuilder;
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


  std::string solution_method =
      appParams->sublist("Problem").get("Solution Method", "Steady");

  Teuchos::RCP<Teuchos::ParameterList> defaultSolverParams = rcp(new Teuchos::ParameterList());
  setSolverParamDefaults(defaultSolverParams.get(), comm->getRank());
  appParams->setParametersNotAlreadySet(*defaultSolverParams);

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
  std::string solution_method =
      appParams->sublist("Problem").get("Solution Method", "Steady");

  Teuchos::RCP<Teuchos::ParameterList> defaultSolverParams = rcp(new Teuchos::ParameterList());
  setSolverParamDefaults(defaultSolverParams.get(), comm->getRank());
  appParams->setParametersNotAlreadySet(*defaultSolverParams);

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
    return piroFactory.createSolver<ST>(
        piroParams, modelWithSolve, solMgr, observer_);
  } else {
    return piroFactory.createSolver<ST>(
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
        appComm, appParams, initial_guess));
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
