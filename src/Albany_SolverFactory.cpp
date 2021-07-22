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
#include "Stratimikos_FROSch_decl.hpp"
#include "Stratimikos_FROSch_def.hpp"
#endif /* ALBANY_FROSCH */

#ifdef ALBANY_TEKO
#include "Teko_StratimikosFactory.hpp"
#endif

#include "Thyra_DefaultModelEvaluatorWithSolveFactory.hpp"
#include "Thyra_DetachedVectorView.hpp"

#include "Teuchos_TestForException.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#include "Teuchos_YamlParameterListHelpers.hpp"

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
 : m_out(Teuchos::VerboseObjectBase::getDefaultOStream())
{
  // Set up application parameters: read and broadcast XML file, and set
  // defaults
  // Teuchos::RCP<Teuchos::ParameterList> input_
  auto appParams = Teuchos::createParameterList("Albany Parameters");

  std::string const input_extension = getFileExtension(inputFile);

  if (input_extension == "yaml" || input_extension == "yml") {
    Teuchos::updateParametersFromYamlFileAndBroadcast(
        inputFile, appParams.ptr(), *comm);
  } else {
    Teuchos::updateParametersFromXmlFileAndBroadcast(
        inputFile, appParams.ptr(), *comm);
  }

  setup(appParams,comm);
}

SolverFactory::
SolverFactory(const Teuchos::RCP<Teuchos::ParameterList>& appParams,
              const Teuchos::RCP<const Teuchos_Comm>&     comm)
 : m_out(Teuchos::VerboseObjectBase::getDefaultOStream())
{
  setup (appParams,comm);
}

void SolverFactory::
setup(const Teuchos::RCP<Teuchos::ParameterList>& appParams,
      const Teuchos::RCP<const Teuchos_Comm>&     comm)
{
  std::string solution_method =
      appParams->sublist("Problem").get("Solution Method", "Steady");

  //Initialize "Number Of Parameters" to zero if the "Parameters" sublist is not present
  if(!appParams->sublist("Problem").isSublist("Parameters"))
     appParams->sublist("Problem").sublist("Parameters").set("Number Of Parameters", 0);
  
  //Initialize "Number Of Responses" to zero if the "Responses" sublist is not present
  if(!appParams->sublist("Problem").isSublist("Response Functions"))
     appParams->sublist("Problem").sublist("Response Functions").set("Number Of Responses", 0);

  Teuchos::RCP<Teuchos::ParameterList> defaultSolverParams = rcp(new Teuchos::ParameterList());
  setSolverParamDefaults(defaultSolverParams.get(), comm->getRank());
  appParams->setParametersNotAlreadySet(*defaultSolverParams);

  if (!appParams->isParameter("Build Type")) {
    if (comm->getRank()==0) {
      *m_out << "\nWARNING! You have not set the entry 'Build Type' in the input parameter list. This will cause Albany to *assume* a Tpetra build.\n"
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

  m_appParams = appParams;
}

Teuchos::RCP<Application>
SolverFactory::
createApplication (const Teuchos::RCP<const Teuchos_Comm>& appComm,
                   const Teuchos::RCP<const Thyra_Vector>& initial_guess)
{
  auto albanyApp = Teuchos::rcp(new Application(appComm, m_appParams, initial_guess));

  return albanyApp;
}

Teuchos::RCP<ModelEvaluator>
SolverFactory::createModel (const Teuchos::RCP<Application>& app)
{
  // Validate Response list
  // TODO: may move inside individual Problem class
  const auto problemParams = Teuchos::sublist(m_appParams, "Problem");
  problemParams->sublist("Response Functions")
      .validateParameters(*getValidResponseParameters(), 0);

  return Teuchos::rcp(new ModelEvaluator(app,m_appParams));
}

Teuchos::RCP<Thyra::ResponseOnlyModelEvaluatorBase<ST>>
SolverFactory::
createSolver (const Teuchos::RCP<ModelEvaluator>&     model,
              const Teuchos::RCP<const Teuchos_Comm>& solverComm)
{
  const Teuchos::RCP<Teuchos::ParameterList> problemParams = Teuchos::sublist(m_appParams, "Problem");
  const std::string solutionMethod = problemParams->get("Solution Method", "Steady");

  const auto piroParams = Teuchos::sublist(m_appParams, "Piro");
  const auto stratList =  Piro::extractStratimikosParams(piroParams);

  // If not explicitly specified, determine which Piro solver to use from the
  // problem parameters
  if (!piroParams->getPtr<std::string>("Solver Type")) {

    /* TODO: this should be a boolean, not a string ! */
    const std::string secondOrder = problemParams->get("Second Order", "No");
    TEUCHOS_TEST_FOR_EXCEPTION(secondOrder != "No", std::logic_error,
        "Second Order is not supported.\n");

    // Populate the Piro parameter list accordingly to inform the Piro solver
    // factory
    std::string piroSolverToken;
    if (solutionMethod == "Steady") {
      piroSolverToken = "NOX";
    } else if (solutionMethod == "Transient") {
      piroSolverToken = "Tempus";
    } else {
      // Piro cannot handle the corresponding problem
      piroSolverToken = "Unsupported";
    }

    ALBANY_ASSERT(piroSolverToken != "Unsupported",
        "Unsupported Solution Method: " << solutionMethod);

    piroParams->set("Solver Type", piroSolverToken);
  }

  if (Teuchos::is_null(stratList)) {
    *m_out << "Error: cannot locate Stratimikos solver parameters in the input "
            "file."
         << std::endl;
    *m_out << "Printing the Piro parameter list:" << std::endl;
    piroParams->print(*m_out);
    // GAH: this is an error - should be fatal
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
        "Error: cannot locate Stratimikos solver parameters in the input file.\n")
  }

  // TODO: remove this error when Belos + Ifpack2/Amesos2 is fixed
  if (stratList->isParameter("Linear Solver Type") && stratList->isParameter("Preconditioner Type") &&
      stratList->isParameter("Preconditioner Types")) {
    if (stratList->get<std::string>("Linear Solver Type") == "Belos" &&
        stratList->get<std::string>("Preconditioner Type") == "Ifpack2") {
      const auto ifpack2List = Teuchos::sublist(Teuchos::sublist(stratList, "Preconditioner Types"), "Ifpack2");
      if (ifpack2List->isParameter("Prec Type"))
        TEUCHOS_TEST_FOR_EXCEPTION(ifpack2List->get<std::string>("Prec Type") == "Amesos2", std::logic_error,
            "Belos + Ifpack2/Amesos2 is currently not working in Trilinos.\n");
    }
  }

  Teuchos::RCP<Thyra_ModelEvaluator> modelWithSolve;
  if (Teuchos::nonnull(model->get_W_factory())) {
    modelWithSolve = model;
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

    modelWithSolve = rcp(new Thyra::DefaultModelEvaluatorWithSolveFactory<ST>(model, lowsFactory));
  }

  const auto app    = model->getAlbanyApp();
  const auto solMgr = app->getAdaptSolMgr();

  Piro::SolverFactory piroFactory;
  m_observer = Teuchos::rcp(new PiroObserver(app, modelWithSolve));
  return piroFactory.createSolver<ST>(
       piroParams, modelWithSolve, Teuchos::null, m_observer);
  TEUCHOS_TEST_FOR_EXCEPTION(
      true,
      std::logic_error,
      "Reached end of createModel()"
          << "\n");

  // Silence compiler warning in case it wasn't used (due to ifdef logic)
  (void) solverComm;

  return Teuchos::null;
}

void SolverFactory::
setSolverParamDefaults(Teuchos::ParameterList* appParams,
                       int            myRank)
{
  // Set the nonlinear solver method
  Teuchos::ParameterList& piroParams = appParams->sublist("Piro");
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
  const int maxRegression = 10;
  for (int i = 0; i < maxRegression; i++) {
    validPL->sublist(strint("Regression For Response", i), false, "Regression Results sublist");
  }
  validPL->sublist("VTK", false, "DEPRECATED  VTK sublist");
  validPL->sublist("Piro", false, "Piro sublist");
  validPL->sublist("Coupled System", false, "Coupled system sublist");
  validPL->sublist("Alternating System", false, "Alternating system sublist");
  validPL->sublist("Block ID to Physics ID Mapping", false, "Block to physics sublist");
  validPL->sublist("Physics Blocks", false, "Physics blocks sublist");
  validPL->sublist("Field Order", false, "Field order sublist");

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
  validPL->set<int>("Write Solution to MatrixMarket", 0, "Solution Number to Dump to MatrixMarket");
  validPL->set<int>("Write Jacobian to Standard Output", 0, "Jacobian Number to Dump to Standard Output");
  validPL->set<int>("Write Residual to Standard Output", 0, "Residual Number to Dump to Standard Output");
  validPL->set<int>("Derivative Check", 0, "Derivative check");
  validPL->set<int>("Write Solution to MatrixMarket", 0, "Solution Number to Dump to MatrixMarket");
  validPL->set<bool>("Write Distributed Solution and Map to MatrixMarket", false, "Flag to Write Distributed Solution and Map to MatrixMarket");
  validPL->set<int>("Write Solution to Standard Output", 0, "Residual Number to Dump to Standard Output");
  validPL->set<bool>("Analyze Memory", false, "Flag to Analyze Memory");
  validPL->set<bool>("Report Timers", true, "Whether to report timers at the end of execution");
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
  validPL->set<int>("Number Of Responses", 0);
  validPL->set<bool>("Observe Responses", true);
  validPL->set<int>("Responses Observation Frequency", 1);
  Teuchos::Array<unsigned int> defaultDataUnsignedInt;
  validPL->set<Teuchos::Array<unsigned int>>(
      "Relative Responses Markers",
      defaultDataUnsignedInt,
      "Array of responses for which relative change will be obtained");
  const int maxNumResponses = 50;
  for (int i = 0; i < maxNumResponses; i++)
    validPL->sublist(strint("Response", i), false, "Response sublist");
  return validPL;
}

} // namespace Albany
