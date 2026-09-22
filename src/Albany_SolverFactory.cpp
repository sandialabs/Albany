//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_SolverFactory.hpp"
#include "Albany_PiroObserver.hpp"
#include "Albany_PiroTempusObserver.hpp"
#include "Albany_ModelEvaluator.hpp"
#include "Albany_ExplicitODEModelEvaluator.hpp"
#include "Albany_AbstractProblem.hpp"
#include "Albany_Application.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_Macros.hpp"
#include "Albany_StringUtils.hpp"

#include "Piro_ProviderBase.hpp"
#include "Piro_NOXSolver.hpp"
#include "Piro_SolverFactory.hpp"
#include "Piro_StratimikosUtils.hpp"
#include "Piro_ProductModelEval.hpp"

#include "Stratimikos_DefaultLinearSolverBuilder.hpp"

#ifdef ALBANY_IFPACK2
#include "Teuchos_AbstractFactoryStd.hpp"
#include "Thyra_Ifpack2PreconditionerFactory.hpp"
#endif /* ALBANY_IFPACK2 */

#ifdef ALBANY_MUELU
#include "Stratimikos_MueLuHelpers.hpp"
#endif /* ALBANY_MUELU */

#ifdef ALBANY_FROSCH
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
enableMueLu(
    Stratimikos::DefaultLinearSolverBuilder&    linearSolverBuilder)
{
#ifdef ALBANY_MUELU
  Stratimikos::enableMueLu<ST, LO, Tpetra_GO, KokkosNode>(linearSolverBuilder);
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

Teuchos::RCP<Thyra_LOWS_Factory>
buildLinearSolveStrategy(const Teuchos::RCP<Teuchos::ParameterList>& stratParams)
{
  const auto linearSolverBuilder = Teuchos::rcp(new Stratimikos::DefaultLinearSolverBuilder);
  enableMueLu(*linearSolverBuilder);
  enableFROSch(*linearSolverBuilder);
#ifdef ALBANY_TEKO
  Teko::addTekoToStratimikosBuilder(*linearSolverBuilder, "Teko");
#endif
  linearSolverBuilder->setParameterList(stratParams);
  return createLinearSolveStrategy(*linearSolverBuilder);
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

  Teuchos::RCP<Teuchos::ParameterList> defaultSolverParams = Teuchos::rcp(new Teuchos::ParameterList());
  setSolverParamDefaults(defaultSolverParams.get(), comm->getRank());
  appParams->setParametersNotAlreadySet(*defaultSolverParams);

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
SolverFactory::createModel (const Teuchos::RCP<Application>& app,
		            const bool adjoint_model)
{
  // Validate Response list
  // TODO: may move inside individual Problem class
  const auto problemParams = Teuchos::sublist(m_appParams, "Problem");
  problemParams->sublist("Response Functions")
      .validateParameters(*getValidResponseParameters(), 0);

  return Teuchos::rcp(new ModelEvaluator(app,m_appParams,adjoint_model));
}

Teuchos::RCP<Thyra::ResponseOnlyModelEvaluatorBase<ST>>
SolverFactory::
createSolver (const Teuchos::RCP<ModelEvaluator>&     model_tmp,
              const Teuchos::RCP<ModelEvaluator>&     adjointModel_tmp,
              const bool                              forwardMode)
{
  const auto piroParams = Teuchos::sublist(m_appParams, "Piro");
  const auto stratList =  Piro::extractStratimikosParams(piroParams);

  Teuchos::RCP<Thyra::ModelEvaluator<ST>> model, adjointModel;
  if ( model_tmp->Np() > 1) {
    std::vector<int> p_indices;
    if (forwardMode) {
      p_indices.resize(model_tmp->Np());
      for(int i=0; i<model_tmp->Np(); ++i) {
        p_indices[i] = i;
      } 
    } else {
      auto rolParams = piroParams->sublist("Analysis").sublist("ROL");
      int num_parameters = rolParams.isParameter("Number Of Parameters") ? rolParams.get<int>("Number Of Parameters") : model_tmp->Np();
      p_indices.resize(num_parameters);
      for(int i=0; i<num_parameters; ++i) {
        std::ostringstream ss; ss << "Parameter Vector Index " << i;
        p_indices[i] = rolParams.get<int>(ss.str(), i);
      } 
    }
    model = Teuchos::rcp(new Piro::ProductModelEvaluator<double>(model_tmp,p_indices));
    adjointModel = adjointModel_tmp.is_null() ? Teuchos::null : Teuchos::rcp(new Piro::ProductModelEvaluator<double>(adjointModel_tmp,p_indices));
  }
  else {
    model = model_tmp;
    adjointModel = adjointModel_tmp;
  }

  const Teuchos::RCP<Teuchos::ParameterList> problemParams = Teuchos::sublist(m_appParams, "Problem");
  const std::string solutionMethod = problemParams->get("Solution Method", "Steady");


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

  // Explicit Tempus stepper on a problem with an algebraic constraint (a semi-explicit
  // index-1 DAE, e.g. LandIce::StokesFOThickness): expose the model to Tempus as an
  // explicit ODE for the differential dofs, solving the constraint at every stage (e.g.,
  // explicit ice thickness evolution, with the FO velocity solved from H). The monolithic
  // model cannot be used with an explicit stepper at all, since its mass matrix is singular.
  Teuchos::RCP<ExplicitODEModelEvaluator> explicitOdeModel;
  if (solutionMethod=="Transient" && piroParams->isSublist("Tempus") &&
      usesExplicitTempusStepper(piroParams->sublist("Tempus")))
  {
    const auto albanyApp = model_tmp->getAlbanyApp();
    Teuchos::RCP<Thyra_Vector> algebraicMask, differentialMask;
    const bool is_dae = albanyApp->getProblem()->getDAEMasks(*albanyApp->getDiscretization(),
                                                             algebraicMask,differentialMask);
    if (is_dae) {
      TEUCHOS_TEST_FOR_EXCEPTION (Teuchos::nonnull(adjointModel), std::logic_error,
          "Error! Explicit time integration of a DAE does not support adjoint models.\n");

      auto& tempusParams = piroParams->sublist("Tempus");

      // Options: everything but the mass matrix comes from the Problem list. The mass
      // matrix options are the ones Piro uses for explicit steppers, so that they mean
      // the same thing whether or not the model is a DAE.
      auto explicitParams = Teuchos::rcp(new Teuchos::ParameterList("Explicit Time Integration"));
      if (problemParams->isSublist("Explicit Time Integration")) {
        *explicitParams = problemParams->sublist("Explicit Time Integration");
      }
      TEUCHOS_TEST_FOR_EXCEPTION (explicitParams->isParameter("Lump Mass Matrix") ||
                                  explicitParams->isParameter("Constant Mass Matrix"), std::logic_error,
          "Error! The mass matrix options of explicit time integration are read from the\n"
          "       Piro->Tempus list ('Lump Mass Matrix', 'Constant Mass Matrix'), and not\n"
          "       from Problem->Explicit Time Integration.\n");
      explicitParams->set("Lump Mass Matrix",     tempusParams.get<bool>("Lump Mass Matrix",false));
      explicitParams->set("Constant Mass Matrix", tempusParams.get<bool>("Constant Mass Matrix",false));

      // The algebraic (Newton) solve is configured as a steady solve, through Piro->NOX,
      // using its Stratimikos list if present (otherwise, the Piro->Tempus one).
      // The (consistent) mass matrix solve uses the Piro->Tempus->Stratimikos list.
      const auto noxParams = Teuchos::sublist(piroParams, "NOX");
      auto algebraicStratParams = stratList;
      if (noxParams->isSublist("Direction") &&
          noxParams->sublist("Direction").isSublist("Newton") &&
          noxParams->sublist("Direction").sublist("Newton").isSublist("Stratimikos Linear Solver") &&
          noxParams->sublist("Direction").sublist("Newton").sublist("Stratimikos Linear Solver").isSublist("Stratimikos")) {
        algebraicStratParams = Teuchos::sublist(Teuchos::sublist(Teuchos::sublist(Teuchos::sublist(
            noxParams, "Direction"), "Newton"), "Stratimikos Linear Solver"), "Stratimikos");
      }

      explicitOdeModel = Teuchos::rcp(new ExplicitODEModelEvaluator(
          model, algebraicMask, differentialMask, explicitParams, noxParams,
          buildLinearSolveStrategy(algebraicStratParams),
          buildLinearSolveStrategy(stratList)));
      model = explicitOdeModel;

      // The model already returns x_dot: Piro must not wrap it in an InvertMassMatrixDecorator
      tempusParams.set("Invert Mass Matrix", false);

      *m_out << "Note: the problem has an algebraic constraint and the Tempus stepper is explicit,\n"
             << "      so Albany::ExplicitODEModelEvaluator is used: the constraint is solved "
             << (explicitParams->isParameter("Algebraic Update") && (explicitParams->get<std::string>("Algebraic Update")=="Once Per Step")
                 ? "once per step,\n" : "at every stage,\n")
             << "      and Piro's mass matrix inversion is turned off (the model returns x_dot).\n";
    }
  }

  Teuchos::RCP<Thyra_ModelEvaluator> modelWithSolve;
  Teuchos::RCP<Thyra_ModelEvaluator> adjointModelWithSolve = Teuchos::null;
  if (Teuchos::nonnull(model->get_W_factory())) {
    modelWithSolve = model;
    adjointModelWithSolve = adjointModel; 
  } else {
    // Setup linear solver
    const auto linearSolverBuilder = Teuchos::rcp(new Stratimikos::DefaultLinearSolverBuilder);
    enableMueLu(*linearSolverBuilder);
    enableFROSch(*linearSolverBuilder);
#ifdef ALBANY_TEKO
    // Note: We are not currently building Teko preconditioners through Stratimikos because this would require a block discretization
    // (i.e. a discretization that builds physically blocked Jacobians/vectors). This might be something we want to do in the future.
    // Instead, we will construct our own Teko block preconditioner for our problem of interest inside the Albany application. We'll
    // keep this here so we can use the same input schema.
    Teko::addTekoToStratimikosBuilder(*linearSolverBuilder, "Teko");
#endif
    linearSolverBuilder->setParameterList(stratList);

    const Teuchos::RCP<Thyra_LOWS_Factory> lowsFactory =
        createLinearSolveStrategy(*linearSolverBuilder);

    const auto app = model_tmp->getAlbanyApp();
    if (app->suppliesPreconditioner()) {
      app->setLinearSolverBuilder(linearSolverBuilder);
      lowsFactory->unsetPreconditionerFactory();
    }

    modelWithSolve = Teuchos::rcp(new Thyra::DefaultModelEvaluatorWithSolveFactory<ST>(model, lowsFactory));
    if (adjointModel != Teuchos::null) {
      adjointModelWithSolve = Teuchos::rcp(new Thyra::DefaultModelEvaluatorWithSolveFactory<ST>(adjointModel, lowsFactory));
    }
  }

  const auto app = model_tmp->getAlbanyApp();

  Teuchos::RCP<PiroObserver> observer;

  if (solutionMethod=="Transient") {
    auto tempusObserver = Teuchos::rcp(new PiroTempusObserver(app, modelWithSolve));
    if (Teuchos::nonnull(explicitOdeModel)) {
      tempusObserver->setExplicitODEModel(explicitOdeModel);
    }
    observer = tempusObserver;
  } else {
    observer = Teuchos::rcp(new PiroObserver(app, modelWithSolve));
  }

  Piro::SolverFactory piroFactory;
  return piroFactory.createSolver<ST>(
       piroParams, modelWithSolve, adjointModelWithSolve, observer);
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
}

Teuchos::RCP<const Teuchos::ParameterList>
SolverFactory::getValidAppParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL = Teuchos::rcp(new Teuchos::ParameterList("ValidAppParams"));
  validPL->sublist("Problem", false, "Problem sublist");
  validPL->sublist("Debug Output", false, "Debug Output sublist");
  validPL->sublist("Scaling", false, "Jacobian/Residual Scaling sublist");
  validPL->sublist("Discretization", false, "Discretization sublist");
  const int maxRegression = 10;
  for (int i = 0; i < maxRegression; i++) {
    validPL->sublist(util::strint("Regression For Response", i), false, "Regression Results sublist");
  }
  validPL->sublist("Piro", false, "Piro sublist");

  return validPL;
}


Teuchos::RCP<const Teuchos::ParameterList>
SolverFactory::getValidDebugParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL = Teuchos::rcp(new Teuchos::ParameterList("ValidDebugParams"));
  validPL->set<int>("Write Jacobian to MatrixMarket", 0, "Jacobian Number to Dump to MatrixMarket");
  validPL->set<int>("Compute Jacobian Condition Number", 0, "Jacobian Condition Number to Compute");
  validPL->set<int>("Write Residual to MatrixMarket", 0, "Residual Number to Dump to MatrixMarket");
  validPL->set<int>("Write Solution to MatrixMarket", 0, "Solution Number to Dump to MatrixMarket");
  validPL->set<int>("Write Jacobian to Standard Output", 0, "Jacobian Number to Dump to Standard Output");
  validPL->set<int>("Write Residual to Standard Output", 0, "Residual Number to Dump to Standard Output");
  validPL->set<int>("Derivative Check", 0, "Derivative check");
  validPL->set<int>("Write Solution to MatrixMarket", 0, "Solution Number to Dump to MatrixMarket");
  validPL->set<bool>("Write Distributed Solution and Map to MatrixMarket", false, "Flag to Write Distributed Solution and Map to MatrixMarket");
  validPL->set<bool>("Write DgDp to MatrixMarket", false, "Flag to Write DgDp to MatrixMarket");
  validPL->set<int>("Write Solution to Standard Output", 0, "Residual Number to Dump to Standard Output");
  validPL->set<bool>("Analyze Memory", false, "Flag to Analyze Memory");
  validPL->set<bool>("Report Timers", true, "Whether to report timers at the end of execution");
  validPL->set<bool>("Report Parameter Changes", true, "Whether to report changes in parameters during execution");
  validPL->set<bool>("Report MPI Info", false, "Whether to report MPI processor name and rank");
  return validPL; 
}

Teuchos::RCP<const Teuchos::ParameterList>
SolverFactory::getValidScalingParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL = Teuchos::rcp(new Teuchos::ParameterList("ValidScalingParams"));
  validPL->set<double>("Scale", 0.0, "Value of Scaling to Apply to Jacobian/Residual");
  validPL->set<bool>("Scale BC Dofs", false, "Flag to Scale Jacobian/Residual Rows Corresponding to DBC Dofs");
  validPL->set<std::string>("Type", "Constant", "Scaling Type (Constant, Diagonal, AbsRowSum)");
  return validPL;
}

Teuchos::RCP<const Teuchos::ParameterList>
SolverFactory::getValidParameterParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL = Teuchos::rcp(new Teuchos::ParameterList("ValidParameterParams"));
  ;

  validPL->set<int>("Number", 0);
  const int maxParameters = 100;
  for (int i = 0; i < maxParameters; i++) {
    validPL->set<std::string>(util::strint("Parameter", i), "");
  }
  return validPL;
}

Teuchos::RCP<const Teuchos::ParameterList>
SolverFactory::getValidResponseParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL = Teuchos::rcp(new Teuchos::ParameterList("ValidResponseParams"));
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
    validPL->sublist(util::strint("Response", i), false, "Response sublist");
  return validPL;
}

} // namespace Albany
