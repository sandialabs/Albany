//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_SolverFactory.hpp"
#include "Albany_ObserverFactory.hpp"
#include "Albany_PiroObserver.hpp"
#include "Albany_PiroObserverT.hpp"
#include "Albany_SaveEigenData.hpp"
#include "Albany_ModelFactory.hpp"

#include "Piro_Epetra_SolverFactory.hpp"
#include "Piro_ProviderBase.hpp"

#include "Piro_SolverFactory.hpp"
#include "Piro_NOXSolver.hpp"
#include "Piro_NullSpaceUtils.hpp"

#include "Stratimikos_DefaultLinearSolverBuilder.hpp"

#ifdef ALBANY_IFPACK2
#include "Teuchos_AbstractFactoryStd.hpp"
#include "Thyra_Ifpack2PreconditionerFactory.hpp"
#endif /* ALBANY_IFPACK2 */

#ifdef ALBANY_MUELU
#include "Stratimikos_MueluTpetraHelpers.hpp"
#endif /* ALBANY_MUELU */

#ifdef ALBANY_QCAD
  #include "QCAD_Solver.hpp"
#endif

#include "Thyra_EpetraModelEvaluator.hpp"
#include "Thyra_DefaultModelEvaluatorWithSolveFactory.hpp"

#include "Thyra_DetachedVectorView.hpp"

#include "Teuchos_XMLParameterListHelpers.hpp"
#include "Teuchos_TestForException.hpp"

#include "NOX_Epetra_Observer.H"
#include "Rythmos_IntegrationObserverBase.hpp"

#include "Albany_Application.hpp"
#include "Albany_Utils.hpp"

#ifdef ALBANY_QCAD
#include "QCAD_CoupledPoissonSchrodinger.hpp"
#endif

//#include <stdexcept>


namespace Albany {

class NOXObserverConstructor : public Piro::ProviderBase<NOX::Epetra::Observer> {
public:
  explicit NOXObserverConstructor(const Teuchos::RCP<Application> &app) :
    factory_(app),
    instance_(Teuchos::null)
  {}

  virtual Teuchos::RCP<NOX::Epetra::Observer> getInstance(
      const Teuchos::RCP<Teuchos::ParameterList> &params);

private:
  NOXObserverFactory factory_;
  Teuchos::RCP<NOX::Epetra::Observer> instance_;
};

Teuchos::RCP<NOX::Epetra::Observer>
NOXObserverConstructor::getInstance(const Teuchos::RCP<Teuchos::ParameterList> &/*params*/)
{
  if (Teuchos::is_null(instance_)) {
    instance_ = factory_.createInstance();
  }
  return instance_;
}

class RythmosObserverConstructor : public Piro::ProviderBase<Rythmos::IntegrationObserverBase<double> > {
public:
  explicit RythmosObserverConstructor(const Teuchos::RCP<Application> &app) :
    factory_(app),
    instance_(Teuchos::null)
  {}

  virtual Teuchos::RCP<Rythmos::IntegrationObserverBase<double> > getInstance(
      const Teuchos::RCP<Teuchos::ParameterList> &params);

private:
  RythmosObserverFactory factory_;
  Teuchos::RCP<Rythmos::IntegrationObserverBase<double> > instance_;
};

Teuchos::RCP<Rythmos::IntegrationObserverBase<double> >
RythmosObserverConstructor::getInstance(const Teuchos::RCP<Teuchos::ParameterList> &/*params*/)
{
  if (Teuchos::is_null(instance_)) {
    instance_ = factory_.createInstance();
  }
  return instance_;
}

class SaveEigenDataConstructor : public Piro::ProviderBase<LOCA::SaveEigenData::AbstractStrategy> {
public:
  SaveEigenDataConstructor(
      Teuchos::ParameterList &locaParams,
      StateManager* pStateMgr,
      const Teuchos::RCP<Piro::ProviderBase<NOX::Epetra::Observer> > &observerProvider) :
    locaParams_(locaParams),
    pStateMgr_(pStateMgr),
    observerProvider_(observerProvider)
  {}

  virtual Teuchos::RCP<LOCA::SaveEigenData::AbstractStrategy> getInstance(
      const Teuchos::RCP<Teuchos::ParameterList> &params);

private:
  Teuchos::ParameterList &locaParams_;
  StateManager* pStateMgr_;

  Teuchos::RCP<Piro::ProviderBase<NOX::Epetra::Observer> > observerProvider_;
};

Teuchos::RCP<LOCA::SaveEigenData::AbstractStrategy>
SaveEigenDataConstructor::getInstance(const Teuchos::RCP<Teuchos::ParameterList> &params)
{
  const Teuchos::RCP<NOX::Epetra::Observer> noxObserver = observerProvider_->getInstance(params);
  return Teuchos::rcp(new SaveEigenData(locaParams_, noxObserver, pStateMgr_));
}

} // namespace Albany

using Teuchos::RCP;
using Teuchos::rcp;
using Teuchos::ParameterList;


Albany::SolverFactory::SolverFactory(
			  const std::string& inputFile,
			  const Albany_MPI_Comm& mcomm)
  : out(Teuchos::VerboseObjectBase::getDefaultOStream())
{
  RCP<Teuchos::Comm<int> > tcomm = Albany::createTeuchosCommFromMpiComm(mcomm);

  // Set up application parameters: read and broadcast XML file, and set defaults
  appParams = Teuchos::createParameterList("Albany Parameters");
  Teuchos::updateParametersFromXmlFileAndBroadcast(inputFile, appParams.ptr(), *tcomm);

  RCP<ParameterList> defaultSolverParams = rcp(new ParameterList());
  setSolverParamDefaults(defaultSolverParams.get(), tcomm->getRank());
  appParams->setParametersNotAlreadySet(*defaultSolverParams);

  appParams->validateParametersAndSetDefaults(*getValidAppParameters(),0);
}


Teuchos::RCP<EpetraExt::ModelEvaluator>
Albany::SolverFactory::create(
  const Teuchos::RCP<const Epetra_Comm>& appComm,
  const Teuchos::RCP<const Epetra_Comm>& solverComm,
  const Teuchos::RCP<const Epetra_Vector>& initial_guess)
{
  Teuchos::RCP<Albany::Application> dummyAlbanyApp;
  return createAndGetAlbanyApp(dummyAlbanyApp, appComm, solverComm, initial_guess);
}

Teuchos::RCP<Thyra::ResponseOnlyModelEvaluatorBase<ST> >
Albany::SolverFactory::createT(
  const Teuchos::RCP<const Epetra_Comm>& appComm,
  const Teuchos::RCP<const Epetra_Comm>& solverComm,
  const Teuchos::RCP<const Epetra_Vector>& initial_guess)
{
  Teuchos::RCP<Albany::Application> dummyAlbanyApp;
  return createAndGetAlbanyAppT(dummyAlbanyApp, appComm, solverComm, initial_guess);
}

Teuchos::RCP<EpetraExt::ModelEvaluator>
Albany::SolverFactory::createAndGetAlbanyApp(
  Teuchos::RCP<Albany::Application>& albanyApp,
  const Teuchos::RCP<const Epetra_Comm>& appComm,
  const Teuchos::RCP<const Epetra_Comm>& solverComm,
  const Teuchos::RCP<const Epetra_Vector>& initial_guess)
{
    const RCP<ParameterList> problemParams = Teuchos::sublist(appParams, "Problem");
    const std::string solutionMethod = problemParams->get("Solution Method", "Steady");

    if (solutionMethod == "Multi-Problem") {
      // QCAD::Solve is only example of a multi-app solver so far
#ifdef ALBANY_QCAD
      return rcp(new QCAD::Solver(appParams, solverComm));
#else /* ALBANY_QCAD */
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Must activate QCAD\n");
#endif /* ALBANY_QCAD */
    }

    if (solutionMethod == "Poisson-Schrodinger") {
#ifdef ALBANY_QCAD
      const RCP<QCAD::CoupledPoissonSchrodinger> ps_model = rcp(new QCAD::CoupledPoissonSchrodinger(appParams, solverComm, initial_guess));
      RCP<Albany::Application> poisson_app = ps_model->getPoissonApp();
      const RCP<ParameterList> piroParams = Teuchos::sublist(appParams, "Piro");

      // Create and setup the Piro solver factory
      Piro::Epetra::SolverFactory piroFactory;
      {
        // Do we need: Observers for output from time-stepper ??

	const RCP<Piro::ProviderBase<NOX::Epetra::Observer> > noxObserverProvider =
	  rcp(new NOXObserverConstructor(poisson_app));
	piroFactory.setSource<NOX::Epetra::Observer>(noxObserverProvider);

	// LOCA auxiliary objects -- needed?
      }
      return piroFactory.createSolver(piroParams, ps_model);

#else /* ALBANY_QCAD */
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Must activate QCAD\n");
#endif /* ALBANY_QCAD */
    }


    // Solver uses a single app, create it here along with observer
    RCP<Albany::Application> app;
    const RCP<EpetraExt::ModelEvaluator> model = createAlbanyAppAndModel(app, appComm, initial_guess);

    //Pass back albany app so that interface beyond ModelEvaluator can be used.
    // This is essentially a hack to allow additional in/out arguments beyond
    //  what ModelEvaluator specifies.
    albanyApp = app;

    const RCP<ParameterList> piroParams = Teuchos::sublist(appParams, "Piro");

    if (solutionMethod == "Continuation") {
      ParameterList& locaParams = piroParams->sublist("LOCA");
      if (app->getDiscretization()->hasRestartSolution()) {
        // Pick up problem time from restart file
        locaParams.sublist("Stepper").set("Initial Value", app->getDiscretization()->restartDataTime());
      }
    }

    // Create and setup the Piro solver factory
    Piro::Epetra::SolverFactory piroFactory;
    {
      // Observers for output from time-stepper
      const RCP<Piro::ProviderBase<Rythmos::IntegrationObserverBase<double> > > rythmosObserverProvider =
        rcp(new RythmosObserverConstructor(app));
      piroFactory.setSource<Rythmos::IntegrationObserverBase<double> >(rythmosObserverProvider);

      const RCP<Piro::ProviderBase<NOX::Epetra::Observer> > noxObserverProvider =
        rcp(new NOXObserverConstructor(app));
      piroFactory.setSource<NOX::Epetra::Observer>(noxObserverProvider);

      // LOCA auxiliary objects
      {
        const RCP<Albany::AdaptiveSolutionManager> adaptMgr = app->getAdaptSolMgr();
        piroFactory.setSource<Piro::Epetra::AdaptiveSolutionManager>(adaptMgr);

        const RCP<Piro::ProviderBase<LOCA::SaveEigenData::AbstractStrategy> > saveEigenDataProvider =
          rcp(new SaveEigenDataConstructor(piroParams->sublist("LOCA"), &app->getStateMgr(), noxObserverProvider));
        piroFactory.setSource<LOCA::SaveEigenData::AbstractStrategy>(saveEigenDataProvider);
      }
    }

    return piroFactory.createSolver(piroParams, model);
}

Teuchos::RCP<Thyra::ModelEvaluator<double> >
Albany::SolverFactory::createThyraSolverAndGetAlbanyApp(
    Teuchos::RCP<Application>& albanyApp,
    const Teuchos::RCP<const Epetra_Comm>& appComm,
    const Teuchos::RCP<const Epetra_Comm>& solverComm,
    const Teuchos::RCP<const Epetra_Vector>& initial_guess)
{
  const RCP<ParameterList> piroParams = Teuchos::sublist(appParams, "Piro");
  const Teuchos::Ptr<const std::string> solverToken(piroParams->getPtr<std::string>("Solver Type"));

  if (Teuchos::nonnull(solverToken) && *solverToken == "ThyraNOX") {
    piroParams->set("Solver Type", "NOX");

    RCP<Albany::Application> app;
    const RCP<EpetraExt::ModelEvaluator> model = createAlbanyAppAndModel(app, appComm, initial_guess);

    // Pass back albany app so that interface beyond ModelEvaluator can be used.
    // This is essentially a hack to allow additional in/out arguments beyond
    // what ModelEvaluator specifies.
    albanyApp = app;

    Stratimikos::DefaultLinearSolverBuilder linearSolverBuilder;
    linearSolverBuilder.setParameterList(Piro::extractStratimikosParams(piroParams));

    const RCP<Thyra::LinearOpWithSolveFactoryBase<double> > lowsFactory =
      createLinearSolveStrategy(linearSolverBuilder);

    const RCP<Thyra::ModelEvaluator<double> > thyraModel = Thyra::epetraModelEvaluator(model, lowsFactory);
    const RCP<Piro::ObserverBase<double> > observer = rcp(new PiroObserver(app));

    return rcp(new Piro::NOXSolver<double>(piroParams, thyraModel, observer));
  }

  const Teuchos::RCP<EpetraExt::ModelEvaluator> epetraSolver =
    this->createAndGetAlbanyApp(albanyApp, appComm, solverComm, initial_guess);
  return Thyra::epetraModelEvaluator(epetraSolver, Teuchos::null);
}

Teuchos::RCP<Thyra::ResponseOnlyModelEvaluatorBase<ST> >
Albany::SolverFactory::createAndGetAlbanyAppT(
  Teuchos::RCP<Albany::Application>& albanyApp,
  const Teuchos::RCP<const Epetra_Comm>& appComm,
  const Teuchos::RCP<const Epetra_Comm>& solverComm,
  const Teuchos::RCP<const Epetra_Vector>& initial_guess)
{
  const RCP<ParameterList> problemParams = Teuchos::sublist(appParams, "Problem");
  const std::string solutionMethod = problemParams->get("Solution Method", "Steady");

  RCP<Albany::Application> app;
  const RCP<Thyra::ModelEvaluator<ST> > modelT =
    createAlbanyAppAndModelT(app, appComm, initial_guess);

  // Pass back albany app so that interface beyond ModelEvaluator can be used.
  // This is essentially a hack to allow additional in/out arguments beyond
  // what ModelEvaluator specifies.
  albanyApp = app;

  const RCP<ParameterList> piroParams = Teuchos::sublist(appParams, "Piro");

  RCP<Thyra::ModelEvaluator<ST> > modelWithSolveT;
  if (Teuchos::nonnull(modelT->get_W_factory())) {
    modelWithSolveT = modelT;
  } else {
    // Setup linear solver
    Stratimikos::DefaultLinearSolverBuilder linearSolverBuilder;
#ifdef ALBANY_IFPACK2
    {
      typedef Thyra::PreconditionerFactoryBase<double> Base;
      typedef Thyra::Ifpack2PreconditionerFactory<Tpetra::CrsMatrix<double> > Impl;

      linearSolverBuilder.setPreconditioningStrategyFactory(Teuchos::abstractFactoryStd<Base, Impl>(), "Ifpack2");
    }
#endif /* ALBANY_IFPACK2 */
#ifdef ALBANY_MUELU
    Stratimikos::enableMueLuTpetra(linearSolverBuilder);
#endif /* ALBANY_MUELU */
    linearSolverBuilder.setParameterList(Piro::extractStratimikosParams(piroParams));

    const RCP<Thyra::LinearOpWithSolveFactoryBase<ST> > lowsFactory =
      createLinearSolveStrategy(linearSolverBuilder);

    modelWithSolveT =
      rcp(new Thyra::DefaultModelEvaluatorWithSolveFactory<ST>(modelT, lowsFactory));
  }

  Piro::SolverFactory piroFactory;
  const RCP<Piro::ObserverBase<double> > observer = rcp(new PiroObserverT(app));
  return piroFactory.createSolver<ST>(piroParams, modelWithSolveT, observer);
}

Teuchos::RCP<EpetraExt::ModelEvaluator>
Albany::SolverFactory::createAlbanyAppAndModel(
  Teuchos::RCP<Albany::Application>& albanyApp,
  const Teuchos::RCP<const Epetra_Comm>& appComm,
  const Teuchos::RCP<const Epetra_Vector>& initial_guess)
{
  // Create application
  albanyApp = rcp(new Albany::Application(appComm, appParams, initial_guess));

  // Validate Response list: may move inside individual Problem class
  const RCP<ParameterList> problemParams = Teuchos::sublist(appParams, "Problem");
  problemParams->sublist("Response Functions").
    validateParameters(*getValidResponseParameters(),0);

  // Create model evaluator
  Albany::ModelFactory modelFactory(appParams, albanyApp);

  return modelFactory.create();

}

Teuchos::RCP<Thyra::ModelEvaluator<ST> >
Albany::SolverFactory::createAlbanyAppAndModelT(
  Teuchos::RCP<Albany::Application>& albanyApp,
  const Teuchos::RCP<const Epetra_Comm>& appComm,
  const Teuchos::RCP<const Epetra_Vector>& initial_guess)
{
  // Create application
  albanyApp = rcp(new Albany::Application(appComm, appParams, initial_guess));

  // Validate Response list: may move inside individual Problem class
  RCP<ParameterList> problemParams = Teuchos::sublist(appParams, "Problem");
  problemParams->sublist("Response Functions").
    validateParameters(*getValidResponseParameters(),0);

  // If not explicitly specified, determine which Piro solver to use from the problem parameters
  const RCP<ParameterList> piroParams = Teuchos::sublist(appParams, "Piro");
  if (!piroParams->getPtr<std::string>("Solver Type")) {
    const std::string solutionMethod = problemParams->get("Solution Method", "Steady");
    TEUCHOS_TEST_FOR_EXCEPTION(
        solutionMethod != "Steady" &&
        solutionMethod != "Transient",
        std::logic_error,
        "Solution Method must be Steady or Transient, not : " <<
        solutionMethod <<
        "\n");

    const std::string secondOrder = problemParams->get("Second Order", "No");
    TEUCHOS_TEST_FOR_EXCEPTION(
        secondOrder != "No",
        std::logic_error,
        "Second Order is not supported" <<
        "\n");

    // Populate the Piro parameter list accordingly to inform the Piro solver factory
    std::string piroSolverToken;
    if (solutionMethod == "Steady") {
      piroSolverToken = "NOX";
    } else if (solutionMethod == "Transient") {
      piroSolverToken = "Rythmos";
    } else {
      // Piro cannot handle the corresponding problem
      piroSolverToken = "Unsupported";
    }

    piroParams->set("Solver Type", piroSolverToken);
  }

  // Create model evaluator
  Albany::ModelFactory modelFactory(appParams, albanyApp);
  return modelFactory.createT();
}

int Albany::SolverFactory::checkSolveTestResults(
  int response_index,
  int parameter_index,
  const Epetra_Vector* g,
  const Epetra_MultiVector* dgdp) const
{
  ParameterList* testParams = getTestParameters(response_index);

  int failures = 0;
  int comparisons = 0;
  const double relTol = testParams->get<double>("Relative Tolerance");
  const double absTol = testParams->get<double>("Absolute Tolerance");

  // Get number of responses (g) to test
  const int numResponseTests = testParams->get<int>("Number of Comparisons");
  if (numResponseTests > 0) {
    if (g == NULL || numResponseTests > g->MyLength()) failures += 1000;
    else { // do comparisons
      Teuchos::Array<double> testValues =
        testParams->get<Teuchos::Array<double> >("Test Values");

      TEUCHOS_TEST_FOR_EXCEPT(numResponseTests != testValues.size());
      for (int i=0; i<testValues.size(); i++) {
        failures += scaledCompare((*g)[i], testValues[i], relTol, absTol);
        comparisons++;
      }
    }
  }

  // Repeat comparisons for sensitivities
  Teuchos::ParameterList *sensitivityParams;
  std::string sensitivity_sublist_name =
    Albany::strint("Sensitivity Comparisons", parameter_index);
  if (parameter_index == 0 && !testParams->isSublist(sensitivity_sublist_name))
    sensitivityParams = testParams;
  else
    sensitivityParams = &(testParams->sublist(sensitivity_sublist_name));
  const int numSensTests =
    sensitivityParams->get<int>("Number of Sensitivity Comparisons");
  if (numSensTests > 0) {
    if (dgdp == NULL || numSensTests > dgdp->MyLength()) failures += 10000;
    else {
      for (int i=0; i<numSensTests; i++) {
        Teuchos::Array<double> testSensValues =
          sensitivityParams->get<Teuchos::Array<double> >(Albany::strint("Sensitivity Test Values",i));
        TEUCHOS_TEST_FOR_EXCEPT(dgdp->NumVectors() != testSensValues.size());
        for (int j=0; j<dgdp->NumVectors(); j++) {
          failures += scaledCompare((*dgdp)[j][i], testSensValues[j], relTol, absTol);
          comparisons++;
        }
      }
    }
  }

  storeTestResults(testParams, failures, comparisons);

  return failures;
}

int Albany::SolverFactory::checkDakotaTestResults(
  int response_index,
  const Teuchos::SerialDenseVector<int,double>* drdv) const
{
  ParameterList* testParams = getTestParameters(response_index);

  int failures = 0;
  int comparisons = 0;
  const double relTol = testParams->get<double>("Relative Tolerance");
  const double absTol = testParams->get<double>("Absolute Tolerance");

  const int numDakotaTests = testParams->get<int>("Number of Dakota Comparisons");
  if (numDakotaTests > 0 && drdv != NULL) {

    if (numDakotaTests > drdv->length()) {
      failures += 100000;
    } else {
      // Read accepted test results
      Teuchos::Array<double> testValues =
        testParams->get<Teuchos::Array<double> >("Dakota Test Values");

      TEUCHOS_TEST_FOR_EXCEPT(numDakotaTests != testValues.size());
      for (int i=0; i<numDakotaTests; i++) {
        failures += scaledCompare((*drdv)[i], testValues[i], relTol, absTol);
        comparisons++;
      }
    }
  }

  storeTestResults(testParams, failures, comparisons);

  return failures;
}

int Albany::SolverFactory::checkAnalysisTestResults(
  int response_index,
  const Teuchos::RCP<Thyra::VectorBase<double> >& tvec) const
{
  ParameterList* testParams = getTestParameters(response_index);

  int failures = 0;
  int comparisons = 0;
  const double relTol = testParams->get<double>("Relative Tolerance");
  const double absTol = testParams->get<double>("Absolute Tolerance");

  int numPiroTests = testParams->get<int>("Number of Piro Analysis Comparisons");
  if (numPiroTests > 0 && tvec != Teuchos::null) {

     // Create indexable thyra vector
      ::Thyra::DetachedVectorView<double> p(tvec);

    if (numPiroTests > p.subDim()) failures += 300000;
    else { // do comparisons
      // Read accepted test results
      Teuchos::Array<double> testValues =
        testParams->get<Teuchos::Array<double> >("Piro Analysis Test Values");

      TEUCHOS_TEST_FOR_EXCEPT(numPiroTests != testValues.size());
      for (int i=0; i<numPiroTests; i++) {
        failures += scaledCompare(p[i], testValues[i], relTol, absTol);
        comparisons++;
      }
    }
  }

  storeTestResults(testParams, failures, comparisons);

  return failures;
}

int Albany::SolverFactory::checkSGTestResults(
  int response_index,
  const Teuchos::RCP<Stokhos::EpetraVectorOrthogPoly>& g_sg,
  const Epetra_Vector* g_mean,
  const Epetra_Vector* g_std_dev) const
{
  ParameterList* testParams = getTestParameters(response_index);

  int failures = 0;
  int comparisons = 0;
  const double relTol = testParams->get<double>("Relative Tolerance");
  const double absTol = testParams->get<double>("Absolute Tolerance");

  int numSGTests = testParams->get<int>("Number of Stochastic Galerkin Comparisons", 0);
  if (numSGTests > 0 && g_sg != Teuchos::null) {
    if (numSGTests > (*g_sg)[0].MyLength()) failures += 10000;
    else {
      for (int i=0; i<numSGTests; i++) {
        Teuchos::Array<double> testSGValues =
          testParams->get<Teuchos::Array<double> >
            (Albany::strint("Stochastic Galerkin Expansion Test Values",i));
        TEUCHOS_TEST_FOR_EXCEPT(g_sg->size() != testSGValues.size());
	for (int j=0; j<g_sg->size(); j++) {
	  failures +=
	    scaledCompare((*g_sg)[j][i], testSGValues[j], relTol, absTol);
          comparisons++;
        }
      }
    }
  }

  // Repeat comparisons for SG mean statistics
  int numMeanResponseTests = testParams->get<int>("Number of Stochastic Galerkin Mean Comparisons", 0);
  if (numMeanResponseTests > 0) {

    if (g_mean == NULL || numMeanResponseTests > g_mean->MyLength()) failures += 30000;
    else { // do comparisons
      Teuchos::Array<double> testValues =
        testParams->get<Teuchos::Array<double> >("Stochastic Galerkin Mean Test Values");

      TEUCHOS_TEST_FOR_EXCEPT(numMeanResponseTests != testValues.size());
      for (int i=0; i<testValues.size(); i++) {
        failures += scaledCompare((*g_mean)[i], testValues[i], relTol, absTol);
        comparisons++;
      }
    }
  }

  // Repeat comparisons for SG standard deviation statistics
  int numSDResponseTests = testParams->get<int>("Number of Stochastic Galerkin Standard Deviation Comparisons", 0);
  if (numSDResponseTests > 0) {

    if (g_std_dev == NULL || numSDResponseTests > g_std_dev->MyLength()) failures +=50000;
    else { // do comparisons
      Teuchos::Array<double> testValues =
        testParams->get<Teuchos::Array<double> >("Stochastic Galerkin Standard Deviation Test Values");

      TEUCHOS_TEST_FOR_EXCEPT(numSDResponseTests != testValues.size());
      for (int i=0; i<testValues.size(); i++) {
        failures += scaledCompare((*g_std_dev)[i], testValues[i], relTol, absTol);
        comparisons++;
      }
    }
  }

  storeTestResults(testParams, failures, comparisons);

  return failures;
}

ParameterList* Albany::SolverFactory::getTestParameters(int response_index) const
{
  ParameterList* result;

  if (response_index == 0 && appParams->isSublist("Regression Results")) {
    result = &(appParams->sublist("Regression Results"));
  } else {
    result = &(appParams->sublist(Albany::strint("Regression Results", response_index)));
  }

  TEUCHOS_TEST_FOR_EXCEPTION(result->isType<string>("Test Values"), std::logic_error,
    "Array information in XML file must now be of type Array(double)\n");
  result->validateParametersAndSetDefaults(*getValidRegressionResultsParameters(),0);

  return result;
}

void Albany::SolverFactory::storeTestResults(
  ParameterList* testParams,
  int failures,
  int comparisons) const
{
  // Store failures in param list (this requires mutable appParams!)
  testParams->set("Number of Failures", failures);
  testParams->set("Number of Comparisons Attempted", comparisons);
  *out << "\nCheckTestResults: Number of Comparisons Attempted = "
       << comparisons << endl;
}

int Albany::SolverFactory::scaledCompare(double x1, double x2, double relTol, double absTol) const
{
  double diff = fabs(x1 - x2) / (0.5*fabs(x1) + 0.5*fabs(x2) + fabs(absTol));

  if (diff < relTol) return 0; //pass
  else               return 1; //fail
}


void Albany::SolverFactory::setSolverParamDefaults(
              ParameterList* appParams_, int myRank)
{
    // Set the nonlinear solver method
    ParameterList& piroParams = appParams_->sublist("Piro");
    ParameterList& noxParams = piroParams.sublist("NOX");
    noxParams.set("Nonlinear Solver", "Line Search Based");

    // Set the printing parameters in the "Printing" sublist
    ParameterList& printParams = noxParams.sublist("Printing");
    printParams.set("MyPID", myRank);
    printParams.set("Output Precision", 3);
    printParams.set("Output Processor", 0);
    printParams.set("Output Information",
		    NOX::Utils::OuterIteration +
		    NOX::Utils::OuterIterationStatusTest +
		    NOX::Utils::InnerIteration +
		    NOX::Utils::Parameters +
		    NOX::Utils::Details +
		    NOX::Utils::LinearSolverDetails +
		    NOX::Utils::Warning +
		    NOX::Utils::Error);

    // Sublist for line search
    ParameterList& searchParams = noxParams.sublist("Line Search");
    searchParams.set("Method", "Full Step");

    // Sublist for direction
    ParameterList& dirParams = noxParams.sublist("Direction");
    dirParams.set("Method", "Newton");
    ParameterList& newtonParams = dirParams.sublist("Newton");
    newtonParams.set("Forcing Term Method", "Constant");

    // Sublist for linear solver for the Newton method
    ParameterList& lsParams = newtonParams.sublist("Linear Solver");
    lsParams.set("Aztec Solver", "GMRES");
    lsParams.set("Max Iterations", 43);
    lsParams.set("Tolerance", 1e-4);
    lsParams.set("Output Frequency", 20);
    lsParams.set("Preconditioner", "Ifpack");

    // Sublist for status tests
    ParameterList& statusParams = noxParams.sublist("Status Tests");
    statusParams.set("Test Type", "Combo");
    statusParams.set("Combo Type", "OR");
    statusParams.set("Number of Tests", 2);
    ParameterList& normF = statusParams.sublist("Test 0");
    normF.set("Test Type", "NormF");
    normF.set("Tolerance", 1.0e-8);
    normF.set("Norm Type", "Two Norm");
    normF.set("Scale Type", "Unscaled");
    ParameterList& maxiters = statusParams.sublist("Test 1");
    maxiters.set("Test Type", "MaxIters");
    maxiters.set("Maximum Iterations", 10);

}

RCP<const ParameterList>
Albany::SolverFactory::getValidAppParameters() const
{
  RCP<ParameterList> validPL = rcp(new ParameterList("ValidAppParams"));;
  validPL->sublist("Problem",            false, "Problem sublist");
  validPL->sublist("Debug Output",       false, "Debug Output sublist");
  validPL->sublist("Discretization",     false, "Discretization sublist");
  validPL->sublist("Quadrature",         false, "Quadrature sublist");
  validPL->sublist("Regression Results", false, "Regression Results sublist");
  validPL->sublist("VTK",                false, "DEPRECATED  VTK sublist");
  validPL->sublist("Piro",               false, "Piro sublist");
  validPL->sublist("Coupled System",     false, "Coupled system sublist");

  // validPL->set<string>("Jacobian Operator", "Have Jacobian", "Flag to allow Matrix-Free specification in Piro");
  // validPL->set<double>("Matrix-Free Perturbation", 3.0e-7, "delta in matrix-free formula");

  return validPL;
}

RCP<const ParameterList>
Albany::SolverFactory::getValidRegressionResultsParameters() const
{
  using Teuchos::Array;
  RCP<ParameterList> validPL = rcp(new ParameterList("ValidRegressionParams"));;
  Array<double> ta;; // string to be converted to teuchos array

  validPL->set<double>("Relative Tolerance", 1.0e-4,
          "Relative Tolerance used in regression testing");
  validPL->set<double>("Absolute Tolerance", 1.0e-8,
          "Absolute Tolerance used in regression testing");

  validPL->set<int>("Number of Comparisons", 0,
          "Number of responses to regress against");
  validPL->set<Array<double> >("Test Values", ta,
          "Array of regression values for responses");

  validPL->set<int>("Number of Sensitivity Comparisons", 0,
          "Number of sensitivity vectors to regress against");

  const int maxSensTests=10;
  for (int i=0; i<maxSensTests; i++) {
    validPL->set<Array<double> >(
       Albany::strint("Sensitivity Test Values",i), ta,
       Albany::strint("Array of regression values for Sensitivities w.r.t parameter",i));
  }

  validPL->set<int>("Number of Dakota Comparisons", 0,
          "Number of paramters from Dakota runs to regress against");
  validPL->set<Array<double> >("Dakota Test Values", ta,
          "Array of regression values for final parameters from Dakota runs");

  validPL->set<int>("Number of Piro Analysis Comparisons", 0,
          "Number of paramters from Analysis to regress against");
  validPL->set<Array<double> >("Piro Analysis Test Values", ta,
          "Array of regression values for final parameters from Analysis runs");

  validPL->set<int>("Number of Stochastic Galerkin Comparisons", 0,
          "Number of stochastic Galerkin expansions to regress against");

  const int maxSGTests=10;
  for (int i=0; i<maxSGTests; i++) {
    validPL->set<Array<double> >(
       Albany::strint("Stochastic Galerkin Expansion Test Values",i), ta,
       Albany::strint("Array of regression values for stochastic Galerkin expansions",i));
  }

  validPL->set<int>("Number of Stochastic Galerkin Mean Comparisons", 0,
          "Number of SG mean responses to regress against");
  validPL->set<Array<double> >("Stochastic Galerkin Mean Test Values", ta,
          "Array of regression values for SG mean responses");
  validPL->set<int>(
    "Number of Stochastic Galerkin Standard Deviation Comparisons", 0,
    "Number of SG standard deviation responses to regress against");
  validPL->set<Array<double> >(
    "Stochastic Galerkin Standard Deviation Test Values", ta,
    "Array of regression values for SG standard deviation responses");

  // These two are typically not set on input, just output.
  validPL->set<int>("Number of Failures", 0,
     "Output information from regression tests reporting number of failed tests");
  validPL->set<int>("Number of Comparisons Attempted", 0,
     "Output information from regression tests reporting number of comparisons attempted");

  return validPL;
}

RCP<const ParameterList>
Albany::SolverFactory::getValidParameterParameters() const
{
  RCP<ParameterList> validPL = rcp(new ParameterList("ValidParameterParams"));;

  validPL->set<int>("Number", 0);
  const int maxParameters = 100;
  for (int i=0; i<maxParameters; i++) {
    validPL->set<std::string>(Albany::strint("Parameter",i), "");
  }
  return validPL;
}

RCP<const ParameterList>
Albany::SolverFactory::getValidResponseParameters() const
{
  RCP<ParameterList> validPL = rcp(new ParameterList("ValidResponseParams"));;

  validPL->set<int>("Number of Response Vectors", 0);
  validPL->set<int>("Number", 0);
  validPL->set<int>("Equation", 0);
  const int maxParameters = 100;
  for (int i=0; i<maxParameters; i++) {
    validPL->set<std::string>(Albany::strint("Response",i), "");
    validPL->sublist(Albany::strint("ResponseParams",i));
    validPL->sublist(Albany::strint("Response Vector",i));
  }
  return validPL;
}
