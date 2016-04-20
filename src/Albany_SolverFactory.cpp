//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/12/14: Epetra ifdef'ed out!
//No epetra if setting ALBANY_EPETRA_EXE off.

#include "Albany_SolverFactory.hpp"
#if defined(ALBANY_EPETRA)
#  include "Albany_PiroObserver.hpp"
#  include "Piro_Epetra_SolverFactory.hpp"
#  include "Petra_Converters.hpp"
#  include "Albany_SaveEigenData.hpp"
#  include "Albany_ObserverFactory.hpp"
#  include "NOX_Epetra_Observer.H"
#endif
#include "Albany_PiroObserverT.hpp"
#include "Albany_ModelFactory.hpp"

#include "Piro_ProviderBase.hpp"

#include "Piro_SolverFactory.hpp"
#include "Piro_NOXSolver.hpp"
#include "Piro_StratimikosUtils.hpp"

#include "Stratimikos_DefaultLinearSolverBuilder.hpp"

#ifdef ALBANY_IFPACK2
#  include "Teuchos_AbstractFactoryStd.hpp"
#  include "Thyra_Ifpack2PreconditionerFactory.hpp"
#endif /* ALBANY_IFPACK2 */

#ifdef ALBANY_MUELU
#  include "Stratimikos_MueLuHelpers.hpp"
#endif /* ALBANY_MUELU */

#ifdef ALBANY_TEKO
#  include "Teko_StratimikosFactory.hpp"
#endif

#ifdef ALBANY_QCAD
#if defined(ALBANY_EPETRA)
  #include "QCAD_Solver.hpp"
  #include "QCAD_CoupledPoissonSchrodinger.hpp"
  #include "QCAD_CoupledPSObserver.hpp"
  #include "QCAD_GenEigensolver.hpp"
#endif
  #include "QCADT_CoupledPoissonSchrodinger.hpp"
#endif

#include "Albany_ModelEvaluatorT.hpp"
#ifdef ALBANY_ATO
  #include "ATO_Solver.hpp"
#endif

#if defined(ALBANY_LCM) && defined(HAVE_STK)
  #include "Schwarz_Multiscale.hpp"
  #include "Schwarz_PiroObserverT.hpp"
#endif

#ifdef ALBANY_AERAS
  #include "Aeras/Aeras_HVDecorator.hpp"
#endif

#include "Thyra_DefaultModelEvaluatorWithSolveFactory.hpp"
#include "Thyra_DetachedVectorView.hpp"

#include "Teuchos_XMLParameterListHelpers.hpp"
#include "Teuchos_TestForException.hpp"

#include "Rythmos_IntegrationObserverBase.hpp"

#include "Albany_Application.hpp"
#include "Albany_Utils.hpp"


extern bool TpetraBuild;

#if defined(ALBANY_EPETRA)
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

class NOXStatelessObserverConstructor :
    public Piro::ProviderBase<NOX::Epetra::Observer> {
public:
  explicit NOXStatelessObserverConstructor (
    const Teuchos::RCP<Application> &app)
    : factory_(app),
      instance_(Teuchos::null)
  {}

  virtual Teuchos::RCP<NOX::Epetra::Observer> getInstance (
    const Teuchos::RCP<Teuchos::ParameterList> &params)
  {
    if (Teuchos::is_null(instance_)) instance_ = factory_.createInstance();
    return instance_;
  }
private:
  NOXStatelessObserverFactory factory_;
  Teuchos::RCP<NOX::Epetra::Observer> instance_;
};

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
#endif

using Teuchos::RCP;
using Teuchos::rcp;
using Teuchos::ParameterList;



Albany::SolverFactory::SolverFactory(
        const std::string& inputFile,
        const RCP<const Teuchos_Comm>& tcomm)
  : out(Teuchos::VerboseObjectBase::getDefaultOStream())
{

  // Set up application parameters: read and broadcast XML file, and set defaults
  //RCP<ParameterList> input_
  appParams = Teuchos::createParameterList("Albany Parameters");
  Teuchos::updateParametersFromXmlFileAndBroadcast(inputFile, appParams.ptr(), *tcomm);

  // do not set default solver parameters for QCAD::Solver or ATO::Solver problems,
  // ... as they handle this themselves
  std::string solution_method = appParams->sublist("Problem").get("Solution Method", "Steady");
  if (solution_method != "QCAD Multi-Problem" &&
      solution_method != "ATO Problem" ) {
    RCP<ParameterList> defaultSolverParams = rcp(new ParameterList());
    setSolverParamDefaults(defaultSolverParams.get(), tcomm->getRank());
    appParams->setParametersNotAlreadySet(*defaultSolverParams);
  }

  appParams->validateParametersAndSetDefaults(*getValidAppParameters(),0);
}


Albany::SolverFactory::SolverFactory(
                  const Teuchos::RCP<Teuchos::ParameterList>& input_appParams,
        const RCP<const Teuchos_Comm>& tcomm)
  : appParams(input_appParams), out(Teuchos::VerboseObjectBase::getDefaultOStream())
{

  // do not set default solver parameters for QCAD::Solver or ATO::Solver problems,
  // ... as they handle this themselves
  std::string solution_method = appParams->sublist("Problem").get("Solution Method", "Steady");
  if (solution_method != "QCAD Multi-Problem" &&
      solution_method != "ATO Problem" ) {
    RCP<ParameterList> defaultSolverParams = rcp(new ParameterList());
    setSolverParamDefaults(defaultSolverParams.get(), tcomm->getRank());
    appParams->setParametersNotAlreadySet(*defaultSolverParams);
  }

  appParams->validateParametersAndSetDefaults(*getValidAppParameters(),0);
}

Albany::SolverFactory::~SolverFactory(){

#if defined(ALBANY_EPETRA)
  // Release the model to eliminate RCP circular reference
  if(Teuchos::nonnull(thyraModelFactory))
    thyraModelFactory->releaseModel();
#endif

#ifdef ALBANY_DEBUG
  *out << "Calling destructor for Albany_SolverFactory" << std::endl;
#endif
}


#if defined(ALBANY_EPETRA)
Teuchos::RCP<EpetraExt::ModelEvaluator>
Albany::SolverFactory::create(
  const Teuchos::RCP<const Teuchos_Comm>& appCommT,
  const Teuchos::RCP<const Teuchos_Comm>& solverCommT,
  const Teuchos::RCP<const Tpetra_Vector>& initial_guess)
{
  Teuchos::RCP<Albany::Application> dummyAlbanyApp;
  return createAndGetAlbanyApp(dummyAlbanyApp, appCommT, solverCommT, initial_guess);
}
#endif

Teuchos::RCP<Thyra::ResponseOnlyModelEvaluatorBase<ST> >
Albany::SolverFactory::createT(
  const Teuchos::RCP<const Teuchos_Comm>& appComm,
  const Teuchos::RCP<const Teuchos_Comm>& solverComm,
  const Teuchos::RCP<const Tpetra_Vector>& initial_guess)
{
  Teuchos::RCP<Albany::Application> dummyAlbanyApp;
//  Teuchos::RCP<Albany::ApplicationT> dummyAlbanyApp;
  return createAndGetAlbanyAppT(dummyAlbanyApp, appComm, solverComm, initial_guess);
}

#if defined(ALBANY_EPETRA)
Teuchos::RCP<EpetraExt::ModelEvaluator>
Albany::SolverFactory::createAndGetAlbanyApp(
  Teuchos::RCP<Albany::Application>& albanyApp,
  const Teuchos::RCP<const Teuchos_Comm>& appCommT,
  const Teuchos::RCP<const Teuchos_Comm>& solverCommT,
  const Teuchos::RCP<const Tpetra_Vector>& initial_guess,
  bool createAlbanyApp)
{
    const RCP<ParameterList> problemParams = Teuchos::sublist(appParams, "Problem");
    const std::string solutionMethod = problemParams->get("Solution Method", "Steady");
    const Teuchos::RCP<const Epetra_Comm> appComm = Albany::createEpetraCommFromTeuchosComm(appCommT);
    const Teuchos::RCP<const Epetra_Comm> solverComm = Albany::createEpetraCommFromTeuchosComm(solverCommT);

    if (solutionMethod == "QCAD Multi-Problem") {
#ifdef ALBANY_QCAD
      RCP<Epetra_Vector> initial_guessE;
      if(Teuchos::nonnull(initial_guess))
        Petra::TpetraVector_To_EpetraVector(initial_guess, *initial_guessE, appComm);
      return rcp(new QCAD::Solver(appParams, solverComm, initial_guessE));
#else /* ALBANY_QCAD */
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Must activate QCAD\n");
#endif /* ALBANY_QCAD */
    }

    if (solutionMethod == "QCAD Poisson-Schrodinger") {
#ifdef ALBANY_QCAD
      RCP<Epetra_Vector> initial_guessE;
      if(Teuchos::nonnull(initial_guess))
        Petra::TpetraVector_To_EpetraVector(initial_guess, *initial_guessE, appComm);
      const RCP<QCAD::CoupledPoissonSchrodinger> ps_model = rcp(new QCAD::CoupledPoissonSchrodinger(appParams, solverComm, initial_guessE));
      const RCP<ParameterList> piroParams = Teuchos::sublist(appParams, "Piro");

      // Create and setup the Piro solver factory
      Piro::Epetra::SolverFactory piroEpetraFactory;
      {
        // Do we need: Observers for output from time-stepper ??
  const RCP<Piro::ProviderBase<NOX::Epetra::Observer> > noxObserverProvider =
    rcp(new QCAD::CoupledPS_NOXObserverConstructor(ps_model));
    //  rcp(new NOXObserverConstructor(poisson_app));
  piroEpetraFactory.setSource<NOX::Epetra::Observer>(noxObserverProvider);

  // LOCA auxiliary objects -- needed?
      }
    // Piro::Epetra::SolverFactory
      return piroEpetraFactory.createSolver(piroParams, ps_model);

#else /* ALBANY_QCAD */
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Must activate QCAD\n");
#endif /* ALBANY_QCAD */
    }

    if (solutionMethod == "Eigensolve") {
#ifdef ALBANY_QCAD

      RCP<Albany::Application> app;
      if(createAlbanyApp) {
        app = rcp(new Albany::Application(appCommT, appParams, initial_guess));
        albanyApp = app;
      }
      else app = albanyApp;

      const RCP<EpetraExt::ModelEvaluator> model = createModel(app, appCommT);


      //QCAD::GenEigensolver uses a state manager as an observer (for now)
      RCP<Albany::StateManager> observer = rcp( &(app->getStateMgr()), false);

      // Currently, QCAD eigensolver just uses LOCA's eigensolver list under Piro -- maybe give it it's own list
      //   outside of Piro?
      const RCP<ParameterList> eigensolveParams = rcp(&(appParams->sublist("Piro").sublist("LOCA").sublist("Stepper").sublist("Eigensolver")), false);
      const RCP<QCAD::GenEigensolver> es_model = rcp(new QCAD::GenEigensolver(eigensolveParams, model, observer, solverComm));
      return es_model;

#else /* ALBANY_QCAD */
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Must activate QCAD\n");
#endif /* ALBANY_QCAD */
    }
#if defined(ALBANY_LCM)
    if (solutionMethod == "Coupled Schwarz") {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Coupled Schwarz Solution Method does not work with Albany executable!  Please re-run with AlbanyT Executable. \n");
    }
#endif
    if (solutionMethod == "ATO Problem") {
#ifdef ALBANY_ATO
//IK, 10/16/14: need to convert ATO::Solver to Tpetra
      RCP<Epetra_Vector> initial_guessE;
      if(Teuchos::nonnull(initial_guess))
        Petra::TpetraVector_To_EpetraVector(initial_guess, *initial_guessE, appComm);
      return rcp(new ATO::Solver(appParams, solverComm, initial_guessE));
#else /* ALBANY_ATO */
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Must activate ATO (topological optimization)\n");
#endif /* ALBANY_ATO */
    }

    // Solver uses a single app, create it here along with observer
    RCP<Albany::Application> app;

    if(createAlbanyApp) {
      app = rcp(new Albany::Application(appCommT, appParams, initial_guess));

      //Pass back albany app so that interface beyond ModelEvaluator can be used.
      // This is essentially a hack to allow additional in/out arguments beyond
      //  what ModelEvaluator specifies.
      albanyApp = app;
    }
    else app = albanyApp;

    const RCP<EpetraExt::ModelEvaluator> model = createModel(app, appCommT);

    const RCP<ParameterList> piroParams = Teuchos::sublist(appParams, "Piro");

    if (solutionMethod == "Continuation") {
      ParameterList& locaParams = piroParams->sublist("LOCA");
      if (app->getDiscretization()->hasRestartSolution()) {
        // Pick up problem time from restart file
        locaParams.sublist("Stepper").set("Initial Value", app->getDiscretization()->restartDataTime());
      }
    }

    // Create and setup the Piro solver factory
    Piro::Epetra::SolverFactory piroEpetraFactory;
    {
      // Observers for output from time-stepper
      const RCP<Piro::ProviderBase<Rythmos::IntegrationObserverBase<double> > > rythmosObserverProvider =
        rcp(new RythmosObserverConstructor(app));
      piroEpetraFactory.setSource<Rythmos::IntegrationObserverBase<double> >(rythmosObserverProvider);

      const RCP<Piro::ProviderBase<NOX::Epetra::Observer> > noxObserverProvider =
        rcp(new NOXObserverConstructor(app));
      piroEpetraFactory.setSource<NOX::Epetra::Observer>(noxObserverProvider);

      // LOCA auxiliary objects
      {
        const RCP<AAdapt::AdaptiveSolutionManager> adaptMgr = app->getAdaptSolMgr();
        piroEpetraFactory.setSource<Piro::Epetra::AdaptiveSolutionManager>(adaptMgr);

        const RCP<Piro::ProviderBase<NOX::Epetra::Observer> >
          noxStatelessObserverProvider = rcp(
            new NOXStatelessObserverConstructor(app));
        const RCP<Piro::ProviderBase<LOCA::SaveEigenData::AbstractStrategy> > saveEigenDataProvider =
          rcp(new SaveEigenDataConstructor(piroParams->sublist("LOCA"), &app->getStateMgr(),
                                           noxStatelessObserverProvider));
        piroEpetraFactory.setSource<LOCA::SaveEigenData::AbstractStrategy>(saveEigenDataProvider);
      }
    }

    // Piro::Epetra::SolverFactory
    return piroEpetraFactory.createSolver(piroParams, model);
}

Teuchos::RCP<Thyra::ModelEvaluator<double> >
Albany::SolverFactory::createThyraSolverAndGetAlbanyApp(
    Teuchos::RCP<Application>& albanyApp,
    const Teuchos::RCP<const Teuchos_Comm>& appCommT,
    const Teuchos::RCP<const Teuchos_Comm>& solverCommT,
    const Teuchos::RCP<const Tpetra_Vector>& initial_guess,
    bool createAlbanyApp)
{
  const Teuchos::RCP<const Epetra_Comm> appComm = Albany::createEpetraCommFromTeuchosComm(appCommT);

  const RCP<ParameterList> piroParams = Teuchos::sublist(appParams, "Piro");
  const Teuchos::Ptr<const std::string> solverToken(piroParams->getPtr<std::string>("Solver Type"));

  const RCP<ParameterList> problemParams = Teuchos::sublist(appParams, "Problem");
  const std::string solutionMethod = problemParams->get("Solution Method", "Steady");


  if (Teuchos::nonnull(solverToken) && *solverToken == "ThyraNOX") {
    piroParams->set("Solver Type", "NOX");

    RCP<Albany::Application> app;


    //WARINING: if createAlbanyApp==true, then albanyApp will be constructed twice, here and below, when calling
    //    createAndGetAlbanyApp. Why? (Mauro)
    if(createAlbanyApp) {
      app = rcp(new Albany::Application(appCommT, appParams, initial_guess));

      // Pass back albany app so that interface beyond ModelEvaluator can be used.
      // This is essentially a hack to allow additional in/out arguments beyond
      // what ModelEvaluator specifies.
      albanyApp = app;
    }
    else app = albanyApp;

    // Creates the Albany::ModelEvaluator
    const RCP<EpetraExt::ModelEvaluator> model = createModel(app, appCommT);



    Stratimikos::DefaultLinearSolverBuilder linearSolverBuilder;
    linearSolverBuilder.setParameterList(Piro::extractStratimikosParams(piroParams));

    const RCP<Thyra::LinearOpWithSolveFactoryBase<double> > lowsFactory =
      createLinearSolveStrategy(linearSolverBuilder);

    if ( solutionMethod == "QCAD Multi-Problem" ||
         solutionMethod == "QCAD Poisson-Schrodinger" ||
         solutionMethod == "ATO Problem" ) {
       // These QCAD and ATO solvers do not contain a primary Albany::Application instance and so albanyApp is null.
       // For now, do not resize the response vectors. FIXME sort out this issue.
       const RCP<Thyra::ModelEvaluator<double> > thyraModel = Thyra::epetraModelEvaluator(model, lowsFactory);
       const RCP<Piro::ObserverBase<double> > observer = rcp(new PiroObserver(app));
       return rcp(new Piro::NOXSolver<double>(piroParams, thyraModel, observer));
    }
    else {
//      const RCP<AAdapt::AdaptiveModelFactory> thyraModelFactory = albanyApp->getAdaptSolMgr()->modelFactory();
      thyraModelFactory = albanyApp->getAdaptSolMgr()->modelFactory();
      const RCP<Thyra::ModelEvaluator<double> > thyraModel = thyraModelFactory->create(model, lowsFactory);
      if(TpetraBuild){
        const RCP<Piro::ObserverBase<double> > observer = rcp(new PiroObserverT(app, thyraModel));
        return rcp(new Piro::NOXSolver<double>(piroParams, thyraModel, observer));
      }
      else {
        const RCP<Piro::ObserverBase<double> > observer = rcp(new PiroObserver(app));
        return rcp(new Piro::NOXSolver<double>(piroParams, thyraModel, observer));
      }
    }
  }

  const Teuchos::RCP<EpetraExt::ModelEvaluator> epetraSolver =
    this->createAndGetAlbanyApp(albanyApp, appCommT, solverCommT, initial_guess, createAlbanyApp);

  if ( solutionMethod == "QCAD Multi-Problem" ||
       solutionMethod == "QCAD Poisson-Schrodinger" ||
       solutionMethod == "ATO Problem" ) {
    return Thyra::epetraModelEvaluator(epetraSolver, Teuchos::null);
  }
  else {
//    const RCP<AAdapt::AdaptiveModelFactory> thyraModelFactory = albanyApp->getAdaptSolMgr()->modelFactory();
    thyraModelFactory = albanyApp->getAdaptSolMgr()->modelFactory();
    return thyraModelFactory->create(epetraSolver, Teuchos::null);
  }
}
#endif

namespace {

void enableIfpack2(Stratimikos::DefaultLinearSolverBuilder& linearSolverBuilder)
{
#ifdef ALBANY_IFPACK2
  typedef Thyra::PreconditionerFactoryBase<ST> Base;
  typedef Thyra::Ifpack2PreconditionerFactory<Tpetra_CrsMatrix> Impl;
  linearSolverBuilder.setPreconditioningStrategyFactory(Teuchos::abstractFactoryStd<Base, Impl>(), "Ifpack2");
#endif
}

void enableMueLu(Teuchos::RCP<Albany::Application>& albanyApp,
                 const Teuchos::RCP<Teuchos::ParameterList>& stratList,
                 Stratimikos::DefaultLinearSolverBuilder& linearSolverBuilder)
{
#ifdef ALBANY_MUELU
  Stratimikos::enableMueLu<LO, GO, KokkosNode>(linearSolverBuilder);
#endif
}
} // namespace

Teuchos::RCP<Thyra::ResponseOnlyModelEvaluatorBase<ST> >
Albany::SolverFactory::createAndGetAlbanyAppT(
  Teuchos::RCP<Albany::Application>& albanyApp,
  const Teuchos::RCP<const Teuchos_Comm>& appComm,
  const Teuchos::RCP<const Teuchos_Comm>& solverComm,
  const Teuchos::RCP<const Tpetra_Vector>& initial_guess,
  bool createAlbanyApp)
{
  const RCP<ParameterList> problemParams = Teuchos::sublist(appParams, "Problem");
  const std::string solutionMethod = problemParams->get("Solution Method", "Steady");

  if (solutionMethod == "QCAD Multi-Problem") {
#ifdef ALBANY_QCAD
     TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "QCAD Multi-Problem does not work with AlbanyT executable!  QCAD::Solver class needs to be implemented with Thyra::ModelEvaluator instead of EpetraExt. \n");
    //IK, 8/26/14: need to implement QCAD::SolverT class that returns Thyra::ModelEvaluator instead of EpetraExt one
    //and takes in Tpetra / Teuchos_Comm objects.
    //return rcp(new QCAD::SolverT(appParams, solverComm, initial_guess));
#else /* ALBANY_QCAD */
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Must activate QCAD\n");
#endif /* ALBANY_QCAD */
    }

    if (solutionMethod == "QCAD Poisson-Schrodinger") {
#ifdef ALBANY_QCAD
     TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "QCAD Poisson-Schrodinger does not work with AlbanyT executable!  QCAD::CoupledPoissonSchrodinger class needs to be implemented with Thyra::ModelEvaluator instead of EpetraExt. \n");
      std::cout <<"In Albany_SolverFactory: solutionMethod = QCAD Poisson-Schrodinger!" << std::endl;
      const RCP<ParameterList> piroParams = Teuchos::sublist(appParams, "Piro");
      const Teuchos::RCP<Teuchos::ParameterList> stratList = Piro::extractStratimikosParams(piroParams);
      // Create and setup the Piro solver factory
      Piro::SolverFactory piroFactory;
      // Setup linear solver
      Stratimikos::DefaultLinearSolverBuilder linearSolverBuilder;
      //FIXME, IKT, 5/22/15: inject Ifpack2, MueLu, Teko into Stratimikos.
      linearSolverBuilder.setParameterList(stratList);
      const RCP<Thyra::LinearOpWithSolveFactoryBase<ST> > lowsFactory =
        createLinearSolveStrategy(linearSolverBuilder);
      const RCP<QCADT::CoupledPoissonSchrodinger> ps_model =
            rcp(new QCADT::CoupledPoissonSchrodinger(appParams, solverComm, initial_guess, lowsFactory));
     //FIXME, IKT, 5/22/15: add observer!
      //const RCP<QCAD::CoupledPoissonSchrodinger> ps_model = rcp(new QCAD::CoupledPoissonSchrodinger(appParams, solverComm, initial_guess));
      //const RCP<ParameterList> piroParams = Teuchos::sublist(appParams, "Piro");

      // Create and setup the Piro solver factory -- need to convert to not be based on Epetra!
      //Piro::Epetra::SolverFactory piroFactory;
      // Replace above with Piro::SolverFactory piroFactory; ?
      /*{
        // Do we need: Observers for output from time-stepper ??
         const RCP<Piro::ProviderBase<NOX::Epetra::Observer> > noxObserverProvider =
          rcp(new QCAD::CoupledPS_NOXObserverConstructor(ps_model));
          //  rcp(new NOXObserverConstructor(poisson_app));
          piroFactory.setSource<NOX::Epetra::Observer>(noxObserverProvider);

        // LOCA auxiliary objects -- needed?
         }
      */
      // Piro::SolverFactory
//      return piroFactory.createSolver<ST>(piroParams, ps_model);
      RCP<Thyra::AdaptiveSolutionManager> solMgrT = Teuchos::null;
      return piroFactory.createSolver<ST, LO, GO, KokkosNode>(piroParams, ps_model, solMgrT, Teuchos::null /* observer */);

#else /* ALBANY_QCAD */
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Must activate QCAD\n");
#endif /* ALBANY_QCAD */
    }

      if (solutionMethod == "Eigensolve") {
#ifdef ALBANY_QCAD
     TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Eigensolve does not work with AlbanyT executable!  QCAD::GenEigensolver class needs to be implemented with Thyra::ModelEvaluator instead of EpetraExt. \n");

      //RCP<Albany::Application> app;
      //const RCP<Thyra::ModelEvaluator<ST> > modelT = createAlbanyAppAndModelT(app, appComm, initial_guess);
      //albanyApp = app;

      //QCAD::GenEigensolver uses a state manager as an observer (for now)
      //RCP<Albany::StateManager> observer = rcp( &(app->getStateMgr()), false);

      // Currently, QCAD eigensolver just uses LOCA's eigensolver list under Piro -- maybe give it it's own list
      // outside of Piro?
      //const RCP<ParameterList> eigensolveParams = rcp(&(appParams->sublist("Piro").sublist("LOCA").sublist("Stepper").sublist("Eigensolver")), false);
      //const RCP<QCAD::GenEigensolver> es_model = rcp(new QCAD::GenEigensolver(eigensolveParams, modelT, observer, solverComm));
      //return es_model;

#else /* ALBANY_QCAD */
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Must activate QCAD\n");
#endif /* ALBANY_QCAD */
    }

//IK, 10/16/14: ATO::Solver needs to be converted to Tpetra?
// if (solutionMethod == "ATO Problem") {
//#ifdef ALBANY_ATO
//      return rcp(new ATO::Solver(appParams, solverComm, initial_guess));
//#else /* ALBANY_ATO */
//      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Must activate ATO (topological optimization)\n");
//#endif /* ALBANY_ATO */
//    }

#ifdef ALBANY_AERAS
  if (solutionMethod == "Aeras Hyperviscosity") {
    //std::cout <<"In Albany_SolverFactory: solutionMethod = Aeras Hyperviscosity" << std::endl;
    //Check if HV coefficient tau is zero of "Explicit HV" is false. Then there is no need for Aeras HVDecorator.

    double tau;
    bool useExplHyperviscosity;

    std::string swProblem_name    = "Shallow Water Problem",
            hydroProblem_name = "Hydrostatic Problem";

    bool swProblem = problemParams->isSublist(swProblem_name);
    bool hydroProblem = problemParams->isSublist(hydroProblem_name);

    if( (!swProblem) && (!hydroProblem) ){

      *out << "Error: Hyperviscosity can only be used with Aeras:Shallow Water or Aeras:Hydrostatic." << std::endl;
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
          "Error: cannot locate " << swProblem_name <<" or " << hydroProblem_name << " sublist in the input file." << "\n");
    }

    if ( swProblem ){
       tau = problemParams->sublist(swProblem_name).get<double>("Hyperviscosity Tau", 0.0);
       useExplHyperviscosity = problemParams->sublist(swProblem_name).get<bool>("Use Explicit Hyperviscosity", false);
       *out << "Reading Shallow Water Problem List: Using explicit hyperviscosity? " <<useExplHyperviscosity << "\n";
    }
    if ( hydroProblem ){
       tau = problemParams->sublist(hydroProblem_name).get<double>("Hyperviscosity Tau", 0.0);
       useExplHyperviscosity = problemParams->sublist(hydroProblem_name).get<bool>("Use Explicit Hyperviscosity", false);
       *out << "Reading Hydrostatic Problem List: Using explicit hyperviscosity? " <<useExplHyperviscosity << "\n";
    }


    if( (useExplHyperviscosity) && (tau != 0.0) ){

///// make a solver, repeated code
    const RCP<ParameterList> piroParams = Teuchos::sublist(appParams, "Piro");
    const Teuchos::RCP<Teuchos::ParameterList> stratList = Piro::extractStratimikosParams(piroParams);
    // Create and setup the Piro solver factory
    Piro::SolverFactory piroFactory;
    // Setup linear solver
    Stratimikos::DefaultLinearSolverBuilder linearSolverBuilder;
    enableIfpack2(linearSolverBuilder);
    enableMueLu(albanyApp, stratList, linearSolverBuilder);
    linearSolverBuilder.setParameterList(stratList);
    const RCP<Thyra::LinearOpWithSolveFactoryBase<ST> > lowsFactory = createLinearSolveStrategy(linearSolverBuilder);

///// create an app and a model evaluator

    RCP<Albany::Application> app;

    app = rcp(new Albany::Application(appComm, appParams, initial_guess));
    RCP<Thyra::ModelEvaluatorDefaultBase<ST> > modelHV(new Aeras::HVDecorator(app, appParams));

    albanyApp = app;

    RCP<Thyra::ModelEvaluator<ST> > modelWithSolveT;

    modelWithSolveT =
      rcp(new Thyra::DefaultModelEvaluatorWithSolveFactory<ST>(modelHV, lowsFactory));

    const RCP<Piro::ObserverBase<double> > observer = rcp(new PiroObserverT(albanyApp, modelWithSolveT));

    // Piro::SolverFactory
    return piroFactory.createSolver<ST, LO, GO, KokkosNode>(piroParams, modelWithSolveT, Teuchos::null, observer);

    }//if useExplHV=true and tau <>0.

  }//if Aeras HyperViscosity
#endif

#if defined(ALBANY_LCM) && defined(HAVE_STK)
  if (solutionMethod == "Coupled Schwarz") {

    std::cout <<"In Albany_SolverFactory: solutionMethod = Coupled Schwarz!" << std::endl;

#ifndef ALBANY_DTK
    if (appComm->getSize() > 1)
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
        "Error: cannot run Coupled Schwarz problem on > 1 procs when DTK is disabled.  "
        <<"Rebuild Trilinos and Albany with DTK to run Coupled Schwarz in parallel." << "\n");
#endif //ALBANY_DTK

    //IKT: We are assuming the "Piro" list will come from the main coupled Schwarz input file (not the sub-input
    //files for each model).  This makes sense I think.
    const RCP<ParameterList> piroParams = Teuchos::sublist(appParams, "Piro");

    const Teuchos::RCP<Teuchos::ParameterList> stratList = Piro::extractStratimikosParams(piroParams);
    // Create and setup the Piro solver factory
    Piro::SolverFactory piroFactory;
    // Setup linear solver
    Stratimikos::DefaultLinearSolverBuilder linearSolverBuilder;
    enableIfpack2(linearSolverBuilder);
    enableMueLu(albanyApp, stratList, linearSolverBuilder);

#ifdef ALBANY_TEKO
    Teko::addTekoToStratimikosBuilder(linearSolverBuilder, "Teko");
#endif
    linearSolverBuilder.setParameterList(stratList);

    const RCP<Thyra::LinearOpWithSolveFactoryBase<ST> > lowsFactory =
        createLinearSolveStrategy(linearSolverBuilder);

    const RCP<LCM::SchwarzMultiscale> coupled_model_with_solveT = rcp(new LCM::SchwarzMultiscale(appParams, solverComm,
                                                                         initial_guess, lowsFactory));

    const RCP<Piro::ObserverBase<double> > observer = rcp(new LCM::Schwarz_PiroObserverT(coupled_model_with_solveT));

    // WARNING: Coupled Schwarz does not contain a primary Albany::Application instance and so albanyApp is null.
    // Piro::SolverFactory
//    return piroFactory.createSolver<ST>(piroParams, coupled_model_with_solveT, observer);
    return piroFactory.createSolver<ST, LO, GO, KokkosNode>(piroParams, coupled_model_with_solveT, Teuchos::null, observer);
  }
#endif /* LCM and Schwarz */

  RCP<Albany::Application> app = albanyApp;
  const RCP<Thyra::ModelEvaluator<ST> > modelT =
    createAlbanyAppAndModelT(app, appComm, initial_guess, createAlbanyApp);
  // Pass back albany app so that interface beyond ModelEvaluator can be used.
  // This is essentially a hack to allow additional in/out arguments beyond what
  // ModelEvaluator specifies.
  albanyApp = app;

  const RCP<ParameterList> piroParams = Teuchos::sublist(appParams, "Piro");
  const Teuchos::RCP<Teuchos::ParameterList> stratList = Piro::extractStratimikosParams(piroParams);

  if(Teuchos::is_null(stratList)){

  *out << "Error: cannot locate Stratimikos solver parameters in the input file." << std::endl;
    *out << "Printing the Piro parameter list:" << std::endl;
    piroParams->print(*out);
// GAH: this is an error - should be fatal
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
        "Error: cannot locate Stratimikos solver parameters in the input file." << "\n");
  }



  RCP<Thyra::ModelEvaluator<ST> > modelWithSolveT;
  if (Teuchos::nonnull(modelT->get_W_factory())) {
    modelWithSolveT = modelT;
  } else {
    // Setup linear solver
    Stratimikos::DefaultLinearSolverBuilder linearSolverBuilder;
    enableIfpack2(linearSolverBuilder);
    enableMueLu(albanyApp, stratList, linearSolverBuilder);
#ifdef ALBANY_TEKO
    Teko::addTekoToStratimikosBuilder(linearSolverBuilder, "Teko");
#endif
    linearSolverBuilder.setParameterList(stratList);

    const RCP<Thyra::LinearOpWithSolveFactoryBase<ST> > lowsFactory =
      createLinearSolveStrategy(linearSolverBuilder);

    modelWithSolveT =
      rcp(new Thyra::DefaultModelEvaluatorWithSolveFactory<ST>(modelT, lowsFactory));
  }

  const RCP<Thyra::AdaptiveSolutionManager> solMgrT = app->getAdaptSolMgrT();
  Piro::SolverFactory piroFactory;

  if(solMgrT->isAdaptive()){
    if(TpetraBuild){
      const RCP<Piro::ObserverBase<double> > observer = rcp(new PiroObserverT(app, modelWithSolveT));
      return piroFactory.createSolver<ST, LO, GO, KokkosNode>(piroParams, modelWithSolveT, solMgrT, observer);
    }
#if defined(ALBANY_EPETRA)
    else {
      const RCP<Piro::ObserverBase<double> > observer = rcp(new PiroObserver(app));
      return piroFactory.createSolver<ST, LO, GO, KokkosNode>(piroParams, modelWithSolveT, solMgrT, observer);
    }
#endif
  }
  else {
    if(TpetraBuild){
      const RCP<Piro::ObserverBase<double> > observer = rcp(new PiroObserverT(app, modelWithSolveT));
      return piroFactory.createSolver<ST, LO, GO, KokkosNode>(piroParams, modelWithSolveT, Teuchos::null, observer);
    }
#if defined(ALBANY_EPETRA)
    else {
      const RCP<Piro::ObserverBase<double> > observer = rcp(new PiroObserver(app));
      return piroFactory.createSolver<ST, LO, GO, KokkosNode>(piroParams, modelWithSolveT, Teuchos::null, observer);
    }
#endif
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
        "Reached end of createAndGetAlbanyAppT()" << "\n");
  return Teuchos::null;
}

#if defined(ALBANY_EPETRA)
Teuchos::RCP<EpetraExt::ModelEvaluator>
Albany::SolverFactory::createAlbanyAppAndModel(
  Teuchos::RCP<Albany::Application>& albanyApp,
  const Teuchos::RCP<const Teuchos_Comm>& appCommT,
  const Teuchos::RCP<const Tpetra_Vector>& initial_guess)
{
  // Create application
  albanyApp = rcp(new Albany::Application(appCommT, appParams, initial_guess));

  return createModel(albanyApp,appCommT);
}


Teuchos::RCP<EpetraExt::ModelEvaluator>
Albany::SolverFactory::createModel(
  const Teuchos::RCP<Albany::Application>& albanyApp,
  const Teuchos::RCP<const Teuchos_Comm>& appCommT)
{
  // Validate Response list: may move inside individual Problem class
  const RCP<ParameterList> problemParams = Teuchos::sublist(appParams, "Problem");
  problemParams->sublist("Response Functions").
    validateParameters(*getValidResponseParameters(),0);

  // Create model evaluator
  Albany::ModelFactory modelFactory(appParams, albanyApp);

  return modelFactory.create();

}
#endif

Teuchos::RCP<Thyra::ModelEvaluator<ST> >
Albany::SolverFactory::createAlbanyAppAndModelT(
  Teuchos::RCP<Albany::Application>& albanyApp,
  const Teuchos::RCP<const Teuchos_Comm>& appComm,
  const Teuchos::RCP<const Tpetra_Vector>& initial_guess,
  const bool createAlbanyApp)
{
  if (createAlbanyApp) {
    // Create application
    albanyApp = rcp(new Albany::Application(appComm, appParams, initial_guess));
    //  albanyApp = rcp(new Albany::ApplicationT(appComm, appParams, initial_guess));
  }

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

int Albany::SolverFactory::checkSolveTestResultsT(
  int response_index,
  int parameter_index,
  const Tpetra_Vector* g,
  const Tpetra_MultiVector* dgdp) const
{
  ParameterList* testParams = getTestParameters(response_index);

  int failures = 0;
  int comparisons = 0;
  const double relTol = testParams->get<double>("Relative Tolerance");
  const double absTol = testParams->get<double>("Absolute Tolerance");

  // Get number of responses (g) to test
  const int numResponseTests = testParams->get<int>("Number of Comparisons");
  if (numResponseTests > 0) {
    if (g == NULL || numResponseTests > g->getGlobalLength()) failures += 1000;
    else { // do comparisons
      Teuchos::Array<double> testValues =
        testParams->get<Teuchos::Array<double> >("Test Values");

      TEUCHOS_TEST_FOR_EXCEPT(numResponseTests != testValues.size());

      Teuchos::ArrayRCP<const double> gv = g->get1dView();
      for (int i=0; i<testValues.size(); i++) {
        failures += scaledCompare(gv[i], testValues[i], relTol, absTol);
        comparisons++;
      }
    }
  }

  // Repeat comparisons for sensitivities
  Teuchos::ParameterList *sensitivityParams = 0;
  std::string sensitivity_sublist_name =
    Albany::strint("Sensitivity Comparisons", parameter_index);
  if (parameter_index == 0 && !testParams->isSublist(sensitivity_sublist_name))
    sensitivityParams = testParams;
  else if(testParams->isSublist(sensitivity_sublist_name))
    sensitivityParams = &(testParams->sublist(sensitivity_sublist_name));
  if(sensitivityParams != 0) {
    const int numSensTests =
      sensitivityParams->get<int>("Number of Sensitivity Comparisons",0);
    if (numSensTests > 0) {
      if (dgdp == NULL || numSensTests > dgdp->getGlobalLength()) failures += 10000;
      else {
        for (int i=0; i<numSensTests; i++) {
          Teuchos::Array<double> testSensValues =
            sensitivityParams->get<Teuchos::Array<double> >(Albany::strint("Sensitivity Test Values",i));
          TEUCHOS_TEST_FOR_EXCEPT(dgdp->getNumVectors() != testSensValues.size());

          Teuchos::ArrayRCP<Teuchos::ArrayRCP<const double> > dgdpv = dgdp->get2dView();
          for (int j=0; j<dgdp->getNumVectors(); j++) {
            failures += scaledCompare(dgdpv[j][i], testSensValues[j], relTol, absTol);
            comparisons++;
          }
        }
      }
    }
  }

  storeTestResults(testParams, failures, comparisons);

  return failures;
}

#if defined(ALBANY_EPETRA)
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
  Teuchos::ParameterList *sensitivityParams = 0;
  std::string sensitivity_sublist_name =
    Albany::strint("Sensitivity Comparisons", parameter_index);
  if (parameter_index == 0 && !testParams->isSublist(sensitivity_sublist_name))
    sensitivityParams = testParams;
  else if(testParams->isSublist(sensitivity_sublist_name))
    sensitivityParams = &(testParams->sublist(sensitivity_sublist_name));

  if(sensitivityParams != 0) {
    const int numSensTests =
      sensitivityParams->get<int>("Number of Sensitivity Comparisons", 0);
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
  }

  storeTestResults(testParams, failures, comparisons);

  return failures;
}
#endif

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

#if defined(ALBANY_EPETRA)
#ifdef ALBANY_STOKHOS
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
#endif
#endif

ParameterList* Albany::SolverFactory::getTestParameters(int response_index) const
{
  ParameterList* result;

  if (response_index == 0 && appParams->isSublist("Regression Results")) {
    result = &(appParams->sublist("Regression Results"));
  } else {
    result = &(appParams->sublist(Albany::strint("Regression Results", response_index)));
  }

  TEUCHOS_TEST_FOR_EXCEPTION(result->isType<std::string>("Test Values"), std::logic_error,
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
       << comparisons << std::endl;
}

int Albany::SolverFactory::scaledCompare(double x1, double x2, double relTol, double absTol) const
{
  const double d = fabs(x1 - x2);
  return (d <= 0.5*(fabs(x1) + fabs(x2))*relTol ||
          d <= fabs(absTol)) ? 0 : 1;
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
  validPL->sublist("Scaling",            false, "Jacobian/Residual Scaling sublist");
  validPL->sublist("DataTransferKit",    false, "DataTransferKit sublist");
  validPL->sublist("DataTransferKit",    false, "DataTransferKit sublist").sublist("Consistent Interpolation",    false, "DTK Consistent Interpolation sublist");
  validPL->sublist("DataTransferKit",    false, "DataTransferKit sublist").sublist("Search",    false, "DTK Search sublist");
  validPL->sublist("DataTransferKit",    false, "DataTransferKit sublist").sublist("L2 Projection",    false, "DTK L2 Projection sublist");
  validPL->sublist("DataTransferKit",    false, "DataTransferKit sublist").sublist("Point Cloud",    false, "DTK Point Cloud sublist");
  validPL->sublist("Discretization",     false, "Discretization sublist");
  validPL->sublist("Quadrature",         false, "Quadrature sublist");
  validPL->sublist("Regression Results", false, "Regression Results sublist");
  validPL->sublist("VTK",                false, "DEPRECATED  VTK sublist");
  validPL->sublist("Piro",               false, "Piro sublist");
  validPL->sublist("Coupled System",     false, "Coupled system sublist");

  // validPL->set<std::string>("Jacobian Operator", "Have Jacobian", "Flag to allow Matrix-Free specification in Piro");
  // validPL->set<double>("Matrix-Free Perturbation", 3.0e-7, "delta in matrix-free formula");

  return validPL;
}

RCP<const ParameterList>
Albany::SolverFactory::getValidRegressionResultsParameters() const
{
  using Teuchos::Array;
  RCP<ParameterList> validPL = rcp(new ParameterList("ValidRegressionParams"));;
  Array<double> ta;; // std::string to be converted to teuchos array

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
    validPL->sublist(Albany::strint("Sensitivity Comparisons",i), false, "Sensitivity Comparisons sublist");
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
  validPL->set<std::string>("Collection Method", "Sum Responses");
  validPL->set<int>("Number of Response Vectors", 0);
  validPL->set<bool>("Observe Responses", true);
  validPL->set<int>("Responses Observation Frequency", 1);
  Teuchos::Array<unsigned int>  defaultDataUnsignedInt;
  validPL->set<Teuchos::Array<unsigned int> >("Relative Responses Markers", defaultDataUnsignedInt,
		  "Array of responses for which relative change will be obtained");

  validPL->set<int>("Number", 0);
  validPL->set<int>("Equation", 0);
  const int maxParameters = 500;
  for (int i=0; i<maxParameters; i++) {
    validPL->set<std::string>(Albany::strint("Response",i), "");
    validPL->sublist(Albany::strint("ResponseParams",i));
    validPL->sublist(Albany::strint("Response Vector",i));
  }
  return validPL;
}
