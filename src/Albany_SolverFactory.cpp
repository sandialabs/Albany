/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/


#include "Albany_SolverFactory.hpp"
#include "ENAT_SGNOXSolver.hpp"
#include "Albany_RythmosObserver.hpp"
#include "Albany_NOXObserver.hpp"
#include "Thyra_DetachedVectorView.hpp"
#include "Albany_SaveEigenData.hpp"

#include "Piro_Epetra_NOXSolver.hpp"
#include "Piro_Epetra_LOCASolver.hpp"
#include "Piro_Epetra_RythmosSolver.hpp"
#include "Piro_Epetra_VelocityVerletSolver.hpp"
#include "Piro_Epetra_TrapezoidRuleSolver.hpp"
#include "QCAD_Solver.hpp"

#include "Teuchos_XMLParameterListHelpers.hpp"
#include "Teuchos_TestForException.hpp"

#include "Rythmos_IntegrationObserverBase.hpp"
#include "Albany_Application.hpp"
#include "Albany_Utils.hpp"

#include "NOX_Epetra_Observer.H"


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
  appParams = rcp(new ParameterList("Albany Parameters"));
  Teuchos::updateParametersFromXmlFileAndBroadcast(inputFile, appParams.get(), *tcomm);

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


Teuchos::RCP<EpetraExt::ModelEvaluator>  
Albany::SolverFactory::createAndGetAlbanyApp(
  Teuchos::RCP<Albany::Application>& albanyApp,
  const Teuchos::RCP<const Epetra_Comm>& appComm,
  const Teuchos::RCP<const Epetra_Comm>& solverComm,
  const Teuchos::RCP<const Epetra_Vector>& initial_guess)
{
    // Get solver type
    ParameterList& problemParams = appParams->sublist("Problem");
    string solutionMethod = problemParams.get("Solution Method", "Steady");
    TEST_FOR_EXCEPTION(solutionMethod != "Steady" &&
            solutionMethod != "Transient" && solutionMethod != "Continuation" &&
	    solutionMethod != "Multi-Problem",  
            std::logic_error, "Solution Method must be Steady, Transient, "
            << "Continuation, or Multi-Problem not : " << solutionMethod);
    bool stochastic = problemParams.get("Stochastic", false);
    string secondOrder = problemParams.get("Second Order", "No");

    //set up parameters
    ParameterList& parameterParams = problemParams.sublist("Parameters");
    parameterParams.validateParameters(*getValidParameterParameters(),0);

    numParameters = parameterParams.get("Number", 0);
    RCP< Teuchos::Array<std::string> > free_param_names;
    if (numParameters>0) {
      free_param_names = rcp(new Teuchos::Array<std::string>);
      for (int i=0; i<numParameters; i++) {
        free_param_names->push_back(parameterParams.get(Albany::strint("Parameter",i), "??"));
      }
    }
    *out << "Number of Parameters in ENAT = " << numParameters << endl;

    //set up SG parameters
    ParameterList& sgParams =
      problemParams.sublist("Stochastic Galerkin");
    ParameterList& sg_parameterParams =
      sgParams.sublist("SG Parameters");
 
    sg_parameterParams.validateParameters(*getValidParameterParameters(),0);
    int sg_numParameters = sg_parameterParams.get("Number", 0);
    RCP< Teuchos::Array<std::string> > sg_param_names;
    if (sg_numParameters>0) {
      sg_param_names = rcp(new Teuchos::Array<std::string>);
      for (int i=0; i<sg_numParameters; i++) {
      sg_param_names->push_back(sg_parameterParams.get(Albany::strint("Parameter",i), "??"));
      }
    }
    *out << "Number of SG Parameters in ENAT = " << sg_numParameters << endl;

    // Validate Response list: may move inside individual Problem class
    problemParams.sublist("Response Functions").
      validateParameters(*getValidResponseParameters(),0);

    

    Teuchos::RCP<Albany::Application> app;
    Teuchos::RCP<EpetraExt::ModelEvaluator> model;

    typedef double Scalar;
    RCP<Rythmos::IntegrationObserverBase<Scalar> > Rythmos_observer;
    RCP<NOX::Epetra::Observer > NOX_observer;

    // QCAD::Solve is only example of a multi-app solver so far
    bool bSingleAppSolver = (solutionMethod != "Multi-Problem");

    //If solver uses a single app, create it here along with observer
    if (bSingleAppSolver) {

      // Create application
      app = rcp(new Albany::Application(appComm, appParams, initial_guess));

      //Pass back albany app so that interface beyond ModelEvaluator can be used.
      // This is essentially a hack to allow additional in/out arguments beyond 
      //  what ModelEvaluator specifies.
      albanyApp = app; 

      // Create model evaluator
      model = rcp(new Albany::ModelEvaluator(app, free_param_names, sg_param_names));

      // Create observer for output from time-stepper
      if (solutionMethod=="Transient" && secondOrder=="No")
	Rythmos_observer = rcp(new Albany_RythmosObserver(app));
      else
	NOX_observer = rcp(new Albany_NOXObserver(app));
    }

    TEST_FOR_EXCEPTION(stochastic && solutionMethod!="Steady", std::logic_error,
         "Stochastic problems only implemented for Steady NOX solves so far\n");

    if (stochastic && solutionMethod=="Steady")
      return  rcp(new ENAT::SGNOXSolver(appParams, model, solverComm, NOX_observer));
    else if (solutionMethod== "Continuation") { // add save eigen data here as in Piro test
      Teuchos::ParameterList& locaParams = (*appParams).sublist("LOCA");
        RCP<LOCA::SaveEigenData::AbstractStrategy> saveEigs =
	  rcp(new Albany::SaveEigenData( locaParams, NOX_observer, &app->getStateMgr() ));
        return  rcp(new Piro::Epetra::LOCASolver(appParams, model, NOX_observer, saveEigs));
	//return  rcp(new Piro::Epetra::LOCASolver(appParams, model, NOX_observer));
    }
    else if (solutionMethod== "Transient" && secondOrder=="No") 
      return  rcp(new Piro::Epetra::RythmosSolver(appParams, model, Rythmos_observer));
    else if (solutionMethod== "Transient" && secondOrder=="Velocity Verlet")
      return  rcp(new Piro::Epetra::VelocityVerletSolver(appParams, model, NOX_observer));
    else if (solutionMethod== "Transient" && secondOrder=="Trapezoid Rule")
      return  rcp(new Piro::Epetra::TrapezoidRuleSolver(appParams, model, NOX_observer));
    else if (solutionMethod== "Multi-Problem")
      return  rcp(new QCAD::Solver(appParams, solverComm));
    else if (solutionMethod== "Transient") {
      TEST_FOR_EXCEPTION(secondOrder!="No", std::logic_error,
         "Invalid value for Second Order: (No, Velocity Verlet, Trapezoid Rule): "
         << secondOrder << "\n");
      return Teuchos::null;
      }
    else
      return  rcp(new Piro::Epetra::NOXSolver(appParams, model, NOX_observer));
}


int Albany::SolverFactory::checkTestResults(
  const Epetra_Vector* g,
  const Epetra_MultiVector* dgdp,
  const Teuchos::SerialDenseVector<int,double>* drdv,
  const Teuchos::RCP<Thyra::VectorBase<double> >& tvec,
  const Teuchos::RCP<Stokhos::EpetraVectorOrthogPoly>& g_sg) const
{
  ParameterList& testParams = appParams->sublist("Regression Results");
  TEST_FOR_EXCEPTION(testParams.isType<string>("Test Values"), std::logic_error,
    "Array information in XML file must now be of type Array(double)\n");
  testParams.validateParametersAndSetDefaults(*getValidRegressionResultsParameters(),0);

  int failures = 0;
  int comparisons = 0;
  double relTol = testParams.get<double>("Relative Tolerance");
  double absTol = testParams.get<double>("Absolute Tolerance");


  // Get number of responses (g) to test
  int numResponseTests = testParams.get<int>("Number of Comparisons");
  if (numResponseTests > 0 && g != NULL) {

    if (numResponseTests > g->MyLength()) failures +=1000;
    else { // do comparisons
      Teuchos::Array<double> testValues =
        testParams.get<Teuchos::Array<double> >("Test Values");
      
      TEST_FOR_EXCEPT(numResponseTests != testValues.size());
      for (int i=0; i<testValues.size(); i++) {
        failures += scaledCompare((*g)[i], testValues[i], relTol, absTol);
        comparisons++;
      }
    }
  }

  // Repeat comparisons for sensitivities
  int numSensTests = testParams.get<int>("Number of Sensitivity Comparisons");
  if (numSensTests > 0 && dgdp != NULL) {

    if (numSensTests > dgdp->MyLength() ||
        numParameters != dgdp->NumVectors() ) failures += 10000;
    else {
      for (int i=0; i<numSensTests; i++) {
        Teuchos::Array<double> testSensValues =
          testParams.get<Teuchos::Array<double> >(Albany::strint("Sensitivity Test Values",i));
        TEST_FOR_EXCEPT(numParameters != testSensValues.size());
        for (int j=0; j<numParameters; j++) {
          failures += scaledCompare((*dgdp)[j][i], testSensValues[j], relTol, absTol);
          comparisons++;
        }
      }
    }
  }

  // Repeat comparisons for Dakota runs
  int numDakotaTests = testParams.get<int>("Number of Dakota Comparisons");
  if (numDakotaTests > 0 && drdv != NULL) {

    if (numDakotaTests > drdv->length()) failures += 100000;
    else { // do comparisons
      // Read accepted test results
      Teuchos::Array<double> testValues =
        testParams.get<Teuchos::Array<double> >("Dakota Test Values");

      TEST_FOR_EXCEPT(numDakotaTests != testValues.size());
      for (int i=0; i<numDakotaTests; i++) {
        failures += scaledCompare((*drdv)[i], testValues[i], relTol, absTol);
        comparisons++;
      }
    }
  }

  // Repeat comparisons for Piro Analysis runs
  int numPiroTests = testParams.get<int>("Number of Piro Analysis Comparisons");
  if (numPiroTests > 0 && tvec != Teuchos::null) {

     // Create indexable thyra vector
      ::Thyra::DetachedVectorView<double> p(tvec);

    if (numPiroTests > p.subDim()) failures += 300000;
    else { // do comparisons
      // Read accepted test results
      Teuchos::Array<double> testValues =
        testParams.get<Teuchos::Array<double> >("Piro Analysis Test Values");

      TEST_FOR_EXCEPT(numPiroTests != testValues.size());
      for (int i=0; i<numPiroTests; i++) {
        failures += scaledCompare(p[i], testValues[i], relTol, absTol);
        comparisons++;
      }
    }
  }

  // Repeat comparisons for SG expansions
  int numSGTests = testParams.get<int>("Number of Stochastic Galerkin Comparisons");
  if (numSGTests > 0 && g_sg != Teuchos::null) {
    if (numSGTests > (*g_sg)[0].MyLength()) failures += 10000;
    else {
      for (int i=0; i<numSGTests; i++) {
        Teuchos::Array<double> testSGValues = 
          testParams.get<Teuchos::Array<double> >
            (Albany::strint("Stochastic Galerkin Expansion Test Values",i));
        TEST_FOR_EXCEPT(g_sg->size() != testSGValues.size());
	for (int j=0; j<g_sg->size(); j++) {
	  failures += 
	    scaledCompare((*g_sg)[j][i], testSGValues[j], relTol, absTol);
          comparisons++;
        }
      }
    }
  }

  // Store failures in param list (this requires mutable appParams!)
  testParams.set("Number of Failures", failures);
  testParams.set("Number of Comparisons Attempted", comparisons);
  *out << "\nCheckTestResults: Number of Comparisons Attempted = "
       << comparisons << endl;
  return failures;
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
    ParameterList& noxParams = appParams_->sublist("NOX");
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
  validPL->sublist("Discretization",     false, "Discretization sublist");
  validPL->sublist("Quadrature",         false, "Quadrature sublist");
  validPL->sublist("Regression Results", false, "Regression Results sublist");
  validPL->sublist("VTK",                false, "DEPRECATED  VTK sublist");
  validPL->sublist("Rythmos",            false, "Rythmos sublist");
  validPL->sublist("Velocity Verlet",    false, "Piro Velocity Verlet sublist");
  validPL->sublist("Trapezoid Rule",     false, "Piro Trapezoid Rule sublist");
  validPL->sublist("LOCA",               false, "LOCA sublist");
  validPL->sublist("NOX",                false, "NOX sublist");
  validPL->sublist("Analysis",           false, "Analysis sublist");

  validPL->set<string>("Jacobian Operator", "Have Jacobian", "Flag to allow Matrix-Free specification in Piro");
  validPL->set<double>("Matrix-Free Perturbation", 3.0e-7, "delta in matrix-free formula");

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

  // These two are typically not set on input, just output.
  validPL->set<int>("Number of Failures", 0,
     "Output information from regression tests reporting number of failed tests");
  validPL->set<int>("Number of Comparisons", 0,
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

  validPL->set<int>("Number", 0);
  validPL->set<int>("Equation", 0);
  const int maxParameters = 100;
  for (int i=0; i<maxParameters; i++) {
    validPL->set<std::string>(Albany::strint("Response",i), "");
    validPL->sublist(Albany::strint("ResponseParams",i));
  }
  return validPL;
}
