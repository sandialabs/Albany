//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_RegressionTests.hpp"
#include "Albany_PiroObserver.hpp"
#include "Albany_ModelEvaluator.hpp"
#include "Albany_Application.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_Macros.hpp"

#include "Thyra_DetachedVectorView.hpp"

#include "Teuchos_TestForException.hpp"

namespace Albany
{

RegressionTests::
RegressionTests(const Teuchos::RCP<Teuchos::ParameterList>& appParams_)
 : appParams(appParams_)
 , out(Teuchos::VerboseObjectBase::getDefaultOStream())
{
  // Nothing to do
}

int RegressionTests::
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
    if (numSensTests != dgdp->range()->dim()) {
      ALBANY_ASSERT(
          false,
          "Number of sensitivity tests ("
              << numSensTests << ") != number of sensitivities ["
              << response_index << "][" << parameter_index << "] ("
              << dgdp->range()->dim() << ") !");
    }
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

int RegressionTests::checkAnalysisTestResults(
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
    if (testParams->get<bool>("Piro Analysis Test Two Norm",false)) {
      const auto norm = tvec->norm_2();
      *out << "Parameter Vector Two Norm: " << norm << std::endl;
      failures += scaledCompare(norm, testValues[0], relTol, absTol, "Piro Analysis Test Two Norm");
      comparisons++;
    }
    else {
      for (int i = 0; i < numPiroTests; i++) {
        auto s = std::string("Piro Analysis Test ") + std::to_string(i);
        failures += scaledCompare(p[i], testValues[i], relTol, absTol, s);
        comparisons++;
      }
    }
  }

  storeTestResults(testParams, failures, comparisons);

  return failures;
}

Teuchos::ParameterList*
RegressionTests::getTestParameters(int response_index) const
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

void RegressionTests::storeTestResults(
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

bool RegressionTests::scaledCompare(
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

Teuchos::RCP<const Teuchos::ParameterList>
RegressionTests::getValidRegressionResultsParameters() const
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
      "Number of Piro Analysis Comparisons",
      0,
      "Number of parameters from Analysis to regress against");
  validPL->set<bool>(
      "Piro Analysis Test Two Norm",
      false,
      "Test l2 norm of final parameters from Analysis runs");
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

} // namespace Albany
