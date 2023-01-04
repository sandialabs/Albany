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
#include "Albany_StringUtils.hpp"
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

std::pair<int,int> RegressionTests::checkResponse(
    int                                           response_index,
    const Teuchos::RCP<const Thyra_Vector>&       g) const
{
  const Teuchos::ParameterList* testParams = getTestParameters(response_index);

  int          failures    = 0;
  int          comparisons = 0;

  if (testParams != NULL) {
    const double relTol      = testParams->isParameter("Relative Tolerance") ?
        testParams->get<double>("Relative Tolerance") : 1.0e-8;
    const double absTol      = testParams->isParameter("Absolute Tolerance") ?
        testParams->get<double>("Absolute Tolerance") : 1.0e-8;

    // Test the response (g)
    ALBANY_ASSERT(
        g != Teuchos::null,
        "There is Response Test but the response vector is null!");
    ALBANY_ASSERT(
        g->space()->dim() == 1,
        "The responses are assumed to be scalar responses; the response " << response_index
        << " is not scalar and its dimension is "
        << g->space()->dim() << " !");
    double testValue =
        testParams->get<double>("Test Value");

    Teuchos::ArrayRCP<const ST> g_view = getLocalData(g);
    auto s = std::string("Response Test ") + std::to_string(response_index);
    failures += scaledCompare(g_view[0], testValue, relTol, absTol, s);
    comparisons++;
  }

  return std::pair<int,int>(failures,comparisons);
}

std::pair<int,int> RegressionTests::checkSensitivity(
    int                                           response_index,
    int                                           parameter_index,
    const Teuchos::RCP<const Thyra_MultiVector>&  dgdp) const
{
  const Teuchos::ParameterList* testParams = getTestParameters(response_index);

  int          failures    = 0;
  int          comparisons = 0;

  if (testParams != NULL) {
    const double relTol      = testParams->isParameter("Relative Tolerance") ?
        testParams->get<double>("Relative Tolerance") : 1.0e-8;
    const double absTol      = testParams->isParameter("Absolute Tolerance") ?
        testParams->get<double>("Absolute Tolerance") : 1.0e-8;

    std::string sensitivity_sublist_name =
        util::strint("Sensitivity For Parameter", parameter_index);

    if (testParams->isSublist(sensitivity_sublist_name)) {
      // Repeat comparisons for sensitivities
      const Teuchos::ParameterList* sensitivityParams = &(testParams->sublist(sensitivity_sublist_name));
      ALBANY_ASSERT(
          dgdp != Teuchos::null,
          "There are Sensitivity Tests but the sensitivity vector ("
          << response_index << ", " << parameter_index << ") is null!");
      const int numSensTests = dgdp->range()->dim();
      const int numVecs = dgdp->domain()->dim();
      for (int i = 0; i < numSensTests; i++) {
        Teuchos::Array<double> testSensValues;

        const Teuchos::ParameterList& paramList = appParams->sublist("Problem").sublist("Parameters");
        const Teuchos::ParameterList& paramSublist = paramList.sublist(util::strint("Parameter", parameter_index));
        const std::string parameterType = paramSublist.isParameter("Type") ?
            paramSublist.get<std::string>("Type") : std::string("Scalar");
        if (parameterType == "Vector") {
          if (sensitivityParams->isType<Teuchos::Array<double>>("Test Values")) {
            int dimension = paramSublist.get<int>("Dimension");
            testSensValues = sensitivityParams->get<Teuchos::Array<double>>("Test Values");
            TEUCHOS_TEST_FOR_EXCEPTION(
                testSensValues.size() != dimension,
                Teuchos::Exceptions::InvalidParameter,
                std::endl
                << "Error!  In Albany::RegressionTests:: checkSolveTestResults:  "
                << "The dimension of the parameter vector "<< parameter_index << " which is " << dimension
                << " and the size of the \"Test Values\" which is " << testSensValues.size()
                << " are not equal."
                << std::endl);
          }
          else {
            TEUCHOS_TEST_FOR_EXCEPTION(
                !sensitivityParams->isType<Teuchos::Array<double>>("Test Values"),
                Teuchos::Exceptions::InvalidParameter,
                std::endl
                << "Error!  In Albany::RegressionTests:: checkSolveTestResults:\n"
                << "Expected Parameter \"Test Values\" of type Teuchos::Array<double> in \"" << sensitivity_sublist_name << "\" sublist."
                << std::endl);
          }
        }
        else if (parameterType == "Scalar" || parameterType == "Distributed") {
          if (sensitivityParams->isType<double>("Test Value")) {
            testSensValues.resize(1);
            testSensValues[0] = sensitivityParams->get<double>("Test Value");
          }
          else {
            TEUCHOS_TEST_FOR_EXCEPTION(
                !sensitivityParams->isType<double>("Test Value"),
                Teuchos::Exceptions::InvalidParameter,
                std::endl
                << "Error!  In Albany::RegressionTests:: checkSolveTestResults:\n"
                << "Expected Parameter \"Test Value\" of type double in \n"
                "\"" << sensitivity_sublist_name << "\" sublist for response " << response_index << "."
                << std::endl);
          }
        }
        else
          TEUCHOS_TEST_FOR_EXCEPTION(
              true,
              Teuchos::Exceptions::InvalidParameter,
              std::endl
              << "Error!  In Albany::RegressionTests:: checkSolveTestResults:  "
              << "Parameter "<< parameter_index << " is of type \"" << parameterType << "\" which is not supported."
              << std::endl);

        auto dgdp_view = getLocalData(dgdp);

        for (int jvec = 0; jvec < numVecs; jvec++) {
          auto s = std::string("Sensitivity (") + std::to_string(response_index) + std::string(",") + std::to_string(parameter_index) +
              std::string("), Test ") + std::to_string(i) + "," + std::to_string(jvec);
          failures +=
              scaledCompare(dgdp_view[jvec][i], testSensValues[i], relTol, absTol, s);
          comparisons++;
        }
      }
    }
  }
  return std::pair<int,int>(failures,comparisons);
}

void RegressionTests::
assertNoSensitivityTests(
    int                                           response_index,
    int                                           parameter_index,
    const std::string&                            error_msg) const
{
  Teuchos::ParameterList* testParams = getTestParameters(response_index);

  if (testParams != NULL) {
    std::string sensitivity_sublist_name =
        util::strint("Sensitivity For Parameter", parameter_index);

    ALBANY_ASSERT(!testParams->isSublist(sensitivity_sublist_name), error_msg);
  }
}

std::pair<int,int> RegressionTests::checkAnalysisTestResults(
    int                                            response_index,
    const Teuchos::RCP<Thyra::VectorBase<double>>& tvec) const
{
  Teuchos::ParameterList* testParams = getTestParameters(response_index);

  int          failures    = 0;
  int          comparisons = 0;

  if (testParams != NULL) {
    const double relTol      = testParams->get<double>("Relative Tolerance", 1.0e-8);
    const double absTol      = testParams->get<double>("Absolute Tolerance", 1.0e-8);

    int numPiroTests = testParams->isType<Teuchos::Array<double>>("Piro Analysis Test Values") ?
        testParams->get<Teuchos::Array<double>>("Piro Analysis Test Values").size() :
        0;

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

      if (testParams->get<bool>("Piro Analysis Test Two Norm",false)) {
        const auto norm = tvec->norm_2();
        *out << "Parameter Vector Two Norm: " << std::setprecision(12) << norm << std::endl;
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
  }

  return std::pair<int,int>(failures,comparisons);
}

Teuchos::ParameterList*
RegressionTests::getTestParameters(int response_index) const
{
  Teuchos::ParameterList* result = &(appParams->sublist(
      util::strint("Regression For Response", response_index)));
  if(result->isParameter("Test Value"))
    result->validateParameters(
        *getValidRegressionResultsParameters(), 0);
  else
    result = NULL;

  return result;
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

  validPL->set<double>(
      "Test Value", 0., "Regression value for responses");

  const int maxSensTests = 10;
  for (int i = 0; i < maxSensTests; i++) {
    std::string sublist_name = util::strint("Sensitivity For Parameter", i);
    validPL->sublist(sublist_name, false, "Sensitivity regression sublist");

    validPL->sublist(sublist_name).set<double>(
        util::strint("Test Value", i),
        0.,
        util::strint(
            "Array of regression values for Sensitivities w.r.t parameter", i));
    validPL->sublist(sublist_name).set<Array<double>>(
        util::strint("Test Values", i),
        ta,
        util::strint(
            "Array of regression values for Sensitivities w.r.t parameters", i));
  }

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
        util::strint("Stochastic Galerkin Expansion Test Values", i),
        ta,
        util::strint(
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
