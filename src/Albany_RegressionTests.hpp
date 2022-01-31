//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_REGRESSION_TESTS_HPP
#define ALBANY_REGRESSION_TESTS_HPP

#include "Albany_ThyraTypes.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_FancyOStream.hpp"

//! Albany driver code, problems, discretizations, and responses
namespace Albany {

/*!
 * \brief A factory class to instantiate AbstractSolver objects
 */
class RegressionTests {
public:
  //! Constructor
  RegressionTests (const Teuchos::RCP<Teuchos::ParameterList>& appParams);

  //! Destructor
  virtual ~RegressionTests () = default;

  /** \brief Function that does regression testing for a response.
   * returns the pair containing the number of failures and comparisons*/
  std::pair<int,int> checkResponse(
      int                                           response_index,
      const Teuchos::RCP<const Thyra_Vector>&       g) const;

  /** \brief Function that does regression testing for a sensitivity.
   * returns the pair containing the number of failures and comparisons*/
  std::pair<int,int> checkSensitivity(
      int                                           response_index,
      int                                           parameter_index,
      const Teuchos::RCP<const Thyra_MultiVector>&  dgdp) const;

  /** \brief Function asserting there are no sensitivity tests for a given response and parameter. */
  void assertNoSensitivityTests(
      int                                           response_index,
      int                                           parameter_index,
      const std::string&                            error_msg) const;

  /** \brief Function that does regression testing for Analysis runs.
   * returns the pair containing the number of failures and comparisons*/
  std::pair<int,int> checkAnalysisTestResults(
      int                                            response_index,
      const Teuchos::RCP<Thyra_Vector>& tvec) const;

protected:
  // Private functions to set default parameter values
  Teuchos::RCP<const Teuchos::ParameterList>
  getValidRegressionResultsParameters() const;

  /** \brief Testing utility that compares two numbers using two tolerances */
  bool scaledCompare (double             x1,
                      double             x2,
                      double             relTol,
                      double             absTol,
                      std::string const& name) const;

  Teuchos::ParameterList* getTestParameters(int response_index)const;

  Teuchos::RCP<Teuchos::ParameterList>  appParams;

  Teuchos::RCP<Teuchos::FancyOStream>   out;
};

}  // namespace Albany

#endif // ALBANY_REGRESSION_TESTS_HPP
