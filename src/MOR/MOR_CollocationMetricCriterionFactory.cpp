//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "MOR_CollocationMetricCriterionFactory.hpp"

#include "MOR_ContainerUtils.hpp"

#include "Teuchos_Tuple.hpp"
#include "Teuchos_TestForException.hpp"

#include <string>
#include <stdexcept>

namespace MOR {

CollocationMetricCriterionFactory::CollocationMetricCriterionFactory(
    const Teuchos::RCP<Teuchos::ParameterList> &params) :
  params_(params)
{}

Teuchos::RCP<CollocationMetricCriterion>
CollocationMetricCriterionFactory::instanceNew(int rankMax)
{
  const Teuchos::Tuple<std::string, 2> allowedTypes = Teuchos::tuple<std::string>("Two Norm", "Frobenius");
  const std::string type = params_->get("Collocation Metric Criterion", allowedTypes[0]);
  TEUCHOS_TEST_FOR_EXCEPTION(!contains(allowedTypes, type),
      std::out_of_range,
      type + " not in " + allowedTypes.toString());

  if (type == "Two Norm") {
    return Teuchos::rcp(new TwoNormCriterion(rankMax));
  } else if (type == "Frobenius") {
    return Teuchos::rcp(new FrobeniusNormCriterion);
  }

  TEUCHOS_TEST_FOR_EXCEPT_MSG(true, "Should not happen");
}

} // end namespace MOR
