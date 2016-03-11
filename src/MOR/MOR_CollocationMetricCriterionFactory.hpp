//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef MOR_COLLOCATIONMETRICCRITERIONFACTORY_HPP
#define MOR_COLLOCATIONMETRICCRITERIONFACTORY_HPP

#include "MOR_CollocationMetricCriterion.hpp"

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

namespace MOR {

class CollocationMetricCriterionFactory {
public:
  explicit CollocationMetricCriterionFactory(const Teuchos::RCP<Teuchos::ParameterList> &params);

  Teuchos::RCP<CollocationMetricCriterion> instanceNew(int rankMax);

private:
  Teuchos::RCP<Teuchos::ParameterList> params_;
};

} // end namespace MOR

#endif /* MOR_COLLOCATIONMETRICCRITERIONFACTORY_HPP */
