//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_DISCRETIZATIONSAMPLEDOFLISTPROVIDER_HPP
#define ALBANY_DISCRETIZATIONSAMPLEDOFLISTPROVIDER_HPP

#include "MOR_SampleDofListFactory.hpp"

#include "Albany_AbstractDiscretization.hpp"

namespace Albany {

class DiscretizationSampleDofListProvider : public MOR::SampleDofListFactory::DofListProvider {
public:
  explicit DiscretizationSampleDofListProvider(const Teuchos::RCP<const AbstractDiscretization> &disc);
  virtual Teuchos::Array<int> operator()(const Teuchos::RCP<Teuchos::ParameterList> &params);

private:
  Teuchos::RCP<const AbstractDiscretization> disc_;
};

} // end namespace Albany

#endif /* ALBANY_DISCRETIZATIONSAMPLEDOFLISTPROVIDER_HPP */
