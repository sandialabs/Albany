//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef MOR_SAMPLEDOFLISTFACTORY_HPP
#define MOR_SAMPLEDOFLISTFACTORY_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_Array.hpp"
#include "Teuchos_ParameterList.hpp"

#include <string>
#include <map>

namespace MOR {

class SampleDofListFactory {
public:
  SampleDofListFactory();

  Teuchos::Array<int> create(const Teuchos::RCP<Teuchos::ParameterList> &params);

  class DofListProvider;
  void extend(const std::string &id, const Teuchos::RCP<DofListProvider> &provider);

private:
  typedef std::map<std::string, Teuchos::RCP<DofListProvider> > ProviderMap;
  ProviderMap providers_;
};

class SampleDofListFactory::DofListProvider {
public:
  virtual Teuchos::Array<int> operator()(const Teuchos::RCP<Teuchos::ParameterList> &params) = 0;
  virtual ~DofListProvider() {}
};

} // namespace MOR

#endif /* MOR_SAMPLEDOFLISTFACTORY_HPP */
